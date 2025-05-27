"""
Mock CRS Device - UDP Streaming Logic.
Manages UDP packet generation and streaming for MockCRS.
"""
import asyncio
import socket
import struct
import time
import threading
import array
import signal
import atexit
from datetime import datetime
import numpy as np # For np.clip in generate_packet

# Import DfmuxPacket and related constants from streamer
from ..streamer import (
    DfmuxPacket, Timestamp, TimestampPort,
    STREAMER_MAGIC, STREAMER_VERSION, NUM_CHANNELS, SS_PER_SECOND,
    STREAMER_PORT # Default port for streaming
)

# Import enhanced scaling constants
from . import mock_constants as const

def _dfmuxpacket_to_bytes(self: DfmuxPacket) -> bytes:
    """
    Serialize a DfmuxPacket into 8240 bytes, matching streamer.py 'from_bytes()'.
    Layout:
      - 16-byte header (<IHHBBBBI)
      - 8192-byte channel data (NUM_CHANNELS*2 int32)
      - 32-byte timestamp (<8I)
    """
    # Ensure ts.c is an int before bitwise operations
    c_val = int(self.ts.c) if self.ts.c is not None else 0
    c_masked = c_val & 0x1FFFFFFF
    
    source_map = {
        TimestampPort.BACKPLANE: 0,
        TimestampPort.TEST: 1,
        TimestampPort.SMA: 2,
        # Using TEST as a fallback if source is not in map or is None
    }
    source_val = source_map.get(self.ts.source, source_map[TimestampPort.TEST])
    c_masked |= (source_val << 29)
    
    if self.ts.recent:
        c_masked |= 0x80000000

    hdr_struct = struct.Struct("<IHHBBBBI")
    hdr = hdr_struct.pack(
        self.magic if self.magic is not None else STREAMER_MAGIC,
        self.version if self.version is not None else STREAMER_VERSION,
        int(self.serial) if self.serial is not None else 0, # Ensure serial is int
        self.num_modules if self.num_modules is not None else 1,
        self.block if self.block is not None else 0,
        self.fir_stage if self.fir_stage is not None else 6, # Default FIR stage
        self.module if self.module is not None else 0, # Module index
        self.seq if self.seq is not None else 0
    )

    # Channel data: s should be an array.array('i')
    if not isinstance(self.s, array.array) or self.s.typecode != 'i':
        # If s is not the correct type (e.g. list of floats from model), convert it
        # This is a critical point for data integrity.
        # Assuming s contains float values that need to be scaled and converted to int32
        temp_arr = array.array("i", [0] * (NUM_CHANNELS * 2))
        scale = 32767 # Example scale, might need adjustment based on expected data range
        for i in range(NUM_CHANNELS * 2):
            if i < len(self.s):
                # This assumes s is flat [i0, q0, i1, q1, ...]
                # If s is [[i0,q0], [i1,q1]...], logic needs change
                val = self.s[i] * scale 
                temp_arr[i] = int(np.clip(val, -2147483648, 2147483647))
            else:
                temp_arr[i] = 0 # Pad if s is too short
        body_bytes = temp_arr.tobytes()

    else: # self.s is already an array.array('i')
        body_bytes = self.s.tobytes()

    if len(body_bytes) != NUM_CHANNELS * 2 * 4: # 4 bytes per int32
        raise ValueError(f"Channel data must be {NUM_CHANNELS*2*4} bytes. Got {len(body_bytes)}")
    
    ts_struct = struct.Struct("<8I")
    ts_data = ts_struct.pack(
        self.ts.y if self.ts.y is not None else 0,
        self.ts.d if self.ts.d is not None else 0,
        self.ts.h if self.ts.h is not None else 0,
        self.ts.m if self.ts.m is not None else 0,
        self.ts.s if self.ts.s is not None else 0,
        self.ts.ss if self.ts.ss is not None else 0,
        c_masked,
        self.ts.sbs if self.ts.sbs is not None else 0
    )

    packet = hdr + body_bytes + ts_data
    if len(packet) != 8240:
        raise ValueError(f"Packet length mismatch: {len(packet)} != 8240")
    return packet

# Monkey-patch the to_bytes method onto DfmuxPacket
DfmuxPacket.to_bytes = _dfmuxpacket_to_bytes

# Global registry to track all active UDP streamers for cleanup
_active_streamers = []
_cleanup_registered = False

def _emergency_cleanup():
    """Emergency cleanup function called on program exit."""
    global _active_streamers
    if _active_streamers:
        #print(f"[UDP] Emergency cleanup: stopping {len(_active_streamers)} active UDP streamers")
        for streamer in _active_streamers[:]:  # Copy list to avoid modification during iteration
            try:
                streamer.emergency_stop()
            except Exception as e:
                print(f"[UDP] Error during emergency cleanup: {e}")

def _register_global_cleanup():
    """Register global cleanup handlers (called once)."""
    global _cleanup_registered
    if not _cleanup_registered:
        atexit.register(_emergency_cleanup)
        # Register signal handlers for graceful shutdown
        for sig in [signal.SIGTERM, signal.SIGINT]:
            try:
                signal.signal(sig, lambda signum, frame: _emergency_cleanup())
            except (ValueError, OSError):
                pass  # Signal handling may not be available in all contexts
        _cleanup_registered = True

class MockCRSUDPStreamer(threading.Thread):
    """Streams UDP packets with S21 data based on MockCRS state"""
    
    def __init__(self, mock_crs, host='127.0.0.1', port=STREAMER_PORT, modules_to_stream=None):
        super().__init__(daemon=True)
        self.mock_crs = mock_crs # Reference to MockCRS instance
        self.host = host
        self.port = port
        
        # For MockCRS, always use unicast to 127.0.0.1 for reliability
        self.unicast_host = '127.0.0.1'
        self.unicast_port = port # Use the provided port
        
        # If modules_to_stream is specified, only stream those modules
        # Otherwise, we'll determine which modules to stream based on configured channels
        self.modules_to_stream = modules_to_stream  # Can be None or a list like [1] or [1,2]
        
        self.socket = None
        self.running = False # Controlled by start/stop
        self.seq_counters = {m: 0 for m in range(1, 5)}  # Per-module sequence numbers (1-4)
        self.packets_sent = 0
        
        # Register this streamer for global cleanup
        global _active_streamers
        _active_streamers.append(self)
        _register_global_cleanup()
        
        #print(f"[UDP] MockCRSUDPStreamer initialized - unicast to {self.unicast_host}:{self.unicast_port}")
        
    def _init_socket(self):
        """Initialize the UDP socket."""
        if self.socket is None:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # SO_SNDBUF might not be necessary for local unicast but doesn't hurt
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4_000_000)
    
    def _cleanup_socket(self):
        """Clean up the UDP socket."""
        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                print(f"[UDP] Error closing socket: {e}")
            finally:
                self.socket = None
        
    def _get_configured_modules(self):
        """Determine which modules have configured channels"""
        configured_modules = set()
        
        # Check frequencies, amplitudes, and phases dictionaries
        for (module, channel) in self.mock_crs.frequencies.keys():
            configured_modules.add(module)
        for (module, channel) in self.mock_crs.amplitudes.keys():
            configured_modules.add(module)
        for (module, channel) in self.mock_crs.phases.keys():
            configured_modules.add(module)
        
        # Convert to sorted list and ensure it's within valid range (1-4)
        modules = sorted([m for m in configured_modules if 1 <= m <= 4])
        
        if not modules:
            # If no modules configured, default to module 1
            modules = [1]
            
        return modules
    
    def stop(self):
        """Stop the streaming thread gracefully."""
        print("[UDP] Stopping UDP streamer...")
        self.running = False
        
    def emergency_stop(self):
        """Emergency stop - force shutdown immediately."""
        print(f"[UDP] Emergency stop for streamer {id(self)}")
        self.running = False
        self._cleanup_socket()
        # Remove from global registry
        global _active_streamers
        if self in _active_streamers:
            _active_streamers.remove(self)
        
    def run(self):
        """Stream UDP packets at the configured sample rate with proper cleanup."""
        #print("[UDP] UDP streaming thread started")
        self.running = True
        packet_count_interval = 0 # Packets sent in the current 1-second interval
        last_log_time = time.time()
        
        try:
            # Initialize socket
            self._init_socket()
            
            while self.running:
                try:
                    start_time_loop = time.perf_counter()
                    
                    # Determine which modules to stream
                    if self.modules_to_stream:
                        # Use explicitly specified modules
                        modules_to_process = self.modules_to_stream
                    else:
                        # Auto-detect: only stream modules that have configured channels
                        modules_to_process = self._get_configured_modules()
                    
                    # Generate and send packet for each active module
                    for module_num in modules_to_process:
                        if not self.running:  # Check if we should stop
                            break
                            
                        try:
                            packet_bytes = self.generate_packet_for_module(module_num, self.seq_counters[module_num])
                            bytes_sent = self.socket.sendto(packet_bytes, (self.unicast_host, self.unicast_port))
                            
                            self.seq_counters[module_num] += 1
                            self.packets_sent += 1
                            packet_count_interval += 1
                            
                            #if self.packets_sent <= 20: # Log first few packets (e.g., 5 per module)
                                #print(f"[UDP] Sent unicast packet #{self.packets_sent}: module={module_num}, seq={self.seq_counters[module_num]-1}, size={bytes_sent} bytes to {self.unicast_host}:{self.unicast_port}")
                            
                        except Exception as e:
                            if self.running:  # Only log if we're still supposed to be running
                                print(f"[UDP] ERROR generating/sending packet for module {module_num}: {e}")
                                import traceback
                                traceback.print_exc()
                    
                    current_time = time.time()
                    if current_time - last_log_time >= 1.0:
                        # if self.running:  # Only log if still running
                        #     print(f"[UDP] Status: {packet_count_interval} packets sent in last second (total {self.packets_sent})")
                        packet_count_interval = 0
                        last_log_time = current_time
                    
                    # Sleep to maintain sample rate (only if still running)
                    if self.running:
                        fir_stage = self.mock_crs.get_fir_stage()
                        sample_rate_per_channel = 625e6 / (256 * 64 * (2**fir_stage))
                        frame_time = 1.0 / sample_rate_per_channel
                        
                        elapsed_loop = time.perf_counter() - start_time_loop
                        sleep_duration = frame_time - elapsed_loop
                        if sleep_duration > 0:
                            time.sleep(sleep_duration)
                
                except KeyboardInterrupt:
                    print("[UDP] Received KeyboardInterrupt, stopping...")
                    break
                except Exception as e:
                    if self.running:
                        print(f"[UDP] Unexpected error in main loop: {e}")
                        import traceback
                        traceback.print_exc()
                        time.sleep(0.1)  # Brief pause before retrying
            
        except Exception as e:
            print(f"[UDP] Fatal error in UDP streamer: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # ALWAYS clean up, regardless of how we exit
            print(f"[UDP] UDP streaming thread stopped. Total packets sent: {self.packets_sent}")
            self._cleanup_socket()
            
            # Remove from global registry
            global _active_streamers
            if self in _active_streamers:
                _active_streamers.remove(self)
    
    def generate_packet_for_module(self, module_num, seq):
        """Generate a DfmuxPacket for a specific module."""
        # Create channel data array (NUM_CHANNELS = 1024, so 2048 int32s for I & Q)
        # This array should store int32 values.
        iq_data_arr = array.array("i", [0] * (NUM_CHANNELS * 2))
        
        # Get configured channels for this module to avoid processing all 1024
        configured_channels = set()
        for (mod, ch) in self.mock_crs.frequencies.keys():
            if mod == module_num:
                configured_channels.add(ch)
        for (mod, ch) in self.mock_crs.amplitudes.keys():
            if mod == module_num:
                configured_channels.add(ch)
        for (mod, ch) in self.mock_crs.phases.keys():
            if mod == module_num:
                configured_channels.add(ch)
        
        # Pre-calculate constants
        full_scale_counts = const.SCALE_FACTOR * 2**8  # 32 bit number instead of 24 bit
        noise_level = const.UDP_NOISE_LEVEL  # Base noise level (in ADC counts)
        nco_freq = self.mock_crs.get_nco_frequency(module=module_num) or 0
        
        # Generate noise for all channels first (this is fast)
        # Using vectorized operations for better performance
        noise_array = np.random.normal(0, noise_level, NUM_CHANNELS * 2)
        for i in range(NUM_CHANNELS * 2):
            iq_data_arr[i] = int(np.clip(noise_array[i], -2147483648, 2147483647))
        
        non_zero_channels_count = len(configured_channels)
        
        # Only process configured channels (this is the expensive part)
        for channel_num_1_based in configured_channels:
            ch_idx_0_based = channel_num_1_based - 1
            
            # Get current settings for this channel from MockCRS
            freq = self.mock_crs.frequencies.get((module_num, channel_num_1_based), 0)
            amp = self.mock_crs.amplitudes.get((module_num, channel_num_1_based), 0)
            phase_deg = self.mock_crs.phases.get((module_num, channel_num_1_based), 0)
            
            # Skip if amplitude is 0 (no signal)
            if amp == 0:
                continue
                
            total_freq = freq + nco_freq

            # Calculate the S21 response using the resonator model
            s21_complex = self.mock_crs.resonator_model.calculate_channel_response(
                module_num, channel_num_1_based, total_freq, amp, phase_deg
            )
            
            # Scale and add to existing noise
            i_val_scaled = s21_complex.real * full_scale_counts + iq_data_arr[ch_idx_0_based * 2]
            q_val_scaled = s21_complex.imag * full_scale_counts + iq_data_arr[ch_idx_0_based * 2 + 1]
            
            # Clip to int32 range and convert to int
            iq_data_arr[ch_idx_0_based * 2] = int(np.clip(i_val_scaled, -2147483648, 2147483647))
            iq_data_arr[ch_idx_0_based * 2 + 1] = int(np.clip(q_val_scaled, -2147483648, 2147483647))
        
        if seq == 0 and module_num == 1: # Log for first packet of first module
            print(f"[UDP] First packet details (module {module_num}): {non_zero_channels_count} non-zero channels.")
            print(f"[UDP] Using enhanced scale factor: {full_scale_counts:.2e}")
        
        now = datetime.now()
        ts = Timestamp(
            y=now.year % 100,
            d=now.timetuple().tm_yday,
            h=now.hour, m=now.minute, s=now.second,
            ss=int(now.microsecond * SS_PER_SECOND / 1e6), # Scaled sub-seconds
            c=0, # Carrier phase, typically 0 for mock
            sbs=0, # Sub-block sequence, typically 0 for mock
            source=TimestampPort.TEST, # Mock data source
            recent=True
        )
        
        packet = DfmuxPacket(
            magic=STREAMER_MAGIC,
            version=STREAMER_VERSION,
            serial=int(self.mock_crs.serial) if self.mock_crs.serial and self.mock_crs.serial.isdigit() else 0,
            num_modules=1, # Packet is for one module's data
            block=0, # Block number, typically 0 for continuous streaming
            fir_stage=self.mock_crs.get_fir_stage(),
            module=module_num - 1,  # DfmuxPacket module is 0-indexed
            seq=seq,
            s=iq_data_arr, # This should be the array.array('i')
            ts=ts
        )
        
        return packet.to_bytes()


class MockUDPManager:
    """Manages the lifecycle of the MockCRSUDPStreamer thread with proper cleanup."""
    def __init__(self, mock_crs):
        self.mock_crs = mock_crs
        self.udp_thread = None
        self._udp_streaming_active = False # Internal flag

    async def start_udp_streaming(self, host='127.0.0.1', port=STREAMER_PORT):
        """Start UDP streaming with proper error handling."""
        try:
            if not self._udp_streaming_active:
                self._udp_streaming_active = True
                # Ensure host is 127.0.0.1 for mock reliability
                self.udp_thread = MockCRSUDPStreamer(self.mock_crs, '127.0.0.1', port)
                self.udp_thread.start()
                await asyncio.sleep(0.1) # Give thread time to start
                print(f"[Manager] UDP Streaming started to 127.0.0.1:{port}")
                return True
            else:
                print("[Manager] UDP Streaming already active.")
                return False # Already active
        except Exception as e:
            print(f"[Manager] Error starting UDP streaming: {e}")
            self._udp_streaming_active = False
            if self.udp_thread:
                self.udp_thread.emergency_stop()
                self.udp_thread = None
            raise
    
    async def stop_udp_streaming(self):
        """Stop UDP streaming with proper cleanup and timeout."""
        if self._udp_streaming_active and self.udp_thread:
            try:
                self._udp_streaming_active = False
                self.udp_thread.stop() # Signal thread to stop
                
                # Wait for thread to finish with timeout
                try:
                    await asyncio.to_thread(self.udp_thread.join, timeout=2.0)
                except Exception as e:
                    print(f"[Manager] Error during thread join: {e}")
                
                if self.udp_thread.is_alive():
                    print("[Manager] Warning: UDP thread did not stop in time, forcing emergency stop")
                    self.udp_thread.emergency_stop()
                
                self.udp_thread = None
                print("[Manager] UDP Streaming stopped.")
                return True
                
            except Exception as e:
                print(f"[Manager] Error stopping UDP streaming: {e}")
                # Force cleanup in case of error
                if self.udp_thread:
                    self.udp_thread.emergency_stop()
                    self.udp_thread = None
                return False
        else:
            print("[Manager] UDP Streaming not active or thread missing.")
            return False # Not active or no thread

    def get_udp_streaming_status(self):
        return {
            "active": self._udp_streaming_active,
            "thread_alive": self.udp_thread is not None and self.udp_thread.is_alive()
        }
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if self._udp_streaming_active and self.udp_thread:
            print("[Manager] Destructor cleanup: stopping UDP streamer")
            self.udp_thread.emergency_stop()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self._udp_streaming_active and self.udp_thread:
            print("[Manager] Context exit cleanup: stopping UDP streamer")
            self.udp_thread.emergency_stop()
