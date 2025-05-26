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
from datetime import datetime
import numpy as np # For np.clip in generate_packet

# Import DfmuxPacket and related constants from streamer
from ..streamer import (
    DfmuxPacket, Timestamp, TimestampPort,
    STREAMER_MAGIC, STREAMER_VERSION, NUM_CHANNELS, SS_PER_SECOND,
    STREAMER_PORT # Default port for streaming
)

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
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # SO_SNDBUF might not be necessary for local unicast but doesn't hurt
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4_000_000)
        
        self.running = False # Controlled by start/stop
        self.seq_counters = {m: 0 for m in range(1, 5)}  # Per-module sequence numbers (1-4)
        self.packets_sent = 0
        print(f"[UDP] MockCRSUDPStreamer initialized - unicast to {self.unicast_host}:{self.unicast_port}")
        
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
        """Stop the streaming thread"""
        print("[UDP] Stopping UDP streamer...")
        self.running = False
        
    def run(self):
        """Stream UDP packets at the configured sample rate"""
        print("[UDP] UDP streaming thread started")
        self.running = True
        packet_count_interval = 0 # Packets sent in the current 1-second interval
        last_log_time = time.time()
        
        while self.running:
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
                try:
                    packet_bytes = self.generate_packet_for_module(module_num, self.seq_counters[module_num])
                    bytes_sent = self.socket.sendto(packet_bytes, (self.unicast_host, self.unicast_port))
                    
                    self.seq_counters[module_num] += 1
                    self.packets_sent += 1
                    packet_count_interval += 1
                    
                    if self.packets_sent <= 20: # Log first few packets (e.g., 5 per module)
                        print(f"[UDP] Sent unicast packet #{self.packets_sent}: module={module_num}, seq={self.seq_counters[module_num]-1}, size={bytes_sent} bytes to {self.unicast_host}:{self.unicast_port}")
                    
                except Exception as e:
                    print(f"[UDP] ERROR generating/sending packet for module {module_num}: {e}")
                    import traceback
                    traceback.print_exc()
            
            current_time = time.time()
            if current_time - last_log_time >= 1.0:
                print(f"[UDP] Status: {packet_count_interval} packets sent in last second (total {self.packets_sent})")
                packet_count_interval = 0
                last_log_time = current_time
            
            # Sleep to maintain sample rate
            fir_stage = self.mock_crs.get_fir_stage()
            # sample_rate = 625e6 / (256 * 64 * (2**fir_stage)) # Original calculation
            # This sample rate is per channel. Packet rate is per module.
            # Assuming one packet per module per frame.
            # If NUM_SAMPLES_PER_PACKET = 1 (as implied by DfmuxPacket structure for PFB data)
            # then packet rate = sample_rate.
            # Let's assume a fixed packet rate for simplicity in mock, e.g., 100 Hz per module
            # This means each module sends a packet every 10ms.
            # Since we send for 4 modules in a loop, the loop should run 400 times per second.
            # Or, if packets are for *all* channels in a module at once, then it's frame rate.
            
            # The original mock.py used this:
            sample_rate_per_channel = 625e6 / (256 * 64 * (2**fir_stage))
            # This is the rate at which individual channel data is updated.
            # A DfmuxPacket contains data for ALL channels for one module at one time instant.
            # So, the packet rate per module is sample_rate_per_channel.
            frame_time = 1.0 / sample_rate_per_channel
            
            elapsed_loop = time.perf_counter() - start_time_loop
            sleep_duration = frame_time - elapsed_loop
            if sleep_duration > 0:
                time.sleep(sleep_duration)
        
        self.socket.close() # Close socket when thread stops
        print(f"[UDP] UDP streaming thread stopped. Total packets sent: {self.packets_sent}")
    
    def generate_packet_for_module(self, module_num, seq):
        """Generate a DfmuxPacket for a specific module."""
        # Create channel data array (NUM_CHANNELS = 1024, so 2048 int32s for I & Q)
        # This array should store int32 values.
        iq_data_arr = array.array("i", [0] * (NUM_CHANNELS * 2))
        
        non_zero_channels_count = 0
        for ch_idx_0_based in range(NUM_CHANNELS): # 0 to 1023
            channel_num_1_based = ch_idx_0_based + 1
            
            # Get current settings for this channel from MockCRS
            # These are the commanded settings
            freq = self.mock_crs.frequencies.get((module_num, channel_num_1_based), 0)
            amp = self.mock_crs.amplitudes.get((module_num, channel_num_1_based), 0)
            phase_deg = self.mock_crs.phases.get((module_num, channel_num_1_based), 0)
            nco_freq = self.mock_crs.get_nco_frequency(module=module_num) or 0
            total_freq = freq + nco_freq

            # Calculate the S21 response using the resonator model
            s21_complex = self.mock_crs.resonator_model.calculate_channel_response(
                module_num, channel_num_1_based, total_freq, amp, phase_deg
            )
            
            # Convert complex S21 to scaled int32 I and Q values
            # This scaling needs to be consistent with how real hardware/parser expects it.
            # A common approach is to scale to a fraction of the int32 range.
            # Max int32 is 2^31 - 1. Let's use a scale factor.
            # Original mock.py used scale = 32767 (for 16-bit idea, but stored in int32)
            # Let's use a larger portion of int32 range for better resolution if needed.
            # Example: scale to fill half of 24-bit range (common ADC bit depth)
            # Max 24-bit signed is 2^23 - 1 = 8388607
            # If s21_complex.real/imag are typically around [-1, 1], then:
            scale_factor = 8000000.0 # Scale to roughly +/- 8 million
            
            # Add random noise even when amplitude is 0
            # This simulates ADC noise that's always present
            noise_level = 100.0  # Base noise level (in ADC counts)
            i_noise = np.random.normal(0, noise_level)
            q_noise = np.random.normal(0, noise_level)
            
            i_val_scaled = s21_complex.real * scale_factor + i_noise
            q_val_scaled = s21_complex.imag * scale_factor + q_noise
            
            # Clip to int32 range and convert to int
            iq_data_arr[ch_idx_0_based * 2] = int(np.clip(i_val_scaled, -2147483648, 2147483647))
            iq_data_arr[ch_idx_0_based * 2 + 1] = int(np.clip(q_val_scaled, -2147483648, 2147483647))
            
            if iq_data_arr[ch_idx_0_based * 2] != 0 or iq_data_arr[ch_idx_0_based * 2 + 1] != 0:
                non_zero_channels_count += 1
        
        if seq == 0 and module_num == 1: # Log for first packet of first module
            print(f"[UDP] First packet details (module {module_num}): {non_zero_channels_count} non-zero channels.")
        
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
    """Manages the lifecycle of the MockCRSUDPStreamer thread."""
    def __init__(self, mock_crs):
        self.mock_crs = mock_crs
        self.udp_thread = None
        self._udp_streaming_active = False # Internal flag

    async def start_udp_streaming(self, host='127.0.0.1', port=STREAMER_PORT):
        if not self._udp_streaming_active:
            self._udp_streaming_active = True
            # Ensure host is 127.0.0.1 for mock reliability
            self.udp_thread = MockCRSUDPStreamer(self.mock_crs, '127.0.0.1', port)
            self.udp_thread.start()
            await asyncio.sleep(0.1) # Give thread time to start
            print(f"[Manager] UDP Streaming started to 127.0.0.1:{port}")
            return True
        print("[Manager] UDP Streaming already active.")
        return False # Already active
    
    async def stop_udp_streaming(self):
        if self._udp_streaming_active and self.udp_thread:
            self._udp_streaming_active = False
            self.udp_thread.stop() # Signal thread to stop
            # Wait for thread to finish. Note: join() is blocking.
            # If called from async, need to run join in executor or handle carefully.
            # For simplicity here, assuming short join or that blocking is acceptable in mock context.
            # A more robust async shutdown would use an event.
            await asyncio.to_thread(self.udp_thread.join, timeout=1.0)
            if self.udp_thread.is_alive():
                print("[Manager] Warning: UDP thread did not stop in time.")
            self.udp_thread = None
            print("[Manager] UDP Streaming stopped.")
            return True
        print("[Manager] UDP Streaming not active or thread missing.")
        return False # Not active or no thread

    def get_udp_streaming_status(self):
        return {
            "active": self._udp_streaming_active,
            "thread_alive": self.udp_thread is not None and self.udp_thread.is_alive()
        }
