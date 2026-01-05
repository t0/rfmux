"""
Mock CRS Device - UDP Streaming Logic.
Manages UDP packet generation and streaming for MockCRS.
"""
import asyncio
import socket
import struct
import time
import threading
import signal
import atexit
from datetime import datetime
import numpy as np
import platform

# Import ReadoutPacket and related constants from streamer
from ..streamer import (
    ReadoutPacket, Timestamp, TimestampSource,
    STREAMER_MAGIC,
    SHORT_PACKET_VERSION, LONG_PACKET_VERSION,
    SHORT_PACKET_CHANNELS, LONG_PACKET_CHANNELS,
    SS_PER_SECOND,
    STREAMER_PORT, # Default port for streaming
)


# Global registry to track all active UDP streamers for cleanup
_active_streamers = []
_cleanup_registered = False

def _emergency_cleanup():
    """Emergency cleanup function called on program exit."""
    global _active_streamers
    if _active_streamers:
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
    
    def __init__(self, mock_crs, host='239.192.0.2', port=STREAMER_PORT, modules_to_stream=None, use_multicast=True):
        super().__init__(daemon=True)
        self.mock_crs = mock_crs # Reference to MockCRS instance
        self.host = host
        self.port = port
        self.use_multicast = use_multicast
        
        # Configure for multicast or unicast
        if use_multicast:
            # Use multicast for better hardware emulation
            self.multicast_group = host if host.startswith('239.') else '239.192.0.2'
            self.multicast_port = port
            print(f"[UDP] MockCRSUDPStreamer initialized - multicast to {self.multicast_group}:{self.multicast_port}")
        else:
            # Fall back to unicast for specific testing scenarios
            self.unicast_host = host if host != '239.192.0.2' else '127.0.0.1'
            self.unicast_port = port
            print(f"[UDP] MockCRSUDPStreamer initialized - unicast to {self.unicast_host}:{self.unicast_port}")
        
        # If modules_to_stream is specified, only stream those modules
        # Otherwise, we'll determine which modules to stream based on configured channels
        self.modules_to_stream = modules_to_stream  # Can be None or a list like [1] or [1,2]
        
        self.socket = None
        self.running = False # Controlled by start/stop
        self.seq_counters = {m: 0 for m in range(1, 5)}  # Per-module sequence numbers (1-4)
        self.packets_sent = 0
        
        # Track total elapsed time for continuous timestamps across decimation changes
        self.total_elapsed_time = {m: 0.0 for m in range(1, 5)}  # Total time elapsed per module
        self.last_decimation = None

        self.timestamp_stream = None
        
        # Register for global cleanup
        global _active_streamers
        _active_streamers.append(self)
        _register_global_cleanup()
        
    def _init_socket(self):
        """Initialize the UDP socket for multicast or unicast."""
        if self.socket is None:
            if self.use_multicast:
                # Create socket for multicast
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

                
                # Enable multicast loopback so local processes can receive
                self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
                
                # Set socket send buffer size for performance
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4_000_000)

                if platform.system() == "Windows":
                    try:
                        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
                        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton("127.0.0.1"))
                    except:
                        print("Issue setting up socket on windows")
                    
                
                # Bind multicast to loopback interface (lo) for local testing
                # Using if_nametoindex to get the interface index for 'lo'  
                else:
                    for iface in ("lo", "lo0"): 
                        # lo is for Linux and lo0 is for Mac 
                        try:
                            lo_index = socket.if_nametoindex(iface)
                            # Use IP_MULTICAST_IF with the loopback address
                            # Note: IP_MULTICAST_IF expects an IP address, not an interface index
                            # For interface index, we'd use IPV6_MULTICAST_IF, but we're using IPv4
                            # So we stick with the loopback IP address
                            self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton('127.0.0.1'))
                            print(f"[UDP] Multicast bound to loopback interface ({iface}, index {lo_index})")
                            break
                        except OSError:
                            continue
                    else:
                        # Fallback if 'lo or lo0' interface not found (shouldn't happen on Linux or Mac)
                        print("[UDP] Warning: Could not find 'lo' or 'lo0' interface")
                        print("[UDP] Warning: The packets are now being launched on your network, rather than on the loopback interface")
                        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton('0.0.0.0'))
                    
                    print(f"[UDP] Multicast socket initialized for {self.multicast_group}:{self.multicast_port}")
            else:
                # Create socket for unicast
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4_000_000)
                print(f"[UDP] Unicast socket initialized")
    
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
        for (module, channel) in self.mock_crs._frequencies.keys():
            configured_modules.add(module)
        for (module, channel) in self.mock_crs._amplitudes.keys():
            configured_modules.add(module)
        for (module, channel) in self.mock_crs._phases.keys():
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
        self.running = True
        
        # Initialize start time for deterministic timestamps
        self.start_datetime = datetime.now()
        
        last_decimation = None
        
        try:
            # Initialize socket
            self._init_socket()
            
            while self.running:
                try:
                    start_time_loop = time.perf_counter()
                    
                    # Get current decimation (dynamically check each iteration)
                    dec = self.mock_crs._fir_stage
                    if dec is None:
                        dec = 6  # Default only if None, not if 0
                    sample_rate = 625e6 / (256 * 64 * (2**dec))
                    frame_time = 1.0 / sample_rate  # Time between frames
                    
                    # Log if decimation changed
                    if dec != last_decimation:
                        # Before changing decimation, save the elapsed time for each module
                        if last_decimation is not None:
                            old_sample_rate = 625e6 / (256 * 64 * (2**last_decimation))
                            for m in self.seq_counters:
                                # Add the time elapsed with the old decimation rate
                                self.total_elapsed_time[m] += self.seq_counters[m] / old_sample_rate
                        
                        print(f"[UDP] Decimation changed to stage {dec}, streaming at {sample_rate:.1f} Hz")
                        last_decimation = dec
                        self.last_decimation = dec
                        
                        # Reset sequence counters on decimation change (but keep total_elapsed_time)
                        for m in self.seq_counters:
                            self.seq_counters[m] = 0
                    
                    # Determine which modules to stream
                    if self.modules_to_stream:
                        # Use explicitly specified modules
                        modules_to_process = self.modules_to_stream
                    else:
                        # Auto-detect: only stream modules that have configured channels
                        modules_to_process = self._get_configured_modules()
                    
                    # Generate and send packets for ALL active modules quickly
                    # Each module gets its own packet at the same time instant
                    for module_num in modules_to_process:
                        if not self.running:  # Check if we should stop
                            break
                            
                        try:
                            # Generate and send packet
                            s_time = time.perf_counter()

                            packet_bytes = self.generate_packet_for_module(module_num, self.seq_counters[module_num], dec)
                            
                            if self.use_multicast:
                                if self.socket:
                                    try:
                                        bytes_sent = self.socket.sendto(packet_bytes, (self.multicast_group, self.multicast_port))
                                    except OSError as e:
                                        if e.errno == 10051: ## unreachable network on Windows, corner case on first start and for tests
                                            continue
                                        else:
                                            raise
                                else:
                                    raise RuntimeError("Socket not initialized")
                            else:
                                if self.socket:
                                    bytes_sent = self.socket.sendto(packet_bytes, (self.unicast_host, self.unicast_port))
                                else:
                                    raise RuntimeError("Socket not initialized")
                            
                            self.seq_counters[module_num] += 1
                            self.packets_sent += 1
                            
                        except Exception as e:
                            if self.running:  # Only log if we're still supposed to be running
                                print(f"[UDP] ERROR generating/sending packet for module {module_num}: {e}")
                                import traceback
                                traceback.print_exc()
                    
                    # Sleep to maintain sample rate
                    if self.running:
                        elapsed_loop = time.perf_counter() - start_time_loop
                        sleep_duration = frame_time - elapsed_loop
                        
                        if sleep_duration > 0:
                            # Simple sleep approach - let the OS handle timing
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
    

    def generate_packet_for_module(self, module_num, seq, dec):
        """Generate a ReadoutPacket for a specific module with coupled channels."""

        # Detailed timing
        timing_start = time.perf_counter()
        
        if self.mock_crs._short_packets:
            num_channels = SHORT_PACKET_CHANNELS
            version = SHORT_PACKET_VERSION
        else:
            num_channels = LONG_PACKET_CHANNELS
            version = LONG_PACKET_VERSION

        timing_array_create = time.perf_counter()

        # Pre-calculate constants from unified configuration (SoT)
        cfg = getattr(self.mock_crs, "physics_config", {}) or {}
        scale_factor = cfg.get("scale_factor", 2**21)
        full_scale = scale_factor  # Full scale in normalized units
        noise_level = cfg.get("udp_noise_level", 10.0)  # Base noise level

        # Noise is in normalized units
        noise_i = np.random.normal(0, noise_level, num_channels)
        noise_q = np.random.normal(0, noise_level, num_channels)
        channel_samples = noise_i + 1j * noise_q

        timing_noise = time.perf_counter()

        # Use module-wide coupled calculation if available
        if hasattr(self.mock_crs._resonator_model, 'calculate_module_response_coupled'):
            # Get sample rate for time-varying signals

            dec = dec
            if dec is None:
                dec = 6  # Default only if None, not if 0
            sample_rate = 625e6 / 256 / 64 / (2**dec)
            
            # Calculate time for this packet (based on sequence number)
            # Each packet represents one sample in time
            t = seq / sample_rate
            
            # Update QP densities based on current time (for pulse events)
            if hasattr(self.mock_crs._resonator_model, 'update_qp_densities_for_time'):
                self.mock_crs._resonator_model.update_qp_densities_for_time(t)
            
            # Count configured channels for debugging
            num_configured = 0
            for (mod, ch) in self.mock_crs._frequencies.keys():
                if mod == module_num:
                    num_configured += 1
            
            # Get all channel responses at once (includes coupling effects and time-varying beats)
            # Now using the unified method with start_time parameter
            channel_responses = self.mock_crs._resonator_model.calculate_module_response_coupled(
                module_num, 
                num_samples=1,  # Single sample per packet
                sample_rate=sample_rate,
                start_time=t   # Pass the current time for this packet
            )
            timing_physics = time.perf_counter()
            
            # Process each channel's response
            for ch_num_1, signal in channel_responses.items():
                ch_idx_0 = ch_num_1 - 1  # Convert to 0-based index
                
                # Scale signal from normalized units to ADC counts and add to noise
                # signal is in normalized units, full_scale converts to ADC counts
                channel_samples[ch_idx_0] += signal * full_scale
                
            timing_fill_array = time.perf_counter()
            non_zero_channels_count = len(channel_responses)
        else:
            print("ERROR WITH COUPLING CODE - this shouldn't happen with updated resonator model")
            non_zero_channels_count = 0
            timing_physics = timing_fill_array = time.perf_counter()
        
        # Calculate deterministic timestamp based on sequence number and sampling rate
        # Get the decimation stage and calculate sample rate

        dec = dec
        if dec is None:
            dec = 6  # Default only if None, not if 0
        sample_rate = 625e6 / 256 / 64 / (2**dec)
        
        # Calculate total elapsed time including previous decimation periods
        # This ensures continuous timestamps even when decimation changes
        current_period_elapsed = seq / sample_rate
        total_elapsed_seconds = self.total_elapsed_time.get(module_num, 0.0) + current_period_elapsed
        
        # Add elapsed time to start datetime to get packet timestamp
        # Use the start_datetime that was set when streaming began
        if hasattr(self, 'start_datetime') and self.start_datetime:
            base_time = self.start_datetime
        else:
            # Fallback if start_datetime not set (shouldn't happen)
            base_time = datetime.now()
            self.start_datetime = base_time
        
        # Calculate the timestamp for this packet using total elapsed time
        from datetime import timedelta
        packet_datetime = base_time + timedelta(seconds=total_elapsed_seconds)
        
        # Convert to timestamp fields
        ts = Timestamp(
            y=int(packet_datetime.year % 100),
            d=int(packet_datetime.timetuple().tm_yday),
            h=int(packet_datetime.hour),
            m=int(packet_datetime.minute),
            s=int(packet_datetime.second),
            ss=int(packet_datetime.microsecond * SS_PER_SECOND / 1e6), # Scaled sub-seconds
            c=0, # Carrier phase, typically 0 for mock
            sbs=0, # Sub-block sequence, typically 0 for mock
            source=TimestampSource.TEST, # Mock data source
            recent=True
        )

        timing_timestamp = time.perf_counter()
        packet = ReadoutPacket(
            magic=STREAMER_MAGIC,
            version=version,
            serial=int(self.mock_crs._serial) if self.mock_crs._serial and self.mock_crs._serial.isdigit() else 0,
            num_modules=1, # Packet is for one module's data
            flags=0,
            fir_stage=dec,
            module=module_num-1,  # ReadoutPacket module is 0-indexed
            seq=seq)

        # Clip and assign complex samples to packet
        packet.samples = (np.clip(channel_samples.real, -8388608, 8388607) +
                1j*np.clip(channel_samples.imag, -8388608, 8388607)).tolist()
        packet.ts = ts

        timing_packet_create = time.perf_counter()

        packet_bytes = bytes(packet)
        self.mock_crs._last_timestamp = ts
        timing_serialize = time.perf_counter()
        
        # Removed detailed timing logs - no longer needed
        
        return packet_bytes


class MockUDPManager:
    """Manages the lifecycle of the MockCRSUDPStreamer thread with proper cleanup."""
    def __init__(self, mock_crs):
        self.mock_crs = mock_crs
        self.udp_thread = None
        self._udp_streaming_active = False # Internal flag

    async def start_udp_streaming(self, host='239.192.0.2', port=STREAMER_PORT, use_multicast=True):
        """Start UDP streaming with proper error handling.
        
        Args:
            host: Multicast group address (default: '239.192.0.2') or unicast IP
            port: UDP port number (default: STREAMER_PORT)
            use_multicast: If True, use multicast; if False, use unicast
        """
        try:
            if not self._udp_streaming_active:
                self._udp_streaming_active = True
                
                # Create the streamer with multicast or unicast configuration
                self.udp_thread = MockCRSUDPStreamer(
                    self.mock_crs, 
                    host=host, 
                    port=port,
                    use_multicast=use_multicast
                )
                self.udp_thread.start()
                await asyncio.sleep(0.1) # Give thread time to start
                
                mode = "multicast" if use_multicast else "unicast"
                print(f"[Manager] UDP Streaming started ({mode}) to {host}:{port}")
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
