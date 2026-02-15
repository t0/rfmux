"""
Mock CRS Unified Streamer.

Single thread emits both slow (ReadoutPacket) and fast (PFBPacket) data,
eliminating lock contention and ensuring perfect time synchronization.

Architecture
------------
The streamer always emits slow ReadoutPackets at the decimation-determined
cadence.  When PFB is enabled (via ``enable_pfb()``), it also emits
PFBPackets in fixed batches of ``PFB_BATCH`` = 32 time samples — the
natural ratio between PFB rate and the dec-0 slow rate.

At decimation stage *d*, there are ``2^d`` PFB batches per slow sample:

    PFB rate:  625 MHz / 512  ≈ 1.22 MHz
    Slow rate: 625 MHz / 256 / 64 / 2^d
    Ratio:     32 × 2^d  PFB samples per slow sample
    Batches:   2^d  (each of 32 PFB samples)

Both packet types share the same simulation clock, giving perfect
timestamp synchronization with zero physics-lock contention.
"""

import asyncio
import socket
import time
import threading
import signal
import atexit
from datetime import datetime, timedelta
import numpy as np
import platform

from ..streamer import (
    ReadoutPacket, PFBPacket, Timestamp, TimestampSource,
    STREAMER_MAGIC, PFB_PACKET_MAGIC,
    SHORT_PACKET_VERSION, LONG_PACKET_VERSION,
    SHORT_PACKET_CHANNELS, LONG_PACKET_CHANNELS,
    SS_PER_SECOND,
    STREAMER_PORT, PFB_STREAMER_PORT,
)

# ── Global cleanup registry ──────────────────────────────────────

_active_streamers = []
_cleanup_registered = False


def _emergency_cleanup():
    """Emergency cleanup function called on program exit."""
    for s in _active_streamers[:]:
        try:
            s.emergency_stop()
        except Exception as e:
            print(f"[Streamer] Error during emergency cleanup: {e}")


def _register_global_cleanup():
    """Register global cleanup handlers (called once)."""
    global _cleanup_registered
    if not _cleanup_registered:
        atexit.register(_emergency_cleanup)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, lambda *_: _emergency_cleanup())
            except (ValueError, OSError):
                pass
        _cleanup_registered = True


# ── Constants ─────────────────────────────────────────────────────

PFB_RATE = 625e6 / 512     # ≈1.22 MHz per PFB bin
PFB_BATCH = 32              # PFB samples per dec-0 slow sample (fundamental quantum)


class MockCRSStreamer(threading.Thread):
    """Unified mock streamer: emits slow and/or PFB packets from one thread.

    Always emits slow ReadoutPackets at the decimation-determined cadence.
    When PFB is enabled (via ``enable_pfb()``), also emits PFBPackets at
    ``PFB_RATE`` in fixed batches of ``PFB_BATCH`` = 32 samples.

    At decimation stage *d*, there are ``2^d`` PFB batches per slow frame.
    Both packet types share the same simulation clock, giving perfect
    timestamp synchronization with zero lock contention.
    """

    def __init__(self, mock_crs, host='239.192.0.2', port=STREAMER_PORT,
                 modules_to_stream=None, use_multicast=True):
        super().__init__(daemon=True)
        self.mock_crs = mock_crs
        self.host = host
        self.port = port
        self.use_multicast = use_multicast
        self.modules_to_stream = modules_to_stream

        # ── Slow state ────────────────────────────────────────
        self.slow_socket = None
        self.seq_counters = {m: 0 for m in range(1, 5)}
        self.packets_sent = 0

        # ── PFB state (toggled at runtime) ────────────────────
        self.pfb_enabled = False
        self.pfb_channels: list = []
        self.pfb_module: int = 1
        self.pfb_socket = None
        self.pfb_seq = 0
        self.pfb_packets_sent = 0

        # ── Timing ────────────────────────────────────────────
        self.start_datetime = None
        self.frame_index = 0
        self.total_elapsed_time = {m: 0.0 for m in range(1, 5)}
        self.last_decimation = None

        # ── Lifecycle ─────────────────────────────────────────
        self.running = False
        _active_streamers.append(self)
        _register_global_cleanup()

        mode = "multicast" if use_multicast else "unicast"
        print(f"[Streamer] MockCRSStreamer initialized ({mode}) → {host}:{port}")

    # ── Socket management ─────────────────────────────────────

    def _make_multicast_socket(self):
        """Create a UDP socket configured for multicast on loopback."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4_000_000)

        if platform.system() == "Windows":
            try:
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                                socket.inet_aton("127.0.0.1"))
            except Exception:
                print("[Streamer] Issue setting up socket on Windows")
        else:
            for iface in ("lo", "lo0"):
                try:
                    socket.if_nametoindex(iface)
                    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                                    socket.inet_aton('127.0.0.1'))
                    break
                except OSError:
                    continue
            else:
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                                socket.inet_aton('0.0.0.0'))
        return sock

    def _init_slow_socket(self):
        if self.slow_socket is None:
            self.slow_socket = self._make_multicast_socket()
            print(f"[Streamer] Slow socket ready → {self.host}:{self.port}")

    def _init_pfb_socket(self):
        if self.pfb_socket is None:
            self.pfb_socket = self._make_multicast_socket()
            print(f"[Streamer] PFB socket ready → {self.host}:{PFB_STREAMER_PORT}")

    def _cleanup_sockets(self):
        for name, sock in [("slow", self.slow_socket), ("pfb", self.pfb_socket)]:
            if sock:
                try:
                    sock.close()
                except Exception as e:
                    print(f"[Streamer] Error closing {name} socket: {e}")
        self.slow_socket = None
        self.pfb_socket = None

    # ── PFB toggle (called from MockUDPManager) ──────────────

    def enable_pfb(self, channels, module):
        """Enable PFB packet emission (safe to call while thread is running)."""
        self.pfb_channels = list(channels)[:4]
        self.pfb_module = module
        self.pfb_seq = 0
        self.pfb_packets_sent = 0
        self._init_pfb_socket()
        self.pfb_enabled = True
        n = len(self.pfb_channels)
        mode_str = {1: 'PFB1', 2: 'PFB2', 4: 'PFB4'}.get(n, f'PFB{n}')
        print(f"[Streamer] PFB enabled — {mode_str} ch={self.pfb_channels} module={module}")

    def disable_pfb(self):
        """Disable PFB packet emission (safe to call while thread is running)."""
        was_enabled = self.pfb_enabled
        self.pfb_enabled = False
        if was_enabled:
            print(f"[Streamer] PFB disabled (sent {self.pfb_packets_sent} PFB packets)")

    # ── Lifecycle ─────────────────────────────────────────────

    def stop(self):
        """Signal the streamer to stop gracefully."""
        print("[Streamer] Stopping...")
        self.running = False

    def emergency_stop(self):
        """Force-stop immediately."""
        print(f"[Streamer] Emergency stop {id(self)}")
        self.running = False
        self._cleanup_sockets()
        if self in _active_streamers:
            _active_streamers.remove(self)

    # ── Main loop ─────────────────────────────────────────────

    def run(self):
        self.running = True
        self.start_datetime = datetime.now()
        self._init_slow_socket()

        try:
            while self.running:
                try:
                    t_wall_start = time.perf_counter()

                    # ── Read current decimation ───────────────
                    dec = self.mock_crs._fir_stage
                    if dec is None:
                        dec = 6
                    slow_rate = 625e6 / 256 / 64 / (2 ** dec)
                    frame_time = 1.0 / slow_rate

                    # Track decimation changes for continuous timestamps
                    if dec != self.last_decimation:
                        if self.last_decimation is not None:
                            old_rate = 625e6 / 256 / 64 / (2 ** self.last_decimation)
                            for m in self.seq_counters:
                                self.total_elapsed_time[m] += self.seq_counters[m] / old_rate
                            # Reset sequence counters for new rate
                            self.seq_counters = {m: 0 for m in range(1, 5)}
                        print(f"[Streamer] Decimation → stage {dec}, "
                              f"slow rate {slow_rate:.1f} Hz")
                        self.last_decimation = dec

                    # Simulation time for this frame
                    t_frame = (self.total_elapsed_time.get(1, 0.0)
                               + self.seq_counters.get(1, 0) / slow_rate)

                    # ── PFB batches (if enabled) ──────────────
                    if self.pfb_enabled and self.pfb_channels:
                        n_batches = 2 ** dec  # 1 at dec=0, 64 at dec=6
                        for batch_idx in range(n_batches):
                            if not self.running:
                                break
                            t_batch = t_frame + batch_idx * PFB_BATCH / PFB_RATE
                            self._emit_pfb_packet(t_batch, dec)

                    # ── Slow packets (always) ─────────────────
                    if self.modules_to_stream:
                        modules = self.modules_to_stream
                    else:
                        modules = self._get_configured_modules()

                    for module_num in modules:
                        if not self.running:
                            break
                        self._emit_slow_packet(module_num, t_frame, dec)

                    self.frame_index += 1

                    # ── Pace to real time ─────────────────────
                    elapsed = time.perf_counter() - t_wall_start
                    sleep_dur = frame_time - elapsed
                    if sleep_dur > 0:
                        time.sleep(sleep_dur)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    if self.running:
                        print(f"[Streamer] Error in main loop: {e}")
                        import traceback
                        traceback.print_exc()
                        time.sleep(0.01)

        finally:
            print(f"[Streamer] Stopped. Slow packets: {self.packets_sent}, "
                  f"PFB packets: {self.pfb_packets_sent}")
            self._cleanup_sockets()
            if self in _active_streamers:
                _active_streamers.remove(self)

    # ── Slow packet emission ──────────────────────────────────

    def _emit_slow_packet(self, module_num, t_frame, dec):
        """Generate and send one slow ReadoutPacket for *module_num*."""
        if self.mock_crs._short_packets:
            num_channels = SHORT_PACKET_CHANNELS
            version = SHORT_PACKET_VERSION
        else:
            num_channels = LONG_PACKET_CHANNELS
            version = LONG_PACKET_VERSION

        seq = self.seq_counters[module_num]
        slow_rate = 625e6 / 256 / 64 / (2 ** dec)

        # ── Physics ───────────────────────────────────────────
        cfg = getattr(self.mock_crs, '_physics_config', {}) or {}
        scale_factor = cfg.get('scale_factor', 2 ** 21)
        full_scale = scale_factor * 256.0
        noise_level = cfg.get('udp_noise_level', 10.0)

        noise_i = np.random.normal(0, noise_level, num_channels)
        noise_q = np.random.normal(0, noise_level, num_channels)
        channel_samples = noise_i + 1j * noise_q

        model = self.mock_crs._resonator_model
        channel_responses = model.calculate_module_response_coupled(
            module_num, num_samples=1, sample_rate=slow_rate,
            start_time=t_frame,
        )

        for ch_num_1, signal_val in channel_responses.items():
            ch_idx_0 = ch_num_1 - 1
            channel_samples[ch_idx_0] += signal_val * full_scale

        # ── Timestamp ─────────────────────────────────────────
        total_elapsed = (self.total_elapsed_time.get(module_num, 0.0)
                         + seq / slow_rate)
        pkt_dt = self.start_datetime + timedelta(seconds=total_elapsed)
        ts = Timestamp(
            y=int(pkt_dt.year % 100),
            d=int(pkt_dt.timetuple().tm_yday),
            h=int(pkt_dt.hour),
            m=int(pkt_dt.minute),
            s=int(pkt_dt.second),
            ss=int(pkt_dt.microsecond * SS_PER_SECOND / 1e6),
            c=0, sbs=0,
            source=TimestampSource.TEST,
            recent=True,
        )

        # ── Build and send packet ─────────────────────────────
        pkt = ReadoutPacket(
            magic=STREAMER_MAGIC,
            version=version,
            serial=(int(self.mock_crs._serial)
                    if self.mock_crs._serial and self.mock_crs._serial.isdigit()
                    else 0),
            num_modules=1,
            flags=0,
            fir_stage=dec,
            module=module_num - 1,  # 0-indexed
            seq=seq,
        )
        pkt.samples = (
            np.clip(channel_samples.real, -8388608, 8388607)
            + 1j * np.clip(channel_samples.imag, -8388608, 8388607)
        ).tolist()
        pkt.ts = ts
        self.mock_crs._last_timestamp = ts

        packet_bytes = bytes(pkt)
        if self.slow_socket:
            try:
                self.slow_socket.sendto(packet_bytes, (self.host, self.port))
            except OSError as e:
                if e.errno == 10051:  # Unreachable network (Windows corner case)
                    return
                raise

        self.seq_counters[module_num] += 1
        self.packets_sent += 1

    # ── PFB packet emission ───────────────────────────────────

    def _emit_pfb_packet(self, t_batch, dec):
        """Generate and send one PFB packet (PFB_BATCH=32 time samples)."""
        channels = self.pfb_channels
        n_groups = len(channels)
        N = PFB_BATCH  # 32 time samples

        # ── Physics ───────────────────────────────────────────
        cfg = getattr(self.mock_crs, '_physics_config', {}) or {}
        scale_factor = cfg.get('scale_factor', 2 ** 21)
        noise_level = cfg.get('udp_noise_level', 10.0)
        pfb_noise_scale = cfg.get('pfb_noise_scale', 64.0)
        full_scale = scale_factor * 256.0

        model = self.mock_crs._resonator_model
        responses = model.calculate_module_response_coupled(
            self.pfb_module,
            num_samples=N,
            sample_rate=PFB_RATE,
            start_time=t_batch,
            pulse_time=t_batch,
        )

        # ── Interleave: [ch1_s0, ch2_s0, …, ch1_s1, …] ──────
        total = N * n_groups
        interleaved = np.zeros(total, dtype=np.complex128)

        for slot_idx, ch in enumerate(channels):
            if ch in responses:
                sig = responses[ch]
                if not isinstance(sig, np.ndarray):
                    sig = np.full(N, sig, dtype=np.complex128)
                slot_data = sig * full_scale
            else:
                slot_data = np.zeros(N, dtype=np.complex128)

            slot_noise = (
                np.random.normal(0, noise_level * pfb_noise_scale, N)
                + 1j * np.random.normal(0, noise_level * pfb_noise_scale, N)
            )
            slot_data = slot_data + slot_noise
            interleaved[slot_idx::n_groups] = slot_data

        # ── Build PFBPacket ───────────────────────────────────
        pkt = PFBPacket()
        pkt.magic = PFB_PACKET_MAGIC
        pkt.version = 1
        pkt.mode = {1: 0, 2: 1, 4: 2}.get(n_groups, 2)
        pkt.serial = (
            int(self.mock_crs._serial)
            if self.mock_crs._serial and self.mock_crs._serial.isdigit()
            else 0
        )
        pkt.slot1 = channels[0] if len(channels) > 0 else 0
        pkt.slot2 = channels[1] if len(channels) > 1 else 0
        pkt.slot3 = channels[2] if len(channels) > 2 else 0
        pkt.slot4 = channels[3] if len(channels) > 3 else 0
        pkt.num_samples = total
        pkt.module = self.pfb_module - 1  # 0-indexed
        pkt.seq = self.pfb_seq

        clipped = (
            np.clip(interleaved.real, -8388608, 8388607)
            + 1j * np.clip(interleaved.imag, -8388608, 8388607)
        )
        pkt.samples = clipped

        # ── Timestamp ─────────────────────────────────────────
        pkt_dt = self.start_datetime + timedelta(seconds=t_batch)
        ts = Timestamp(
            y=int(pkt_dt.year % 100),
            d=int(pkt_dt.timetuple().tm_yday),
            h=int(pkt_dt.hour),
            m=int(pkt_dt.minute),
            s=int(pkt_dt.second),
            ss=int(pkt_dt.microsecond * SS_PER_SECOND / 1e6),
            c=0, sbs=0,
            source=TimestampSource.TEST,
            recent=True,
        )
        pkt.ts = ts
        self.mock_crs._last_timestamp = ts

        if self.pfb_socket:
            self.pfb_socket.sendto(bytes(pkt), (self.host, PFB_STREAMER_PORT))

        self.pfb_seq += 1
        self.pfb_packets_sent += 1

    # ── Module detection ──────────────────────────────────────

    def _get_configured_modules(self):
        """Return sorted list of modules that have configured channels."""
        configured = set()
        for (mod, _ch) in self.mock_crs._frequencies.keys():
            configured.add(mod)
        for (mod, _ch) in self.mock_crs._amplitudes.keys():
            configured.add(mod)
        for (mod, _ch) in self.mock_crs._phases.keys():
            configured.add(mod)
        modules = sorted(m for m in configured if 1 <= m <= 4)
        return modules or [1]


# ── Manager ───────────────────────────────────────────────────────

class MockUDPManager:
    """Manages the lifecycle of the unified MockCRSStreamer.

    The streamer always emits slow packets.  PFB emission is toggled
    on/off without stopping the thread — just a flag flip checked each
    iteration of the main loop.
    """

    def __init__(self, mock_crs):
        self.mock_crs = mock_crs
        self._streamer = None
        self._streaming_active = False

    async def start_udp_streaming(self, host='239.192.0.2', port=STREAMER_PORT,
                                   use_multicast=True):
        """Start the unified streamer thread (slow packets begin immediately)."""
        if self._streaming_active:
            print("[Manager] Streaming already active")
            return False

        try:
            self._streaming_active = True
            self._streamer = MockCRSStreamer(
                self.mock_crs, host=host, port=port,
                use_multicast=use_multicast,
            )
            self._streamer.start()
            await asyncio.sleep(0.1)
            print(f"[Manager] Streaming started → {host}:{port}")
            return True
        except Exception as e:
            print(f"[Manager] Error starting streaming: {e}")
            self._streaming_active = False
            if self._streamer:
                self._streamer.emergency_stop()
                self._streamer = None
            raise

    async def stop_udp_streaming(self):
        """Stop the unified streamer thread entirely."""
        if not self._streaming_active or not self._streamer:
            return False

        try:
            self._streaming_active = False
            self._streamer.stop()

            try:
                await asyncio.to_thread(self._streamer.join, timeout=2.0)
            except Exception as e:
                print(f"[Manager] Error during thread join: {e}")

            if self._streamer.is_alive():
                print("[Manager] Warning: streamer did not stop in time, "
                      "forcing emergency stop")
                self._streamer.emergency_stop()

            self._streamer = None
            print("[Manager] Streaming stopped.")
            return True
        except Exception as e:
            print(f"[Manager] Error stopping streaming: {e}")
            if self._streamer:
                self._streamer.emergency_stop()
                self._streamer = None
            return False

    def get_udp_streaming_status(self):
        """Return status dict for both slow and PFB streams."""
        s = self._streamer
        return {
            "active": self._streaming_active,
            "thread_alive": s is not None and s.is_alive(),
            "pfb_active": s.pfb_enabled if s else False,
            "pfb_thread_alive": s is not None and s.is_alive() and s.pfb_enabled,
        }

    # ── PFB toggle (no thread restart) ───────────────────────

    async def start_pfb_streaming(self, channels, module, host=None,
                                   port=None, use_multicast=True):
        """Enable PFB packet emission on the already-running streamer."""
        if not self._streaming_active or not self._streamer:
            raise RuntimeError("Cannot enable PFB: slow streamer not running. "
                               "Call start_udp_streaming() first.")
        self._streamer.enable_pfb(channels, module)
        await asyncio.sleep(0.05)
        return True

    async def stop_pfb_streaming(self):
        """Disable PFB packet emission (slow continues uninterrupted)."""
        if self._streamer:
            self._streamer.disable_pfb()
        return True

    # ── Cleanup ───────────────────────────────────────────────

    def _emergency_stop_all(self):
        if self._streamer:
            self._streamer.emergency_stop()
            self._streamer = None
        self._streaming_active = False

    def __del__(self):
        if self._streaming_active and self._streamer:
            print("[Manager] Destructor cleanup")
            self._emergency_stop_all()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._streaming_active and self._streamer:
            print("[Manager] Context exit cleanup")
            self._emergency_stop_all()
