"""
Mock CRS Unified Streamer.

Single thread emits both slow (ReadoutPacket) and fast (PFBPacket) data,
eliminating lock contention and ensuring perfect time synchronization.

Architecture
------------
The streamer always emits slow ReadoutPackets at the decimation-determined
cadence.  When PFB is enabled (via ``enable_pfb()``), it also generates
the PFB samples of each slow frame in one physics call and sends them as
1000-sample PFBPackets, the size the hardware sends, carrying any
remainder into the next frame.

At decimation stage *d*, a slow frame holds ``2^d`` sub-batches of
``PFB_BATCH`` = 64 PFB samples, the grid on which pulse triggers are
checked:

    PFB rate:  625 MHz / 256  ≈ 2.44 MHz  (complex samples)
    Slow rate: 625 MHz / 256 / 64 / 2^d
    Frame:     64 × 2^d  PFB samples per slow sample

Both packet types share the same simulation clock, so their timestamps
agree, and one thread means no physics-lock contention between them.
"""

import asyncio
import socket
import time
import threading
import signal
import atexit
import traceback
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
from .config import MOCK_DEFAULTS
from ..core.transferfunctions import (
    CIC1_DECIMATION, PFB_SAMPLING_FREQ, decimated_stream_delay_s,
    decimation_to_sampling)

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

PFB_BATCH = 64              # PFB samples per dec-0 slow sample (fundamental quantum)

#: Wire mode for each channel count the PFB packet can express.
PFB_MODE_FOR_CHANNELS = {1: 0, 2: 1, 4: 2}


class MockCRSStreamer(threading.Thread):
    """One thread, one simulation clock: slow ReadoutPackets always, PFBPackets when enabled."""

    def __init__(self, mock_crs, host='239.192.0.2', port=STREAMER_PORT,
                 modules_to_stream=None):
        super().__init__(daemon=True)
        self.mock_crs = mock_crs
        self.host = host
        self.port = port
        self.modules_to_stream = modules_to_stream

        # ── Slow state ────────────────────────────────────────
        self.slow_socket = None
        self.seq_counters = {m: 0 for m in range(1, 5)}   # wire sequence numbers
        self.packets_sent = 0
        self._failing = set()       # streams (module number or "PFB") in an outage

        # ── PFB state (owned by the streamer thread) ──────────
        self.pfb_enabled = False
        self.pfb_channels: list = []
        self.pfb_module: int = 1
        self.pfb_socket = None
        self.pfb_seq = 0
        self.pfb_packets_sent = 0
        # enable_pfb() is called from the server thread; it leaves its
        # (channels, module) here and the streamer thread adopts it at
        # its next block, so a frame is never cut with one channel set
        # and stamped with another.
        self._pfb_request = None
        self._pfb_adopted = None
        # Samples generated but not yet sent, and the time of the first
        # of them: a frame's physics is one call, packets are cut from
        # it at the hardware's size.
        self._pfb_buf = None
        self._pfb_buf_t0 = 0.0

        # ── Timing ────────────────────────────────────────────
        # Stream seconds at the start of the next block.  The physics,
        # the slow stamps and the PFB stamps all read it, so the streams
        # agree whichever modules carry tones.
        self.start_datetime = None
        self.t_stream = 0.0
        self.last_decimation = None

        # ── Lifecycle ─────────────────────────────────────────
        self.running = False
        _active_streamers.append(self)
        _register_global_cleanup()

        print(f"[Streamer] MockCRSStreamer initialized → {host}:{port}")

    # ── Socket management ─────────────────────────────────────

    def _make_multicast_socket(self):
        """UDP send socket; the multicast options are inert for a
        unicast destination."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        # TTL 0 means "never leaves this host". The mock streams to the
        # same group real boards use, so an escaped packet would reach
        # colleagues' receivers -- and at stage 0 that is tens of MB/s of
        # simulated detectors on the lab network.
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 0)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4_000_000)

        if platform.system() == "Windows":
            try:
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
        """Request PFB emission of *channels* on *module*; the streamer
        thread adopts it at its next block."""
        if len(channels) not in PFB_MODE_FOR_CHANNELS:
            raise ValueError(f"PFB packets carry 1, 2 or 4 channels, not "
                             f"{len(channels)}")
        self._init_pfb_socket()
        self._pfb_request = (tuple(channels), module)
        self.pfb_enabled = True
        print(f"[Streamer] PFB enabled — PFB{len(channels)} ch={list(channels)} "
              f"module={module}")

    @property
    def pfb_adopted(self):
        return self._pfb_request is self._pfb_adopted

    def _adopt_pfb_request(self):
        req = self._pfb_request
        if req is None or req is self._pfb_adopted:
            return
        self._pfb_adopted = req
        self.pfb_channels, self.pfb_module = list(req[0]), req[1]
        self.pfb_seq = 0
        self.pfb_packets_sent = 0
        self._pfb_buf = None

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
                t_wall_start = time.perf_counter()
                try:
                    stream_seconds = self._run_block()
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    if not self.running:
                        break
                    print(f"[Streamer] Error in main loop: {e}")
                    traceback.print_exc()
                    stream_seconds = self.SLOW_BLOCK_SECONDS

                # Pace to real time whether or not the block succeeded,
                # so a failure cannot speed up the streams that work.
                sleep_dur = stream_seconds - (time.perf_counter() - t_wall_start)
                if sleep_dur > 0:
                    time.sleep(sleep_dur)

        finally:
            print(f"[Streamer] Stopped. Slow packets: {self.packets_sent}, "
                  f"PFB packets: {self.pfb_packets_sent}")
            self._cleanup_sockets()
            if self in _active_streamers:
                _active_streamers.remove(self)

    def _run_block(self):
        """One block of frames on every stream; returns the stream
        seconds it covered."""
        dec = self.mock_crs._fir_stage
        if dec is None:
            dec = 6
        slow_rate = decimation_to_sampling(dec)

        if dec != self.last_decimation:
            # Sequence numbers restart with the rate; the clock runs on.
            self.seq_counters = {m: 0 for m in range(1, 5)}
            print(f"[Streamer] Decimation → stage {dec}, "
                  f"slow rate {slow_rate:.1f} Hz")
            self.last_decimation = dec
            self._pfb_buf = None

        t_block = self.t_stream
        n_block = self.slow_block_len(dec)
        # The clock advances before anything is emitted: a stream that
        # fails must not hold the others at a repeated block time.
        self.t_stream = t_block + n_block / slow_rate

        self._adopt_pfb_request()
        pfb = self.pfb_enabled and bool(self.pfb_channels)
        if pfb:
            # Schedule the block's pulses on the PFB trigger grid before
            # either stream is evaluated.  advance_pulses_to keeps only
            # the pulses alive at the start of the span it is given, and
            # the slow block, evaluated from t_block, must see every
            # pulse the PFB frames do.
            dt = PFB_BATCH / PFB_SAMPLING_FREQ
            n_steps = n_block * 2 ** dec
            self.mock_crs._resonator_model.advance_pulses_to(
                t_block + (n_steps - 1) * dt, n_steps, dt)
        else:
            self._pfb_buf = None

        modules = self.modules_to_stream or self._get_configured_modules()
        for module_num in modules:
            if not self.running:
                break
            self._emit_guarded(module_num, self._emit_slow_block,
                               module_num, t_block, dec, n_block)

        if pfb:
            for k in range(n_block):
                if not self.running:
                    break
                if not self._emit_guarded("PFB", self._emit_pfb_frame,
                                          t_block + k / slow_rate, dec):
                    break

        return n_block / slow_rate

    def _emit_guarded(self, stream, emit, *args):
        """Run one stream's emitter; a failure is said once per outage
        and leaves the other streams their pace.  True on success."""
        try:
            emit(*args)
        except Exception as e:
            if stream not in self._failing:
                self._failing.add(stream)
                print(f"[Streamer] {stream} physics failed, its packets "
                      f"stop until it succeeds: {e!r}")
            return False
        self._failing.discard(stream)
        return True

    # ── Shared packet pieces ──────────────────────────────────

    def _timestamp_at(self, seconds):
        pkt_dt = self.start_datetime + timedelta(seconds=seconds)
        return Timestamp(
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

    def _serial(self):
        serial = self.mock_crs._serial
        return int(serial) if serial and serial.isdigit() else 0

    def _scale_and_noise(self):
        """(full scale of a unit response in ADC counts, slow-stream noise
        sigma): the scale get_samples reports, so a consumer reading the
        stream through the receiver sees the same counts it would read
        through get_samples."""
        cfg = getattr(self.mock_crs, '_physics_config', {}) or {}
        return (cfg.get('scale_factor', MOCK_DEFAULTS['scale_factor']),
                cfg.get('udp_noise_level', MOCK_DEFAULTS['udp_noise_level']))

    # ── Slow packet emission ──────────────────────────────────

    #: The slow stream is generated a block of frames at a time: one
    #: physics call per block, then one packet per frame.  Per-sample
    #: overhead dominated the single-frame path at many tones (100
    #: tones: 7 ms a sample against a 1.7 ms budget at stage 6, a
    #: quarter of real time; 0.7 ms a sample in blocks of 32).  About
    #: this much stream time per block, so the latency it adds stays
    #: under what the board's own buffering already is.
    SLOW_BLOCK_SECONDS = 0.05
    SLOW_BLOCK_MAX = 128

    @classmethod
    def slow_block_len(cls, dec) -> int:
        return max(1, min(cls.SLOW_BLOCK_MAX,
                          int(round(cls.SLOW_BLOCK_SECONDS
                                    * decimation_to_sampling(dec)))))

    def _emit_slow_block(self, module_num, t_frame, dec, n):
        """*n* consecutive slow frames for *module_num* from one physics
        call, sent as *n* packets with their own sequence numbers and
        timestamps."""
        if self.mock_crs._short_packets:
            num_channels = SHORT_PACKET_CHANNELS
        else:
            num_channels = LONG_PACKET_CHANNELS
        slow_rate = decimation_to_sampling(dec)
        full_scale, noise_level = self._scale_and_noise()

        model = self.mock_crs._resonator_model
        # Pulses that would start inside the block are scheduled first,
        # on the frame grid; with PFB on, _run_block has already
        # scheduled them on the finer PFB grid and this only prunes.
        if n > 1:
            model.advance_pulses_to(t_frame + (n - 1) / slow_rate, n,
                                    1.0 / slow_rate)
        # pulse_time is explicit (same escape hatch the PFB emitter uses):
        # update_qp_densities_for_time is a monotonic ratchet, and
        # advance_pulses_to has already moved last_update_time to the
        # end of the block — without this the slow stream would be
        # evaluated there, skewing it by up to one block.
        channel_responses = model.calculate_module_response_coupled(
            module_num, num_samples=n, sample_rate=slow_rate,
            start_time=t_frame, pulse_time=t_frame,
        )
        signal = np.zeros((n, num_channels), dtype=complex)
        for ch_num_1, signal_val in channel_responses.items():
            signal[:, ch_num_1 - 1] = np.atleast_1d(signal_val) * full_scale

        noise = (np.random.normal(0, noise_level, (n, num_channels))
                 + 1j * np.random.normal(0, noise_level, (n, num_channels)))
        for k in range(n):
            self._send_slow_packet(module_num, dec, signal[k] + noise[k],
                                   t_frame + k / slow_rate)

    def _send_slow_packet(self, module_num, dec, channel_samples, t_frame):
        """Stamp, build and send one slow packet whose samples describe
        stream time *t_frame*; advances the module's sequence number."""
        version = (SHORT_PACKET_VERSION if self.mock_crs._short_packets
                   else LONG_PACKET_VERSION)
        seq = self.seq_counters[module_num]

        # The samples describe the frame time; the hardware stamps the
        # packet later by the CIC group delay, so the mock does too.
        # TEMPORARY firmware behaviour: delete this term when the RTL
        # timestamps the decimated stream at its filter centroid (and
        # zero the default in DualPulseCaptureSession alongside).
        ts = self._timestamp_at(t_frame + decimated_stream_delay_s(dec))

        pkt = ReadoutPacket(
            magic=STREAMER_MAGIC,
            version=version,
            serial=self._serial(),
            num_modules=1,
            flags=0,
            fir_stage=dec | (0x8 if self.mock_crs._short_packets else 0),
            module=module_num - 1,  # 0-indexed
            seq=seq,
        )
        pkt[:] = (
            np.clip(channel_samples.real, -8388608, 8388607)
            + 1j * np.clip(channel_samples.imag, -8388608, 8388607)
        )
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

    #: Time samples per PFB packet: the hardware sends 1000 complex
    #: samples a packet, and the receiver reads exactly that many bytes
    #: per datagram, so the mock cuts its stream the same way.
    PFB_PACKET_SAMPLES = 1000

    def _emit_pfb_frame(self, t_frame, dec):
        """Generate one slow frame's worth of PFB samples in one physics
        call and send whatever full packets that makes.

        A frame is ``2**dec`` sub-batches of PFB_BATCH samples.  Trigger
        checks run on that grid before the physics, so pulse start
        times do not depend on the frame length; the physics is then
        evaluated once across the frame.  The
        remainder that does not fill a packet is carried to the next
        frame, with the time of its first sample.
        """
        channels = self.pfb_channels
        n_groups = len(channels)
        n_sub = 2 ** dec
        n_time = n_sub * PFB_BATCH
        full_scale, noise_level = self._scale_and_noise()
        # The slow stream is this one decimated by CIC1_DECIMATION *
        # 2**dec, so for white readout noise the PFB sigma is root that
        # many times the slow floor udp_noise_level names.
        pfb_noise = noise_level * np.sqrt(CIC1_DECIMATION * n_sub)
        model = self.mock_crs._resonator_model
        model.advance_pulses_to(t_frame + (n_sub - 1) * PFB_BATCH / PFB_SAMPLING_FREQ,
                                n_sub, PFB_BATCH / PFB_SAMPLING_FREQ)
        responses = model.calculate_module_response_coupled(
            self.pfb_module,
            num_samples=n_time,
            sample_rate=PFB_SAMPLING_FREQ,
            start_time=t_frame,
            pulse_time=t_frame,
        )
        # ── Interleave: [ch1_s0, ch2_s0, …, ch1_s1, …] ──────
        interleaved = np.zeros(n_time * n_groups, dtype=np.complex128)
        for slot_idx, ch in enumerate(channels):
            if ch in responses:
                sig = responses[ch]
                if not isinstance(sig, np.ndarray):
                    sig = np.full(n_time, sig, dtype=np.complex128)
                slot_data = sig * full_scale
            else:
                slot_data = np.zeros(n_time, dtype=np.complex128)
            slot_noise = (np.random.normal(0, pfb_noise, n_time)
                          + 1j * np.random.normal(0, pfb_noise, n_time))
            interleaved[slot_idx::n_groups] = slot_data + slot_noise
        if self._pfb_buf is None or len(self._pfb_buf) == 0:
            self._pfb_buf = interleaved
            self._pfb_buf_t0 = t_frame
        else:
            self._pfb_buf = np.concatenate((self._pfb_buf, interleaved))
        per_packet = (self.PFB_PACKET_SAMPLES // n_groups) * n_groups
        while len(self._pfb_buf) >= per_packet and self.running:
            self._send_pfb_packet(self._pfb_buf[:per_packet],
                                  self._pfb_buf_t0)
            self._pfb_buf = self._pfb_buf[per_packet:]
            self._pfb_buf_t0 += (per_packet // n_groups) / PFB_SAMPLING_FREQ

    def _send_pfb_packet(self, interleaved, t_first):
        """Build and send one PFBPacket holding *interleaved* samples,
        stamped with the time of its first sample."""
        channels = self.pfb_channels
        pkt = PFBPacket()
        pkt.magic = PFB_PACKET_MAGIC
        pkt.version = 1
        pkt.mode = PFB_MODE_FOR_CHANNELS[len(channels)]
        pkt.serial = self._serial()
        # Slot fields are 0-indexed on the wire, like the module field.
        pkt.slot1 = channels[0] - 1 if len(channels) > 0 else 0
        pkt.slot2 = channels[1] - 1 if len(channels) > 1 else 0
        pkt.slot3 = channels[2] - 1 if len(channels) > 2 else 0
        pkt.slot4 = channels[3] - 1 if len(channels) > 3 else 0
        pkt.num_samples = len(interleaved)
        pkt.module = self.pfb_module - 1  # 0-indexed
        pkt.seq = self.pfb_seq
        clipped = (
            np.clip(interleaved.real, -8388608, 8388607)
            + 1j * np.clip(interleaved.imag, -8388608, 8388607)
        )
        pkt[:] = clipped
        ts = self._timestamp_at(t_first)
        pkt.ts = ts
        self.mock_crs._last_timestamp = ts
        if self.pfb_socket:
            self.pfb_socket.sendto(bytes(pkt), (self.host, PFB_STREAMER_PORT))
        self.pfb_seq += 1
        self.pfb_packets_sent += 1

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

LOOPBACK_UNICAST = "127.0.0.1"


def select_stream_destination(*, use_multicast=True):
    """Multicast if this machine can, loopback unicast if it cannot.

    Real hardware always multicasts. Mock mode streams the same way so
    that the transport under test is the transport people actually run,
    and so that a machine whose multicast is broken says so here --
    where it is cheap to debug -- rather than the first time a board is
    plugged in.

    Falling back is not silent: multicast trouble is common and easy to
    mistake for "the mock is broken", so the failing step and how to fix
    it are printed, along with the fact that the fallback is available
    to the mock and not to a CRS.
    """
    if not use_multicast:
        print(f"[Streamer] Multicast disabled; using unicast on "
              f"{LOOPBACK_UNICAST}")
        return LOOPBACK_UNICAST

    from ..streamer import MULTICAST_GROUP, check_multicast_loopback

    check = check_multicast_loopback()
    if check.ok:
        return MULTICAST_GROUP

    print("[Streamer] Multicast does not work on this machine; falling "
          "back to unicast on " + LOOPBACK_UNICAST + ".")
    print("[Streamer] The mock will run normally, but a real CRS "
          "multicasts and has no fallback:")
    print(check.report())
    return LOOPBACK_UNICAST


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

    async def start_udp_streaming(self, host=None, port=STREAMER_PORT,
                                   use_multicast=True):
        """Start the unified streamer thread (slow packets begin immediately).

        ``host=None`` picks the destination: the multicast group if
        multicast works on this machine, loopback unicast if it does
        not. Pass a host explicitly to override, or
        ``use_multicast=False`` to skip the check entirely.
        """
        if self._streaming_active:
            print("[Manager] Streaming already active")
            return False

        if host is None:
            host = select_stream_destination(use_multicast=use_multicast)

        try:
            self._streaming_active = True
            self._streamer = MockCRSStreamer(self.mock_crs, host=host, port=port)
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
        s = self._streamer
        return {
            "active": self._streaming_active,
            "thread_alive": s is not None and s.is_alive(),
        }

    # ── PFB toggle (no thread restart) ───────────────────────

    async def start_pfb_streaming(self, channels, module):
        """Enable PFB packet emission on the already-running streamer."""
        if not self._streaming_active or not self._streamer:
            raise RuntimeError("Cannot enable PFB: slow streamer not running. "
                               "Call start_udp_streaming() first.")
        self._streamer.enable_pfb(channels, module)
        # Return once the streamer thread has adopted the request, so
        # readers of its state see the new channel set.
        deadline = time.monotonic() + 0.5
        while not self._streamer.pfb_adopted and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
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
