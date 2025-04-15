"""
py_get_samples: an experimental pure-Python, client-side implementation of the
get_samples() call with dynamic determination of the multicast interface IP address.

This code retrieves time-domain samples from the CRS device. It can optionally
compute a spectrum (via Welch) in either 'psd' (V^2/Hz) or 'ps' (V^2) form.
If reference='relative', we report them in dBc or dBc/Hz (carrier power = DC bin).
If reference='absolute', we report in dBm or dBm/Hz.

Important notes:
  - "average=True" => returns time-domain mean/std dev only (no spectrum).
  - "channel=None" => returns data for all channels.
  - "scaling" => 'psd' or 'ps', determines whether we interpret Welch's data as
    power spectral density (V^2/Hz) or power spectrum (V^2).
  - "nsegments" => how many segments to use for the Welch method (like nperseg),
    or for chunk-based logic. By default 1 => no segmenting beyond the entire data array.
  - "spectrum_cutoff" => fraction of Nyquist (0..1) to keep. Default 0.9 => up to 0.9*(fs/2).

The final output:
  - time-domain arrays in 'results["i"]', 'results["q"]'.
  - if return_spectrum=True, then 'results["spectrum"]' includes frequency axes
    and spectral data, e.g. 'freq_iq', 'freq_dsb' plus entries named:
      '{scaling}_i', '{scaling}_q', '{scaling}_dual_sideband'
    in either dBc or dBm units, depending on 'reference'.

"""

import array
import asyncio
import contextlib
import ctypes
import enum
import numpy as np
import socket
import struct
import sys
import warnings

from ...core.hardware_map import macro
from ...core.schema import CRS
from ...tuber.codecs import TuberResult
from ...core.transferfunctions import VOLTS_PER_ROC, spectrum_from_slow_tod

from dataclasses import dataclass, asdict, astuple

STREAMER_PORT = 9876
STREAMER_HOST = "239.192.0.2"
STREAMER_LEN = 8240
STREAMER_MAGIC = 0x5344494B
STREAMER_VERSION = 5
NUM_CHANNELS = 1024
SS_PER_SECOND = 125000000

# This seems like a long timeout - in practice, we can see long delays when
# packets are flowing normally
STREAMER_TIMEOUT = 60

# Source-specific multicasting support was added in 3.12.0
if not hasattr(socket, 'IP_ADD_SOURCE_MEMBERSHIP'):
    raise NotImplementedError(
            "Module 'socket' doesn't have source-specific multicasting (SSM) "
            "support, which was added in Python 3.12.0. Refer to "
            "https://github.com/python/cpython/issues/89415 and the Python "
            "3.12.0 release notes for details."
    )


class InAddr(ctypes.Structure):
    _fields_ = [("s_addr", ctypes.c_uint32)]

    def __init__(self, ip: bytes):
        # Accept IP address as a string
        self.s_addr = int.from_bytes(socket.inet_aton(ip), byteorder='little')


class IPMreqSource(ctypes.Structure):
    '''
    The order of fields in this structure is implementation-specific.
    Notably, fields have different order in Windows and Linux.
    '''

    match sys.platform:
        case 'win32':
            _fields_ = [
                ("imr_multiaddr", InAddr),
                ("imr_sourceaddr", InAddr),
                ("imr_interface", InAddr),
        ]
        case 'linux':
            _fields_ = [
                ("imr_multiaddr", InAddr),
                ("imr_interface", InAddr),
                ("imr_sourceaddr", InAddr),
            ]
        case _:
            raise NotImplementedError(f"Source-specific multicast support for {sys.platform} is incomplete.")


class TimestampPort(str, enum.Enum):
    BACKPLANE = "BACKPLANE"
    TEST = "TEST"
    SMA = "SMA"
    GND = "GND"


@dataclass(order=True)
class Timestamp:
    y: np.int32 # 0-99
    d: np.int32 # 1-366
    h: np.int32 # 0-23
    m: np.int32 # 0-59
    s: np.int32 # 0-59
    ss: np.int32 # 0-SS_PER_SECOND-1
    c: np.int32
    sbs: np.int32

    source: TimestampPort
    recent: bool

    def renormalize(self):
        """
        Normalizes the timestamp fields, carrying over seconds->minutes->hours->days
        as needed. Ignores leap years for day-of-year handling.
        """
        old = astuple(self)

        carry, self.ss = divmod(self.ss, SS_PER_SECOND)
        self.s += carry

        carry, self.s = divmod(self.s, 60)
        self.m += carry

        carry, self.m = divmod(self.m, 60)
        self.h += carry

        carry, self.h = divmod(self.h, 24)
        self.d += carry

        self.d -= 1  # convert to zero-indexed
        carry, self.d = divmod(self.d, 365)  # does not work on leap day
        self.d += 1  # restore to 1-indexed
        self.y += carry
        self.y %= 100  # and roll over


    @classmethod
    def from_bytes(cls, data):
        # Unpack the timestamp data from the last portion of the packet
        args = struct.unpack("<8I", data)
        ts = Timestamp(*args, recent=False, source=TimestampPort.GND)

        # Decode the source from the 'c' field
        source_bits = (ts.c >> 29) & 0x3
        if source_bits == 0:
            ts.source = TimestampPort.BACKPLANE
        elif source_bits == 1:
            ts.source = TimestampPort.TEST
        elif source_bits == 2:
            ts.source = TimestampPort.SMA
        elif source_bits == 3:
            ts.source = TimestampPort.GND
        else:
            raise RuntimeError("Unexpected timestamp source!")

        # Decode the 'recent' flag from the 'c' field
        ts.recent = bool(ts.c & 0x80000000)

        # Mask off the higher bits to get the actual count
        ts.c &= 0x1FFFFFFF

        return ts

    @classmethod
    def from_TuberResult(cls, ts):
        # Convert the TuberResult into a dictionary we can use
        data = { f: getattr(ts, f) for f in dir(ts) if not f.startswith('_')}
        return Timestamp(**data)


@dataclass
class DfmuxPacket:
    magic: np.uint32
    version: np.uint16
    serial: np.uint16

    num_modules: np.uint8
    block: np.uint8
    fir_stage: np.uint8
    module: np.uint8

    seq: np.uint32

    s: array.array

    ts: Timestamp

    @classmethod
    def from_bytes(cls, data):
        """
        Parses the entire DfmuxPacket from raw bytes, which contain:
          - a header (<IHHBBBBI)
          - channel data (NUM_CHANNELS * 2 * sizeof(int))
          - a timestamp (8 * sizeof(uint32))
        """
        # Ensure the data length matches the expected packet length
        assert (
            len(data) == STREAMER_LEN
        ), f"Packet had unexpected size {len(data)} != {STREAMER_LEN}!"

        # Unpack the packet header
        header = struct.Struct("<IHHBBBBI")
        header_args = header.unpack(data[: header.size])

        # Extract the body (channel data)
        body = array.array("i")
        bodysize = NUM_CHANNELS * 2 * body.itemsize
        body.frombytes(data[header.size : header.size + bodysize])

        # Parse the timestamp from the remaining data
        ts = Timestamp.from_bytes(data[header.size + bodysize :])

        # Return a DfmuxPacket instance with the parsed data
        return DfmuxPacket(*header_args, s=body, ts=ts)


def get_local_ip(crs_hostname):
    """
    Determines the local IP address used to reach the CRS device.

    Args:
        crs_hostname (str): The hostname or IP address of the CRS device.

    Returns:
        str: The local IP address of the network interface used to reach the CRS device.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            # Connect to the CRS hostname on an arbitrary port to determine the local IP
            s.connect((crs_hostname, 1))
            local_ip = s.getsockname()[0]
        except Exception:
            raise Exception("Could not determine local IP address!")
    return local_ip


@macro(CRS, register=True)
async def py_get_samples(crs: CRS,
                         num_samples: int,
                         average: bool = False,
                         channel: int = None,
                         module: int = None,
                         *,
                         return_spectrum: bool = False,
                         scaling: str = 'psd',
                         nsegments: int = 1,
                         reference: str = 'relative',
                         spectrum_cutoff: float = 0.9):
    """
    Asynchronously retrieves samples from the CRS device.

    Parameters
    ----------
    crs : CRS
        The CRS device instance.
    num_samples : int
        Number of samples to collect.
    average : bool, optional
        If True, returns average and std dev only (time-domain).
        If False, returns the full timeseries (and possibly the spectrum).
    channel : int, optional
        Specific channel number to collect data from (1..1024). If None, collects all channels.
    module : int, optional
        The module number from which to retrieve samples. Must be in crs.modules.module.
    return_spectrum : bool, optional
        If True, also compute and return spectral data using welch + droop corrections.
    scaling : {'psd','ps'}, optional
        'psd' => interpret Welch results as V^2/Hz => final dBc/Hz or dBm/Hz
        'ps' => interpret Welch results as V^2 => final dBc or dBm
    nsegments : int, optional
        Number of Welch segments => nperseg = num_samples//nsegments. 
        Default 1 => entire data is one segment. For plots with lots of samples 10 is a good place to start.
    reference : {'relative','absolute'}, optional
        'relative' => dBc or dBc/Hz with DC bin as carrier in spectra, readout counts in TOD
        'absolute' => dBm or dBm/Hz with absolute scaling in spectra, volts in TOD
    spectrum_cutoff : float, optional
        Fraction of Nyquist to retain. Default=0.9 => up to 0.9*(fs/2).

    Returns
    -------
    TuberResult
        Contains the time-domain data (in 'i','q') plus optionally a 'spectrum' dict, with:
          - freq_iq, freq_dsb: frequency axes
          - {scaling}_i, {scaling}_q, {scaling}_dual_sideband
        in dBc or dBm units, depending on 'reference'.
        If channel=None, data arrays contain all channels.

    Notes
    -----
    - If average=True, we only return time-domain mean and std dev (no spectrum).
    - If reference='absolute', we convert time-domain data to volts (VOLTS_PER_ROC).
    - If reference='relative', the data are referenced to the total carrier power. 
      The DC bin is also overwritten to show the DC power rather than a density. 
      For the dual-sideband data this will be exactly 0dB.
    """
    # Ensure 'module' parameter is valid
    assert (
        module in crs.modules.module
    ), f"Unspecified or invalid module! Available modules: {crs.modules.module}"

    if channel is not None:
        assert 1 <= channel <= NUM_CHANNELS, f"Invalid channel {channel}!"

    # We need to ensure all data we grab from the network was emitted "now" or
    # later, from the perspective of control flow. Because of delays in the
    # signal path and network, we might actually get stale data.  We want to
    # avoid the following pathology:
    #
    # >>> # cryostat is in "before" condition"
    # >>> d.set_amplitude(...)
    # >>> # cryostat is in "after" condition
    # >>> x = await d.get_samples(...) # "x" had better reflect "after" condition!
    #
    # There are two ways data can be stale:
    #
    # 1. Buffers in the end-to-end network mean that any packets we grab "now"
    # might actually be pretty old. We can fix this by grabbing a "now"
    # timestamp from the board, and tossing any data packets that are older
    # than it.
    #
    # 2. Adjusting the "now" timestamp to add a little smidgen of signal-path
    # delay. This is because the signal path experiences group delay due to
    # decimation filters (PFB, CIC) and the timestamp doesn't. There are also
    # digital delays in the data converters (upsampling, downsampling) and
    # analog delays in the system.

    async with crs.tuber_context() as tc:
        ts = tc.get_timestamp()
        high_bank = tc.get_analog_bank()
        (ts, high_bank) = await tc()


    if high_bank:
        assert module > 4, \
                f"Can't retrieve samples from module {module} with set_analog_bank(True)"
        module -= 4
    else:
        assert module <= 4, \
                f"Can't retrieve samples from module {module} with set_analog_bank(False)"

    # Because IRIG-B takes a full second to decode a timestamp, it's polite to
    # stall here if we don't see a recent timestamp for up to a couple of
    # seconds. This allows a set_timestamp_port() call followed by a
    # py_get_samples() call to succeed, which may occur when a board is first
    # initialized after power-up. Note that old packets (with stale timestamps)
    # still need to traverse the network even if the board sees fresh ones,
    # this PC may receive old ones until they flush out.
    attempt = 5
    while attempt>0 and not ts.recent:
        warnings.warn(f"Got a stale timestamp. Trying again ({attempt} attempts remaining...)")
        await asyncio.sleep(1)
        ts = await crs.get_timestamp()
        attempt -= 1

    # Math on timestamps only works if they are valid
    assert ts.recent, "Timestamp wasn't recent - do you have a valid timestamp source?"

    # Ingest timestamp into Pythonic representation and nudge to compensate for
    # FIR delay.
    ts = Timestamp.from_TuberResult(ts)
    ts.ss += np.uint32(.02 * SS_PER_SECOND) # 20ms, per experiments at FIR6
    ts.renormalize()

    # Determine the local IP address to use for the multicast interface
    multicast_interface_ip = get_local_ip(crs.tuber_hostname)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:

        # To use asyncio, we need a non-blocking socket
        loop = asyncio.get_running_loop()
        sock.setblocking(False)

        # Configure the socket for multicast reception
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Bind the socket to all interfaces on the specified port
        sock.bind(("", STREAMER_PORT))

        # Set a large receive buffer size
        rcvbuf = 16777216
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)

        # Ensure SO_RCVBUF didn't just fail silently
        actual = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        if sys.platform=='linux':
            # Linux returns a doubled value
            actual /= 2
        if actual != rcvbuf:
            warnings.warn(
                f"Unable to set SO_RCVBUF to {rcvbuf} (got {actual}). Consider "
                "'sudo sysctl net.core.rmem_max=67108864' or similar. This setting "
                "can be made persistent across reboots by configuring "
                "/etc/sysctl.conf or /etc/sysctl.d."
            )

        # Set the interface to receive multicast packets
        sock.setsockopt(
            socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(multicast_interface_ip)
        )

        # Join the multicast group on the specified interface
        mreq = IPMreqSource(
                imr_multiaddr=InAddr(STREAMER_HOST),
                imr_interface=InAddr(multicast_interface_ip),
                imr_sourceaddr=InAddr(socket.gethostbyname(crs.tuber_hostname)))

        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_SOURCE_MEMBERSHIP, bytes(mreq))

        # Variable to track the previous sequence number for continuity checks
        prev_seq = None

        # Allow up to 10 packet-loss retries
        retries = 10

        packets = []
        # Start receiving packets
        while len(packets) < num_samples:
            data = await asyncio.wait_for(
                loop.sock_recv(sock, STREAMER_LEN), STREAMER_TIMEOUT
            )

            # Parse the received packet
            p = DfmuxPacket.from_bytes(data)

            if p.serial != int(crs.serial):
                warnings.warn(
                    f"Packet serial number {p.serial} didn't match CRS serial number {crs.serial}! Two boards on the network? IGMPv3 capable router will fix this warning."
                )

            # Filter packets by module
            if p.module != module - 1:
                continue  # Skip packets from other modules

            # Sanity checks on the packet
            if p.magic != STREAMER_MAGIC:
                raise RuntimeError(f"Invalid packet magic! {p.magic} != {STREAMER_MAGIC}")

            if p.version != STREAMER_VERSION:
                raise RuntimeError(f"Invalid packet version! {p.version} != {STREAMER_VERSION}")

            # Check if this packet is older than our "now" timestamp
            assert ts.source == p.ts.source, f"Timestamp source changed! {ts.source} vs {p.ts.source}"
            if ts > p.ts:
                continue

            # Update the sequence number continuity check
            if prev_seq is not None and prev_seq + 1 != p.seq:
                if retries > 0:
                    warnings.warn(
                        f"Discontinuous packet capture! Previous sequence {prev_seq} -> current sequence {p.seq}. "
                        f"Retrying capture ({retries} attempts remain.)"
                    )
                    retries -= 1
                    packets = []
                    prev_seq = None
                    continue
                else:
                    raise RuntimeError(
                        f"Discontinuous packet capture! Previous sequence {prev_seq} -> current sequence {p.seq}"
                    )

            # Update the previous sequence number
            prev_seq = p.seq

            # Append the valid packet to the list
            packets.append(p)

    # If average => just return time-domain averages
    if average:
        mean_i = np.zeros(NUM_CHANNELS)
        mean_q = np.zeros(NUM_CHANNELS)
        std_i = np.zeros(NUM_CHANNELS)
        std_q = np.zeros(NUM_CHANNELS)

        for c in range(NUM_CHANNELS):
            mean_i[c] = np.mean([p.s[2*c]/256 for p in packets])
            mean_q[c] = np.mean([p.s[2*c+1]/256 for p in packets])
            std_i[c] = np.std([p.s[2*c]/256 for p in packets])
            std_q[c] = np.std([p.s[2*c+1]/256 for p in packets])

        # If reference='absolute', convert to volts
        if reference == 'absolute':
            mean_i *= VOLTS_PER_ROC
            mean_q *= VOLTS_PER_ROC
            std_i *= VOLTS_PER_ROC
            std_q *= VOLTS_PER_ROC

        if channel is None:
            results = {
                "mean": TuberResult(dict(i=mean_i, q=mean_q)),
                "std": TuberResult(dict(i=std_i, q=std_q)),
            }
        else:
            # single channel => pick out channel-1
            results = {
                "mean": TuberResult(dict(i=mean_i[channel-1], q=mean_q[channel-1])),
                "std": TuberResult(dict(i=std_i[channel-1], q=std_q[channel-1])),
            }
        return TuberResult(results)

    # Otherwise build the normal time-domain results
    results = dict(ts=[TuberResult(asdict(p.ts)) for p in packets])

    if channel is None:
        # Return data for all channels
        results["i"] = []
        results["q"] = []
        for c in range(NUM_CHANNELS):
            i_channel = np.array([p.s[2*c]/256 for p in packets])
            q_channel = np.array([p.s[2*c+1]/256 for p in packets])
            if reference == 'absolute':
                i_channel=i_channel*VOLTS_PER_ROC
                q_channel=q_channel*VOLTS_PER_ROC
            results["i"].append(i_channel.tolist())
            results["q"].append(q_channel.tolist())
    else:
        i_channel = np.array([p.s[2*(channel-1)]/256 for p in packets])
        q_channel = np.array([p.s[2*(channel-1)+1]/256 for p in packets])
        if reference=='absolute':
            i_channel=i_channel*VOLTS_PER_ROC
            q_channel=q_channel*VOLTS_PER_ROC
        results["i"] = i_channel.tolist()
        results["q"] = q_channel.tolist()

    # Optionally compute the spectrum
    if return_spectrum:
        # Convert nsegments => nperseg for Welch
        nperseg = num_samples // nsegments

        # Retrieve decimation stage => helps define final sampling freq
        dec_stage = await crs.get_fir_stage()

        spec_data = {}
        if channel is None:
            # multi-channel => store a list of spectra for each channel
            spec_data["freq_iq"] = None
            spec_data["freq_dsb"] = None
            i_ch_spectra=[]
            q_ch_spectra=[]
            c_ch_spectra=[]
            for c in range(NUM_CHANNELS):
                i_data = results["i"][c]
                q_data = results["q"][c]
                d = spectrum_from_slow_tod(
                    i_data, q_data, dec_stage,
                    scaling=scaling,
                    nperseg=nperseg if nperseg else num_samples,
                    reference=reference,
                    spectrum_cutoff=spectrum_cutoff
                )
                # Store freq_iq, freq_dsb once
                if spec_data["freq_iq"] is None:
                    spec_data["freq_iq"] = d["freq_iq"].tolist()
                if spec_data["freq_dsb"] is None:
                    # shift freq_dsb for plotting
                    spec_data["freq_dsb"] = np.fft.fftshift(d["freq_dsb"]).tolist()

                i_ch_spectra.append(d["psd_i"].tolist())
                q_ch_spectra.append(d["psd_q"].tolist())
                c_ch_spectra.append(np.fft.fftshift(d["psd_dual_sideband"]).tolist())

            # store under {scaling}_i, {scaling}_q, {scaling}_dual_sideband
            spec_data[f"{scaling}_i"] = i_ch_spectra
            spec_data[f"{scaling}_q"] = q_ch_spectra
            spec_data[f"{scaling}_dual_sideband"] = c_ch_spectra
        else:
            i_data = results["i"]
            q_data = results["q"]
            d = spectrum_from_slow_tod(
                i_data, q_data, dec_stage,
                scaling=scaling,
                nperseg=nperseg if nperseg else num_samples,
                reference=reference,
                spectrum_cutoff=spectrum_cutoff
            )
            spec_data["freq_iq"] = d["freq_iq"].tolist()
            spec_data[f"{scaling}_i"] = d["psd_i"].tolist()
            spec_data[f"{scaling}_q"] = d["psd_q"].tolist()

            spec_data["freq_dsb"] = np.fft.fftshift(d["freq_dsb"]).tolist()
            spec_data[f"{scaling}_dual_sideband"] = np.fft.fftshift(d["psd_dual_sideband"]).tolist()

        # attach spectrum data to results
        results["spectrum"] = TuberResult(spec_data)

    return TuberResult(results)
