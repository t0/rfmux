"""
py_get_samples: an experimental pure-Python, client-side implementation of the
get_samples() call with dynamic determination of the multicast interface IP address.
"""

import array
import contextlib
import enum
import numpy as np
import socket
import struct
import warnings

from ...core.hardware_map import macro
from ...core.schema import CRS
from ...tuber.codecs import TuberResult
from ...core.utils.transferfunctions import VOLTS_PER_ROC

from dataclasses import dataclass, asdict, astuple

# Added import for PSD computation
from scipy.signal import welch

STREAMER_PORT = 9876
STREAMER_HOST = "239.192.0.2"
STREAMER_LEN = 8240
STREAMER_MAGIC = 0x5344494B
STREAMER_VERSION = 5
NUM_CHANNELS = 1024
SS_PER_SECOND = 125000000

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
        # Unpack the timestamp data
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


#
# New: A small helper to compute the single-stage CIC correction
# (used internally by _compute_spectrum).
#
def _cic_correction(frequencies, f_in, R=64, N=6):
    """
    Internal helper for single-stage CIC correction.

    frequencies : array of frequency bins (Hz)
    f_in : input sampling rate prior to decimation
    R, N : decimation rate, number of stages
    """
    freq_ratio = frequencies / f_in
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = np.sin(np.pi * freq_ratio)
        denominator = np.sin(np.pi * freq_ratio / R)
        correction = (numerator / denominator) ** N
        # Replace NaNs at DC with the ideal DC gain = R^N
        correction[np.isnan(correction)] = R ** N
    return correction / (R ** N)


#
# New: A local function to compute spectrum in dBc or dBm.
# This is only called if return_spectrum=True.
#
def _compute_spectrum(i_data, q_data, fs, dec_stage,
                     scaling='psd', nperseg=None,
                     reference='relative',
                     spectrum_cutoff=0.9):
    """
    Internal function to compute both the I/Q single-sideband PSD and
    the dual-sideband complex PSD in dBc or dBm.

    Parameters
    ----------
    i_data : list or ndarray
        Time-domain I (real) samples.
    q_data : list or ndarray
        Time-domain Q (imag) samples.
    fs : float
        Sampling frequency in Hz.
    dec_stage : int
        Decimation stage to determine second CIC's decimation factor.
    scaling : str
        'psd' for power spectral density (V²/Hz),
        'ps' for power spectrum (V²).
    nperseg : int or None
        Number of samples per segment for Welch. Defaults to len(i_data).
    reference : str
        'relative' for dBc, 'absolute' for dBm.
    spectrum_cutoff : float
        Fraction of Nyquist frequency to retain in the spectrum (default: 0.9).

    Returns
    -------
    dict
        Dictionary with:
          'freq_iq' : frequency bins for the single-sideband (I/Q) PSD
          'i_psd'   : PSD for I
          'q_psd'   : PSD for Q
          'freq_c'  : frequency bins for the complex (dual-sideband) PSD
          'complex_psd' : complex PSD
    """
    # Set default nperseg to num_samples if not provided
    if nperseg is None:
        nperseg = len(i_data)

    # Convert 'psd'/'ps' to scipy's 'density'/'spectrum'
    scipy_scaling = 'density' if scaling.lower() == 'psd' else 'spectrum'

    arr_i = np.asarray(i_data)
    arr_q = np.asarray(q_data)
    arr_complex = arr_i + 1j * arr_q

    # Compute the carrier power if in 'relative' mode
    if reference == 'relative':
        carrier_power = np.abs(np.mean(arr_complex)) ** 2
    else:
        carrier_power = None

    # Define CIC decimation parameters
    R1 = 64
    R2 = 2 ** dec_stage
    f_in1 = 625e6 / 256.0  # Original f_in before first CIC
    f_in2 = f_in1 / R1     # f_in before second CIC

    #
    # Single-sideband I/Q
    #
    freq_i, psd_i = welch(
        arr_i, fs=fs, nperseg=nperseg, scaling=scipy_scaling, return_onesided=True
    )
    freq_q, psd_q = welch(
        arr_q, fs=fs, nperseg=nperseg, scaling=scipy_scaling, return_onesided=True
    )

    # freq_i == freq_q, so we can unify them
    freq_iq = freq_i

    # Correct frequency scaling for the two CIC stages
    cic1_corr = _cic_correction(freq_iq * R1 * R2, f_in=f_in1, R=R1, N=6)
    cic2_corr = _cic_correction(freq_iq * R2, f_in=f_in2, R=R2, N=6)
    correction_iq = cic1_corr * cic2_corr

    # Apply correction
    psd_i_corrected = psd_i / correction_iq
    psd_q_corrected = psd_q / correction_iq

    # Apply spectrum cutoff
    nyquist = fs / 2
    cutoff_freq = spectrum_cutoff * nyquist
    cutoff_idx_iq = freq_iq <= cutoff_freq

    freq_iq = freq_iq[cutoff_idx_iq]
    psd_i_corrected = psd_i_corrected[cutoff_idx_iq]
    psd_q_corrected = psd_q_corrected[cutoff_idx_iq]

    # Convert to dBc or dBm
    if reference == 'relative':
        i_psd_db = 10.0 * np.log10(psd_i_corrected / (carrier_power + 1e-30))
        q_psd_db = 10.0 * np.log10(psd_q_corrected / (carrier_power + 1e-30))
    else:
        # absolute => convert V^2 to W => W to dBm
        p_i = psd_i_corrected / 50.0
        p_q = psd_q_corrected / 50.0
        i_psd_db = 10.0 * np.log10(p_i / 1e-3 + 1e-30)
        q_psd_db = 10.0 * np.log10(p_q / 1e-3 + 1e-30)

    #
    # Dual-sideband complex
    #
    freq_c, psd_c = welch(
        arr_complex, fs=fs, nperseg=nperseg, scaling=scipy_scaling, return_onesided=False
    )

    # We'll apply corrections and cutoff using absolute value of freq
    freq_abs = np.abs(freq_c)
    cic1_corr_c = _cic_correction(freq_abs * R1 * R2, f_in=f_in1, R=R1, N=6)
    cic2_corr_c = _cic_correction(freq_abs * R2, f_in=f_in2, R=R2, N=6)
    correction_c = cic1_corr_c * cic2_corr_c

    psd_c_corrected = psd_c / correction_c

    cutoff_idx_c = freq_abs <= cutoff_freq
    freq_c = freq_c[cutoff_idx_c]
    psd_c_corrected = psd_c_corrected[cutoff_idx_c]

    if reference == 'relative':
        complex_psd_db = 10.0 * np.log10(psd_c_corrected / (carrier_power + 1e-30))
    else:
        p_c = psd_c_corrected / 50.0
        complex_psd_db = 10.0 * np.log10(p_c / 1e-3 + 1e-30)

    return {
        "freq_iq": freq_iq,
        "i_psd": i_psd_db,
        "q_psd": q_psd_db,
        "freq_c": freq_c,
        "complex_psd": complex_psd_db,
    }


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

    Args:
        crs (CRS): The CRS device instance.
        num_samples (int): Number of samples to collect.
        average (bool, optional): If True, returns average and std dev only (time-domain).
        channel (int, optional): Specific channel number to collect data from (optional).
        module (int, optional): The module number from which to retrieve samples.
        return_spectrum (bool, optional): If True, also compute and return the PSD or PS.
        scaling (str, optional): Specifies density vs spectrum. 'psd' => V^2/Hz; 'ps' => V^2.
        nsegments (int, optional) : Number of segments for averaging.
            By default there is no averaging. For plots with lots of samples 10 is a good place to start.
        reference (str, optional): 'relative' to report spectra in dBc and time-domain in counts,
            'absolute' to report spectra in dBm and time-domain in volts.
        spectrum_cutoff (float, optional): Fraction of Nyquist frequency to retain in the spectrum (default: 0.9).
            The CIC corrections go to infinity at Nyqyuist. This cutoff avoids amplifying noise at the extrema.
            If you are measuring large signals above the noise, then beyond-cutoff values still make sense, and
            this can be adjusted to recover those signals.

    Returns:
        TuberResult: The collected time-domain samples and timestamps, plus optional spectral data.
                     - In 'relative' mode:
                         - Time-domain 'i' and 'q' in counts.
                         - Spectrum in dBc.
                     - In 'absolute' mode:
                         - Time-domain 'i' and 'q' in volts.
                         - Spectrum in dBm.
                     - If channel=None, data arrays contain all channels.
    """
    # Ensure 'module' parameter is specified and valid
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

    ts = Timestamp.from_TuberResult(ts.result())
    high_bank = high_bank.result()

    if high_bank:
        assert module > 4, \
                f"Can't retrieve samples from module {module} with set_analog_bank(True)"
        module -= 4
    else:
        assert module <= 4, \
                f"Can't retrieve samples from module {module} with set_analog_bank(False)"

    # Math on timestamps only works if they are valid
    assert ts.recent, "Timestamp wasn't recent - do you have a valid timestamp source?"

    ts.ss += np.uint32(.02 * SS_PER_SECOND) # 20ms, per experiments at FIR6
    ts.renormalize()

    # Determine the local IP address to use for the multicast interface
    multicast_interface_ip = get_local_ip(crs.tuber_hostname)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:

        # Configure the socket for multicast reception
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Bind the socket to all interfaces on the specified port
        sock.bind(("", STREAMER_PORT))

        # Set a large receive buffer size
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16777216)

        # Set the interface to receive multicast packets
        sock.setsockopt(
            socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(multicast_interface_ip)
        )

        # Join the multicast group on the specified interface
        mc_ip = socket.inet_aton(STREAMER_HOST)
        mreq = struct.pack("4s4s", mc_ip, socket.inet_aton(multicast_interface_ip))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        # Set a timeout on the socket to prevent indefinite blocking
        sock.settimeout(5)  # Timeout after 5 seconds

        # Variable to track the previous sequence number for continuity checks
        prev_seq = None

        # Allow up to 10 packet-loss retries
        retries = 10

        packets = []
        # Start receiving packets
        while len(packets) < num_samples:
            try:
                data = sock.recv(STREAMER_LEN)
            except socket.timeout:
                if retries > 0:
                    # Retry receiving packets after timeout
                    retries -= 1
                    continue
                else:
                    raise RuntimeError("Socket timed out waiting for data. No more retries left.")
            except Exception as e:
                raise

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

            # Drop packets until the board's time equals the network's time
            assert ts.source == p.ts.source, f"Timestamp source changed! Reference {ts.source}, packet {p.ts.source}"
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

    if average:
        mean_i = np.zeros(NUM_CHANNELS)
        mean_q = np.zeros(NUM_CHANNELS)
        std_i = np.zeros(NUM_CHANNELS)
        std_q = np.zeros(NUM_CHANNELS)

        for c in range(NUM_CHANNELS):
            mean_i[c] = np.mean([p.s[2 * c] / 256 for p in packets])
            mean_q[c] = np.mean([p.s[2 * c + 1] / 256 for p in packets])
            std_i[c] = np.std([p.s[2 * c] / 256 for p in packets])
            std_q[c] = np.std([p.s[2 * c + 1] / 256 for p in packets])

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
            results = {
                "mean": TuberResult(dict(i=mean_i[channel-1], q=mean_q[channel-1])),
                "std": TuberResult(dict(i=std_i[channel-1], q=std_q[channel-1])),
            }
            
        return TuberResult(results)

    # Build the results dictionary with timestamps
    results = dict(ts=[TuberResult(asdict(p.ts)) for p in packets])

    if channel is None:
        # Return data for all channels
        results["i"] = []
        results["q"] = []
        for c in range(NUM_CHANNELS):
            # Extract 'i' and 'q' data for each channel across all packets
            i_channel = np.array([p.s[2 * c] / 256 for p in packets])
            q_channel = np.array([p.s[2 * c + 1] / 256 for p in packets])
            if reference == 'absolute':
                i_channel *= VOLTS_PER_ROC
                q_channel *= VOLTS_PER_ROC
            results["i"].append(i_channel.tolist())
            results["q"].append(q_channel.tolist())
    else:
        i_channel = np.array([p.s[2 * (channel - 1)] / 256 for p in packets])
        q_channel = np.array([p.s[2 * (channel - 1) + 1] / 256 for p in packets])
        if reference == 'absolute':
            i_channel *= VOLTS_PER_ROC
            q_channel *= VOLTS_PER_ROC
        results["i"] = i_channel.tolist()
        results["q"] = q_channel.tolist()

    #
    # New: Optionally compute the spectrum and store in TuberResult
    #
    if return_spectrum:

        # Convert nsegments to nperseg for welch
        nperseg = num_samples // nsegments

        # Retrieve decimation stage to determine sampling frequency
        dec_stage = await crs.get_fir_stage()
        fs = 625e6 / (256 * 64 * 2**dec_stage)

        spec_data = {}
        if channel is None:
            spec_data["freq_iq"] = None  # set once
            spec_data["freq_c"] = None   # set once
            i_ch_spectra = []
            q_ch_spectra = []
            c_ch_spectra = []
            for c in range(NUM_CHANNELS):
                i_data = results["i"][c]
                q_data = results["q"][c]
                d = _compute_spectrum(
                    i_data, q_data, fs, dec_stage,
                    scaling=scaling,
                    nperseg=nperseg if nperseg is not None else num_samples,
                    reference=reference,
                    spectrum_cutoff=spectrum_cutoff
                )
                if spec_data["freq_iq"] is None:
                    spec_data["freq_iq"] = d["freq_iq"].tolist()
                if spec_data["freq_c"] is None:
                    spec_data["freq_c"] = np.fft.fftshift(d["freq_c"]).tolist()
                i_ch_spectra.append(d["i_psd"].tolist())
                q_ch_spectra.append(d["q_psd"].tolist())
                c_ch_spectra.append(np.fft.fftshift(d["complex_psd"]).tolist())

            spec_data["i_psd"] = i_ch_spectra
            spec_data["q_psd"] = q_ch_spectra
            spec_data["complex_psd"] = c_ch_spectra
        else:
            i_data = results["i"]
            q_data = results["q"]
            d = _compute_spectrum(
                i_data, q_data, fs, dec_stage,
                scaling=scaling,
                nperseg=nperseg if nperseg is not None else num_samples,
                reference=reference,
                spectrum_cutoff=spectrum_cutoff
            )
            spec_data["freq_iq"] = d["freq_iq"].tolist()
            spec_data["i_psd"] = d["i_psd"].tolist()
            spec_data["q_psd"] = d["q_psd"].tolist()

            # FFT shift these to make plotting easier from left to right
            spec_data["freq_c"] = np.fft.fftshift(d["freq_c"]).tolist()
            spec_data["complex_psd"] = np.fft.fftshift(d["complex_psd"]).tolist()

        results["spectrum"] = TuberResult(spec_data)

    return TuberResult(results)
