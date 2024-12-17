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

from dataclasses import dataclass, asdict, astuple

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


@macro(CRS, register=True)
async def py_get_samples(crs : CRS,
                         num_samples, average : bool = False,
                         channel : int = None,
                         module : int = None):
    """
    Asynchronously retrieves samples from the CRS device.

    Args:
        crs: The CRS device instance.
        num_samples: Number of samples to collect.
        channel: Specific channel number to collect data from (optional).
        module: Specific module number to collect data from.

    Returns:
        TuberResult: The collected samples and timestamps.
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

    # Adjust the timestamp by a small amount of wall time to avoid stale-data
    # problems. The time "error" is due to delays the signal path sees (group
    # delay in the CICs) that the timestamp path does not see.

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
                if retries:
                    warnings.warn(
                        f"Discontinuous packet capture! Previous sequence {prev_seq} -> current sequence {p.seq}. "
                        f"Retrying ({retries} attempts remain.)"
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

        if channel is None:
            return {
                "mean": dict(i=mean_i, q=mean_q),
                "std": dict(i=std_i, q=std_q),
            }
        return {
            "mean": dict(i=mean_i[channel-1], q=mean_q[channel-1]),
            "std": dict(i=std_i[channel-1], q=std_q[channel-1]),
        }

    # Build the results dictionary with timestamps
    results = dict(ts=[TuberResult(asdict(p.ts)) for p in packets])

    if channel is None:
        # Return data for all channels
        results["i"] = []
        results["q"] = []
        for c in range(NUM_CHANNELS):
            # Extract 'i' and 'q' data for each channel across all packets
            results["i"].append([p.s[2 * c] / 256 for p in packets])
            results["q"].append([p.s[2 * c + 1] / 256 for p in packets])
    else:
        # Return data for the specified channel
        results["i"] = [p.s[2 * (channel - 1)] / 256 for p in packets]
        results["q"] = [p.s[2 * (channel - 1) + 1] / 256 for p in packets]

    return TuberResult(results)
