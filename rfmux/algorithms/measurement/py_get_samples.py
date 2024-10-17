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

from dataclasses import dataclass, asdict

STREAMER_PORT = 9876
STREAMER_HOST = "239.192.0.2"
STREAMER_LEN = 8240
STREAMER_MAGIC = 0x5344494B
STREAMER_VERSION = 5
NUM_CHANNELS = 1024


class TimestampPort(str, enum.Enum):
    BACKPLANE = "BACKPLANE"
    TEST = "TEST"
    SMA = "SMA"
    GND = "GND"


@dataclass
class Timestamp:
    y: np.int32
    d: np.int32
    h: np.int32
    m: np.int32
    s: np.int32
    ss: np.int32
    c: np.int32
    sbs: np.int32

    source: TimestampPort
    recent: bool

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
async def py_get_samples(crs, num_samples, channel=None, module=None):
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

    # Determine the local IP address to use for the multicast interface
    multicast_interface_ip = get_local_ip(crs.tuber_hostname)

    with contextlib.closing(
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    ) as sock:
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
                raise RuntimeError(
                    f"Packet serial number {p.serial} didn't match CRS serial number {crs.serial}!"
                )

            # Filter packets by module
            if p.module != module - 1:
                continue  # Skip packets from other modules

            # Sanity checks on the packet
            if p.magic != STREAMER_MAGIC:
                raise RuntimeError(f"Invalid packet magic! {p.magic} != {STREAMER_MAGIC}")

            if p.version != STREAMER_VERSION:
                raise RuntimeError(f"Invalid packet version! {p.version} != {STREAMER_VERSION}")

            # Check for packet sequence continuity
            with np.errstate(over="ignore"):
                if packets and packets[-1].seq + 1 != p.seq:
                    if retries:
                        warnings.warn(
                            f"Discontinuous packet capture! Index {len(packets)}, sequence {packets[-1].seq} -> {p.seq}. "
                            f"Retrying ({retries} attempts remain.)"
                        )
                        retries -= 1
                        packets = []
                        continue
                    else:
                        raise RuntimeError(
                            f"Discontinuous packet capture! Index {len(packets)}, sequence {packets[-1].seq} -> {p.seq}"
                        )

            packets.append(p)

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
