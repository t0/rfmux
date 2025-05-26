import array
import ctypes
import dataclasses
import enum
import numpy as np
import socket
import struct
import sys
import warnings

# Constants
STREAMER_PORT = 9876
STREAMER_HOST = "239.192.0.2"
LONG_PACKET_SIZE = 8240  # expected packet length in bytes
SHORT_PACKET_SIZE = 1072  # expected packet length in bytes
STREAMER_MAGIC = 0x5344494B

LONG_PACKET_VERSION = 5
SHORT_PACKET_VERSION = 6
LONG_PACKET_CHANNELS = 1024
SHORT_PACKET_CHANNELS = 128

SS_PER_SECOND = 156250000

# This seems like a long timeout - in practice, we can see long delays when
# packets are flowing normally
STREAMER_TIMEOUT = 60

# Source-specific multicasting support was added in 3.12.0
if not hasattr(socket, "IP_ADD_SOURCE_MEMBERSHIP"):
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
        self.s_addr = int.from_bytes(socket.inet_aton(ip), byteorder="little")


class IPMreqSource(ctypes.Structure):
    """
    The order of fields in this structure is implementation-specific.
    Notably, fields have different order in Windows and Linux.
    """

    match sys.platform:
        case "win32":
            _fields_ = [
                ("imr_multiaddr", InAddr),
                ("imr_sourceaddr", InAddr),
                ("imr_interface", InAddr),
            ]
        case "linux":
            _fields_ = [
                ("imr_multiaddr", InAddr),
                ("imr_interface", InAddr),
                ("imr_sourceaddr", InAddr),
            ]
        case _:
            raise NotImplementedError(
                f"Source-specific multicast support for {sys.platform} is incomplete."
            )


class TimestampPort(str, enum.Enum):
    BACKPLANE = "BACKPLANE"
    TEST = "TEST"
    SMA = "SMA"
    GND = "GND"


@dataclasses.dataclass(order=True)
class Timestamp:
    y: np.int32  # 0-99
    d: np.int32  # 1-366
    h: np.int32  # 0-23
    m: np.int32  # 0-59
    s: np.int32  # 0-59
    ss: np.int32  # 0-SS_PER_SECOND-1
    c: np.int32
    sbs: np.int32

    source: TimestampPort
    recent: bool

    def renormalize(self):
        """
        Normalizes the timestamp fields, carrying over seconds->minutes->hours->days
        as needed. Ignores leap years for day-of-year handling.
        """

        # if the timestamp wasn't recent, fields aren't trustworthy and we shouldn't bother
        if not self.recent:
            return

        old = dataclasses.astuple(self)

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
        data = {f: getattr(ts, f) for f in dir(ts) if not f.startswith("_")}
        return Timestamp(**data)


@dataclasses.dataclass
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
          - channel data (num_channels * 2 * sizeof(int))
          - a timestamp (8 * sizeof(uint32))
        """
        # Ensure the data length matches the expected packet length
        assert (
            len(data) in {LONG_PACKET_SIZE, SHORT_PACKET_SIZE}
        ), f"Packet had unexpected size {len(data)}!"

        # Unpack the packet header - we need this to determine the channel count
        header = struct.Struct("<IHHBBBBI")
        header_args = (magic, version, serial, *_) = header.unpack(data[: header.size])

        if magic != STREAMER_MAGIC:
            raise RuntimeError(f"Invalid packet magic 0x{magic:08}")

        if version == LONG_PACKET_VERSION:
            num_channels = LONG_PACKET_CHANNELS
        elif version == SHORT_PACKET_VERSION:
            num_channels = SHORT_PACKET_CHANNELS
        else:
            raise RuntimeError(f"Invalid packet version 0x{version:04x}")

        # Extract the body (channel data)
        body = array.array("i")
        bodysize = num_channels * 2 * body.itemsize
        body.frombytes(data[header.size : header.size + bodysize])

        # Parse the timestamp from the remaining data
        ts = Timestamp.from_bytes(data[header.size + bodysize :])

        # Return a DfmuxPacket instance with the parsed data
        return DfmuxPacket(*header_args, s=body, ts=ts)


    def get_num_channels(self):
        if self.version == LONG_PACKET_VERSION:
            return LONG_PACKET_CHANNELS
        elif self.version == SHORT_PACKET_VERSION:
            return SHORT_PACKET_CHANNELS
        raise RuntimeError(f"Invalid packet version 0x{self.version:04x}")


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


def get_multicast_socket(crs_hostname):
    multicast_interface_ip = get_local_ip(crs_hostname)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    # Configure the socket for multicast reception
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Bind the socket to all interfaces on the specified port
    sock.bind(("", STREAMER_PORT))

    # Set a large receive buffer size
    rcvbuf = 16777216
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)

    # Ensure SO_RCVBUF didn't just fail silently
    actual = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    if sys.platform == "linux":
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
        socket.IPPROTO_IP,
        socket.IP_MULTICAST_IF,
        socket.inet_aton(multicast_interface_ip),
    )

    # Join the multicast group on the specified interface
    mreq = IPMreqSource(
        imr_multiaddr=InAddr(STREAMER_HOST),
        imr_interface=InAddr(multicast_interface_ip),
        imr_sourceaddr=InAddr(socket.gethostbyname(crs_hostname)),
    )

    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_SOURCE_MEMBERSHIP, bytes(mreq))

    return sock
