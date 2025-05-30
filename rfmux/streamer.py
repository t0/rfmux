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
STREAMER_LEN = 8240  # expected packet length in bytes
STREAMER_MAGIC = 0x5344494B
STREAMER_VERSION = 5
NUM_CHANNELS = 1024
SS_PER_SECOND = 125000000

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

    def __init__(self, ip: str):
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
        crs_hostname (str): The hostname or IP address of the CRS device, optionally with port.

    Returns:
        str: The local IP address of the network interface used to reach the CRS device.
    """
    # Parse hostname to extract just the host part if port is included
    if ':' in crs_hostname:
        hostname = crs_hostname.split(':')[0]
    else:
        hostname = crs_hostname
    
    # Special handling for localhost
    if hostname in ["127.0.0.1", "localhost", "::1"]:
        return "127.0.0.1"
    
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            # Connect to the hostname on an arbitrary port to determine the local IP
            s.connect((hostname, 1))
            local_ip = s.getsockname()[0]
        except Exception:
            raise Exception("Could not determine local IP address!")
    return local_ip


def get_multicast_socket(crs_hostname):
    # Extract just the hostname part for source address resolution
    if ':' in crs_hostname:
        hostname_only = crs_hostname.split(':')[0]
    else:
        hostname_only = crs_hostname
    
    # Check if this is a localhost connection (MockCRS uses unicast)
    if hostname_only in ["127.0.0.1", "localhost", "::1"]:
        # Create a unicast socket for MockCRS
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind to the streamer port on localhost
        sock.bind(("127.0.0.1", STREAMER_PORT))
        
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
        
        return sock
    
    # Original multicast socket code for real hardware
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
        imr_sourceaddr=InAddr(socket.gethostbyname(hostname_only)),
    )

    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_SOURCE_MEMBERSHIP, bytes(mreq))

    return sock
