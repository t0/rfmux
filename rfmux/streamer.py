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
        case "win32" | "darwin":
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


    def to_bytes(self) -> bytes:
        """
        Serialize a DfmuxPacket into bytes.
        Layout:
          - 16-byte header (<IHHBBBBI)
          - channel data (NUM_CHANNELS*2 int32)
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
            self.version if self.version is not None else LONG_PACKET_VERSION,
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
            temp_arr = array.array("i", [0] * (self.get_num_channels() * 2))
            scale = 32767 # Example scale, might need adjustment based on expected data range
            for i in range(self.get_num_channels() * 2):
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

        if len(body_bytes) != self.get_num_channels() * 2 * 4: # 4 bytes per int32
            raise ValueError(f"Channel data must be {self.get_num_channels()*2*4} bytes. Got {len(body_bytes)}")

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
        if len(packet) not in {SHORT_PACKET_SIZE, LONG_PACKET_SIZE}:
            raise ValueError(f"Packet length mismatch: {len(packet)}")
        return packet


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


def _set_socket_buffer_size(sock, desired_size=16777216):
    """
    Attempt to set socket receive buffer size, trying progressively smaller values if needed.
    This handles platform limitations, particularly on macOS which has lower limits than Linux.
    
    Args:
        sock: The socket to configure
        desired_size: The desired buffer size in bytes (default 16MB)
    
    Returns:
        The actual buffer size that was set
    """
    # Buffer sizes to try, in descending order
    # macOS typically supports up to ~7.4MB, while Linux can go much higher
    buffer_sizes = [
        desired_size,  # 16MB
        8388608,       # 8MB
        7430000,       # ~7.4MB (near macOS limit)
        4194304,       # 4MB
    ]
    
    actual_size = None
    set_size = None
    
    for size in buffer_sizes:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, size)
            set_size = size
            
            # Verify what was actually set
            actual_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            if sys.platform == "linux":
                # Linux returns a doubled value
                actual_size //= 2
            
            # If we successfully set a buffer size, we're done
            break
            
        except OSError as e:
            # This size was rejected, try the next smaller one
            continue
    
    if set_size is None:
        # This should rarely happen, but handle it gracefully
        warnings.warn(
            "Unable to set socket receive buffer to any reasonable size. "
            "Network performance may be severely impacted."
        )
        return 0
    
    # Warn if we couldn't set the desired size or if actual differs from requested
    if set_size < desired_size:
        if sys.platform == "darwin":
            warnings.warn(
                f"macOS UDP buffer size limit prevented setting SO_RCVBUF to {desired_size} bytes. "
                f"Set to {set_size} bytes instead (actual: {actual_size} bytes). "
                f"This is a known macOS limitation. To increase the limit, you can try: "
                f"'sudo sysctl -w kern.ipc.maxsockbuf=16777216' (temporary) or add "
                f"'kern.ipc.maxsockbuf=16777216' to /etc/sysctl.conf (permanent)."
            )
        else:
            warnings.warn(
                f"Unable to set SO_RCVBUF to {desired_size} bytes. "
                f"Set to {set_size} bytes instead (actual: {actual_size} bytes). "
                f"Consider increasing system limits."
            )
    elif actual_size != set_size:
        # Size was accepted but kernel adjusted it
        warnings.warn(
            f"SO_RCVBUF was adjusted by the kernel from {set_size} to {actual_size} bytes. "
            f"Consider 'sudo sysctl net.core.rmem_max=67108864' or similar. This setting "
            f"can be made persistent across reboots by configuring /etc/sysctl.conf or /etc/sysctl.d."
        )
    
    return actual_size


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
        
        # Set receive buffer size with platform-aware handling
        _set_socket_buffer_size(sock)
        
        return sock
    
    # Original multicast socket code for real hardware
    multicast_interface_ip = get_local_ip(crs_hostname)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    # Configure the socket for multicast reception
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Bind the socket to all interfaces on the specified port
    sock.bind(("", STREAMER_PORT))

    # Set receive buffer size with platform-aware handling
    _set_socket_buffer_size(sock)

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
