"""
CRS Packet Parser

Python wrapper around the _packets C++ extension module.
"""

import socket
from typing import Optional

from ._packets import (
    PFBPacket,
    PFBPacketReceiver,
    PacketQueue,
    PacketQueueStats,
    PacketReceiverStats,
    ReadoutPacket,
    ReadoutPacketReceiver,
    Timestamp,
    TimestampSource,
    LONG_PACKET_CHANNELS,
    LONG_PACKET_SIZE,
    MULTICAST_GROUP,
    PFB_PACKET_MAGIC,
    PFB_STREAMER_PORT,
    PFBPACKET_NSAMP_MAX,
    READOUT_PACKET_MAGIC,
    SHORT_PACKET_CHANNELS,
    SHORT_PACKET_SIZE,
    STREAMER_PORT,
    ip_mreq_source,
)

__all__ = [
    'PFBPacket',
    'PFBPacketReceiver',
    'PacketQueue',
    'PacketQueueStats',
    'PacketReceiverStats',
    'ReadoutPacket',
    'ReadoutPacketReceiver',
    'Timestamp',
    'TimestampSource',
    'LONG_PACKET_CHANNELS',
    'LONG_PACKET_SIZE',
    'MULTICAST_GROUP',
    'PFB_PACKET_MAGIC',
    'PFB_STREAMER_PORT',
    'PFBPACKET_NSAMP_MAX',
    'READOUT_PACKET_MAGIC',
    'SHORT_PACKET_CHANNELS',
    'SHORT_PACKET_SIZE',
    'STREAMER_PORT',
    'create_multicast_socket',
]


def create_multicast_socket(
    hostname: Optional[str] = None,
    interface: Optional[str] = None,
    port: int = STREAMER_PORT,
    buffer_size: Optional[int] = None
) -> socket.socket:
    """
    Create and configure a multicast socket for receiving CRS packets.

    This handles all the platform-specific setup including:
    - Creating UDP socket
    - Setting socket options (reuse address, buffer size)
    - Binding to port
    - Joining source-specific multicast group

    Args:
        hostname: CRS board hostname or IP address (used as multicast source).
                 If provided, auto-discovers the local interface to use.
        interface: Local interface IP address to use for multicast.
                  If provided, receives from all sources on that interface.
                  Either hostname or interface must be specified.
        port: Port to bind to (typically STREAMER_PORT or PFB_STREAMER_PORT)
        buffer_size: Optional receive buffer size in bytes.
                    If None, uses 16MB on Linux, 7MB on macOS

    Returns:
        Configured socket ready for receiving packets

    Examples:
        >>> # Auto-discover interface from hostname
        >>> sock = create_multicast_socket(hostname='rfmux0033.local')

        >>> # Explicit interface
        >>> sock = create_multicast_socket(interface='eth0')
    """
    if hostname is None and interface is None:
        raise ValueError("Must specify either hostname or interface")

    if hostname is not None and interface is not None:
        raise ValueError("Cannot specify both hostname and interface")

    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    # Allow address reuse
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Set receive buffer size
    if buffer_size is None:
        # Default: 16MB on Linux, less on macOS which has lower limits
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16777216)
        except OSError:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 7340032)
    else:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)

    # Bind to port
    sock.bind(('', port))

    if hostname is not None:
        # Resolve source IP from hostname
        source_ip = socket.gethostbyname(hostname)

        # Determine local interface IP. Connects a temporary socket to the source
        # to find which interface to use
        temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            temp_sock.connect((source_ip, port))
            local_ip = temp_sock.getsockname()[0]
        finally:
            temp_sock.close()
    else:
        # Explicit interface specified
        local_ip = interface
        # For interface-only mode, we don't filter by source
        # Join the multicast group to receive from any source
        mreq = socket.inet_aton(MULTICAST_GROUP) + socket.inet_aton(local_ip)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        return sock

    # Join source-specific multicast group (when hostname is specified)
    # This uses the platform-specific ip_mreq_source implementation
    mreq = ip_mreq_source(multiaddr=MULTICAST_GROUP, sourceaddr=source_ip, interface=local_ip)
    sock.setsockopt(
        socket.IPPROTO_IP,
        ip_mreq_source.IP_ADD_SOURCE_MEMBERSHIP,
        mreq.to_bytes()
    )

    return sock
