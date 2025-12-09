"""
CRS Streaming Protocol

Unified API for CRS packet streaming, including:
- Packet structures (ReadoutPacket, PFBPacket, Timestamp)
- High-performance C++ packet receiver
- Socket utilities for multicast configuration
- Protocol constants
"""

# Import C++ packet receiver and structures
from ._receiver import (
	# Packet classes
	ReadoutPacket,
	PFBPacket,
	Timestamp,
	TimestampSource,
	Packet,

	# Receivers
	ReadoutPacketReceiver,
	PFBPacketReceiver,
	PacketReceiver,

	# Queues and stats
	PacketQueue,
	PacketQueueStats,
	PacketReceiverStats,

	# Socket utilities (from C++)
	ip_mreq_source,

	# Constants
	MULTICAST_GROUP,
	READOUT_PACKET_MAGIC,
	PFB_PACKET_MAGIC,
	STREAMER_PORT,
	PFB_STREAMER_PORT,
	LONG_PACKET_SIZE,
	SHORT_PACKET_SIZE,
	LONG_PACKET_CHANNELS,
	SHORT_PACKET_CHANNELS,
	LONG_PACKET_VERSION,
	SHORT_PACKET_VERSION,
	PFBPACKET_NSAMP_MAX,
	SS_PER_SECOND,
)

# Import socket utilities
from .socket import (
	get_multicast_socket,
	get_local_ip,
)

# Backwards compatibility aliases
DfmuxPacket = ReadoutPacket
STREAMER_MAGIC = READOUT_PACKET_MAGIC
STREAMER_HOST = MULTICAST_GROUP
STREAMER_TIMEOUT = 60  # seconds

__all__ = [
	# Packet classes
	'ReadoutPacket',
	'PFBPacket',
	'Timestamp',
	'TimestampSource',
	'Packet',

	# Receivers
	'ReadoutPacketReceiver',
	'PFBPacketReceiver',
	'PacketReceiver',

	# Queues
	'PacketQueue',
	'PacketQueueStats',
	'PacketReceiverStats',

	# Socket utilities
	'get_multicast_socket',
	'get_local_ip',
	'ip_mreq_source',

	# Constants
	'MULTICAST_GROUP',
	'READOUT_PACKET_MAGIC',
	'PFB_PACKET_MAGIC',
	'STREAMER_PORT',
	'PFB_STREAMER_PORT',
	'LONG_PACKET_SIZE',
	'SHORT_PACKET_SIZE',
	'LONG_PACKET_CHANNELS',
	'SHORT_PACKET_CHANNELS',
	'LONG_PACKET_VERSION',
	'SHORT_PACKET_VERSION',
	'PFBPACKET_NSAMP_MAX',
	'SS_PER_SECOND',

	# Backwards compatibility
	'DfmuxPacket',
	'STREAMER_MAGIC',
	'STREAMER_HOST',
	'STREAMER_TIMEOUT',
]

__version__ = '1.0.0'
