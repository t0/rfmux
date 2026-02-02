"""
CRS Streaming Protocol

Unified API for CRS packet streaming, including:
- Packet structures (ReadoutPacket, PFBPacket, Timestamp)
- High-performance C++ packet receiver
- Socket utilities for multicast configuration
- Protocol constants
"""

# Import C++ packet receiver and structures, and ensure version parity
_PY_API_VERSION = 0  # must match _SO_API_VERSION in bindings.cpp
try:
    from ._receiver import _SO_API_VERSION
except ModuleNotFoundError as e:
    import textwrap
    raise ModuleNotFoundError(textwrap.dedent(
        '''
        rfmux recently integrated a c++ extension for faster packet processing.
        This extension requires a compile/install step that was not previously
        necessary.

        Try `pip install -e .` from the repository root, or see README.md for
        details.
        ''')) from e
except ImportError as e:
    _SO_API_VERSION = 0

if _SO_API_VERSION != _PY_API_VERSION:
    import textwrap
    raise ImportError(textwrap.dedent(
    f'''
    C++ fastpath: API version mismatch; {_SO_API_VERSION=}, {_PY_API_VERSION=}

    You probably need to recompile the _receiver extension. You can do this
    with something like

        pip install -e . --force-reinstall

    from the repository root. _receiver is a fairly recent addition to rfmux;
    please see README.md for details.
    '''))

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
	PFB_PACKET_SIZE,
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
