"""
Socket utilities for CRS multicast streaming.

Provides cross-platform multicast socket configuration.
"""

import socket
import sys
import warnings


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
			f"SO_RCVBUF was adjusted to a smaller buffer size by the kernel from {set_size} to {actual_size} bytes. "
			f"To avoid packet loss, set 'sudo sysctl net.core.rmem_max=67108864' or similar. This setting "
			f"can be made persistent across reboots by configuring /etc/sysctl.conf or /etc/sysctl.d."
		)

	return actual_size


def get_multicast_socket(crs_hostname, port=None, interface=None, buffer_size=None):
	"""
	Create and configure a multicast socket for receiving CRS packets.

	Args:
		crs_hostname (str): The hostname or IP address of the CRS device
		port (int, optional): Port number. If None, uses STREAMER_PORT from constants
		interface (str, optional): Local interface IP address. If None, auto-detects from hostname
		buffer_size (int, optional): Socket receive buffer size in bytes. If None, uses default (16MB)

	Returns:
		socket.socket: Configured multicast socket
	"""
	# Import here to avoid circular dependency
	from . import STREAMER_PORT, MULTICAST_GROUP
	from ._receiver import ip_mreq_source

	if port is None:
		port = STREAMER_PORT

	# Extract just the hostname part for source address resolution
	hostname_only = None
	if crs_hostname:
		if ':' in crs_hostname:
			hostname_only = crs_hostname.split(':')[0]
		else:
			hostname_only = crs_hostname

	# Check if this is a localhost connection (MockCRS uses multicast)
	if hostname_only and hostname_only in ["127.0.0.1", "localhost", "::1"]:
		# Create a multicast socket for MockCRS (using multicast on loopback)
		sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
		sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

		# For multicast on Linux, we may need SO_REUSEPORT as well
		try:
			sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
		except AttributeError:
			pass  # SO_REUSEPORT not available on all platforms

		# Bind to all interfaces on the streamer port (needed for multicast)
		sock.bind(("", port))

		# Set receive buffer size with platform-aware handling
		if buffer_size:
			_set_socket_buffer_size(sock, buffer_size)
		else:
			_set_socket_buffer_size(sock)

		# Enable multicast loopback to receive our own packets
		sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)

		# Set the multicast interface to loopback for receiving
		sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton("127.0.0.1"))

		# Join the multicast group specifically on loopback interface
		import struct
		mreq_lo = struct.pack("4s4s", socket.inet_aton(MULTICAST_GROUP), socket.inet_aton("127.0.0.1"))
		sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq_lo)

		return sock

	# Original multicast socket code for real hardware
	multicast_interface_ip = interface if interface else get_local_ip(crs_hostname)

	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

	# Configure the socket for multicast reception
	sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

	try:
		# Will only work Linux > 3.9 and Mac > 10.6
		sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
	except:
		pass

	# Bind the socket to all interfaces on the specified port
	sock.bind(("", port))

	# Set receive buffer size with platform-aware handling
	if buffer_size:
		_set_socket_buffer_size(sock, buffer_size)
	else:
		_set_socket_buffer_size(sock)

	# Set the interface to receive multicast packets
	sock.setsockopt(
		socket.IPPROTO_IP,
		socket.IP_MULTICAST_IF,
		socket.inet_aton(multicast_interface_ip),
	)

	# Join the multicast group on the specified interface
	if hostname_only:
		# Source-specific multicast (SSM) - only receive from this source
		mreq = ip_mreq_source(
			multiaddr=MULTICAST_GROUP,
			sourceaddr=socket.gethostbyname(hostname_only),
			interface=multicast_interface_ip
		)
		sock.setsockopt(socket.IPPROTO_IP, ip_mreq_source.IP_ADD_SOURCE_MEMBERSHIP, mreq.to_bytes())
	else:
		# Regular multicast - receive from all sources
		import struct
		mreq = struct.pack("4s4s", socket.inet_aton(MULTICAST_GROUP), socket.inet_aton(multicast_interface_ip))
		sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

	return sock
