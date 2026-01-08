# Network Configuration Guide

This guide covers network tuning and troubleshooting for optimal rfmux performance.

## UDP Buffer Sizes

For reliable long-duration data captures, increase your system's UDP receive buffer size.

### Linux

**Temporary (until reboot):**
```bash
sudo sysctl net.core.rmem_max=67108864
```

**Permanent configuration:**

Add to `/etc/sysctl.conf` or create a file in `/etc/sysctl.d/` (e.g., `/etc/sysctl.d/99-rfmux.conf`):

```bash
net.core.rmem_max=67108864
```

Then reload:
```bash
sudo sysctl -p
```

### Windows

Increase buffer parameters via registry:

1. Press `Win+R`, type `regedit`, hit Enter
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\Afd\Parameters`
3. Create or modify DWORD (32-bit) values:
   - `DefaultReceiveWindow` = 67108864 (decimal)
   - `MaximumBufferSize` = 67108864 (decimal)
   - `MaximumDynamicBufferSize` = 67108864 (decimal)
4. Reboot for changes to take effect

### macOS

```bash
sudo sysctl -w kern.ipc.maxsockbuf=67108864
sudo sysctl -w net.inet.udp.recvspace=67108864
```

## Multicast Configuration

rfmux uses Source-Specific Multicast (SSM) for data streaming:

- **Multicast group:** 239.192.0.2
- **Data port:** 9876 (downsampled readout data)
- **PFB port:** 9877 (polyphase filterbank data)
- **Filtering:** By CRS serial number

SSM allows multiple CRS boards on the same network without interference.

## MTU (Maximum Transmission Unit)

**Important:** Mismatched MTU settings can cause packet loss or connectivity issues.

### Linux

Check your interface MTU:
```bash
ip link show eth0
```

Set jumbo frames (if your network supports it):
```bash
sudo ip link set eth0 mtu 9000
```

### Windows

**Warning:** Windows does not accept MTU hints from DHCP, and this limitation extends to Hyper-V guests.

To check/set MTU:
```powershell
netsh interface ipv4 show interfaces
netsh interface ipv4 set subinterface "Ethernet" mtu=9000 store=persistent
```

If you're experiencing packet resolution issues on a jumbo-frame network, verify your host MTU matches the network configuration.

## Ethernet Interrupt Coalescing

For high-decimation captures (stage 6+), tune Ethernet interrupt coalescing to reduce low-volume packet loss:

**Temporary:**
```bash
sudo ethtool -C eth0 rx-usecs 150000
```

**Permanent (systemd):**

Create `/etc/systemd/network/10-eth0.link`:

```ini
[Match]
MACAddress=aa:bb:cc:dd:ee:ff  # Replace with your interface's MAC address

[Link]
Name=eth0
ReceiveQueues=4096      # Default may be smaller (check: ethtool -g eth0)
RxCoalesceSec=150ms     # Default may be smaller (check: ethtool -c eth0)
```

Restart networking or reboot to apply.

## Monitoring Packet Loss

Use the `parser` tool to monitor dropped packets:

```bash
parser --drop-stats
```

This reports statistics on:
- Total packets received
- Packets dropped by kernel
- Sequence number gaps
- Buffer overruns

High drop rates indicate:
1. Insufficient UDP buffer size
2. MTU mismatch
3. Network congestion
4. Slow packet processing (check CPU usage)

## Firewall Configuration

Ensure your firewall allows:
- **UDP multicast** on group 239.192.0.2
- **Ports 9876 and 9877** for incoming data
- **IGMP** for multicast group management

### Linux (ufw)

```bash
sudo ufw allow 9876/udp
sudo ufw allow 9877/udp
```

### Linux (iptables)

```bash
sudo iptables -A INPUT -p udp --dport 9876 -j ACCEPT
sudo iptables -A INPUT -p udp --dport 9877 -j ACCEPT
sudo iptables -A INPUT -p igmp -j ACCEPT
```

### Windows

Windows may prompt for firewall permission when running `py_get_samples` for the first time. Grant access to Python.

## CRS Board Discovery

CRS boards advertise via mDNS/Bonjour as `rfmux<serial>.local`:

```bash
# Ping a CRS board
ping rfmux0033.local

# Check mDNS resolution (Linux)
avahi-browse -a

# Check mDNS resolution (macOS)
dns-sd -B _http._tcp
```

If mDNS doesn't work, specify the IP address directly in your hardware map YAML:

```yaml
!HardwareMap
- !CRS
  serial: "0033"
  hostname: "192.168.1.100"  # Use IP instead of rfmux0033.local
```

## Troubleshooting

### No packets received

1. Check firewall rules
2. Verify multicast routing: `ip route` should show multicast routes
3. Check interface is multicast-capable: `ip link show`
4. Verify CRS is streaming: check board status LEDs

### Intermittent packet loss

1. Increase UDP buffer size
2. Check interrupt coalescing settings
3. Monitor CPU usage during capture
4. Verify MTU matches network configuration

### High latency

1. Check network switch configuration
2. Disable flow control if not needed
3. Check for network congestion (use `iperf3` to test bandwidth)
4. Verify no VLANs causing routing delays

## Performance Tuning Summary

For optimal performance:

```bash
# Increase UDP buffers
sudo sysctl -w net.core.rmem_max=134217728

# Set jumbo frames (if supported)
sudo ip link set eth0 mtu 9000

# Tune interrupt coalescing
sudo ethtool -C eth0 rx-usecs 150000

# Monitor for drops
parser --drop-stats
```

Adjust values based on your specific data rates and network environment.
