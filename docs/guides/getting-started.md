# Getting Started with rfmux

This guide will walk you through basic usage patterns for rfmux.

## Hardware Map Sessions

rfmux models your hardware setup using SQLAlchemy ORM objects. You create a session from YAML configuration:

```python
import rfmux

# Load hardware map from YAML
s = rfmux.load_session('!HardwareMap [ !CRS { serial: "0033" } ]')

# Query for CRS board
crs = s.query(rfmux.CRS).one()

# Resolve remote connection (async)
await crs.resolve()
```

## Mock Mode for Offline Development

Mock mode emulates CRS hardware and KID arrays without physical hardware:

```python
s = rfmux.load_session("""
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "MOCK0001" }
""")

crs = s.query(rfmux.CRS).one()
await crs.resolve()
```

Mock mode emulates:
- KID non-linear inductance
- Tuning algorithms
- Signal processing at baseband
- Real-time visualization

## Hardware Hierarchy

rfmux models hardware with SQLAlchemy ORM:

```python
Crate                    # Physical crate containing CRS boards
└── CRS                  # Control and Readout System board (identified by serial number)
    └── ReadoutModule    # Module on a CRS board (1-4 per board)
        └── ReadoutChannel  # Individual channel (1-1024 per module)

Wafer                    # Detector wafer
└── Resonator            # Individual KID resonator
    └── ChannelMapping   # Maps resonators to readout channels
```

Query examples:

```python
# Get a specific CRS board
crs = s.query(rfmux.CRS).filter_by(serial="0033").one()

# Get all modules on a CRS
modules = s.query(rfmux.ReadoutModule).filter_by(crs=crs).all()

# Get specific channels
channels = s.query(rfmux.ReadoutChannel).filter_by(
    module=modules[0],
    channel_number=[1, 2, 3]
).all()
```

## Common Operations

### Network Analysis

```python
from rfmux.algorithms.measurement import take_netanal

# Run network analysis sweep
result = await take_netanal(crs, module=1, channel=1)
```

### Acquiring Samples

```python
# Get time-domain samples
samples = await crs.get_samples(num_samples=1000, channel=1, module=1)

# Python-based UDP receiver (platform-independent)
samples = await crs.py_get_samples(num_samples=1000, channel=1, module=1)
```

### Biasing KIDs

```python
from rfmux.algorithms.measurement import bias_kids

# Auto-bias resonators
await bias_kids(crs, module=1, channels=[1, 2, 3])
```

## IPython Integration

rfmux integrates with IPython via the `awaitless` module, which automatically transforms `await` expressions:

```python
# In IPython/Jupyter, these are equivalent:
await crs.resolve()
crs.resolve()  # awaitless automatically adds await

# To use awaitless, start IPython with:
ipython --TerminalIPythonApp.exec_lines='import rfmux.awaitless'
```

## YAML Configuration Tags

Custom YAML tags define hardware:

```yaml
!HardwareMap
- !CRS
  serial: "0033"
  hostname: "rfmux0033.local"

- !Wafer
  name: "test_wafer"
  resonators: !Resonators
    csv_file: "resonators.csv"

- !ChannelMappings
  csv_file: "mappings.csv"
```

Common tags:
- `!HardwareMap` - Top-level hardware configuration
- `!CRS` - CRS board definition
- `!Wafer` - Detector wafer
- `!Resonator` - Individual resonator
- `!Resonators` - Bulk import from CSV
- `!ChannelMappings` - Channel-to-resonator mappings from CSV
- `!flavour "rfmux.core.mock"` - Enable mock mode

## Network Configuration

For reliable data streaming, you may need to increase UDP receive buffer sizes and configure multicast. See the [Networking Guide](networking.md) for platform-specific instructions.

## Next Steps

- [Launch Periscope](periscope.md) for real-time visualization
- [Configure networking](networking.md) for optimal data streaming
- [Flash firmware](firmware.md) to update CRS boards
- Explore example notebooks in `home/Demos/`
