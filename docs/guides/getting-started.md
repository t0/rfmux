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

For an interactive session it is usually easier to use the helper, which builds
the session, resolves it, configures simulated resonators and starts streaming
in one call:

```python
from rfmux.mock.helpers import create_mock_crs

crs = await create_mock_crs(module=1, config={"num_resonances": 10})
```

Note that a simulated CRS is *created*, not discovered: its RPC port is assigned
at startup, so a second `create_mock_crs()` gives you a second, unrelated
simulation — and both stream to the same UDP port, where one reader sees the two
interleaved with nothing raised. Start one per machine, and attach to it by
hostname if another process needs it.

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
└── HWMResonator         # Individual KID resonator (hardware-map row)
    └── ChannelMapping   # Maps resonators to readout channels
```

Query examples:

```python
# Get a specific CRS board
crs = s.query(rfmux.CRS).filter_by(serial="0033").one()

# Get all modules on a CRS
modules = s.query(rfmux.ReadoutModule).filter_by(crs=crs).all()

# Get one channel, or a set of them
channel = s.query(rfmux.ReadoutChannel).filter_by(
    module=modules[0], channel=1
).one()

channels = s.query(rfmux.ReadoutChannel).filter(
    rfmux.ReadoutChannel.module == modules[0],
    rfmux.ReadoutChannel.channel.in_([1, 2, 3]),
).all()
```

## Common Operations

### Network Analysis

Most measurement algorithms are registered on the CRS object itself, so you call
them as methods rather than importing them — `module` is keyword-only:

```python
# Sweep 600 MHz - 1.1 GHz and return frequencies, iq_complex, phase_degrees
result = await crs.take_netanal(
    amp=0.001, fmin=0.6e9, fmax=1.1e9, npoints=50000, module=1
)
```

### Acquiring Samples

```python
# Get time-domain samples
samples = await crs.get_samples(num_samples=1000, channel=1, module=1)

# Python-based UDP receiver (platform-independent)
samples = await crs.py_get_samples(num_samples=1000, channel=1, module=1)
```

### Biasing KIDs

`bias_kids` picks an operating point per detector and programs the hardware. It
works from multisweep results — it needs each resonator characterised before it
can choose where to park a carrier — so it is the last step of the tuning
sequence, not a standalone call.

It is one of the algorithms that is *not* registered on `CRS`, so unlike the
calls above it is imported and takes `crs` as an argument:

```python
from rfmux.algorithms.measurement.bias_kids import bias_kids

sweeps = await crs.multisweep(
    center_frequencies=resonance_frequencies,
    span_hz=500e3, npoints_per_sweep=500, amp=0.001, module=1,
)
bias_results = await bias_kids(crs=crs, multisweep_results=sweeps, module=1)

# {detector_index: {"bias_frequency": ..., "df_calibration": ..., ...}}
```

The full sequence — sweep, unwrap cable delay, find resonances, multisweep, fit,
bias, measure noise — is documented and runnable in
`rfmux/reference-notebooks/Demos/simplified_tuning_flow.md`.

## IPython Integration

rfmux depends on the `awaitless` package, which lets you call coroutines in
IPython and Jupyter without writing `await`. `import rfmux` loads it for you
whenever it detects an IPython session — there is nothing to enable:

```python
# In IPython/Jupyter, these are equivalent:
await crs.resolve()
crs.resolve()  # awaitless supplies the await

# In a plain .py script neither applies: use asyncio.run() around an async main().
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
  hwm_resonators: !HWMResonators
    csv_file: "resonators.csv"

- !ChannelMappings
  csv_file: "mappings.csv"
```

Common tags:
- `!HardwareMap` - Top-level hardware configuration
- `!CRS` - CRS board definition
- `!Wafer` - Detector wafer
- `!HWMResonators` - Bulk import of resonators from CSV
- `!ChannelMappings` - Channel-to-resonator mappings from CSV
- `!flavour "rfmux.mock"` - Enable mock mode

## Network Configuration

For reliable data streaming, you may need to increase UDP receive buffer sizes and configure multicast. See the [Networking Guide](networking.md) for platform-specific instructions.

## Next Steps

- Launch Periscope for real-time visualization: `python -m rfmux.tools.periscope`
  (add `--mock` for a simulated board)
- [Configure networking](networking.md) for optimal data streaming
- [Flash firmware](firmware.md) to update CRS boards
- Work through the runnable reference notebooks in
  `rfmux/reference-notebooks/Demos/`:
  - `simplified_tuning_flow.md` — sweep, find, fit, bias and measure noise
  - `pulse_capture.md` — detect and record detector pulses

  These are jupytext markdown, not `.ipynb`. Periscope's Jupyter panel opens
  them as notebooks on double-click; in your own JupyterLab use right-click →
  *Open With* → *Notebook*.
