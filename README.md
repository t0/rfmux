# rfmux

[rfmux](https://github.com/t0/rfmux) is the Python API for
[t0.technology](https://t0.technology)'s Control and Readout System (CRS), a
hardware platform designed for operating large arrays of Kinetic Inductance
Detectors (KIDs) used in radio astronomy and quantum sensing applications.

The CRS + MKIDs firmware is described in
[this conference proceedings](https://arxiv.org/abs/2406.16266).

## Quick Start

### Installation

**New in 2025:** rfmux is now available on PyPI. We recommend using [uv](https://github.com/astral-sh/uv) for installation:

```bash
# Install uv (if not already installed)
$ curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS or WSL
# Or: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Create virtual environment 
uv venv ### or uv venv my-env-name
source .venv/bin/activate # On Windows: .venv/Scripts/activate
# or source my-env-name/bin/activate

# Install rfmux
$ uv pip install rfmux
```

**Note:** rfmux now uses a C++ extension for packet processing.  PyPI hosts
pre-built binaries (wheels) for common platforms (Linux x86_64, macOS,
Windows). If wheels aren't available for your platform, you'll need a C++
compiler. See [Installation Guide](docs/installation.md) for details.

### Interactive GUI

To launch the Periscope GUI, run:

```bash
$ uv run periscope # or periscope
```

https://github.com/user-attachments/assets/581d4ff8-5ea2-493a-9c9c-c93d6ca847e2

### Scripting with Mock Mode

If you do not have a CRS board (or cryogenic detectors) handy, you can use
"mock" mode for a software emulation:

```python
# Emulate CRS hardware for offline development
s = rfmux.load_session("""
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "MOCK0001" }
""")
```

### Scripting with CRS Hardware

To control a single network-attached CRS from your PC's Python prompt, use:

```python
import rfmux

# Connect to a CRS board
s = rfmux.load_session('!HardwareMap [ !CRS { serial: "0033" } ]')
crs = s.query(rfmux.CRS).one()
await crs.resolve()

# Acquire samples
samples = await crs.get_samples(1000, channel=1, module=1)
```

## Documentation

- **[Installation Guide](docs/installation.md)** - Detailed installation for all platforms, building from source
- **[Getting Started](docs/guides/getting-started.md)** - Usage patterns, hardware hierarchy, common operations
- **[Networking Guide](docs/guides/networking.md)** - UDP tuning, multicast configuration, troubleshooting
- **[Firmware Guide](docs/guides/firmware.md)** - Fetching, managing, and flashing firmware
- **[Release Notes](docs/release-notes/)** - What each release lets you do
- **[Tuning KIDs](rfmux/reference-notebooks/Demos/simplified_tuning_flow.md)** - Sweeping, fitting, biasing, measuring noise
- **[Pulse Capture](rfmux/reference-notebooks/Demos/pulse_capture.md)** - Detecting and recording detector pulses

The last two are runnable notebooks that ship with the package. See
[reference-notebooks/README.md](rfmux/reference-notebooks/README.md) for how to
open them.

## Repository Structure

```
rfmux/
├── docs/                    # Documentation
├── firmware/                # Firmware binaries (Git LFS)
├── rfmux/                   # Main Python package
│   ├── algorithms/          # Network analysis, fitting, biasing
│   ├── core/                # Hardware schema, sessions, transfer functions
│   ├── mock/                # Physics-based CRS simulator
│   ├── pulse_capture/       # Pulse detection, HDF5 capture, histograms
│   ├── reference-notebooks/ # Runnable demos, shipped with the package
│   ├── streamer/            # C++ packet receiver library
│   └── tools/               # Periscope GUI, CLI, firmware, QC suite
└── test/                    # Test suite
```

RPC to the board is provided by the separate
[tuber](https://pypi.org/project/tuber-client/) package.

## Contributing & Feedback

rfmux is permissively licensed; see LICENSE for details.

We actively encourage contributions and feedback. Understanding operator needs
is how we determine what to add to rfmux.

- **Pull Requests:** Your contributions are welcome
- **Issues:** Please submit tickets for bugs or enhancement suggestions
- **Collaborator Slack:** Join #crs-collaboration - email Joshua@t0.technology with your name, affiliation, and project

## Citation

When citing rfmux or CRS, please reference:
> CRS + MKIDs Conference Proceedings: https://arxiv.org/abs/2406.16266
