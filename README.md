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

https://github.com/user-attachments/assets/fa8eb68c-8593-43b1-af96-285720ed0d5f

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

## Repository Structure

```
rfmux/
├── docs/                # Documentation
├── firmware/            # Firmware binaries (Git LFS)
├── home/                # Jupyter Hub content (demos, docs)
├── rfmux/               # Main Python package
│   ├── algorithms/      # Network analysis, fitting, biasing
│   ├── core/            # Hardware schema, sessions, mock infrastructure
│   ├── packets/         # C++ packet receiver library
│   ├── tools/           # Periscope GUI and other tools
│   └── tuber/           # RPC/remote-object communication
└── test/                # Test suite (unit, integration, QC)
```

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
