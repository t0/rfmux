# Installation Guide

## Pre-Requisites

### macOS: OpenMP (optional)

For optimal threading performance in mock mode:

```bash
brew install libomp

# Add to ~/.zshrc (or ~/.bash_profile):
export NUMBA_THREADING_LAYER=omp
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
# For Intel Macs: /usr/local/opt/libomp/lib:$DYLD_LIBRARY_PATH

# Reload your shell config:
source ~/.zshrc
```

### Linux: GUI Support for Periscope

```bash
sudo apt-get install libxcb-cursor0
```

## Installation from PyPI

**New (Jan 2025):** rfmux is now available on PyPI with pre-built wheels for
common platforms (Linux x86_64, macOS, Windows). No compiler needed for most
users.

### Using uv (recommended)

We recommend [uv](https://github.com/astral-sh/uv) for installing rfmux. Modern
Linux distributions (Debian, Ubuntu, etc.) now prevent `pip install` from
modifying system packages (PEP 668), and `uv` elegantly handles this by
automatically managing virtual environments. It's also significantly faster and
provides consistent behavior across all platforms.

```bash
# Install uv (once)
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install rfmux
uv pip install rfmux
```

Or create a virtual environment explicitly:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install rfmux
```

### Using traditional pip

**Note:** System-wide `pip install` (without a virtual environment) is blocked
on modern Linux distributions. Always use a virtual environment.

If you prefer traditional `pip`:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install rfmux
pip install rfmux
```

## Development Installation

### Pre-Requisites

To compile from source, you will need build tools for your platform:

#### Linux
```bash
sudo apt-get install build-essential python3-dev
```

#### macOS
```bash
xcode-select --install
```

#### Windows

Install [Visual Studio Community Edition](https://visualstudio.microsoft.com/downloads/).

### Build Instructions

```bash
# Clone repository
git clone https://github.com/t0/rfmux.git
cd rfmux

# Create virtual environment and install in editable mode
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```
