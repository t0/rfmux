# CLAUDE.md - rfmux Project Reference

## Project Overview

**rfmux** is the Python API for the t0.technology Control and Readout System (CRS), used to operate Kinetic Inductance Detectors (KIDs) for astrophysics and particle physics applications.

### Core Components
- **Python API** (`rfmux/core/`): Hardware abstraction for CRS boards
- **Algorithms** (`rfmux/algorithms/`): KID measurement algorithms (network analysis, multisweep, pulse capture)
- **Periscope** (`rfmux/tools/periscope/`): Real-time PyQt6 GUI for data visualization
- **Streamer** (`rfmux/streamer/`): C++ extension for high-performance packet reception
- **Mock System** (`rfmux/mock/`): Physics-based CRS simulator with Numba JIT

## Current Active Work

**Branch**: `buffer_exploration`
**Project**: Pulse Capture → Periscope Integration

Integrating real-time pulse detection into Periscope with:
- Streaming HDF5 persistence (write-as-you-go)
- Live histogram visualization
- Callback-driven PulseCapture engine
- Session browser integration

**Phase 1** (next): Callback-driven PulseCapture refactor + PulseHDF5Writer/Reader + PulseHistogramSet

See `memory-bank/pulse_capture_periscope_integration.md` for full specification.

## Development Setup

```bash
# Requires Python 3.10+, Git LFS
pip install -e .

# Linux: Required for Periscope GUI
sudo apt-get install libxcb-cursor0

# Optional: Increase UDP buffer for long captures
sudo sysctl net.core.rmem_max=67108864
```

## Code Style & Conventions

### Python Style
- **4 spaces** indentation (vim modeline: `sts=4 ts=4 sw=4 tw=78 smarttab expandtab`)
- **Type annotations** on function signatures
- **snake_case** for functions/variables, **CamelCase** for classes
- **Leading underscore** for private/internal methods (`_helper_method`)
- Minimal docstrings; prefer self-documenting code

### Periscope GUI Patterns

**Panels** inherit from `QWidget + ScreenshotMixin`:
```python
class MyPanel(QtWidgets.QWidget, ScreenshotMixin):
    def __init__(self, parent=None, *, crs=None, dark_mode=False):
        super().__init__(parent)
        self._setup_ui()  # Calls _setup_toolbar(), _setup_plot_area()
```

**Toolbars**: Plain `QWidget` with `QHBoxLayout` (not QToolBar)

**Tasks**: `QThread` subclasses with signal objects:
```python
class MyTaskSignals(QObject):
    progress = pyqtSignal(int, float)
    completed = pyqtSignal(dict)
    error = pyqtSignal(str)

class MyTask(QThread):
    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # ... async CRS operations
```

**Thread safety**:
- UI updates only in GUI thread (via signals)
- Use `queue.Queue` for GUI → worker thread data
- Use `pyqtSignal` for worker → GUI thread events

**File dialogs**: Use non-modal `dlg.open()` (not `exec()`) to prevent Linux hangs

### Naming Conventions
- `*Panel` - Dockable widget
- `*Dialog` - Modal configuration window
- `*Task` - Background QThread worker
- `*Signals` - Signal container for tasks
- `*Mixin` - Composable functionality

### Plot Colors (utils.py)
- `IQ_COLORS`: I="#3366CC" (blue), Q="#CC6633" (orange)
- `TABLEAU10_COLORS`: Default series colors
- `AMPLITUDE_COLORMAP_THRESHOLD = 3`: Use distinct colors if ≤3 amplitudes

## Key Architecture Patterns

### Data Flow
```
CRS Hardware → UDP Multicast → Streamer (C++) → Python API → Periscope
                                                    ↓
                                              Algorithms (async macros)
```

### Algorithm Registration (@macro decorator)
```python
from rfmux.core.hardware_map import macro

@macro(CRS)
async def my_algorithm(crs, module, channels, **params):
    # Called as: await crs.my_algorithm(module=1, channels=[1,2])
```

### Results Data Structure
`results_by_detector`: `{detector_id: {iteration_index: {data + amplitude, direction}}}`

**Critical**: Dictionary keys are indices, not frequencies. Extract actual values:
```python
# WRONG: freq = list(results.keys())[0]  # This is an index!
# CORRECT: freq = results[idx]['bias_frequency']
```

### Unit Conversion (UnitConverter class)
- Raw ADC counts ↔ Volts ↔ dBm
- Use `convert_amplitude()` with `unit_mode` parameter
- Normalization at display time, preserve raw data

### Session System
```python
# Export path in session folder
path = session_mgr.get_export_path("category", "label", ".pkl")
# Auto-appears in Session Browser panel
```

## Important Technical Notes

### MockCRS Physics
- `jit_physics.py` requires Numba; no Python fallback
- `compute_s21_parallel()` handles attenuation internally — don't double-apply
- Single convergence loop: `converged_lekid_parameters()`
- Reproducibility requires concrete `resonator_random_seed` in config

### Streaming
- Slow stream: ~38 kHz at dec=1, port 9876, `ReadoutPacket`
- PFB stream: ~1.22 MHz, port 9877, `PFBPacket`
- `get_multicast_socket()` uses `SO_REUSEPORT` — multiple listeners OK
- Mock streams to the same multicast group as hardware, with TTL 0 so it
  cannot leave the host. If multicast does not work on the machine it
  falls back to loopback unicast and prints which step failed —
  `check_multicast_loopback()` in `rfmux/streamer`

### Threading
- Periscope: Qt event loop + asyncio integration
- Long operations: `QThread` with own asyncio loop
- h5py not thread-safe — write from single thread only

## File Structure

```
rfmux/
├── core/           # CRS, session, hardware_map, schema
├── algorithms/     # measurement/ (fitting, multisweep, pulse_detection)
├── tools/
│   └── periscope/  # GUI panels, dialogs, tasks, utils
├── mock/           # MockCRS, resonator physics
├── streamer/       # C++ packet receiver
├── tuber/          # RPC framework
└── mr_resonator/   # Numba physics (subset of external lib)
```

## Testing

```bash
pytest --tier=quick                 # Edit loop: 537 tests, ~50 s, zero skips
pytest --tier=portable              # No CRS, no GUI: 37 tests, ~9 s
pytest --tier=full                  # All 553 that run without a board, ~9.5 min
pytest --tier=acquisition           # MockCRS server + real UDP: 16 tests, ~8.5 min (inside full)
pytest --tier=hardware --serial 0024  # 75 tests, needs a real CRS
pytest test/pulse_capture/          # One subsystem
python -m rfmux.tools.periscope     # Launch Periscope
```

`--tier` (defined in `test/conftest.py`) names an invocation; every tier but
`hardware`/`all` excludes the board tests, so they report zero skips. Markers
tag tests: `portable`, `slow_acquisition`, `hardware` — the last applied
automatically to anything using the `crs`/`live_session`/`serial` fixtures,
so don't add it by hand. A bare `pytest` still runs 553 + ~75 hardware skips.
`full` contains `acquisition`; running both pays the slow set twice.

`test/` is organized into subdirectories mirroring the package under test
(`core/`, `streamer/`, `mock/`, `algorithms/`, `periscope/`, `notebooks/`),
plus `pulse_capture/` for the subsystem that spans algorithms and Periscope.
See `test/README.md` for the layout, markers, and Qt/notebook conventions.

Mock mode: `periscope --mock` or configure via Mock Configuration dialog

## Common Tasks

### Add a new Periscope panel
1. Create `rfmux/tools/periscope/my_panel.py`
2. Inherit from `QWidget, ScreenshotMixin`
3. Implement `_setup_ui()`, `apply_theme()`
4. Add menu entry in `app.py`
5. Register with dock manager

### Add a new CRS algorithm
1. Create function in `rfmux/algorithms/measurement/`
2. Decorate with `@macro(CRS)`
3. Add to `__all__` in `algorithms/__init__.py`

### Debug pulse capture
- `pytest test/pulse_capture/` (assertions) or
  `pytest --tier=acquisition` (end to end, real UDP)
- Check `threshold_sigma` (3-5 typical, 50 is debug artifact)
- Verify noise estimation with `estimate_noise_stats()`
