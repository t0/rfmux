# rfmux Project Reference

## Project Overview

**rfmux** is the Python API for the t0.technology Control and Readout System (CRS), used to operate Kinetic Inductance Detectors (KIDs) for astrophysics and particle physics applications.

### Core Components
- **Python API** (`rfmux/core/`): Hardware abstraction for CRS boards
- **Algorithms** (`rfmux/algorithms/`): KID measurement algorithms (network analysis, multisweep, df calibration, streamer configuration, one-shot `trigger_capture`)
- **Pulse capture** (`rfmux/pulse_capture/`): the trigger engine, its compiled per-sample walk (`walk.py`), the stream sources, the dual-stream session, and the HDF5 record
- **Periscope** (`rfmux/tools/periscope/`): Real-time PyQt6 GUI for data visualization
- **Streamer** (`rfmux/streamer/`): C++ extension for high-performance packet reception
- **Mock System** (`rfmux/mock/`): Physics-based CRS simulator with Numba JIT

## Development Setup

```bash
# Requires Python 3.10+, Git LFS
uv pip install -e .   # rebuilds the C++ extension

# Linux: Required for Periscope GUI
sudo apt-get install libxcb-cursor0

# Optional: Increase UDP buffer for long captures
sudo sysctl net.core.rmem_max=268435456
```

## After every change: the review pass

Every feature, bug fix or improvement gets this pass before it is
reported, without being asked. It is what the explicit review passes on
this project have kept asking for.

- **Claims against the code, not the summary.** Every statement in
  prose, comments, tooltips, docs and figures is checked against the
  source it describes: units, defaults, which stream, which attribute.
  A stale or wrong statement is a bug.
- **Measure, don't assert.** A performance or behaviour claim comes
  with a number from a run, before and after. The report says what was
  not verified and why (a port held, no board).
- **Net code.** Prefer deletion and simplification. If a change is
  growing a lot of new code, stop and say so. Never re-create what an
  existing path does (the toolchain, a helper, a kernel): call it.
- **Where code lives.** Orchestration and anything that can run
  headlessly lives in `rfmux/algorithms` or `rfmux/mock`; Periscope is
  a thin caller. A mock property is not a Periscope property.
- **Comments say what and why**, never the history of how it got there
  ("used to", "previously"). Narrative belongs in the commit message.
  Minimal docstrings. The release note is a how-to, not a changelog.
- **Tests pin contracts, not re-enactments.** Assert what a caller
  relies on, one behaviour per test; incidental details (packet
  boundaries, dispatch counts) are not contracts unless stated. Every
  bug fix carries the test that fails without it.
- **No unnecessary dialogs or warnings.** Routine outcomes go to the
  status line or the console; a dialog is for a failure the user must
  act on. No artificial caps: use the system's value and warn if it is
  too low.
- **Defaults and docs move together.** Changing a default updates every
  example, tooltip, validation message and figure that cites it, and
  the tier counts when tests are added.
- **Judgement calls are listed** in the report with the choice made, so
  they can be overruled.
- **Commit hygiene.** No attribution trailers. Stage explicit paths.
  Gate a commit on its test run in one command. Do not push unless
  asked.
- **The report** leads with what changed and what was measured, states
  what was left out and why, and shows test failures verbatim.

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

**Toolbars**: Plain `QWidget` with `FlowLayout` from `layouts.py` (not
QToolBar), so controls wrap on a laptop screen; `grouped`/`labelled` there
keep a label with its control

**Board work is a session cell.** A panel acts on the board only by
handing the embedded console a cell, so the console shows every
operation as the Python that reproduces it, and a user can type the
same line. `run_python_then(code, comment, done)` in `app_runtime.py`
runs the cell and calls `done(future)` on the GUI thread; the result is
read back from `session_namespace()` under the name the cell assigned.
```python
self.run_python_then(
    f"netanal_0[{amp!r}] = await crs.take_netanal(amp={amp!r}, module={bank!r}, "
    f"**periscope.netanal_hooks({window_id!r}, {amp!r}))",
    "Network Analysis #1", done)
```
The rules:
- A cell is flat calls with literal values, written with `repr`. No
  loops or comprehensions: if a panel needs a loop or a `gather`, that
  orchestration belongs in the algorithm (as `take_netanal` takes a
  module list), and the cell stays one call. An argument may be a plain
  function of a result already named in the session, such as
  `center_frequencies=bias_frequencies(multisweep_0[(0.001, 'upward')])`,
  which is how one step feeds the next without pasting its numbers.
- Values are plain Python where they are produced. A numpy scalar
  renders as `np.float64(...)` and fails in the cell; convert at the
  source, not in the f-string.
- Argument checks live in the algorithm, which raises `ValueError`;
  the dialog parses text into typed values and does not re-validate.
- GUI callbacks are passed in the open, as `**periscope.<hooks>(...)`,
  so the line is what ran; dropping them gives the line a user types.
- Cells run one at a time on the interpreter thread, so the console is
  busy during an operation and typed input waits, as in Vivado. The
  GUI thread is never blocked.
- **A panel's data is its named result.** The cell assigns a session
  name (`netanal_0`, `multisweep_0`, `noise_m1`); the panel holds
  `result_name` and derives its view from the value with `set_result`.
  An export carries `name`, `result` (the value) and `cells` (what built
  it) beside the display-oriented keys older readers use. Loading a
  file binds the name again with a `load_result(...)` cell, so a loaded
  panel's data can feed the next step just like a live one.

**Tasks**: the remaining `QThread` subclasses (`MultisweepTask`,
`BiasKidsTask`, `PulseCaptureTask`) hold orchestration that has not yet
moved into `rfmux/algorithms`; new board work should not add one.

**Thread safety**:
- UI updates only in GUI thread (via signals)
- Callbacks handed to a cell run on the interpreter thread: emit a
  `pyqtSignal` from them, never touch a widget
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

@macro(CRS, register=True)
async def my_algorithm(crs, module, channels, **params):
    # Called as: await crs.my_algorithm(module=1, channels=[1,2])
```
Without `register=True` the function is not attached to `CRS`: import it
and call `await my_algorithm(crs, module=1, channels=[1,2])`.

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
- `compute_s21_parallel()` handles attenuation internally: do not apply it again
- Single convergence loop: `converged_lekid_parameters()`, run by
  `converge_tones()` for every tone of an evaluation at once, each seeded
  with the currents its resonators last had under it (the model's state
  memory) and stepped adaptively, so a bifurcated resonance is
  hysteretic: currents are a small fraction of Istar (a linewidth of
  shift is a 1e-4 change in Lk)
- `envelope_parameters()` gives each resonator's Kerr-resonator
  parameters (omega_r, kappa, K, D, c) from its linear response;
  `rfmux/mock/kerr.py` linearises about a driven state (rates, probe
  response, idler, gain); with `envelope_dynamics` on, a pulsed
  resonator's current rings down to each new steady state and the
  transient reaches the stream (visible at dec 0 and on PFB)
- Tones sharing a resonator each shift the resonance the others see:
  each tone's solve adds twice the other tones' |I|^2 per resonator
  (cross-phase modulation of an instantaneous nonlinearity), from the
  states they last left
- Reproducibility requires concrete `resonator_random_seed` in config

### Streaming
- Slow stream: ~38 kHz at dec=0, halving per stage, port 9876, `ReadoutPacket`
- PFB stream: ~2.44 MHz, port 9877, `PFBPacket`
- The C++ receiver hands Periscope and the PFB source one demuxed array per queue read (`pop_readout_batch`, `pop_pfb_batch`); the per-packet conversions remain the reference
- `get_multicast_socket()` uses `SO_REUSEPORT`, so several listeners can share the port
- Mock slow stream is generated a ~50 ms block of frames at a time (one
  physics call, one packet per frame); 100 tones run at real time
- Mock slow stream carries every module 1-4 with a configured channel
  (frequency, amplitude or phase set), module 1 if none; the mock PFB
  stream carries the module passed to `set_pfb_streamer`
- Mock streams to the same multicast group as hardware, with TTL 0 so it
  cannot leave the host. If multicast does not work on the machine it
  falls back to loopback unicast and prints which step failed
  (`check_multicast_loopback()` in `rfmux/streamer`)

### Threading
- Periscope: Qt event loop on the GUI thread; one interpreter thread
  with one asyncio loop (`console_kernel.Interpreter`) where every
  console cell, typed or panel-issued, executes. The console is created
  at startup and its dock is visible by default.
- The in-process kernel has no interrupt of its own; `ConsoleKernel`
  tracks the running `await` cell's task so Ctrl-C and a panel's cancel
  can cancel it. A typed blocking call (`time.sleep`) cannot be stopped.
- Raised from IPython or Jupyter, that session is the console: cells run
  on the interpreter thread in its namespace and are printed.
- h5py is not thread-safe: write from one thread

## File Structure

```
rfmux/
├── core/           # CRS, session, hardware_map, schema
├── algorithms/     # measurement/ (fitting, multisweep, streamer config)
├── pulse_capture/  # detection engine, walk, sources, session, hdf5
├── tools/
│   └── periscope/  # GUI panels, dialogs, tasks, utils
├── mock/           # MockCRS, resonator physics
├── streamer/       # C++ packet receiver
└── mr_resonator/   # Numba physics (subset of external lib)
```

## Testing

```bash
pytest --tier=quick                 # Edit loop: 1075 tests, ~1 min
pytest --tier=portable              # No CRS, no GUI: 43 tests, ~9 s
pytest --tier=full                  # All 1097 that run without a board, ~4 min
pytest --tier=acquisition           # MockCRS server + real UDP: 22 tests, ~3 min (inside full)
pytest --tier=hardware --serial 0024  # 75 tests, needs a real CRS
pytest test/pulse_capture/          # One subsystem
python -m rfmux.tools.periscope     # Launch Periscope
```

`--tier` (defined in the root `conftest.py`) names an invocation; every tier
but `hardware`/`all` excludes the board tests; on Linux with fastrx built and
the test group and the `dirfile` extra installed they report zero skips.
Markers tag tests: `portable`, `slow_acquisition`, `hardware`; the last is
applied automatically to anything using the `crs`/`live_session`/`serial`
fixtures, so don't add it by hand. A bare `pytest` runs the quick tier plus
~75 hardware skips: `addopts` in `pyproject.toml` deselects the acquisition
tests.
`full` contains `acquisition`; running both pays the slow set twice.

`test/` is organized into subdirectories mirroring the package under test
(`core/`, `streamer/`, `mock/`, `algorithms/`, `periscope/`, `notebooks/`),
plus `pulse_capture/` for the subsystem that spans algorithms and Periscope.
See `test/README.md` for the layout, markers, and Qt/notebook conventions.

Mock mode: `periscope MOCK`, or the mock connection in the startup dialog

## Common Tasks

### Add a new Periscope panel
1. Create `rfmux/tools/periscope/my_panel.py`
2. Inherit from `QWidget, ScreenshotMixin`
3. Implement `_setup_ui()`, `apply_theme()`
4. Add menu entry in `app.py`
5. Register with dock manager

### Add a new CRS algorithm
1. Create function in `rfmux/algorithms/measurement/`
2. Decorate with `@macro(CRS, register=True)`
3. Add to `__all__` in `algorithms/__init__.py`

### Debug pulse capture
- `pytest test/pulse_capture/` (assertions) or
  `pytest --tier=acquisition` (end to end, real UDP)
- Check `threshold_sigma` (default 5.0; 3-5 typical)
- Verify noise estimation with `estimate_noise_stats()`
