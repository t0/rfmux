# rfmux test suite

Run everything from the repo root. `pyproject.toml` sets `testpaths = ["test"]`,
so a bare `pytest` finds this directory.

```bash
pytest                      # default tier, ~20 s
pytest -m mock_e2e test/    # heavy tier: spawns MockCRS servers, streams UDP
pytest test/pulse_capture/  # one subsystem
./test.sh                   # tox across Python versions (offline tests only)
```

## Layout

Directories mirror the package under test, so a new test goes where its module
lives. The one exception is `pulse_capture/`, whose subsystem deliberately
spans `rfmux/algorithms/measurement/` and `rfmux/tools/periscope/`.

| Directory | Covers |
| --- | --- |
| `core/` | `rfmux/core/` — API surface, schema, threading, hardware spotchecks |
| `streamer/` | `rfmux/streamer/` — packet decode |
| `mock/` | `rfmux/mock/` — simulator fidelity, config plumbing, TLS noise, JIT dispatch |
| `algorithms/` | `rfmux/algorithms/measurement/` — measurement flows, streamer config |
| `periscope/` | `rfmux/tools/periscope/` — panels, dialogs, fonts, embedded console |
| `pulse_capture/` | pulse detection + its Periscope panel and dialog |
| `notebooks/` | Jupyter-based quantitative tests (see below) |

`conftest.py` stays at this level: it registers `--serial` and the
`live_session` / `crs` fixtures for every directory.

## Tiers and markers

Markers are declared in `pyproject.toml`. The default run is
`-m "not mock_e2e"`, so the slow tier is opt-in.

- `mock_e2e` — spawns a MockCRS server and streams real UDP. Slow (~70 s),
  excluded by default, run explicitly in CI.
- `offline` — no hardware and no server; what `tox`/`test.sh` runs.
- `integration` — multi-component flows.
- `qc_stage1` / `qc_stage2` — the QC suite, which lives in `rfmux/tools/qc/`
  and runs via `rfmux qc`, not via this directory.

Tests that need real hardware take the `crs` fixture and skip unless you pass
a board serial:

```bash
pytest --serial 0024        # unskips the hardware tests
```

Roughly 75 tests are hardware-gated, so a large skip count in a normal run is
expected, not a problem.

## Qt tests

GUI tests set `QT_QPA_PLATFORM=offscreen` at import time and
`pytest.importorskip("PyQt6")`. They share one `QApplication` via a `qt_app`
fixture and call `_spin(qt_app)` after closing a panel — without that, Qt's
deferred deletion runs during interpreter teardown and pyqtgraph's ViewBox
cleanup writes tracebacks to stderr.

## Notebook tests

`notebooks/` executes JupyterLab notebooks so that quantitative checks can
carry plots and prose. Notebooks are stored as `.md` (jupytext) and only the
`.md` form belongs in version control; executed `.ipynb` results are written
next to them and are gitignored.

Requires the dev dependency group (`jupytext`, `nbclient`, `nbformat`):

```bash
pip install -e '.[dev]'   # or: uv sync --group dev
```

To edit a notebook:

```bash
jupytext -o test_py_get_samples.ipynb test_py_get_samples.md
# ...edit in JupyterLab...
jupytext -o test_py_get_samples.md test_py_get_samples.ipynb
```

## Diagnostics are not tests

`diagnostics/` at the repo root holds plot-generating, eyeball-it scripts —
they print and save PNGs rather than assert. They are named `diag_*.py` so
pytest never collects them, and they are run by hand:

```bash
python diagnostics/diag_trigger_capture_e2e.py
```

If a diagnostic establishes an invariant worth defending, port it into a real
test here rather than leaving it as a script.
