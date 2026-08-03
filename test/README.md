# rfmux test suite

Run everything from the repo root. `pyproject.toml` sets `testpaths = ["test"]`,
so a bare `pytest` finds this directory.

## Which command do I want?

Ask for a tier by name. Times are wall clock from a warm checkout on a
developer laptop; treat them as orders of magnitude.

| Command | Runs | Time | Use when |
| --- | --- | --- | --- |
| `pytest --tier=portable` | 9 | **~6 s** | Sanity check on an unfamiliar Python. No CRS, no GUI, minimal deps. |
| `pytest --tier=quick` | 224 | **~20 s** | The normal edit/run loop. |
| `pytest --tier=acquisition` | 12 | **~1 min** | You touched streaming, decimation, the PFB path, or pulse capture. |
| `pytest --tier=full` | 236 | **~1 min 45 s** | Everything runnable without a board. Run this before pushing. |
| `pytest --tier=hardware --serial 0024` | 75 | needs a board | You have a CRS in front of you. |
| `pytest --tier=all --serial 0024` | 311 | needs a board | Belt and braces before a release. |

Every tier except `hardware` and `all` excludes the board tests, so all of the
above report **zero skips** — a bare pass/fail, rather than a result buried
under ~75 "no `--serial`" skips.

`--tier` is a shorthand for a marker expression, nothing more; `pytest --help`
lists what each one expands to. Reach for `-m` directly when you want something
the tiers don't cover:

```bash
pytest test/pulse_capture/         # just one subsystem
pytest --tier=quick -k baseline    # narrow within a tier
pytest -m "portable or hardware"   # an expression no tier covers
```

Passing both `--tier` and `-m` is an error rather than one silently winning.

A bare `pytest` with no arguments still behaves as it always has — the `quick`
tier plus the hardware tests skipping, so 224 passed and ~75 skipped. That is
normal, not a problem; use `--tier=quick` when the skip count is drowning out
the signal.

`./test.sh` is a separate thing — it drives tox to run the `portable` tier
across every supported Python version. Use it when changing packaging or
touching version-sensitive code, not in the edit loop.

## What each tier actually covers

**`portable`** — hardware-map YAML/CSV parsing, schema validation, and session
threading. Pure library behavior with no I/O, which is why it is the subset tox
can run on Python 3.9 through 3.12.

**default** (everything not `slow_acquisition`) — the bulk of the suite: packet
decode, mock config plumbing, TLS 1/f noise, JIT dispatch, all the Periscope
panels and dialogs, and the pulse detection/analysis logic. Fast because it
never spawns a server: Qt runs offscreen and detection is driven by synthetic
arrays and `AsyncMock`.

**`slow_acquisition`** — the data-acquisition path for real, which is where the
minute goes. These spawn a MockCRS server subprocess and stream UDP over
loopback, so they cover what no unit test can: streamer configuration actually
taking effect, the slow (~38 kHz) and PFB (~1.22 MHz) sources actually feeding a
session, decimation stage constraints, carrier-scale parity between the two
streams, and end-to-end pulse capture in slow, fast, and both modes.

**`hardware`** — the same API exercised against a real CRS, plus mock-vs-real
attribute and signature comparison. Skipped unless you pass `--serial`.

The QC suite is not here at all: it lives in `rfmux/tools/qc/`, carries the
`qc_stage1`/`qc_stage2` markers, and runs via `rfmux qc`.

## Markers

Markers tag *tests*; `--tier` names *invocations*. The tiers above are defined
in `conftest.py` as marker expressions over these:

Declared in `pyproject.toml`. The default run applies
`-m "not slow_acquisition"` via `addopts`, so the acquisition tier is opt-in —
passing your own `-m` on the command line replaces that filter entirely.

- `portable` — needs no CRS and no GUI.
- `slow_acquisition` — slow because it spawns a MockCRS server and streams UDP.
- `hardware` — needs a real board. **Applied automatically**: `conftest.py`
  marks any test whose fixture closure includes `live_session`, `crs`, or
  `serial`. Do not add it by hand; request the fixture and the marker follows.
  This exists so the hardware tier is addressable — the tests are gated by a
  skip *inside* those fixtures, which otherwise leaves nothing to write
  after `-m`.

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

`conftest.py` stays at this level: it registers `--serial`, the `live_session` /
`crs` fixtures, and the automatic `hardware` marking, for every directory.

Mark at the narrowest scope that is true. A module-level
`pytestmark = pytest.mark.slow_acquisition` on a file where only one class
spawns a server exiles the fast tests in that file for a cost they never incur —
and because `pytestmark` gates *selection* and not *import*, the suite still
pays their setup on every default run. Put it on the class.

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

Glob the notebooks relative to `__file__`, never the working directory. A
CWD-relative glob silently yields an empty parameter set from the repo root,
which reports as one skip rather than an error — this suite's notebook test
went unrun in CI that way.

## Diagnostics are not tests

`diagnostics/` at the repo root holds plot-generating, eyeball-it scripts —
they print and save PNGs rather than assert. They are named `diag_*.py` so
pytest never collects them, and they are run by hand:

```bash
python diagnostics/diag_trigger_capture_e2e.py
```

If a diagnostic establishes an invariant worth defending, port it into a real
test here rather than leaving it as a script.
