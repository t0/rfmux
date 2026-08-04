# rfmux test suite

**Run from the repo root.** `pyproject.toml` sets `testpaths = ["test"]`, which
pytest only consults when you give it no path — and only relative to the rootdir
when the rootdir is where you are. From a subdirectory, pytest treats the
current directory as the target instead, so `cd rfmux && pytest` quietly
collects the QC suite (which then demands `--serial`) rather than running
anything here. That is pytest's normal behaviour, not a misconfiguration; it
just means "run from the root" is load-bearing advice.

Command-line options (`--tier`, `--serial`) are declared in the **root**
`conftest.py` rather than this directory's, because pytest only honours
`pytest_addoption` from a conftest it loads *before* parsing arguments. Declared
down here they vanished — `pytest --tier=quick` from a subdirectory failed with
"unrecognized arguments" instead of running.

## Which command do I want?

Ask for a tier by name. Times are wall clock from a warm checkout on a
developer laptop; treat them as orders of magnitude.

| Command | Runs | Time | Use when |
| --- | --- | --- | --- |
| `pytest --tier=portable` | 9 | **~6 s** | Sanity check on an unfamiliar Python. No CRS, no GUI, minimal deps. |
| `pytest --tier=quick` | 229 | **~20 s** | The normal edit/run loop. |
| `pytest --tier=acquisition` | 12 | **~1 min** | You touched streaming, decimation, the PFB path, or pulse capture. |
| `pytest --tier=full` | 241 | **~1 min 40 s** | Everything runnable without a board. Run this before pushing. |
| `pytest --tier=hardware --serial 0024` | 75 | needs a board | You have a CRS in front of you. |
| `pytest --tier=all --serial 0024` | 316 | needs a board | Belt and braces before a release. |

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
tier plus the hardware tests skipping, so 229 passed and ~75 skipped. That is
normal, not a problem; use `--tier=quick` when the skip count is drowning out
the signal.

`./test.sh` is a separate thing — it drives tox to run the `portable` tier
across every supported Python version. Use it when changing packaging or
touching version-sensitive code, not in the edit loop.

### One acquisition run at a time

The mock streamer binds fixed ports — 9876 slow, 9877 PFB — and
`get_multicast_socket()` sets `SO_REUSEPORT`. A second listener therefore does
*not* collide: it joins the same multicast group and the two readers split the
packet stream. Neither errors, both see partial data, and the tests report it as
a pulse-detector fault (`no matched pairs`, thousands of phantom pulses,
nonsensical elapsed times) with nothing pointing at the real cause. Two
investigations on this branch were lost to that misdirection.

You should not be able to hit it silently any more. Three things changed:

- **The tier refuses to start** if 9876/9877 are already held. The guard lives in
  `conftest.py` here and prints the `ss` command to find the holder. Override
  with `--allow-busy-streamer-ports` if you genuinely want two readers.
- **A failing test no longer leaks its reader.** `test_pulse_capture_fast.py`
  used to stop its tap thread and capture task only on the success path, so a
  failed assertion left them holding both ports for the rest of the process — one
  leaked run was measured still holding them 419 s later. The `stream_guard`
  fixture now tears them down unconditionally.
- **Exiting is fast.** `rfmux/mock/server.py` used to register one `atexit`
  handler per mock session, each with its own `terminate()`/`join(2.0)` grace
  period, so ~20 sessions serialised into a minute-plus of teardown *after* the
  summary line — all of it with the sockets still open. One hook now terminates
  the fleet and joins against a single deadline.

If an acquisition failure still looks like a detector bug, check for a stranded
reader before believing it:

```bash
ss -ulnp | grep -E '9876|9877'               # must be empty
ps -eo pid,etimes,cmd | grep '[p]ytest'      # a finished run can still be dying
```

A clean solo `--tier=full` is 241 passed in ~1 min 40 s. Re-run solo with the
ports verified free before drawing any conclusion about the code.

### Rate-invariance caveat in the rolling baseline

Not a failure today, but worth knowing when reading pulse-capture results.
`PulseCaptureConfig.buf_size` floors the ring at `_MIN_BUF = 1000` samples, and
`baseline_window_samples` returns `BASELINE_MIN_RINGS (8) * buf_size`. That
floor stops scaling down with the sample rate, so the baseline window in *time*
is:

| stream | rate | baseline window |
| --- | --- | --- |
| slow, dec=6 | 596 Hz | **13.42 s** |
| slow, dec=1 | 19 kHz | 0.42 s |
| slow, dec=0 | 38 kHz | 0.24 s |
| fast (PFB) | 1.22 MHz | 0.24 s |

Every rate wants ~0.24 s except dec=6, which demands 56× more, because 8000
samples is a long time at 596 Hz. Until that much data has accumulated the
median baseline has nothing solid to sit on, which shortens the margin on the
slow stream at high decimation stages more than the numbers suggest.

When reading a decimation stage back, test it against `None`, not falsiness.
**Stage 0 is a valid decimation** — the ~38 kHz rate `test_streamer_config`
exercises — so the tempting `dec = await crs.get_decimation() or 6` rewrites
stage 0 to stage 6 and derives a sample rate 64× off from the stream's. Write
the explicit check, as `trigger_capture.py` does:

```python
dec = await crs.get_decimation()
if dec is None:      # genuinely unset; not the same as stage 0
    dec = 6
```

`get_decimation()` really does return `None` before any `set_decimation` call
(see `test/core/test_spotcheck.py`), so the fallback itself is needed — it just
has to distinguish "unset" from "stage 0".

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

`conftest.py` stays at this level for the `live_session` / `crs` fixtures and
the automatic `hardware` marking. The `--serial` and `--tier` *options* live in
the root `conftest.py` — see the note at the top of this file.

Mark at the narrowest scope that is true. A module-level
`pytestmark = pytest.mark.slow_acquisition` on a file where only one class
spawns a server exiles the fast tests in that file for a cost they never incur —
and because `pytestmark` gates *selection* and not *import*, the suite still
pays their setup on every default run. Put it on the class.

Own your sockets and threads from a fixture, not from the success path. Anything
that binds a streamer port — a `PulseCaptureTask`, a thread running
`run_slow_source` — must be stopped in fixture teardown, so a failing assertion
cannot strand it. Cleaning up after the last assertion means the cleanup is
skipped exactly when something has already gone wrong, and a stranded reader
turns one local failure into corruption of everything that follows. See
`stream_guard` in `pulse_capture/test_pulse_capture_fast.py`.

## Qt tests

GUI tests set `QT_QPA_PLATFORM=offscreen` at import time and
`pytest.importorskip("PyQt6")`. They share one `QApplication` via a `qt_app`
fixture and call `_spin(qt_app)` after closing a panel, to let Qt drain
deferred deletions before the next assertion.

### Panels must not be strongly referenced by their own ViewBoxes

If panel tests start printing tracebacks ending in:

```
RuntimeError: wrapped C/C++ object of type QComboBox has been deleted
```

something has reintroduced a reference cycle between a panel and one of its
plots, and the suite is one `gc.collect()` away from a segfault rather than a
traceback.

Panels give their ViewBox a back-pointer so the double-click handler in
`utils.py` can reach the window:

```python
vb = ClickableViewBox()
vb.parent_window = self        # noise_spectrum, network_analysis,
                               # detector_digest, multisweep
```

Held *strongly*, that closes a cycle — ViewBox → panel → PlotWidget → ViewBox —
so tearing the panel down goes through Python's **cyclic** collector, which
finalizes PyQt objects in arbitrary order and frees C++ objects out from under
live wrappers. Building and dropping one panel in a bare script was enough to
segfault the interpreter; under pytest the same cycle usually collapses at a
survivable moment and only prints the traceback above.

`ClickableViewBox.parent_window` is therefore a **weakref-backed property**.
Reads are unchanged (`getattr(vb, 'parent_window', None)` still works and still
returns the window while it lives); it simply no longer keeps the panel alive.

`test/periscope/test_viewbox_lifetime.py` pins both halves: the weakref contract
directly, and the build/drop/`gc.collect()` loop that used to crash. That loop
runs in a **subprocess** on purpose — a regression is a SIGSEGV, which in-process
would take the whole pytest session down instead of failing one test.

Dead ends, recorded so nobody repeats them — none of these address the cycle:

- `_spin(qt_app)` after `close()` — Qt's event queue is not the trigger; Python's
  GC is.
- Holding panel references to prevent collection — made it *worse*.
- A session-scoped `pyqtgraph.cleanup()` fixture — too late; the collections
  happen mid-run.
- Dropping `name=` from the plot widgets — moved the segfault from the first
  panel to the second.
- Filtering `sys.excepthook` / `sys.unraisablehook` — the traceback comes from
  inside Qt's C++ slot invocation; replacing those hooks **core-dumps**. It would
  also have hidden a crash.

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
