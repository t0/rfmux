# rfmux test suite

## Running

Ask for a tier by name. Counts and times are from a warm checkout on a
developer laptop.

| Command | Runs | Time | Use when |
| --- | --- | --- | --- |
| `pytest --tier=portable` | 38 | ~9 s | Sanity check on an unfamiliar Python. No CRS, no GUI. |
| `pytest --tier=quick` | 462 | ~50 s | The normal edit/run loop. |
| `pytest --tier=acquisition` | 13 | ~5 min | You touched streaming, decimation, the PFB path, or pulse capture. |
| `pytest --tier=full` | 475 | ~6 min | Everything runnable without a board. Before pushing. |
| `pytest --tier=hardware --serial 0024` | 75 | needs a board | A CRS is in front of you. |
| `pytest --tier=all --serial 0024` | 550 | needs a board | Before a release. |

Thirteen acquisition tests take five minutes because each spawns a MockCRS
server and streams UDP in real time — the cost is wall clock, not CPU, and
`full` is dominated by it.

`--tier` is shorthand for a marker expression. Use `-m` directly for anything
the tiers don't cover; passing both is an error rather than one silently
winning.

```bash
pytest test/pulse_capture/         # one subsystem
pytest --tier=quick -k baseline    # narrow within a tier
pip install --group test           # nbclient, nbformat, pytest_check
```

Without the test group, collection *aborts* rather than skipping: pytest
imports every module before deselection applies, so one absent import costs
the whole run.

## What each tier covers

**portable** — hardware-map YAML/CSV parsing, schema validation, session
threading. No I/O, which is why tox can run it on Python 3.9–3.12.

**quick** — the bulk: packet decode, mock config plumbing, TLS noise, JIT
dispatch, Periscope panels and dialogs, pulse detection and analysis. Fast
because it never spawns a server — Qt runs offscreen, detection is driven by
synthetic arrays.

**acquisition** — the real data path: a MockCRS server subprocess streaming UDP
over loopback. Covers what no unit test can — streamer config actually taking
effect, the slow (~38 kHz) and PFB (~1.22 MHz) sources feeding a session,
decimation constraints, and end-to-end pulse capture in slow, fast and both
modes.

**hardware** — the same API against a real CRS, plus mock-vs-real attribute and
signature comparison. Skipped unless you pass `--serial`.

Run acquisition tiers **one at a time**: the mock binds fixed ports 9876/9877,
and a second reader silently takes the whole stream. A session guard in
`conftest.py` refuses to start if the ports are held.

## Markers

Declared in `pyproject.toml`; the default run applies `-m "not
slow_acquisition"`.

- `portable` — needs no CRS and no GUI.
- `slow_acquisition` — spawns a MockCRS server and streams UDP.
- `hardware` — needs a board. **Applied automatically** to any test whose
  fixtures include `live_session`, `crs` or `serial`. Don't add it by hand.

Mark at the narrowest scope that is true, and own sockets and threads from a
fixture rather than the success path — cleanup after the last assertion is
skipped exactly when something has already gone wrong.

## Layout

Directories mirror the package under test.

| Directory | Covers |
| --- | --- |
| `core/` | `rfmux/core/` — API surface, schema, threading |
| `streamer/` | `rfmux/streamer/` — packet decode, port conflict probes |
| `mock/` | `rfmux/mock/` — simulator fidelity, config plumbing, TLS noise, JIT dispatch, multicast selection |
| `algorithms/` | `rfmux/algorithms/measurement/` — measurement flows, streamer config |
| `periscope/` | `rfmux/tools/periscope/` — panels, dialogs, receiver, shutdown |
| `pulse_capture/` | `rfmux/pulse_capture/` — detection, session, ingest, HDF5 — plus its Periscope panel and task |
| `notebooks/` | Jupyter-based tests (below) |

## Notebook tests

Two kinds, both executed as tests rather than checked in as `.ipynb`:

- `test/notebooks/test_*.md` — quantitative checks written as notebooks.
- `rfmux/reference-notebooks/Demos/*.md` — the user-facing demos, executed in
  the acquisition tier so they cannot rot.

They are jupytext markdown. Edit them in JupyterLab, or as text. A demo that
writes output must write to a temp directory — the reference copies are
provisioned read-only.

## Platform skips

A handful of tests skip on macOS or Windows because they pin behaviour that
only exists elsewhere: `recvmmsg` blocking on a silent socket (Linux),
`SO_REUSEPORT` (absent on Windows), and `SIGINT` (Windows delivers Ctrl+C as a
`CTRL_C_EVENT` to a process group). On Linux every tier below `hardware`
reports zero skips.

## CI

`.github/workflows/periscope-tests.yml` runs the quick tier, the acquisition
tier, and `test_periscope_flow.py` on ubuntu, windows and macos, with
`fail-fast: false`.

It triggers on push and pull request against `main`, plus `workflow_dispatch`.
A long-lived branch gets **no CI until it opens a PR** — run the tiers locally,
or dispatch the workflow against the branch by hand.
