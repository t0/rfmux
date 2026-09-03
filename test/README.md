# rfmux test suite

## Running

Ask for a tier by name. Counts and times are from a warm checkout on a
developer laptop.

| Command | Runs | Time | Use when |
| --- | --- | --- | --- |
| `pytest --tier=portable` | 37 | ~9 s | Changing packaging, dependencies, or the Python floor. This is what `tox` runs on 3.10-3.12. |
| `pytest --tier=quick` | 626 | ~1 min | Default while editing. |
| `pytest --tier=acquisition` | 16 | ~3 min | After changing streaming, decimation, the PFB path, or pulse capture. A subset of `full`: run one or the other, not both. |
| `pytest --tier=full` | 642 | ~4 min | Before pushing. Everything that runs without a board, the acquisition tier included. |
| `pytest --tier=hardware --serial 0024` | 75 | needs a board | Against a connected board; see *Hardware tests*. |
| `pytest --tier=all --serial 0024` | 717 | needs a board | Before a release. |

```bash
pytest test/pulse_capture/         # one subsystem
pytest --tier=quick -k baseline    # narrow within a tier
pip install --group test           # nbclient, nbformat, pytest_check
```

`--tier` is shorthand for a marker expression. Passing both `--tier` and `-m` is
an error rather than one overriding the other.

**Without the test group** the notebook and mock-vs-real tests skip rather than
fail, so a CI runner missing it stops covering them without going red.

## What each tier covers

- **portable:** hardware-map YAML/CSV parsing, schema validation, session
  threading. Runs on a bare install: no PyQt6, no board. This is the subset
  `tox` runs on Python 3.10, 3.11 and 3.12, which is what keeps
  `requires-python = ">=3.10"` honest. The floor is 3.10 because
  `rfmux/core/crs.py` uses `match`.
- **quick:** packet decode, mock config plumbing, TLS noise, JIT dispatch,
  Periscope panels and dialogs, pulse detection and analysis. Fast because it
  never spawns a server: Qt runs offscreen, detection runs on synthetic arrays.
- **acquisition:** a MockCRS server subprocess streaming UDP over loopback.
  Covers what no unit test can: streamer config taking effect, the slow
  (~38 kHz) and PFB (~2.44 MHz) sources feeding a session, decimation
  constraints, and end-to-end pulse capture in slow, fast and both modes.
- **hardware:** the same API against a real CRS. Skipped unless you pass
  `--serial`.

A test marked `portable` that needs PyQt6 or a board does not fail there, it
*skips*, so the matrix goes green having tested nothing.

## Hardware tests

`--tier=hardware --serial 0024` needs **a board on the network and nothing
else**: no cryostat, no detectors, nothing connected to the RF ports. It checks
plumbing, not physics: NCO round-trips (64 of the 75 tests, eight modules by
eight frequencies), decimation round-trip, sample array shapes, sequence
continuity, and packet reception. No assertion touches signal content.

**It is not read-only.** The tests write NCO frequencies across all eight
modules, and the `crs` fixture resets the analog bank and forces decimation to
stage 6 long packets on teardown. Don't point it at a board mid-experiment.

- `test_high_sampling_rate` is the one that fails for environmental reasons. It
  streams at stage 0 (~38 kHz) and demands zero sequence gaps across 1000
  samples, so a slow machine or an undersized UDP buffer fails it and the
  failure reads as a code bug. Set `net.core.rmem_max` first.
- `test/mock/test_mock_vs_real.py --serial <n>` compares every mock attribute
  and signature against a real board, and is the only check that catches the two
  disagreeing. Run it after changing the mock's API, or when board firmware
  changes. It only reads attributes, so it is safe against a board in use.
- The measurement algorithms are **not** covered here, or anywhere against
  real hardware: network analysis, multisweep, fitting and `bias_kids` all run
  against the mock only. `test_mock_vs_real` is what keeps that arbiter
  trustworthy.

## One acquisition run at a time

The mock binds fixed ports 9876/9877, and `SO_REUSEPORT` lets a second run bind
them too. Neither outcome raises: on the unicast loopback fallback one reader is
starved, and where multicast works both readers receive both simulations
interleaved. Either way the symptom is a pulse-detection failure that looks like
a detector bug. A session guard in `conftest.py` refuses to start if the ports
are held.

## Markers

Declared in `pyproject.toml`; the default run applies
`-m "not slow_acquisition"`.

- `portable`: imports on a bare install (no board, no PyQt6).
- `slow_acquisition`: spawns a MockCRS server and streams UDP.
- `hardware`: needs a board. **Applied automatically** to any test whose
  fixtures include `live_session`, `crs` or `serial`. Don't add it by hand.

Mark at the narrowest scope that is true. Own sockets and threads from a
fixture rather than the success path: cleanup after the last assertion is
skipped exactly when something has already gone wrong.

## Layout

Directories mirror the package under test.

| Directory | Covers |
| --- | --- |
| `core/` | `rfmux/core/` — API surface, schema, threading |
| `streamer/` | `rfmux/streamer/` — packet decode, the batched getters, port conflict probes |
| `mock/` | `rfmux/mock/` — simulator fidelity, config plumbing, TLS noise, JIT dispatch, multicast selection |
| `algorithms/` | `rfmux/algorithms/measurement/` — measurement flows, streamer config |
| `periscope/` | `rfmux/tools/periscope/` — panels, dialogs, receiver, shutdown |
| `pulse_capture/` | `rfmux/pulse_capture/` — detection, session, ingest, HDF5, plus its Periscope panel and task |
| `notebooks/` | Jupyter-based tests |

## Notebook tests

Two kinds, both executed as tests rather than checked in as `.ipynb`:

- `test/notebooks/test_*.md`: quantitative checks written as notebooks.
- `rfmux/reference-notebooks/Demos/*.md`: the user-facing demos, executed in
  the acquisition tier so they cannot rot.

Both are jupytext markdown; edit them in JupyterLab or as text. A demo that
writes output must write to a temp directory, since the reference copies are
provisioned read-only. The `.py` scripts beside the demos are not executed by
any test; run one against `MOCK` by hand when its notebook changes.

## Platform skips

A handful of tests skip on macOS or Windows because they pin behaviour that only
exists elsewhere: `recvmmsg` blocking on a silent socket (Linux), `SO_REUSEPORT`
(absent on Windows), and `SIGINT` (Windows delivers Ctrl+C as a `CTRL_C_EVENT`
to a process group). On Linux every tier below `hardware` reports zero skips.

## CI

`.github/workflows/periscope-tests.yml` runs the quick tier and the acquisition
tier on ubuntu, windows and macos, with `fail-fast: false`. Between them that is
every test that does not need a board.

It triggers on push and pull request against `main`, plus `workflow_dispatch`. A
long-lived branch gets **no CI until it opens a PR**, so run the tiers locally
or dispatch the workflow against the branch by hand.
