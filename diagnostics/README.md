# Diagnostics

Plot-generating, eyeball-it scripts. These are **not** tests: they print
tables and save PNGs instead of asserting, and nothing here runs in CI.

They are named `diag_*.py` on purpose — pytest's default `python_files` is
`test_*.py`, so this directory can never be collected by accident. (It used to
be `test_debug_scripts/`, where ~90 scripts named `test_*.py` sat one stray
`pytest .` away from being imported.)

Run them from anywhere; each resolves the repo root from `__file__`:

```bash
python diagnostics/diag_trigger_capture_e2e.py
python diagnostics/diag_pfb_timestream.py
python diagnostics/diag_tls_baseline_sweep.py --quick
```

Output (`*.png`, `*.h5`) lands next to the script and is gitignored.

| Script | What it shows |
| --- | --- |
| `diag_trigger_capture_e2e.py` | Drives `trigger_capture` against a MockCRS in slow, fast (PFB), and both modes; plots every captured pulse as its own subplot with threshold bands and SNR, and round-trips the results through HDF5. |
| `diag_pfb_timestream.py` | Raw slow and PFB I/Q timestreams via `py_run_pfb_streamer` / `py_get_samples`, bypassing pulse detection entirely — use it to tell a physics/streamer problem apart from a detection problem. |
| `diag_tls_baseline_sweep.py` | Sweeps TLS 1/f drift amplitude against the rolling-median baseline window, reporting detection efficiency, false triggers, and stuck fraction per cell. Synthetic signal, no mock server, so it is fast. |

## Adding one

Keep the `diag_*.py` name and make the script self-contained: generate your own
data from the mock or synthetically rather than reading a pickle from some
earlier session. Two previous scripts (`debug_bifurcation_detector.py`,
`analyze_multisweep_fits.py`) were dropped precisely because they hardcoded
`multisweep_updown_v2.pkl`, a file that no longer exists.

If a diagnostic pins down an invariant worth defending, port it into `test/` as
a real assertion instead of leaving it here.
