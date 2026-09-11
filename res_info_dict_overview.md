# `res_info_dict` — the rfmux resonator registry

An overview of what `res_info_dict` contains and how it is passed from measurement
to measurement in the Periscope GUI (resonator finding → multisweep → fitting →
bias optimization → bias application → noise/df display → save/reload).

Paths below are relative to `/home/maclean/code/rfmux`.

---

## What it is

`res_info_dict` is the **resonator registry**: one small dict per detector, keyed
by a random unique 4-letter uppercase code (e.g. `"AXQR"`), generated in
`_generate_unique_detector_ids` (`rfmux/algorithms/measurement/multisweep.py:33`).

The code *is* the detector's name — it's the bookkeeping handle that ties a
physical resonator across every sweep, fit, bias, and file save.

The registry is deliberately **lightweight and canonical**: it holds only the
current state of each detector, never sweep data. The heavy data lives in a
parallel dict keyed by the same codes:

* `multisweep_data_dict` on the algorithm side
* `MultisweepPanel.results_by_detector` in the GUI
  (`{code: {iteration_index: entry_dict}}`)

---

## Fields, and who writes them

| Field | Written by | Meaning |
|---|---|---|
| `bias_frequency` | `multisweep` at creation (`multisweep.py:238`); refined by `find_bias_points` (`bias_kids.py:824`); quantized by `apply_bias` (`bias_kids.py:921`) | The canonical tone frequency. Before Find Bias it is just the sweep centre; after, it is the max-IQ-derivative point; after Apply Bias it is the actual hardware-quantized value (`_BASE_FREQ_HZ ≈ 298.023 mHz` grid) |
| `bias_amplitude` | same three | Normalized DAC amplitude (0–1) selected as the operating point |
| `channel_number` | `multisweep` only, 1-based, assigned in creation order | **Permanent** hardware channel binding; never rewritten |
| `bias_found` | `find_bias_points` / custom-bias path | Gate flag — everything downstream (Apply Bias enablement, plot overlays, hardware programming, file re-loading) keys off it |
| `dI_df`, `dQ_df` | `find_bias_points` | IQ slopes at the bias point (V/Hz) |
| `df_calibration` | `find_bias_points` | `1/(dI_df + j·dQ_df)` (Hz/V) — the *sole* source for Periscope's df display mode |
| `bifurcated_at` | `find_bias_points` | Amplitude at which bifurcation was first detected, else `None` |
| `nonlinear_fit_params`, `nonlinear_fit_success` | `find_bias_points` | Diagnostic fit on the selected amplitude |
| `iq_rotation_angle` | `_iq_rotation_completed` after `ComputeIQRotationTask` (`multisweep_panel.py:3281`) | Radians; maximizes variance in Q |

### What is *not* in the registry

Per-sweep quantities live in `results_by_detector[code][iteration]`:

* snapshots taken at sweep time — `bias_frequency`, `bias_amplitude`,
  `channel_number` (`multisweep.py:451`)
* what this particular trace actually used — `sweep_center_frequency`,
  `sweep_amplitude_normalized`, `sweep_amplitude_dbm`, `sweep_direction`
* the data — `frequencies`, `iq_counts`, `iq_volts`, `phase_degrees`
* GUI-injected extras — `iteration`, `sweep_power_dbm`, `is_bifurcated`
* `fits` → `{'skewed': …, 'nonlinear': …}` written by `RunFitsTask`
* `bias_finding` → all Find Bias diagnostics (arc-length speed arrays, spike
  thresholds, peak indices, and the settings used), attached only to the
  selected-amplitude entry (`bias_kids.py:779`)

So a saved file records both "what the detector was" and "what this trace
measured".

---

## How it moves between measurements

### 1. Resonator finding

The netanal panel runs `fitting.find_resonances` and stores a plain frequency
list (`rfmux/tools/periscope/network_analysis_panel.py:861`). No registry exists
yet — resonators are still anonymous.

### 2. First multisweep (Option A)

The dialog produces `sweep_center_frequencies` + `amp_arrays`; `MultisweepTask`
calls `crs.multisweep` with explicit `center_frequencies`
(`rfmux/tools/periscope/tasks.py:565`). The algorithm mints codes, assigns
channels 1..N, and returns a fresh registry.

### 3. Subsequent iterations (Option B)

`MultisweepTask` captures the returned registry and re-feeds it on every
following amplitude/direction (`tasks.py:530`, `tasks.py:553`) — this is what
keeps codes and channel numbers stable across an entire multi-amplitude run.

In Option B, `multisweep` reads `bias_frequency` / `bias_amplitude` /
`channel_number` back out and **does not modify** the dict (`multisweep.py:185`).
For multi-module runs all modules share one registry (codes and channels are
module-agnostic); each module gets its own data dict.

### 4. Into the panel

Each iteration's `data_update` signal carries
`(multisweep_data_dict, res_info_dict)`; the panel replaces its live registry
wholesale and files sweep data under `results_by_detector[code][iteration]`
(`multisweep_panel.py:1226`).

### 5. Find Bias

`FindBiasTask` deep-copies the registry, runs `find_bias_points` over
`results_by_detector`, and emits the result; the panel **merges field-by-field**
back into the live dict rather than replacing it (`multisweep_panel.py:2955`).
Codes present in the data but not in the registry are skipped with a warning
(`bias_kids.py:558`). Completion auto-saves the pickle.

The algorithm itself: sorts entries by amplitude, steps up until bifurcation is
detected, selects the amplitude one step below, then locates `bias_frequency` by
maximizing the IQ arc-length speed (or its log) and computes `df_calibration`
there.

### 6. Apply Bias

`apply_bias` walks only `bias_found == True` entries and programs
`(channel_number, bias_frequency − NCO, bias_amplitude, phase 0)` in one tuber
batch, writing the quantized frequency back into the registry so it stays
consistent with hardware (`bias_kids.py:908`).

The panel then re-keys `df_calibration` **from code to `channel_number`** and
emits it to Periscope for df units (`multisweep_panel.py:3210`), and auto-kicks
`ComputeIQRotationTask`, which writes `iq_rotation_angle` back per code.

### 7. Fitting

`RunFitsTask` writes results into `results_by_detector[code][iter]['fits']`,
never into the registry — but it *reads* the registry when fit mode is
"bias amplitude", matching each detector's `bias_amplitude` to the iteration
whose `sweep_amplitude_normalized` equals it (`tasks.py:1147`).

### 8. Persistence

`_prepare_export_data` writes `res_info_dict` as a top-level pickle key
(`multisweep_panel.py:1885`). On load the registry is restored verbatim, Apply
Bias is re-enabled if any `bias_found`, `df_calibration` is re-derived per
channel, and rotation angles are re-published
(`app_runtime.py:1385`–`1466`). Legacy files without a registry (top-level
`bias_kids_output`, integer detector keys) are handled by fallback branches.

### 9. Re-run

The panel seeds the dialog with registry frequencies sorted by `channel_number`,
clears only `bias_found` (keeping codes, channels, and `bias_amplitude` so
Option B still knows what amplitude to sweep at), and re-injects the registry
(`multisweep_panel.py:2074`–`2091`).

This is what makes bias frequencies chain: each Find Bias refines the value the
next sweep centres on. The dialog also uses `bias_amplitude` as the base for
"multiplicative scaling" amplitude arrays (`multisweep_dialog.py:1504`).

### 10. Custom bias / Bias KIDs

The in-panel custom path bypasses the algorithm: N user pairs are matched to the
N existing resonators **in ascending frequency order**, overwriting only
`bias_frequency` / `bias_amplitude` / `bias_found` and preserving codes,
channels, and all other calibration data (`multisweep_panel.py:3100`).

The standalone Bias KIDs dialog reads `bias_found` entries out of a pickle
(sorted by `channel_number`) to prefill frequencies/amplitudes and to recover the
channel assignment (`app.py:1587`). With CSV or manual entry it just assigns
channels 1..N and no registry is involved.

### 11. Display

* Grid helpers get the registry only when "Show Bias Info" is checked, and read
  `bias_amplitude` / `bias_frequency` gated on `bias_found` to highlight the
  chosen trace and draw the bias line (`multisweep_grid_helpers.py:171`).
* The noise-spectrum panel builds a `{channel_number: bias_frequency}` map from
  it (`multisweep_panel.py:2448`).
* Rotated-IQ plotting pulls `iq_rotation_angle` per code
  (`multisweep_panel.py:1429`).

---

## Design notes and gotchas

* **Identity is only as durable as the code.** Any path that goes through
  Option A mints new codes. The clearest case is a re-run with custom
  frequencies, which deliberately drops the registry
  (`multisweep_panel.py:2090`) — the detectors survive physically but lose their
  names and all accumulated calibration.

* **`channel_number` is the interop key.** Codes never leave the multisweep
  panel: `df_calibration` and hardware programming are re-keyed to channel
  number at the boundary. The registry is the only place the two namespaces are
  joined.

* **Two dicts, one key space.** The registry is "current truth, one row per
  detector"; `results_by_detector` is "history, one row per detector per sweep,
  with a snapshot of truth at that moment". Most fragile lookups in the GUI are
  fallback chains between the two (`bias_frequency` → `sweep_center_frequency` →
  legacy `original_center_frequency`).

* **Known inconsistency:** `iq_rotation_ready` publishes angles keyed by
  **detector code**, and `Periscope.iq_rotation_calibrations` stores them that
  way (`app.py:1803`), but `_convert_iq_data` looks them up by **channel
  number**, with a fallback loop guarded on `isinstance(code, int)` that can
  never match a 4-letter string (`app_runtime.py:58`). Rotated-IQ mode therefore
  silently falls through to plain volts. `df_calibration` avoids this because
  the panel re-keys it to `channel_number` before emitting; the rotation path
  does not.

---

## Scripted equivalent

`rfmux/reference-notebooks/Demos/simplified_tuning_flow.py` shows the same flow
without the GUI:

```python
res_info_dict, multisweep_results = await crs.multisweep(**MULTISWEEP_PARAMS)
find_bias_points(results_by_detector, res_info_dict)      # updates in place
await apply_bias(crs, module, res_info_dict=res_info_dict)
ch = res_info_dict[code]['channel_number']                 # 1-based
```
