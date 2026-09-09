# Periscope onto the headless tuning flow: a roadmap

Working document, 2026-09-09. Companion to `tuning_refactor_design.md` (the
plan for the library, largely built), `tuning_revamp_todo.md` (deferred
work items), `res_info_dict_overview.md` (how the old GUI flow worked) and
`main_merge_conflict_survey.md` (the 2026-09-08 merge and its decisions).

Goal: Periscope calls `rfmux.tuning` and the two drivers the way a notebook
does, with very little tuning logic of its own, and the plotting and
usability reach the bar `mr_multisweep_section_amplitudes` set. This
document says what exists on each side, what has to go, and the order to do
it in. Every stage leaves Periscope working and tested.

---

## 1. Where things stand

### 1.1 The library is ready to be called

Everything the GUI needs for the basic flow exists and is tested
(`test/tuning`: 343 passed, 74 s; `test_flow_on_standard_array.py` runs the
whole sequence against the standard mock array over RPC, in the quick tier).

| Step | Call | Comes back as | Hooks a GUI wants |
|---|---|---|---|
| Network analysis | `crs.take_netanal(amp, fmin, fmax, npoints, nsamps, sweep_direction=, module=, progress_callback=, data_callback=, save=, label=)` | container, trace at `[module_id]["results"]` (descending when `sweep_direction="downward"`) | `data_callback(module, freqs, amps, phases)` for live plotting |
| Find resonances | `find_resonances_in_netanal(module_netanal, min_dip_depth_db=, min_Q=, max_Q=, min_separation_hz=, require_isolation=, expected_resonances=)` | `ResonanceSearch`: `candidates`, `rejected` (with `rejected_because`), `frequencies_hz`, `magnitude_db`; written into the netanal as `trace["resonance_search"]` | `candidate.index` into the trace, for markers |
| Seed the array | `search.to_catalog(module, amplitude, names=)` or `ResonatorCatalog.from_frequencies(...)` | `ResonatorCatalog` (module-scoped, named resonators, one `BiasPoint` each) | `copy()` for workers; `update_bias_point()`; `remove()`; `to_dict`/`from_dict`/`to_csv`/`from_csv` |
| Amplitude schedule | `AmplitudeSchedule()` / `(x)` / `.explicit()` / `.ramp()` / `.multiplicative()` | frozen dataclass | `.describe(catalog, n_directions, dac_scale_dbm)` and `.validate(...)` for a live dialog preview; `.resolve_steps()` |
| Multisweep | one `crs.multisweep(catalog, span_hz=, npoints_per_sweep=, nsamps=, amp=schedule, sweep_direction=(...), progress_callback=, data_callback=, sweep_callback=, save=, label=)` | container, `[module_id]["results"][step][direction][name]` with seven measurement keys | `sweep_callback(record)` once per sweep (`step, direction, amplitudes, factor, completed, total`); `data_callback(module, partial, step, direction)`; progress across the whole call |
| Read sweeps | `collect_amplitude_iterations_for`, `get_amplitudes_at_iteration`, `find_iteration_matching_amplitude` | plain dicts | the accessors the design doc's `SweepSet` was going to be |
| Fit | `fit_sweeps(module_sweeps, models=, progress_callback=(completed,total), ...)`, `fit_sweeps_at_bias_amplitude(...)` | `FitReport`; params written to `entry["fits"][model]` | `skewed_model_magnitude(entry)`, `nonlinear_model_iq(entry)`, `centered_iq`, `gain_corrected_iq` rebuild curves; `failed_because` per fit |
| Find bias | `find_bias_points(module_sweeps, amplitude_method=, frequency_method=, spike_prominence_factor=, noise_gate_factor=, max_discrepancy=, compare=, max_distance_hz=)` | `BiasReport`: a new `catalog`, `findings` with `checks` per step, `flagged`; written into the sweeps as `bias_report` | `BifurcationCheck.metric`/`threshold` per step for a diagnostics view; `iq_arc_speed`, `normalized_arc_speed` |
| Apply | `await crs.apply_bias(report.catalog)` | nothing; raises if it cannot | owns the NCO |
| Files | `store.save/load/maybe_save`, `set_output_directory`, `set_created_by` | pickle of builtins plus ndarrays, `file_metadata` inside each module block | analyses re-save in place |

Also relevant: `rfmux/mock/standard_array.py` gives tests a seeded array
that builds in under a second with no UDP, and
`Demos/example_plotting_{netanal,multisweep,fits,bias}.py` show, in
matplotlib, what every product looks like when plotted from the library
alone. They are the reference for what the pyqtgraph panels should show.

### 1.2 Periscope on this branch does not run the flow

This branch's Periscope is main's Periscope (11 lines differ), and main's
Periscope predates the library. It imports nothing from `rfmux.tuning`. Every
module imports cleanly; the breaks are at runtime:

* `MultisweepTask` passes `bias_frequency_method` and `rotate_saved_data` to
  `crs.multisweep` (`tasks.py:706`). `TypeError` before a sweep starts.
* `NetworkAnalysisTask` reads `result['frequencies']` off the top of what
  `take_netanal` returns (`tasks.py:479`). `KeyError`; the completion signal
  never fires, so a multi-amplitude netanal stops after its first amplitude
  and nothing is exported.
* Everything downstream reads `iq_complex`, `phase_degrees`, `bias_frequency`,
  `is_bifurcated`, `recalculation_method_applied`, `rotation_tod` off sweep
  entries. None of those exist any more. The multisweep dialog's Bias
  Frequency Method combo and Rotate Saved Data checkbox drive nothing.
* Find Resonances and Bias KIDs still work in signature, through the
  deprecated shims, and would flood the console with `DeprecationWarning`
  the moment they received data (Periscope runs as `__main__`).
* Nothing caught either break, because the only test of this path was a
  mocked smoke test whose data and scaffolding lived in shipped code:
  `test_dialog_params`, `_ui_mock_context` and `run_ui_mock_smoke_test` in
  `app_runtime.py`, 530 lines, injecting a `results_by_detector` keyed by
  `(0.1, "up")` with a literal `"some_data": [1, 2, 3]`, with every dialog,
  task and signal on the path a MagicMock and `assert True` at the end. It
  could not fail, and it made the old dict shape a contract of the package
  rather than of a test. Deleted in stage 0, with its test and the
  `unittest.mock` import from the GUI.

Beyond the breaks, this Periscope carries its own amplitude loop, its own
re-centring history (`_get_closest_remembered_cf`), inline fitting during the
sweep, model-curve generation and denormalisation in the digest, a
hand-rolled `apply_bias_output`, three different restructurings of the same
result dict, and a shared `MultisweepSignals` object that only the most
recent panel receives.

### 1.3 `mr_multisweep_section_amplitudes` sets the usability bar

That branch diverged from main (87 commits its own, 39 main's it lacks). Its
Periscope is much larger (`multisweep_panel.py` 3758 lines, `multisweep_dialog.py`
1612, `multisweep_grid_helpers.py` 1299) and its tuning logic lives almost
entirely in the GUI, which is what the library was built to replace. What it
does well, and what this port must reproduce:

**Display.** Seven tabs: Mag vs Freq grid, IQ Circles grid, Mag/Phase
overview, Histograms, Detector Digest, IQ Derivatives, Fit Results. Grid rule
`ncols = max(min(4, n), ceil(sqrt(n)))`; cached plot widgets; kHz offset axes
`f - f_central`; titles with name and centre frequency; amplitude colours
from TABLEAU10 up to three amplitudes and `inferno` above, with a shared
colorbar labelled in dBm and normalised units; direction as line style
(solid up, dotted down); stable colours across a live run; square IQ axes;
zoom box mode; legend entries carrying the amplitude and the fitted `a`;
bias overlays (bold chosen trace, bias-frequency line, cross on the IQ
loop, star in the legend); a Fit Results tab with measured and model curves
and three-line legends; an IQ Derivatives tab showing the arc-speed
diagnostics with the bias frequency marked; a digest with three plots and
two parameter tables; Q histograms on shared log bins.

**Ergonomics.** Non-modal, always-on-top settings panels for Find Resonances,
Fits and Bias, persisted through QSettings with Reset to Defaults; a live
validation label (green tick, amber warning, red cross) that disables Start
and explains why in the tooltip; Return starts the sweep from anywhere in the
dialog; batch navigation with a subplots-per-page spinbox; sort by frequency
or by name; measurement names with a custom suffix and a live filename
preview; the filename in the panel title; transient green status labels
instead of popups; mutual locking of Find Bias and Run Fit while one runs;
a "Find bias after sweep" checkbox; keyboard navigation in the digest
(left/right detector, up/down amplitude); double-click a subplot to open its
digest; non-blocking file dialogs; netanal magnitude normalised by sweep
power so the display is true dB; a "Load bias amplitudes" button that pulls
the found amplitudes into the next sweep.

**Not to be ported.** The dialog builds amplitude schedules and reads
`res_info_dict.values()` in dict order to do it (`AmplitudeSchedule` does
this, keyed by name). The task loops over amplitudes and directions itself
(`multisweep` does). `identify_bifurcation(threshold_factor=7)` runs during
the sweep and disagrees with the Find Bias detector (`bifurcated_by_either`
runs at bias time). Two near-duplicate fit orchestrators generate and store
model curves (`fit_sweeps` stores parameters; the readers rebuild curves).
Custom bias matches user pairs to resonators in frequency order (a catalog
edit). `apply_bias_output` programs tones without quantisation beside an
`apply_bias` that does (one `apply_bias` now). `find_resonances` runs on the
GUI thread. Several `asyncio.run` calls sit on the GUI thread in the noise
path. `ComputeIQRotationTask` publishes angles keyed by code and looks them
up by channel, so Rotated IQ silently falls through to volts.

### 1.4 Which Periscope to start from

Start from this branch's smaller Periscope and port the section-amplitudes
branch's display code and ergonomics into it, tab by tab, rewritten against
the container shape. The alternative, merging that branch's Periscope and
then gutting it, brings 10,000 lines whose plumbing is exactly what has to
go. The grid helpers, colorbar, digest, histograms and settings panels port
well because they are mostly rendering; the panel and task rewrite is where
the deletion happens.

---

## 2. Rules for the port

These follow from the design doc's §1 (the pulse-capture layering) and §9,
and from the merge decisions of 2026-09-08.

1. **The panel holds a `ResonatorCatalog` and one module's output block.**
   No `results_by_detector`, no `res_info_dict`, no integer detector indices,
   no `"amp:direction"` string keys. Every tab reads through the
   `sweep_results` accessors or walks `results[step][direction][name]`.
   Resonators are referred to by name everywhere the user sees them.
2. **A task runs one library call and re-emits its callbacks.** Shaped like
   `pulse_capture_task.py`: no loops over amplitudes, no fitting, no
   restructuring. `sweep_callback` and `data_callback` become signals;
   `progress_callback` is passed straight through.
3. **Dialogs are views over library arguments.** The multisweep dialog is a
   view over `AmplitudeSchedule` (its five constructors, `describe()` for the
   preview, `validate()` for the status label). The fit and bias settings
   panels expose exactly the keyword arguments of `fit_sweeps` and
   `find_bias_points`, with the library's defaults, and nothing the library
   does not accept.
4. **Analysis is a button, never a side effect of measuring.** Fits run on
   Run Fit (or the "fit after sweep" checkbox, which presses the button for
   you). Bias finding runs on Find Bias. The sweep is a measurement.
5. **Workers get `catalog.copy()`; the GUI swaps its reference on
   `completed`.** Find Bias produces a new catalog (`report.catalog`); the
   panel adopts it. There is no field-by-field merge.
6. **Periscope writes and reads through `store`.** `set_created_by("periscope")`
   at startup; the session folder is the output directory; a file Periscope
   writes opens in a notebook and vice versa. Analyses re-save in place via
   `file_metadata`, which is what the session manager's
   overwrite-the-same-file logic was doing by hand.
7. **Nothing goes into `rfmux/tuning` for Periscope's sake unless it is
   toolkit-agnostic and a notebook would use it too.** Plotting stays in
   Periscope. Number-to-number display helpers that the four
   `example_plotting_*.py` files already duplicate (amplitude colour
   normalisation, `offset_khz`, batching, `panels_per_row`) are the one
   candidate; see §5.
8. **A stage is done when the flow test drives it against the standard mock
   array**, through the real tasks, with the real drivers, in the quick tier.
   The deprecated modules are deleted in the last stage, with their tests,
   in one commit.
9. **Nothing that changes a DAC or ADC phase.** IQ rotation returns as a
   separate, later step that records `iq_rotation_deg` on the `BiasPoint`.
10. **Periscope reads what the library wrote and reconstructs nothing.** No
    reader that rebuilds a shape out of a file, opens a file to guess what it
    holds, back-fills a missing key, or re-derives a quantity the container
    already carries. Each such reader is deleted in the same commit as the
    panel that used it — netanal's in stage 1, multisweep's in stage 2 — so no
    stage leaves Periscope parsing a shape the library does not produce. The
    instances known today: the `results_by_iteration` loader
    (`app_runtime.py:1618`), the `bias_kids_output` df extraction
    (`app_runtime.py:1568`, `:1636`), the two dialogs' bare `pickle.load`
    payload readers (`multisweep_dialog.py:37`, `bias_kids_dialog.py:33`), and
    `session_manager.py`'s own `pickle.dump` (`:346`), `pickle.load` (`:546`)
    and open-it-and-look file typing (`:555`), which `store`'s `file_metadata`
    replaces. That is what has been found so far, not what exists; §8 says how
    to recognise the rest.

---

## 3. What goes, what changes, what stays

### Deleted from Periscope

| Where | What | Replaced by |
|---|---|---|
| `tasks.py` `MultisweepTask.run` | the amplitude/direction loop, `iteration_index`, Option A/B switching, `results_for_history` | one `crs.multisweep` call |
| `tasks.py` | `_apply_fitting_analysis`, `_process_fitting_async`, thread pool, model-curve generation, `'nan'`-string success test | `fit_sweeps` in a `RunFitsTask` |
| `tasks.py` | `identify_bifurcation` call and `is_bifurcated` | `BiasReport.findings[].checks` |
| `tasks.py` | `BiasKidsTask`, `DfCalibrationTask` | `FindBiasTask`, `ApplyBiasTask` |
| `multisweep_panel.py` | `results_by_detector`, `update_data`'s injection of `amplitude/direction/iteration`, the three restructurings, `_get_closest_remembered_cf`, `last_output_cfs_by_amp_and_conceptual_idx`, `_get_fit_frequencies`, `_fits_present`, dead `_intermediate_*` | catalog plus module block, `sweep_results` readers |
| `multisweep_dialog.py` | Bias Frequency Method combo, Rotate Saved Data checkbox, `_get_frequencies` precedence chain, legacy `results_by_iteration` reader, `resonance_frequencies` legacy key | `AmplitudeSchedule` view; catalog in, catalog out |
| `multisweep_panel.py` | `handle_error`'s modal `QMessageBox.critical` (`:1163`), which blocks the GUI thread until dismissed and deadlocks a headless run | a transient status label, per AGENTS.md's rule on dialogs |
| `detector_digest_panel.py` | skewed-fit denormalisation guess, `gain_complex` re-multiplication, `rotation_tod` plot, `is_bifurcated` rows, `float(key.split(":")[0])` | `skewed_model_magnitude`, `nonlinear_model_iq`, `BiasFinding` |
| `app.py` | `apply_bias_output`, `_set_bias` NCO midpoint, mock-mode `_start_df_calibration`, `handle_bias_kids` payload plumbing | `crs.apply_bias(catalog)`; see §6 for mock-mode df |
| `app_runtime.py` | the legacy loaders (`results_by_iteration`, `bias_kids_output`, `iq_volts` back-fill, flat fit keys), NCO placement for loaded multisweeps, `iq_complex` reads in `_convert_iq_data` | `store.load`; `apply_bias` owns the NCO |
| `network_analysis_export.py`, `network_analysis_panel.py` | the private `parameters/modules` export payload, `raw_data` tuples, `iq = amps * exp(j phase)` reconstruction, GUI-thread `find_resonances` | the netanal container; `find_resonances_in_netanal` in a task |
| `find_resonances_dialog.py`, `utils.py` | Data Exponent field, `DEFAULT_DATA_EXPONENT`, `min_resonance_separation_hz` | `find_resonances` kwargs |
| `app_runtime.py` | `run_ui_mock_smoke_test`, `_ui_mock_context`, the module-scope `unittest.mock` import, and with them `test/periscope/test_periscope_flow.py` | the flow test, on real widgets offscreen (stage 0) |
| `multisweep_dialog.py`, `bias_kids_dialog.py` | the bare `pickle.load` payload readers | `store.load`, `ResonatorCatalog.from_dict` |
| `notebook_panel.py` | the starter notebook's `pickle.load` helper | a `store.load` line, so the panel teaches the supported reader |
| `session_manager.py` | pickle writing and `_last_exported_per_identifier` overwrite tracking | `store` with the session folder as output directory |

### Deleted from the library, last

`algorithms/measurement/bias_kids.py`, `fitting.py`, `fitting_nonlinear.py`,
`df_calibration.py`, `_legacy.py`, and `test/algorithms/test_bias_kids_fits.py`,
`test_df_calibration*.py`, `test_find_resonances.py` (the eleven shim tests;
their behaviour is pinned in `test/tuning/test_find_resonances.py`),
`test_nonlinear_fit_on_mock.py` (restated in the flow test),
`test_measurement_flow.py` with `simplified_tuning_flow.{md,py}` (rewritten
later against a `tune_resonators` front door, per the design doc §11 step 5).
The re-exported model functions `s21_skewed`, `nonlinear_iq`,
`fit_nonlinear_iq`, `get_y_nonlinear` need their remaining importers moved to
`rfmux.tuning.fits` first.

### Stays

`session_manager.py`'s session folder, browser, screenshot registry and
`data_ready` fan-out (it just stops writing pickles itself). `dock_manager`,
`layouts.FlowLayout`, `ScreenshotMixin`, `UnitConverter` (display units are
the GUI's business), theming, the noise spectrum panel (it reads a
channel-to-frequency map, which the catalog provides), the netanal dialog's
amplitude group and DAC-scale fetch, the pulse capture panel and task
(untouched; out of scope).

---

## 4. The stages

Each stage is one or a few commits, each gated on `pytest --tier=quick` plus
the new flow test, and each leaves the tree runnable. Sizes are relative.

### Stage 0. Guardrails (small) — done

The things that let every later stage be checked rather than asserted, and
the removal of the one thing that made the old shape look checked when it was
not.

What landed differed from the plan in four ways, each recorded where it
belongs: the mocked scaffolding was 530 lines rather than 350, because
`test_dialog_params` turned out to be part of it (its five `self.*_params`
dicts are assigned nowhere else in the app, and only `_ui_mock_context`
called it); `store` already supported a flat output directory, so that
library change was not needed (§5 item 1); driving a real board from a worker
thread needed the hardware map warmed first (§5 item 1b); and
`MultisweepPanel.handle_error` was found to open a modal dialog from a signal
handler, which deadlocks a headless run (§3, stage 2). The two runtime breaks
are now strict xfails that name the stage which clears them.

* **Delete the mocked smoke test and its scaffolding, first.**
  `test/periscope/test_periscope_flow.py`, `run_ui_mock_smoke_test`
  (`app_runtime.py:2648`), `_ui_mock_context` (`app_runtime.py:2403`) and the
  `unittest.mock` import at `app_runtime.py:8`: about 350 lines, most of them
  shipped GUI code. Two reasons, and the second is the one that matters. It
  cannot catch a break, because every dialog, task and signal on the path is a
  MagicMock and the only assertion is `assert True` — which is how the two
  runtime breaks of §1.2 got through. And it does not merely imitate the old
  data shape: it makes that shape a contract of production code, so deleting
  `results_by_detector` in stage 2 would break `app_runtime.py` rather than a
  test. Periscope is to hold what the library returns and nothing else, which
  leaves no place in `rfmux/` for a hand-built
  `{(0.1, "up"): {"some_data": [1, 2, 3]}}`. What the helper nominally
  covered — dialogs construct, windows register in `netanal_windows` and
  `multisweep_windows` — the flow test covers with real widgets offscreen.
* **A real Periscope flow test.** `test/periscope/test_tuning_flow.py`
  builds the panels and tasks against `standard_array()` (RPC only, so quick
  tier) with the offscreen Qt platform, and drives the real drivers through
  the real tasks: netanal, find resonances, multisweep schedule, fits, bias,
  apply. Nothing on the data path is mocked, and the assertions compare panel
  state against the container the driver returned rather than restating a
  shape in the test, so the test cannot drift into describing a structure the
  library stopped producing. It starts by pinning that the current netanal and
  multisweep tasks fail (the two runtime breaks), so it is red before stage 1
  and green after each stage extends it.
* **`store` and the session folder.** `store.set_created_by("periscope")` at
  startup. The session manager sets `store.set_output_directory(session_dir)`
  when a session starts and clears it when it ends. `store.session_directory()`
  currently adds an `ipy_session_YYYYMMDD` day folder under the output
  directory, which Periscope does not want inside a session folder: this
  needs one small library change, an option for a caller-owned flat
  directory (see §5). File names follow `store`'s
  `{type}_{date}_{time}_{label}.pkl`, with the measurement name as the label,
  so a session folder reads the same as a notebook's.
* **Per-panel signals.** Replace the single shared `MultisweepSignals` on the
  Periscope instance with one signals object per task, constructed by the
  caller, the way `PulseCaptureTask` does it. Fixes the "only the newest
  panel gets updates" defect before anything is built on top of it.

### Stage 1. Netanal and Find Resonances on the container (medium)

* `NetworkAnalysisTask` keeps the live `data_callback` plotting and, on
  completion, emits the module's block from the container. The panel stores
  the block per (module, amplitude) instead of `raw_data` tuples; magnitude
  and phase are `abs` and `np.angle` of `iq_counts` at draw time. The export
  button and the session export save the container through `store`;
  loading reads it back. `network_analysis_export.py` loses its private
  payload and the cable-delay unwrap re-derives phase from `iq_counts`.
* Find Resonances runs `find_resonances_in_netanal` in a small task, not on
  the GUI thread. Markers come from `search.candidates`; rejected candidates
  are drawn differently with `rejected_because` in the tooltip; the count
  goes in the plot title. The search is already persisted into the netanal
  block, so the re-export-in-place the panel does today is a `store.save` of
  the same block.
* The Find Resonances settings become the persistent settings panel from
  the section-amplitudes branch, as a view over the finder's arguments:
  delete Data Exponent; relabel Min Separation as the collision cut it now
  is, with blank meaning 0 Hz, next to the `require_isolation` checkbox;
  halve the default dip depth to reproduce what the GUI used to find (the
  migration note in `tuning_revamp_todo.md`, "Retire the legacy resonance
  finder"); keep the amplitude-iteration selector, which now selects which
  netanal block to search. Min Q and Max Q stay, with the todo's caveat that
  they rarely bite at netanal resolution.
* Add/Subtract Resonances by double-click stays. It edits the frequency list
  that seeds the catalog, not the search (§6, judgement call 1).
* **Take Multisweep hands over a `ResonatorCatalog`**, built with
  `search.to_catalog(module, amplitude=<probe amplitude>)`, or from the
  edited list with `ResonatorCatalog.from_frequencies`. The netanal panel
  keeps that catalog per module, and the session export writes it beside the
  netanal as a `catalog` file.
* QoL in this stage: netanal magnitude normalised by sweep power (true dB in
  dBm mode), the filename in the plot title, measurement name plus custom
  suffix with a live filename preview, Enter as OK in the dialog.
* Test: flow test steps 1 and 2; `test_find_resonances_dialog.py` updated
  for the new fields; a display test on a saved netanal block.

### Stage 2. Multisweep as one call (large; the centre of the port)

* **Task.** `MultisweepTask` takes a `catalog.copy()`, an `AmplitudeSchedule`,
  a direction tuple and the sweep parameters, and makes one
  `crs.multisweep` call. `sweep_callback` is re-emitted as
  `sweep_completed(record)`; `data_callback(module, partial, step, direction)`
  as `partial_data(...)` for live curves on the overview tab;
  `progress_callback` straight to the bar. Cancel is `requestInterruption`
  cancelling the coroutine, as now, plus a visible Cancel button, which
  neither branch has. `save=False` in the call; the panel saves through
  `store` on completion so the session manager stays in charge of when.
* **Dialog.** A view over `AmplitudeSchedule`. The amplitude group has one
  radio per constructor: at each resonator's catalog amplitude (the default,
  `AmplitudeSchedule()`), one absolute amplitude, an explicit list, a ramp
  (start, stop, steps, linear or log), multiplicative (start factor, stop
  factor, steps, log or linear). `describe(catalog, n_directions, dac_scale_dbm)`
  fills a live summary line (sweeps, sections, amplitude and power range);
  `validate()` feeds the tick/warning/cross status label and enables Start.
  Direction is two checkboxes producing the tuple. Span, points and nsamps
  as now, with the library's defaults. A "Custom frequencies" mode builds a
  fresh catalog with `from_frequencies` and therefore needs an amplitude and
  mints new names, which the dialog says. Measurement name becomes the
  `label`. Checkboxes "Fit after sweep" and "Find bias after sweep" press
  the stage 3 and 4 buttons on completion. Return starts the sweep.
* **Panel state.** `self.catalog` (the array as swept) and
  `self.module_sweeps` (the block). Nothing else about the data.
* **Tabs, ported from the section-amplitudes grid helpers and rewritten
  against the block:** Mag vs Freq grid, IQ Circles grid, Mag/Phase
  overview. The grid rule, widget caching, kHz offset axes, titles
  `NAME (f_central = ... MHz)`, TABLEAU10-then-inferno with the threshold
  of three, the shared colorbar with dBm and normalised labels, line style
  for direction, stable colours from `schedule.resolve_steps()` known before the
  first sweep, square IQ axes, zoom box, batch navigation with a
  subplots-per-page spinbox, sort by frequency or name, Normalize Traces,
  units radios, double-click to digest. The digest and histograms tabs come
  back in stage 3 when there is something to show in them beyond the
  traces; until then the digest shows the sweep plots only.
* **Re-run** reopens the dialog seeded with the panel's current catalog.
  That is the iterative multisweep: after stage 4 the current catalog is
  `report.catalog`, so the re-run centres on the found bias frequencies at
  the found amplitudes without any history bookkeeping. A "from fitted fr"
  option follows the library item in §5.
* **Files.** Save is `store.save(container, "multisweep", label=)` into the
  session folder; the catalog goes beside it. Load is `store.load` and
  `ResonatorCatalog.from_dict(block["call_params"]["catalog"])`, so a loaded
  sweep always knows its array. Legacy readers are deleted.
* **Multi-module** runs one panel, one task and one call per module (a
  catalog is one module).
* Test: flow test step 3 runs the schedule `multiplicative(0.5, 8, 5)` in both
  directions through the task and checks the panel's block is the driver's.
* **Test: the same measurement, run both ways, writes the same file.** Sweep
  the same seeded mock array twice — once through Periscope in mock mode,
  once headlessly the way a notebook does — and compare the two pickles
  `store` wrote. That is one test covering three things a rendering test does
  not: the file (the `{type}_{date}_{time}_{label}.pkl` naming and the
  `file_metadata` keys), the data (the seven measurement keys, their dtypes
  and their values), and the tuning results derived from them (fits, and the
  bias points and `bias_report` once stages 3 and 4 land). If Periscope ever
  grows a private shape again, this is what says so, and it says it about the
  artefact a user actually keeps.

  Two fields are *expected* to differ and should be asserted to differ rather
  than compared: `created_by`, which is `"periscope"` on one and `"script"` on
  the other, and the timestamp in the name. Everything else agrees or the port
  has gone wrong.

  Two practical notes for whoever writes it. The seed fixes the *array*, not
  the readout: `STANDARD_ARRAY` leaves the simulator's noise on deliberately,
  so an exact value comparison needs noise turned off through `overrides`,
  while the structural comparison — keys, dtypes, shapes, `call_params` —
  holds either way and is the half that catches a re-packaging. And one array
  per process, so the two runs are two processes or one array driven twice.

  This replaces the plan's rendering test on the shipped
  `multisweep_*_demo_biasfind1.pkl`, which the demo notebooks are no longer
  guaranteed to ship.

### Stage 3. Fits on a button (medium)

* Run Fit button and a persistent Fit Settings panel as a view over
  `fit_sweeps`: models (three checkboxes), `approx_Qr`, `normalize`,
  `fit_nonlinearity`, `n_extrema_points`, `max_residual`, and the amplitude
  policy: all sweeps, one iteration, or the bias amplitude
  (`fit_sweeps_at_bias_amplitude`). `RunFitsTask` makes the one call with
  `progress_callback(completed, total)`; the panel's Fitting label reads
  from it. Fits re-save the block in place through `store`.
* Fit Results tab: measured magnitude, skewed and nonlinear model curves
  from `skewed_model_magnitude` and `nonlinear_model_iq` on a finer grid,
  `fr` lines, three-line legends with `a`; an amplitude selector populated
  from iterations that have a fit; failed fits shown with `failed_because`.
* Detector Digest: the three plots and two parameter tables, reading
  `entry["fits"][model]["params"]` and `errors`; keyboard navigation;
  Check Noise stays (it is a `get_samples` call, not analysis). The
  bifurcation column reads `a` against `BIFURCATION_A` until stage 4 gives
  it the report.
* Histograms: fr scatter, Qr/Qc/Qi on shared log bins, coloured by
  amplitude with the same colorbar, an amplitude selector, and a
  `BIFURCATION_A` reference where `a` is shown.
* Mutual locking of Run Fit and Find Bias while one runs; transient
  "Fits complete" label.
* Test: flow test step 4 fits the schedule and checks the panel reads the
  same params `fit_sweeps` wrote; `test_histogram_display.py` updated.

### Stage 4. Find Bias and Apply Bias (medium)

* Find Bias button and a persistent Bias Settings panel as a view over
  `find_bias_points`: `amplitude_method` (both, derivative, hysteresis),
  `frequency_method` (iq_derivative, minimum), `direction`,
  `spike_prominence_factor` (default 0.5; the old GUI's 2.0 is the
  reciprocal, do not copy it), `noise_gate_factor`, `max_discrepancy`,
  `compare`, `max_distance_hz` (as absolute or fraction of span, resolved in
  the dialog). `FindBiasTask` makes one call on `self.module_sweeps` and
  returns the `BiasReport`; the panel adopts `report.catalog` and re-saves
  the block, which now carries `bias_report`.
* The report is shown, not popped up: a status line with counts and the
  flagged names, and flag markers on the affected subplots, with
  `flagged_because` in the tooltip.
* Overlays (Show Bias Info): bold chosen-amplitude trace, bias-frequency
  line in the grid magnitude plots, cross on the IQ loop, star and `a` in the
  legend, dashed lines on the overview.
* IQ Derivatives tab from `normalized_arc_speed` and `iq_arc_speed` for each
  step with the bias frequency marked, and a verdict view: for each
  resonator, each step's `BifurcationCheck.metric` against `threshold`, in
  the form `example_plotting_bias.plot_bifurcation_verdict_map` draws. This
  is the tool for calibrating the thresholds against a real array, which
  the todo file says has not been done.
* **The fit judges the choice** (merge survey §9.4): after Find Bias, run
  `fit_sweeps_at_bias_amplitude` for the nonlinear model and show `a` next
  to each finding, coloured against `BIFURCATION_A`.
* Apply Bias runs `crs.apply_bias(self.catalog)` in `ApplyBiasTask`. On
  success the panel publishes `{r.channel: r.bias.df_calibration}` for df
  units, enables Get Noise Spectrum, and flashes "Bias applied". No
  quantisation or NCO logic in the GUI. `apply_bias_output` and `_set_bias`
  are deleted.
* The main-window Bias KIDs button becomes "Apply Bias from File": load a
  catalog (`from_dict` from a catalog or multisweep file, `from_csv` for a
  CSV), show it in a table for editing (frequency and amplitude per name,
  `update_bias_point`), then `apply_bias`. Custom bias in the multisweep panel is the
  same table over `self.catalog`. No frequency-order matching.
* "Load bias amplitudes" in the multisweep dialog is the default
  `AmplitudeSchedule()` once the panel's catalog is the report's.
* Test: flow test steps 5 and 6 (bias report has no warnings on the schedule;
  every tone lands where the catalog says); `test_bias_kids_dialog.py`
  replaced by a catalog-table test.

### Stage 5. Delete the legacy path (small, one commit)

* Remove the four deprecated modules, `_legacy.py`, and their tests (§3).
* Mock-mode startup df calibration: replaced per §6, judgement call 3.
* Rewrite the "Results Data Structure" section of `AGENTS.md` to describe
  the container shape and the catalog (merge survey §4.1), and update the
  Periscope `README.md` flow section and tooltips. Update tier counts.
* Move the todo entries this port closes out of `tuning_revamp_todo.md`.

### Stage 6. Afterwards, as separate pieces of work

Not part of enabling the basic flow, listed so they are not lost.

* **IQ rotation** as `rfmux/tuning/rotation.py` (angle from a timestream,
  pure) plus a capture step, recording `iq_rotation_deg` on the `BiasPoint`;
  Rotated IQ display mode reads it by channel, fixing the code/channel
  mismatch by construction.
* **`tune_resonators` front door** and a Tune button that runs the whole
  sequence with one progress bar; `simplified_tuning_flow` rewritten against
  it and run in CI.
* **Run-level folder** in `store` tying a catalog to its sweeps (todo,
  "`store.py` owns persistence").
* **Bias frequency and calibration off the fit** (merge survey §9.1 items
  three and four) as options on `find_bias_frequency` and `find_bias_points`;
  the Bias Settings panel gains the radio when the library gains the method.
* **Measured df calibration by tone step** (merge survey §9.2), as an
  optional step of `apply_bias`, when wanted.
* **Noise spectrum panel** reading the catalog for channel-to-frequency and
  moving its `asyncio.run` calls off the GUI thread into a task.
* **`import rfmux` drags in Qt** (todo): lazy `tools` import, so scripts
  using `rfmux.tuning` do not need PyQt6.

---

## 5. Library changes the port needs

Small, and each belongs in the library rather than in Periscope.

1. ~~**`store`: a flat output directory.**~~ Already there:
   `set_output_directory` adds no dated folder, and `session_directory()`
   only reaches for `ipy_session_YYYYMMDD` when nothing was set. The session
   manager just calls it (stage 0).
1b. **The hardware map has to be readable from a worker thread** (done in
   stage 0). Every Periscope measurement runs on a QThread, and this branch's
   drivers name their output block with `crs.module[m].index()`, an ORM read
   (`take_netanal.py:417`, `multisweep.py:878`). Main has both drivers but
   keys its outputs by module number, so it never reads the map from a task.
   The map is one in-memory SQLite database that only the thread which opened
   it may read, so the first such read from a task raised
   `sqlite3.ProgrammingError`. `warm_for_threads(crs)` in
   `core/hardware_map.py` loads those attributes on the map's own thread, where
   they stay cached on the instance; `Periscope.__init__` calls it when it
   takes a board. Two tests in `test_tuning_flow.py` hold it: one drives a
   netanal from a worker, one expires an attribute and pins the
   `ProgrammingError` that the warm-up prevents.

   Decided against making the engine itself thread-shared
   (`check_same_thread=False` plus a `StaticPool`), which would remove the
   class of problem but change core behaviour for real boards. The warm-up
   holds only while nothing commits the session after startup —
   `expire_on_commit` defaults to true, and a commit would un-warm every
   instance — and the only `hwm.commit()` is at load time
   (`core/session.py:278`). Revisit if that changes.
2. **`find_bias_frequency(method="fit")`**: the fitted `fr` from
   `entry["fits"]`, listed in the todo as the reason the function takes an
   entry rather than two arrays. Needed for the re-run dialog's "from fitted
   fr" option and the Bias Settings radio.
3. **Display helpers without a toolkit.** `amplitude colour normalisation`
   (the `0.3 + 0.7 t` dark and `0.75 t` light mapping and the threshold of
   three), `offset_khz`, batching and `panels_per_row` are duplicated across
   the four `example_plotting_*.py` files and would be duplicated a fifth
   time in pyqtgraph. A small pure module, importing neither matplotlib nor
   Qt, that the notebooks and Periscope both call, keeps the two looks
   identical. Judgement call 2 below; the doc's rule that plotting belongs
   to callers is about plotting, and these are arithmetic.
4. **Nothing in `multisweep`.** `sweep_callback`, the four-argument
   `data_callback` and whole-call progress are already what the task needs.
5. **Possibly `ResonatorCatalog.with_names(mapping)`** for attaching a name
   map to a freshly found array (design doc §13 open question). Not needed
   for the basic flow.

---

## 6. Judgement calls

Listed so they can be overruled.

1. **Manual add/subtract of resonances edits the seed list, not the
   search.** The `ResonanceSearch` stays what the algorithm found; the
   catalog is built from the edited list, and the catalog file records the
   difference implicitly. Alternative: record manual additions in
   `Resonator.notes`. Recommended: the former, with a note in `notes`
   ("added by hand") since it costs one line.
2. **Shared display arithmetic goes in a pure library module.** See §5
   item 3. Alternative: duplicate it in Periscope's `utils.py` and accept
   drift. Recommended: the module.
3. **Mock-mode df units at startup.** Today `DfCalibrationTask` steps every
   tone at startup in mock mode so df units work without tuning. Its
   replacement in catalog terms: build the catalog from the simulator's
   playing tones (as `standard_array` does), run a one-step multisweep at
   those amplitudes, read `iq_derivatives_at` at each catalog frequency onto
   the bias points, publish `df_calibration` per channel. That is a
   catalog-producing measurement, so it can be a small library function
   rather than GUI code. Alternative: drop startup calibration and require
   Find Bias in mock mode too. Recommended: the former, because pulse capture
   stores calibrated channels in hertz and the mock demo path relies on it.
4. **Old Periscope pickles do not load.** No converter for
   `results_by_detector` or `bias_kids_output` files. They open on the branch
   that wrote them. Alternative: a one-shot converter in `store`. Recommended:
   none, per the todo's preference for deletion over documenting fallbacks.
5. **Start from this branch's Periscope, port the other branch's display.**
   §1.4. Alternative: merge `mr_multisweep_section_amplitudes` first.
   Recommended: do not merge it.
6. **Find bias after sweep and fit after sweep are checkboxes that press
   the buttons**, not flags on the drivers. Keeps rule 4 intact and the
   drivers measurement-only.
7. **The bifurcation thresholds ship uncalibrated**, with the verdict view
   in stage 4 as the tool to calibrate them. The finding in the todo (both
   detectors fire early on the standard array) is a library question, not a
   port question, and the GUI should not paper over it with its own
   detector.

---

## 7. Test plan

| Stage | Adds | Where |
|---|---|---|
| 0 (done) | deleted the mocked smoke test and its shipped scaffolding; flow test pinning the two runtime breaks as strict xfails; a worker thread driving a warmed board, and the `ProgrammingError` the warm-up prevents; per-panel signals; the session folder as `store`'s output directory | `test/periscope/test_tuning_flow.py`, `test_multisweep_signals_per_task.py`, `test_session_store_directory.py` |
| 1 | flow steps 1-2; dialog fields; netanal block rendering | `test/periscope/` |
| 2 | flow step 3 through the task; Periscope's pickle against a headless one on the same seeded array (file, data and derived results); dialog as a view over `AmplitudeSchedule` (describe/validate wiring) | `test/periscope/` |
| 3 | flow step 4; fit panel reads what `fit_sweeps` wrote; histograms | `test/periscope/` |
| 4 | flow steps 5-6; bias table dialog; overlays present after a report | `test/periscope/` |
| 5 | deletions; tier counts in `AGENTS.md` and `test/README.md` | root |

Everything runs in the quick tier: the standard array is RPC-only and Qt
runs offscreen. Nothing in the port touches the acquisition tier, pulse
capture or `simplified_tuning_flow`.

---

## 8. Holdovers to keep watching for

The mocked smoke test was not an isolated defect, it was one instance of a
habit. Old Periscope carried the tuning flow itself, so it grew a layer of
code whose job was to package, re-package, reconstruct and guess at data — a
layer the library makes unnecessary. §3 lists the instances found so far. The
rest will surface while porting, and the reason for writing this down is that
the response is deletion, not translation: when a stage uncovers one it goes
in that stage's commit, and the row is added to §3 rather than to a follow-up
list.

The shapes it takes, so they are recognisable at a glance:

* **Reconstruction.** Rebuilding something the container already carries:
  `iq = amps * exp(j phase)` in the netanal export, denormalising a skewed fit
  in the digest, re-multiplying `gain_complex`, generating model curves during
  a sweep. The readers in `fits.py` and `sweep_results.py` rebuild curves from
  stored parameters; call them.
* **Re-packaging.** Restructuring a result into a second private shape to hand
  it to the next panel: `results_by_detector`, `res_info_dict`, the
  `"amp:direction"` string keys, the three restructurings in
  `multisweep_panel.py`. A container plus a `ResonatorCatalog` is the only
  shape that crosses a boundary inside Periscope.
* **Guessing.** Deciding what a file holds by opening it and looking,
  back-filling `iq_volts`, precedence chains like the dialog's
  `_get_frequencies`, a `'nan'`-string test for whether a fit succeeded.
  `file_metadata`, typed returns and `failed_because` replace all of it.
* **Fabrication.** Test or demo data hand-written in the shape of a
  measurement, anywhere under `rfmux/` — the stage 0 deletion. A test that
  needs an array calls `standard_array()`, which builds one in under a second
  with no UDP; a test that needs sweeps measures them, and compares them
  against the same measurement made the other way (stage 2).
* **A private copy of a library job.** A second bifurcation detector, a second
  fit orchestrator, an amplitude loop, a re-centring history, an
  `apply_bias_output` that programs tones beside an `apply_bias` that already
  does.

Two greps worth running at the end of each stage. `grep -rn "unittest.mock"
rfmux/` should be empty from stage 0 onwards. `grep -rn "pickle\." rfmux/tools/`
should be empty by stage 5, including the starter notebook that
`notebook_panel.py` writes, which should hand the user `store.load`.
