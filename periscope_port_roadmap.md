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
| `detector_digest_panel.py` | the whole module, its `ui.py` export, the Digest tab, `_open_detector_digest_for_index`, `detector_digest_windows`, `_navigate_digest_to_detector`, the digest `eventFilter` and double-click handler, the auto-open on the noise load path, and the Check Noise button with `_take_noise_samps`, its only caller (stage 2) | nothing in stage 2; rebuilt from scratch against the block and the catalog once fits and bias points exist (§6, judgement call 23) |
| `parameter_histograms_panel.py` | the Histograms tab, `_generate_histograms`, `_ensure_histogram_panel` and the cache invalidation (stage 2) | a tab built in stage 3 over `entry["fits"]`, when there is something to bin |
| `app.py` | `apply_bias_output`, `_set_bias` NCO midpoint, mock-mode `_start_df_calibration`, `handle_bias_kids` payload plumbing | `crs.apply_bias(catalog)`; see §6 for mock-mode df |
| `app_runtime.py` | the legacy loaders (`results_by_iteration`, `bias_kids_output`, `iq_volts` back-fill, flat fit keys), NCO placement for loaded multisweeps, `iq_complex` reads in `_convert_iq_data` | `store.load`; `apply_bias` owns the NCO |
| `network_analysis_export.py`, `network_analysis_panel.py` | the private `parameters/modules` export payload, `raw_data` tuples, `iq = amps * exp(j phase)` reconstruction, GUI-thread `find_resonances` | the netanal container; `find_resonances_in_netanal` in a task |
| `network_analysis_export.py` | `build_export_dict`, `_export_to_pickle`, `_export_to_csv` and the Export-As dialog with its update-suppression dance (stage 1) | `save_netanal()`, one `store.save` |
| `app.py` | `_start_network_analysis`'s modal refusal when `self.dac_scales` is unset (stage 1), which blocks the GUI thread with nobody there to dismiss it, and is set only by the dialogs | nothing; the panel warns that it cannot show dBm and the sweep runs |
| `network_analysis_panel.py`, `app.py` | `_last_export_filename` and `export_data(filename_override=)` for netanal (stage 1) | `store`'s re-save in place, off `file_metadata` |
| `network_analysis_dialog.py`, `app.py` | `dac_scales_used` in the file, and the `"modules" in params` test that told a loaded payload from a new measurement (stage 1) | the board's DAC scale; `dialog.loaded_container` |
| `session_manager.py` | the netanal `pickle.dump` path, `pickle.load` in `load_file`, and the `'parameters' and 'modules'` file typing (stage 1) | `store.load` and `file_metadata`'s `measurement_type` |
| `find_resonances_dialog.py`, `utils.py` | the whole modal dialog, the Data Exponent field, and every `DEFAULT_*` finder constant (stage 1) | `FindResonancesSettingsPanel`, over `find_resonances`' own kwargs and defaults |
| `network_analysis_panel.py` | `_run_and_plot_resonances`' GUI-thread `fitting.find_resonances` call and its four `QMessageBox`es, `_use_loaded_resonances`, the faux resonance legend entry and `_update_resonance_checkbox_text` (stage 1) | `FindResonancesTask`, one `draw_search`, the count in the plot title, a status label |
| `network_analysis_export.py` | `_show_multisweep_dialog`'s "No Resonances" modal (stage 1) | the toolbar status line, which also covers a sweep that has not finished |
| `network_analysis_panel.py` | `resonance_freqs`, the parallel seed list, and `hand_added_freqs` beside it (stage 1) | the `ResonanceSearch` itself: accepted candidates are the array, `accept`/`reject` are the edits |
| `extract_params.py` | `ParamKeyExtractor`, 102 lines of AST parsing to find the keys a dialog's `get_parameters` builds (stage 1) | calling `get_parameters()` and looking at the dict |
| `network_analysis_dialog.py` | Cable Length field, "Clear all channels first" checkbox (stage 1) | nothing; neither is `take_netanal`'s to do |
| `network_analysis_export.py` | the board write at the end of the cable-delay unwrap (stage 1) | nothing; the unwrap adjusts the display only |
| `utils.py` | `NETANAL_UPDATE_INTERVAL` (stage 1), a throttle whose interval was recomputed and then ignored | nothing |
| `test/periscope/test_netanal_export_pairing.py` | hand-built sweep tuples (stage 1) | the same contract in the flow test, over measured sweeps |
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
the GUI's business), theming, the noise spectrum panel and the multisweep
toolbar's Get Noise Spectrum button that opens it (it reads a
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

* **The data path (done).** `NetworkAnalysisTask` emits the module's measured
  trace, live and on completion, as one `data_update(module, trace)` signal.
  The panel stores it per module in `netanal_traces` instead of `raw_data`
  tuples; magnitude and phase are `abs` and `np.angle` of `iq_counts` at draw
  time, in the panel, once. The cable-delay unwrap re-derives phase from
  `iq_counts`. What differed from the plan is in §5 item 6 and §6 judgement
  calls 7-10.
* **One netanal is one probe amplitude (done).** `take_netanal` takes a scalar
  `amp` and returns one trace tagged with it; there is no amplitude axis to
  iterate. Periscope's ladder is gone with it: the per-window amplitude queues
  and `current_amp_index`, the completion handler that started the next
  amplitude, `_start_next_amplitude_task` (now `_start_netanal_task`, one task
  per module), the `amp_curves`/`phase_curves` dicts and `_amplitude_color`
  that coloured them, the "Amplitude n/N" progress label, and the two
  precedence chains that picked which of several sweeps to search and to fit a
  delay from. The legend keeps its job -- it names the power the trace was
  probed at, read off the trace's own `sweep_amplitude` -- and the export holds
  one `sweep` per module rather than an index. §6 judgement call 12 covers the
  dialogs, which still offer a list.
* **Files through `store` (done).** The completion signal carries the
  container, the panel keeps it (`netanal_container`, a union keyed by module
  identifier), and Save writes it with `store.save(container, "netanal",
  label=)`. The measurement name in the dialogs is that `label`, so the file is
  `netanal_YYYYMMDD_HHMMSS_<name>.pkl` wherever it is written from. Saving the
  same panel twice overwrites the same file, which is the `file_metadata`
  mechanism that replaces `filename_override` and `_last_export_filename`, and
  a re-run clears the container so it writes a new file. Loading is
  `store.load` plus the blocks the driver wrote. Gone with it:
  `build_export_dict`'s six-unit `parameters/modules` payload, the CSV export
  (three unit copies of one trace, per §6 judgement call 13), the
  Export-As file dialog and its update-suppression dance, `dac_scales_used`,
  the session manager's own `pickle.dump` for netanal and its
  open-it-and-look netanal typing, and the starter notebook's `pickle.load`
  helper. §6 judgement calls 13-16 cover the choices. Since the DAC scale is
  no longer carried in the file, the modal dialog that refused to start a
  sweep without one is gone too: it is a legend, not a measurement.
* **Find Resonances on the container (done).** `FindResonancesTask` runs
  `find_resonances_in_netanal` on the module's own netanal output, off the
  GUI thread, and the panel draws `search.candidates` as the dashed lines it
  always drew, now with the depth and Q estimate in a tooltip. Rejected
  candidates are crosses on the magnitude curve, hoverable for
  `rejected_because`; they are crosses and not lines because a search can
  reject far more than it keeps. The count is in the magnitude plot's title,
  which retired the faux legend entry that carried it. The search goes into
  the block, so a netanal that has been saved is re-saved in place; one that
  has not been is left to the Save button. The four `QMessageBox` calls on
  the path -- no module selected, no data, finder raised, nothing found --
  are one transient status label in the toolbar. Gone with the old path:
  `fitting.find_resonances` as Periscope's last caller, the legacy shim's
  `resonance_frequencies` dict, `_use_loaded_resonances` (loading now calls
  the same `draw_search`), and `extract_params.py`, whose only caller was
  the deleted dialog test. §6 judgement calls 17-19 cover the choices.
* **The settings are a persistent panel (done).** `FindResonancesSettingsPanel`
  is a non-modal, always-on-top window over the finder's keyword arguments,
  behind a `⚙` beside the button, persisted as JSON through `settings.py` with
  Reset to Defaults. Data Exponent is gone with `DEFAULT_DATA_EXPONENT`; Min
  Separation is now "Collision cut", in kHz, beside the `require_isolation`
  checkbox, with "Off" for 0 Hz; Min Q and Max Q stay, with the todo's caveat
  in the Max Q tooltip. The defaults are read out of `find_resonances`'
  signature at import, so they cannot drift from the library -- which is also
  what halves the dip depth the old dialog shipped (2.0 dB, whose effective
  floor was 1.0, to the library's 1.0). The other `DEFAULT_*` finder
  constants in `utils.py` went with it. Dropped from the plan: the
  amplitude-iteration selector, per judgement call 17.
* **Add/Subtract Resonances by double-click edits the search (done).**
  Left-click accepts, right-click or shift-click rejects, through
  `ResonanceSearch.accept`/`reject` — so removing a resonance moves it to
  `rejected` with a reason and draws it as one of the crosses, where clicking
  it again brings it back exactly as the finder measured it, and adding one
  accepts an arbitrary frequency with nothing claimed about it. §6 judgement
  call 1 records the overrule; judgement call 20 covers why a hand-accepted
  frequency carries `nan` rather than measurements.
* **Take Multisweep hands over a `ResonatorCatalog` (done).** The search is
  the array: `_show_multisweep_dialog` calls
  `search.to_catalog(module, amplitude=<the netanal's probe amplitude>)` at the
  press and puts the catalog in `params["catalog"]`, where stage 2's task picks
  it up. The dialog is given that catalog's own bias frequencies, so the list
  it shows is a projection of it rather than a second source.
  **The catalog is not written anywhere of its own**: `multisweep` records the
  catalog it swept in its own output (`call_params["catalog"]`), and the
  netanal holds the search it was built from, so a third file that has to be
  kept paired with both is exactly what `find_resonances_in_netanal`'s own
  reasoning rules out. Double-click add/subtract edits the search
  (judgement call 1, overruled), through `accept`/`reject`; the panel writes
  the edited search back into the netanal block and re-saves through `store`
  when there is a file, as the search itself does. Gone with
  it: the panel's parallel `resonance_freqs` seed list, the `hand_added_freqs`
  set beside it, and the "No Resonances" modal, now the toolbar's status
  line. §6 judgement calls 20-22 cover the rest.
* QoL in this stage: netanal magnitude normalised by sweep power (true dB in
  dBm mode), the filename in the plot title, measurement name plus custom
  suffix with a live filename preview, Enter as OK in the dialog.
* Test: flow test steps 1 and 2; `test_find_resonances_dialog.py` updated
  for the new fields; a display test on a saved netanal block.

### Stage 2. Multisweep as one call (large; the centre of the port)

* **Task (done).** `MultisweepTask` takes a `catalog.copy()`, an
  `AmplitudeSchedule`, a direction tuple and the sweep parameters, and makes
  one `crs.multisweep` call. `sweep_callback` is re-emitted as
  `sweep_completed(record)`; `data_callback(module, partial, step, direction)`
  as `partial_data(...)` for live curves on the overview tab;
  `progress_callback` straight to the bar. `save=False` in the call; the panel
  saves through `store` on completion so the session manager stays in charge of
  when. Gone with the loop: the per-amplitude and per-direction iteration, the
  re-centring history it read off the panel (`_get_closest_remembered_cf`,
  `conceptual_section_frequencies`), the `window` argument that let a worker
  read the GUI, the inline fitting on a thread pool of its own, and the
  restructuring into `results_for_plotting`/`results_for_history` -- 354 lines
  of `tasks.py` out against 82 in, and with them the module's `fitting`,
  `fitting_nonlinear`, `os` and `concurrent.futures` imports.

  Four things differed from the plan. The task is not told its module: a
  catalog belongs to one, so it reads `catalog.module` and there is no second
  place for the two to disagree. Cancel emits nothing -- the panel that pressed
  it is the one that knows, and a cancelled sweep is a routine outcome, not an
  error -- while a call that finishes as Cancel arrives is still handed over
  rather than thrown away. Two arguments moved into the dialog ahead of its own
  rewrite, because they are the library's own forms and the alternative was a
  translation layer in between: an amplitude list becomes
  `AmplitudeSchedule.explicit`, and "Both" becomes `("upward", "downward")`.
  And frequencies typed into the dialog by hand, which no search named, are
  minted into a catalog in `_start_multisweep_analysis` until the dialog's
  Custom frequencies mode does it where it can say so.

  What the sweep no longer does, until the stages that own it: fits during the
  measurement (stage 3's button), drawing (the panel's `update_data` is off the
  measure path, still on the load path), and the session auto-export that
  `all_sweeps_completed` fired (this stage's Files bullet). The panel keeps `multisweep_container`,
  `module_sweeps` and `catalog`, reads the catalog back out of
  `call_params["catalog"]` rather than holding a second copy, and reports an
  error on its status line -- `handle_error`'s modal, opened from a signal
  handler, is what deadlocked a headless run.
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

  Four controls went with the task rewrite rather than waiting for this
  bullet, because a control that drives nothing is worse than a missing one:
  Bias Frequency Method and Rotate Saved Data (`multisweep` measures, and
  nothing on this branch rotates a saved sweep), and the Apply Skewed Fit and
  Apply Nonlinear Fit checkboxes (stage 3's button). `_direction_text` came
  with them, so a re-run seeded from a previous call's own arguments finds
  "Both" in the combo rather than falling back to Upward.
  `test_multisweep_dialog_params.py` pins what the dialog emits, including
  that it emits nothing `multisweep` would refuse.
* **Panel state.** Three attributes about the data, all of them the library's:
  `self.multisweep_container` (what the call returned), `self.module_sweeps`
  (this module's block out of it) and `self.catalog` (read back from
  `module_sweeps["call_params"]["catalog"]`). Beside them, view state that
  describes what is on screen and never the measurement — unit mode, the
  Normalize flag, sort order, batch index, the names and steps and directions
  currently selected, the pyqtgraph widget cache. `self._live` is a fourth data
  attribute that exists only while a sweep runs; see below. Deleted with the
  rewrite: `results_by_detector`,
  `last_output_cfs_by_amp_and_conceptual_idx`,
  `current_amplitude_being_processed`, `current_iteration_being_processed`, and
  the six-argument `update_data` that filled them.
* **Every plot reads the block, on every redraw.** Nothing is copied out of it,
  nothing is precomputed into a per-trace structure, and no panel hands another
  panel anything but the block and the catalog. A redraw walks the results and
  throws away what it built, and the walk is the library's own:

  ```python
  for name in self._selected_names():        # catalog order, or sorted by frequency
      measured = collect_amplitude_iterations_for(self.module_sweeps, name)
      for step, by_direction in measured.items():
          for direction, sweep in by_direction.items():
              ...                            # sweep is the seven-key entry
  ```

  Everything a trace needs is on `sweep`:

  | What the plot needs | What it reads |
  |---|---|
  | x, in the grids | `(sweep["frequencies"] - sweep["original_center_frequency"]) / 1e3`, kHz either side of the sweep's own centre |
  | x, in the overview | `sweep["frequencies"]`, unshifted, so the array shares one axis |
  | magnitude | `UnitConverter.convert_amplitude(abs(sweep["iq_counts"]), ...)` for all three units, the same converter the netanal panel uses; volts there are `convert_roc_to_volts`, one constant scale, so the result equals `abs(sweep["iq_volts"])` |
  | phase | `np.angle(sweep["iq_counts"])`, at draw time. A sweep has no phase key on purpose: it would be the readout chain's phase, not the resonator's |
  | IQ circle | `sweep["iq_counts"]`, scaled by `convert_roc_to_volts` unless the units are counts — the same numbers `iq_volts` holds, and a sweep still being measured has only counts |
  | Normalize Traces | the panel's own meaning, kept: the first point of each trace is the reference (subtracted in dBm, divided otherwise), through `UnitConverter.convert_amplitude`, and the peak magnitude for an IQ loop. Not `example_plotting_multisweep.sweep_iq`'s divide-by-drive — see §6, judgement call 24 |
  | trace colour | `sweep["sweep_amplitude"]`, through the scale below |
  | line style | `sweep["sweep_direction"]`: solid upward, dotted downward |
  | panel title | the name the entry is keyed by, and its `original_center_frequency` |

  That is the whole reader, and it is what the `example_plotting_*.py` files do
  in matplotlib three lines at a time. §5 item 3 lets the two *look* different;
  it does not let them compute different things.

  Two selections above the walk, and they are the only place a step number is
  used as an index: which steps and directions to draw (checkboxes, all by
  default), and which names, which is a slice of `catalog.names()` for the
  current batch. `find_iteration_matching_amplitude` answers "the step this
  resonator is biased at" when stages 3 and 4 need one, so the panel never
  matches an amplitude by hand.
* **The colour scale exists before the first point arrives.** Colour means
  drive amplitude, so it is graded over every amplitude the call will produce,
  and `schedule.resolve_steps(catalog)` gives all of them at the press of Start
  without touching a board — one `AmplitudeStep` per step, `amplitudes` keyed
  by name. So a trace's colour does not shift as later sweeps land, and a
  resonator under a multiplicative schedule is coloured by what *it* is driven
  at rather than by which step it is on: those are different numbers, which is
  why `sweep_amplitude` is per-entry and not per-step. Log-normalised;
  TABLEAU10 up to `AMPLITUDE_COLORMAP_THRESHOLD` steps and a colormap above it;
  one shared colourbar per grid, labelled in normalized units and, when a DAC
  scale is known, in dBm through `UnitConverter.normalize_to_dbm`.
* **A live trace is the same read, one key short.** `partial_data(module,
  partial, step, direction)` carries `{name: {"frequencies", "iq_counts",
  "original_center_frequency"}}` for the resonators in the NCO region currently
  being swept — each from its first point up to the latest, resent in full every
  time, with finished regions not resent and unstarted ones absent. So the panel
  assigns per name rather than appending to an array or replacing the step. It
  is stored under the container's own nesting,
  `self._live[name][(step, direction)] = partial[name]`, so the reader above
  reaches it the same way and a half-drawn sweep and a finished one are one code
  path. A callback arrives per point and a grid takes longer to draw than a
  point takes to measure, so redraws are coalesced on a 100 ms single-shot
  timer.
  The one key it lacks is `sweep_amplitude`, and a live trace's colour comes
  from `resolve_steps`' amplitude for that name and step — the number the driver
  will write into the entry. `self._live` is cleared when `completed` arrives
  and `module_sweeps` becomes the only source. It is a cursor, not a second copy
  of the measurement: it holds the callback payload verbatim, it is never saved,
  and it does not outlive the sweep.
* **The three tabs, from that reader.** Mag vs Freq grid, IQ Circles grid,
  Mag/Phase overview. Ported from the section-amplitudes grid helpers as
  *rendering*: the grid rule `ncols = max(min(4, n), ceil(sqrt(n)))`, widget
  caching, titles `NAME (f_central = ... MHz)`, square IQ axes, zoom box, batch
  navigation with a subplots-per-page spinbox, sort by frequency or name.
  Nothing on these tabs is derived from a fit, a bias point or a bifurcation
  test: a sweep is a measurement, and the overlays that read those arrive with
  the buttons that produce them in stages 3 and 4.
* **The detector digest is not wired up at all** (maclean, 2026-09-10). It is
  not shown, nothing is computed for it, and no code path reaches it; it is
  rebuilt from scratch once fits and bias points exist, against the block and
  the catalog, rather than ported. So stage 2 removes the Digest tab and its
  placeholder, `_open_detector_digest_for_index`, `detector_digest_windows`,
  `_navigate_digest_to_detector`, the `eventFilter` that gave it keyboard
  navigation, the double-click handler that opened it, the digest invalidation
  in the old `update_data`, the theme loop over its windows, the auto-open on
  the noise load path (`app.py:2929`), and `detector_digest_panel.py` itself
  with its export from `ui.py`. Two consequences worth naming rather than
  discovering: the **Check Noise** button lives *inside* the digest and goes
  with it, taking `MultisweepPanel._take_noise_samps`, its only caller, along —
  while **Get Noise Spectrum** is a toolbar button of the multisweep panel
  reaching `NoiseSpectrumPanel` directly, and is untouched. Double-click on a
  subplot therefore does nothing in stage 2; it is reconnected when there is a
  digest to open.
* **Histograms are unwired on the same terms.** There is nothing to bin until
  fits exist, `parameter_histograms_panel.py` reads `results_by_detector` for
  what it bins, and stage 3 is where the tab earns its place — so stage 2
  removes the tab, `_generate_histograms`, `_ensure_histogram_panel` and the
  cache invalidation, and stage 3 builds it against `entry["fits"]`. Recorded
  as §6 judgement call 23; the digest's treatment was instructed, this one is
  the same reasoning applied one panel over, and is the piece to overrule if
  you would rather keep a histogram tab through the port.
* **What this deletes.** Every read of `results_by_detector` in
  `multisweep_panel.py` (44 today), and with them `_redraw_sweep_grid`'s
  detector-index dictionaries, `_prepare_export_data`'s payload,
  `_get_fit_frequencies`, `_get_closest_remembered_cf` and
  `_toggle_cf_lines_visibility`'s history lines — plus the two panels above and
  their 29 and 7 reads of the same shape. After it, `grep -rn
  "results_by_detector\|results_by_iteration" rfmux/tools/periscope/` should
  reach the dialogs' loaders and `app.py`'s load path and nothing else, which
  is the Files bullet's work.
* Test: a redraw on a block measured by the real task puts one curve per
  (name, step, direction) on the axes, with the entry's own frequencies and
  `abs(iq_counts)` on it; a live `partial_data` for a step draws a shorter curve
  in the same colour the finished sweep gets; and switching units, Normalize and
  batch changes what is drawn without touching `module_sweeps`. Two existing
  cases go with the deletions above, since they construct the panels:
  `test_viewbox_lifetime.py`'s `DetectorDigestPanel` entry and
  `test_laptop_fit.py`'s `ParameterHistogramsPanel` one. Both are per-panel
  checks of a general rule, so the rule keeps its other cases; they come back
  with the panels.
* **Re-run** reopens the dialog seeded with the panel's current catalog.
  That is the iterative multisweep: after stage 4 the current catalog is
  `report.catalog`, so the re-run centres on the found bias frequencies at
  the found amplitudes without any history bookkeeping. A "from fitted fr"
  option follows the library item in §5.

  **The control for it already exists and reads the old shape.** The dialog's
  "Use resonance fit frequency" entry calls `_get_frequencies(payload,
  raw_section_centers=False)`, which pulls `fit_params['fr']` and
  `nonlinear_fit_params['fr']` out of a `results_by_detector` or
  `results_by_iteration` payload, choosing between them by the `apply_*_fit`
  settings recorded in the file. Every part of that is gone: `fit_sweeps`
  writes `entry["fits"][model]["params"]["fr"]`, and a block does not record
  whether a fit was *asked* for because a fit that ran is in it. When stage 3
  hooks fits up, this is rewritten against the new shape and against the
  panel's own `module_sweeps` rather than a re-opened file -- not extended to
  read both. It is one of §8's precedence chains, and the response there is
  deletion, not translation. Noted at the function.
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
* Histograms, built new against the block: fr scatter, Qr/Qc/Qi on shared log
  bins, coloured by amplitude with the same colorbar as the grids, a step
  selector, and a `BIFURCATION_A` reference where `a` is shown. Read through
  `entry["fits"][model]["params"]` and `errors`, skipping entries whose fit
  carries `failed_because`.
* **Detector Digest, written from scratch** — stage 2 deleted it rather than
  porting it (§6, judgement call 23), and it is designed once the pieces it
  shows exist rather than reassembled from the old one. A resonator's sweeps
  at every step, its fitted parameters and errors, keyboard navigation between
  resonators and steps, and Check Noise back with it (a `get_samples` call, not
  analysis, so it needs `_take_noise_samps` again). The bifurcation reading is
  `a` against `BIFURCATION_A` until stage 4 hands it the report's own checks.
  Scope this when stage 3 starts, not now; if the fit tabs turn out to say
  enough on their own, the honest outcome is that it does not come back.
* Mutual locking of Run Fit and Find Bias while one runs; transient
  "Fits complete" label.
* Test: flow test step 4 fits the schedule and checks the panel reads the
  same params `fit_sweeps` wrote, and that a fit that failed is shown as
  failed rather than skipped silently; the histogram tab bins what
  `entry["fits"]` holds. Not `test_histogram_display.py` — that one covers the
  main window's live amplitude histogram and its df units, and has nothing to
  do with this tab.

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
3. ~~**Display helpers without a toolkit.**~~ **Not doing this** (maclean,
   2026-09-10): the GUI and the notebooks are allowed to look different.
   `amplitude colour normalisation` (the `0.3 + 0.7 t` dark and `0.75 t`
   light mapping and the threshold of three), `offset_khz`, batching and
   `panels_per_row` are duplicated across the four `example_plotting_*.py`
   files, and pyqtgraph duplicates them a fifth time. A shared module would
   keep the two looks identical, which is the thing that is not wanted: a
   panel the operator drives at the board and a figure in a notebook answer
   to different constraints, and pinning them together makes every later
   change to one a negotiation with the other. The arithmetic is a few lines
   in each place. Revisit only if the two are found drifting in a way that
   confuses rather than suits — a future to-do, not a prerequisite.
4. **Nothing in `multisweep`.** `sweep_callback`, the four-argument
   `data_callback` and whole-call progress are already what the task needs.
4b. **Editing a search by hand** (done in stage 1).
   `ResonanceSearch.reject(frequency_hz, reason=)` moves the nearest accepted
   candidate to `rejected`; `accept(frequency_hz)` restores a rejected one
   exactly as it was found, or accepts an arbitrary frequency with its measured
   fields `nan` (§6, judgement call 20). In the library because a notebook
   edits a search for the same reasons a GUI does. **Nothing else was needed:**
   writing the edit back is
   `netanal_trace(block)["resonance_search"] = search.to_dict()` and then
   `store.save`, which overwrites the file the block already knows it came
   from. A `record_search` helper wrapping those two lines was written and
   removed again — `store`'s save-in-place is the whole mechanism, and a
   function to hide one dict assignment earns nothing.
5. **Possibly `ResonatorCatalog.with_names(mapping)`** for attaching a name
   map to a freshly found array (design doc §13 open question). Not needed
   for the basic flow.
6. **`take_netanal`'s `data_callback` hands over a partial trace** (done in
   stage 1). It computed `np.abs` and `np.degrees(np.angle(...))` and passed
   `(module, freqs, amps, phases)`, so the live path carried a polar
   projection of data the return value carries as IQ, under names the block
   does not use, and every consumer round-tripped it back
   (`amps * exp(1j * radians(phases))` in the netanal panel). It now passes
   `(module, partial)` with the block's own keys, `frequencies` and
   `iq_counts`, growing as points arrive — the shape `multisweep`'s
   `data_callback` already used. No measurement algorithm computes magnitude
   or phase now; the returns never did.

   Note the word: `amp` is the DAC drive amplitude and `amp_array` was |S21|,
   twelve lines apart in the same function. Where a GUI needs it, it is
   *magnitude*.

---

## 6. Judgement calls

Listed so they can be overruled.

1. ~~**Manual add/subtract of resonances edits the seed list, not the
   search.**~~ **Overruled (maclean, 2026-09-10): they edit the search.** The
   operator is the last rejection pass and works the way the automatic ones do
   — `ResonanceSearch.reject` moves a candidate to `rejected` with a reason,
   `accept` brings one back or measures a new one off the searched trace — so
   the search stays the whole record of what was found and what was decided
   about it, either edit undoes the other, and both survive a save because the
   search is already in the netanal. What this replaces: a parallel frequency
   list in the panel, a set beside it tracking which entries were added by
   hand, and a `Resonator.notes` entry to carry that distinction into the
   catalog. All three are gone; nothing is deleted from a search any more.
2. ~~**Shared display arithmetic goes in a pure library module.**~~
   **Overruled (maclean, 2026-09-10): duplicate it.** See §5 item 3. The GUI
   plots need not match the notebook plots, so the shared module's benefit —
   one look — is not one. Periscope carries its own copy of the arithmetic.
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
7. **Nothing Periscope does to the board that the driver does not.** The
   netanal task called `crs.clear_channels()` and
   `crs.set_cable_length()` before every sweep; neither is in `take_netanal`,
   which sets the NCO it needs and zeroes its own tones on the way out.
   Both are gone, with the dialog's Cable Length field. Cable length is
   ruled out entirely for now under §2 rule 9 — it rotates the phase of
   everything the board reads — so the unwrap in `network_analysis_export.py`
   still fits a delay and adjusts the *displayed* phase, but no longer writes
   the result to the board. `SetCableLengthTask` and `SetCableLengthSignals`
   stay in `tasks.py` with no callers. If cable length comes back it comes
   back as an analysis helper that removes a phase slope from measured data,
   never as a board write.
8. **One `data_update` signal, carrying the trace.** `data_update` and
   `data_update_with_amp` were the same payload one float apart, both emitted
   at both emit sites, and the panel stored each trace twice — under
   `'default'` and under `f"{module}_{amplitude}"`. One signal now, always
   with the amplitude. It carries the trace (`block["results"]`) rather than
   the whole block, because that is the part the panel plots and it is the
   shape the live partial has; the container is what gets saved, and
   `completed` grows to carry it when the save moves to `store`.
9. **Three copies of the redraw became one.** `_toggle_normalization`,
   `_update_unit_mode` and `_redraw_all_plots` each walked `raw_data` with
   the same body and the same `'default'` special case. They call
   `_redraw_magnitudes(module)`. With the amplitude ladder gone the
   single-trace `amp_curve`/`phase_curve` plot items are the curves again, and
   the per-amplitude dicts beside them are deleted.
10. **`UnitConverter.convert_amplitude` takes an `iq_data` argument it never
    reads.** Left alone: it is called from four panels, and removing a dead
    parameter across all of them belongs with whichever stage touches them,
    not with netanal's.
11. **The bifurcation thresholds ship uncalibrated**, with the verdict view
    in stage 4 as the tool to calibrate them. The finding in the todo (both
   detectors fire early on the standard array) is a library question, not a
   port question, and the GUI should not paper over it with its own
   detector.
12. **The netanal dialogs are left offering a list of amplitudes**, and
    `_start_netanal_task` takes the first. They are rewritten in this stage
    anyway -- measurement name, filename preview, the persistent settings
    panel -- and the shared `NetworkAnalysisDialogBase` amplitude group is
    still a multisweep amplitude ladder until stage 2 replaces it with an
    `AmplitudeSchedule` view. One resolving line in the caller beats a flag
    threaded through a base class that is about to go. It is the one place
    left where a netanal knows the word `amps`.
13. **The CSV export is deleted rather than rewritten** (stage 1). It wrote
    one metadata file plus three data files per module — the same trace in
    counts, volts and dBm, with a phase column derived from the IQ it did not
    write — which is `build_export_dict`'s unit fan-out in another format. A
    netanal file opens with `store.load` and converts in two lines. Alternative:
    a four-column CSV (frequency, I, Q) from the trace. Recommended: none until
    someone asks for it; the request will say which columns.
14. **Save writes through `store`'s naming, with the user naming the label.**
    There is no Save-As: the button writes
    `netanal_YYYYMMDD_HHMMSS_<measurement name>.pkl` into `store`'s output
    directory, which is the session folder while a session is open. That is
    what makes a Periscope file and a notebook file the same file. Alternative:
    a `path=` argument on `store.save` so a file dialog can name it. Not added;
    the filename is the library's to compose, and the label is the part a user
    actually wants to choose.
15. **The DAC scale is not written into the file** (stage 1). It was carried as
    `dac_scales_used` and restored on load, which is a display setting of the
    board taking a ride in a measurement file. A loaded netanal shows dBm from
    the connected board's scale and says it cannot when there is none. If the
    probe power in dBm turns out to be provenance worth keeping, the driver
    should record it, not the GUI.
16. **A netanal is saved once, when it finishes** (stage 1). The re-save after
    Find Resonances is restored in the same commit as the search task, when the
    search is written into the block by `find_resonances_in_netanal`; the
    legacy shim's frequency list is a panel attribute with nowhere in the
    container to live, and inventing a key for it is the re-packaging this port
    is for deleting.
17. **No amplitude-iteration selector on the Find Resonances settings**
    (stage 1). The plan kept it, "which now selects which netanal block to
    search". One netanal is one probe amplitude and one panel tab is one
    module, so the tab already picks the block and a selector beside it would
    be a second, disagreeing way to say the same thing. Alternative: keep it
    as a module selector, so a search can run on a tab that is not showing.
    Overrule this if searching several modules from one place turns out to be
    what the operator wants.
18. **A search re-saves only a netanal that has a file** (stage 1). The block
    changed, so the file that holds it is out of date; overwriting it is the
    `file_metadata` mechanism working as intended. A panel that has never been
    saved is left alone rather than given a file by a search, because a file
    appearing from an analysis button is a surprise. A measured netanal is
    saved when it finishes, so in practice the search updates it.
19. **Rejected candidates are crosses, not lines** (stage 1). The plan said
    "drawn differently". A search rejects by collision and by count, and a
    collision cut a little too wide rejects most of an array, so full-height
    lines would bury the trace; one hoverable scatter item per module carries
    every reason. Alternative: lines in a lighter pen behind a checkbox of
    their own.
20. **A hand-accepted frequency is blank, not measured** (stage 1;
    **maclean, 2026-09-10**, overruling a first attempt that measured it).
    `accept` at a frequency the finder had no candidate for records that
    frequency and sets `depth_db`, `width_hz` and `q_estimate` to `nan`. The
    point of the gesture is that **an arbitrary frequency can join the array** —
    it is a place someone wants a tone, not a claim that a resonator is there —
    so there is nothing to characterise and `nan` says so. What this replaces:
    reading prominence and width off the searched trace with the same scipy
    calls the finder uses, plus a one-point nudge onto the local minimum,
    because scipy reads no prominence anywhere but a local minimum and a dip's
    minimum falls between samples. All of it deleted. `frequency_hz` is now
    exactly what was asked for, unrounded, and is quantized once at the end,
    where `to_catalog` builds a `BiasPoint`. `index` remains the nearest
    searched point, which is what plots the marker.

    Two consequences worth knowing. `nan` is never equal to itself, so two
    hand-accepted candidates do not compare `==` even after a faithful round
    trip through `to_dict`; the round-trip test compares them field by field.
    And a `nan` depth is what the netanal panel's marker tooltip reads to say
    "added by hand" instead of three nans — the only thing that now
    distinguishes a hand-accepted candidate from a found one, since
    `ResonanceCandidate` has no provenance field.
21. **`to_catalog` is called at the press, and its names are not kept**
    (stage 1). Every press mints a fresh catalog, so a dialog opened and
    cancelled renames the array. That is only visible in a name, the sweep
    that runs is the one that records its catalog, and keeping one would mean
    a second piece of state to hold against an editable search. Overrule this
    if operators start referring to resonators by name before the first sweep.
22. **`params["catalog"]` has no reader until stage 2** (stage 1). The
    catalog crosses the boundary now because the press is the handover;
    `MultisweepTask` still runs off the frequency list the dialog produces,
    and picks the catalog up in stage 2 when that list goes.
23. **The digest and histogram tabs are unwired in stage 2 rather than
    ported** (maclean, 2026-09-10, for the digest). Both read
    `results_by_detector`, which the grids stop producing, and both show
    quantities that do not exist until stage 3 — so translating them buys a
    tab that displays nothing, and leaving them buys two shipped panels
    reading a shape nothing writes, which is the §1.2 state stage 0 existed to
    end. They come back built against the block and the catalog: the digest
    from scratch when fits and bias points exist, the histograms in stage 3
    over `entry["fits"]`. The instruction covered the digest; extending it to
    the histograms is the same reasoning one panel over, and is the half to
    overrule.
24. **Normalize Traces keeps the panel's meaning, not the notebook's**
    (stage 2). `example_plotting_multisweep.sweep_iq` divides a sweep by the
    drive that produced it; Periscope's checkbox references each trace to its
    own first point, which is off resonance and where |S21| is proportional to
    drive — so it collapses an amplitude ladder onto one another for the same
    reason, and also removes the gain the traces have in common. It is what the
    shipped control has always done, it lives in `UnitConverter` where the
    netanal panel uses it too, and §5 item 3 already says the GUI and the
    notebooks may look different. The plan said `sweep_iq`; the code is right
    and the plan was corrected.
25. **Every trace is coloured by its drive, including the only one**
    (stage 2). The old grid drew a single-amplitude sweep in the foreground
    colour and switched to the amplitude colours when a second one arrived, so
    a live trace changed colour mid-run — which is what resolving the schedule
    before the first point exists to prevent. Colour now always means drive; a
    one-step sweep is drawn in TABLEAU10's first colour rather than in black or
    white.

---

## 7. Test plan

| Stage | Adds | Where |
|---|---|---|
| 0 (done) | deleted the mocked smoke test and its shipped scaffolding; flow test pinning the two runtime breaks as strict xfails; a worker thread driving a warmed board, and the `ProgrammingError` the warm-up prevents; per-panel signals; the session folder as `store`'s output directory | `test/periscope/test_tuning_flow.py`, `test_multisweep_signals_per_task.py`, `test_session_store_directory.py` |
| 1 (done) | the netanal step, no longer an xfail; the trace reaching the panel carries the driver's keys and complex IQ; the panel stores it and draws `abs(iq_counts)`; the cable-delay unwrap runs over that trace; the completion signal carries the container; a saved netanal reads back through `store.load` as the measured sweep, under store's name with the user's label; a second save writes the same file; a finished netanal lands in the session folder and is registered there; it loads back into a panel with its resonance search; the session browser types it from `file_metadata`; the measurement name is the label, and Import fills the dialog in from the container. Then the search: it finds the array through the real task and marks what it found, the settings panel is what it runs with, rejected candidates are drawn with their reason, a search updates the file the netanal is in and writes none when there is no file; the settings panel asks for exactly the finder's arguments with the finder's defaults and remembers them; the status line clears itself off the label's own slot. Then the handover: `to_catalog` names the accepted candidates at the probe amplitude, and a double-click rejects a resonance rather than deleting it, accepts a rejected one back as it was found, accepts an arbitrary frequency with `nan` measurements and a tooltip that says so, and updates the netanal file the search is in (and writes none when there is no file). The library side is in `test/tuning/test_find_resonances.py`. Still to come: the remaining QoL | `test/periscope/test_tuning_flow.py`, `test_find_resonances_settings.py`, `test_netanal_status_line.py` |
| 2 | flow step 3 through the task; the panel's redraw over a measured block (one curve per name/step/direction, carrying the entry's own frequencies and `abs(iq_counts)`), a live `partial_data` curve in the colour its finished sweep gets, and units/Normalize/batch changing the drawing without touching `module_sweeps`; Periscope's pickle against a headless one on the same seeded array (file, data and derived results); dialog as a view over `AmplitudeSchedule` (describe/validate wiring); the digest and histogram cases removed from `test_viewbox_lifetime.py` and `test_laptop_fit.py` with the panels | `test/periscope/` |
| 3 | flow step 4; fit panel reads what `fit_sweeps` wrote, and a failed fit reads as failed; a histogram tab built new over `entry["fits"]` | `test/periscope/` |
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

---

## 9. What the digest and histogram tabs showed, for when they come back

Both were deleted in stage 2 (§6, judgement call 23) rather than ported, and
both are to be designed again against the block and the catalog rather than
reassembled from this list. So this is an inventory of *what a user could see*,
not a specification: it is here so that rebuilding starts from the questions
these tabs answered instead of from whatever the new plumbing makes easy. Where
a thing was wrong, it says so — those are the parts not to bring back.

The code is in git: `detector_digest_panel.py` (1208 lines) and
`parameter_histograms_panel.py` (781) as of `8f72fc3`.

### 9.1 Detector digest — one resonator, in detail

A separate dockable window, opened by double-clicking a subplot, and also a tab
of the multisweep panel.

**Three plots, side by side.**

1. *Sweep vs frequency* — magnitude against `f - f_bias` in Hz, every amplitude
   step of that one resonator overplotted, legend per trace.
2. *IQ plane* — the same sweeps as loops, aspect locked. **Check Noise**
   over-plotted 100 live I/Q samples per detector here, so the operator could
   see where the tone actually sits on the loop it was biased on. That is the
   one control on the panel that measured something, and the reason it was
   worth having: it answers "is this detector still where I left it?" without
   leaving the panel.
3. *Bias amplitude optimisation* — `|S21|` in dB against `f - f_bias`, over the
   amplitude steps, optionally normalised to the first point of each trace.
   Double-clicking a curve here **made that amplitude the active trace** in the
   other two plots and the tables — the panel's best idea, and the one to keep:
   it made choosing a bias amplitude a thing you did by looking at the sweeps
   rather than by typing a number.

**Two parameter tables**, Parameter / Value / Description, side by side under
the plots in a splitter:

* *Skewed Lorentzian*: Status, `fr` (MHz), `Qr`, `Qc`, `Qi`, Bifurcation.
* *Nonlinear*: Status, `fr_nl`, `Qr_nl`, `Qc_nl`, `Qi_nl`, `a`, `φ` (deg),
  `I0`, `Q0`, Bifurcation. The `a` row's tooltip named 4√3/9 ≈ 0.77 as where
  the fitted resonance goes multivalued, which is `BIFURCATION_A` in
  `rfmux.tuning.fits` — the description is right, and the constant now has one
  home to read it from.

The tables' Status row and the `'nan'`-string test behind it are replaced by
`failed_because`, and the Bifurcation rows by the `BifurcationCheck` a
`BiasReport` carries. Both tables denormalised a skewed fit by hand and
re-multiplied `gain_complex` to draw the curves; `skewed_model_magnitude` and
`nonlinear_model_iq` do that now, so a rebuilt digest calls them.

**Navigation**: Previous/Next buttons and left/right arrows between resonators,
a spinbox to jump to one by number, up/down arrows between amplitude traces, a
"n of N" counter and a title carrying the resonance frequency in MHz.
Resonators were addressed by *integer detector index* throughout, which is what
`ResonatorCatalog` names replace — a rebuilt digest navigates
`catalog.names()`, and the spinbox becomes a name box.

### 9.2 Parameter histograms — the array, in aggregate

A tab of the multisweep panel, over the fitted parameters of every resonator.

* *Frequency scatter*: fitted `fr` against detector ID — the array's frequency
  layout, and the plot that shows a collision or a gap at a glance.
* *Qr, Qc, Qi histograms*: one panel each, a bin-count spinbox, and shared
  ranges computed across amplitude steps (`_compute_global_q_ranges`) so the
  bins do not move when you change step.
* An amplitude selector listing each sweep as amplitude and direction, plus an
  all-sweeps mode that stacked the histograms per step.

What to keep: the shared bins across steps, and the all-steps stacked view —
seeing `Qi` shift as drive rises is the measurement. What to drop: the detector
ID as the scatter's x-axis (frequency against name, or against index in the
catalog's own order), and `_extract_params_for_sweep`'s walk over
`results_by_detector` keyed by `"amp:direction"` strings, which is
`collect_amplitude_iterations_for` plus `entry["fits"][model]["params"]`.

Both panels are per-panel cases in `test_viewbox_lifetime.py` (the digest) and
`test_laptop_fit.py` (the histograms), removed with them; the general rules
those tests hold keep their other cases, and the panels rejoin them when they
return.
