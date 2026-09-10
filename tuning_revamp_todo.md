# Tuning revamp: running to-do list

Follow-ups deferred during the headless-tuning work on `tuning_headless_revamp`.
Things noticed while doing something else, kept here so they are not rediscovered
from scratch. Each entry says where the code is and what the decision is, so it
can be picked up cold.

Delete an entry when it lands. Design *questions* (as opposed to work items)
live in `tuning_refactor_design.md` §13 — this file is for work.

---

## Closing VS Code leaves the mock streamer running, and the notebook tests pay

Editing a `Demos/*.md` by running its paired `.ipynb` in VS Code starts a mock
CRS that streams to UDP 9876. Closing VS Code does **not** stop it — the kernel
goes away, the streamer does not. Do that a few times and several orphaned
streamers are all transmitting to the same port.

What that costs, next time the tests run:

* `test_notebooks.py::test_reference_demo_notebook[pulse_capture.md]` and
  `[simplified_tuning_flow.md]` fail, along with
  `test_measurement_flow.py::TestIntegration::test_mock_mode_execution` (in
  0.2 s — it never starts). All three call `find_streamer_conflict`
  (`rfmux/streamer.py`), which is working exactly as designed: a second
  simulation would interleave with the first and corrupt the data silently.
* The diagnosis is invisible at the pytest level. The failure reads only
  "Reference notebook X failed! See /tmp/pytest-of-…/….ipynb" — the actual
  `RuntimeError` about port 9876 is buried in the executed notebook's cell
  output, which has to be dug out of the JSON. Reproducing the reported
  failure takes several four-minute runs before it becomes clear the cause is
  not in the repository at all.
* `multisweep.md` and
  `network_analysis_find_resonances.md` pass throughout,
  because they do not stand up a streamer. That split — which notebooks call the
  guard — is the quickest way to recognise this.

Confirm it in one line before hunting anything else:

    python -c "from rfmux.streamer import find_streamer_conflict; print(find_streamer_conflict())"

Note that `ss`/`lsof` show nothing: no socket is *bound* to 9876. The orphans
are senders, which is what the guard's second probe (a short read) exists to
catch. So "the port looks free" is not evidence.

Worth fixing at the source rather than documenting forever. Options: have the
mock streamer die with the kernel that made it (a parent-death watch, or a
heartbeat the server times out on), and/or surface the conflict as a pytest
`skip`/`error` with the real message rather than an opaque notebook assertion.

## Periscope's multisweep `data_callback` is two arguments too narrow

`multisweep` calls `data_callback(module, partial_results, step, direction)`.
The last pair is not decoration: a consumer plotting partial data inside a
multi-amplitude sweep has no way to tell which amplitude step and direction the
points belong to, which is exactly what the live multisweep grid needs, and a
single sweep's `(0, "upward")` is a fact about it rather than padding.

Nothing in Periscope passes a `data_callback` to `multisweep` today — only
`take_netanal`, whose callback now hands over `(module, partial)` with the
trace's keys, the same idea one level up — so this is not broken right now. It
becomes load-bearing when Periscope is rewired (step 5
of `tuning_multisweep_amplitudes_plan.md`, a plan that landed and was removed 2026-09-08; it is in git history):

* `MultisweepTask.run` (`tools/periscope/tasks.py:632`) loops over amplitude
  steps and directions itself, calling `crs.multisweep` once per sweep. That
  whole loop goes away in favour of one call passing `amp=AmplitudeSchedule(…)`
  and `sweep_direction=("upward", "downward")`.
* Its `data_callback` and the `multisweep_signals` it re-emits need the two new
  coordinates plumbed through, replacing what it currently derives from its own
  loop counter.
* `sweep_callback(record)` is the replacement for the task's per-step
  bookkeeping — it carries `step`, `direction`, `amplitudes`, `factor`,
  `completed` and `total`, which is everything the progress UI reads.
* `progress_callback` now runs across the whole call rather than resetting per
  sweep, so the task's progress bar no longer has to be rescaled by hand.

## multisweep's measurement loop is nearly untested

`test/algorithms/test_multisweep.py` covers input resolution only, by design —
"the measurement loop below it needs a board and is not exercised here". The one
exception is now `test/algorithms/test_multisweep_channels.py`
(`slow_acquisition`), which drives the loop against a MockCRS to pin down which
channels the sweep may silence.

That test exists because the behaviour was changed and needed a guard, not
because the loop is now covered. Still unexercised: NCO region splitting (the
`MAX_NCO_SPAN_HZ` cut and the no-phase-stitching seam between regions). The
MockCRS route is cheap once resonators are generated at known frequencies —
`crs.generate_resonators({"num_resonances": n, "auto_bias_kids": False})`
returns the list.

(The recalculation arithmetic, `rotate_saved_data` and `apply_df_calibration`
used to be on this list. They are gone from the macro — see below — so there is
nothing left to cover.)

## multisweep measures, and does nothing else

The macro no longer rotates, re-centres or df-calibrates. Removed: the
`bias_frequency_method`, `rotate_saved_data` and `apply_df_calibration`
arguments, the `_get_recalculated_center_freq` helper, and the whole per-NCO-
region TOD acquisition that fed the rotation. A section entry is now
`channel`, `frequencies`, `iq_counts`, `iq_volts`, `original_center_frequency`,
`sweep_direction`, `sweep_amplitude` — and nothing else. The then-separate
`multiamp_multisweep` lost the same three pass-throughs, its packer dropped them
from `call_params`, and `RESULTS_SCHEMA_VERSION` went to 2. (That second macro
has since been folded into `multisweep` itself, and the two packers into
`pack_multisweep` — schema 5.)

Two things to bring back, deliberately, when there is something to bring them
back *for*:

1. **Re-centring across amplitude steps.** The point of the old
   recalculation was to let a sweep centre follow a resonance that moves
   between amplitude steps. When it returns it adjusts the *sweep centre* of
   the next step, decided by whatever analysis found the dip — not a
   `bias_frequency` reported out of a sweep. The bias frequency lives in the
   catalog's `BiasPoint`.
2. **df calibration**, off the sweep it was fit to. The fitting layer it comes
   from now exists — `rfmux/tuning/fits.py` — so this is unblocked; see the
   fitting entry below.

### Consumers still reading the old contract

None of these is a regression from this change alone — all of them predate it
and are already on the list to be rewired — but they now fail sooner and more
loudly, so check them off when their rewrite lands:

* **Periscope's `MultisweepTask`** (`tools/periscope/tasks.py:669`) passes
  `bias_frequency_method` and `rotate_saved_data` straight into
  `crs.multisweep`; `app_runtime.py:2365` builds the same pair. Both are now a
  `TypeError`. `multisweep_dialog.py` has the checkbox and combo that produce
  them, and `multisweep_panel.py` plots `iq_complex`. Part of step 5.
* **The legacy analysis stack** — `fitting.fit_skewed_multisweep` and
  `fitting_nonlinear.fit_nonlinear_iq_multisweep` have been replaced by
  `rfmux/tuning/fits.py` and now carry a `DeprecationWarning`; they still read
  `iq_complex`, so they work on pre-schema-2 pickles and nothing newer. See the
  fitting entry below for what is left. `bias_kids` still reads `iq_complex`
  and `bias_frequency` and has not been touched.
* **`reference-notebooks/Demos/simplified_tuning_flow.{py,md}`** passes the
  removed kwargs and then indexes results by integer, which the catalog revamp
  had already broken. `test/algorithms/test_measurement_flow.py` keeps passing
  because it mocks `crs.multisweep`, so CI will not catch either.

All of the above now have a second layer to absorb: a sweep no longer returns
`{name: entry}` at all. It returns `{module_id: {..., "results": {iteration:
{direction: {name: entry}}}}}` — see `tuning_sweep_result_shape_plan.md` in git history — the plan landed and was removed 2026-09-08. So
each rewrite starts with `sweeps[crs.module[m].index()]["results"][…]`, or
better, the readers in `rfmux/tuning/sweep_results.py`. `bias_kids.py:231`
reaching for `multisweep_results['results_by_detector']` is the oldest of these
and pre-dates even the catalog revamp.

A trap worth knowing about while doing any of it: `test_multisweep_channels.py`
is in the `slow_acquisition` tier, which `addopts` deselects by default, so it
went on passing vacuously while asserting on a return shape that no longer
existed. Anything that changes what a macro returns needs
`pytest -m slow_acquisition test/` as well as a default run.

## Retire the legacy resonance finder

`algorithms/measurement/fitting.py:394` `find_resonances` is now a deprecating
shim that forwards to `rfmux/tuning/find_resonances.py` and rebuilds the old
`{'resonance_frequencies', 'resonances_details'}` dict. One caller still goes
through it. Once it moves, the shim and everything below it goes.

Periscope's two — the netanal panel's `_run_and_plot_resonances` and
`FindResonancesDialog` — are done, in the stage 1 commit that put
`FindResonancesTask` and `FindResonancesSettingsPanel` in their place. The
migration notes recorded here are settled with them: the settings panel reads
its defaults out of `find_resonances`' signature, which drops Data Exponent
and halves the dip-depth floor to the library's 1.0 dB (the old dialog said
2.0 and the old code did not scale the threshold by the exponent, so its
effective floor was 1.0); and the separation is now labelled "Collision cut",
defaulting to the library's 0 Hz rather than the dialog's 10 kHz, which under
the new every-member rule would have silently discarded both halves of any
pair inside 10 kHz. Whether a non-zero collision cut is worth shipping is
still a physics call about the readout, now made in the panel by whoever is
looking at the array.

1. **`simplified_tuning_flow`** — `reference-notebooks/Demos/`, the `.py` at
   line 267 and the `.md` companion at lines 33, 79, 375, 401 and 777. This is
   the one that reads `resonances_details`, so it becomes `.candidates`, whose
   fields are `frequency_hz` / `depth_db` / `width_hz` / `q_estimate`. Its
   `FIND_RES_PARAMS` still passes `data_exponent`, and the `.md` at line 375
   describes the finder as working on `-|S21|**data_exponent` — both stale. Per the
   design doc's step 5 this demo is due to be rewritten against
   `tune_resonators` anyway — worth doing in one pass rather than two.
2. Then: delete the shim, and the four mocked
   `{'resonance_frequencies': …, 'resonances_details': …}` dicts in
   `test/algorithms/test_measurement_flow.py` (lines 71, 124, 250, 320).

## Retire the legacy fit walkers

`rfmux/tuning/fits.py` is the fitting layer. It fits sweeps that already exist,
by hand, and writes each model's results into the sweep entry's `fits` subdict
keyed by model — `skewed`, `nonlinear`, `circle`. The per-trace maths moved
there wholesale; `algorithms/measurement/fitting.py` and `fitting_nonlinear.py`
re-export it and keep only their old dict-walking API, now deprecated. Also
deleted on the way past: the ad-hoc `test_*` / `run_all_tests` functions in both
modules, including the dead `fitting.test_find_resonances` that used to have its
own entry here — it planted 30 kHz dips on a 200 kHz grid and `print`ed a ✗
instead of asserting. `test/tuning/test_fits.py` covers the replacement.

What is left:

1. **Periscope.** `MultisweepTask._apply_fitting_analysis`
   (`tools/periscope/tasks.py:818`) fits inline during the sweep, through the
   deprecated walkers, and writes flat keys that `detector_digest_panel`,
   `parameter_histograms_panel` and `multisweep_dialog` all read. It becomes a
   call to `fit_sweeps` on a finished multisweep, on a button rather than on
   every sweep — `fit_sweeps` takes a `progress_callback(completed, total)` for
   the progress UI. The panels then read `entry["fits"][model]["params"]`, and
   the model curves they cache come from `skewed_model_magnitude` /
   `nonlinear_model_iq` instead of a stored array. Part of step 5.
2. **df calibration**, which §11 of the design doc wants off the fit — the
   piece `multisweep` gave up when it stopped calibrating on the way past.
   Nothing in `fits.py` computes it yet.
3. **`identify_bifurcation`** (`algorithms/measurement/fitting.py`) is still
   there, because Periscope calls it (`tools/periscope/tasks.py:687`). It is
   sweep analysis rather than fitting, and `tuning/bias.py` now does the job
   properly with `bifurcated_by_derivative` / `bifurcated_by_hysteresis`. Note
   the nonlinear fit's `a` answers the same question better — bifurcation is at
   `a ≈ 0.77` — so the flag may not survive the rewire at all.

   Its wrapper `add_bifurcation_flags_to_multisweep_data` is **gone**: it had no
   callers and walked `results_by_iteration`, a shape retired at
   `RESULTS_SCHEMA_VERSION = 2`. It was also the last of the ad-hoc
   `pickle_filepath_or_data` / `output_pickle_filepath` file handling, which
   `store.py` replaces.
4. **A flag on `multisweep`** to fit as it goes. Deliberately not built: the
   macro measures, and a caller who wants every amplitude step fitted calls
   `fit_sweeps` on what came back. If it is ever added it should take the
   fitting arguments and hand them straight over, so there is one fitter and
   not two.

## Calibrate the bifurcation thresholds against a real array

**2026-09-08, on the standard simulated array (`rfmux/mock/standard_array.py`,
`test/notebooks/test_standard_mock_array.md`):** both detectors fire too
early. Over the schedule `multiplicative(0.5, 8, 5)` the nonlinear fit puts every
one of the eight resonators' bifurcation between steps 3 and 4 (`a` ≈ 0.4 then
≈ 0.85–0.9), yet `derivative` picks step 0–2 on seven of eight and
`hysteresis` scatters from step 0 to 4; `both` agrees with the fit on 0 of 8,
and three resonators come back flagged "the quietest amplitude measured was
already bifurcated". The paragraph below about the simulator's two passes being
identical is no longer true: main's mock now has TLS 1/f frequency wander on
by default (`tls_noise_enabled`, ~1e-7 df/f), so upward and downward traces
taken seconds apart differ by a drift the hysteresis test reads as a jump.
The derivative test is presumably reading the readout noise (`udp_noise_level`
11 counts) on a 5 dB dip. Next steps, in this order: rerun the notebook's
comparison with `tls_noise_enabled=False` and with `nqp_noise_enabled=False`
to attribute each detector's false positives; then decide whether the
thresholds move, the metrics change, or the fitted `a` becomes the check the
post-merge plan describes (survey §9.4). The notebook's comparison table is
the harness for this; nothing else needs building first.


`rfmux/tuning/bias.py` ships two tests for spotting a bifurcated amplitude step
— plus `both`, which runs them and takes either verdict, so it has no threshold
of its own and inherits whatever these two are calibrated to. Neither default
has been checked against a cryostat.

* **`derivative`** carries the GUI's long-standing prominence bar, restated as
  `spike_prominence_factor=0.5` multiplying the span of the arc speed. The GUI
  *divided* the span by a `spike_prominence_factor` of 2.0, so turning its knob
  up made the test more sensitive; same argument name here, reciprocal value,
  identical threshold. Anything quoting the GUI's 2.0 needs converting rather
  than copying — a 2.0 passed to this one asks for a spike twice the whole
  range, which nothing clears.
  The GUI's companion `spike_height_factor=3.0` is *not* ported: the prominence
  bar was the binding one at every amplitude on the shipped sweep, so the height
  bar only ever added a knob that never decided anything. On the sweep shipped with
  `Demos/bias_finding.md` it behaves: `metric/threshold` runs 0.4–0.6 on the
  clean amplitude steps and 1.8–1.9 on the jumped ones, and all four resonators
  flip between 2 mV and 4 mV. But the margin is a factor of two, not orders of
  magnitude, and the one step that sits above 1.0 without being called
  bifurcated (R0003 at 1 mV, 1.24) is held back only by the adjacency rule.
  Two known ways to fool it, both documented on the function: a sweep too
  coarse to resolve the resonance (a dip crossed in two samples *is* a
  discontinuity), and a sweep with no resonance in it at all (the largest noise
  excursion then sets the scale a threshold is a fraction of).
* **`hysteresis`** — new here, no ancestor to inherit a number from —
  defaults to `max_discrepancy=0.25` loop radii. It cannot be exercised against
  the simulator at all: MockCRS computes each sweep point independently, so its
  upward and downward traces are identical up to noise and the metric sits at
  0.03–0.08 at every amplitude, clean or jumped. That is a noise floor with
  roughly a factor of three of headroom under the default, and nothing more.

Both report `metric` and `threshold` on every `BifurcationCheck` precisely so
this can be settled by reading them across the amplitude steps of a resonator
that is known to bifurcate. Note that `derivative`'s `metric` and `threshold`
are not quite the same quantity — the metric is the largest positive jump,
measured from zero, and the threshold is compared against a *prominence*,
measured from the spike's own neighbourhood — so the ratio is a margin and not
a verdict. Reporting the prominence itself would make the pair commensurate and
is worth doing if the ratio turns out to be what a real calibration is read
off. Worth settling before either default is quoted anywhere
as a recommendation, and worth reconciling with the nonlinear fit's `a`
(bifurcation at ≈ 0.77) — see the fit-walker entry above, which asks the same
question from the other side.

## Decide whether the width window earns its keep at real sampling

`tuning/find_resonances.py` bounds dip width by `frequencies / min_Q` and
`frequencies / max_Q`, converted to samples. On a realistic netanal — 5,000
points over 2.35 GHz, so 470 kHz spacing — a Q=1e4 resonator at 1.5 GHz is
150 kHz wide, well under one sample. Both bounds `ceil` to 1, so the window is
`[1, 1]` and the test is very nearly a no-op; it only bites on oversampled
sweeps. Options: keep it as harmless, warn when the sweep cannot resolve the
requested Q range, or drop the parameters in favour of something that means
something at netanal resolution. Needs a look at real data before choosing.

## Resonance-finder plots

Not ported from hidfmux's `analysis/find_resonances.py`, which draws the sweep
with candidates circled and a per-candidate detail grid with the peak properties
overlaid — both genuinely useful for tuning the finder's parameters by eye. They
belong in Periscope (or a notebook), not in the analysis module.
`ResonanceSearch` already carries what a plotter needs: `frequencies_hz`,
`magnitude_db`, and each candidate's `index` into them, plus `rejected` with
reasons so discarded candidates can be drawn differently.

## `store.py` owns persistence; what is left is a run-level layout

Done. `rfmux/tuning/store.py` writes one dated folder per day
(`ipy_session_YYYYMMDD`) under a configurable root, names files
`{type}_{date}_{time}_{label}.pkl`, and stamps a `file_metadata` block into each
module's block of what it writes. The drivers and analyses save by default
through `save=` / `label=`; `rfmux/config_template.yaml` and `$RFMUX_DATA_DIR`
say where and whether. Both rules this section was holding onto survived: the
dict goes in the file, never the object, and the `schema_version` check comes
along with it. §13's question is answered for now as *pickle, behind
`store.py`*, so the format can move to HDF5 later without touching callers.

What is still missing is a level up: a **folder per tuning run** that ties a
catalog to the sweeps and the settings that produced it. Today each measurement
is a file that knows its own name and nothing about its siblings, so
reconstructing "the run that produced this catalog" means reading timestamps.
Design doc §2 and §11 step 4 describe the folder; `store.py` is where it goes.

One pairing that used to need it does not any more: `find_resonances_in_netanal`
writes its search into the netanal, under `resonance_search` beside the trace it
searched, and saves the netanal back over its own file. Analysis that annotates
a measurement now looks the same wherever you meet it — one module's output in,
the annotation written into it, the file it came from updated — so `fit_sweeps`
and the resonance finder are the same shape of call, and there is no
`find_resonances_*.pkl` to keep matched with the netanal it came from.

Two smaller follow-ups it should pick up:

* **The legacy readers** still inline in `app_runtime.py` — top-level
  `bias_kids_output`, integer detector keys, missing `iq_volts`, flat fit keys —
  which the design doc lists as `store.py`'s (§7).
* **A wheel built from a configured checkout carries `rfmux/config.yaml`**,
  because `wheel.packages = ["rfmux"]` copies the whole package directory.
  Gitignored, so it never reaches a commit, but `scikit-build-core`'s
  `wheel.exclude` would close it properly — it needs >=0.5.0, and
  `pyproject.toml` currently floors at 0.3.3.

## Rewire Periscope onto the netanal container shape

`take_netanal` now returns `{module_id: {schema_version, measurement, module,
call_params, results}}` like every other driver, with the trace *at* `results`
and its arrays named `iq_counts`/`iq_volts`. It also takes `sweep_direction`,
one direction per call, and a downward netanal comes back descending. That is a
breaking change to the shape Periscope reads, and Periscope was deliberately
left on the old one. This is stage 1 of `periscope_port_roadmap.md`, which
stages the whole port; what follows is the netanal part of it.

* ~~`tools/periscope/tasks.py:479` unpacks `result['frequencies']`,
  `result['iq_complex']` and `result['phase_degrees']` straight off the top.~~
  Done. `NetworkAnalysisTask` reads the module's trace out of the container and
  emits it, live and on completion, as one
  `data_update(module, amplitude, trace)`; the panel keeps it in
  `netanal_traces` keyed by probe amplitude and takes `abs`/`np.angle` at draw
  time. `take_netanal`'s `data_callback` was changed to match: it passed
  `(module, freqs, amps, phases)` and now passes `(module, partial)` with the
  trace's own keys, so no measurement algorithm computes magnitude or phase.
* `detector_digest_panel.py` reads `iq_complex` off stored sweep entries
  (`:653`, `:810`), which is the multisweep half of the same change (stage 2).
* `network_analysis_panel.py` and `network_analysis_export.py` carry their own
  `parameters`/`modules` payload, which `call_params` now duplicates. Still
  open: `build_export_dict` walks the traces but writes that payload, and the
  loader in `app.py` reads it back. `store.save`/`store.load` replace both, and
  that is the rest of stage 1's file work.

Phase is `np.angle(iq_counts)` at the point of use; the module number is in
each module's output rather than passed alongside. `store.py`'s `_blocks` has
already lost its list case, which existed for the old netanal return and had no
producer left; saving a list now raises and says to key it by module.

Two smaller things deferred with it:

* **`Demos/simplified_tuning_flow.py`** still reads the old flat shape, and
  `test/algorithms/test_measurement_flow.py` drives it with `AsyncMock`s that
  return the old shape too. The mocked tests still pass, since they never reach
  the real driver; `test_mock_mode_execution` (slow_acquisition) does and will
  fail until the demo is rewired.
* **`take_netanal`'s `rotate_phase_to_0`** is the last rotation the measurement
  applies to its own data — the NCO stitch is gone. It is recorded in
  `call_params` so a file says whether it ran, but by the same argument that
  moved rotation out of `multisweep` it belongs in an analysis, not here.

## Carry the "How to use this document" section into later notebooks

`Demos/network_analysis_find_resonances.md` opens with a
"How to use this document" section covering the things every reader of a jupytext
demo trips over: run the cells in order, no saved outputs, how to open a `.md` in
JupyterLab versus VS Code (pair it with jupytext sync), and how to tell which
copy of rfmux your kernel actually imported. Reuse it in the notebooks this
revamp still needs, rather than rewriting it each time — and if it changes, keep
the copies in step. `pulse_capture.md` has an earlier version of the same
section, worth folding the kernel and VS Code points into.

## `import rfmux` drags in Qt

`rfmux/__init__.py` eagerly does `from . import … tools`, which imports PyQt6 and
pyqtgraph, so no pure module can be imported without the GUI stack — including
everything in `rfmux/tuning/`, whose entire point is to be usable from a plain
script. Already recorded as an xfail in `test/core/test_resonators.py`; it flips
to XPASS when the import becomes lazy.

## Cable length as an analysis, never a board write

Periscope used to set the cable length on the board twice: before every netanal
(`NetworkAnalysisTask`, from a dialog field) and again at the end of the
cable-delay unwrap, which fitted a residual delay off the trace and wrote the
new length back. Both are gone as of stage 1 of `periscope_port_roadmap.md`,
under §2 rule 9 — the board's cable length rotates the phase of everything it
reads, and this branch changes no DAC or ADC phase. Nothing calls
`SetCableLengthTask` or `SetCableLengthSignals` in `tools/periscope/tasks.py`
any more; they are left in place, unwired.

The unwrap itself stays and is still useful: it fits the delay and adjusts the
*displayed* phase, which is the honest version of what it was doing. If cable
length comes back it comes back like that — a helper that removes a known phase
slope from measured data, taking the length as an argument and returning
corrected phase, so a notebook and Periscope get the same correction and the
board is never touched. `Demos/simplified_tuning_flow.py:258` still calls
`crs.set_cable_length`; it is on the roadmap's delete-and-rewrite list.
