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
  `network_analyses_find_resonances_make_resonator_catalog.md` pass throughout,
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

## Periscope's `data_callback` is two arguments too narrow for a ladder

`multiamp_multisweep` calls `data_callback(module, partial_results, step,
direction)`, where `multisweep` calls it `(module, partial_results)`. The extra
pair is not decoration: inside a ladder a consumer plotting partial data has no
way to tell which amplitude step and direction the points belong to, which is
exactly what the live multisweep grid needs.

`multisweep` itself is unchanged, so nothing is broken today — but a Periscope
task that passes its narrow callback to the driver will `TypeError` on the first
partial-data emission. To carry across when Periscope is rewired (step 5 of
`tuning_multisweep_amplitudes_plan.md`):

* `MultisweepTask.run` (`tools/periscope/tasks.py:508`) is where the loop over
  amplitude rungs lives today; it goes away in favour of one driver call.
* Its `data_callback` and the `multisweep_signals` it re-emits need the two new
  coordinates plumbed through, replacing whatever it currently derives from its
  own loop counter.
* `sweep_callback(record)` is the replacement for the task's per-rung
  bookkeeping — it carries `step`, `direction`, `amplitudes`, `factor`,
  `completed` and `total`, which is everything the progress UI reads.

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
`sweep_direction`, `sweep_amplitude` — and nothing else. `multiamp_multisweep`
lost the same three pass-throughs, `pack_results` dropped them from
`call_params`, and `RESULTS_SCHEMA_VERSION` went to 2.

Two things to bring back, deliberately, when there is something to bring them
back *for*:

1. **Re-centring across an amplitude ladder.** The point of the old
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

## Two `fits.py` guards become dead letters once the sweep shapes unify

When `multisweep` returns the same envelope as `multiamp_multisweep` — one
shape, `results[iteration][direction][name]`, with a single sweep sitting at
iteration 0 in its own direction — two errors that exist only to explain the
difference between the shapes stop being reachable. Both were deliberately
skipped when the unification landed. Decide each on its own merits:

1. **`iterations=` / `directions=` filters on a single sweep.** `_select`
   (`tuning/fits.py:651`) raises "This is a single multisweep return: it is one
   iteration in one direction, so there is nothing to select between." Once
   there is always an iteration and always a direction, `iterations=0` is
   simply true and `iterations=5` falls through to the generic "selected none
   of the N sweeps" message, which already names what is available. Probably
   just delete the guard — but it is a deliberate deletion, not a silent one.
2. **`fit_sweeps_at_bias_amplitude` on a single sweep.** The `_is_packed` check
   at `tuning/fits.py:365` refuses it outright: one amplitude per resonator,
   nothing to match. Unified, it would work and always land on iteration 0,
   because nearest-wins over a set of one. That is arguably right — a ladder
   that does not bracket the bias amplitude already behaves this way, and the
   docstring says so — but "the iteration nearest your bias amplitude" quietly
   meaning "the only one you took" is worth a decision rather than a
   side-effect. `test/tuning/test_fits.py:387` guards the current behaviour and
   goes with it.

Related: `SweepFit.iteration` and `.direction` (`tuning/fits.py:167`) are
`int | None` / `str | None` purely for the bare case, and `where` branches on
it. Both can lose the `None` once every sweep has coordinates.

## Retire the legacy resonance finder

`algorithms/measurement/fitting.py:394` `find_resonances` is now a deprecating
shim that forwards to `rfmux/tuning/find_resonances.py` and rebuilds the old
`{'resonance_frequencies', 'resonances_details'}` dict. Two callers still go
through it. Once both move, the shim and everything below it goes.

1. **Periscope netanal panel** —
   `tools/periscope/network_analysis_panel.py:683`, in
   `_run_and_plot_resonances`. It only reads `resonance_frequencies`, so it maps
   onto `ResonanceSearch.resonance_frequencies_hz` directly.
2. **`FindResonancesDialog`** — `tools/periscope/find_resonances_dialog.py`.
   The **Data Exponent** field now controls nothing: the parameter was removed
   from the finder (it was a multiplier in dB, so it scaled dips and noise
   together and could not change a candidate), and the shim accepts it only to
   keep the old call signature working. Delete the field and
   `DEFAULT_DATA_EXPONENT` in `utils.py:124`. Note the migration is not quite
   neutral for a GUI user: the old code did *not* scale the prominence threshold
   by the exponent, so at the default of 2.0 the effective dip-depth floor was
   half what the box said. Halving `DEFAULT_MIN_DIP_DEPTH_DB` (`utils.py:120`,
   currently 2.0) reproduces what the GUI used to find.
   Parameter names changed too (`min_resonance_separation_hz` → `min_separation_hz`),
   and the *meaning* changed: it is now a collision cut that removes every
   member of a too-close group, where it used to keep the tallest. The dialog
   always sends `DEFAULT_MIN_RESONANCE_SEPARATION_HZ = 1e4` (`utils.py:123`), so
   under the new rule a GUI user silently gets both members of any pair inside
   10 kHz discarded. Relabel the field so it says what it does now ("Collision
   cut", not "Min Separation"), let blank mean the 0 Hz default, and revisit
   whether 10 kHz is the right number to ship — that is a physics call about the
   readout, not a UI default.
3. **`simplified_tuning_flow`** — `reference-notebooks/Demos/`, the `.py` at
   line 267 and the `.md` companion at lines 33, 79, 375, 401 and 777. This is
   the one that reads `resonances_details`, so it becomes `.candidates`, whose
   fields are `frequency_hz` / `depth_db` / `width_hz` / `q_estimate`. Its
   `FIND_RES_PARAMS` still passes `data_exponent`, and the `.md` at line 375
   describes the finder as working on `-|S21|**data_exponent` — both stale. Per the
   design doc's step 5 this demo is due to be rewritten against
   `tune_resonators` anyway — worth doing in one pass rather than two.
4. Then: delete the shim, and the four mocked
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
3. **`add_bifurcation_flags_to_multisweep_data`** and `identify_bifurcation`
   (`algorithms/measurement/fitting.py`) were left where they are: they are
   sweep analysis rather than fitting, and `identify_bifurcation` is what
   Periscope calls today. Note the nonlinear fit's `a` answers the same
   question better — bifurcation is at `a ≈ 0.77` — so the flag may not survive
   the rewire at all.
4. **A flag on `multiamp_multisweep`** to fit as it goes. Deliberately not
   built: the driver measures, and a caller who wants the ladder fitted calls
   `fit_sweeps` on what came back. If it is ever added it should take the
   fitting arguments and hand them straight over, so there is one fitter and
   not two.

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

## `store.py` should own catalog persistence, and the notebook should point at it

Section 6 of `Demos/network_analyses_find_resonances_make_resonator_catalog.md`
currently hand-rolls `pickle.dump(catalog.to_dict(), f)`, which makes a demo the
de facto spec for how a catalog reaches disk. That is fine while nothing else
writes one, and wrong once `store.py` exists (design doc §2, §11 step 4).

Two things to carry across when it lands:

* **Pickle the dict, never the object.** `pickle.dump(catalog, f)` round-trips,
  but unpickling does not call `__init__`, so none of the catalog's invariants
  run — a file holding two resonators on one frequency restores without
  complaint, verified. It also bakes the class's import path into the file, so
  renaming or moving `ResonatorCatalog` strands every old file. `to_dict` /
  `from_dict` avoids both and gets the `schema_version` check for free.
* The design doc's §13 question — pickle now and HDF5 later, or HDF5 straight
  away — is still open, and `store.py` is where it gets answered. The notebook
  should then show `store.save(catalog, ...)` rather than the `pickle` module.

## Carry the "How to use this document" section into later notebooks

`Demos/network_analyses_find_resonances_make_resonator_catalog.md` opens with a
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
