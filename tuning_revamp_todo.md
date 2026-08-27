# Tuning revamp: running to-do list

Follow-ups deferred during the headless-tuning work on `tuning_headless_revamp`.
Things noticed while doing something else, kept here so they are not rediscovered
from scratch. Each entry says where the code is and what the decision is, so it
can be picked up cold.

Delete an entry when it lands. Design *questions* (as opposed to work items)
live in `tuning_refactor_design.md` §13 — this file is for work.

---

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
   Parameter names changed (`min_resonance_separation_hz` → `min_separation_hz`),
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
   fields are `frequency_hz` / `depth_db` / `width_hz` / `q_estimate`. Per the
   design doc's step 5 this demo is due to be rewritten against
   `tune_resonators` anyway — worth doing in one pass rather than two.
4. Then: delete the shim, and the four mocked
   `{'resonance_frequencies': …, 'resonances_details': …}` dicts in
   `test/algorithms/test_measurement_flow.py` (lines 71, 124, 250, 320).

## `fitting.test_find_resonances` is dead

`algorithms/measurement/fitting.py:862`. It plants four resonators of width
~30 kHz on a 200 kHz grid, so each dip is a fraction of a sample and the finder
returns nothing — it reports "0 out of 4" and `print`s a ✗ instead of asserting,
which is why it went unnoticed. It behaves identically on the pre-revamp code,
so this is not a regression. Delete it with the shim, or rebuild it at sampling
that resolves a dip (`test/tuning/test_find_resonances.py` does).

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

## `import rfmux` drags in Qt

`rfmux/__init__.py` eagerly does `from . import … tools`, which imports PyQt6 and
pyqtgraph, so no pure module can be imported without the GUI stack — including
everything in `rfmux/tuning/`, whose entire point is to be usable from a plain
script. Already recorded as an xfail in `test/core/test_resonators.py`; it flips
to XPASS when the import becomes lazy.
