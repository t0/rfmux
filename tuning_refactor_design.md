# Resonator tuning: a plain-Python design sketch

Working document. Companion to `res_info_dict_overview.md`, which describes how
the tuning flow works **today** (branch `mr_multisweep_section_amplitudes`,
where most of the logic lives in the Periscope GUI).

Goal: move the tuning functionality out of the GUI and into plain Python, so a
user writing their own script has the same capability the GUI has — following
the layering the `buffer_exploration` branch uses for pulse capture.

---

## 0. Status — where the code has overtaken this document

The data model is built, in `rfmux/core/resonators.py` with tests in
`test/core/test_resonators.py`. It departs from the sketch below in ways worth
reading before the rest of this document:

* **Location.** The types are `BiasPoint`, `Resonator` and `ResonatorCatalog`,
  in `rfmux/core/`, not in `rfmux/tuning/` as §2 and §3 propose. The data model
  is useful beyond tuning, so it belongs in `core`; `rfmux/tuning/` is still
  where the *functions* that operate on it will live. The hardware-map ORM row
  that used to be called `rfmux.core.schema.Resonator` is now
  `HWMResonator` — renamed so the two are visibly different things, and
  because it may be removed outright later. Everything named after it moved
  with it: the YAML tag is `!HWMResonators`, the wafer key is
  `hwm_resonators:`, the ChannelMappings CSV column is `hwm_resonator`, the
  relationship attributes are `hwm_resonator` / `hwm_resonators`, and the table
  is `hwm_resonators`. Existing hardware maps need those four edits. The name
  `Resonator` in `rfmux/core/__init__.py` is consequently free; the new type is
  still not re-exported there, so that `rfmux.Resonator` fails loudly rather
  than silently resolving to a different class than a caller expects.
* **Identity is a `name`, not a random code.** Resonators are `R0001…` in
  frequency order, so §3's "the 4-letter code is random and regenerated" problem
  does not arise. `ResonatorCatalog.rekey()` is therefore not implemented.
* **`BiasPoint` is a real type, and it is frozen.** The operating point and the
  calibration measured at it are one object, so §3's list of loose per-detector
  calibration fields is now a single `Resonator.bias`. Stale calibration is
  unrepresentable rather than merely avoided.
* **One frequency per resonator, and it lives on the bias point.** §3's
  `bias_frequency` *and* a separate sweep centre have collapsed into
  `Resonator.bias.frequency_hz`. `from_frequencies(freqs, module, amplitude)`
  seeds it from `find_resonances`; multisweep and bias finding refine it; and
  every one of those lands on the tone grid as it is set, rather than waiting
  for apply-bias to quantize. There is no `center_frequency_hz` field, because the
  sweep centre is multisweep's decision and a second frequency is a second
  thing to keep in agreement. A consequence: `amplitude` is required at
  construction — a probe power is a measurement choice with no safe default.
* **`Resonator.bias` is required, so there is no unbiased state.** A resonator
  we cannot name a frequency for is not a resonator we know about. §3's
  `biased()` filter and `clear_biases()` are therefore not implemented, and
  neither `to_dict` nor the CSV has a null-bias branch. To forget the tuning
  you re-seed from `find_resonances`; there is nothing to clear to.
* **No sweep storage at all.** §4's `SweepSet` / `SweepEntry` are not built and
  are not planned in this form. Analysis reduces a sweep to the scalars that
  belong on a `BiasPoint`; the traces stay with the caller. This keeps the
  catalog cheap to copy — the threading rule in §9 depends on that.
* **The catalog is module-scoped**, which resolves §13's open question: one
  `ResonatorCatalog` per module, rather than a module-agnostic catalog or an
  optional per-`Resonator` module field. Today's `multisweep` shares one
  registry across modules, so that is an interface change still to make.
* **Provenance deferred.** A `source` field recording how a bias was arrived at
  was considered and left out until there is a caller that needs it.

§10's table of fixed bugs describes the state of `mr_multisweep_section_amplitudes`.
Those defects are not present on this branch — the registry layer never existed
here — so they are avoided by construction rather than repaired.

**Settled: the tone grid is `transferfunctions.BASE_FREQUENCY`** (≈596.046 Hz).
There is now one definition, and `BiasPoint`, `bias_kids.py` and
Periscope's `apply_bias_output` all use it. The three sites previously carried
their own copy of the literal `298.0232238769531` — half of `BASE_FREQUENCY` —
so quantization is a factor of two coarser than it was, and the maximum shift
from the requested frequency rises from ≈149 Hz to ≈298 Hz. `BASE_FREQUENCY`
still carries a "TODO: verify still appropriate" comment in
`transferfunctions.py`; it is now the single line to change if the firmware
says otherwise.

---

## 1. The pattern we are emulating

From `buffer_exploration`, pulse capture is split into:

* `rfmux/pulse_capture/` — a package beside `streamer/` and `mock/`.
  Imports no Qt and no `CRS`; it never talks to a board. Commit `e5f1c44`
  states the intent outright: *"pulse capture is not in the algorithms layer
  any more."*
* `rfmux/algorithms/measurement/trigger_capture.py` — a thin `@macro` front
  door. Validates arguments, talks to the board, drives the library, returns a
  result dataclass.
* `rfmux/tools/periscope/pulse_capture_task.py` — a QThread that only queues,
  feeds and re-emits. The GUI is a *third caller*, not a second implementation.

| Role | Pulse capture | The rule |
|---|---|---|
| Engine | `detection.py` | State machine, ring buffers. No CRS, no Qt. |
| Pure per-event math | `analysis.py` | One definition of each derived scalar — the writer, the histograms and the GUI all derive from it "so they can never disagree" |
| Running products | `accumulators.py` | O(1) per event |
| Persistence | `hdf5.py` | One writer / reader pair |
| I/O adapters | `sources.py` | `SlowIngest` is shared by the socket loop *and* the GUI tap, "so the two cannot drift" |
| Config | `PulseCaptureConfig` | Physical units in; `describe(rate)` and `validate(rate)` out |
| Front door | `crs.trigger_capture()` | One call, result dataclass back |
| GUI dialog | `pulse_capture_settings_dialog.py` | *"Thin view over `PulseCaptureConfig`: every derived number and every rule comes from the config object"* |

**Deliberately not adopted:** a session/orchestrator object. Pulse capture
needs one because it has a live lifecycle (noise training → capturing → stop)
driven by a sample stream. Tuning does not: each step is a discrete call whose
inputs and outputs are data. An **output folder** plus a passed-along catalog
covers what we need, and the user decides when to start a new folder.

---

## 2. Proposed layout

```
rfmux/tuning/
    catalog.py      Resonator + Catalog — the array bookkeeping
    find_resonances.py  netanal sweep → candidate resonances (built)
    sweeps.py       the heavy side: sweep entries and sets
    bias.py         bias-point analysis (pure; lifted from bias_kids.py)
    fits.py         fit dispatch policy
    rotation.py     IQ rotation angle from a timestream (pure)
    config.py       SweepConfig / BiasConfig / FitConfig
    store.py        the output folder: writer, reader, legacy migrations
```

Board-facing code stays where it is:

```
rfmux/algorithms/measurement/
    multisweep.py       (takes/returns a Catalog instead of a bare dict)
    bias_kids.py        (legacy; superseded by tuning/bias.py + operation/apply_bias.py)
    tune_resonators.py  (new: the one-call front door)

rfmux/algorithms/operation/
    apply_bias.py       (new: puts a Catalog's tones on the air)
```

`operation` is a second algorithms category, added when `apply_bias` was
written: a measurement asks the array a question and returns the answer, an
operation tells the array how to be and returns nothing. Future array-control
work (rotation, clearing, NCO planning) goes here rather than beside the
sweeps.

---

## 3. `Resonator` and `Catalog`

The core of the change. `res_info_dict` becomes two real types: a `Resonator`
per detector, and a `Catalog` holding the collection.

Today the registry is an untyped dict whose invariants live only in comments
spread across six files, and every consumer re-derives them.

### `Resonator`

Roughly the current per-code fields, typed:

```python
@dataclass
class Resonator:
    code: str                       # 4-letter identity, e.g. "AXQR"
    channel_number: int             # 1-based; permanent hardware binding
    bias_frequency: float           # Hz — sweep centre, then refined, then quantized
    bias_amplitude: float           # normalized DAC units, 0-1

    name: str | None = None         # user-facing label (NEW)

    bias_found: bool = False
    df_calibration: complex | None = None   # Hz/V — sole source for df units
    iq_rotation_angle: float | None = None  # radians
```

**Decided: the Find-Bias diagnostics stay with the sweep set, not here.**
`dI_df`, `dQ_df`, `bifurcated_at` and `nonlinear_fit_*` are results of one
analysis of one sweep set, not permanent properties of the detector, and
keeping them out avoids cluttering the catalog. This also matches where they
already partly live: `find_bias_points` writes a `bias_finding` sub-dict onto
the selected sweep entry today (`bias_kids.py:779`), holding the same
quantities plus the diagnostic arrays and the settings used.

The dividing line that falls out of this: **the catalog holds identity, the
operating point, and calibrations that downstream measurements need**
(`df_calibration` for df units, `iq_rotation_angle` for Rotated IQ). Everything
derived from a particular sweep lives with that sweep. Reversible — if a
diagnostic turns out to be wanted array-wide, promoting it later is additive.

### `Catalog`

```python
@dataclass
class Catalog:
    resonators: dict[str, Resonator]      # keyed by code
    module: int | None = None
    # provenance: what produced this, and from where
```

What it should own — each item is currently open-coded in the GUI, usually more
than once:

* **Code minting and channel allocation.** Today in `multisweep.py:33` plus a
  separate 1..N assignment in `app.py:1586`.
* **Ordering helpers** — `by_channel()`, `by_frequency()`, `biased()`.
  Replaces four hand-written
  `sorted(..., key=lambda c: ...get('channel_number', 0))` sites.
* **Projections** — `df_calibration_by_channel()`, `rotation_by_channel()`.
  This is where a live bug sits: `df_calibration` is re-keyed code→channel at
  `multisweep_panel.py:3210` and works; rotation angles are published keyed by
  code (`app.py:1803`) but looked up by channel (`app_runtime.py:58`), so
  Rotated-IQ mode silently falls through to plain volts. One projection method
  and the whole class of bug goes away.
* **Merge** — `catalog.update_from(other)`, replacing the field-by-field loop in
  `_find_bias_completed`.
* **Re-key instead of discard.** A re-run with custom frequencies currently
  *drops* the registry (`multisweep_panel.py:2090`) and mints fresh codes, so
  the detectors lose their identity and all accumulated calibration.
  `catalog.rekey(frequencies)` should keep codes and channels and record that
  the mapping moved.
* **Explicit state transitions** — `catalog.clear_bias_results()` for the
  re-run rule that currently lives in a comment at `multisweep_panel.py:2068`
  ("clear `bias_found` but keep `bias_amplitude`, because the multisweep
  Option-B path needs it"). Written down as a method, it becomes testable.
* **Serialization** — `to_dict()` / `from_dict()`, so `store.py` and the pickle
  format have one definition.

### Naming and identity

The 4-letter code is currently the only handle a detector has, and it is random
and regenerated on any path that goes through multisweep's Option A. Adding
`name` to `Resonator` and making re-keying explicit is the concrete step toward
real array bookkeeping — being able to say "this is the same detector I
measured last week" rather than "this is the fourth code in channel order".

Open question: should `Catalog` be able to load a name map (CSV of
frequency → name) and apply it? That is how names would first get attached to a
freshly found array.

---

## 4. `sweeps.py` — the heavy side

The sweep data stays a separate structure keyed by the same codes
(`{code: {iteration: entry}}` — `MultisweepPanel.results_by_detector` today).
`SweepSet` / `SweepEntry` wrap it with accessors that retire the fallback chains
now scattered through the GUI:

* `entry.center_frequency` — replaces
  `bias_frequency` → `sweep_center_frequency` → legacy `original_center_frequency`
* `sweeps.sorted_by_amplitude(code)` — the amplitude sort done inline in three
  places
* `sweeps.entry_at_bias_amplitude(code, catalog)` — the lookup inline at
  `tasks.py:1171`
* backfill of `iq_volts` / `sweep_power_dbm` for older files, inline at
  `app_runtime.py:1368`
* format migration for legacy flat fit keys

It also **owns the Find-Bias diagnostics** (§3): the per-entry `bias_finding`
sub-dict — `dI_df`/`dQ_df` at the bias point, `bifurcated_at`, the arc-length
speed arrays, the spike thresholds and peak indices, the settings used — plus
the `fits` sub-dict. Both are products of analysing one sweep, so they belong
beside it. `SweepSet` should expose them through accessors rather than leaving
callers to dig through nested dicts, which is what the IQ-Derivatives and Fit
Results tabs do today.

This is the `analysis.py` role: one definition of every derived quantity, so the
plot grid, the fitter, the exporter and the digest panel cannot disagree.

---

## 5. `config.py`

Modelled on `PulseCaptureConfig`: physical units in, derived numbers and
validation out.

```python
@dataclass
class BiasConfig:
    spike_prominence_factor: float = 2.0
    spike_height_factor: float = 3.0
    max_deriv_distance_hz: float | None = None     # absolute…
    max_deriv_distance_fraction: float = 0.5       # …or fraction of the span
    reference_freq_source: str = "bias_frequency"
    bias_freq_method: str = "iq_derivative"
    fit_selected_amplitude: bool = True

    def distance_hz(self, span_hz) -> float: ...
    def describe(self, span_hz) -> dict: ...
    def validate(self, span_hz) -> list[tuple[str, str]]: ...
```

Also `SweepConfig` (span, npoints, nsamps, direction, amplitude schedule) and
`FitConfig` (which fitters, which amplitude policy).

What this deletes from the GUI:

* the fraction→Hz resolution with its `warnings.warn` fallback, currently in
  `multisweep_panel.py:2900`
* the ~200 lines of amplitude-schedule construction in
  `MultisweepDialog.get_parameters` — including multiplicative scaling, which
  today reaches into `self.params['res_info_dict']` from inside a dialog
  (`multisweep_dialog.py:1504`). As `SweepConfig.amplitude_arrays(catalog)`, a
  script author gets the same ladder the GUI builds.
* per-dialog validation rules; dialogs render `describe()` live and show
  `validate()` through the existing issue banner, exactly as the pulse-capture
  settings dialog does.

---

## 6. Pure analysis modules

* **`bias.py`** — `find_bias_points`, `detect_bifurcation_derivative`,
  `find_max_derivative_frequency`, `compute_*_derivative_spline`. Lifted
  wholesale out of `algorithms/measurement/bias_kids.py`, which today mixes
  pure analysis with one hardware call. Signature becomes
  `find_bias_points(sweeps, catalog, config) -> BiasReport` — returning results
  rather than mutating in place, with the caller deciding when to merge.
  (`apply_bias` stays in the algorithms layer; it is hardware programming and
  belongs there.)
* **`fits.py`** — the three amplitude policies now embedded in
  `RunFitsTask.run` ("all", "index N", "bias amplitude") as one entry point
  over the existing `fitting.py` / `fitting_nonlinear.py`.
* **`rotation.py`** — angle from an IQ timestream. The *capture* of that
  timestream stays a board call in the algorithms layer.

---

## 7. Persistence: an output folder, no session object

Instead of a session object holding state, the state is:

* a `Catalog` — passed from call to call, and saved to the output folder
* sweep data files — written to the same folder
* configs — recorded alongside, so a run is reproducible

`store.py` owns:

* the folder layout and file naming
* `save_catalog()` / `load_catalog()`
* `save_sweeps()` / `load_sweeps()`
* the legacy readers currently inline in `app_runtime.py` (top-level
  `bias_kids_output`, integer detector keys, missing `iq_volts`, flat fit keys)
* a version stamp on the payload

The user decides when to start a new folder; that decision is what would
otherwise have been "starting a new session". Periscope's `session_manager`
already works this way (auto-export into a session directory, overwriting the
file a panel was loaded from), so the GUI side is a rename more than a rewrite.

Open question: pickle or HDF5. Pickle is what exists and what loads today;
`pulse_capture/hdf5.py` next door argues for HDF5 as the end state. Suggest
keeping the current pickle payload initially but putting it behind `store.py`
and versioning it, so the format can move later without touching callers.

---

## 8. Algorithms layer

* `multisweep` — unchanged in spirit, but takes and returns a `Catalog`. Its
  Option A / Option B split (build a fresh registry vs. re-use one) is exactly
  `Catalog` construction vs. `Catalog` consumption.
* `apply_bias` — rewritten against the catalog, in `algorithms/operation/`, and
  without the write-back: the catalog's frequencies are already the quantized
  ones, so it reads them and programs the tones. It also owns the module's NCO,
  which has to reach every tone and sit on the tone grid for those frequencies
  to mean what they say. **Landed.**
* `tune_resonators` (new) — the `trigger_capture` analogue: one call that runs
  netanal → find resonances → sweep ladder → find bias → apply → rotation, and
  returns a result dataclass with the catalog, the sweeps and the output path.
  Users who want the whole routine call this; users who want the pieces import
  from `rfmux.tuning`.

A script then reads roughly:

```python
catalog, sweeps = await crs.multisweep(config=sweep_cfg, catalog=catalog, module=1)
report = find_bias_points(sweeps, catalog, bias_cfg)
catalog.update_from(report.catalog)
await crs.apply_bias(module=1, catalog=catalog)
store.save(outdir, catalog, sweeps)
```

— which is the same sequence the GUI buttons perform, in the same order.

---

## 9. Periscope as a caller

`MultisweepPanel` holds a `Catalog` and an output folder instead of a raw dict
plus a dozen ad-hoc attributes. Buttons call the same library functions a script
would. `tasks.py` shrinks to what `pulse_capture_task.py` is: run the work off
the GUI thread and re-emit progress as signals. The deep-copy-then-merge dance
in `FindBiasTask` + `_find_bias_completed` becomes `catalog.update_from(...)`.
Dialogs become views over the config dataclasses.

---

## 10. What this fixes from the overview

| Problem today | Fixed by |
|---|---|
| Rotation angles keyed by code, looked up by channel → Rotated IQ silently broken | `Catalog.rotation_by_channel()` |
| Custom-frequency re-run mints new codes, detectors lose identity | `Catalog.rekey()` |
| Frequency lookups are 3-deep fallback chains in every consumer | `SweepEntry` accessors |
| "Clear `bias_found`, keep `bias_amplitude`" is a comment | `Catalog.clear_bias_results()` |
| Amplitude ladders and fraction→Hz resolution only exist inside dialogs | config dataclasses |
| Legacy file formats handled inline in the loader | `store.py` |
| No detector names | `Resonator.name` |

---

## 11. Suggested order

Each step is independently shippable and behaviour-preserving.

1. `catalog.py` + `sweeps.py`; rewire the current GUI to them. Pure refactor —
   and it retires the rotation bug and the fallback chains on its own.
2. `config.py` with `validate` / `describe`; dialogs become views. Biggest
   visible reduction in GUI code.
3. Split `bias_kids.py`: analysis to `tuning/bias.py`, applying to
   `algorithms/operation/apply_bias.py`.
   `fits.py` and `rotation.py` follow the same cut.
4. `store.py` — one reader/writer owning the folder layout and the legacy
   formats.
5. `tune_resonators` front door, with `simplified_tuning_flow` rewritten
   against it and run in CI — following the pulse-capture convention that the
   demo is both the documentation and a test.

Tests mirror the split: `test/tuning/` for the pure layers (catalog invariants,
re-key behaviour, config validation, bias-point selection on synthetic sweeps)
and `test/algorithms/` for the macros against a MockCRS.

---

## 12. Decisions so far

* **Find-Bias diagnostics live with the sweep set**, not on `Resonator` (§3).
  The catalog holds identity, operating point and downstream calibrations;
  everything derived from a particular sweep stays beside that sweep.
  Revisitable — promoting a field later is additive.
* **`Catalog` stays module-agnostic** for now, matching `res_info_dict` today:
  one catalog is shared across modules in a multi-module multisweep, and
  `channel_number` is the only per-module field. See the open question below.
* **Resonance finding moved to `rfmux/tuning/find_resonances.py`**, resolving
  §13's open question. It is analysis over two arrays, not a board operation, so
  it belongs with the other pure layers; `algorithms/measurement/fitting.py`
  keeps a deprecated shim that forwards to it and returns the old dict, so the
  Periscope panel and `simplified_tuning_flow` still work. The search takes
  arrays and the `find_resonances_in_netanal` wrapper unpacks a `take_netanal`
  result for it, which keeps the algorithm usable on any sweep from any source.
  `ResonanceSearch.to_catalog()` is where anonymous dips become a catalog.

## 13. Open questions

* **Should `Catalog` carry a module?** Left module-agnostic for now, but worth
  revisiting. Making it module-scoped would be a real semantic change, not just
  bookkeeping: today a multi-module multisweep resolves one registry and hands
  the same object to every module (`multisweep.py:273`), returning it once
  alongside a per-module data list. Arguments for adding it later: a catalog
  loaded from disk currently has no record of which module its channel numbers
  refer to, so the GUI infers it from the panel; and an array spanning several
  modules has no way to say so. Possible middle ground — an optional
  per-`Resonator` `module` field, so a catalog can span modules without forcing
  a single value on the whole collection.
* Should `Catalog` load and apply a name map (frequency → name)? That is how
  names would first get attached to a freshly found array.
* Pickle now and HDF5 later, or HDF5 straight away?
