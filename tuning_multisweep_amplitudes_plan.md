# Sweeping a catalog at several amplitudes

Working document, in the same family as `tuning_refactor_design.md` (the overall
plan) and `tuning_revamp_todo.md` (the work queue). This one covers a single
question: how a headless user sweeps an array at more than one probe amplitude.

The short version: `multisweep` keeps doing exactly one sweep, a separate macro
walks a ladder of amplitudes and calls it once per rung, and a pure module in
`rfmux/tuning/` builds both the ladder and the output dict.

---

## 1. What exists today, and where

All amplitude iteration currently lives in the Periscope GUI on branch
`mr_multisweep_section_amplitudes`. The algorithm there takes
`amp: float | list[float] | None` and performs one sweep; the ladder is
constructed in `MultisweepDialog.get_parameters` (`multisweep_dialog.py:1359`)
and executed by `MultisweepTask.run` (`tasks.py:508`).

Three modes, selected by radio button:

| Mode | Inputs | Result |
|---|---|---|
| Single iteration | one global amplitude, **or** a per-section array (one value per section) | one sweep |
| Uniform sweep | start, stop, steps, linear (`np.linspace`) or log (`np.geomspace`) | N sweeps, every section at the same amplitude per rung |
| Multiplicative scaling | start factor, stop factor, steps (`np.linspace` only) | N sweeps of `bias_amplitude × factor`, base read from `res_info_dict` |

The dialog emits one flat `amp_arrays: list[list[float]]` — a rung per outer
element, a section per inner element — plus loose metadata keys
(`iteration_mode`, `num_steps`, `uniform_*`, `scale_*`, `base_amplitude_*`).
The task loops rungs × directions and tags each result with a single
`iteration` integer.

Things that are worth *not* carrying over:

* Per-section amplitudes exist only in single-iteration mode. "Per-resonator
  base × a ladder" is reachable only through scaling, which forces the base to
  come from the registry. The three modes are two orthogonal choices — what the
  base amplitudes are, and what ladder is applied to them — collapsed into one
  radio group.
* `iteration` fuses amplitude and direction into one integer, so every consumer
  re-derives which is which from the entry body.
* The reported amplitude is `amp_array[0]`, the first section's — meaningless
  for a per-section array.
* `if single_iteration or num_steps == 1:` (`multisweep_dialog.py:1434`) takes
  the single-iteration branch in every mode, but `base_amp_array` is `None`
  there for uniform and scaling — so uniform mode with one step ships
  `amp=None` to the algorithm.
* Amplitudes are positional lists that must match `res_info_dict.keys()` order.
* The dialog reaches into `self.params['res_info_dict']` to build a ladder
  (`multisweep_dialog.py:1504`) — already noted in `tuning_refactor_design.md` §5.

---

## 2. Decisions

* **The driver is a separate macro**, not a mode of `multisweep`. A `multisweep`
  that sometimes returns one sweep and sometimes a stack of them regrows the
  Option A / Option B branching the revamp is removing.
* **Measurement outputs stay plain dicts**, pickled — the same contract
  `take_netanal` and today's `multisweep` have. No `SweepSet` / `SweepEntry`
  classes for now. `rfmux/tuning/sweeps.py` owns the *shape* of those dicts and
  the accessors over them, so there is one definition, but what crosses the
  boundary is builtins and ndarrays.
* **`multisweep` takes a catalog and does not modify it.** It reads the bias
  frequency, amplitude and channel of each resonator and returns data. Anything
  that *learns* something — fitting, bias finding — updates the catalog itself,
  later, and is out of scope here.
* **A bare frequency list stays a first-class second mode**, with `amp` as one
  value or one per frequency. Not everything worth sweeping is a tuned array:
  resonances may not have been found yet, or there may be none to find. Its
  sweeps are named `S0001…` (S for section) unless `names` says otherwise —
  strings either way, so downstream code has one key type, but visibly not a
  catalog's `R0001…`. The two modes are identical once the measurement starts.
* **Amplitude and direction are separate labelled axes** in the output, not one
  fused iteration index. **A step is one amplitude**, numbered from 0 in the
  order it was measured; a step is swept once or twice, and direction is a
  subkey *beneath* a step. So `len(schedule)` counts amplitudes, and the sweep
  count is that times the number of directions.
* **Per-resonator *absolute* ladders are not representable.** Proportional
  per-resonator ladders — what a bifurcation walk wants — are
  `multiplicative(base={...})`; non-proportional ones (R0001 goes 1→2→4 µ while R0002
  goes 3→3.5→4) have yet to find a use that justifies the extra state. Widening
  `ladder` to hold scalars-or-mappings is additive if one turns up.
* **One "sweep" is a whole multisweep measurement; the per-resonator pieces are
  "sweep sections", or "sections".** Worth being strict about, because the loose
  reading actively misleads here: an `AmplitudeStep` repr saying "10 sweeps at
  0.0005" reads as ten passes at that amplitude, when it means one pass
  containing ten sections. So `describe()` reports `n_sweeps` (steps ×
  directions) alongside `n_sections`, and messages counting the per-resonator
  pieces say "sections".

---

## 3. The layers

```
rfmux/tuning/multisweep_amplitudes.py   pure: the amplitudes going in, and the shape coming back
rfmux/algorithms/measurement/
    multisweep.py                       one catalog, one amplitude vector, one direction
    multiamp_multisweep.py              the driver: steps × directions, calls multisweep
```

There is no `sweeps.py`. It was a leftover from `tuning_refactor_design.md` §4,
where `SweepSet`/`SweepEntry` were classes wrapping *all* sweep data across the
GUI; once this plan decided measurement outputs stay plain dicts, the file kept
its name and its slot while losing the reason to be separate. What was left for
it splits in two, and neither half wants a file of its own:

* **Readers over the driver's dict** live beside `AmplitudeSchedule`, because
  the schedule's own `to_dict()` is *inside* that dict — a reader resolving
  `ladder[iteration]` has to agree with the schedule about what a rung means,
  and two files agreeing about one contract is one file too many.
* **Readers over a single sweep entry** — the `bias_frequency` →
  `sweep_center_frequency` → `original_center_frequency` fallback chain,
  `iq_volts` backfill, legacy flat fit keys — are **not being written**. They
  exist to read files this revamp is replacing, and carrying compatibility for
  them costs readability now against a benefit nobody has asked for. If a real
  caller turns up, that is when it earns a home.

`rfmux/tuning/` imports no `CRS` and no Qt, which is why the loop that actually
talks to the board is in the algorithms layer even though everything it decides
is computed in `tuning/`.

### `AmplitudeSchedule`

Splits the GUI's three modes back into their two axes. A schedule is exactly a
**base** — what each resonator would be swept at with no iteration — and the
**steps** applied to it, which either multiply the base (`relative=True`) or
*are* the amplitude (`relative=False`).

The forms that do not iterate are the plain constructor, because `base` is the
first field and the only one a caller sets by hand with any regularity:

```python
AmplitudeSchedule()                                      # each resonator's own amplitude
AmplitudeSchedule(0.005)                                 # one amplitude for all
AmplitudeSchedule({"R0001": 0.004, "R0002": 0.006})      # per-resonator
```

The iterating forms are classmethods over the same two fields, so there is a
single code path to test:

```python
AmplitudeSchedule.multiplicative(0.5, 2.0, 5)                        # × each resonator's own
AmplitudeSchedule.multiplicative(0.5, 2.0, 5, base={"R0001": 0.004}) # × a base you chose
AmplitudeSchedule.ramp(1e-3, 1e-2, 6)                                # absolute, generated
AmplitudeSchedule.explicit([0.001, 0.003, 0.01])                     # absolute, arbitrary
```

`multiplicative` rather than `scaled`: it says what the factors *do*, and it
does not read as a near-synonym of `ramp`.

`spacing` defaults to `"log"` (equal ratios, so equal steps in dB) with
`"linear"` available; it is recorded for provenance only and excluded from
equality, so two schedules that measure the same amplitudes compare equal
however they were spelled. A schedule that does not iterate reports `"none"`,
since no spacing rule generated its single step.

There was a `fixed()` classmethod covering the non-iterating cases. Once `base`
became the first field it was the same thing spelled longer, so it is gone
rather than kept as a second way to say one thing.

**The absolute forms take no base**: their steps already are the amplitudes, so
there is nothing left for a base to contribute. That is the one asymmetry in the
"two axes" story, and it is enforced rather than papered over.

`schedule.steps(target)` returns one `AmplitudeStep` per amplitude, carrying
`step` (execution order), `amplitudes: dict[name, float]` and `factor` (`None`
when the rung is absolute). `amplitudes` drops straight into
`multisweep(amp=...)` with no adaptation, which is what lets the driver be a
loop and nothing more. `describe(target, n_directions, dac_scale_dbm)` and
`validate(target, n_directions)` follow the `PulseCaptureConfig` idiom, and are
what a Periscope dialog renders instead of hand-rolling its own status text;
`validate` never raises, so a live preview of a half-entered form gets text
rather than a traceback.

*target* is a catalog, **or a list of sweep names** for the bare
`center_frequencies` mode — the same names `multisweep` will key its results by.
A bare list has no `bias.amplitude` to fall back on, so `base=None` is an error
there, while `ramp`/`explicit` need no base and work unchanged. As in
`multisweep`, a *positional* base is accepted alongside names (the caller's own
ordering) and refused alongside a catalog.

Keying amplitudes by resonator **name** rather than by position retires the
"must match `res_info_dict.keys()` order" fragility, and makes per-resonator
amplitudes available in every mode rather than only in the single-sweep one.
Because `ResonatorCatalog` requires an amplitude at construction, `multiplicative` has no
"no registry yet, run a plain sweep first" failure mode.

Two failures are caught here that would otherwise reach the hardware, since
`multisweep` only rejects non-positive amplitudes and nothing on the `amp` path
enforces an upper bound: a resolved amplitude outside the `(0, 1]`
`BiasPoint` requires — reported per step and per name, so the answer is "R0007
overshoots at step 5" and not a failure twenty minutes into the run — and
`nsteps=1` between two *different* endpoints, which is the bug at
`multisweep_dialog.py:1434` restated as a `ValueError`.

### The driver macro

```python
await crs.multiamp_multisweep(
    catalog,
    span_hz=200e3,
    npoints_per_sweep=101,
    amp_schedule=AmplitudeSchedule.ramp(1e-3, 1e-2, 6),
    directions=("upward", "downward"),
)
```

Walks steps × directions, calls `multisweep` once per combination, and returns
them in one dict. It mutates nothing: choosing an operating amplitude is bias
finding's job, fitting is `fits.py`', writing files is `store.py`'.

No `amp` argument: amplitude is the schedule's job and nothing else's, so there
is one way to say it. `amp_schedule` defaults to `AmplitudeSchedule()`, which
makes the useful degenerate call an up-and-down pair at the catalog's own
amplitudes with nothing else said. Spelled `amp_schedule` rather than `schedule`
because a bare `schedule` in a measurement macro reads like a time.

The driver's only contact with the board is `crs.multisweep` — no
`tuber_context`, no `get_samples`. That keeps hardware knowledge one layer down,
and means the loop is exercised in full by a fake CRS with a single async
method.

Step outer, direction inner: each step's up-and-down pair is measured together
and amplitude marches monotonically, which is what a bifurcation walk wants.
That is a physical choice, not an implementation accident, so it belongs in the
docstring — and the deferred early-stop question in §7 depends on it.

`directions` is an explicit tuple rather than the magic string `"both"`, so the
product is honestly a product and each result is labelled with both coordinates.

Callbacks, three of them:

* `progress_callback(module, pct)` is forwarded untouched, so it keeps meaning
  *progress within the current sweep* and resets once per sweep.
* `data_callback(module, partial_results, step, direction)` is **two arguments
  wider than `multisweep`'s**. Inside a ladder the bare `(module, partial)` form
  is ambiguous — a consumer plotting live cannot tell which sweep the points
  belong to. Callers written against the narrow form need updating; Periscope is
  tracked in `tuning_revamp_todo.md`.
* `sweep_callback(record)` fires once per completed **sweep**, not per step,
  carrying `step`, `direction`, `amplitudes`, `factor`, `completed`, `total` and
  `data`. Per-sweep is the finer granularity and a per-step consumer can
  accumulate; the reverse is not true. A script ignores it, a notebook prints
  from it, Periscope re-emits it as the signals it has now.

That last one is also why the driver does not try to return partial results when
a sweep fails: every sweep that finished has already been handed over, so the
exception can propagate rather than the driver inventing a result dict with
holes in it that reads as complete.

---

## 4. Data shapes

`multisweep` returns what it returns today, keyed by resonator name:

```python
{
    "R0001": {
        "name": "R0001",
        "channel": 1,
        "frequencies": ndarray,          # Hz
        "iq_complex": ndarray,           # readout counts
        "iq_complex_volts": ndarray,
        "phase_degrees": ndarray,
        "original_center_frequency": float,   # the catalog's bias frequency
        "sweep_amplitude": float,             # what this sweep actually used
        "sweep_direction": str,
        ...                                   # unchanged: bias_frequency,
                                              # df_calibration, rotation_tod, …
    },
    ...
}
```

The driver wraps those:

```python
{
    "schema_version": 1,
    "module": 2,                          # resolved, never None
    "call_params": {                      # verbatim, as the driver was called
        "catalog": catalog.to_dict(),     # or None
        "center_frequencies": None,       # or the list, in that mode
        "names": None,                    # as passed, not as resolved
        "amp_schedule": amp_schedule.to_dict(),
        "directions": ["upward", "downward"],
        "span_hz": ..., "npoints_per_sweep": ..., "nsamps": ...,
        "module": ...,
    },
    "results": {
        0: {                              # the amplitude step, in measured order
            "upward":   { ...one multisweep return... },
            "downward": { ...one multisweep return... },
        },
        1: {...},
    },
}
```

Four things this shape is deliberate about:

* **`results`, not `steps`** — it is the main data repository, and the name
  should say so. Integer keys in measurement order; a step holds one entry per
  direction swept **and nothing else**, so a step measured once and a step
  measured twice have the same shape.
* **Nothing is duplicated into the step level.** What each resonator was probed
  at is already `sweep_amplitude` in its own entry; the rung that produced it is
  `call_params["amp_schedule"]["ladder"][step]`. Storing a step-level
  `amplitudes` too would be a second copy to keep in step. `tuning/sweeps.py`
  wraps those lookups; until it exists they are the documented route.
* **`call_params` is verbatim**, including the `None`s — it says what was
  *asked for*. Anything resolved from it (the module, the section names) is
  either top-level or already in the data.
* **Sweep centres are recorded only as passed.** A later step may re-centre
  between amplitudes, at which point a top-level copy of the centres would be a
  lie while each sweep's own `original_center_frequency` cannot be.

Plain builtins and ndarrays throughout, so `pickle.dump` is the whole
persistence story until `store.py` exists.

Direction keys are `multisweep`'s own `"upward"`/`"downward"`, so there is one
vocabulary rather than a mapping at the boundary.

`pack_results` in `tuning/multisweep_amplitudes.py` writes this, and three
readers beside it get things back out, so consumers stop re-deriving them
inline:

* `collect_amplitude_iterations_for(results, name)` — one resonator's sweeps
  across every iteration, `{iteration: {direction: sweep}}`, in the order
  measured rather than sorted by amplitude (an `explicit` ladder may run in any
  order, and re-sorting silently would lose what actually happened).
* `get_amplitudes_at_iteration(results, iteration)` — `{name: amplitude}`, read
  from each sweep's own `sweep_amplitude`. This is why the packed dict carries
  no iteration-level copy: it is reconstructed on demand instead of stored
  twice.
* `find_iteration_matching_amplitude(results, name, amplitude=None)` —
  nearest match, defaulting to that resonator's bias amplitude from the catalog
  snapshot. It takes a *name* because a relative ladder gives every resonator
  its own amplitudes: R0001 walking 1→2→4 µ and R0002 walking 3→6→12 µ share an
  iteration number and nothing else, so "the iteration at 4 µ" is only a
  question about one of them. Nearest rather than exact because a ladder's
  floats rarely compare equal — `0.001 × 4` is not `0.004`.

---

## 5. Deliberately out of scope

* Choosing a bias amplitude from the ladder (bifurcation walk) — that is
  `find_bias_points`, and it will update the catalog.
* Fitting, and the "which amplitude do we fit" policy.
* File layout and naming (`store.py`).
* Anything that writes back into the catalog.

---

## 6. Order of work

1. **`multisweep` accepts a catalog** — one sweep, input read-only. Amplitude
   defaults to each resonator's `bias.amplitude`; an explicit `amp` overrides,
   as a scalar for all or a mapping per resonator. A bare `center_frequencies`
   list plus an `amp` (scalar, or one per frequency) stays a supported mode
   rather than a deprecated one: it is how you sweep something that is not a
   tuned array yet, and it keeps Periscope and `simplified_tuning_flow` working
   unchanged. Demonstrated in
   `reference-notebooks/Demos/multisweep.md`.
2. **`AmplitudeSchedule`** in `rfmux/tuning/multisweep_amplitudes.py`, with
   tests. Pure, no hardware, ships on its own.
3. **`multiamp_multisweep`** macro over 1–2, with the sweep callback.
4. **`pack_results` and its readers**, in `tuning/multisweep_amplitudes.py`.
   Written after the driver rather than before it, so the shape was settled
   against a working call instead of guessed; then the packing moved back here,
   so one module writes the dict and reads it.
5. **Periscope rewired** to call 3, its dialog reduced to a view over
   `AmplitudeSchedule`.

Steps 1–4 have landed; 3 and 4 swapped places on the way. Step 5 is not started,
and the set still owes a notebook: `Demos/multisweep.md` covers one sweep, and a
multi-amplitude demo belongs beside it.

---

## 7. Open questions

* Does `AmplitudeSchedule` stay in its own module, or fold into the
  `config.py` of `tuning_refactor_design.md` §5 as `SweepConfig.amplitudes` once
  that file exists? Its own module for now; folding it in later is a re-export.
* Should the driver stop early on bifurcation rather than always walking every
  rung? That is the expensive-measurement argument for it, and the argument
  against is that it puts an analysis decision inside the driver. Deferred until
  `find_bias_points` is ported and we can see what it needs.
* Multi-module. A catalog is per-module, so the driver is per-module too, and
  running four modules is four calls (or one `asyncio.gather`). Whether that
  wants a helper is a question for when there is a caller.
