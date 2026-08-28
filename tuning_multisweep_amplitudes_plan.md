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
  fused iteration index.

---

## 3. The layers

```
rfmux/tuning/amplitudes.py      AmplitudeSchedule — pure; catalog in, rungs out
rfmux/tuning/sweeps.py          packs the driver's output dict; accessors over it
rfmux/algorithms/measurement/
    multisweep.py               one catalog, one amplitude vector, one direction
    multisweep_amplitudes.py    the driver: rungs × directions, calls multisweep
```

`rfmux/tuning/` imports no `CRS` and no Qt, which is why the loop that actually
talks to the board is in the algorithms layer even though everything it decides
is computed in `tuning/`.

### `AmplitudeSchedule`

Splits the GUI's three modes back into their two axes — base amplitudes, and the
ladder applied to them:

```python
AmplitudeSchedule.fixed()                                    # the catalog's own amplitudes
AmplitudeSchedule.fixed(0.005)                               # one override for all
AmplitudeSchedule.fixed({"R0001": 0.004, "R0002": 0.006})    # per-resonator
AmplitudeSchedule.ramp(1e-3, 1e-2, steps=6, spacing="log")   # absolute ladder
AmplitudeSchedule.scaled(0.5, 2.0, steps=5)                  # × each resonator's own
AmplitudeSchedule.explicit([0.001, 0.003, 0.01])             # arbitrary rungs
```

`schedule.steps(catalog)` returns one record per rung carrying `index`,
`amplitudes: dict[name, float]` and `factor` (`None` when the rung is absolute).
`describe(catalog)` and `validate(catalog)` follow the `PulseCaptureConfig`
idiom, and are what a Periscope dialog renders instead of hand-rolling its own
status text.

Keying amplitudes by resonator **name** rather than by position retires the
"must match `res_info_dict.keys()` order" fragility, and makes per-resonator
amplitudes available in every mode rather than only in the single-sweep one.
Because `ResonatorCatalog` requires an amplitude at construction, `scaled` has no
"no registry yet, run a plain sweep first" failure mode.

### The driver macro

```python
await crs.multisweep_amplitudes(
    catalog,
    span_hz=200e3,
    npoints_per_sweep=101,
    schedule=AmplitudeSchedule.ramp(1e-3, 1e-2, steps=6, spacing="log"),
    directions=("upward", "downward"),
)
```

Walks rungs × directions, calls `multisweep` once per combination, hands each
result to `tuning/sweeps.py` to pack, and returns the packed dict. It mutates
nothing: choosing an operating amplitude is bias finding's job, fitting is
`fits.py`', writing files is `store.py`'.

`directions` is an explicit tuple rather than the magic string `"both"`, so the
product is honestly a product and each result is labelled with both coordinates.

Callbacks: `progress_callback(module, pct)` stays as it is for within-a-sweep
progress, and a step-level `step_callback(step_record)` fires as each rung
lands. A script ignores it, a notebook prints from it, Periscope re-emits it as
the signals it has now.

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
    "module": 2,
    "catalog": catalog.to_dict(),        # snapshot as swept, for provenance
    "span_hz": ..., "npoints_per_sweep": ..., "nsamps": ...,
    "schedule": schedule.to_dict(),
    "steps": [
        {
            "step": 0,
            "direction": "upward",
            "amplitudes": {"R0001": 0.004, ...},
            "factor": 0.5,               # None for an absolute rung
            "data": { ...one multisweep return... },
        },
        ...
    ],
}
```

Plain builtins and ndarrays throughout, so `pickle.dump` is the whole
persistence story until `store.py` exists. `tuning/sweeps.py` supplies the
readers — sweeps for one resonator in amplitude order, the rung at a given
amplitude, and so on — so consumers stop re-deriving them inline.

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
2. **`AmplitudeSchedule`** in `rfmux/tuning/amplitudes.py`, with tests. Pure, no
   hardware, ships on its own.
3. **`rfmux/tuning/sweeps.py`** — the packer and the accessors over the output
   dict.
4. **`multisweep_amplitudes`** macro over 1–3, with the step callback.
5. **Periscope rewired** to call 4, its dialog reduced to a view over
   `AmplitudeSchedule`.

Step 1 is in progress; the rest are not started.

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
