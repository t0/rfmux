# One shape for every sweep result

Working document, in the same family as `tuning_refactor_design.md` (the overall
plan), `tuning_multisweep_amplitudes_plan.md` (the amplitude ladder) and
`tuning_revamp_todo.md` (the work queue). This one covers a single question:
what a sweep macro returns.

The short version: `multisweep` and `multiamp_multisweep` return the same
thing — a dict keyed by module identifier, each value a self-describing envelope
whose `results` are keyed `[iteration][direction][name]`. A single sweep is one
iteration in one direction, which it genuinely is. Analysis functions take one
module's envelope, never the container.

---

## 1. What exists today, and where

Two shapes, and everything downstream pays for the difference.

**`multisweep`** (`algorithms/measurement/multisweep.py:213`) returns a flat
`{name: entry}`, where an entry is `channel`, `frequencies`, `iq_counts`,
`iq_volts`, `original_center_frequency`, `sweep_direction`, `sweep_amplitude`
(built at `multisweep.py:588`). Nothing records the module, the span, the point
count or `nsamps`, so a pickled sweep cannot say what produced it. With
`center_frequencies` and `module=[1, 2]` it returns a bare *list* of those
dicts (`multisweep.py:395`), with nothing but argument order to say which
element is which module.

**`multiamp_multisweep`** (`algorithms/measurement/multiamp_multisweep.py:106`)
returns the envelope `pack_results` builds
(`tuning/multisweep_amplitudes.py:688`): `schema_version`, `module`,
`call_params`, and `results[iteration][direction][name]`. It refuses a module
list outright, so it is always single-module.

The cost of the split is concentrated in `tuning/fits.py`, which accepts both:

* `_is_packed` (`fits.py:569`) discriminates on `"results" in sweeps and
  "call_params" in sweeps`.
* `_walk` (`fits.py:578`) has a branch per shape, plus a `TypeError` for the
  multi-module list telling callers to index it positionally.
* `_select` (`fits.py:651`) special-cases `iteration is None` to refuse
  `iterations=` / `directions=` filters.
* `SweepFit.iteration` and `.direction` (`fits.py:167`) are `int | None` /
  `str | None` purely so the bare shape has something to put there, and `where`
  branches on it.
* `fit_sweeps_at_bias_amplitude` (`fits.py:365`) refuses a bare multisweep.

---

## 2. Decisions

* **One shape, not two accepted shapes.** The reason `fits.py` sniffs is that
  there are two things to sniff. Unify the shapes and the sniffing has nothing
  left to do.
* **A plain multisweep sits at iteration 0, in its own direction.** This is not
  a padded slot: one call *is* one amplitude step in one direction, so
  `results[0]["upward"]` states a fact. `multiamp_multisweep` becomes what it
  claims to be in its own docstring — a collector.
* **Module identifier outermost, always — including for one module.** A caller
  who writes `for module_id, module_sweeps in sweeps.items():` has written the
  same code for one module and for four. A convenience that flattens the
  single-module case would make the common script *differ* from the general
  one, which is the class of thing this revamp keeps deleting.
* **Analysis and plotting functions take one module's envelope.** Stepping into
  the module you mean is the caller's job, and it is one subscript. The
  alternative — passing the container plus a `module=` argument — puts a
  coordinate in a function signature that the data structure already carries,
  and gives every reader a "which module did you mean" branch.
* **The container is recognized only to be refused.** Handing `fit_sweeps` the
  whole dict gets an error naming the module keys it found and showing the
  subscript. Recognizing a shape in order to reject it is not dispatching on it:
  there is still exactly one accepted input.
* **`call_params` stays verbatim** — what was asked for, not what was worked out
  from it — as `pack_results` already documents.
* **Entries keep `sweep_direction` and `sweep_amplitude`** even though the
  envelope now records direction too. They are what survives an entry being
  lifted out of its nesting, and amplitude is per-resonator regardless.

---

## 3. The shape

```python
{
    "crs0030_rmod1": {
        "schema_version": 3,
        "module": 1,                 # the int, for handing back to hardware calls
        "call_params": {...},        # verbatim, as the macro was called
        "results": {
            0: {"upward": {"R0001": {...}, "R0002": {...}}},
        },
    },
    "crs0030_rmod2": {...},          # only when module=[1, 2]
}
```

Identical from both macros. A plain `multisweep` has exactly one iteration and
one direction; `multiamp_multisweep` has one iteration per rung and one entry
per direction swept. Under a direction is the `{name: entry}` dict that
`multisweep` returns today, unchanged field for field.

`call_params` differs between the two macros in one place, and no reader depends
on it:

| | `multisweep` | `multiamp_multisweep` |
|---|---|---|
| amplitude spec | `amp` (verbatim: `None`, float, list or mapping) | `amp_schedule` (`to_dict`) |
| direction spec | `sweep_direction` (str) | `directions` (list) |
| shared | `catalog`, `center_frequencies`, `names`, `span_hz`, `npoints_per_sweep`, `nsamps`, `module` | — |

`multisweep` does **not** synthesise a one-rung `amp_schedule` to make the two
identical. It has no schedule; inventing one would be fiction of the kind this
codebase keeps out. `find_iteration_matching_amplitude` only needs
`call_params["catalog"]`, which both carry.

**Duplication across modules is accepted.** A multi-module call is only ever
possible with `center_frequencies` — a catalog belongs to one module and a
module list is refused — so `call_params` is identical across modules except the
`module` field, and cannot diverge in any way that matters. In exchange, each
module's value is complete: lift it out, pickle it, hand it to a fitter.

**`data_callback` partials stay bare** `{name: {frequencies, iq_counts,
original_center_frequency}}` (`multisweep.py:565`). A partial is not a result,
and wrapping it would emit a `call_params` block dozens of times per sweep.

---

## 4. The module identifier

`ReadoutModule.index()` (`core/schema.py:222`) already exists and is documented
as "a shorthand string representation for this readout module":

```python
return "crs%s_rmod%d" % (self.crs.serial, self.module)   # 'crs0030_rmod1'
```

It has no callers today. That is an argument for adopting it rather than minting
a second convention beside a documented one nobody got around to using.

Reachable from inside a macro as `crs.module[m].index()`, which is also the
idiom for demos and interactive work — no hardcoded serial, and a module that
was not swept raises instead of quietly picking one:

```python
module_sweeps = sweeps[crs.module[2].index()]
```

**One fix first.** `crs.serial` can be `None`: a hostname-only HWM is supported,
and `tuber_hostname` (`core/schema.py:179`) falls back hostname → serial →
slot/crate. Today `index()` would silently produce `"crsNone_rmod1"`. Give it
the same cascade so the identifier is always meaningful. This is its own small
change to `schema.py`, landed before anything below depends on it.

---

## 5. Where the shape lives

`pack_results` and the readers are in `tuning/multisweep_amplitudes.py`, a
module named for the amplitude ladder. After this change the shape is not the
ladder's — it is every sweep's — so leaving the packer there misfiles it.

Move the shape-owning code to a new `rfmux/tuning/sweep_results.py`: the schema
version, the packers, and the four readers (`_iterations`, `_section_names`,
`collect_amplitude_iterations_for`, `get_amplitudes_at_iteration`,
`find_iteration_matching_amplitude`, `_bias_amplitude_of`). `tuning/__init__.py`
re-exports the same names it does today, so the public path is unchanged from
outside.

**Landed, with one correction.** The move was going to leave re-exports behind in
`multisweep_amplitudes.py`, but that would have been a cycle: `pack_results`
calls `amp_schedule.to_dict()` and both it and `_bias_amplitude_of` use the
`_named` error-message helper, so `sweep_results` imports *from*
`multisweep_amplitudes` and the arrow cannot also point back. `_named` straddles
the two halves (three uses in the schedule, two in the readers) and stayed where
it is rather than being moved or duplicated.

So the direction is `sweep_results → multisweep_amplitudes`, and the four
importers were updated instead of shimmed: `tuning/__init__.py`, `fits.py`,
`multiamp_multisweep.py`, and two test modules. That is the honest arrangement —
the shape depends on the schedule, and a re-export would have hidden which way
round that goes.

Two packers, since the two macros have different `call_params`:

```python
def pack_sweep(sections, *, module_id, module, call_params, direction) -> dict
def pack_results(sweeps, *, module_id, module, ...) -> dict   # as today, plus module_id
```

Both return the one-key container, so the macros never assemble a dict literal
themselves.

---

## 6. What changes, file by file

**`core/schema.py`** — `ReadoutModule.index()` gains the `tuber_hostname`
fallback cascade so it never renders `None`.

**`tuning/sweep_results.py`** (new) — the moved shape code, plus `pack_sweep`.
`RESULTS_SCHEMA_VERSION` 2 → 3.

**`tuning/multisweep_amplitudes.py`** — keeps `AmplitudeSchedule`; the four
importers of the moved names point at `sweep_results` instead. No re-exports;
see §5.

**`algorithms/measurement/multisweep.py`** — returns
`pack_sweep(results, ...)` instead of the flat dict (`multisweep.py:608`). The
multi-module branch (`multisweep.py:366`) becomes a dict union over the gathered
per-module containers rather than a list — each child already returns a one-key
container, so merging is `{k: v for c in containers for k, v in c.items()}` and
the list return disappears. Returns docstring (`multisweep.py:325`) rewritten.

**`algorithms/measurement/multiamp_multisweep.py`** — indexes its own resolved
module out of each child on the way past
(`child[module_id]["results"][0][direction]`, replacing the raw `data` at
`multiamp_multisweep.py:329`) and packs a one-key container at the end. Its
"one module per call" rule (`_resolve_module`) is unchanged. The
`sweep_callback` record's `data` field: keep it as the bare `{name: entry}` for
that one sweep, matching `data_callback`'s partials — a per-sweep hand-over is
not a result either. Note this explicitly in the docstring, since the field
currently carries whatever `multisweep` returned.

**`tuning/fits.py`** — the simplification this is for:

* `_is_packed` deleted.
* `_walk` loses its branches: one nesting, always. It gains a check that refuses
  the container with a message naming the module keys and showing
  `fit_sweeps(sweeps['crs0030_rmod1'])`.
* The multi-module `TypeError` for lists (`fits.py:580`) becomes that message,
  and stops telling people to index positionally.
* `SweepFit.iteration` / `.direction` become plain `int` / `str`; `where` loses
  its branch and always reads `R0001@0 upward`.
* `FitReport.settings` records the module id, read off the envelope, so a
  printed report says which module without the caller threading it.
* `_select`'s `bare` guard and `fit_sweeps_at_bias_amplitude`'s `_is_packed`
  refusal are **left alone for now** — deliberately. See the entry in
  `tuning_revamp_todo.md`; both become unreachable-as-written and need a
  decision each, not a silent deletion.

**Tests** — `test/algorithms/test_multiamp_multisweep.py:51`'s fake CRS must
return the envelope shape, or the driver's unwrapping has nothing to unwrap.
`test/algorithms/test_multisweep_channels.py:74` and `:94` index the return.
`test/tuning/test_fits.py` exercises both input shapes throughout;
`test_fits.py:387` guards the bias-amplitude refusal and stays until that
question is answered. `test/algorithms/test_measurement_flow.py:172` and `:229`
mock `crs.multisweep` with bare dicts, so they pass today whatever the real
macro does — worth fixing while in there.

**Demos** — `Demos/multisweep.md` (both macros, ~10 call sites),
`Demos/fitting_resonators.md` (including the "`fit_sweeps` also takes what a
single `crs.multisweep` returns" paragraph at `:993`, which stops being a
special case worth mentioning), `Demos/simplified_tuning_flow.{md,py}` (already
broken against the current contract).

**Periscope** — `tools/periscope/tasks.py:730` and `multisweep_panel.py` read
the old shape. Both already fail against the current macro and are on the
step-5 rewire in `tuning_multisweep_amplitudes_plan.md`; this adds to what that
rewire has to absorb rather than breaking something that works.

---

## 7. What this costs

Hand-walking one trace from a plain sweep is six subscripts:

```python
sweeps["crs0030_rmod1"]["results"][0]["upward"]["R0001"]["iq_counts"]
```

That is the accumulated price of the envelope, the unification and the module
layer together. It is worth paying, but it changes what "the supported way to
read a result" means: the readers in `sweep_results.py` stop being a
convenience and become the interface. `collect_amplitude_iterations_for` now
works on a plain multisweep too, which it cannot today — so the supported path
gets uniformly better while hand-walking gets worse, which is the trade being
made on purpose. The demos should lead with the readers rather than the
subscripts.

---

## 8. Order of work

1. **`ReadoutModule.index()` fallback** in `core/schema.py`, with a test for the
   hostname-only case. Independent of everything else; lands on its own.
2. **`tuning/sweep_results.py`** — the pure move, re-exports in place, no
   behaviour change. Tests should pass untouched.
3. **`pack_sweep` and the container**, with `RESULTS_SCHEMA_VERSION` → 3.
   Pure, no hardware, tested on its own.
4. **`multisweep` returns the container**, multi-module branch merged into one
   dict. `test_multisweep_channels.py` updated.
5. **`multiamp_multisweep` collects**, fake CRS updated.
6. **`fits.py` simplified** — `_is_packed` gone, `_walk` single-shape, `SweepFit`
   coordinates non-optional. The two deferred guards left as they are.
7. **Demos rewritten**, leading with the readers.

Steps 4 and 5 must land together or the driver breaks; the rest are separable.

---

## 9. Deliberately out of scope

* The two `fits.py` guards — tracked in `tuning_revamp_todo.md`.
* Periscope's rewire, which is step 5 of
  `tuning_multisweep_amplitudes_plan.md`.
* `store.py` and how a container is written to disk. The container is what it
  will be handed; how it lays that out is that document's question.
* The legacy fit walkers, already deprecated and already reading a contract that
  no longer exists.

---

## 10. Open questions

* Does a `FitReport` from `fit_sweeps` want the module id in `where` after all,
  once several reports are being compared side by side in a notebook? Left out
  because it is constant within a report; the `settings` entry is there to be
  promoted if it turns out to be wanted.
* Should the container be a plain `dict`, or a small class that knows how to
  say which modules it holds and raises well on a missing key? A plain dict for
  now — it loops with `.items()`, which is the whole point, and a class is a
  later wrapper rather than a different shape.
