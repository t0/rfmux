# Merging `main` into `tuning_headless_revamp`: conflict survey

Written 2026-09-08, against `origin/main` = `efb4407` and
`tuning_headless_revamp` = `3a26dd2`. Nothing has been merged yet. Every
claim about conflicts below was checked with a no-touch dry run
(`git merge-tree --write-tree --merge-base=1b1f5ce HEAD origin/main`) and by
reading both sides' code; the merged tree it produced is `daf5831`.

There are two layers to this. §3 is the eight files git will stop on. §4 is
the larger problem: the two branches made different design decisions about
the same things, mostly in files that do not overlap at all. Git merges those
without a word, and the result is a tree where two tuning systems coexist and
disagree. Read §4 before deciding how to resolve §3.

## 1. The topology problem, and why it matters before anything else

The buffer exploration branch went into `main` as a **squash merge**:

```
efb4407  Pulse Techniques Implementation / Buffer Exploration (#78)
```

It has exactly one parent (`e46fc41`). `origin/buffer_exploration` (`91b2b86`)
is *not* an ancestor of `main`, so git cannot see that the work is already
represented there.

Our branch forked from buffer exploration at `1b1f5ce` and has 74 commits
since; buffer exploration went another 166 commits before landing. So there
are three different "bases" in play:

| Comparison | Merge base git picks | Conflicted files |
|---|---|---|
| `HEAD` × `origin/main` (naive) | `385eee9`, far too old | **250+** |
| `HEAD` × `origin/buffer_exploration` | `1b1f5ce`, correct | 7 |
| `HEAD` × `main` with base forced to `1b1f5ce` | `1b1f5ce`, correct | **8** |

The naive `git merge origin/main` re-conflicts every buffer-exploration change
our branch already contains. Do not run it.

Also note: **`main` is not simply `buffer_exploration` + fastrx.** The PR
picked up review changes on the way in: 64 files differ between
`origin/buffer_exploration` and `origin/main`, including `bias_kids.py`,
`multisweep.py`, `take_netanal.py`, the pulse-capture modules, and a
repo-convention change (`CLAUDE.md` is now an 11-byte `@AGENTS.md` pointer,
with the real content in a new `AGENTS.md`). Merging `buffer_exploration`
is **not** a substitute for merging `main`.

## 2. Recommended mechanism

Teach git the squash, locally and reversibly, then merge `main` normally:

```bash
git replace --graft efb4407 e46fc41 91b2b86
git merge origin/main        # merge base is now 1b1f5ce; 8 conflicts
git replace -d efb4407
```

The replacement ref is local-only and is not pushed; the merge commit records
the honest parents `[HEAD, efb4407]`. Caveat: `refs/replace/*` lives in the
shared `.git` directory and is visible to the other worktrees while it exists.
Add it, merge, delete it.

To *preview* without touching anything (what this survey used):

```bash
git merge-tree --write-tree --merge-base=1b1f5ce HEAD origin/main
```

`main`'s non-buffer work (fastrx, AF_XDP, PacketWriter, firmware verbs) has
zero file overlap with our 70 changed files and contributes no conflicts.

## 3. The eight textual conflicts

Only 12 files were modified by both sides since `1b1f5ce`. Eight conflict;
four auto-merge cleanly (`core/transferfunctions.py`, `test/conftest.py`,
`tools/periscope/tasks.py`, `docs/guides/getting-started.md`; the first two
were inspected and are fine).

| File | Hunks | Character |
|---|---|---|
| `algorithms/measurement/multisweep.py` | 4 | API redesign vs. in-place edits; see §4.2 |
| `algorithms/measurement/fitting_nonlinear.py` | 2 | we moved it, they extended it |
| `algorithms/measurement/fitting.py` | 3 | we moved it, they extended it |
| `reference-notebooks/README.md` | 2 | prose, both rewrote |
| `Demos/simplified_tuning_flow.md` | 2 | prose, both rewrote |
| `algorithms/measurement/bias_kids.py` | 2 | **semantic**: tone grid constant; see §3.1 |
| `algorithms/measurement/take_netanal.py` | 1 | import block, trivial |
| `tools/periscope/app.py` | 1 | follows the tone grid decision |

### 3.1 The tone grid: decide this first, it is not a rename

`bias_kids.py` and `app.py` conflict on one constant, and the two sides are
**numerically different**:

| Side | Definition | Value |
|---|---|---|
| ours | `BASE_FREQUENCY = COMB_SAMPLING_FREQ / 256 / 2**12` (`core/transferfunctions.py`) | 596.046 Hz |
| main | `TONE_GRID_HZ = decimation_to_sampling(6) / 2` (`bias_kids.py`) | 298.023 Hz |

Exactly a factor of two apart. Main's version is a deliberate change with a
stated rationale (tones on multiples of half the slow stream's frame rate at
decimation 6, so intermodulation products land on the grid too). Ours is a
consolidation commit, `2bfecce` "One definition of the tone grid, and it is
BASE_FREQUENCY", which made `BASE_FREQUENCY` load-bearing across
`core/resonators.py` (`on_grid`, and every `BiasPoint` is quantized onto it
at construction), `algorithms/operation/apply_bias.py`,
`tools/periscope/app.py`, and the catalog round-trip.

**Decision 2026-09-08: keep ours.** Both values are valid tone grids; the
choice is style. `BASE_FREQUENCY` in `core/transferfunctions.py` stays the one
definition, and no file may re-declare the grid as a local constant (main's
`TONE_GRID_HZ` in `bias_kids.py` is replaced by the import). Changing the
number later is then one line.

Background: main has the physics argument;
our branch has the wider blast radius, and every saved catalog on our branch
has frequencies quantized to the 596 Hz grid. The likely resolution is to take
main's *value and rationale* but keep our *single definition*: define the grid
once in `core/transferfunctions.py` and point everything at it. Whoever
resolves this should confirm which grid the board actually wants, because
picking wrong silently moves every tone.

Main itself went back and forth: `298dde0` removed the rounding, saying 298 Hz
*"is the slow stream's frame rate at decimation 7, not a tone grid, and the
board's frequency resolution is far finer than any bias choice"*; `3087df9`
then reinstated it with the intermodulation rationale, noting a resonator
narrower than about 6 kHz gets a single grid step for its calibration and
leans on the fit's curvature correction. So the 298 Hz grid is a considered
choice on main, not an accident, and the argument for it is about
intermodulation products, not hardware resolution.

### 3.2 `fitting.py` and `fitting_nonlinear.py`: move vs. modify

Our branch relocated these algorithms into `rfmux/tuning/fits.py` and left
thin deprecated forwarders that re-import and re-export the moved names:

| File | base | ours | main |
|---|---|---|---|
| `fitting.py` | 1098 | **394** | 1140 |
| `fitting_nonlinear.py` | 854 | **301** | 831 |
| `multisweep.py` | 604 | **991** | 579 |

Main kept editing the algorithms in place. Git shows a few enormous hunks;
the real work is porting main's changes into `rfmux/tuning/fits.py` and
`rfmux/tuning/find_resonances.py` by hand. Main added no *new* top-level
functions to either file, so this is edits to existing algorithms, not new
API.

**Imports survive, on one condition.** Main's new `df_calibration.py` and
four of its tests import `s21_skewed`, `nonlinear_iq`, `fit_nonlinear_iq`,
`get_y_nonlinear` from the old locations. Our forwarders export all of them
via `from ...tuning.fits import (...)` plus `__all__`, so these imports work
*as long as the conflict resolution keeps our forwarder block*. (An earlier
draft of this survey called this a hard `ImportError`; it is not.)

**What main adds that we must not lose:** `find_resonances` gained a
`require_isolation` flag (commit `2eac96c`), with a Periscope checkbox, a
default in `app_runtime.py`, and 11 tests. Its `True` branch, drop *both*
members of a collided pair, is what our `find_resonances.py`
`_separation_pass` already does unconditionally. Its `False` default, scipy's
`distance=` thinning that keeps the deepest of a close group, **has no
equivalent in our implementation**. The signature merges cleanly (the merged
forwarder does accept `require_isolation`, confirmed in `daf5831`) but the
body does not, so post-merge the argument is accepted and ignored.

### 3.3 `multisweep.py`

The largest conflict. Hunk 4 is main's 235-line post-sweep block (TOD
acquisition, rotation, bifurcation flag, bias frequency) against our 3-line
`packed(results)` / `store.maybe_save` tail. This is not a textual problem;
it is §4.2, and the resolution follows from the decision made there.

### 3.4 `take_netanal.py`

Import block only; union the two sides. Main's change is a better
crest-factor headroom warning (`CREST_FACTOR * amp * sqrt(N/2)`), which is
orthogonal to our restructuring and should be kept.

### 3.5 The two prose conflicts

`reference-notebooks/README.md`: ours describes the four new demo notebooks;
ours should win with main's pulse-capture lines folded in.

`simplified_tuning_flow.md` is on the do-not-touch list for this branch, and
main changed it by 280 lines during review. Take main's side wholesale.

## 4. Conceptual conflicts: same question, different answers, different files

This is the important section. Since `1b1f5ce`, both branches kept working
on resonator tuning. Ours moved it into a new `rfmux/tuning/` package with a
new result shape and a catalog model; main kept building it out in place in
`algorithms/measurement/` and Periscope. Almost none of this overlaps
textually. After the merge, both systems will be in the tree, and main's will
read our data structures and get defaults instead of errors.

### 4.1 Where tuning code is allowed to live

Main's new `AGENTS.md` states a repo policy: *"Orchestration and anything
that can run headlessly lives in `rfmux/algorithms` or `rfmux/mock`;
Periscope is a thin caller."* It also lists "df calibration" as an
`rfmux/algorithms/` algorithm and documents the result shape as
`{detector_id: {iteration_index: {data + amplitude, direction}}}` with
`results[idx]['bias_frequency']` marked **CORRECT**.

Our branch put tuning in `rfmux/tuning/` and `rfmux/core/resonators.py`, and
its result shape is keyed by module then iteration then direction then
resonator *name*, with no `bias_frequency` on an entry at all. Adopting
`AGENTS.md` as written (survey step 7) would instruct every future agent
session that our branch's layout and data structure are wrong. `AGENTS.md`
needs to be rewritten for this branch, not adopted.

Meanwhile main made Periscope *less* of a thin caller for tuning: it now runs
a df calibration at startup (`app.py:_start_df_calibration`), and gained
bias-kids, find-resonances and df-cal dialogs and tasks with their own tests.

### 4.2 What a sweep is allowed to do on the way past

This is the deepest disagreement, and it is a contract, not an
implementation detail.

**Main:** `multisweep` does the sweep and then, per resonator, acquires a
TOD, rotates the IQ data, computes a bias frequency (`bias_frequency_method`
= `max-diq`/`min-s21`), and runs `identify_bifurcation`. Every entry comes
back carrying `is_bifurcated`, `bias_frequency`,
`recalculation_method_applied`, `rotation_tod`, `applied_rotation_degrees`,
`phase_degrees`. Main's `test_multisweep_then_bias.py` **asserts** that a
bare `multisweep` returns `is_bifurcated` and a finite `bias_frequency` on
every entry.

**Ours:** `sweep_results.py` schema bump 2 says, verbatim, *"sweeps stopped
rotating, re-centring and df-calibrating themselves,"* and `fits.py` says
*"no sweep fits itself on the way past ... the same argument that emptied
`multisweep` of its side jobs."* A sweep is measurement only; bifurcation,
bias frequency and calibration are separate analysis steps over saved data.

These cannot both be true. Every downstream piece of main's tuning code
(`bias_kids`, `df_calibration.py`, the Periscope digest, `AGENTS.md`) is
written against the first contract. Resolving hunk 4 of `multisweep.py` in
our favour, which is the only resolution consistent with our branch, makes
all of that code run against data that lacks the keys it expects.

### 4.3 Who decides a resonator is bifurcated, and when

| | main | ours |
|---|---|---|
| Detector | `identify_bifurcation`: a 5σ jump in one IQ trace | `bifurcated_by_derivative`, `bifurcated_by_hysteresis`, or `both` (the default) |
| Needs | one sweep, one direction | an `AmplitudeSchedule`; hysteresis needs both directions |
| Runs | inside `multisweep`, at sweep time | in `tuning/bias.py`, at bias time, over saved data |
| Stored as | `entry['is_bifurcated']` | `BiasPoint.bifurcated_at`, `BiasFinding`, `sweeps['bias_report']` |

Same word, two detectors, two moments. After the merge, main's `bias_kids`
still does `det_data.get('is_bifurcated', False)`. Our sweeps never set that
key, so **every amplitude reads as not bifurcated**, and its amplitude search
degenerates to "pick the loudest". No error is raised.

### 4.4 How the bias amplitude is chosen

**Main:** an amplitude is suitable when its sweep does not jump *and* the
fitted nonlinearity parameter `a` from the nonlinear fit is below
`nonlinear_threshold=0.77` (`bias_kids._suitable`). The criterion is a fit
parameter.

**Ours:** the amplitude one step below where a bifurcation detector first
fires (`find_bias_amplitude`). The criterion is a change between measured
steps; the fit's `a` is reported by `fit_sweeps` but is not used to bias.

Both are defensible physics. They are different answers to "which amplitude?"
and will choose differently on the same data. Someone has to decide whether
the `a < 0.77` criterion is wanted as a third `BIFURCATION_METHODS` entry, a
cross-check, or not at all.

### 4.5 How the bias frequency is chosen

Both sides offer the same two ideas, an IQ-derivative maximum and an |S21|
minimum, under different names and in different places:

| | main | ours |
|---|---|---|
| Names | `bias_frequency_method="max-diq"` / `"min-s21"` | `frequency_method="iq_derivative"` / `"minimum"` |
| Where | inside `multisweep`, then **refined onto the fitted curve** by `df_calibration.bias_frequency_from_fit` in `bias_kids` | `tuning/bias.find_bias_frequency`, over the raw sweep with splines |
| Recorded | `entry['bias_frequency']`, `entry['bias_frequency_source']` | `BiasPoint.frequency_hz` in the report's catalog |

Main's fit-refinement step (read the max-diq point off the fitted model
rather than the sampled grid) has no counterpart in ours and is worth
porting into `find_bias_frequency` as an option.

### 4.6 What a calibration is and where it lives

Same physical quantity, two representations, two homes.

**Main:** `df_calibration` is one complex number per channel, Hz/V:
*"multiply IQ in volts by it to get frequency shift + j dissipation."* It is
the inverse slope of the *fitted* resonator model at the bias frequency,
computed by `df_calibration.df_calibration_for_entry`, or measured standalone
by a new CRS macro `crs.measure_df_calibrations(module=...)` returning
`{channel: complex}`, with a spline fallback and a curvature correction for
stepped-tone measurements (`step_slope_correction`). Periscope calls the
macro at startup. `bias_kids` also has `measure_calibrations_by_step`.

**Ours:** `dI_df` and `dQ_df` in V/Hz, read off the *bias sweep* with
`iq_derivatives_at`, frozen onto a `BiasPoint` with the sweep they came from
(`bias_sweep`), and carried in the catalog. Moving the tone drops them by
construction. `bias_finding.md` notes the relation,
`df_calibration = 1 / (dI_df + j dQ_df)`, so the conversion is trivial; the
disagreement is about which object owns it and whether it comes from a fit or
from the data.

After the merge the `CRS` object will carry both `measure_df_calibrations`
and `apply_bias`, and nothing connects them.

### 4.7 Where fit results go

**Main:** flat keys written onto the sweep entry (`nonlinear_fit_params`,
`nonlinear_fit_success`, `skewed_fit_applied`, `gain_complex`,
`iq_gain_corrected`, ...). `ensure_fits` mutates entries in place and
`bias_kids` *"writes on"* the entries handed to it.

**Ours:** `entry["fits"]["skewed"]["params"]`, written only by an explicit
`fit_sweeps` call; nothing else modifies entries.

Every fit-reading helper in main's `df_calibration.py` (`_has_nonlinear_fit`,
`_fitted_model`, `fits_present`) looks for the flat keys and will report
"no fit" against our entries, which then triggers `ensure_fits` to fit again
and write the flat keys beside our nested ones.

### 4.8 The result shape

Our branch renamed and re-nested the payload:

| Key | ours | main |
|---|---|---|
| `iq_complex`, `phase_degrees` | — | ✓ |
| `iq_counts`, `iq_volts`, `channel` | ✓ | — |
| `bias_frequency`, `is_bifurcated`, `recalculation_method_applied` | — | ✓ |
| `rotation_tod`, `applied_rotation_degrees` | — | ✓ |
| `nonlinear_fit_params` etc. (flat) | — | ✓ |
| `fits` (nested) | ✓ | — |
| Top-level keyed by | module → `results` → iteration → direction → **name** | 1-based **detector index** |

`bias_kids.py` is the sharpest case. We touched it in one commit (the tone
grid, 4+/2-), main in nine (281+/123-), so it merges almost entirely to
main's version, which reads our data with `.get()` defaults and returns
wrong answers rather than raising (§4.3).

### 4.9 `find_resonances` default behaviour

Covered in §3.2. The conceptual part: main's default keeps the deepest peak
of a crowded group; ours always drops the whole group. On a crowded array the
two find different numbers of resonators, and ours does not have a switch.

### 4.10 Saving

Both pickle. Main's Periscope writes `multisweep_module1_HHMMSS.pkl` per
session and its new bias-kids dialog loads `.pkl` files expecting main's
shape. Ours writes `ipy_session_YYYYMMDD/multisweep_<stamp>_<label>.pkl` with
a `file_metadata` block so analyses can save back over their source. Not a
conflict in itself, but a file from one side is unreadable by the other's
loaders.

### 4.11 Periscope, on both sides

Both sides' `tasks.py` still call `crs.multisweep(center_frequencies=...,
bias_frequency_method=..., rotate_saved_data=...)`, which our `multisweep`
does not accept. So Periscope's multisweep task is **already broken on our
branch** before the merge; the merge adds more of it (df-cal at startup, the
dialogs, `app_runtime.py` reading `iq_complex`). Periscope is out of this
branch's scope, but the merge widens the gap and its new tests will fail.

## 5. What merges clean and breaks anyway

Main brings 79 new files. Git adds them without complaint. The ones that
interact with tuning:

- `algorithms/measurement/df_calibration.py`: imports fine (§3.2), runs
  against the wrong shape (§4.6, §4.7).
- Tests that will **fail** rather than error:
  - `test/algorithms/test_multisweep_result.py`, `test_multisweep_then_bias.py`:
    call `multisweep()` positionally and assert §4.2's contract.
  - `test/algorithms/test_find_resonances.py`: 11 tests on `require_isolation`.
  - `test/algorithms/test_df_calibration.py`, `test_df_calibration_entry.py`,
    `test_bias_kids_fits.py`, `test_nonlinear_fit_on_mock.py`.
  - `test/periscope/test_find_resonances_dialog.py`, `test_bias_kids_dialog.py`,
    `test_df_cal_task.py`, `test_close_stops_df_cal.py`.
- `test/algorithms/test_nonlinear_model.py` should pass: it imports the
  re-exported names and tests the model, not the shape.

No basename collision between our `test/tuning/test_find_resonances.py` and
main's `test/algorithms/test_find_resonances.py`; both directories have
`__init__.py`. Neither side deleted a file, and there are no add/add path
collisions between our 48 new files and main's 79.

## 6. Additions that need no merge work

Our whole `rfmux/tuning/` package, `core/resonators.py`,
`algorithms/operation/apply_bias.py`, `config.py`, `resonator_names/`, the
demo notebooks and the `example_plotting_*.py` modules arrive untouched.
Main's pulse-capture, fastrx, streamer and firmware work arrives untouched.

## 7. Decisions to make before touching the merge

These are the §4 questions, as choices. Each has to be answered once; the
textual resolutions in §3 follow from the answers.

1. **Tone grid** (§3.1): 298 Hz or 596 Hz, and defined where.
2. **Sweep contract** (§4.2): does `multisweep` stay measurement-only? If
   yes (the premise of this branch), main's post-sweep block, `bias_kids`
   additions, `df_calibration.py`, and the tests in §5 are re-expressed
   against `rfmux/tuning/` or retired, not merged as-is.
3. **Bifurcation and amplitude criteria** (§4.3, §4.4): does main's fitted
   `a < 0.77` become a third method, a cross-check, or nothing?
4. **Bias frequency from the fit** (§4.5): port main's fit-refinement into
   `find_bias_frequency`?
5. **Calibration** (§4.6): is `crs.measure_df_calibrations` kept as a
   standalone measurement that produces `BiasPoint` calibration fields, or
   retired in favour of the sweep-derived `dI_df`/`dQ_df`?
6. **`require_isolation=False`** (§4.9): add the deepest-of-group thinning
   to our `find_resonances` as the default, or document the difference.
7. **`AGENTS.md`** (§4.1): rewrite the "where code lives" and "results
   data structure" sections for this branch before adopting it.
8. **Periscope** (§4.11): accept that it stays broken for tuning on this
   branch, and say so in the todo file, or scope a port.

## 8. Suggested order

1. Answer §7. Most of it is small; item 2 is the one that decides the shape
   of everything else.
2. Graft and merge (§2). Resolve `simplified_tuning_flow.md` to main,
   `take_netanal.py` as a union, `reference-notebooks/README.md` to ours plus
   main's lines, `bias_kids.py`/`app.py` per the tone grid decision,
   `fitting*.py` keeping our forwarder blocks, `multisweep.py` to ours.
3. Port main's algorithm edits from `fitting.py` / `fitting_nonlinear.py`
   into `rfmux/tuning/fits.py` and `find_resonances.py`, including whatever
   was decided for `require_isolation`.
4. Deal with `df_calibration.py` and main's `bias_kids.py` additions per the
   §7 decisions. Do not leave them reading our shape with defaults.
5. Rewrite `AGENTS.md` for this branch.
6. Run the suite, expecting §5 to need attention. Pulse capture and
   `simplified_tuning_flow` stay out of scope: take main's side and do not
   test-run them.

## 9. What main improved, and what to do with each

Direction decided 2026-09-08: keep this branch's tuning infrastructure, fold
in main's fitting fixes, note the calibration-measurement work without
implementing it, leave Periscope alone for now and adapt it to
`rfmux/tuning/` later. After the merge, use the fits on the chosen bias
amplitude to judge the choice (§9.4). **Nothing that changes a DAC or ADC
phase is implemented on this branch**; where main does so it is noted in
§9.2 and left alone. Under those decisions the §4 items sort into four
piles.

### 9.1 Port now, into `rfmux/tuning/`

**The nonlinear resonator model is wrong on our branch.** `fits.py` still
solves the pre-fork equation `yg = y + a/(1+y²)` by Newton iteration
(`get_y_nonlinear`, `_solve_single_y`). Main's `da910f1` replaced it with
Swenson et al. 2013 eq. 13, `y = yg + a/(1+4y²)`, solved by bisection on the
bracket `[yg, yg+a]` (monotone below bifurcation, so it always converges)
followed by four clipped Newton steps. The factor of four in the denominator
is the physics; the bifurcation threshold `a = 4√3/9 ≈ 0.7698` is only
correct for that equation. Consequences on our branch today: the fitted `a`
does not mean what the threshold assumes, and the pull direction and
magnitude of `fr` are off. Main's `test_nonlinear_model.py` pins the
contract (eq. 13 satisfied to 1e-8, dip pulls *downward* by `0.5·fr/Qr` at
`a≈0.35`, fit recovers `a=0.35±0.02`) and `test_nonlinear_fit_on_mock.py`
checks the pull is downward and `a` rises with drive. Port the solver and
both test files into `rfmux/tuning/` and `test/tuning/`.

**Fit bounds and the `a` guard.** Main widened the bound on `a` to `[0, 0.9]`,
above bifurcation, so an over-driven sweep records a large `a` instead of
pinning at the bound. `nonlinear_fit_success` stays "residual < 0.1" and
does *not* consider `a`; consumers decline a fit with `a ≥ 4√3/9`
(`df_calibration._has_nonlinear_fit`). Adopt the same bound and put the
guard in `fits.py` where `SweepFit` decides `fitted`.

**Bias frequency read off the fitted curve.** `bias_frequency_from_fit`
evaluates the fitted model on a 4001-point grid over the sweep span and
takes the max-|dIQ/df| or min-|S21| there, instead of on the sampled grid
(2 kHz spacing at Periscope defaults, a third of a linewidth). Add as a
`FREQUENCY_METHODS` option, or as a refinement flag on `find_bias_frequency`,
using our `fits` entry.

**Calibration from the fitted slope.** Main's `df_calibration_for_entry`
is the inverse of the fitted model's central difference at `f_bias` over
`±1e-4·linewidth`, in Hz/V. Its docstring records the empirical reason:
*"differentiating a spline through the points instead scatters by a factor
of two on real sweeps."* That is exactly what our `iq_derivatives_at` does.
Offer the fitted slope as the preferred `dI_df`/`dQ_df` source when the
entry carries a nonlinear fit below bifurcation, spline as fallback. The
step-slope correction that goes with a *measured* calibration is part of
§9.2, not this.

**`find_resonances(require_isolation=False)`** (§3.2, §4.9): add scipy's
`distance=` thinning as the default path in `tuning/find_resonances.py`,
with our drop-both behaviour behind the flag. Port the 11 tests.

**`take_netanal` crest-factor warning** (§3.4): keep main's.

**`core/transferfunctions.py`**: auto-merges. Main's `CIC1_*`/`CIC2_*`
constants, `sampling_to_decimation`, `volts_squared_to_dbm` and
`apply_iq_conversion` all arrive cleanly; nothing to port.

### 9.2 Note for the Periscope port, later

**`crs.measure_df_calibrations` macro** (`df_calibration.py:288`,
`@macro(CRS, register=True)`). What it does: for every biased channel on a
module (default: `get_biased_channels`), read the tone frequency, then step
*all* channels together across `±span_hz/2` (default 20 kHz, 500 Hz
resolution, 41 points) with one batched `set_frequency` write and one
`get_samples(n_samples=10)` per point, and restore every tone in a `finally`.
Per channel it warns if `identify_bifurcation` fires, fits the nonlinear
model (`ensure_fits`), and returns `1 / convert_roc_to_volts(slope)` as one
complex Hz/V number; spline fallback with a single summary warning listing
which channels fell back. Returns `{channel: complex}`; channels with no
usable derivative are omitted rather than guessed.

How Periscope uses it: **mock mode only**. `app.py:_df_calibration_measurement`
returns `None` on hardware, with the stated reason that sweeping moves every
tone and is *"not something to do to a tuned array because someone picked a
units option; on hardware the calibration comes from bias_kids."* In mock
mode it runs in a background `DfCalibrationTask` at startup (or as a stage of
the launcher's build-progress dialog, `__main__.py:_build_with_progress`),
and the result lands in `self.df_calibrations[module]` through the same
handler a `bias_kids` result uses. Consumers: `app_runtime.py` converts live
I/Q to df with `apply_iq_conversion(volts, cal)` when `unit_mode == "df"`,
and `trigger_capture` stores pulses as Δf in hertz for calibrated channels.
For our branch the equivalent is a `ResonatorCatalog` whose bias points
carry `dI_df`/`dQ_df`; the macro is the "no catalog, just calibrate what is
playing" path and could later be re-expressed as a catalog-producing
measurement.

**`bias_kids` measured calibration: noted, not implemented.** Why main
measures a calibration the sweep already contains, from commit `3b16198`:
the fit's slope is the *model's*, and on the simulator its direction is up
to 4° off a real frequency step, because the Swenson form is not the
simulator's full kinetic-inductance physics, and a real detector disagrees
with the model in its own way. A few degrees of direction error leaks that
fraction of the frequency signal into dissipation and tilts the noise cloud
in the df view. Stepping the live tone measures the actual complex slope,
model-free, in the frame the ADC phase leaves the samples in, and on the
simulator lands within 0.7° of a true step. It also answers a question the
sweep cannot: the sweep was taken *before* biasing, at a possibly different
tone frequency and with the ADC phase zeroed, whereas the operating point
is where the tone actually sits now.

The same argument applies to our spline-derivative `dI_df`/`dQ_df` too: it
is model-free but reads a sweep from before biasing, on a coarser grid.
How main does it (`measure_calibrations_by_step`, `3b16198`, `298dde0`):
after biasing, step each tone `±calibration_step·linewidth` (default 5 %,
rounded to whole tone-grid steps; a fixed 300 Hz half-step read the slope
13 % low on a 2 kHz linewidth) in lockstep, read the module once at each,
take the inverse complex slope, multiply by `step_slope_correction` from
the fit's curvature. Two reads for the whole module. The fit's value is
kept as `df_calibration_fit`, and one warning names detectors where the two
disagree by more than 5° or a factor outside 0.7–1.4. The fit-derived
calibration is rotated by `exp(+j·phase)` to match the ADC frame. When it
is wanted here, it is an optional step of `apply_bias` (which already has
the catalog and the board) that overwrites `dI_df`/`dQ_df` on the
`BiasPoint` with a measured pair and records the fit's as the cross-check.

**ADC/DAC phase: noted, not implemented.** Main touches the ADC phase in
three places, all inherited into the tree by the merge but none wired into
`rfmux/tuning/` or `apply_bias`: (1) `bias_kids` (`7d03a12`) replaces the
72-step phase scan with a one-shot PCA of streamed (I, Q), sets it with
`ctx.set_phase(..., target=ADC)` when `optimize_phase=True`, and rotates the
fit-derived calibration by `exp(+j·phase)` to stay in the samples' frame;
(2) main's `multisweep` post-sweep block zeroes the ADC phase on the channels
it sweeps so the sweep's frame is phase zero (we take our `multisweep`, so
this does not come across); (3) `measure_calibrations_by_step` measures in
whatever frame the ADC phase leaves. Our branch's position is that a
calibration is read in the sweep's frame and the tone's phase is not
touched. If phase optimisation is ever wanted, it goes in as a separate
`tuning/` step that records the phase on the `BiasPoint` (there is already
an `iq_rotation_deg` field) rather than as a side effect of biasing.

**Periscope dialogs, tasks and tests** for find-resonances (`require_isolation`
checkbox), bias-kids and df-cal, plus `layouts.py`: leave as main has them.
They will not work against our data shape until the port.

### 9.3 Do not adopt

`multisweep`'s post-sweep block (TOD acquisition, rotation, `is_bifurcated`,
`bias_frequency` on the entry); `bias_kids`' amplitude rule via fitted
`a < 0.77` as the *primary* criterion (offer it as a method, §4.4);
`AGENTS.md`'s "Results Data Structure" section as written. Note that main
itself moved partway toward our design: `441bd29` "bias_kids works from one
chosen fit; the multisweep fits nothing" took the fitters out of
`multisweep`, so the remaining disagreement is rotation, bifurcation flag and
bias frequency, not fitting.

### 9.4 After the merge: the fits judge the bias amplitude

The plan once the merge is in, as a minor note: run the nonlinear fit on the
sweeps at the amplitude `find_bias_points` chose, and use the fitted `a`
to say whether that choice was a good one, alongside `flagged_because`. The
fit is a *check on* the amplitude selection, not the selector; the
derivative and hysteresis detectors keep that job. Main's rule for
comparison (`bias_kids._suitable`): a fit is consulted only when it
converged with finite `a`, the skewed fit has no `a` and is not asked, and
the threshold defaults to the bifurcation value `4√3/9`. An `a` near or
above that on the chosen step means the detectors let too much power
through; an `a` near zero on the loudest step means the ladder never
reached the nonlinear regime. This depends on §9.1's model fix: `a` from the
pre-fork equation is not comparable to `4√3/9`. `fit_sweeps_at_bias_amplitude`
already exists in `fits.py` and is the natural hook.

## 10. What the merge did (2026-09-08)

Merged `origin/main` (`efb4407`) into `tuning_headless_revamp` via the §2
graft. Merge base `1b1f5ce`, eight conflicts, resolved as follows.

| File | Resolution |
|---|---|
| `multisweep.py` | ours, whole file. Main's post-sweep block does not come across (§4.2). |
| `fitting.py`, `fitting_nonlinear.py` | ours (the forwarders), then main's changes ported into `rfmux/tuning/` (below). |
| `bias_kids.py` | main's version, with its local `TONE_GRID_HZ` replaced by `BASE_FREQUENCY` imported from `core/transferfunctions.py` (§3.1). |
| `tools/periscope/app.py` | ours for the grid line; main's import of the local constant removed. |
| `take_netanal.py` | union of the import blocks; main's crest-factor warning kept. |
| `reference-notebooks/README.md` | ours, with main's acquisition-tier sentence folded in. |
| `Demos/simplified_tuning_flow.md` | main's, wholesale. |

Ported into `rfmux/tuning/`:

- **`fits.py`**: `get_y_nonlinear` now solves Swenson et al. 2013 eq. 13
  (`y = yg + a/(1+4y²)`) by bisection then clipped Newton, as on main;
  `_solve_single_y` is gone. `BIFURCATION_A = 4√3/9` is exported (and
  re-exported by the `fitting_nonlinear` forwarder). Fit success is still the
  residual alone, exactly as main has it; `a` is a reading for consumers, not
  a failure. The `a ∈ [0, 0.9]` bound already matched main.
- **`find_resonances.py`**: `require_isolation` parameter, default `True`
  (our existing drop-both behaviour, unchanged). `False` is a new
  `_thinning_pass` that keeps the deepest of a close group and reports the
  rest in `.rejected`, in Hz. The `fitting.find_resonances` forwarder accepts
  `require_isolation` (default `False`, main's behaviour) and emits main's
  "Dropped N of M" warning when isolation is required. Main's eleven tests
  in `test/algorithms/test_find_resonances.py` pass against it.

Tests:

- `test/tuning/test_bias.py`: one assertion encoded the old equation's
  upward pull; it now asserts the downward pull.
- `test/algorithms/test_nonlinear_model.py`: adapted to our fitter's
  `(params, errors, residual)` return; the model assertions are unchanged
  and pass, as do `test_nonlinear_fit_on_mock.py`'s.
- `test/algorithms/test_bias_kids_fits.py`: the stepped-calibration test's
  expected step and tolerance now follow from `BASE_FREQUENCY` rather than
  assuming 298 Hz.
- `test/algorithms/test_multisweep_result.py` and
  `test_multisweep_then_bias.py`: skipped at module level with a reason
  pointing at §4.2. They pin main's multisweep contract, which this branch
  does not have.

Not touched, and known to be stale on this branch:

- `docs/release-notes/2026-09-pulse-capture-branch.md` still describes
  main's 298 Hz grid and quotes the old model equation. It is main's record
  of its own branch and was left as written.
- `df_calibration.py` defines its own `BIFURCATION_A` rather than importing
  ours; it and `bias_kids.py` still read main's flat fit keys and
  `iq_complex`, so they do not work against our sweep shape (§4.6–§4.8).
  Left for the Periscope port.
- `AGENTS.md` arrived as main wrote it (§4.1); rewriting it is a follow-up.

Results: `test/tuning`, `test/algorithms`, `test/core` all pass. Periscope
and mock suites pass. `test/pulse_capture` was not run (out of scope).

## 11. Test infrastructure, adopted (2026-09-08)

The branch now uses main's tiered test framework as-is: `--tier=portable`
for the synthetic tuning tests (all of `test/tuning/` except the flow module
carries `pytest.mark.portable`), `--tier=quick` for everything that needs no
UDP stream, `acquisition` for what does.

**Standard simulated array.** `rfmux/mock/standard_array.py` fixes one seeded
array (8 resonators, 1.00–1.10 GHz, seed 42, simulator-biased, bath 0.23 K
so Q ≈ 5e4) served over RPC only. `test/tuning/conftest.py` provides it
module-scoped. `test/notebooks/test_standard_mock_array.md` characterises it
end to end and runs in the quick tier. The exploration that produced it is
in that notebook; the two decisions it forced are the bath temperature (the
default 0.12 K gives 4 kHz linewidths a 1 kHz sweep grid cannot resolve) and
one-array-per-process (a second `load_session` detaches the first).

**Adapted by contract.** Main's `test_multisweep_then_bias.py` and
`test_multisweep_result.py`, skipped at the merge, are deleted; their contracts
live in `test/tuning/test_flow_on_standard_array.py` in this branch's terms:
a sweep entry carries no verdicts; every resonator gets a bias point at a
ladder rung, on the tone grid, inside its sweep; the calibration lives on the
`BiasPoint` with the sweep it was read from; bias finding on a bifurcating
ladder warns about nothing; `apply_bias` puts every tone where the catalog
says. Main's `test_nonlinear_fit_on_mock.py` checks (pull downward, `a`
rising with drive) are restated against the same ladder sweeps through
`fit_sweeps`, and a new one pins that the ladder brackets bifurcation for
every resonator. Main's original `test_nonlinear_fit_on_mock.py` still passes
through the forwarders and is left in place.

**Finding.** The notebook's detector-versus-fit table shows the bias finder's
detectors firing well below the rung where the fitted nonlinearity crosses
`BIFURCATION_A`, on all eight resonators. Recorded in `tuning_revamp_todo.md`
with the attribution experiment to run next; it is the §9.4 work item and was
not acted on.

**Demos.** `fitting_resonators.md`, `multisweep.md` and `resonator_catalogs.md`
are the revised drafts; the `_revised` files are gone. All five tuning demos
pass under `test_reference_demo_notebook` in the acquisition tier (the earlier
`multisweep.md` failure was a helper the revision no longer needs).

## 12. Tidying, and the legacy path marked (2026-09-08)

Removed: `estimate_and_remove_gain` (dead, in `fitting_nonlinear.py`), and the
two landed plans `tuning_multisweep_amplitudes_plan.md` and
`tuning_sweep_result_shape_plan.md` (history keeps them; the todo file's
references now say so). Kept on request: the `example_plotting_*.py`
companions and `QOL_IMPROVEMENTS.md`.

Marked deprecated, so a Periscope port cannot reach for them unnoticed:
`bias_kids.py`, `fitting.py`, `fitting_nonlinear.py` and `df_calibration.py`
each open with a **DEPRECATED — legacy Periscope tuning path** banner, and
every public function in them is wrapped by `_legacy.deprecated(replacement)`,
which emits a `DeprecationWarning` naming the replacement and puts a
"Deprecated. Use …" line at the head of the docstring. The CRS macros
(`measure_df_calibrations`) are wrapped beneath `@macro`, so the registered
method is the warning one. `find_resonances`, `fit_skewed_multisweep` and
`fit_nonlinear_iq_multisweep` already warned inline with fuller messages and
were left as they were. The re-exported fit functions (`s21_skewed`,
`nonlinear_iq`, …) are *not* deprecated: they are the current implementations
under their old import paths.

These four modules and their `test/algorithms/` tests go together when
Periscope is ported to `rfmux/tuning/` (§4.1, §4.11).
