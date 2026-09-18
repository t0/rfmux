# Track resonant frequency between sweep amplitudes

Status: deferred design, agreed 2026-09-17. No implementation is authorized
by this note. This is a future enhancement, separate from merging main.

## Merge decision

Do not retain main's per-amplitude fitted-center rerun implementation from
the changes through `be0bd49`. In particular, omit its
`fit_frequencies_by_amp`, `fit_frequencies_for`, `sweep_centres`,
`_fit_frequencies_by_amp`, and `use_fit_frequencies` plumbing and adapt or
omit tests that require those legacy interfaces. Preserve unrelated dialog
improvements where they fit this branch's architecture.

Main looks up a previous run's fit at the nearest amplitude. A rerun may
change its amplitudes and other measurement settings, so that table is not
the desired tracking mechanism. The proposed option follows the resonance
using measurements from the run currently in progress.

## Intended behavior

Expose an opt-in control labelled **Track resonant frequency between sweep
amplitudes**, with a matching headless argument. Proposed argument:
`track_resonant_frequency=False`; the spelling is not a published API yet.

1. Start from the caller's selected centers: catalog bias frequencies,
   previous recorded centers, or explicitly supplied centers. Do not change
   the catalog or its bias points.
2. Acquire every requested direction at the current amplitude step around
   the same centers. Do not recenter between directions: upward and downward
   traces at one amplitude must cover the same interval.
3. Fit each completed sweep with the existing nonlinear fitter, preserving
   its direction metadata. For each resonator, prefer the downward sweep's
   usable fit when available; otherwise use a usable upward fit.
4. Estimate that resonator's resonance frequency from the selected nonlinear
   fit and use it as its center at the next amplitude step. Each named
   resonator advances independently, including schedules with different
   amplitudes per resonator.
5. Follow the schedule in its supplied order. Increasing and decreasing
   amplitudes use the same rule, with no assumed direction of frequency
   drift and no nearest-amplitude lookup into a previous run.

This is one-step feedback: step N's measurement determines step N+1's
center. It does not predict the shift caused by the next amplitude. A
sufficiently large shift can still move a resonance outside the next span.

## Sweep metadata and provenance

Never overwrite an acquired entry's `original_center_frequency` with a fit
result. The current acquisition code sets that field to the center used to
construct that trace's frequency grid; retain that meaning. Tracking must
not mutate earlier entries or rewrite their measured frequency arrays.

Proposed additional field: `updated_center_frequency`, in Hz, records the
outgoing center selected after the current amplitude step for the next
step. Both direction entries for a resonator at a given step should record
the same outgoing decision once all directions are complete. The next
step's newly acquired entries record their actual acquisition center in
their own `original_center_frequency` fields.

Example: step 0 is acquired around 1,000,000,000 Hz and selects
999,980,000 Hz. Its original center remains 1,000,000,000 Hz and its updated
center is 999,980,000 Hz. Step 1 is acquired around 999,980,000 Hz; its own
original center is therefore 999,980,000 Hz.

Proposed convention: omit the new field when tracking is disabled. When
enabled, record the selected outgoing center even on the final step, but
document that no further acquisition was performed there. Record the
tracking option in `call_params`; preserve the initial requested centers
and catalog as input provenance. Readers must accept older files without
the optional field. Final saves and completed-step notifications must agree
on these decisions; partial traces cannot promise a center not yet chosen.

## Fitting, failures, and decisions still to settle

- Reuse `rfmux.tuning` nonlinear fitting and its quality checks. Do not add
  a second fitter or revive legacy flat fit fields. Fit results remain
  separate from raw measurement entries under the current architecture.
- Resolve the definition of the tracked frequency before implementation:
  the nonlinear model's `fr` parameter and the frequency of its driven
  transmission minimum need not coincide. Choose the quantity that tracks
  the measured resonance at that power; document and test the distinction.
- Proposed failure policy: try upward if downward is absent or unusable;
  if no usable fit remains, retain that resonator's current center. Report
  the fallback through existing progress/status mechanisms, without a
  dialog or aborting otherwise valid measurements. Do not silently switch
  to skewed fits or a previous run's result.
- Reject nonfinite or out-of-range centers using existing measurement and
  fit validation. Settle whether a fit outside the acquired interval is
  acceptable; do not invent an arbitrary drift cap. Revalidate the next
  step's NCO grouping and hardware frequency limits after centers change.
- Keep source direction and fallback reason available in diagnostics, with
  their storage location decided alongside the optional metadata field.
- The defaults and final-step metadata conventions above are proposed
  implementation choices; the agreed requirements are opt-in tracking,
  nonlinear fitting, downward preference, immutable acquired original
  centers, and omission of main's previous-run amplitude-table mechanism.

## Implementation boundary and verification

Place feedback orchestration in `rfmux/algorithms/measurement/multisweep.py`
at the amplitude-step boundary; use existing `rfmux/tuning` helpers for
analysis. Periscope passes the option and displays progress. Retain tone
shutdown between directions/steps and cleanup on errors or cancellation,
including errors during fitting. Update schema documentation, accessors,
packing/serialization, examples and GUI help together where affected.

Focused contract tests should establish:

- Disabled tracking preserves current centers and output conventions.
- A known fitted shift at step N changes step N+1's acquisition grid.
- Downward wins when both fits are usable, regardless of direction order;
  neither direction at the current step is recentered midway through it.
- Increasing/decreasing schedules and independent resonators track without
  reading a previous run's amplitude table or modifying catalog biases.
- Missing/failed fits retain the current center or use the documented
  direction fallback, with truthful diagnostics.
- Original centers and measured arrays remain immutable; updated centers
  round-trip through saved results, and old files remain readable.
- Updated targets respect hardware constraints, and fitting failure or
  cancellation still silences owned tones.

Use the `rfmux-tuning` environment. After deterministic contract tests,
measure a seeded mock amplitude ladder with tracking off/on: report fit
successes, resonance displacement within each span, and added fit time.
Do not claim improved tracking or performance before those measurements.
