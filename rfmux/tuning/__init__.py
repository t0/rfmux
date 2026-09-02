"""
Tuning: turning sweeps into a tuned array, in plain Python.

This is analysis, not a measurement.  Nothing in here talks to a board,
registers a ``@macro`` or imports :class:`~rfmux.core.schema.CRS` — each
module takes data in and hands data back, so every step can be run on a
saved sweep as easily as on a live one.  The operations that *drive* the
board stay in ``rfmux.algorithms.measurement`` (``crs.take_netanal(...)``,
``crs.multisweep(...)``), and Periscope is a third caller.  That is why
the package sits beside :mod:`rfmux.pulse_capture` rather than under
``algorithms``.

The layers, in the order a tuning run uses them::

    find_resonances       locate the dips in a network-analysis sweep
    multisweep_amplitudes decide the amplitude steps of a multi-amplitude sweep
    sweep_results         pack what a sweep measured, and read it back out
    fits                  fit resonator models to the sweeps that came back

The array bookkeeping those steps pass between each other —
``Resonator``, ``BiasPoint``, ``ResonatorCatalog`` — lives in
:mod:`rfmux.core.resonators`, because a typed resonator is useful well
beyond tuning.

Typical headless use::

    from rfmux.tuning import find_resonances_in_netanal, fit_sweeps

    netanal = await crs.take_netanal(module=2, amp=0.001, fmin=1e9, fmax=2e9)
    found = find_resonances_in_netanal(netanal, min_dip_depth_db=1.0)
    catalog = found.to_catalog(module=2, amplitude=0.001)

    sweeps = await crs.multiamp_multisweep(catalog, span_hz=200e3,
                                           npoints_per_sweep=101)
    fit_sweeps(sweeps)   # writes each sweep's fits alongside the sweep

See ``tuning_refactor_design.md`` in the repository root for the plan this
package is being built out against.
"""

from .find_resonances import (
    ResonanceCandidate,
    ResonanceSearch,
    find_resonances,
    find_resonances_in_netanal,
    magnitude_db,
)
from .fits import (
    MODELS,
    FitFailed,
    FitReport,
    SweepFit,
    centered_iq,
    fit_section,
    fit_sweeps,
    fit_sweeps_at_bias_amplitude,
    gain_corrected_iq,
    nonlinear_model_iq,
    skewed_model_magnitude,
)
from .multisweep_amplitudes import (
    AmplitudeSchedule,
    AmplitudeStep,
)
from .sweep_results import (
    RESULTS_SCHEMA_VERSION,
    collect_amplitude_iterations_for,
    find_iteration_matching_amplitude,
    get_amplitudes_at_iteration,
    merge_modules,
    pack_results,
    pack_sweep,
)

__all__ = [
    "ResonanceCandidate",
    "ResonanceSearch",
    "find_resonances",
    "find_resonances_in_netanal",
    "magnitude_db",
    "MODELS",
    "FitFailed",
    "FitReport",
    "SweepFit",
    "centered_iq",
    "fit_section",
    "fit_sweeps",
    "fit_sweeps_at_bias_amplitude",
    "gain_corrected_iq",
    "nonlinear_model_iq",
    "skewed_model_magnitude",
    "AmplitudeSchedule",
    "AmplitudeStep",
    "RESULTS_SCHEMA_VERSION",
    "collect_amplitude_iterations_for",
    "find_iteration_matching_amplitude",
    "get_amplitudes_at_iteration",
    "merge_modules",
    "pack_results",
    "pack_sweep",
]
