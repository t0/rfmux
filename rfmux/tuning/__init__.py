"""Analyze measured sweeps, fit resonators, and choose bias points.

Analysis functions take one module's result, selected from the container
returned by ``crs.take_netanal`` or ``crs.multisweep``::

    from rfmux.tuning import find_resonances_in_netanal, find_bias_points

    netanal = await crs.take_netanal(module=2, amp=0.001, fmin=1e9, fmax=2e9)
    search = find_resonances_in_netanal(netanal[crs.module[2].index()])
    catalog = search.to_catalog(module=2, amplitude=0.001)
    sweeps = await crs.multisweep(catalog)
    report = find_bias_points(
        sweeps[crs.module[2].index()], amplitude_method="derivative")

``report.catalog`` contains the chosen bias points; ``report.flagged`` lists
points needing review. Apply them with ``await crs.apply_bias(report.catalog)``.
See the reference notebooks for amplitude schedules and fitting examples.
"""

from .bias import (
    BIFURCATION_METHODS,
    FREQUENCY_METHODS,
    HYSTERESIS_COMPARISONS,
    NEEDS_BOTH_DIRECTIONS,
    AmplitudeChoice,
    BiasFinding,
    BiasReport,
    BifurcationCheck,
    bifurcated_by_derivative,
    bifurcated_by_either,
    bifurcated_by_hysteresis,
    find_bias_amplitude,
    find_bias_frequency,
    find_bias_points,
    iq_arc_speed,
    iq_derivatives,
    iq_derivatives_at,
    normalized_arc_speed,
    hysteresis_separation,
)
from .find_resonances import (
    ResonanceCandidate,
    ResonanceSearch,
    find_resonances,
    find_resonances_in_netanal,
    find_sweeps_with_nearby_resonances,
    magnitude_db,
    netanal_trace,
)
from .fits import (
    FIT_PARAMS,
    MODELS,
    FitFailed,
    FitReport,
    SweepFit,
    centered_iq,
    collect_fit_params,
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
from . import store
from .tuning_record import (catalog_from_tuning, multisweep_from_tuning,
                            tuning_rows)
from .sweep_results import (
    RESULTS_SCHEMA_VERSION,
    collect_amplitude_iterations_for,
    find_iteration_matching_amplitude,
    get_amplitudes_at_iteration,
    merge_modules,
    pack_multisweep,
)

__all__ = [
    "BIFURCATION_METHODS",
    "FREQUENCY_METHODS",
    "HYSTERESIS_COMPARISONS",
    "NEEDS_BOTH_DIRECTIONS",
    "AmplitudeChoice",
    "BiasFinding",
    "BiasReport",
    "BifurcationCheck",
    "bifurcated_by_derivative",
    "bifurcated_by_either",
    "bifurcated_by_hysteresis",
    "find_bias_amplitude",
    "find_bias_frequency",
    "find_bias_points",
    "iq_arc_speed",
    "iq_derivatives",
    "iq_derivatives_at",
    "normalized_arc_speed",
    "hysteresis_separation",
    "ResonanceCandidate",
    "ResonanceSearch",
    "find_resonances",
    "find_resonances_in_netanal",
    "find_sweeps_with_nearby_resonances",
    "magnitude_db",
    "netanal_trace",
    "FIT_PARAMS",
    "MODELS",
    "FitFailed",
    "FitReport",
    "SweepFit",
    "centered_iq",
    "collect_fit_params",
    "fit_section",
    "fit_sweeps",
    "fit_sweeps_at_bias_amplitude",
    "gain_corrected_iq",
    "nonlinear_model_iq",
    "skewed_model_magnitude",
    "AmplitudeSchedule",
    "AmplitudeStep",
    # Keep file operations namespaced as store.save() and store.load().
    "store",
    "RESULTS_SCHEMA_VERSION",
    "collect_amplitude_iterations_for",
    "find_iteration_matching_amplitude",
    "get_amplitudes_at_iteration",
    "merge_modules",
    "pack_multisweep",
    "catalog_from_tuning",
    "multisweep_from_tuning",
    "tuning_rows",
]
