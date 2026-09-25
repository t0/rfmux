"""Pack and read network analysis and multisweep results.

Both measurements return ``{module_id: block}``. Each block records the
measurement type, module, DAC scale, call parameters, and results. A network
analysis holds one trace; a multisweep holds ``results[step][direction][name]``.

The readers below take one module's multisweep output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .multisweep_amplitudes import AmplitudeSchedule, _named

__all__ = [
    "RESULTS_SCHEMA_VERSION",
    "DIRECTIONS",
    "resolve_direction",
    "pack_multisweep",
    "pack_netanal",
    "merge_modules",
    "collect_amplitude_iterations_for",
    "find_iteration_matching_amplitude",
    "get_amplitudes_at_iteration",
]


# Bump when the saved measurement format changes incompatibly.
RESULTS_SCHEMA_VERSION = 11


# The directions a sweep can run in. Here rather than in either driver, because
# both record one in the shape this module defines.
DIRECTIONS = ("upward", "downward")


def _call_params(
    *,
    catalog,
    center_frequencies,
    names,
    span_hz,
    npoints_per_sweep,
    nsamps,
    requested_module,
) -> dict:
    """Record requested sweep settings and a snapshot of the resolved catalog."""
    return {
        "catalog": catalog.to_dict(),
        "center_frequencies": (
            {name: float(f) for name, f in center_frequencies.items()}
            if isinstance(center_frequencies, Mapping) else
            [float(f) for f in center_frequencies]
            if center_frequencies is not None
            else None
        ),
        "names": list(names) if names is not None else None,
        "span_hz": float(span_hz),
        "npoints_per_sweep": int(npoints_per_sweep),
        # Capture tuning rows may not record the sample count.
        "nsamps": int(nsamps) if nsamps is not None else None,
        "module": requested_module,
    }


def _packed(
    module_id: str,
    module: int,
    call_params: dict,
    results: dict,
    *,
    measurement: str,
    dac_scale_dbm: float | None,
) -> dict:
    """Wrap one module's measurement in ``{module_id: block}``.

    ``dac_scale_dbm`` is the measured DAC full-scale power, or None if
    unavailable. It converts the recorded amplitude fractions to power.
    """
    return {
        module_id: {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "measurement": measurement,
            "module": int(module),
            "dac_scale_dbm": (
                None if dac_scale_dbm is None else float(dac_scale_dbm)),
            "call_params": call_params,
            "results": results,
        }
    }


def pack_multisweep(
    sweeps: Mapping[int, Mapping[str, dict]],
    *,
    module_id: str,
    module: int,
    amp_schedule: AmplitudeSchedule,
    directions: Sequence[str],
    span_hz: float,
    npoints_per_sweep: int,
    nsamps: int,
    catalog,
    center_frequencies: Sequence[float] | Mapping[str, float] | None = None,
    names: Sequence[str] | None = None,
    requested_module: int | None = None,
    dac_scale_dbm: float | None = None,
) -> dict:
    """Pack measured sweeps as ``{module_id: block}``.

    ``sweeps`` is ``{step: {direction: {name: entry}}}``, with steps numbered
    from zero in acquisition order. It becomes the block's ``results``.
    The block also holds ``schema_version``, ``measurement``, ``module``,
    ``dac_scale_dbm``, and ``call_params``.

    ``catalog`` and ``amp_schedule`` are the resolved objects and are recorded
    with ``to_dict()``. Other call parameters record the requested settings,
    including ``requested_module`` (which may be None or a module list).
    ``module`` and ``module_id`` identify the module actually measured.
    ``directions`` records acquisition order. ``dac_scale_dbm`` is the board's
    DAC full-scale power, or None when unavailable.

    Each sweep entry records its own amplitude and original centre. Use
    :func:`collect_amplitude_iterations_for` to read a resonator's sweeps.
    """
    call_params = _call_params(
        catalog=catalog,
        center_frequencies=center_frequencies,
        names=names,
        span_hz=span_hz,
        npoints_per_sweep=npoints_per_sweep,
        nsamps=nsamps,
        requested_module=requested_module,
    )
    call_params["amp_schedule"] = amp_schedule.to_dict()
    call_params["directions"] = list(directions)

    return _packed(
        module_id,
        module,
        call_params,
        {int(i): dict(by_direction) for i, by_direction in sweeps.items()},
        measurement="multisweep",
        dac_scale_dbm=dac_scale_dbm,
    )


def resolve_direction(sweep_direction) -> str:
    """Validate and return one sweep direction.

    A network-analysis call measures one direction. To measure both, make
    separate calls with ``"upward"`` and ``"downward"``.
    """
    if sweep_direction in DIRECTIONS:
        return sweep_direction

    if isinstance(sweep_direction, str):
        raise ValueError(
            f"Invalid sweep_direction: {sweep_direction!r}. Must be one of "
            f"{DIRECTIONS}."
        )

    raise TypeError(
        f"sweep_direction must be one of {DIRECTIONS}, got "
        f"{type(sweep_direction).__name__}. A netanal measures the band once "
        f"per call, so both directions is two calls."
    )


def pack_netanal(
    trace: Mapping,
    *,
    module_id: str,
    module: int,
    amp: float,
    fmin: float,
    fmax: float,
    npoints: int,
    nsamps: int,
    max_chans: int,
    max_span: float,
    rotate_phase_to_0: bool,
    sweep_direction: str,
    requested_module=None,
    dac_scale_dbm: float | None = None,
) -> dict:
    """Pack one network-analysis trace as ``{module_id: block}``.

    The block stores the trace directly under ``results``. The trace contains
    frequency, IQ counts, IQ volts, amplitude, power, and sweep direction.
    There are no amplitude-step or direction dictionaries around it.

    ``module`` and ``module_id`` identify the measured module. The remaining
    settings go into ``call_params``, including ``requested_module`` exactly
    as supplied by the caller. ``dac_scale_dbm`` records the board's DAC
    full-scale power, or None when unavailable.
    """
    call_params = {
        "amp": float(amp),
        "fmin": float(fmin),
        "fmax": float(fmax),
        "npoints": int(npoints),
        "nsamps": int(nsamps),
        "max_chans": int(max_chans),
        "max_span": float(max_span),
        "rotate_phase_to_0": bool(rotate_phase_to_0),
        "sweep_direction": resolve_direction(sweep_direction),
        "module": requested_module,
    }

    return _packed(
        module_id,
        module,
        call_params,
        dict(trace),
        measurement="netanal",
        dac_scale_dbm=dac_scale_dbm,
    )


def merge_modules(containers) -> dict:
    """Combine measurement containers, keeping their module identifiers.

    Raise ValueError if an identifier appears more than once.
    """
    merged: dict = {}
    for container in containers:
        for module_id, output in container.items():
            if module_id in merged:
                raise ValueError(
                    f"{module_id!r} appears twice. Each module comes back under "
                    f"its own key, so a repeat would overwrite one module's "
                    f"data with another's."
                )
            merged[module_id] = output
    return merged


def _is_container(obj) -> bool:
    """Return whether the input is a nonempty mapping of module result blocks.

    Each block must contain ``results`` and ``call_params``. A single module
    block has ``results`` at the top level and is not a container.
    """
    return (
        isinstance(obj, Mapping)
        and bool(obj)
        and "results" not in obj
        and all(
            isinstance(v, Mapping) and "results" in v and "call_params" in v
            for v in obj.values()
        )
    )


def _refuse_container(
    obj, *, what: str = "multisweep output", variable: str = "multisweep_output"
) -> None:
    """Reject a container where one module's result is required.

    Use ``what`` and ``variable`` to show the correct indexing in the error.
    """
    if _is_container(obj):
        keys = list(obj)
        raise TypeError(
            f"This is the whole {what}, keyed by module "
            f"({_named(keys)}). Pass one module's data: {variable}[{keys[0]!r}]."
        )


def _refuse_netanal(obj) -> None:
    """Reject a network-analysis block where multisweep results are required.

    Both have a ``results`` field, but only multisweeps organize it by
    amplitude step, direction, and resonator name.
    """
    if isinstance(obj, Mapping) and obj.get("measurement") == "netanal":
        raise TypeError(
            "This is a netanal, not a sweep. A netanal measures one wideband "
            "trace rather than a section per resonator, so there is nothing "
            "here to look up by name — its arrays are netanal['results']. To "
            "find resonances in it, use "
            "rfmux.tuning.find_resonances_in_netanal()."
        )


def collect_amplitude_iterations_for(
    ms_module_output: Mapping, name: str
) -> dict:
    """Return all measured sweeps for one resonator.

    The result is ``{iteration: {direction: sweep}}`` in acquisition order,
    which may differ from amplitude order. Sweep dictionaries are shared
    with the input. Raise KeyError if the name was not swept.
    """
    _refuse_container(ms_module_output)
    _refuse_netanal(ms_module_output)
    collected = {}
    for iteration, by_direction in ms_module_output["results"].items():
        entries = {
            direction: sections[name]
            for direction, sections in by_direction.items()
            if name in sections
        }
        if entries:
            collected[iteration] = entries

    if not collected:
        available = {
            section_name
            for by_direction in ms_module_output["results"].values()
            for sections in by_direction.values()
            for section_name in sections
        }
        raise KeyError(
            f"{name!r} was not swept. The section names in play are "
            f"{_named(sorted(available))}."
        )
    return collected


def get_amplitudes_at_iteration(
    ms_module_output: Mapping, iteration: int
) -> dict:
    """What every sweep was probed at on one iteration.

    Reads each sweep's own ``sweep_amplitude`` rather than a stored copy, which
    is why the packed dict does not carry one.

    Args:
        ms_module_output: one module's output from ``multisweep``.
        iteration: which amplitude iteration.

    Returns:
        dict: ``{name: amplitude}`` in normalized DAC units.

    Raises:
        KeyError: if there is no such iteration.
    """
    _refuse_container(ms_module_output)
    _refuse_netanal(ms_module_output)
    iterations = ms_module_output["results"]
    if iteration not in iterations:
        raise KeyError(
            f"No iteration {iteration}. This result has "
            f"{sorted(iterations)}."
        )

    # Every direction of one iteration was swept at the same amplitudes, so the
    # first one answers the question.
    for sections in iterations[iteration].values():
        return {name: float(s["sweep_amplitude"]) for name, s in sections.items()}
    return {}


def find_iteration_matching_amplitude(
    ms_module_output: Mapping, name: str, amplitude: float | None = None
) -> tuple[dict, int]:
    """Return the measured step nearest a resonator's target amplitude.

    ``amplitude`` is in fractions of DAC full scale. If omitted, use the
    resonator's bias amplitude from the catalog in ``call_params``.
    Matching is per resonator because amplitudes can differ within a step.

    Return ``({direction: sweep}, iteration)``. There is no maximum matching
    distance; inspect a returned sweep's ``sweep_amplitude`` if needed.
    Equal-distance matches use the first step in acquisition order.

    Raise KeyError if the name was not swept or the recorded catalog lacks
    its bias amplitude, or ValueError if no amplitude can be matched.
    """
    collected = collect_amplitude_iterations_for(ms_module_output, name)
    iteration = _iteration_matching_amplitude(
        ms_module_output, name, amplitude, collected
    )
    return collected[iteration], iteration


def _iteration_matching_amplitude(
    ms_module_output: Mapping,
    name: str,
    amplitude: float | None,
    collected: Mapping | None = None,
) -> int:
    """Return only the nearest amplitude-step index, for fitting and other readers."""
    if amplitude is None:
        resonators = ms_module_output["call_params"]["catalog"]["resonators"]
        amplitude = float(resonators[name]["bias"]["amplitude"])
    if collected is None:
        collected = collect_amplitude_iterations_for(ms_module_output, name)

    per_iteration = {
        iteration: float(next(iter(entries.values()))["sweep_amplitude"])
        for iteration, entries in collected.items()
    }
    if not per_iteration:
        raise ValueError(f"No sweep sections for {name!r} to match against.")

    return min(per_iteration, key=lambda i: abs(per_iteration[i] - amplitude))
