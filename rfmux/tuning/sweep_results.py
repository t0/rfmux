"""The shape a sweep comes back in, written and read in one place.

``multiamp_multisweep`` produces the dict :func:`pack_results` assembles; the
readers under it are the supported way to get things back out. One module owns
both ends, because a reader resolving ``ladder[iteration]`` has to agree with
the packer about what a rung means, and two files agreeing about one contract is
one file too many.

This lived in :mod:`rfmux.tuning.multisweep_amplitudes` while a ladder was the
only thing that produced it. It is not the ladder's shape any more — it is every
sweep's — so it has its own file, and the amplitudes module is back to being
about amplitudes.

Nothing here needs a board. Everything can be built, printed, validated and
unit-tested with no hardware and no GUI in sight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .multisweep_amplitudes import AmplitudeSchedule, _named

__all__ = [
    "RESULTS_SCHEMA_VERSION",
    "pack_sweep",
    "pack_results",
    "merge_modules",
    "collect_amplitude_iterations_for",
    "find_iteration_matching_amplitude",
    "get_amplitudes_at_iteration",
]


# Bumped when the packed dict changes shape in a way a reader cannot absorb.
#
# 2: sweeps stopped rotating, re-centring and df-calibrating themselves. A
#    section entry lost 'name', 'phase_degrees', 'bias_frequency',
#    'recalculation_method_applied', 'rotation_tod', 'applied_rotation_degrees',
#    'df_calibration' and 'calibrated_tod_df'; 'iq_complex'/'iq_complex_volts'
#    became 'iq_counts'/'iq_volts'; and call_params lost the three arguments
#    that drove all of it.
#
# 3: a plain multisweep returns this shape too, and everything is wrapped in a
#    dict keyed by module identifier. A single sweep is one iteration in one
#    direction — which is what it is — so the nesting no longer says which macro
#    produced it, and the multi-module form is keyed rather than a bare list.
RESULTS_SCHEMA_VERSION = 3


# The iteration a plain multisweep's one sweep sits at. Not a placeholder: one
# call is one amplitude step, so 0 is its number in a ladder of length one.
SINGLE_SWEEP_ITERATION = 0


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
    """The ``call_params`` fields both macros record, recorded identically.

    Verbatim throughout — what was asked for, not what was worked out from it —
    including the ``None``s, which is why *requested_module* is separate from
    the module that was actually swept.
    """
    return {
        "catalog": catalog.to_dict() if catalog is not None else None,
        "center_frequencies": (
            [float(f) for f in center_frequencies]
            if center_frequencies is not None
            else None
        ),
        "names": list(names) if names is not None else None,
        "span_hz": float(span_hz),
        "npoints_per_sweep": int(npoints_per_sweep),
        "nsamps": int(nsamps),
        "module": requested_module,
    }


def _packed(module_id: str, module: int, call_params: dict, results: dict) -> dict:
    """One module's envelope, in the container both macros return.

    Always a container, even for the one module that is the usual case, so a
    caller who writes ``for module_id, module_sweeps in sweeps.items():`` has
    written the same code for one module and for four. A convenience that
    flattened the single-module case would make the common script differ from
    the general one.
    """
    return {
        module_id: {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "module": int(module),
            "call_params": call_params,
            "results": results,
        }
    }


def pack_sweep(
    sections: Mapping[str, dict],
    *,
    module_id: str,
    module: int,
    sweep_direction: str,
    span_hz: float,
    npoints_per_sweep: int,
    nsamps: int,
    amp=None,
    catalog=None,
    center_frequencies: Sequence[float] | None = None,
    names: Sequence[str] | None = None,
    requested_module: int | None = None,
) -> dict:
    """Assemble what a single ``multisweep`` returns.

    The same shape :func:`pack_results` builds, holding the one iteration in the
    one direction that a single sweep is. Reading it needs no knowledge of which
    macro produced it, which is the point: the readers below and the fitters
    take either without asking.

    Args:
        sections: ``{name: entry}`` — what the sweep measured.
        module_id: the board-and-module identifier this comes back under, from
            ``crs.module[m].index()``.
        module: the module actually swept — resolved, never None.
        sweep_direction: the direction this sweep was taken in, which is the key
            *sections* sits under.
        amp: the ``amp`` argument verbatim — None, a number, a list or a
            mapping. What each resonator was actually probed at is already
            ``sweep_amplitude`` in its own entry, so nothing is lost by not
            resolving it here, and recording the request keeps *call_params*
            meaning what was asked for.
        requested_module: the ``module`` argument as the caller passed it, None
            whenever it came from the catalog instead.

    Returns:
        dict: ``{module_id: envelope}``, the envelope holding
        ``schema_version``, ``module``, ``call_params`` and ``results``.
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
    call_params["amp"] = amp
    call_params["sweep_direction"] = sweep_direction

    return _packed(
        module_id,
        module,
        call_params,
        {SINGLE_SWEEP_ITERATION: {sweep_direction: dict(sections)}},
    )


def merge_modules(containers) -> dict:
    """One container from several, for a sweep that ran on several modules.

    Each per-module call already returns a container of its own, so merging is a
    union — and a keyed one, which is what the multi-module return used to lack:
    it was a bare list, with nothing but argument order to say which element was
    which module.

    Raises:
        ValueError: on a repeated module identifier, which would otherwise
            overwrite a module's data with another's.
    """
    merged: dict = {}
    for container in containers:
        for module_id, envelope in container.items():
            if module_id in merged:
                raise ValueError(
                    f"{module_id!r} appears twice. Each module comes back under "
                    f"its own key, so a repeat would overwrite one module's "
                    f"data with another's."
                )
            merged[module_id] = envelope
    return merged


def _is_container(obj) -> bool:
    """Is this the whole return, keyed by module, rather than one envelope?

    An envelope carries ``results`` and ``call_params`` at the top; a container
    carries envelopes. Recognized only in order to be refused — nothing
    dispatches on it, so there is still exactly one accepted input everywhere.
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


def _refuse_container(obj) -> None:
    """Raise if handed the container where one module's envelope was wanted."""
    if _is_container(obj):
        keys = list(obj)
        raise TypeError(
            f"This is the whole sweep result, keyed by module "
            f"({_named(keys)}). Pass one module's data: sweeps[{keys[0]!r}]."
        )


def pack_results(
    sweeps: Mapping[int, Mapping[str, dict]],
    *,
    module_id: str,
    module: int,
    amp_schedule: AmplitudeSchedule,
    directions: Sequence[str],
    span_hz: float,
    npoints_per_sweep: int,
    nsamps: int,
    catalog=None,
    center_frequencies: Sequence[float] | None = None,
    names: Sequence[str] | None = None,
    requested_module: int | None = None,
) -> dict:
    """Assemble what ``multiamp_multisweep`` returns.

    Args:
        sweeps: ``{iteration: {direction: {name: entry}}}``, in the order
            measured.
        module_id: the board-and-module identifier this comes back under, from
            ``crs.module[m].index()``.
        module: the module actually swept — resolved, never None.
        requested_module: the ``module`` argument as the caller passed it, which
            is None whenever it came from the catalog instead. Recorded as-is,
            because *call_params* says what was asked for and not what was
            worked out from it.
        catalog: the ``ResonatorCatalog`` swept, or None in frequency-list mode.
            Snapshotted with ``to_dict`` for provenance.

    Returns:
        dict: ``{module_id: envelope}``, the envelope holding
        ``schema_version``, ``module``, ``call_params`` and ``results``.

        ``results`` is keyed by amplitude iteration, numbered from 0 in the
        order measured, and an iteration holds one entry per direction swept
        and nothing else.

        Nothing is duplicated into the iteration level. What a resonator was
        probed at is already ``sweep_amplitude`` in its own entry — see
        :func:`get_amplitudes_at_iteration` — and the rung that produced it is
        ``call_params["amp_schedule"]["ladder"][iteration]``. Sweep centres are
        recorded only as passed: a later step may re-centre between amplitudes,
        at which point a top-level copy would be a lie while each sweep's own
        ``original_center_frequency`` cannot be.
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
    )


def _iterations(results: Mapping) -> dict:
    """The ``results`` block, with a useful error when handed the wrong dict."""
    _refuse_container(results)
    try:
        return results["results"]
    except (TypeError, KeyError):
        raise TypeError(
            "Expected one module's sweep result (with 'results' and "
            "'call_params'), not one of its parts."
        ) from None


def _section_names(results: Mapping) -> list[str]:
    """Every section name that appears in the first sweep, in its order."""
    for by_direction in _iterations(results).values():
        for sections in by_direction.values():
            return list(sections)
    return []


def collect_amplitude_iterations_for(results: Mapping, name: str) -> dict:
    """Every sweep of one resonator, across the amplitude iterations.

    Args:
        results: what ``multiamp_multisweep`` returned.
        name: the resonator or section to pull out.

    Returns:
        dict: ``{iteration: {direction: sweep}}`` — the same shape as
        ``results["results"]``, one resonator deep, in the order measured.
        Measured order, not sorted by amplitude: an ``explicit`` ladder may run
        in any order, and re-sorting silently would lose the order things
        actually happened in.

    Raises:
        KeyError: if *name* was not swept.
    """
    collected = {}
    for iteration, by_direction in _iterations(results).items():
        entries = {
            direction: sections[name]
            for direction, sections in by_direction.items()
            if name in sections
        }
        if entries:
            collected[iteration] = entries

    if not collected:
        available = _section_names(results)
        raise KeyError(
            f"{name!r} was not swept. The section names in play are "
            f"{_named(available)}."
        )
    return collected


def get_amplitudes_at_iteration(results: Mapping, iteration: int) -> dict:
    """What every sweep was probed at on one iteration.

    Reads each sweep's own ``sweep_amplitude`` rather than a stored copy, which
    is why the packed dict does not carry one.

    Args:
        results: what ``multiamp_multisweep`` returned.
        iteration: which amplitude iteration.

    Returns:
        dict: ``{name: amplitude}`` in normalized DAC units.

    Raises:
        KeyError: if there is no such iteration.
    """
    iterations = _iterations(results)
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
    results: Mapping, name: str, amplitude: float | None = None
) -> int:
    """Which amplitude iteration probed *name* closest to *amplitude*.

    Args:
        results: what ``multiamp_multisweep`` returned.
        name: whose amplitudes to match against. Required, because a relative
            ladder gives every resonator its own: R0001 walking 1→2→4 µ and
            R0002 walking 3→6→12 µ share an iteration number and nothing else,
            so "the iteration at 4 µ" is only a question about one of them.
        amplitude: the amplitude to match, in normalized DAC units. Defaults to
            *name*'s own bias amplitude, read from the catalog snapshot in
            ``call_params`` — which is the usual question, "which iteration was
            taken where this resonator is actually biased?"

    Returns:
        int: the iteration number, for indexing ``results["results"]``.

    Nearest wins, and there is always a nearest — floats from a ladder rarely
    compare equal, so matching on equality would find nothing. A caller who
    needs the match to be close should check it:
    ``get_amplitudes_at_iteration(results, i)[name]``.

    Raises:
        KeyError: if *name* was not swept.
        ValueError: if *amplitude* is None and there is no catalog to take a
            bias amplitude from, or if nothing was measured.
    """
    if amplitude is None:
        amplitude = _bias_amplitude_of(results, name)

    per_iteration = {
        iteration: float(next(iter(entries.values()))["sweep_amplitude"])
        for iteration, entries in collect_amplitude_iterations_for(
            results, name
        ).items()
    }
    if not per_iteration:
        raise ValueError(f"No sweep sections for {name!r} to match against.")

    return min(per_iteration, key=lambda i: abs(per_iteration[i] - amplitude))


def _bias_amplitude_of(results: Mapping, name: str) -> float:
    """*name*'s bias amplitude, from the catalog snapshot in call_params."""
    # The one read that does not go through _iterations, so it needs its own
    # guard: a container has no call_params of its own, and without this it
    # would be reported as a sweep that had no catalog rather than as the
    # wrong dict.
    _refuse_container(results)

    catalog = results.get("call_params", {}).get("catalog")
    if catalog is None:
        raise ValueError(
            "No amplitude given and no catalog to take one from — this result "
            "came from a bare center_frequencies sweep, which has no bias "
            "amplitude. Pass amplitude= explicitly."
        )

    for r in catalog["resonators"]:
        if r["name"] == name:
            return float(r["bias"]["amplitude"])

    raise KeyError(
        f"{name!r} is not in the catalog this result was swept from. Its "
        f"resonators are {_named([r['name'] for r in catalog['resonators']])}."
    )
