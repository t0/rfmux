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
    "pack_results",
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
RESULTS_SCHEMA_VERSION = 2


def pack_results(
    sweeps: Mapping[int, Mapping[str, dict]],
    *,
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
        sweeps: ``{iteration: {direction: one multisweep return}}``, in the
            order measured.
        module: the module actually swept — resolved, never None.
        requested_module: the ``module`` argument as the caller passed it, which
            is None whenever it came from the catalog instead. Recorded as-is,
            because *call_params* says what was asked for and not what was
            worked out from it.
        catalog: the ``ResonatorCatalog`` swept, or None in frequency-list mode.
            Snapshotted with ``to_dict`` for provenance.

    Returns:
        dict: ``schema_version``, ``module``, ``call_params`` and ``results``.

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
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "module": module,
        "call_params": {
            "catalog": catalog.to_dict() if catalog is not None else None,
            "center_frequencies": (
                [float(f) for f in center_frequencies]
                if center_frequencies is not None
                else None
            ),
            "names": list(names) if names is not None else None,
            "amp_schedule": amp_schedule.to_dict(),
            "directions": list(directions),
            "span_hz": float(span_hz),
            "npoints_per_sweep": int(npoints_per_sweep),
            "nsamps": int(nsamps),
            "module": requested_module,
        },
        "results": {int(i): dict(by_direction) for i, by_direction in sweeps.items()},
    }


def _iterations(results: Mapping) -> dict:
    """The ``results`` block, with a useful error when handed the wrong dict."""
    try:
        return results["results"]
    except (TypeError, KeyError):
        raise TypeError(
            "Expected the dict multiamp_multisweep returned (with 'results' "
            "and 'call_params'), not one of its parts."
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
