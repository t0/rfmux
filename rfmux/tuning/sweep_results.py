"""Pack and read network analysis and multisweep results.

Both measurements return ``{module_id: block}``. Each block records the
measurement type, module, DAC scale, call parameters, and results. A network
analysis holds one trace; a multisweep holds ``results[step][direction][name]``.

The readers below take one module's block.
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
RESULTS_SCHEMA_VERSION = 9


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
        # None where the sweep came from somewhere that does not record it --
        # a capture file's tuning group. Absent, rather than a number nothing
        # measured.
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
    """One direction, validated — what a netanal measures per call.

    A netanal is a comb: up to a thousand tones are on at once, so a call
    sweeps the band once, in one direction, and both directions is two calls
    whose results a caller keeps side by side. Hence one string in and one
    string out, and a sequence refused with a message that says so rather than
    quietly measuring its first element.
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
    """Assemble what ``take_netanal`` returns.

    The container :func:`pack_multisweep` builds, with the one wideband trace a
    netanal is under ``results`` where a sweep has its amplitude iterations. No
    iteration and no direction key above it: a netanal has no amplitude
    schedule and measures one direction per call, so those levels could only
    ever be constants a reader had to type. Which direction it was is beside
    the arrays, in the trace's own ``sweep_direction``.

    Args:
        trace: the ``frequencies``/``iq_counts``/``iq_volts`` arrays and the
            ``sweep_amplitude``/``sweep_amplitude_dbm``/``sweep_direction``
            scalars, already assembled.
        module_id: the board-and-module identifier this comes back under, from
            ``crs.module[m].index()``.
        module: the module actually measured — resolved, never None.
        sweep_direction: the direction measured, recorded in ``call_params``.
            The copy the trace carries is what a reader of the arrays wants;
            this one is the argument, alongside every other argument.
        requested_module: the ``module`` argument as the caller passed it, which
            is the list itself for a call that fanned out over several. Recorded
            as-is, because *call_params* says what was asked for and not what
            was worked out from it.
        dac_scale_dbm: what DAC full scale was worth on this module, read from
            the board as the netanal was taken. The trace's ``sweep_amplitude``
            is a fraction of it.

    Returns:
        dict: ``{module_id: output}``, one module's output holding
        ``schema_version``, ``measurement``, ``module``, ``dac_scale_dbm``,
        ``call_params`` and ``results``.
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
    """Is this the whole return, keyed by module, rather than one module's?

    One module's output carries ``results`` and ``call_params`` at the top; a
    container carries those. Recognized only in order to be refused — nothing
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


def _refuse_container(obj, *, what: str = "sweep result", variable: str = "sweeps") -> None:
    """Raise if handed the container where one module's output was wanted.

    *what* and *variable* name the measurement in the message, since every
    driver returns this shape: a netanal handed to the resonance finder wants
    to be told about ``netanal[module_id]``, not ``sweeps[module_id]``.
    """
    if _is_container(obj):
        keys = list(obj)
        raise TypeError(
            f"This is the whole {what}, keyed by module "
            f"({_named(keys)}). Pass one module's data: {variable}[{keys[0]!r}]."
        )


def _refuse_netanal(obj) -> None:
    """Raise if handed a netanal's output where a sweep's was wanted.

    Everything down to ``results`` is identical between the two, and a netanal
    has the arrays there where a sweep has its amplitude iterations. So a reader
    that walked a netanal would find ``frequencies`` and ``iq_counts`` where it
    expected iteration numbers and hand back arrays dressed as sweeps — no
    exception anywhere, just results that are wrong. Hence a guard rather than
    a docstring: this is the one confusion the shared container makes possible,
    and it is silent.
    """
    if isinstance(obj, Mapping) and obj.get("measurement") == "netanal":
        raise TypeError(
            "This is a netanal, not a sweep. A netanal measures one wideband "
            "trace rather than a section per resonator, so there is nothing "
            "here to look up by name — its arrays are netanal['results']. To "
            "find resonances in it, use "
            "rfmux.tuning.find_resonances_in_netanal()."
        )


def _iterations(results: Mapping) -> dict:
    """The ``results`` block, with a useful error when handed the wrong dict."""
    _refuse_container(results)
    _refuse_netanal(results)
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
        results: what ``multisweep`` returned, for a single module.
        name: the resonator or section to pull out.

    Returns:
        dict: ``{iteration: {direction: sweep}}`` — the same shape as
        ``results["results"]``, one resonator deep, in the order measured.
        Measured order, not sorted by amplitude: an ``explicit`` schedule may run
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
        results: what ``multisweep`` returned, for a single module.
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
) -> tuple[dict, int]:
    """The sweep of *name* taken closest to *amplitude*.

    Args:
        results: what ``multisweep`` returned, for a single module.
        name: whose amplitudes to match against. Required, because a relative
            schedule gives every resonator its own: BOTA walking 1→2→4 µ and
            KOZR walking 3→6→12 µ share an iteration number and nothing else,
            so "the iteration at 4 µ" is only a question about one of them.
        amplitude: the amplitude to match, in normalized DAC units. Defaults to
            *name*'s own bias amplitude, read from the catalog snapshot in
            ``call_params`` — which is the usual question, "which iteration was
            taken where this resonator is actually biased?"

    Returns:
        tuple: ``({direction: sweep}, iteration)`` — the matching sweeps, one
        per direction measured, and the iteration they were taken at. The sweep
        comes first because it is what a caller wants next; the number is there
        for indexing anything else by the same step, and can be dropped with
        ``sweeps, _ =``.

    Nearest wins, and there is always a nearest — floats from a schedule rarely
    compare equal, so matching on equality would find nothing. A caller who
    needs the match to be close can read it off the sweeps it got back:
    ``sweeps["upward"]["sweep_amplitude"]``.

    Raises:
        KeyError: if *name* was not swept.
        ValueError: if nothing was measured, or if *amplitude* is None and the
            result records no catalog to take a bias amplitude from — which
            only a file older than schema_version 6 does.
    """
    collected = collect_amplitude_iterations_for(results, name)
    iteration = _iteration_matching_amplitude(results, name, amplitude, collected)
    return collected[iteration], iteration


def _iteration_matching_amplitude(
    results: Mapping,
    name: str,
    amplitude: float | None,
    collected: Mapping | None = None,
) -> int:
    """Just the iteration number, for callers indexing by it.

    The matching itself, kept apart from the entry the public reader hands
    back: :func:`~rfmux.tuning.fits.fit_sweeps_at_bias` compares one against
    every section's iteration and has no use for the sweep.
    """
    if amplitude is None:
        amplitude = _bias_amplitude_of(results, name)
    if collected is None:
        collected = collect_amplitude_iterations_for(results, name)

    per_iteration = {
        iteration: float(next(iter(entries.values()))["sweep_amplitude"])
        for iteration, entries in collected.items()
    }
    if not per_iteration:
        raise ValueError(f"No sweep sections for {name!r} to match against.")

    return min(per_iteration, key=lambda i: abs(per_iteration[i] - amplitude))


def _bias_amplitude_of(results: Mapping, name: str) -> float:
    """*name*'s bias amplitude, from the catalog snapshot in call_params."""
    # The one read that does not go through _iterations, so it needs its own
    # guards: a container has no call_params of its own, and a netanal's have no
    # catalog in them, so without these either would be reported as a sweep that
    # had no catalog rather than as the wrong dict.
    _refuse_container(results)
    _refuse_netanal(results)

    catalog = results.get("call_params", {}).get("catalog")
    if catalog is None:
        raise ValueError(
            "No amplitude given and no catalog to take one from. Every "
            "multisweep records one since schema_version 6, so this is an "
            "older result — a bare center_frequencies sweep from back when "
            "those had no catalog at all. Pass amplitude= explicitly."
        )

    # Keyed by name since catalog schema_version 2, and a list of entries each
    # carrying their own name before that. Absorbing the old shape is why the
    # snapshot changing did not have to move RESULTS_SCHEMA_VERSION.
    resonators = catalog["resonators"]
    if not isinstance(resonators, dict):
        resonators = {rd["name"]: rd for rd in resonators}

    if name not in resonators:
        raise KeyError(
            f"{name!r} is not in the catalog this result was swept from. Its "
            f"resonators are {_named(list(resonators))}."
        )
    return float(resonators[name]["bias"]["amplitude"])
