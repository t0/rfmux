"""The shape a sweep comes back in, written and read in one place.

``multisweep`` produces the dict :func:`pack_multisweep` assembles; the readers
under it are the supported way to get things back out. One module owns both
ends, because a reader resolving ``schedule.steps[iteration]`` has to agree with the
packer about what a step means, and two files agreeing about one contract is one
file too many.

``take_netanal`` packs through :func:`pack_netanal` into the same shape, so
every driver in the package returns one container shape. What sits under a
direction differs — a sweep has a section per resonator, a netanal has the one
trace it measured — which is what ``measurement`` is in the output to say.

This lived in :mod:`rfmux.tuning.multisweep_amplitudes` while a schedule was the
only thing that produced it. It is not the schedule's shape any more — it is every
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
    "pack_multisweep",
    "pack_netanal",
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
#
# 4: take_netanal returns this shape too, and a module's output now says which
#    driver made it in 'measurement'. A netanal measures one trace rather than a
#    section per resonator, so under a direction it carries the arrays directly
#    where a sweep carries {name: section} — the one place the two differ, and
#    the reason a reader has to be able to tell them apart. The netanal's own
#    'iq_complex'/'phase_degrees' became 'iq_counts'/'iq_volts' on the way in.
#
# 5: multiamp_multisweep was folded into multisweep, which now takes an
#    AmplitudeSchedule as its 'amp' and a sequence as its 'sweep_direction'. So
#    'measurement' is 'multisweep' whether one amplitude was swept or twenty,
#    and call_params records the pair every sweep now has — 'amp_schedule' and
#    'directions' — in place of the 'amp'/'sweep_direction' a single sweep used
#    to record. A one-step schedule is the faithful record of amp=0.005; what
#    each resonator was actually probed at is, as before, 'sweep_amplitude' on
#    its own entry.
#
# 6: call_params always carries a catalog. A bare center_frequencies sweep used
#    to record None there, and multisweep now generates one from the list, so a
#    result says what array it is of whichever way it was asked for — which is
#    what lets find_bias_points work off the sweep alone, and why this is a bump
#    rather than a field quietly filling in: a reader that needs the catalog
#    cannot absorb a 5 that has none.
RESULTS_SCHEMA_VERSION = 6


# The iteration a netanal's one trace sits at. Not a placeholder: a netanal is
# one amplitude sweeping upward in frequency, so 0 is its number in a schedule of
# length one.
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
    the module that was actually swept. The catalog is the exception, and for
    the same reason ``amp_schedule`` is: it is the resolved form of either way
    of asking, so a bare ``center_frequencies`` sweep records both the list
    that was passed *and* the catalog it became.
    """
    return {
        "catalog": catalog.to_dict(),
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


def _packed(
    module_id: str,
    module: int,
    call_params: dict,
    results: dict,
    *,
    measurement: str,
) -> dict:
    """One module's output, in the container every driver returns.

    Always a container, even for the one module that is the usual case, so a
    caller who writes ``for module_id, module_sweeps in sweeps.items():`` has
    written the same code for one module and for four. A convenience that
    flattened the single-module case would make the common script differ from
    the general one.

    *measurement* names the driver: ``"multisweep"`` or ``"netanal"``. Two
    outputs that are structurally identical down to the direction and then are
    not — a netanal has no sections — so the readers below need a way to tell
    that is not sniffing ``call_params`` for ``span_hz``, which is the kind of
    test this shape exists to delete.
    """
    return {
        module_id: {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "measurement": measurement,
            "module": int(module),
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
    center_frequencies: Sequence[float] | None = None,
    names: Sequence[str] | None = None,
    requested_module: int | None = None,
) -> dict:
    """Assemble what ``multisweep`` returns.

    One packer for one sweep and for twenty, because they are the same
    measurement at different extents. A call that swept one amplitude in one
    direction arrives here as a *sweeps* of one iteration holding one direction
    — which is what it is, not a padded slot — so reading a result needs no
    knowledge of how wide the call that made it was.

    Args:
        sweeps: ``{iteration: {direction: {name: entry}}}``, in the order
            measured.
        module_id: the board-and-module identifier this comes back under, from
            ``crs.module[m].index()``.
        module: the module actually swept — resolved, never None.
        amp_schedule: the schedule the amplitudes came from, normalized — a
            bare ``amp=0.005`` reaches here as the one-step schedule it is.
            Snapshotted with ``to_dict`` for provenance.
        directions: the directions swept, in the order measured.
        requested_module: the ``module`` argument as the caller passed it, which
            is None whenever it came from the catalog instead, and the list
            itself for a call that fanned out over several. Recorded as-is,
            because *call_params* says what was asked for and not what was
            worked out from it.
        catalog: the ``ResonatorCatalog`` swept — required, and for a
            frequency-list sweep the one ``multisweep`` generated from the list
            rather than None. Snapshotted with ``to_dict``, which is the whole
            catalog and not a summary of it, so the array a sweep was taken
            from comes back off a file intact.

    Returns:
        dict: ``{module_id: output}``, one module's output holding
        ``schema_version``, ``measurement``, ``module``, ``call_params`` and
        ``results``.

        ``results`` is keyed by amplitude iteration, numbered from 0 in the
        order measured, and an iteration holds one entry per direction swept
        and nothing else.

        Nothing is duplicated into the iteration level. What a resonator was
        probed at is already ``sweep_amplitude`` in its own entry — see
        :func:`get_amplitudes_at_iteration` — and the step that produced it is
        ``call_params["amp_schedule"]["steps"][iteration]``. Sweep centres are
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
        measurement="multisweep",
    )


def pack_netanal(
    trace: Mapping,
    *,
    module_id: str,
    module: int,
    sweep_direction: str = "upward",
    amp: float,
    fmin: float,
    fmax: float,
    npoints: int,
    nsamps: int,
    max_chans: int,
    max_span: float,
    rotate_phase_to_0: bool,
    requested_module=None,
) -> dict:
    """Assemble what ``take_netanal`` returns.

    The same shape :func:`pack_multisweep` builds, holding the one wideband
    trace a netanal is where a sweep holds a section per resonator. The
    iteration and
    direction levels are kept — a netanal is one amplitude sweeping upward in
    frequency, which is what iteration 0 of ``"upward"`` means — so walking down
    to a measurement is the same walk whichever driver wrote the file.

    Args:
        trace: the ``frequencies``/``iq_counts``/``iq_volts`` arrays and the
            ``sweep_amplitude``/``sweep_direction`` scalars, already assembled.
        module_id: the board-and-module identifier this comes back under, from
            ``crs.module[m].index()``.
        module: the module actually measured — resolved, never None.
        requested_module: the ``module`` argument as the caller passed it, which
            is the list itself for a call that fanned out over several. Recorded
            as-is, because *call_params* says what was asked for and not what
            was worked out from it.

    Returns:
        dict: ``{module_id: output}``, one module's output holding
        ``schema_version``, ``measurement``, ``module``, ``call_params`` and
        ``results``.
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
        "module": requested_module,
    }

    return _packed(
        module_id,
        module,
        call_params,
        {SINGLE_SWEEP_ITERATION: {sweep_direction: dict(trace)}},
        measurement="netanal",
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

    Everything above a direction is identical between the two, and below it a
    netanal has the arrays where a sweep has ``{name: section}``. So a reader
    that walked a netanal would find ``frequencies`` and ``iq_counts`` where it
    expected resonator names and hand back arrays dressed as sweeps — no
    exception anywhere, just results that are wrong. Hence a guard rather than
    a docstring: this is the one confusion the shared shape makes possible,
    and it is silent.
    """
    if isinstance(obj, Mapping) and obj.get("measurement") == "netanal":
        raise TypeError(
            "This is a netanal, not a sweep. A netanal measures one wideband "
            "trace rather than a section per resonator, so there is nothing "
            "here to look up by name — its arrays are "
            "netanal['results'][0]['upward']. To find resonances in it, use "
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
