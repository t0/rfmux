"""multiamp_multisweep: one array, one ladder of probe amplitudes.

The driver over :func:`~rfmux.algorithms.measurement.multisweep.multisweep`.  It
walks the amplitude steps an
:class:`~rfmux.tuning.multisweep_amplitudes.AmplitudeSchedule` produces, sweeps
each one in each requested direction, and returns every sweep in one dict.

Everything it *decides* is computed in :mod:`rfmux.tuning.multisweep_amplitudes`
and everything it *returns* is shaped by :mod:`rfmux.tuning.sweep_results`,
neither of which needs a board; everything it *does* goes through
``crs.multisweep``, which is the only hardware call in this module.  What is left
here is the loop and the callbacks — which is the point: a driver that also knew
how to build a ladder would be two things.

Step outer, direction inner.  Each step's up-and-down pair is measured together
and the amplitude marches monotonically, which is what a bifurcation walk wants.

It mutates nothing.  Choosing an operating amplitude from the ladder is bias
finding's job, fitting is the fitters', writing files is ``store.py``'s.
"""

from collections.abc import Sequence

from ...core.hardware_map import macro
from ...core.schema import CRS
from ...core.resonators import ResonatorCatalog
from ...tuning import store
from ...tuning.multisweep_amplitudes import AmplitudeSchedule
from ...tuning.sweep_results import pack_results
from .multisweep import _resolve_section_names

DIRECTIONS = ("upward", "downward")


def _resolve_directions(directions) -> tuple[str, ...]:
    """Check the direction axis and freeze it.

    An explicit tuple rather than the magic string ``"both"``, so the product
    below is honestly a product and each sweep is labelled with both of its
    coordinates.
    """
    if isinstance(directions, str):
        raise TypeError(
            f"directions={directions!r}: pass a sequence, not a single string "
            f"— ({directions!r},) for one direction, {DIRECTIONS} for both. "
            f"(A bare string would read as a sequence of characters.)"
        )
    if not isinstance(directions, Sequence):
        raise TypeError(
            f"directions must be a sequence of {DIRECTIONS}, got "
            f"{type(directions).__name__}."
        )

    resolved = tuple(directions)
    if not resolved:
        raise ValueError(
            f"directions is empty: nothing would be measured. Pass at least "
            f"one of {DIRECTIONS}."
        )
    unknown = [d for d in resolved if d not in DIRECTIONS]
    if unknown:
        raise ValueError(
            f"Unknown sweep direction(s) {unknown}. Must be one or both of "
            f"{DIRECTIONS}."
        )
    repeated = sorted({d for d in resolved if resolved.count(d) > 1})
    if repeated:
        raise ValueError(
            f"directions repeats {repeated}: each direction is one key in the "
            f"result, so a repeat would overwrite itself rather than measure "
            f"twice."
        )
    return resolved


def _resolve_module(catalog, center_frequencies, module) -> int:
    """One module per call, and say so.

    ``multisweep`` accepts a list of modules alongside ``center_frequencies``
    and returns a list of dicts. A ladder over that would be a list of ladders,
    so this driver stays per-module: a catalog belongs to one module anyway, and
    four modules is four calls (or one ``asyncio.gather``).
    """
    if isinstance(module, (list, tuple)):
        raise ValueError(
            f"module={list(module)}: multiamp_multisweep runs one module per "
            f"call. Sweeping several means one call each — gather them if you "
            f"want them concurrent."
        )

    if catalog is not None:
        if module is None:
            return catalog.module
        if module != catalog.module:
            raise ValueError(
                f"module={module} does not match the catalog's module "
                f"({catalog.module})."
            )
        return module

    if module is None:
        raise ValueError(
            "module is required when sweeping center_frequencies."
        )
    return module


@macro(CRS, register=True)
async def multiamp_multisweep(
    crs: CRS,
    catalog: ResonatorCatalog | None = None,
    *,
    span_hz: float,
    npoints_per_sweep: int,
    amp_schedule: AmplitudeSchedule | None = None,
    directions: Sequence[str] = ("upward",),
    nsamps: int = 10,
    center_frequencies: list[float] | None = None,
    names: list[str] | None = None,
    module: int | None = None,
    progress_callback=None,
    data_callback=None,
    sweep_callback=None,
    save=None,
    label=None,
):
    """Sweep one array at a ladder of probe amplitudes.

    One call per amplitude step per direction, all of them through
    ``crs.multisweep``, returned in a single dict.

    The amplitudes come from *amp_schedule* and nowhere else — there is no
    ``amp`` argument here, because a ladder and a per-call override would be two
    ways to say the same thing::

        from rfmux.tuning import AmplitudeSchedule

        results = await crs.multiamp_multisweep(
            catalog,
            span_hz=200e3,
            npoints_per_sweep=101,
            amp_schedule=AmplitudeSchedule.ramp(1e-3, 1e-2, 6),
            directions=("upward", "downward"),
        )

        module_results = results[crs.module[2].index()]
        module_results["results"][0]["upward"]["R0001"]["iq_counts"]

    Omitting *amp_schedule* sweeps the catalog at its own amplitudes, once — so
    the useful degenerate call is an up-and-down pair with nothing else said::

        results = await crs.multiamp_multisweep(
            catalog, span_hz=200e3, npoints_per_sweep=101,
            directions=("upward", "downward"),
        )

    A bare frequency list works the same way, for an array that is not tuned
    yet. It has no bias amplitude to scale, so the schedule has to supply one —
    ``ramp``/``explicit`` do by construction, while ``multiplicative`` needs an
    explicit ``base``::

        results = await crs.multiamp_multisweep(
            center_frequencies=[1.0e9, 1.1e9], module=2,
            span_hz=200e3, npoints_per_sweep=101,
            amp_schedule=AmplitudeSchedule.ramp(1e-4, 1e-2, 5),
        )
        results[crs.module[2].index()]["results"][0]["upward"]["S0001"]["iq_counts"]

    Args:
        crs (CRS): The CRS object (injected by macro).
        catalog (ResonatorCatalog, optional): What to sweep, as for
            ``multisweep``. Read, never written. Pass this or
            *center_frequencies*, not both.
        span_hz (float): Total frequency width (Hz) of each sweep. The same for
            every step.
        npoints_per_sweep (int): Points measured within each sweep's span.
        amp_schedule (AmplitudeSchedule, optional): The amplitude steps. Defaults
            to ``AmplitudeSchedule()`` — one step, at whatever amplitude the
            catalog already carries. Built through
            ``AmplitudeSchedule.multiplicative/ramp/explicit``, or the plain
            constructor for a single pass; see
            :mod:`rfmux.tuning.multisweep_amplitudes`.
        directions (Sequence[str], optional): Which directions to sweep each
            step in — ``("upward",)`` (the default), ``("downward",)``, or both.
            An explicit sequence rather than a ``"both"`` flag, so each sweep is
            labelled with both of its coordinates. Order is honoured: it is the
            order the sweeps are measured in.
        nsamps (int, optional): Samples averaged per frequency point. Defaults
            to 10.
        center_frequencies (list[float], optional): A bare list of sweep
            centres, for an array that is not tuned yet. Pass this or *catalog*,
            not both.
        names (list[str], optional): Names for the *center_frequencies*, one
            each. Defaults to ``S0001…``. Resolved once, here, and passed
            explicitly to every sweep, so the schedule's keys and the results'
            keys are the same strings by construction. Rejected alongside a
            *catalog*, whose resonators are already named.
        module (int, optional): The readout module. Defaults to the catalog's
            own, and must agree with it when both are given; required when
            sweeping *center_frequencies*. One module per call — a list is
            refused, because a ladder over several modules would be several
            ladders.
        progress_callback (callable, optional): ``(module, pct)``, forwarded to
            each ``multisweep`` untouched, so it keeps meaning *progress within
            the current sweep* and resets once per sweep. For progress across
            the whole ladder use *sweep_callback*, which carries ``completed``
            and ``total``.
        data_callback (callable, optional): ``(module, partial_results, step,
            direction)`` — partial data during acquisition, as ``multisweep``
            emits it, plus the two coordinates saying which sweep it belongs to.

            .. note::
               This is two arguments wider than ``multisweep``'s own
               ``data_callback(module, partial_results)``. Inside a ladder the
               bare form is ambiguous — a consumer plotting live has no way to
               tell which step and direction the points belong to. Callers
               written against the narrow form need updating; Periscope is
               tracked in ``tuning_revamp_todo.md``.
        sweep_callback (callable, optional): ``(record)``, called once per
            completed sweep — not once per step — with a dict of ``step``,
            ``direction``, ``amplitudes``, ``factor``, ``completed``, ``total``
            and ``data``. A script ignores it, a notebook prints from it,
            Periscope re-emits it as the signals it has now. It is also the
            reason this macro does not need to return partial results on
            failure: every sweep that finished has already been handed over.

            ``data`` is the bare ``{name: entry}`` for that one sweep, not the
            envelope ``multisweep`` returns — the module, the span and the rest
            are the same for every sweep in the ladder, and a hand-over is not
            a result. The coordinates that *do* vary are the record's own
            ``step`` and ``direction``.
        save (bool, optional): Write the result to the output folder when the
            whole ladder finishes. Defaults to whatever
            ``rfmux.tuning.store.autosave_enabled()`` says, which is on unless
            your config file or ``$RFMUX_AUTOSAVE`` turns it off. One file for
            the ladder, not one per step — the steps are one measurement.
        label (str, optional): Your name for this ladder, appended to the
            filename. Ignored when nothing is being saved.

    Returns:
        dict: the same shape ``multisweep`` returns, keyed by module identifier
        — one key here, because this driver is one module per call.

        .. code-block:: python

            {
                "crs0042_rmod2": {
                    "schema_version": 3,
                    "module": 2,            # resolved, never None
                    "call_params": {...},   # verbatim, as this driver was called
                    "results": {
                        0: {"upward": {...}, "downward": {...}},
                        1: {"upward": {...}, "downward": {...}},
                    },
                },
            }

        ``results`` is keyed by amplitude iteration, numbered from 0 in the
        order measured, and an iteration holds one entry per direction swept and
        nothing else. Under a direction is what one ``multisweep`` measured:
        ``{name: entry}``, unwrapped from the envelope it arrived in, since the
        module and the sweep parameters are the same for every rung and the
        amplitude is already on each entry.

        :func:`~rfmux.tuning.sweep_results.pack_results` owns this shape
        and documents it in full; the readers beside it —
        ``collect_amplitude_iterations_for``,
        ``find_iteration_matching_amplitude`` and
        ``get_amplitudes_at_iteration`` — are the supported way back out, so
        callers need not walk the nesting by hand. They take *one module's*
        value, not the whole dict, and say so if handed the container.

    Raises:
        ValueError: for an empty or unknown *directions*, a module list, a
            module that disagrees with the catalog, or an amplitude the schedule
            cannot resolve — all before the first sweep runs.
    """
    if amp_schedule is None:
        amp_schedule = AmplitudeSchedule()
    if not isinstance(amp_schedule, AmplitudeSchedule):
        raise TypeError(
            f"amp_schedule must be an AmplitudeSchedule, got "
            f"{type(amp_schedule).__name__}. A bare number or list is not one "
            f"— AmplitudeSchedule({amp_schedule!r}) for a single "
            f"amplitude, or .ramp()/.explicit() for a ladder."
        )

    if (catalog is None) == (center_frequencies is None):
        raise ValueError(
            "Pass a ResonatorCatalog or center_frequencies — exactly one of "
            "the two."
        )
    if catalog is not None and names is not None:
        raise ValueError(
            "names applies to center_frequencies only — a catalog's resonators "
            "are already named. Rename them in the catalog if that is what you "
            "meant."
        )

    directions = _resolve_directions(directions)
    resolved_module = _resolve_module(catalog, center_frequencies, module)

    # A catalog names its own resonators. A bare frequency list does not, so
    # name it once here rather than letting every multisweep call generate
    # S0001… for itself: the schedule is keyed by these names, so they have to
    # be the same strings each time round the loop, not merely equal by
    # coincidence.
    if catalog is not None:
        target = catalog
        section_names = None
    else:
        section_names = _resolve_section_names(center_frequencies, names)
        target = section_names

    # Resolves the whole ladder up front, so an amplitude that overshoots full
    # scale on step 5 is a ValueError now rather than after four steps of data.
    steps = amp_schedule.steps(target)

    # Every sweep this call makes comes back under this one key, and the ladder
    # is packed under it too — the driver is one module per call by construction.
    module_id = crs.module[resolved_module].index()

    total = len(steps) * len(directions)
    completed = 0
    results: dict[int, dict[str, dict]] = {}

    for step in steps:
        per_direction: dict[str, dict] = {}

        for direction in directions:
            # Widen multisweep's (module, partial) to carry the coordinates of
            # the sweep the partial data belongs to.
            if data_callback is None:
                inner_data_callback = None
            else:
                def inner_data_callback(
                    module_, partial, _step=step.step, _direction=direction
                ):
                    data_callback(module_, partial, _step, _direction)

            sweep_kwargs = dict(
                span_hz=span_hz,
                npoints_per_sweep=npoints_per_sweep,
                amp=step.amplitudes,
                nsamps=nsamps,
                sweep_direction=direction,
                module=resolved_module,
                progress_callback=progress_callback,
                data_callback=inner_data_callback,
                # The ladder is one measurement and gets one file, written
                # below. A step that saved itself would leave a folder full of
                # partial sweeps beside the whole one.
                save=False,
            )
            if catalog is not None:
                swept = await crs.multisweep(catalog, **sweep_kwargs)
            else:
                swept = await crs.multisweep(
                    center_frequencies=center_frequencies,
                    names=section_names,
                    **sweep_kwargs,
                )

            # One sweep, in the same shape this driver is about to return: its
            # own module, its one iteration, its one direction. Unwrapped to the
            # sections rather than nested whole, because the envelope's
            # call_params would then be repeated once per step per direction,
            # differing only in the amplitude the entries already carry.
            data = swept[module_id]["results"][0][direction]

            per_direction[direction] = data
            completed += 1

            if sweep_callback is not None:
                sweep_callback({
                    "step": step.step,
                    "direction": direction,
                    "amplitudes": dict(step.amplitudes),
                    "factor": step.factor,
                    "completed": completed,
                    "total": total,
                    "data": data,
                })

        results[step.step] = per_direction

    packed = pack_results(
        results,
        module_id=module_id,
        module=resolved_module,
        amp_schedule=amp_schedule,
        directions=directions,
        span_hz=span_hz,
        npoints_per_sweep=npoints_per_sweep,
        nsamps=nsamps,
        catalog=catalog,
        center_frequencies=center_frequencies,
        names=names,
        requested_module=module,
    )
    store.maybe_save(packed, "multiamp_multisweep", save=save, label=label)
    return packed
