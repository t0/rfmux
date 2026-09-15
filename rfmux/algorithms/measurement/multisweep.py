"""Measure targeted frequency sweeps for a catalog or a list of centres.

``amp`` chooses one probe amplitude or an amplitude schedule;
``sweep_direction`` chooses one or both frequency directions. Results are
keyed by module, then ``results[step][direction][name]``. Fitting and bias
finding are separate analysis steps in :mod:`rfmux.tuning`.
"""

import numpy as np
import asyncio
import warnings

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ...core.dac_scale import dac_scale_dbm
from ...core.hardware_map import macro
from ...core.schema import CRS
from ...core.resonators import BiasPoint, Resonator, ResonatorCatalog
from ...core.transferfunctions import ALLOWED_NCO_BANDWIDTH_HZ, convert_roc_to_volts
from ...tuning import store
from ...tuning.multisweep_amplitudes import AmplitudeSchedule, resolve_amplitudes
from ...tuning.sweep_results import DIRECTIONS, merge_modules, pack_multisweep


@dataclass(frozen=True, slots=True)
class _SweepTarget:
    """One resonator's worth of "what to sweep", normalized.

    The two ways of asking for a sweep — a catalog, or a bare
    ``center_frequencies`` list — are resolved into a list of these at the top
    of the macro, so the measurement body below has exactly one thing to walk.

    No amplitude: a target is *what* is swept, and it is swept once per step of
    the amplitude schedule. How loud each pass is arrives beside the targets,
    as the step's own ``{name: amplitude}``.
    """

    name: str  # identity, and the key this sweep comes back under
    channel: int  # 1-based hardware channel
    center_frequency_hz: float


def _resolve_section_names(
    center_frequencies: list[float],
    names: list[str] | None,
) -> list[str]:
    """Name every entry of a bare frequency list.

    Without *names*, sections are called ``S0001…`` in the order they were
    passed — S for section, and visibly not a catalog's drawn names (``BOTA``,
    ``KOZR``), so a result dict says which of the two it came from at a glance.
    """
    count = len(center_frequencies)

    if names is None:
        return [f"S{i:04d}" for i in range(1, count + 1)]

    names = list(names)
    if len(names) != count:
        raise ValueError(
            f"{len(names)} names for {count} center_frequencies. Pass one name "
            f"per frequency, in the same order, or none at all for S0001…"
        )
    if not all(isinstance(n, str) for n in names):
        raise TypeError(
            "names must be strings — they are the keys the sweeps come back "
            "under."
        )
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate section names {duplicates}: each sweep needs its own "
            f"key, or results would overwrite each other."
        )
    return names


def _check_sweep_input(
    catalog: ResonatorCatalog | None,
    center_frequencies: list[float] | Mapping[str, float] | None,
) -> None:
    """A catalog with optional named centers, or a bare frequency list.

    Checked at the very top of the macro as well as here, because everything
    after it — the module, the amplitudes, the channels — is a question about
    an input that may not exist, and "module is required" is a poor answer to
    a call that named nothing to sweep.
    """
    if catalog is not None and isinstance(center_frequencies, Mapping):
        return
    if catalog is None and isinstance(center_frequencies, Mapping):
        raise ValueError("Named center_frequencies require a catalog.")
    if (catalog is None) == (center_frequencies is None):
        raise ValueError(
            "Pass a ResonatorCatalog or center_frequencies — exactly one of "
            "the two, or a catalog with a mapping of names to sweep centers."
        )


def _resolve_sweep_targets(
    catalog: ResonatorCatalog | None,
    center_frequencies: list[float] | Mapping[str, float] | None,
    names: list[str] | None,
) -> list[_SweepTarget]:
    """Normalize the catalog and bare-frequency-list forms into one list."""

    _check_sweep_input(catalog, center_frequencies)

    if catalog is not None:
        if names is not None:
            raise ValueError(
                "names applies to center_frequencies only — a catalog's "
                "resonators are already named. Rename them in the catalog if "
                "that is what you meant."
            )
        if center_frequencies is not None:
            if set(center_frequencies) != set(catalog.names()):
                raise ValueError(
                    "center_frequencies must name every catalog resonator "
                    "exactly once, with no extra names."
                )
            if any(not np.isfinite(f) or f <= 0
                   for f in center_frequencies.values()):
                raise ValueError("Sweep centers must be positive finite frequencies.")
        # Sections come out in bias-frequency order, matching catalog iteration
        # and `names()`, so a sweep result tabulates the same way the array
        # does. The channel each one is measured on rides along on the target.
        return [
            _SweepTarget(
                name=r.name,
                channel=r.channel,
                center_frequency_hz=float(
                    r.bias.frequency_hz if center_frequencies is None
                    else center_frequencies[r.name]
                ),
            )
            for r in catalog.resonators(order="frequency")
        ]

    # --- a bare list of frequencies: named S0001…, channelled by position ---
    return [
        _SweepTarget(
            name=name,
            channel=channel,
            center_frequency_hz=float(cf),
        )
        for channel, (name, cf) in enumerate(
            zip(_resolve_section_names(center_frequencies, names), center_frequencies),
            start=1,
        )
    ]


def _resolve_catalog(
    catalog: ResonatorCatalog | None,
    targets: list[_SweepTarget],
    amplitudes: Mapping[str, float],
    module: int,
) -> ResonatorCatalog:
    """Return the input catalog, or build one for bare sweep targets.

    Generated bias points keep the exact sweep centres and step-zero
    amplitudes. Later amplitudes are recorded on their sweep entries.
    """
    if catalog is not None:
        return catalog

    return ResonatorCatalog(
        [
            Resonator(
                name=t.name,
                channel=t.channel,
                bias=BiasPoint(
                    frequency_hz=t.center_frequency_hz,
                    amplitude=amplitudes[t.name],
                    bias_frequency_quantized=False,
                ),
            )
            for t in targets
        ],
        module=module,
    )


def _resolve_directions(sweep_direction) -> tuple[str, ...]:
    """Validate a direction string or sequence and return a tuple."""
    if isinstance(sweep_direction, str):
        if sweep_direction not in DIRECTIONS:
            raise ValueError(
                f"Invalid sweep_direction: {sweep_direction}. Must be "
                f"'upward' or 'downward'."
            )
        return (sweep_direction,)

    if not isinstance(sweep_direction, Sequence):
        raise TypeError(
            f"sweep_direction must be one of {DIRECTIONS}, or a sequence of "
            f"them, got {type(sweep_direction).__name__}."
        )

    resolved = tuple(sweep_direction)
    if not resolved:
        raise ValueError(
            f"sweep_direction is empty: nothing would be measured. Pass at "
            f"least one of {DIRECTIONS}."
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
            f"sweep_direction repeats {repeated}: each direction is one key in "
            f"the result, so a repeat would overwrite itself rather than "
            f"measure twice."
        )
    return resolved


def _resolve_schedule(
    amp,
    names: list[str],
    defaults: dict[str, float] | None,
) -> AmplitudeSchedule:
    """The amplitude axis, as a schedule, whichever way it was spelled.

    A bare ``amp`` is a one-step schedule — that is what one amplitude is — so
    the loop below has one thing to walk either way. It is checked here, in
    ``amp``'s own words, before becoming a schedule's ``base``: the two speak
    the same vocabulary (see
    :func:`~rfmux.tuning.multisweep_amplitudes.resolve_amplitudes`) but a
    caller who wrote ``amp=`` should not be told about ``base=``. The resolved
    mapping is then thrown away and *amp* kept verbatim, so ``call_params``
    still records the request rather than what was worked out from it.
    """
    if isinstance(amp, AmplitudeSchedule):
        return amp

    resolve_amplitudes(
        names,
        amp,
        defaults=defaults,
        # A positional list pairs to the caller's own ordering, which exists
        # for center_frequencies and not for a catalog.
        allow_sequence=defaults is None,
        what="amp",
        of="center_frequencies",
    )
    return AmplitudeSchedule(amp)


async def _measure_sweep(
    crs: CRS,
    targets: list[_SweepTarget],
    amplitudes: Mapping[str, float],
    *,
    module: int,
    sweep_direction: str,
    span_hz: float,
    npoints_per_sweep: int,
    nsamps: int,
    step: int,
    report_progress,
    data_callback,
) -> dict:
    """One sweep: every target once, at *amplitudes*, in one direction.

    The only part of this module that touches a board, and the part the loop
    above it calls once per amplitude step per direction. Returns the
    ``{name: entry}`` that step and direction measured; the caller files it.

    *report_progress* takes a fraction in [0, 1] of *this* sweep — turning that
    into progress across the whole call is the caller's arithmetic, since only
    it knows how many sweeps there are.
    """
    # The channels this sweep owns, and the only ones it will ever silence.
    # Everything else on the module is somebody else's: a tone parked by hand,
    # another algorithm's channel, a bias tone left live on purpose. Zeroing
    # the whole module would be tidier for us and destructive for them.
    #
    # The flip side, and the caller's job now: multisweep no longer guarantees
    # a quiet module. A foreign tone left live can intermodulate with the sweep
    # or land inside a span, so a measurement that needs silence has to arrange
    # it — crs.clear_channels(module=...) before the call.
    swept_channels = {t.channel for t in targets}

    # --- Generate sweep frequencies ---
    resonance_data = {}
    for t in targets:
        # Generate points for this sweep based on direction
        if sweep_direction == "upward":
            sweep_points = np.linspace(
                t.center_frequency_hz - span_hz / 2,
                t.center_frequency_hz + span_hz / 2,
                npoints_per_sweep,
                endpoint=True
            )
        else:
            sweep_points = np.linspace(
                t.center_frequency_hz + span_hz / 2,
                t.center_frequency_hz - span_hz / 2,
                npoints_per_sweep,
                endpoint=True
            )

        resonance_data[t.name] = {
            'frequencies': sweep_points,
            'iq_counts': np.zeros(npoints_per_sweep, dtype=np.complex128), # Pre-allocate array
            'original_center_frequency': t.center_frequency_hz,
        }

    # --- Group resonances by NCO regions ---
    # Regions are contiguous runs in frequency order, cut whenever adding the
    # next resonator would push the run past the NCO's instantaneous bandwidth.
    sorted_targets = sorted(targets, key=lambda t: t.center_frequency_hz)
    nco_regions = []
    current_region = [sorted_targets[0]]
    region_min = sorted_targets[0].center_frequency_hz - span_hz / 2

    for t in sorted_targets[1:]:
        if (t.center_frequency_hz + span_hz / 2) - region_min > ALLOWED_NCO_BANDWIDTH_HZ:
            nco_regions.append(current_region)
            current_region = [t]
            region_min = t.center_frequency_hz - span_hz / 2
        else:
            current_region.append(t)
    nco_regions.append(current_region)

    # --- Calculate all NCO frequencies upfront ---
    nco_frequencies = [
        (
            min(t.center_frequency_hz - span_hz / 2 for t in region)
            + max(t.center_frequency_hz + span_hz / 2 for t in region)
        ) / 2
        for region in nco_regions
    ]

    # --- Measurement Loop ---
    total_nco_regions = len(nco_regions)

    for region_idx, region_targets in enumerate(nco_regions):
        # --- Set Current NCO Frequency ---
        current_nco_freq = nco_frequencies[region_idx]
        await crs.set_nco_frequency(current_nco_freq, module=module)

        # --- Sweep Points within the Region ---
        active_res_channels = {t.channel for t in region_targets}

        # Loop through sweep points
        for point_idx in range(npoints_per_sweep):
            # Configure resonance channels for this sweep point
            async with crs.tuber_context() as ctx:
                # Set resonance channels
                for t in region_targets:
                    freq = resonance_data[t.name]['frequencies'][point_idx]
                    freq_rel = freq - current_nco_freq # Use current_nco_freq
                    ctx.set_frequency(freq_rel, channel=t.channel, module=module)
                    if not point_idx: # only set amplitude once per sweep
                        ctx.set_amplitude(
                            amplitudes[t.name], channel=t.channel, module=module
                        )

                # Silence this sweep's *other* NCO regions — their tones would
                # otherwise sit outside the current NCO's band. Only channels
                # this sweep owns, so a tone the caller parked elsewhere on the
                # module survives.
                if not point_idx: # only need to do this once per sweep
                    for ch in sorted(swept_channels - active_res_channels):
                        ctx.set_amplitude(0, channel=ch, module=module) # Zeros freq implicitly if amp=0
                await ctx()

            # Acquire samples for all active resonance channels
            samples = await crs.get_samples(nsamps, average=True, channel=None, module=module)

            # Process samples for each resonance in this region
            for t in region_targets:
                channel_idx = t.channel - 1 # 0-based index
                # Get raw IQ
                i_val = samples.mean.i[channel_idx]
                q_val = samples.mean.q[channel_idx]
                raw_iq_val = i_val + 1j * q_val

                # Store raw IQ value directly
                resonance_data[t.name]['iq_counts'][point_idx] = raw_iq_val

            # --- Progress update ---
            if report_progress:
                report_progress(
                    (region_idx + point_idx / npoints_per_sweep) / total_nco_regions
                )

            # Call data callback with intermediate results if provided
            if data_callback:
                # Partial data for the region being swept, up to and including
                # this point.  Regions already finished are not resent, and
                # regions not yet started have nothing to send.
                n = point_idx + 1
                data_callback(module, {
                    t.name: {
                        'frequencies': resonance_data[t.name]['frequencies'][:n],
                        'iq_counts': resonance_data[t.name]['iq_counts'][:n],
                        'original_center_frequency': t.center_frequency_hz,
                    }
                    for t in region_targets
                }, step, sweep_direction)

    # Record each trace at its measured centre, without analysis or re-centring.
    results = {}
    for t in targets:
        data_entry = resonance_data[t.name]
        iq_counts = data_entry['iq_counts']

        results[t.name] = {
            'channel': t.channel,
            'frequencies': data_entry['frequencies'],
            'iq_counts': iq_counts,
            'iq_volts': convert_roc_to_volts(iq_counts),
            'original_center_frequency': data_entry['original_center_frequency'],
            'sweep_direction': sweep_direction,
            'sweep_amplitude': amplitudes[t.name],  # Amplitude this resonator was swept at
        }

    # --- Hardware Cleanup ---
    try:
        async with crs.tuber_context() as ctx:
            # Only the channels this sweep put a tone on. See swept_channels.
            for ch in sorted(swept_channels):
                ctx.set_amplitude(0, channel=ch, module=module)
            await ctx()
    except Exception as e:
        warnings.warn(f"Hardware cleanup failed for module {module}: {e}")

    return results


def sweep_nco_frequency(center_frequencies, span_hz: float) -> float:
    """The NCO a sweep of *center_frequencies* with *span_hz* each runs
    at: the middle of the band the sweep covers.  Placeholders (NaN)
    in the list are ignored."""
    cfs = [cf for cf in center_frequencies if np.isfinite(cf)]
    lo = min(cf - span_hz / 2 for cf in cfs)
    hi = max(cf + span_hz / 2 for cf in cfs)
    return (lo + hi) / 2


@macro(CRS, register=True)
async def multisweep(
    crs: CRS,
    catalog: ResonatorCatalog | None = None,
    *,
    span_hz: float = 100e3,
    npoints_per_sweep: int = 101,
    amp: float | list[float] | Mapping[str, float] | AmplitudeSchedule | None = None,
    nsamps: int = 10,
    sweep_direction: str | Sequence[str] = "upward",
    center_frequencies: list[float] | Mapping[str, float] | None = None,
    names: list[str] | None = None,
    module=None,
    progress_callback=None,
    data_callback=None,
    sweep_callback=None,
    save=None,
    label=None,
):
    """Measure targeted sweeps at one or more probe amplitudes.

    Resonators within an NCO band are swept together. Wider catalogs are
    measured in groups, without phase stitching between NCO settings.
    The input catalog and its bias points are not modified.

    Example::

        from rfmux.tuning import AmplitudeSchedule

        sweeps = await crs.multisweep(
            catalog, amp=AmplitudeSchedule.ramp(1e-3, 1e-2, 6),
            sweep_direction=("upward", "downward"))
        block = sweeps[crs.module[catalog.module].index()]
        trace = block["results"][0]["upward"][catalog.names()[0]]

    Args:
        crs: CRS instance, supplied by the macro.
        catalog: resonators to sweep, with channel bindings and default
            centres and amplitudes from their bias points.
        span_hz: full sweep width for each resonator, in Hz.
        npoints_per_sweep: frequency points per sweep.
        amp: amplitude in DAC full-scale units, or an ``AmplitudeSchedule``.
            With a catalog, None uses its bias amplitudes; a scalar applies
            to all members and a name mapping must cover every member.
            Without a catalog, supply a scalar, a list matching the centres,
            a name mapping, or a schedule. Lists are rejected with a catalog.
            The entire schedule is validated before measurement starts.
        nsamps: samples averaged per frequency point.
        sweep_direction: ``"upward"``, ``"downward"``, or a sequence of both
            in acquisition order. Every amplitude step uses each direction.
        center_frequencies: with a catalog, an optional name-to-Hz mapping
            covering every member; this overrides sweep centres only.
            Without a catalog, a list of centres in Hz. Channels are assigned
            in list order, starting at 1, and a catalog is recorded using
            these centres and the first step's amplitudes.
        names: names for a bare centre list, defaulting to ``S0001…``.
            Rejected with a catalog, which supplies its own names.
        module: defaults to the catalog's module and must match if supplied.
            Required without a catalog; a list runs modules concurrently.
            Multiple modules require bare centre lists and one analog bank
            (1–4 or 5–8).
        progress_callback: ``(module, percent)`` across the whole schedule.
        data_callback: ``(module, partial_results, step, direction)`` during
            acquisition. Partial results contain the current NCO region's
            resonators, sliced to the points measured so far.
        sweep_callback: ``(record)`` after each completed direction of a step.
            The record holds ``step``, ``direction``, ``amplitudes``, ``factor``,
            ``completed``, ``total``, and ``data`` (``{name: entry}``).
        save: save one file for the completed call, including all modules.
            None uses ``store.autosave_enabled()``.
        label: label appended to the filename when saving.

    Returns:
        dict: ``{module_id: block}``, keyed by ``crs.module[m].index()``.
        Each block holds ``results[step][direction][name]`` and the resolved
        catalog and amplitude schedule in ``call_params``. Sweep entries
        contain frequencies (Hz), complex IQ in counts and volts, channel,
        original centre, amplitude, and direction. See
        :func:`rfmux.tuning.sweep_results.pack_multisweep` for the format.

    Raises:
        ValueError: invalid directions, module selection, centres, or
            amplitudes. These are checked before acquisition.
    """

    # What call_params records: the argument as passed, before the catalog or
    # the list branch below rewrites it into the module actually swept.
    requested_module = module

    _check_sweep_input(catalog, center_frequencies)
    directions = _resolve_directions(sweep_direction)

    # --- Resolve module ------------------------------------------------------
    if catalog is not None:
        if isinstance(module, list):
            raise ValueError(
                f"A catalog belongs to one module ({catalog.module}); sweeping "
                f"modules {module} means one call per module."
            )
        if module is None:
            module = catalog.module
        elif module != catalog.module:
            raise ValueError(
                f"module={module} does not match the catalog's module "
                f"({catalog.module})."
            )
    elif module is None:
        raise ValueError("module is required when sweeping center_frequencies.")

    # --- Handle parallel execution if module is a list ---
    if isinstance(module, list):
        if not module:
            raise ValueError("Module list cannot be empty.")

        # Ensure all modules are in [1..4] or all are in [5..8]
        in_first_bank = all(1 <= m <= 4 for m in module)
        in_second_bank = all(5 <= m <= 8 for m in module)
        if not (in_first_bank or in_second_bank):
            raise ValueError(
                f"Module list must be entirely in [1..4] or [5..8], got: {module}"
            )

        tasks = []
        for m in module:
            # Call the same macro again, but for a single module=m. The whole
            # amplitude schedule and direction axis go with it, so each module
            # runs the same measurement rather than a slice of one.
            tasks.append(crs.multisweep(
                center_frequencies=center_frequencies,
                names=names,
                span_hz=span_hz,
                npoints_per_sweep=npoints_per_sweep,
                amp=amp,
                nsamps=nsamps,
                sweep_direction=sweep_direction,
                module=m, # Pass single module here
                progress_callback=progress_callback,
                data_callback=data_callback,
                sweep_callback=sweep_callback,
                # The per-module calls do not save. One call is one file, so
                # the fan-out saves once, below, over the merged container.
                save=False,
            ))
        # Each of those returns a container of its own, keyed by module, so the
        # several modules merge into one rather than stacking into a list whose
        # order was the only thing saying which element was which.
        merged = merge_modules(await asyncio.gather(*tasks))
        store.maybe_save(merged, "multisweep", save=save, label=label)
        return merged
    # --- End parallel execution handling ---

    # --- Resolve what to sweep ----------------------------------------------
    targets = _resolve_sweep_targets(catalog, center_frequencies, names)

    # --- Resolve how loud, on each pass -------------------------------------
    #
    # A catalog can supply the amplitudes itself; a bare frequency list has
    # nothing to fall back on, which is the difference the defaults carry.
    schedule = _resolve_schedule(
        amp,
        [t.name for t in targets],
        {r.name: float(r.bias.amplitude) for r in catalog.resonators(order="frequency")}
        if catalog is not None
        else None,
    )

    # Resolves the whole schedule up front, so an amplitude that overshoots full
    # scale on step 5 is a ValueError now rather than after four steps of data.
    # A call with nothing to sweep has nothing to resolve it against; the schedule
    # it asked for is still reported below.
    steps = (
        schedule.resolve_steps(
            catalog if catalog is not None else [t.name for t in targets]
        )
        if targets
        else []
    )

    # Both ways of saying what to sweep, as one catalog — see _resolve_catalog.
    # After the schedule because it takes the first step's amplitudes, and before
    # the packing because that is where it is going.
    swept = _resolve_catalog(
        catalog, targets, steps[0].amplitudes if steps else {}, module
    )

    # Every sweep this call makes comes back under this one key.
    module_id = crs.module[module].index()

    # Read here rather than left to whoever opens the file: every amplitude
    # this call sweeps at is a fraction of DAC full scale, and the board that
    # says what full scale is worth is this one, now. Once for the call, since
    # it is one number per module.
    dac_scale = await dac_scale_dbm(crs, module)

    def packed(sweeps):
        """This measurement, in the shape every sweep comes back in."""
        return pack_multisweep(
            sweeps,
            module_id=module_id,
            module=module,
            amp_schedule=schedule,
            directions=directions,
            span_hz=span_hz,
            npoints_per_sweep=npoints_per_sweep,
            nsamps=nsamps,
            catalog=swept,
            center_frequencies=center_frequencies,
            names=names,
            requested_module=requested_module,
            dac_scale_dbm=dac_scale,
        )

    if not targets:
        # Still a well-formed result, with no sections in it. A bare {} would
        # be indistinguishable from a caller's own empty dict, and the
        # provenance of a sweep that measured nothing is worth as much as any
        # other's. The steps and directions asked for are still there, because
        # they are still what was asked for.
        warnings.warn("Nothing to sweep. Returning a result with no sections.")
        empty = packed({
            step: {direction: {} for direction in directions}
            for step in range(schedule.nsteps)
        })
        store.maybe_save(empty, "multisweep", save=save, label=label)
        return empty

    # --- Validate inputs for single module execution ---
    # Check if number of resonances exceeds maximum channels
    dec = await crs.get_decimation()
    if dec <=3:
        max_channels = 128
    else:
        max_channels = 1024

    if len(targets) > max_channels:
        raise ValueError(f"Number of resonances ({len(targets)}) exceeds maximum channels ({max_channels})")

    over = [t for t in targets if t.channel > max_channels]
    if over:
        raise ValueError(
            f"Channel(s) {[t.channel for t in over]} exceed the maximum channel "
            f"({max_channels}) available at decimation {dec}."
        )

    if npoints_per_sweep < 2:
        raise ValueError("npoints_per_sweep must be at least 2.")
    if span_hz <= 0:
        raise ValueError("span_hz must be positive.")
    # --- End input validation ---

    # --- The sweeps: step outer, direction inner ----------------------------
    #
    # Each step's up-and-down pair is measured together and the amplitude
    # marches monotonically, which is what a bifurcation walk wants.
    total = len(steps) * len(directions)
    completed = 0
    results: dict[int, dict[str, dict]] = {}

    for step in steps:
        per_direction: dict[str, dict] = {}

        for direction in directions:
            # Progress is across the whole call, so each sweep reports into its
            # own slice of it — which for a single sweep is all of it, exactly
            # as it was before there was an amplitude axis to share with.
            if progress_callback is None:
                report_progress = None
            else:
                def report_progress(fraction, _done=completed):
                    progress_callback(module, (_done + fraction) / total * 100)

            data = await _measure_sweep(
                crs,
                targets,
                step.amplitudes,
                module=module,
                sweep_direction=direction,
                span_hz=span_hz,
                npoints_per_sweep=npoints_per_sweep,
                nsamps=nsamps,
                step=step.step,
                report_progress=report_progress,
                data_callback=data_callback,
            )

            per_direction[direction] = data
            completed += 1

            if progress_callback is not None:
                progress_callback(module, completed / total * 100)

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

    swept = packed(results)
    store.maybe_save(swept, "multisweep", save=save, label=label)
    return swept
