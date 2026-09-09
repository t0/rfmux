"""
multisweep: A measurement algorithm for performing simultaneous, targeted,
high-resolution frequency sweeps around multiple specified center frequencies.

One call is one *measurement*, which may be one sweep or many. The narrow case —
one amplitude, one direction — is a call that said nothing about either. The
wide case walks a schedule of probe amplitudes, in one or both frequency
directions, and returns every sweep it took in one dict.

That is one macro rather than two because a schedule is not a different kind of
measurement from a sweep; it is more of one. The two used to be ``multisweep``
and ``multiamp_multisweep``, whose arguments were near-identical and whose
outputs were identical — ``results[step][direction][name]`` either way, with a
single sweep sitting at ``results[0]["upward"]`` because that is genuinely what
it is. Two entry points onto that shape asserted a distinction the shape denies.

So the amplitude axis is ``amp``, which takes a number, a list, a mapping *or*
an :class:`~rfmux.tuning.multisweep_amplitudes.AmplitudeSchedule`; and the
direction axis is ``sweep_direction``, which takes one direction or a sequence
of them. A caller who does not want to iterate says nothing and gets one sweep.

It measures and it returns what it measured. It does not fit, rotate,
calibrate, or move a sweep centre onto the dip it found — those are analyses,
they belong to the code that does them, and a sweep that quietly did one of
them on the way past would be a sweep whose output nobody can reason about.

There are two ways to say what to sweep, identical once the measurement starts:

* a :class:`~rfmux.core.resonators.ResonatorCatalog`, which supplies each
  resonator's sweep centre (``bias.frequency_hz``), its probe amplitude
  (``bias.amplitude``, overridable per call) and its permanent hardware
  channel.  Results come back keyed by resonator name.
* a bare list of ``center_frequencies`` plus an ``amp``, for sweeping
  frequencies that are not a tuned array — before resonances have been found,
  or on a system that has none.  Results come back keyed by section name,
  ``S0001…`` unless ``names`` says otherwise.

The second form is resolved into the first before anything is measured, the way
a bare ``amp`` is resolved into an ``AmplitudeSchedule``: the catalog recorded
in ``call_params`` is the one that was swept whichever way the call was
spelled, so a reader downstream — bias finding, above all — has one thing to
look at and never a sweep with no resonators in it.

Either way multisweep reads its input and never modifies it. Updating a catalog
from what a sweep reveals belongs to the analysis that learns it — fitting,
bias finding — not here.
"""

import numpy as np
import asyncio
import warnings

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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


def _require_one_input(
    catalog: ResonatorCatalog | None,
    center_frequencies: list[float] | None,
) -> None:
    """Exactly one of the two ways to say what to sweep.

    Checked at the very top of the macro as well as here, because everything
    after it — the module, the amplitudes, the channels — is a question about
    an input that may not exist, and "module is required" is a poor answer to
    a call that named nothing to sweep.
    """
    if (catalog is None) == (center_frequencies is None):
        raise ValueError(
            "Pass a ResonatorCatalog or center_frequencies — exactly one of "
            "the two."
        )


def _resolve_sweep_targets(
    catalog: ResonatorCatalog | None,
    center_frequencies: list[float] | None,
    names: list[str] | None,
) -> list[_SweepTarget]:
    """Normalize the catalog and bare-frequency-list forms into one list."""

    _require_one_input(catalog, center_frequencies)

    if catalog is not None:
        if names is not None:
            raise ValueError(
                "names applies to center_frequencies only — a catalog's "
                "resonators are already named. Rename them in the catalog if "
                "that is what you meant."
            )
        # Sections come out in bias-frequency order, matching catalog iteration
        # and `names()`, so a sweep result tabulates the same way the array
        # does. The channel each one is measured on rides along on the target.
        return [
            _SweepTarget(
                name=r.name,
                channel=r.channel,
                center_frequency_hz=float(r.bias.frequency_hz),
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
    """The catalog this sweep is of, whichever way it was asked for.

    A bare ``center_frequencies`` list becomes one here, and what
    ``call_params`` records is this rather than the ``None`` it used to: a
    sweep then always carries the array it measured, and the analysis that
    reads it back — ``find_bias_points`` in particular — needs nothing from the
    caller that the measurement did not already hold.

    Everything but the amplitude is already decided, because it is the target:
    the name each section comes back under, the channel it was measured on, and
    the frequency it was centred on. Centres are kept exactly as passed rather
    than quantized — a centre is a number the caller may be doing arithmetic
    with, and it agrees with each entry's ``original_center_frequency`` this
    way. *amplitudes* is step 0's, which is the amplitude the first pass
    actually used; a schedule has no one amplitude, and the amplitude for each later
    step is on its own sweeps.
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
    """Check the direction axis and freeze it.

    One direction is a string, because that is what one direction is. Several
    are a sequence, and an explicit one rather than the magic string ``"both"``,
    so the product below is honestly a product and each sweep is labelled with
    both of its coordinates.
    """
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

    # --- Format final results for each resonance ---
    #
    # NOTE: re-centring is deliberately absent. A schedule of amplitudes wants the
    # sweep centre to follow a resonance that moves between steps, and that will
    # come back — as an adjustment to the *sweep centre* the next step is taken
    # at, made by whatever analysis found the dip. It was previously spelled
    # "recalculate the bias frequency", which conflated two different things:
    # where to point the next sweep, and where the resonator is biased. The bias
    # frequency lives in the catalog and is not a sweep's to report.
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
    center_frequencies: list[float] | None = None,
    names: list[str] | None = None,
    module=None,
    progress_callback=None,
    data_callback=None,
    sweep_callback=None,
    save=None,
    label=None,
):
    """
    Perform simultaneous, high-resolution frequency sweeps around many center
    frequencies at once.

    This algorithm dedicates one channel per resonance and sweeps all resonances
    in parallel. The NCO is re-tuned for different groups of resonances (NCO
    regions) if their combined span exceeds the NCO's instantaneous bandwidth.
    No phase stitching is performed between data collected from different NCO
    regions.

    One call is one measurement, of as many sweeps as its two iterating axes
    ask for: an amplitude schedule in *amp*, and one or both frequency
    directions in *sweep_direction*. Say nothing about either and you get one
    sweep, at one amplitude, upward.

    The input is read, never written — a sweep on its own has not learned
    anything yet, and the analyses that do (fitting, bias finding) update the
    catalog themselves.

    Only the channels this sweep puts a tone on are silenced, on the way in and
    on the way out. A tone the caller parked elsewhere on the module — by hand,
    or by another algorithm — survives the call. The corollary is that
    multisweep does not guarantee a quiet module: if a foreign tone would
    intermodulate with the sweep or sit inside a span, clear it first with
    ``crs.clear_channels(module=...)``.

    Two ways to say what to sweep, identical once the measurement starts.
    With a catalog, which brings its own frequencies, amplitudes and channels::

        catalog = ResonatorCatalog.from_frequencies(found, module=2, amplitude=1e-3)
        sweeps = await crs.multisweep(catalog)

        module_sweeps = sweeps[crs.module[2].index()]
        module_sweeps["results"][0]["upward"]["BOTA"]["iq_counts"]

    Or with a bare list of frequencies, for anything that is not a tuned array
    yet::

        sweeps = await crs.multisweep(
            center_frequencies=[1.0e9, 1.1e9],
            amp=1e-3,                 # or [1e-3, 2e-3], one per frequency
            names=["low", "high"],    # optional; default is S0001, S0002
            span_hz=200e3, npoints_per_sweep=101, module=2,
        )
        sweeps[crs.module[2].index()]["results"][0]["upward"]["low"]["iq_counts"]

    To iterate over amplitude, hand *amp* a schedule instead of a number. Every
    step is swept in every requested direction, and every sweep comes back in
    the one result::

        from rfmux.tuning import AmplitudeSchedule

        sweeps = await crs.multisweep(
            catalog,
            amp=AmplitudeSchedule.ramp(1e-3, 1e-2, 6),
            sweep_direction=("upward", "downward"),
        )

        module_sweeps = sweeps[crs.module[2].index()]
        module_sweeps["results"][3]["downward"]["BOTA"]["iq_counts"]

    which is what bias finding reads, because "the loudest amplitude that has
    not bifurcated yet" is only meaningful against amplitudes that were
    actually measured. An up-and-down pair at the catalog's own amplitudes —
    enough for the hysteresis check — is the degenerate form of the same call::

        sweeps = await crs.multisweep(
            catalog, sweep_direction=("upward", "downward"),
        )

    Args:
        crs (CRS): The CRS object (injected by macro).
        catalog (ResonatorCatalog, optional): What to sweep. Each resonator
            contributes its ``bias.frequency_hz`` as the sweep centre, its
            ``channel`` as the hardware channel, and — unless *amp* overrides
            it — its ``bias.amplitude`` as the probe amplitude. Sections come
            back in bias-frequency order, matching ``catalog.names()``. Pass
            this or *center_frequencies*, not both.
        span_hz (float, optional): Total frequency width (Hz) of each sweep.
            The same for every sweep in the call. Defaults to 100 kHz.
        npoints_per_sweep (int, optional): Number of points to measure within
            each sweep's span. Defaults to 101.
        amp (float | list[float] | Mapping[str, float] | AmplitudeSchedule | None, optional):
            Probe amplitude, in normalized DAC units, for one sweep — or an
            :class:`~rfmux.tuning.multisweep_amplitudes.AmplitudeSchedule` for
            a schedule of them.

            One amplitude, with a *catalog*:

            - ``None`` (default): use each resonator's own ``bias.amplitude``.
            - a number: use it for every resonator.
            - a ``{resonator_name: amplitude}`` mapping: per-resonator, and it
              must name every resonator in the catalog.

            A positional sequence is refused here: a catalog is an unordered
            collection, so pairing to it by position means knowing which order
            it was pulled out in.

            One amplitude, with *center_frequencies*, where the ordering is the
            caller's own:

            - a number: use it for every frequency.
            - a list: one amplitude per frequency, in the same order.
            - a ``{section_name: amplitude}`` mapping, as above.

            Required in that case — there is nothing to fall back to.

            Or a schedule, built through ``AmplitudeSchedule.multiplicative``
            (steps that scale each resonator's own amplitude, so an array
            biased across a spread walks that spread together), ``.ramp`` or
            ``.explicit`` (steps that *are* the amplitude). One sweep per step
            per direction, all in one result. The whole schedule is resolved
            before the first sweep runs, so a step that overshoots full scale
            on step 5 is a ``ValueError`` now rather than after four steps of
            data. A schedule and a bare number are the same argument because
            they answer the same question — a number is a schedule of one step.
        nsamps (int, optional): Number of samples to average per frequency
            point. Defaults to 10.
        sweep_direction (str | Sequence[str], optional): The direction of the
            frequency sweep.

            - ``"upward"`` (the default): sweep from lower to higher
              frequencies.
            - ``"downward"``: sweep from higher to lower frequencies.
            - a sequence of both: sweep each amplitude step in each direction,
              which is what a hysteresis comparison needs. An explicit sequence
              rather than a ``"both"`` flag, so each sweep is labelled with
              both of its coordinates. Order is honoured: it is the order the
              sweeps are measured in.
        center_frequencies (list[float], optional): A bare list of sweep
            centres, for sweeping frequencies that are not a tuned array — no
            resonances found yet, or a system that has none. Hardware channels
            are 1-based positions in this list. Pass this or *catalog*, not
            both. A catalog is generated from the list and recorded in
            ``call_params["catalog"]``, the same way one amplitude is recorded
            as a one-step schedule, so what comes back is the same result a
            catalog would have produced and every analysis downstream works on
            it unchanged. Each section's bias amplitude there is step 0's.
        names (list[str], optional): Names for the *center_frequencies*, one
            each, in the same order — these are the keys the sweeps come back
            under. Defaults to ``S0001…`` (S for section), which is visibly not
            a catalog's drawn names, so a result dict says which of the two
            produced it. Rejected alongside a *catalog*, whose resonators are
            already named.
        module (int | list[int], optional): The target readout module. Defaults
            to the catalog's own ``module``, and must agree with it when both
            are given; required when sweeping *center_frequencies*. A list of
            modules is only accepted with *center_frequencies* — a catalog
            belongs to one module, so sweeping several means one call per
            module. Each module runs the whole schedule, concurrently, and the
            results merge into one dict keyed by module.
        progress_callback (callable, optional): ``(module, pct)`` — progress
            across the whole call, so a schedule of six steps in two directions
            reaches 100 only once, after the twelfth sweep. For *which* sweep
            is being taken, use *sweep_callback*.
        data_callback (callable, optional): ``(module, partial_results, step,
            direction)`` during acquisition, carrying the current NCO region's
            resonators sliced to the points measured so far, plus the two
            coordinates saying which sweep the points belong to.

            .. note::
               The last two arguments are new. Without them a consumer plotting
               live has no way to tell which amplitude step and direction the
               points belong to, and a single sweep's ``(0, "upward")`` is a
               fact about it rather than padding. Callers written against the
               two-argument form need updating.
        sweep_callback (callable, optional): ``(record)``, called once per
            completed sweep — not once per amplitude step — with a dict of
            ``step``, ``direction``, ``amplitudes``, ``factor``, ``completed``,
            ``total`` and ``data``. A script ignores it, a notebook prints from
            it, Periscope re-emits it as signals. It is also the reason this
            macro does not need to return partial results on failure: every
            sweep that finished has already been handed over.

            ``data`` is the bare ``{name: entry}`` for that one sweep, not the
            dict this macro returns — the module, the span and the rest are the
            same for every sweep in the call, and a hand-over is not a result.
            The coordinates that *do* vary are the record's own ``step`` and
            ``direction``.
        save (bool, optional): Write the result to the output folder when the
            whole measurement finishes. Defaults to whatever
            ``rfmux.tuning.store.autosave_enabled()`` says, which is on unless
            your config file or ``$RFMUX_AUTOSAVE`` turns it off. One file per
            call — a schedule's steps are one measurement, and a list of modules
            produces one file covering all of them.
        label (str, optional): Your name for this sweep, appended to the
            filename. Ignored when nothing is being saved.

    Returns:
        dict: keyed by module identifier — ``crs.module[m].index()``, e.g.
        ``crs0042_rmod2`` — with one entry per module swept::

            {
                "crs0042_rmod2": {
                    "schema_version": 7,
                    "measurement": "multisweep",
                    "module": 2,           # resolved, never None
                    "call_params": {...},  # verbatim, as this macro was called,
                                           # plus the resolved catalog and schedule
                    "results": {
                        0: {"upward": {"BOTA": {...}, "KOZR": {...}}},
                        1: {"upward": {...}},
                    },
                },
            }

        Always keyed by module, including for the one module that is the usual
        case, so a caller who writes ``for module_id, module_sweeps in
        sweeps.items():`` has written the same code for one module and for four.

        A list of modules sweeps them concurrently and merges the results into
        one dict of this shape. Each module's output then records its own
        module in ``call_params["module"]`` rather than the list that was
        passed: the call that produced it really was a call for that module,
        and one module's output is meant to stand on its own once lifted out.

        ``results`` is keyed by amplitude step, numbered from 0 in the order
        measured, and a step holds one entry per direction swept and nothing
        else. One sweep is one step in one direction — which is what it is, not
        a padded slot — so nothing downstream has to ask how wide the call was:
        the readers in :mod:`rfmux.tuning.sweep_results` and the fitters in
        :mod:`rfmux.tuning.fits` take a one-sweep result and a twenty-sweep one
        without asking. They take *one module's* value, not the whole dict —
        ``fit_sweeps(sweeps["crs0042_rmod2"])`` — and say so if handed the
        container.

        ``call_params`` records the arguments as they were passed, ``None``s
        and all, beside the two things they were resolved into: ``catalog``,
        which is there whichever form was asked for, and ``amp_schedule``.
        Between them a result says what was measured without reference to the
        call that made it, which is what lets the analysis downstream take a
        result and nothing else.

        Nothing is duplicated into the step level. What a resonator was probed
        at is already ``sweep_amplitude`` in its own entry, and the step that
        produced it is ``call_params["amp_schedule"]["steps"][step]``. The
        readers beside the packer — ``collect_amplitude_iterations_for``,
        ``find_iteration_matching_amplitude`` and
        ``get_amplitudes_at_iteration`` — are the supported way back out, so
        callers need not walk the nesting by hand.

        Under a direction is one entry per resonator, keyed by resonator name
        with a catalog or by section name with a bare frequency list::

            {
                'channel': int,                 # hardware channel swept on
                'frequencies': np.ndarray (Hz), # Sweep frequencies
                'iq_counts': np.ndarray (complex),  # Sweep IQ, in readout counts
                'iq_volts': np.ndarray (complex),   # The same, in volts at the
                                                    # board input port
                'original_center_frequency': float, # Sweep centre, as requested
                'sweep_direction': str, # "upward" or "downward"
                'sweep_amplitude': float, # Normalized amplitude used in this sweep
            }

        A sweep does not say what it is *of*: no phase, no fit, no bias
        frequency, no df calibration. Phase is ``np.angle(iq_counts)`` wherever
        it is wanted, and calling it phase in here invites reading it as the
        resonator's rather than the readout chain's. The entry carries no
        ``name`` either — it is already keyed by one.

    Raises:
        ValueError: for an empty or unknown *sweep_direction*, a module list
            alongside a catalog, a module that disagrees with the catalog, an
            amplitude the schedule cannot resolve, or a *center_frequencies*
            entry that is not a positive frequency — all before the first sweep
            runs.
    """

    # What call_params records: the argument as passed, before the catalog or
    # the list branch below rewrites it into the module actually swept.
    requested_module = module

    _require_one_input(catalog, center_frequencies)
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
