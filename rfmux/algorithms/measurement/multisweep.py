"""
multisweep: A measurement algorithm for performing simultaneous, targeted,
high-resolution frequency sweeps around multiple specified center frequencies.

One call is *one* sweep: one amplitude per resonator, one direction. Sweeping
the same array at a ladder of amplitudes is a separate layer on top — see
``tuning_multisweep_amplitudes_plan.md``.

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

Either way multisweep reads its input and never modifies it. Updating a catalog
from what a sweep reveals belongs to the analysis that learns it — fitting,
bias finding — not here.
"""

import numpy as np
import asyncio
import warnings

from collections.abc import Mapping
from dataclasses import dataclass

from ...core.hardware_map import macro
from ...core.schema import CRS
from ...core.resonators import ResonatorCatalog
from ...core.transferfunctions import convert_roc_to_volts
from ...tuning.sweep_results import merge_modules, pack_sweep


@dataclass(frozen=True, slots=True)
class _SweepTarget:
    """One resonator's worth of "what to sweep", normalized.

    The two ways of asking for a sweep — a catalog, or a bare
    ``center_frequencies`` list — are resolved into a list of these at the top
    of the macro, so the measurement body below has exactly one thing to walk.
    """

    name: str  # identity, and the key this sweep comes back under
    channel: int  # 1-based hardware channel
    center_frequency_hz: float
    amplitude: float  # normalized DAC units


def _resolve_section_names(
    center_frequencies: list[float],
    names: list[str] | None,
) -> list[str]:
    """Name every entry of a bare frequency list.

    Without *names*, sections are called ``S0001…`` in the order they were
    passed — S for section, and visibly not a catalog's ``R0001…``, so a
    result dict says which of the two it came from at a glance.
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


def _resolve_amplitudes(
    names: list[str],
    amp: float | list[float] | Mapping[str, float] | None,
    *,
    defaults: dict[str, float] | None,
    allow_sequence: bool,
) -> dict[str, float]:
    """Decide the probe amplitude for every sweep, keyed by name.

    ``None`` falls back to *defaults* (a catalog's own bias amplitudes) and is
    an error where there are none. A number applies to everything. A mapping
    sets them individually and must name every sweep, because a half-applied
    amplitude override is the kind of thing that is only noticed after the data
    is taken. A positional sequence is only accepted where the caller supplied
    the ordering — i.e. alongside ``center_frequencies``.
    """
    if amp is None:
        if defaults is None:
            raise ValueError(
                "amp is required when sweeping center_frequencies: pass a "
                "single amplitude for all of them, one per frequency, or a "
                "{name: amplitude} mapping. (A ResonatorCatalog carries an "
                "amplitude per resonator, so there amp is optional.)"
            )
        return dict(defaults)

    if isinstance(amp, Mapping):
        unknown = sorted(set(amp) - set(names))
        if unknown:
            raise ValueError(
                f"amp names {unknown} are not being swept. The names in play "
                f"are {names[:4]}{' …' if len(names) > 4 else ''}."
            )
        missing = sorted(set(names) - set(amp))
        if missing:
            raise ValueError(
                f"amp is missing an amplitude for {missing}. Pass every name, "
                f"or a single number for all of them."
            )
        return {n: float(amp[n]) for n in names}

    if isinstance(amp, (list, tuple, np.ndarray)):
        if not allow_sequence:
            raise TypeError(
                "amp cannot be a positional sequence alongside a catalog — the "
                "pairing would depend on catalog ordering, which is not "
                "something a caller should have to know. Pass a "
                "{name: amplitude} mapping, a single number, or None. (A "
                "positional list *is* accepted alongside center_frequencies, "
                "where the ordering is your own.)"
            )
        values = [float(a) for a in amp]
        if len(values) != len(names):
            raise ValueError(
                f"amp has {len(values)} amplitudes for {len(names)} "
                f"center_frequencies. Pass one per frequency, in the same "
                f"order, or a single number for all."
            )
        return dict(zip(names, values))

    return {n: float(amp) for n in names}


def _resolve_sweep_targets(
    catalog: ResonatorCatalog | None,
    center_frequencies: list[float] | None,
    names: list[str] | None,
    amp: float | list[float] | Mapping[str, float] | None,
) -> list[_SweepTarget]:
    """Normalize the catalog and bare-frequency-list forms into one list."""

    if (catalog is None) == (center_frequencies is None):
        raise ValueError(
            "Pass a ResonatorCatalog or center_frequencies — exactly one of "
            "the two."
        )

    if catalog is not None:
        if names is not None:
            raise ValueError(
                "names applies to center_frequencies only — a catalog's "
                "resonators are already named. Rename them in the catalog if "
                "that is what you meant."
            )
        catalog_names = [r.name for r in catalog]  # channel order
        amplitudes = _resolve_amplitudes(
            catalog_names,
            amp,
            defaults={r.name: float(r.bias.amplitude) for r in catalog},
            allow_sequence=False,
        )
        return [
            _SweepTarget(
                name=r.name,
                channel=r.channel,
                center_frequency_hz=float(r.bias.frequency_hz),
                amplitude=amplitudes[r.name],
            )
            for r in catalog
        ]

    # --- a bare list of frequencies: named S0001…, channelled by position ---
    section_names = _resolve_section_names(center_frequencies, names)
    amplitudes = _resolve_amplitudes(
        section_names, amp, defaults=None, allow_sequence=True
    )
    return [
        _SweepTarget(
            name=name,
            channel=channel,
            center_frequency_hz=float(cf),
            amplitude=amplitudes[name],
        )
        for channel, (name, cf) in enumerate(
            zip(section_names, center_frequencies), start=1
        )
    ]


@macro(CRS, register=True)
async def multisweep(
    crs: CRS,
    catalog: ResonatorCatalog | None = None,
    *,
    span_hz: float,
    npoints_per_sweep: int,
    amp: float | list[float] | Mapping[str, float] | None = None,
    nsamps: int = 10,
    sweep_direction: str = "upward", # Options: "upward", "downward"
    center_frequencies: list[float] | None = None,
    names: list[str] | None = None,
    module=None,
    progress_callback=None,
    data_callback=None,
):
    """
    Perform simultaneous, high-resolution frequency sweeps around many center
    frequencies at once.

    This algorithm dedicates one channel per resonance and sweeps all resonances
    in parallel. The NCO is re-tuned for different groups of resonances (NCO
    regions) if their combined span exceeds the NCO's instantaneous bandwidth.
    No phase stitching is performed between data collected from different NCO
    regions.

    One call is one sweep: one amplitude per resonator, one direction. The
    input is read, never written — a sweep on its own has not learned anything
    yet, and the analyses that do (fitting, bias finding) update the catalog
    themselves.

    Only the channels this sweep puts a tone on are silenced, on the way in and
    on the way out. A tone the caller parked elsewhere on the module — by hand,
    or by another algorithm — survives the call. The corollary is that
    multisweep does not guarantee a quiet module: if a foreign tone would
    intermodulate with the sweep or sit inside a span, clear it first with
    ``crs.clear_channels(module=...)``.

    Two ways to say what to sweep, identical once the measurement starts.
    With a catalog, which brings its own frequencies, amplitudes and channels::

        catalog = ResonatorCatalog.from_frequencies(found, module=2, amplitude=1e-3)
        sweeps = await crs.multisweep(catalog, span_hz=200e3, npoints_per_sweep=101)

        module_sweeps = sweeps[crs.module[2].index()]
        module_sweeps["results"][0]["upward"]["R0001"]["iq_counts"]

    Or with a bare list of frequencies, for anything that is not a tuned array
    yet::

        sweeps = await crs.multisweep(
            center_frequencies=[1.0e9, 1.1e9],
            amp=1e-3,                 # or [1e-3, 2e-3], one per frequency
            names=["low", "high"],    # optional; default is S0001, S0002
            span_hz=200e3, npoints_per_sweep=101, module=2,
        )
        sweeps[crs.module[2].index()]["results"][0]["upward"]["low"]["iq_counts"]

    Args:
        crs (CRS): The CRS object (injected by macro).
        catalog (ResonatorCatalog, optional): What to sweep. Each resonator
            contributes its ``bias.frequency_hz`` as the sweep centre, its
            ``channel`` as the hardware channel, and — unless *amp* overrides
            it — its ``bias.amplitude`` as the probe amplitude. Pass this or
            *center_frequencies*, not both.
        span_hz (float): Total frequency width (Hz) of each sweep.
        npoints_per_sweep (int): Number of points to measure within each sweep's span.
        amp (float | list[float] | Mapping[str, float] | None, optional): Probe
            amplitude, in normalized DAC units.

            With a *catalog*:

            - ``None`` (default): use each resonator's own ``bias.amplitude``.
            - a number: use it for every resonator.
            - a ``{resonator_name: amplitude}`` mapping: per-resonator, and it
              must name every resonator in the catalog.

            A positional sequence is refused here, because the pairing would
            depend on catalog ordering.

            With *center_frequencies*, where the ordering is the caller's own:

            - a number: use it for every frequency.
            - a list: one amplitude per frequency, in the same order.
            - a ``{section_name: amplitude}`` mapping, as above.

            Required in that case — there is nothing to fall back to.
        nsamps (int, optional): Number of samples to average per frequency
            point. Defaults to 10.
        sweep_direction (str, optional): The direction of the frequency sweep.
            - "upward": Sweep from lower to higher frequencies.
            - "downward": Sweep from higher to lower frequencies.
            Defaults to "upward".
        center_frequencies (list[float], optional): A bare list of sweep
            centres, for sweeping frequencies that are not a tuned array — no
            resonances found yet, or a system that has none. Hardware channels
            are 1-based positions in this list. Pass this or *catalog*, not
            both.
        names (list[str], optional): Names for the *center_frequencies*, one
            each, in the same order — these are the keys the sweeps come back
            under. Defaults to ``S0001…`` (S for section), which is visibly not
            a catalog's ``R0001…``, so a result dict says which of the two
            produced it. Rejected alongside a *catalog*, whose resonators are
            already named.
        module (int | list[int], optional): The target readout module. Defaults
            to the catalog's own ``module``, and must agree with it when both
            are given; required when sweeping *center_frequencies*. A list of
            modules is only accepted with *center_frequencies* — a catalog
            belongs to one module, so sweeping several means one call per
            module.
        progress_callback (callable, optional): Function called with (module, progress_percentage).
        data_callback (callable, optional): Function called with (module, partial_results)
            during acquisition, carrying the current NCO region's resonators
            sliced to the points measured so far.

    Returns:
        dict: keyed by module identifier — ``crs.module[m].index()``, e.g.
        ``crs0042_rmod2`` — with one entry per module swept::

            {
                "crs0042_rmod2": {
                    "schema_version": 3,
                    "module": 2,           # resolved, never None
                    "call_params": {...},  # verbatim, as this macro was called
                    "results": {
                        0: {"upward": {"R0001": {...}, "R0002": {...}}},
                    },
                },
            }

        Always keyed by module, including for the one module that is the usual
        case, so a caller who writes ``for module_id, module_sweeps in
        sweeps.items():`` has written the same code for one module and for four.

        A list of modules sweeps them concurrently and merges the results into
        one dict of this shape. Each envelope then records its own module in
        ``call_params["module"]`` rather than the list that was passed: the call
        that produced it really was a call for that module, and an envelope is
        meant to stand on its own once lifted out.

        ``results`` is the shape ``multiamp_multisweep`` returns, and one sweep
        is one iteration in one direction — which is what it is, not a padded
        slot. So nothing downstream has to ask which macro produced a result:
        the readers in :mod:`rfmux.tuning.sweep_results` and the fitters in
        :mod:`rfmux.tuning.fits` take either. They take *one module's* value,
        not the whole dict — ``fit_sweeps(sweeps["crs0042_rmod2"])`` — and say
        so if handed the container.

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
    """

    # What call_params records: the argument as passed, before the catalog or
    # the list branch below rewrites it into the module actually swept.
    requested_module = module

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
            # Call the same macro again, but for a single module=m
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
            ))
        # Each of those returns a container of its own, keyed by module, so the
        # several modules merge into one rather than stacking into a list whose
        # order was the only thing saying which element was which.
        return merge_modules(await asyncio.gather(*tasks))
    # --- End parallel execution handling ---

    # --- Resolve what to sweep ----------------------------------------------
    targets = _resolve_sweep_targets(catalog, center_frequencies, names, amp)

    def packed(sections):
        """This sweep, in the shape every sweep comes back in."""
        return pack_sweep(
            sections,
            module_id=crs.module[module].index(),
            module=module,
            sweep_direction=sweep_direction,
            span_hz=span_hz,
            npoints_per_sweep=npoints_per_sweep,
            nsamps=nsamps,
            amp=amp,
            catalog=catalog,
            center_frequencies=center_frequencies,
            names=names,
            requested_module=requested_module,
        )

    if not targets:
        # Still a well-formed result, with no sections in it. A bare {} would
        # be indistinguishable from a caller's own empty dict, and the
        # provenance of a sweep that measured nothing is worth as much as any
        # other's.
        warnings.warn("Nothing to sweep. Returning a result with no sections.")
        return packed({})

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
    if sweep_direction not in ("upward", "downward"):
        raise ValueError(
            f"Invalid sweep_direction: {sweep_direction}. Must be 'upward' or 'downward'."
        )
    for t in targets:
        if t.amplitude <= 0:
            warnings.warn(
                f"Amplitude for {t.name!r} (amp={t.amplitude}) is non-positive. "
                f"Results may be invalid."
            )
    # --- End input validation ---

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

    # --- Define Constants ---
    MAX_NCO_SPAN_HZ = 500e6

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
        if (t.center_frequency_hz + span_hz / 2) - region_min > MAX_NCO_SPAN_HZ:
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
                        ctx.set_amplitude(t.amplitude, channel=t.channel, module=module)

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
            if progress_callback:
                # Calculate progress as percentage
                region_progress = region_idx / total_nco_regions
                point_progress = point_idx / npoints_per_sweep
                overall_progress = (region_progress + point_progress / total_nco_regions) * 100
                progress_callback(module, overall_progress)

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
                })

    # --- Format final results for each resonance ---
    #
    # NOTE: re-centring is deliberately absent. A ladder of amplitudes wants the
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
            'sweep_amplitude': t.amplitude,  # Amplitude this resonator was swept at
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

    return packed(results)
