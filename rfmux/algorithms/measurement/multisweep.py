"""
multisweep: A measurement algorithm for performing simultaneous, targeted,
high-resolution frequency sweeps around multiple specified center frequencies.
Optionally fits resonances and centers IQ data.

One call is *one* sweep: one amplitude per resonator, one direction. Sweeping
the same array at a ladder of amplitudes is a separate layer on top — see
``tuning_multisweep_amplitudes_plan.md``.

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
from ...core.transferfunctions import convert_roc_to_volts, convert_iq_to_df
from .fitting import center_resonance_iq_circle
from typing import Optional, Tuple # Added for type hinting


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


def _get_recalculated_center_freq(
    original_cf_hz: float,
    sweep_freqs_hz: np.ndarray,
    sweep_iq: np.ndarray,
    method: Optional[str]
) -> Tuple[float, str]:
    """
    Helper to recalculate center frequency based on sweep data and method.
    Returns the new frequency and the method string actually applied.
    """
    if sweep_iq.size == 0 or sweep_freqs_hz.size == 0:
        return original_cf_hz, "none"

    actual_method_applied = "none"
    new_frequency_hz = original_cf_hz

    if method == "min-s21":
        s21_mag = np.abs(sweep_iq)
        if np.any(s21_mag):
            min_mag_idx = np.argmin(s21_mag)
            recalculated_freq = sweep_freqs_hz[min_mag_idx]
            if np.isfinite(recalculated_freq):
                new_frequency_hz = recalculated_freq
                actual_method_applied = "min-s21"
            else:
                warnings.warn(f"Recalculated 'min-s21' frequency for original_cf {original_cf_hz*1e-6:.3f} MHz is not finite. Using original.")
        else:
            warnings.warn(f"Cannot recalculate 'min-s21' for original_cf {original_cf_hz*1e-6:.3f} MHz: all S21 magnitudes are zero.")
    elif method == "max-diq":
        if sweep_iq.size < 2 or sweep_freqs_hz.size < 2: # Need at least 2 points for gradient
             warnings.warn(f"Cannot recalculate 'max-diq' for original_cf {original_cf_hz*1e-6:.3f} MHz: not enough sweep points ({sweep_iq.size}).")
             return original_cf_hz, "none"

        # Center the IQ data first to calculate velocity relative to the resonance circle center
        try:
            centered_iq = center_resonance_iq_circle(sweep_iq)
        except Exception as e:
            warnings.warn(f"Failed to center IQ circle for 'max-diq' calculation at {original_cf_hz*1e-6:.3f} MHz: {e}. Using uncentered data.")
            centered_iq = sweep_iq

        # Calculate the velocity of the IQ point as it moves through the complex plane
        # This is now relative to the circle center, not the origin
        i_vals = centered_iq.real
        q_vals = centered_iq.imag

        # Calculate derivatives
        di_df = np.gradient(i_vals, sweep_freqs_hz)
        dq_df = np.gradient(q_vals, sweep_freqs_hz)

        # Total velocity magnitude in IQ plane (relative to circle center)
        iq_velocity = np.sqrt(di_df**2 + dq_df**2)

        if np.any(iq_velocity):
            max_velocity_idx = np.argmax(iq_velocity)
            recalculated_freq = sweep_freqs_hz[max_velocity_idx]
            if np.isfinite(recalculated_freq):
                new_frequency_hz = recalculated_freq
                actual_method_applied = "max-diq"
            else:
                warnings.warn(f"Recalculated 'max-diq' frequency for original_cf {original_cf_hz*1e-6:.3f} MHz is not finite. Using original.")
        else:
            warnings.warn(f"Cannot recalculate 'max-diq' for original_cf {original_cf_hz*1e-6:.3f} MHz: IQ velocity is zero everywhere.")
    elif method is not None:
        warnings.warn(f"Unknown recalculate_center_frequencies method: '{method}'. No recalculation performed.")

    return new_frequency_hz, actual_method_applied

@macro(CRS, register=True)
async def multisweep(
    crs: CRS,
    catalog: ResonatorCatalog | None = None,
    *,
    span_hz: float,
    npoints_per_sweep: int,
    amp: float | list[float] | Mapping[str, float] | None = None,
    nsamps: int = 10,
    bias_frequency_method: Optional[str] = "max-diq", # Options: "min-s21", "max-diq", or None
    rotate_saved_data: bool = False,  # Whether to rotate sweep data based on TOD analysis
    sweep_direction: str = "upward", # Options: "upward", "downward"
    apply_df_calibration: bool = True,  # Enable frequency shift/dissipation calibration
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
        sweeps["R0001"]["iq_complex"]

    Or with a bare list of frequencies, for anything that is not a tuned array
    yet::

        sweeps = await crs.multisweep(
            center_frequencies=[1.0e9, 1.1e9],
            amp=1e-3,                 # or [1e-3, 2e-3], one per frequency
            names=["low", "high"],    # optional; default is S0001, S0002
            span_hz=200e3, npoints_per_sweep=101, module=2,
        )
        sweeps["low"]["iq_complex"]

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
        nsamps (int, optional): Number of samples to average per frequency point for the main sweep.
                                Defaults to 10. The TOD acquisition for rotation uses 1000 samples.
        bias_frequency_method (Optional[str], optional):
            Determines how/if center frequencies are recalculated for biasing.
            - "min-s21": Recalculates center frequency to the point of minimum |S21| in the sweep.
            - "max-diq": Recalculates center frequency to the point of maximum IQ velocity |d(I+jQ)/df| in the sweep.
                         This finds where the IQ trajectory is moving fastest, considering both angular and
                         radial motion.
            - None: No recalculation. Use original center frequency.
            Defaults to "max-diq".  The result is *reported*, not written back
            into the catalog.
        rotate_saved_data (bool, optional):
            Whether to rotate sweep data based on TOD analysis. When True and bias_frequency_method
            is not None:
            - For "min-s21": Rotates to minimize the I component of the TOD's mean.
            - For "max-diq": Rotates to align the principal component of the TOD with the I-axis.
            Defaults to False.
        sweep_direction (str, optional): The direction of the frequency sweep.
            - "upward": Sweep from lower to higher frequencies.
            - "downward": Sweep from higher to lower frequencies.
            Defaults to "upward".
        apply_df_calibration (bool, optional): Whether to apply frequency shift/dissipation calibration.
            When True, converts IQ data to frequency shift and dissipation units using sweep derivatives.
            Defaults to True.
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
        dict: keyed by resonator name with a catalog, or by section name with
              a bare frequency list. Each value is a dictionary containing:
              {
                  'name': str,                    # the key this entry is under
                  'channel': int,                 # hardware channel swept on
                  'frequencies': np.ndarray (Hz), # Sweep frequencies
                  'iq_complex': np.ndarray (complex), # Final, possibly rotated, sweep IQ data
                  'phase_degrees': np.ndarray (degrees), # Derived from final iq_complex
                  'original_center_frequency': float, # Sweep centre, as requested
                  'bias_frequency': float, # Frequency to use for biasing (may be recalculated)
                  'recalculation_method_applied': str, # "min-s21", "max-diq", or "none"
                  'rotation_tod': Optional[np.ndarray], # 1000-sample IQ TOD, if acquired
                  'applied_rotation_degrees': Optional[float], # Rotation applied to sweep data
                  'sweep_direction': str, # "upward" or "downward"
                  'sweep_amplitude': float, # Normalized amplitude used in this sweep
                  'iq_complex_volts': Optional[np.ndarray], # Sweep IQ data in voltage units
                  'df_calibration': Optional[complex], # Calibration factor: multiply IQ data (volts) by this to get freq shift + j*dissipation
                  'calibrated_tod_df': Optional[np.ndarray], # rotation_tod converted to freq shift + j*dissipation
              }

              If running on multiple modules (center_frequencies only),
              returns a list of these dictionaries.
    """

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
                bias_frequency_method=bias_frequency_method,
                rotate_saved_data=rotate_saved_data,
                sweep_direction=sweep_direction,
                apply_df_calibration=apply_df_calibration,
                module=m, # Pass single module here
                progress_callback=progress_callback,
                data_callback=data_callback,
            ))
        # Results will be a list of dictionaries, one per module
        results_list = await asyncio.gather(*tasks)
        return results_list
    # --- End parallel execution handling ---

    # --- Resolve what to sweep ----------------------------------------------
    targets = _resolve_sweep_targets(catalog, center_frequencies, names, amp)

    if not targets:
        warnings.warn("Nothing to sweep. Returning empty dictionary.")
        return {}

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
            'iq_complex': np.zeros(npoints_per_sweep, dtype=np.complex128), # Pre-allocate array
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
                resonance_data[t.name]['iq_complex'][point_idx] = raw_iq_val

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
                        'iq_complex': resonance_data[t.name]['iq_complex'][:n],
                        'original_center_frequency': t.center_frequency_hz,
                    }
                    for t in region_targets
                })

        # --- Post-Sweep Processing for this NCO Region (TOD acquisition and rotation) ---
        if bias_frequency_method is not None: # Only proceed if a method is specified
            # Step 1: Determine all bias frequencies and methods for this region
            targets_needing_tod = [] # Resonances needing TOD acquisition
            for t in region_targets:
                res_data_entry = resonance_data[t.name]
                bias_freq, recalc_method_used = _get_recalculated_center_freq(
                    original_cf_hz=t.center_frequency_hz,
                    sweep_freqs_hz=res_data_entry['frequencies'],
                    sweep_iq=res_data_entry['iq_complex'], # Use raw sweep IQ for recalc
                    method=bias_frequency_method
                )
                res_data_entry['bias_frequency'] = bias_freq
                res_data_entry['recalculation_method_applied'] = recalc_method_used
                res_data_entry['rotation_tod'] = None # Initialize
                res_data_entry['applied_rotation_degrees'] = 0.0 # Initialize

                if recalc_method_used in ["min-s21", "max-diq"]:
                    targets_needing_tod.append((t, bias_freq))

            # Step 2: Acquire TODs in batch if any are needed
            if targets_needing_tod:
                async with crs.tuber_context() as ctx:
                    # Turn off this sweep's channels first, so only the ones
                    # being TOD'd are live. Foreign channels are left alone.
                    for ch_iter in sorted(swept_channels):
                        ctx.set_amplitude(0, channel=ch_iter, module=module)

                    # Set up channels that need TOD
                    for t, bias_freq in targets_needing_tod:
                        freq_for_tod_rel = bias_freq - current_nco_freq
                        ctx.set_frequency(freq_for_tod_rel, channel=t.channel, module=module)
                        ctx.set_amplitude(t.amplitude, channel=t.channel, module=module)
                    await ctx()

                try:
                    # Acquire all TODs simultaneously
                    all_tod_samples = await crs.get_samples(50, average=False, channel=None, module=module)

                    # Distribute TODs to respective resonance_data entries
                    for t, _ in targets_needing_tod:
                        channel_idx_0based = t.channel - 1
                        tod_i_channel_data = np.array(all_tod_samples.i[channel_idx_0based])
                        tod_q_channel_data = np.array(all_tod_samples.q[channel_idx_0based])
                        resonance_data[t.name]['rotation_tod'] = tod_i_channel_data + 1j * tod_q_channel_data
                except Exception as e:
                    warnings.warn(f"Batch TOD acquisition failed for NCO region {region_idx} (module {module}): {e}")
                    # Mark all relevant TODs as None if batch failed
                    for t, _ in targets_needing_tod:
                        resonance_data[t.name]['rotation_tod'] = None

            # Step 3: Calculate and apply rotations using the (now populated) TODs
            for t in region_targets: # Iterate again to apply rotations
                res_data_entry = resonance_data[t.name]
                recalc_method_used = res_data_entry['recalculation_method_applied']

                if recalc_method_used not in ["min-s21", "max-diq"] or res_data_entry['rotation_tod'] is None:
                    # Ensure these are set if no TOD-based rotation happens
                    if res_data_entry.get('bias_frequency') is None : # Should have been set in step 1
                         res_data_entry['bias_frequency'] = t.center_frequency_hz
                    if res_data_entry.get('recalculation_method_applied') is None:
                         res_data_entry['recalculation_method_applied'] = "none"
                    res_data_entry['rotation_tod'] = None # Ensure it's None
                    res_data_entry['applied_rotation_degrees'] = 0.0
                    continue # Skip to next resonance if no valid method or no TOD

                rotation_angle_rad = 0.0
                tod_iq = res_data_entry['rotation_tod']
                cf_original = t.center_frequency_hz

                if tod_iq.size > 0: # Check if TOD has data
                    if recalc_method_used == "min-s21":
                        mean_tod = np.mean(tod_iq)
                        rotation_angle_rad = (np.pi / 2) - np.angle(mean_tod)
                    elif recalc_method_used == "max-diq":
                        if tod_iq.size > 1:
                            data_matrix = np.vstack((tod_iq.real, tod_iq.imag))
                            try:
                                covariance_matrix = np.cov(data_matrix)
                                if np.all(np.isfinite(covariance_matrix)):
                                    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
                                    # Eigenvectors are columns, eigenvector corresponding to largest eigenvalue is pc1
                                    pc1_idx = np.argmax(eigenvalues)
                                    pc1_vector_complex = eigenvectors[0, pc1_idx] + 1j * eigenvectors[1, pc1_idx]
                                    rotation_angle_rad = -np.angle(pc1_vector_complex)
                                else:
                                    warnings.warn(f"Covariance matrix for 'max-diq' rotation of cf {cf_original*1e-6:.3f} MHz contains non-finite values. Skipping rotation.")
                            except np.linalg.LinAlgError:
                                 warnings.warn(f"PCA failed for 'max-diq' rotation of cf {cf_original*1e-6:.3f} MHz. Skipping rotation.")
                        else:
                            warnings.warn(f"Not enough TOD points for 'max-diq' PCA rotation for cf {cf_original*1e-6:.3f} MHz ({tod_iq.size} points). Skipping rotation.")

                # Store the calculated rotation angle regardless of whether we apply it
                res_data_entry['calculated_rotation_degrees'] = np.degrees(rotation_angle_rad)

                # Only apply rotation if rotate_saved_data is True
                if rotate_saved_data and rotation_angle_rad != 0.0:
                    rotation_factor = np.exp(1j * rotation_angle_rad)
                    res_data_entry['iq_complex'] *= rotation_factor

                    # Also rotate the TOD if it exists and is not empty
                    if res_data_entry['rotation_tod'] is not None and res_data_entry['rotation_tod'].size > 0:
                        res_data_entry['rotation_tod'] *= rotation_factor

                    res_data_entry['applied_rotation_degrees'] = np.degrees(rotation_angle_rad)
                else:
                    # If rotation not applied, mark as 0
                    res_data_entry['applied_rotation_degrees'] = 0.0

    # --- Format final results for each resonance ---
    results = {}
    for t in targets:
        data_entry = resonance_data[t.name]
        final_iq_complex = data_entry['iq_complex']
        original_cf = data_entry['original_center_frequency']

        # Get bias frequency (either recalculated or original)
        bias_freq = data_entry.get('bias_frequency', original_cf)
        recalc_method_applied = data_entry.get('recalculation_method_applied', "none")

        # Apply calibration if requested
        iq_complex_volts = None
        df_calibration = None
        calibrated_tod_df = None

        if apply_df_calibration:
            # Convert sweep IQ data from ADC counts to volts
            iq_complex_volts = convert_roc_to_volts(final_iq_complex)

            # Ensure frequencies are in ascending order for interpolation
            # (downward sweeps produce a decreasing sequence which scipy rejects)
            sort_idx = np.argsort(data_entry['frequencies'])
            cal_freqs = data_entry['frequencies'][sort_idx]
            cal_iq_volts = iq_complex_volts[sort_idx]

            # Calculate calibration at bias frequency
            try:
                # Calculate the calibration factor by converting iq=1 (in volts)
                # This gives us the conversion factor: any future IQ data can be
                # multiplied by this to get frequency shift and dissipation
                df_calibration = convert_iq_to_df(
                    np.array([1.0 + 0j]),  # Unit IQ in volts
                    bias_freq,
                    cal_freqs,
                    cal_iq_volts
                )[0]  # Get the single complex value

                # If we have a TOD, calibrate it too
                if data_entry.get('rotation_tod') is not None:
                    rotation_tod_volts = convert_roc_to_volts(data_entry['rotation_tod'])
                    calibrated_tod_df = convert_iq_to_df(
                        rotation_tod_volts,
                        bias_freq,
                        cal_freqs,
                        cal_iq_volts
                    )

            except Exception as e:
                warnings.warn(f"Calibration failed for resonance {t.name!r}: {e}")
                df_calibration = None
                calibrated_tod_df = None

        results[t.name] = {
            'name': t.name,
            'channel': t.channel,
            'frequencies': data_entry['frequencies'],
            'iq_complex': final_iq_complex,
            'phase_degrees': np.degrees(np.angle(final_iq_complex)),
            'original_center_frequency': original_cf,
            'bias_frequency': bias_freq,
            'recalculation_method_applied': recalc_method_applied,
            'rotation_tod': data_entry.get('rotation_tod'),
            'applied_rotation_degrees': data_entry.get('applied_rotation_degrees'),
            'sweep_direction': sweep_direction,
            'sweep_amplitude': t.amplitude,  # Amplitude this resonator was swept at
            'iq_complex_volts': iq_complex_volts,  # Sweep IQ data in voltage units
            'df_calibration': df_calibration,  # Calibration parameters
            'calibrated_tod_df': calibrated_tod_df  # Calibrated TOD data
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
