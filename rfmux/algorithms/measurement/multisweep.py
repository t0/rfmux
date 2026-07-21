"""
multisweep: A measurement algorithm for performing simultaneous, targeted,
high-resolution frequency sweeps around multiple specified center frequencies.

Returns a ``(res_info_dict, multisweep_data_dict)`` tuple where:

* **res_info_dict** – Lightweight resonator registry keyed by unique 4-letter
  uppercase codes (e.g. ``"AXQR"``).  Each entry holds the canonical
  ``bias_frequency``, ``bias_amplitude``, and permanent ``channel_number``
  for one resonator.  This dict is *not* modified by multisweep; it is
  updated by downstream algorithms such as ``bias_kids``.

* **multisweep_data_dict** – Heavy sweep output keyed by the same codes.
  Each entry contains a snapshot of the ``res_info_dict`` state at sweep
  time together with the actual sweep data arrays.
"""

import numpy as np
import random
import string
import asyncio
import warnings

from rfmux.core.hardware_map import macro
from rfmux.core.schema import CRS
from rfmux.core.transferfunctions import convert_roc_to_volts, convert_dac_normalized_to_dbm


# ---------------------------------------------------------------------------
# Helper: resonator ID generation
# ---------------------------------------------------------------------------

def _generate_unique_detector_ids(n: int) -> list[str]:
    """Return *n* unique random 4-letter uppercase detector IDs.

    Example IDs: ``"AXQR"``, ``"BKMT"``.  The IDs are guaranteed unique
    within the returned list.
    """
    existing: set[str] = set()
    result: list[str] = []
    while len(result) < n:
        code = "".join(random.choices(string.ascii_uppercase, k=4))
        if code not in existing:
            existing.add(code)
            result.append(code)
    return result


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

@macro(CRS, register=True)
async def multisweep(
    crs: CRS,
    span_hz: float = 150e3,
    npoints_per_sweep: int = 100,
    center_frequencies: list[float] | None = None,
    amp: float | list[float] | None = None,
    detector_ids: list[str] | None = None,
    res_info_dict: dict | None = None,
    nsamps: int = 10,
    sweep_direction: str = "upward",
    update_center_frequency: bool = False,
    *,
    module,
    progress_callback=None,
    data_callback=None,
) -> tuple[dict, dict]:
    """Simultaneous high-resolution frequency sweeps around multiple resonators.

    One hardware channel is dedicated to each resonance for the duration of the
    sweep.  If resonances span more than 500 MHz the NCO is re-tuned for each
    group (NCO region); no phase stitching is performed across regions.

    There are two equivalent ways to specify what to sweep:

    **Option A – explicit lists** (first sweep / standalone use)::

        res_info, data = await crs.multisweep(
            span_hz=2e6,
            npoints_per_sweep=500,
            center_frequencies=[1.2e9, 1.3e9],
            amp=0.1,
            module=1,
        )

    **Option B – res_info_dict** (re-run / downstream algorithms)::

        res_info, data = await crs.multisweep(
            span_hz=2e6,
            npoints_per_sweep=500,
            res_info_dict=res_info,   # re-uses codes, channels, bias frequencies
            amp=0.05,                 # optional per-iteration amplitude override
            module=1,
        )

    Args:
        crs: CRS object injected by the ``@macro`` decorator.
        span_hz: Total frequency span (Hz) of each individual sweep.
        npoints_per_sweep: Number of frequency points measured per sweep.
        center_frequencies: Center frequencies (Hz) for each resonance.
            Required when *res_info_dict* is not provided.
        amp: Amplitude(s) (normalised DAC units, 0–1) used as the sweep
            amplitude for this call.

            * Single ``float``: applied to every resonator.
            * ``list[float]``: per-resonator, must match the ordering of
              *center_frequencies* (option A) or ``res_info_dict.keys()``
              (option B).
            * ``None`` (option B only): uses each resonator's
              ``bias_amplitude`` from *res_info_dict* as the sweep amplitude.

        detector_ids: Optional list of 4-letter codes to use as resonator
            keys.  Length must match *center_frequencies*.  If ``None``,
            unique codes are generated automatically.  Ignored when
            *res_info_dict* is provided.
        res_info_dict: Pre-existing resonator registry (option B).  When
            supplied, ``bias_frequency``, ``bias_amplitude``, and
            ``channel_number`` are read from each entry.  The dict is
            **not** modified.
        nsamps: Samples averaged per frequency point (default 10).
        sweep_direction: ``"upward"`` (low → high) or ``"downward"``
            (high → low).
        update_center_frequency: Reserved for future use.  Raises
            ``NotImplementedError`` if set to ``True``.
        module: Target readout module(s).  A list triggers parallel
            execution (all modules must be in the same bank: 1–4 or 5–8).
        progress_callback: Called as ``callback(module, pct)`` after each
            frequency point, where *pct* is 0–100.
        data_callback: Called as ``callback(module, intermediate_results)``
            after each frequency point with a code-keyed dict of partial
            sweep data.

    Returns:
        ``(res_info_dict, multisweep_data_dict)`` where:

        * **res_info_dict** – Either the dict passed in (option B, unchanged)
          or a freshly built one (option A).  Keys are 4-letter codes.
          Each value: ``{"bias_frequency": float, "bias_amplitude": float,
          "channel_number": int}``.

        * **multisweep_data_dict** – Keys are the same 4-letter codes.
          Each value::

              {
                  # Snapshots from res_info_dict at sweep time
                  "bias_frequency":         float,
                  "bias_amplitude":         float,
                  "channel_number":         int,
                  # Actual values used for this sweep call
                  "sweep_center_frequency":     float,  # = bias_frequency by default
                  "sweep_amplitude_normalized": float,  # may differ from bias_amplitude
                  "sweep_amplitude_dbm":        float,  # same amplitude in dBm at the
                                                        # DAC output; None if DAC scale
                                                        # could not be queried
                  # Sweep data
                  "sweep_direction":        str,
                  "frequencies":            np.ndarray,   # Hz
                  "iq_counts":              np.ndarray,   # complex128, in readout counts
                  "iq_volts":               np.ndarray,   # complex128, in volts at the
                                                          # board input port, derived by
                                                          # applying VOLTS_PER_ROC from
                                                          # core.transferfunctions
                  "phase_degrees":          np.ndarray,
              }

          For multi-module calls the second element is a *list* of
          ``multisweep_data_dict``, one per module in the same order as
          *module*.
    """

    # -----------------------------------------------------------------------
    # Guard: update_center_frequency not yet supported
    # -----------------------------------------------------------------------
    if update_center_frequency:
        raise NotImplementedError(
            "update_center_frequency=True is not yet supported.  "
            "Center frequency updates will be implemented via bias_kids."
        )

    # -----------------------------------------------------------------------
    # Resolve inputs → codes, cfs, amp_list, channels, res_info_dict
    # -----------------------------------------------------------------------
    if res_info_dict is not None:
        # Option B: extract everything from the registry
        codes = list(res_info_dict.keys())
        cfs = [res_info_dict[c]["bias_frequency"] for c in codes]
        channels = [res_info_dict[c]["channel_number"] for c in codes]

        if amp is None:
            amp_list = [res_info_dict[c]["bias_amplitude"] for c in codes]
        elif isinstance(amp, (int, float)):
            amp_list = [float(amp)] * len(codes)
        else:
            amp_list = list(amp)
            if len(amp_list) != len(codes):
                raise ValueError(
                    f"Length of amp ({len(amp_list)}) must match "
                    f"number of resonators in res_info_dict ({len(codes)})"
                )

    else:
        # Option A: build from explicit lists
        if center_frequencies is None:
            raise ValueError(
                "Must provide either res_info_dict or center_frequencies."
            )
        if amp is None:
            raise ValueError("Must provide amp when not using res_info_dict.")

        cfs = list(center_frequencies)

        if isinstance(amp, (int, float)):
            amp_list = [float(amp)] * len(cfs)
        else:
            amp_list = list(amp)
            if len(amp_list) != len(cfs):
                raise ValueError(
                    f"Length of amp ({len(amp_list)}) must match "
                    f"center_frequencies ({len(cfs)})"
                )

        if detector_ids is None:
            codes = _generate_unique_detector_ids(len(cfs))
        else:
            codes = list(detector_ids)
            if len(codes) != len(cfs):
                raise ValueError(
                    f"Length of detector_ids ({len(codes)}) must match "
                    f"center_frequencies ({len(cfs)})"
                )

        # 1-based permanent channel numbers, one per resonator
        channels = list(range(1, len(cfs) + 1))

        # Build fresh res_info_dict (caller may capture the returned ref)
        res_info_dict = {
            code: {
                "bias_frequency": cf,
                "bias_amplitude": amp_val,
                "channel_number": ch,
            }
            for code, cf, amp_val, ch in zip(codes, cfs, amp_list, channels)
        }

    # Amplitude sanity check
    for code, a in zip(codes, amp_list):
        if a <= 0:
            warnings.warn(
                f"Amplitude for resonator {code!r} (amp={a}) is non-positive. "
                "Results may be invalid."
            )

    # -----------------------------------------------------------------------
    # Parallel module execution
    # -----------------------------------------------------------------------
    if isinstance(module, list):
        if not module:
            raise ValueError("Module list cannot be empty.")
        in_first_bank = all(1 <= m <= 4 for m in module)
        in_second_bank = all(5 <= m <= 8 for m in module)
        if not (in_first_bank or in_second_bank):
            raise ValueError(
                f"Module list must be entirely in [1..4] or [5..8], got: {module}"
            )
        tasks = [
            crs.multisweep(
                span_hz=span_hz,
                npoints_per_sweep=npoints_per_sweep,
                # Use the already-resolved res_info_dict so codes / channels
                # are stable across modules.
                res_info_dict=res_info_dict,
                amp=amp,
                nsamps=nsamps,
                sweep_direction=sweep_direction,
                update_center_frequency=update_center_frequency,
                module=m,
                progress_callback=progress_callback,
                data_callback=data_callback,
            )
            for m in module
        ]
        results_list = await asyncio.gather(*tasks)
        # All modules share the same res_info_dict (codes / channels are
        # module-agnostic).  Return it once alongside a per-module data list.
        return results_list[0][0], [r[1] for r in results_list]

    # -----------------------------------------------------------------------
    # Single-module: validate inputs
    # -----------------------------------------------------------------------
    if not cfs:
        warnings.warn("No center frequencies provided. Returning empty results.")
        return res_info_dict, {}

    # Query the hardware DAC full-scale so we can report amplitude in dBm.
    try:
        dac_scale_dbm = await crs.get_dac_scale('DBM', module=module)
    except Exception:
        dac_scale_dbm = None

    dec = await crs.get_decimation()
    max_channels = 128 if dec <= 3 else 1024

    if len(cfs) > max_channels:
        raise ValueError(
            f"Number of resonances ({len(cfs)}) exceeds maximum "
            f"channels ({max_channels})"
        )
    if npoints_per_sweep < 2:
        raise ValueError("npoints_per_sweep must be at least 2.")
    if span_hz <= 0:
        raise ValueError("span_hz must be positive.")

    # -----------------------------------------------------------------------
    # Build fast lookup dicts (all keyed by code or by CF)
    # -----------------------------------------------------------------------
    code_to_cf = dict(zip(codes, cfs))
    code_to_amp = dict(zip(codes, amp_list))
    code_to_channel = dict(zip(codes, channels))
    # CF-keyed lookups are used during the NCO-region sweep loop
    cf_to_code = {cf: code for code, cf in zip(codes, cfs)}
    cf_to_channel = {cf: ch for cf, ch in zip(cfs, channels)}

    # -----------------------------------------------------------------------
    # Pre-allocate sweep frequency arrays per resonator
    # -----------------------------------------------------------------------
    resonance_data: dict[str, dict] = {}
    for code, cf in zip(codes, cfs):
        if sweep_direction == "upward":
            pts = np.linspace(cf - span_hz / 2, cf + span_hz / 2,
                              npoints_per_sweep, endpoint=True)
        elif sweep_direction == "downward":
            pts = np.linspace(cf + span_hz / 2, cf - span_hz / 2,
                              npoints_per_sweep, endpoint=True)
        else:
            raise ValueError(
                f"Invalid sweep_direction: {sweep_direction!r}. "
                "Must be 'upward' or 'downward'."
            )
        resonance_data[code] = {
            "frequencies": pts,
            "iq_counts": np.zeros(npoints_per_sweep, dtype=np.complex128),
        }

    # -----------------------------------------------------------------------
    # Group resonances into NCO regions (each ≤ 500 MHz wide)
    # -----------------------------------------------------------------------
    MAX_NCO_SPAN_HZ = 500e6
    sorted_cfs = sorted(cfs)
    nco_regions: list[list[float]] = []

    if len(sorted_cfs) == 1:
        nco_regions.append(sorted_cfs)
    else:
        current_region = [sorted_cfs[0]]
        region_min = sorted_cfs[0] - span_hz / 2
        for cf in sorted_cfs[1:]:
            cf_max = cf + span_hz / 2
            if cf_max - region_min > MAX_NCO_SPAN_HZ:
                nco_regions.append(current_region)
                current_region = [cf]
                region_min = cf - span_hz / 2
            else:
                current_region.append(cf)
        nco_regions.append(current_region)

    nco_frequencies = [
        (
            min(cf - span_hz / 2 for cf in region)
            + max(cf + span_hz / 2 for cf in region)
        ) / 2
        for region in nco_regions
    ]

    # -----------------------------------------------------------------------
    # Main measurement loop
    # -----------------------------------------------------------------------
    total_nco_regions = len(nco_regions)

    for region_idx, region_cfs in enumerate(nco_regions):
        current_nco_freq = nco_frequencies[region_idx]
        await crs.set_nco_frequency(current_nco_freq, module=module)

        for point_idx in range(npoints_per_sweep):
            async with crs.tuber_context() as ctx:
                for cf in region_cfs:
                    code = cf_to_code[cf]
                    channel = cf_to_channel[cf]
                    freq_rel = (
                        resonance_data[code]["frequencies"][point_idx]
                        - current_nco_freq
                    )
                    ctx.set_frequency(freq_rel, channel=channel, module=module)
                    if not point_idx:
                        # Set amplitude once at the start of each NCO region
                        ctx.set_amplitude(
                            code_to_amp[code], channel=channel, module=module
                        )

                if not point_idx:
                    # Zero out all channels not active in this region
                    active_channels = {cf_to_channel[cf] for cf in region_cfs}
                    for ch in range(1, max_channels + 1):
                        if ch not in active_channels:
                            ctx.set_amplitude(0, channel=ch, module=module)

                await ctx()

            # Acquire averaged samples for every channel simultaneously
            samples = await crs.get_samples(
                nsamps, average=True, channel=None, module=module
            )

            for cf in region_cfs:
                code = cf_to_code[cf]
                ch_idx = cf_to_channel[cf] - 1  # 0-based index into samples arrays
                resonance_data[code]["iq_counts"][point_idx] = (
                    samples.mean.i[ch_idx] + 1j * samples.mean.q[ch_idx]
                )

            # Progress callback (0–100 %)
            if progress_callback:
                overall_pct = (
                    (region_idx + (point_idx + 1) / npoints_per_sweep)
                    / total_nco_regions
                ) * 100.0
                progress_callback(module, overall_pct)

            # Data callback: partial sweep data keyed by code
            if data_callback:
                n = point_idx + 1
                intermediate = {
                    code: {
                        "frequencies":            resonance_data[code]["frequencies"][:n],
                        "iq_counts":              resonance_data[code]["iq_counts"][:n],
                        "iq_volts":               convert_roc_to_volts(
                                                      resonance_data[code]["iq_counts"][:n]
                                                  ),
                        "sweep_center_frequency": code_to_cf[code],
                    }
                    for code in codes
                }
                data_callback(module, intermediate)

    # -----------------------------------------------------------------------
    # Assemble final output dict
    # -----------------------------------------------------------------------
    multisweep_data_dict: dict[str, dict] = {
        code: {
            # --- Snapshots from res_info_dict at sweep time ---
            "bias_frequency":         res_info_dict[code]["bias_frequency"],
            "bias_amplitude":         res_info_dict[code]["bias_amplitude"],
            "channel_number":         res_info_dict[code]["channel_number"],
            # --- Actual values used for this sweep call ---
            "sweep_center_frequency":     code_to_cf[code],   # = bias_frequency by default
            "sweep_amplitude_normalized": code_to_amp[code],  # may differ from bias_amplitude
            "sweep_amplitude_dbm": (
                convert_dac_normalized_to_dbm(code_to_amp[code], dac_scale_dbm)
                if dac_scale_dbm is not None else None
            ),
            # --- Sweep data ---
            "sweep_direction":        sweep_direction,
            "frequencies":            resonance_data[code]["frequencies"],
            "iq_counts":              resonance_data[code]["iq_counts"],
            "iq_volts":               convert_roc_to_volts(
                                          resonance_data[code]["iq_counts"]
                                      ),
            "phase_degrees":          np.degrees(
                                          np.angle(resonance_data[code]["iq_counts"])
                                      ),
        }
        for code in codes
    }

    # -----------------------------------------------------------------------
    # Hardware cleanup: zero all channels
    # -----------------------------------------------------------------------
    try:
        async with crs.tuber_context() as ctx:
            for ch in range(1, max_channels + 1):
                ctx.set_amplitude(0, channel=ch, module=module)
            await ctx()
    except Exception as exc:
        warnings.warn(f"Hardware cleanup failed for module {module}: {exc}")

    return res_info_dict, multisweep_data_dict
