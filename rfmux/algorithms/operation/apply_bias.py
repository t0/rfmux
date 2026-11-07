"""Program a catalog's bias frequencies and amplitudes on its module.

Call ``await crs.apply_bias(report.catalog)`` after bias finding.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from pathlib import Path

from ...core.dac_scale import dac_scale_setting_dbm
from ...core.hardware_map import macro
from ...core.resonators import ResonatorCatalog, on_grid
from ...core.schema import CRS
from ...core.session_folder import register_export
from ...core.transferfunctions import (
    ALLOWED_NCO_BANDWIDTH_HZ,
    BALUN_LOSS_DB,
    BASE_FREQUENCY,
    FREQ_QUANTUM,
    convert_dbm_to_amplitude,
)
from ...tuning import store


def _unreachable(nco_hz: float, resonators: list) -> list:
    """Return resonators outside the allowed bandwidth around this NCO."""
    reach = ALLOWED_NCO_BANDWIDTH_HZ / 2
    return [r for r in resonators if abs(r.bias.frequency_hz - nco_hz) > reach]


def _aligned(nco_hz: float) -> bool:
    """Check tone-grid alignment to within one DDS frequency quantum."""
    return abs(nco_hz - on_grid(nco_hz)) < FREQ_QUANTUM


def _describe_problem(nco_hz: float, unreachable: list) -> str:
    """Describe tones outside the NCO bandwidth or an NCO off the tone grid."""
    if unreachable:
        worst = max(unreachable, key=lambda r: abs(r.bias.frequency_hz - nco_hz))
        return (
            f"{len(unreachable)} of the catalog's tones are outside the "
            f"{ALLOWED_NCO_BANDWIDTH_HZ / 1e6:.0f} MHz it reaches — {worst.name} "
            f"at {worst.bias.frequency_hz / 1e6:.6f} MHz is "
            f"{(worst.bias.frequency_hz - nco_hz) / 1e6:+.3f} MHz away"
        )
    return (
        f"it is {nco_hz - on_grid(nco_hz):+.4f} Hz off the tone grid "
        f"({BASE_FREQUENCY:.6f} Hz steps), so every offset computed from it "
        f"would be off-grid too and the tones would not land where the catalog "
        f"says they do"
    )


def _at_measured_power(catalog: ResonatorCatalog, setting_dbm, module: int):
    """A copy of *catalog* whose amplitudes give each bias point the power it
    was measured at, under this DAC setting, and the ``(name, old, new)``
    amplitude of every point that had to change.

    A point without a measured power, or a board that reports no DAC scale,
    keeps its amplitude. The calibration is kept: the resonator sees the same
    power it was calibrated at.
    """
    applied = catalog.copy()
    if setting_dbm is None:
        return applied, []
    full_scale_dbm = setting_dbm - BALUN_LOSS_DB
    moved, too_loud = [], []
    for r in applied:
        power = r.bias.bias_power_dbm
        if power is None:
            continue
        amplitude = float(convert_dbm_to_amplitude(power, full_scale_dbm))
        if math.isclose(amplitude, r.bias.amplitude, rel_tol=1e-9):
            continue
        if amplitude > 1:
            too_loud.append(f"{r.name} ({power:.2f} dBm)")
            continue
        moved.append((r.name, r.bias.amplitude, amplitude))
        r.bias = replace(r.bias, amplitude=amplitude)
    if too_loud:
        raise ValueError(
            f"Module {module}'s DAC scale is set to {setting_dbm:g} dBm, so its "
            f"full scale at the output is {full_scale_dbm:g} dBm, below the power "
            f"{len(too_loud)} bias points were measured at: "
            f"{', '.join(too_loud)}. Raise the DAC scale; no tones were applied."
        )
    return applied, moved


def _warn_recomputed(moved, catalog, setting_dbm, module) -> None:
    """Say which amplitudes were recomputed, and from which DAC setting."""
    name, old, new = moved[0]
    power = catalog[name].bias.bias_power_dbm
    measured_at = power - 20 * math.log10(old) + BALUN_LOSS_DB
    warnings.warn(
        f"Module {module}'s DAC scale is set to {setting_dbm:g} dBm, not the "
        f"{measured_at:g} dBm the bias powers were measured at, so "
        f"{len(moved)} tones' amplitudes were recomputed to keep their measured "
        f"powers (e.g. {name}: {old:.4g} -> {new:.4g}).",
        UserWarning, stacklevel=3)


@macro(CRS, register=True)
async def apply_bias(
    crs,
    catalog: ResonatorCatalog,
    *,
    allow_nco_reset: bool = True,
    save: bool | None = None,
    label: str | None = None,
) -> str | None:
    """Program each catalog member's bias frequency and amplitude.

    Other channels are not cleared, and IQ rotation is not applied. To start
    with a quiet module, call ``crs.clear_channels(module=...)`` first.
    The catalog is read without modification.

    A bias point carrying the power it was measured at keeps that power: if
    the module's DAC scale has changed since, its amplitude is recomputed for
    the present scale, with a warning saying so.

    The applied catalog is saved as a ``bias_catalog`` file holding the NCO
    and the DAC scale setting read back from the board, and listed in the
    folder's session metadata so ``rfmux record`` finds it. Read it with
    ``rfmux.tuning.load_bias_catalog``.

    Args:
        catalog: resonators to program, with their module and channel bindings.
        allow_nco_reset: if True, move an unreachable or off-grid NCO to the
            catalog's grid-aligned midpoint. A usable NCO is left unchanged.
            Moving the NCO also shifts tones on channels outside the catalog.
        save: save the bias_catalog file. None uses
            ``store.autosave_enabled()``.
        label: label appended to the filename when saving.

    Returns:
        The bias_catalog file's path; None when nothing was saved.

    Raises:
        ValueError: the catalog is empty, spans more than
            ``ALLOWED_NCO_BANDWIDTH_HZ``, needs a forbidden NCO reset, or holds
            a power above the DAC scale's full scale. No settings have been
            programmed in these cases.
        RuntimeError: the NCO readback remains unreachable or off-grid after
            a reset. The NCO has moved, but no tones have been applied.
    """
    if not isinstance(catalog, ResonatorCatalog):
        # Explain how to extract a catalog if the caller passed a bias report.
        raise TypeError(
            f"apply_bias takes a ResonatorCatalog, not a "
            f"{type(catalog).__name__}. Bias finding hands back a report; the "
            f"catalog is report.catalog."
        )

    if len(catalog) == 0:
        raise ValueError(
            "The catalog is empty, so there is nothing to apply. A catalog "
            "arrives here from bias finding (report.catalog), which carries "
            "one resonator per sweep it was given."
        )

    module = catalog.module
    resonators = list(catalog)  # bias-frequency order

    lowest = min(resonators, key=lambda r: r.bias.frequency_hz)
    highest = max(resonators, key=lambda r: r.bias.frequency_hz)
    span_hz = highest.bias.frequency_hz - lowest.bias.frequency_hz

    if span_hz > ALLOWED_NCO_BANDWIDTH_HZ:
        raise ValueError(
            f"The catalog's bias frequencies span {span_hz / 1e6:.1f} MHz, from "
            f"{lowest.name} at {lowest.bias.frequency_hz / 1e6:.6f} MHz to "
            f"{highest.name} at {highest.bias.frequency_hz / 1e6:.6f} MHz, and "
            f"one NCO reaches {ALLOWED_NCO_BANDWIDTH_HZ / 1e6:.0f} MHz. Module "
            f"{module} plays one NCO at a time, so no NCO frequency puts all of "
            f"these tones on the air together. Build a catalog whose bias "
            f"frequencies fit inside one band and apply that."
        )

    nco_hz = float(await crs.get_nco_frequency(module=module))
    unreachable = _unreachable(nco_hz, resonators)

    if unreachable or not _aligned(nco_hz):
        # The midpoint of the catalog, on the grid. Centring leaves the most
        # room on both sides for a bias point that moves later.
        wanted_hz = on_grid((lowest.bias.frequency_hz + highest.bias.frequency_hz) / 2)

        if not allow_nco_reset:
            raise ValueError(
                f"Module {module}'s NCO is at {nco_hz / 1e6:.6f} MHz and "
                f"{_describe_problem(nco_hz, unreachable)}. allow_nco_reset=False "
                f"forbids moving it, so no tones were applied. Either call again "
                f"with allow_nco_reset=True, or set the NCO yourself: "
                f"await crs.set_nco_frequency({wanted_hz!r}, module={module})."
            )

        await crs.set_nco_frequency(wanted_hz, module=module)
        # Use the measured NCO frequency to calculate tone offsets.
        nco_hz = float(await crs.get_nco_frequency(module=module))

        unreachable = _unreachable(nco_hz, resonators)
        if unreachable or not _aligned(nco_hz):
            raise RuntimeError(
                f"Module {module}'s NCO was set to {wanted_hz / 1e6:.6f} MHz and "
                f"read back as {nco_hz / 1e6:.6f} MHz, and "
                f"{_describe_problem(nco_hz, unreachable)}. The NCO has moved; "
                f"no tones were applied."
            )

    setting_dbm = await dac_scale_setting_dbm(crs, module)
    applied, moved = _at_measured_power(catalog, setting_dbm, module)

    async with crs.tuber_context() as ctx:
        for r in applied:
            ctx.set_frequency(
                r.bias.frequency_hz - nco_hz, channel=r.channel, module=module
            )
            ctx.set_amplitude(r.bias.amplitude, channel=r.channel, module=module)
        await ctx()
    if moved:
        _warn_recomputed(moved, catalog, setting_dbm, module)

    path = store.maybe_save(
        {
            "measurement": "bias_catalog",
            "module": module,
            "nco_frequency_hz": nco_hz,
            "dac_scale_setting_dbm": setting_dbm,
            "catalog": applied.to_dict(),
        },
        "bias_catalog", save=save, label=label, module=module,
    )
    if path is None:
        return None
    register_export(Path(path).parent, Path(path).name, "bias_catalog",
                    f"module{module}")
    return str(path)
