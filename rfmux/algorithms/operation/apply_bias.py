"""Program a catalog's bias frequencies and amplitudes on its module.

Call ``await crs.apply_bias(report.catalog)`` after bias finding.
"""

from __future__ import annotations

from ...core.hardware_map import macro
from ...core.resonators import ResonatorCatalog, on_grid
from ...core.schema import CRS
from ...core.transferfunctions import (
    ALLOWED_NCO_BANDWIDTH_HZ,
    BASE_FREQUENCY,
    FREQ_QUANTUM,
)


def _unreachable(nco_hz: float, resonators: list) -> list:
    """The resonators whose tones fall outside the band this NCO carries."""
    reach = ALLOWED_NCO_BANDWIDTH_HZ / 2
    return [r for r in resonators if abs(r.bias.frequency_hz - nco_hz) > reach]


def _aligned(nco_hz: float) -> bool:
    """Check tone-grid alignment to within one DDS frequency quantum."""
    return abs(nco_hz - on_grid(nco_hz)) < FREQ_QUANTUM


def _describe_problem(nco_hz: float, unreachable: list) -> str:
    """Why this NCO will not do, as a phrase that follows the NCO frequency."""
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


@macro(CRS, register=True)
async def apply_bias(
    crs,
    catalog: ResonatorCatalog,
    *,
    allow_nco_reset: bool = True,
):
    """Program each catalog member's bias frequency and amplitude.

    Other channels are not cleared, and IQ rotation is not applied. To start
    with a quiet module, call ``crs.clear_channels(module=...)`` first.
    The catalog is read without modification. Returns None.

    Args:
        catalog: resonators to program, with their module and channel bindings.
        allow_nco_reset: if True, move an unreachable or off-grid NCO to the
            catalog's grid-aligned midpoint. A usable NCO is left unchanged.
            Moving the NCO also shifts tones on channels outside the catalog.

    Raises:
        ValueError: the catalog is empty, spans more than
            ``ALLOWED_NCO_BANDWIDTH_HZ``, or needs a forbidden NCO reset.
            No settings have been programmed in these cases.
        RuntimeError: the NCO readback remains unreachable or off-grid after
            a reset. The NCO has moved, but no tones have been applied.
    """
    if not isinstance(catalog, ResonatorCatalog):
        # Overwhelmingly this is a BiasReport passed whole. Say so, rather than
        # letting it fail several lines later on an iteration it does not
        # support.
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
        # Read back rather than trusting the number we sent: the offsets below
        # are only right if they are computed from the NCO the board actually
        # settled on.
        nco_hz = float(await crs.get_nco_frequency(module=module))

        unreachable = _unreachable(nco_hz, resonators)
        if unreachable or not _aligned(nco_hz):
            raise RuntimeError(
                f"Module {module}'s NCO was set to {wanted_hz / 1e6:.6f} MHz and "
                f"read back as {nco_hz / 1e6:.6f} MHz, and "
                f"{_describe_problem(nco_hz, unreachable)}. The NCO has moved; "
                f"no tones were applied."
            )

    async with crs.tuber_context() as ctx:
        for r in resonators:
            ctx.set_frequency(
                r.bias.frequency_hz - nco_hz, channel=r.channel, module=module
            )
            ctx.set_amplitude(r.bias.amplitude, channel=r.channel, module=module)
        await ctx()
