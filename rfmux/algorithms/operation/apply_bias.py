"""
apply_bias: put a catalog's tones on the air.

Bias finding decides where each resonator's tone belongs; this is the step that
plays it.  One call programs one module: every resonator in the catalog gets
its bias frequency and its bias amplitude, on its own channel, in a single
tuber context.

Registered as a macro, so it is available as::

    report = find_bias_points(sweeps)
    await crs.apply_bias(report.catalog)

Nothing comes back.  An operation either put the board into the state you asked
for or raised saying why it could not, and there is no third outcome worth
returning a value to describe.

What it deliberately does not do
--------------------------------
* **It does not clear the channels the catalog leaves out.**  A tone parked by
  hand or by another algorithm survives the call — the same rule multisweep
  follows, for the same reason: zeroing the whole module would be tidier for us
  and destructive for them.  The corollary is the caller's job.  Applying a
  bias does not leave the module otherwise quiet, so a run that needs silence
  arranges it with ``crs.clear_channels(module=...)`` first.
* **It does not rotate.**  :attr:`~rfmux.core.resonators.BiasPoint.iq_rotation_deg`
  exists and this never reads it.  The angle comes from a timestream, so
  programming it belongs with the step that measures it.
* **It does not re-quantize.**  A bias point's frequency landed on the tone
  grid when the point was constructed.  Rounding it again here would be a
  second opinion about a settled number.

The NCO is this operation's business
------------------------------------
A module has one NCO and every channel's frequency is programmed as an offset
from it, which makes the NCO part of applying a bias whether we want it to be
or not.  Two things have to be true of it.

**It has to reach every tone.**  Each bias frequency must sit within half of
``ALLOWED_NCO_BANDWIDTH_HZ`` of the NCO.  A catalog whose bias frequencies span
more than that cannot be applied at all — not by choosing a cleverer NCO, and
not in two passes, because one module plays one NCO at a time and these tones
would have to be on the air together.  That raises, and the fix is a catalog
that fits rather than an argument to this call.

**It has to be on the tone grid.**  The grid applies to the offset that gets
programmed, not to the absolute frequency, so an off-grid NCO puts every tone
off-grid however carefully the bias point was quantized.  When we choose the
NCO we snap it, and a catalog's tones then land exactly where the catalog says.

If the current NCO already satisfies both, it is left alone: moving it would
disturb every other channel on the module, including ones this catalog does not
own.  Otherwise it is reset to the quantized midpoint of the catalog's
frequencies — unless ``allow_nco_reset=False``, which turns both conditions
into errors instead.  Note that an off-grid NCO is one of them: a board whose
NCO is not on the grid cannot apply a catalog faithfully, and with the reset
forbidden there is nothing this call can do about it but say so.
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
    """Is this NCO on the tone grid?

    Compared at the DDS's own resolution, ``FREQ_QUANTUM`` — the smallest
    difference between two frequencies the synthesizer can hold, and so the
    finest sense in which two frequencies are the same one. The tone grid is an
    exact multiple of it (both divide ``COMB_SAMPLING_FREQ / 256``, by 2**12 and
    2**32), so a grid point is always a frequency the DDS can land on exactly
    and any remainder below one quantum is arithmetic left over from computing
    it, not a setting the board could act on.
    """
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
    """Program every resonator in ``catalog`` onto its channel.

    Args:
        catalog: the resonators to bias.  Its ``module`` says where; each
            resonator's ``channel`` and ``bias`` say what.  Read, never written.
            The NCO this call settles on is not recorded there — the catalog
            does not carry one, deliberately — so a caller who needs to know it
            afterwards asks the board: ``await crs.get_nco_frequency(module=...)``.
        allow_nco_reset: may this call move the module's NCO?  The default
            moves it when, and only when, the current one cannot carry the
            catalog.  ``False`` forbids it outright: the NCO in place is used
            as it is, or the call raises.

    Raises:
        ValueError: the catalog is empty; or its bias frequencies span more
            than ``ALLOWED_NCO_BANDWIDTH_HZ``, which no single NCO can carry;
            or ``allow_nco_reset=False`` and the NCO in place is unusable.
            Nothing has been programmed in any of these cases.
        RuntimeError: the NCO was set and read back as something that still
            will not do.  The NCO has moved; no tones were applied.
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
    resonators = list(catalog)  # channel order, per ResonatorCatalog

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
