"""The standard simulated array: one reproducible set of resonators for tests
and notebooks to share.

Every test that drives a measurement against the simulator has to build an
array first, and every one that builds its own is a test about a different
array. This module fixes one: a small, seeded array with the simulator's
default physics and noise except for a warmer bath (Q near 5e4 rather than
3e5, so the resonances are resolvable on the default sweep grids), biased by
the simulator itself so a caller can go straight to sweeping. The tests in ``test/tuning/`` and the notebook that
characterises the array, ``test/notebooks/test_standard_mock_array.md``, all
use it through :func:`standard_array`; a test that needs something else (noise
off, a bigger array) passes ``overrides`` and says so in its name.

The configuration is a plain dict so a notebook can print it, and the builder
is a coroutine because that is what a CRS is driven with. Nothing here streams
UDP: the array is served over RPC alone, which keeps it in the quick test tier
(see ``test/README.md``).
"""

from __future__ import annotations

from typing import Any, Mapping

from ..core import resonators as _resonators
from ..core.schema import CRS
from ..core.session import load_session
from ..resonator_names import syllabic_names_from_frequency
from . import config as _config

__all__ = ["STANDARD_ARRAY", "STANDARD_MODULE", "SESSION", "standard_array"]

#: The module the array is served on. One number, so a test never has to ask.
STANDARD_MODULE = 1

#: Overrides on ``rfmux.mock.config.MOCK_DEFAULTS``. Everything not named here
#: is the simulator's default, noise included: the array is meant to look like
#: a board, and a test that wants it quiet turns the noise off itself.
STANDARD_ARRAY: dict[str, Any] = {
    "num_resonances": 8,
    # A 100 MHz band, so one NCO setting covers the whole array and a netanal
    # over it is quick. Eight resonators in it are well separated.
    "freq_start": 1.00e9,
    "freq_end": 1.10e9,
    "resonator_random_seed": 42,      # the same eight resonators every time
    "auto_bias_kids": True,           # the simulator parks a tone on each one
    # The bath temperature sets Q through the quasiparticle density. The
    # default 0.12 K gives Q near 3e5, a 4 kHz linewidth that a 1 kHz sweep
    # grid barely resolves; 0.23 K gives Q near 5e4 and a 20 kHz linewidth,
    # so multisweep's default 100 kHz span holds five linewidths with twenty
    # points across each, and a netanal on a few-kHz grid cannot miss one.
    "T": 0.23,
}

#: The hardware map for a simulated board served over RPC.
SESSION = """
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
"""


async def standard_array(
    overrides: Mapping[str, Any] | None = None,
    *,
    module: int = STANDARD_MODULE,
) -> tuple[CRS, _resonators.ResonatorCatalog]:
    """Stand up the standard array and hand back the board and its catalog.

    The catalog has one resonator per channel, in frequency order, at the tone
    the simulator's own biasing chose and at its bias amplitude: the point the
    tuning flow starts from when the array has already been found once. Names
    are derived from the frequencies, so they are the same every run.

    Args:
        overrides: changes to :data:`STANDARD_ARRAY`, applied on top of it. A
            test that passes any should say what it changed in its name.
        module: which module to serve the array on.

    Returns:
        ``(crs, catalog)``. The CRS is resolved and its resonators generated;
        no UDP streaming has been started.
    """
    cfg = dict(STANDARD_ARRAY)
    if overrides:
        cfg.update(overrides)
    cfg = _config.apply_overrides(cfg)

    session = load_session(SESSION)
    try:
        crs = session.query(CRS).one()
        await crs.resolve()
        await crs.generate_resonators(cfg)

        nco = await crs.get_nco_frequency(module=module)
        tones = [
            nco + await crs.get_frequency(channel=channel, module=module)
            for channel in range(1, cfg["num_resonances"] + 1)
        ]
        # Named from the frequencies, so the same resonator has the same name in
        # every run and a test can say catalog["..."] and mean one of them.
        catalog = _resonators.ResonatorCatalog.from_frequencies(
            tones, module=module, amplitude=float(cfg["bias_amplitude"]),
            names=syllabic_names_from_frequency,
        )
        return crs, catalog
    except BaseException:
        session.close()
        raise
