"""Which channels a multisweep is allowed to silence.

The one test that drives multisweep's measurement loop rather than its input
resolution, because this is behaviour no amount of argument-checking can show:
the loop zeroes channels on the way in (to quiet its other NCO regions) and
again on the way out. Both must stay inside the set of channels the sweep
itself put a tone on.

Spawns a MockCRS server, hence slow_acquisition.
"""

import pytest

import rfmux
from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog

pytestmark = pytest.mark.slow_acquisition

MODULE = 1
# Well clear of the swept channels, and inside the 1024-channel limit that
# applies at the mock's default decimation.
FOREIGN_CHANNEL = 900
FOREIGN_AMPLITUDE = 0.25


@pytest.fixture(scope="module")
def crs_mock():
    session = rfmux.load_session(
        """
        !HardwareMap
        - !flavour "rfmux.mock"
        - !CRS { serial: "0000", hostname: "127.0.0.1" }
        """
    )
    return session.query(rfmux.CRS).one()


async def a_catalog_of_real_resonators(crs, n=3):
    """A catalog on frequencies the mock actually has resonators at, so the
    channel bookkeeping is exercised by a sweep of the shape a user would
    actually take rather than one over a flat baseline."""
    _, frequencies = await crs.generate_resonators(
        {"num_resonances": n, "auto_bias_kids": False}
    )
    assert len(frequencies) >= n, f"mock produced only {len(frequencies)}"

    return ResonatorCatalog(
        [
            Resonator(
                name=f"R{i + 1:04d}",
                channel=i + 1,
                bias=BiasPoint(frequency_hz=float(f), amplitude=1e-3),
            )
            for i, f in enumerate(sorted(frequencies)[:n])
        ],
        module=MODULE,
    )


@pytest.mark.asyncio
async def test_a_tone_the_caller_parked_elsewhere_survives_the_sweep(crs_mock):
    """A user with something live on channel 900 and a 3-resonator catalog does
    not expect the sweep to reach across and silence it."""
    crs = crs_mock
    await crs.resolve()

    catalog = await a_catalog_of_real_resonators(crs)

    await crs.set_frequency(50e6, channel=FOREIGN_CHANNEL, module=MODULE)
    await crs.set_amplitude(
        FOREIGN_AMPLITUDE, channel=FOREIGN_CHANNEL, module=MODULE
    )

    sweeps = await crs.multisweep(
        catalog,
        span_hz=100e3,
        npoints_per_sweep=9,
        nsamps=2,
    )
    assert list(sweeps) == [crs.module[MODULE].index()]
    sections = sweeps[crs.module[MODULE].index()]["results"][0]["upward"]
    assert sorted(sections) == ["R0001", "R0002", "R0003"]

    assert await crs.get_amplitude(
        channel=FOREIGN_CHANNEL, module=MODULE
    ) == pytest.approx(FOREIGN_AMPLITUDE)


@pytest.mark.asyncio
async def test_the_sweeps_own_channels_are_left_silent(crs_mock):
    """The other half of the contract: what multisweep does set, it cleans up."""
    crs = crs_mock
    await crs.resolve()

    catalog = await a_catalog_of_real_resonators(crs)
    await crs.multisweep(
        catalog,
        span_hz=100e3,
        npoints_per_sweep=5,
        nsamps=2,
    )

    for r in catalog:
        assert await crs.get_amplitude(channel=r.channel, module=MODULE) == 0
