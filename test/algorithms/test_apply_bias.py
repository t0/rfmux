"""
Putting a catalog's bias points on the board.

Uses a MockCRS session directly rather than create_mock_crs(): applying a bias
is tuber RPC and nothing else — no samples are taken — so these stay out of the
acquisition tier.

The board state each test asserts on is read back through the same getters a
user would use, so a test that passes says the tones are where the catalog says
they are, not merely that apply_bias called the functions we expected.
"""

import pytest

import rfmux
from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog, on_grid
from rfmux.core.transferfunctions import (
    ALLOWED_NCO_BANDWIDTH_HZ,
    BASE_FREQUENCY,
    FREQ_QUANTUM,
)
from rfmux.tuning import BiasReport

SESSION = """
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
"""

MODULE = 1
CENTRE_HZ = 1e9
AMPLITUDE = 0.01

# Well clear of the catalog's channels, and inside the 1024-channel limit that
# applies at the mock's default decimation.
FOREIGN_CHANNEL = 900
FOREIGN_AMPLITUDE = 0.25

# An NCO that reaches the catalog below but is not the midpoint apply_bias
# would choose, so "unchanged" is a distinguishable outcome from "reset".
SETTLED_NCO_HZ = on_grid(CENTRE_HZ - 10e6)


@pytest.fixture
def crs_mock():
    """A fresh board per test — these mutate the NCO and the channels."""
    session = rfmux.load_session(SESSION)
    return session.query(rfmux.CRS).one()


def a_catalog(*offsets_hz, amplitude=AMPLITUDE):
    """A catalog at CENTRE_HZ + each offset, one channel each from 1."""
    return ResonatorCatalog(
        [
            Resonator(
                name=f"R{i + 1:04d}",
                channel=i + 1,
                bias=BiasPoint(frequency_hz=CENTRE_HZ + offset, amplitude=amplitude),
            )
            for i, offset in enumerate(offsets_hz)
        ],
        module=MODULE,
    )


async def assert_nothing_applied(crs, catalog):
    """No channel the catalog names carries a tone."""
    for r in catalog:
        assert await crs.get_amplitude(channel=r.channel, module=MODULE) is None
        assert await crs.get_frequency(channel=r.channel, module=MODULE) is None


@pytest.mark.asyncio
async def test_every_tone_lands_on_its_channel(crs_mock):
    crs = crs_mock
    await crs.resolve()
    await crs.set_nco_frequency(SETTLED_NCO_HZ, module=MODULE)

    catalog = a_catalog(-1e6, 0.0, +1e6)
    await crs.apply_bias(catalog)

    for r in catalog:
        assert await crs.get_frequency(
            channel=r.channel, module=MODULE
        ) == pytest.approx(r.bias.frequency_hz - SETTLED_NCO_HZ)
        assert await crs.get_amplitude(
            channel=r.channel, module=MODULE
        ) == pytest.approx(r.bias.amplitude)


@pytest.mark.asyncio
async def test_an_nco_that_already_works_is_left_alone(crs_mock):
    """Moving the NCO moves every tone on the module, including ones this
    catalog does not own. So it only moves when it has to."""
    crs = crs_mock
    await crs.resolve()
    await crs.set_nco_frequency(SETTLED_NCO_HZ, module=MODULE)

    await crs.apply_bias(a_catalog(-1e6, +1e6))

    assert await crs.get_nco_frequency(module=MODULE) == pytest.approx(SETTLED_NCO_HZ)


@pytest.mark.asyncio
async def test_a_channel_the_catalog_does_not_name_survives(crs_mock):
    """A user with something live on channel 900 does not expect applying a
    bias to a three-resonator catalog to reach across and silence it."""
    crs = crs_mock
    await crs.resolve()
    await crs.set_nco_frequency(SETTLED_NCO_HZ, module=MODULE)
    await crs.set_frequency(50e6, channel=FOREIGN_CHANNEL, module=MODULE)
    await crs.set_amplitude(FOREIGN_AMPLITUDE, channel=FOREIGN_CHANNEL, module=MODULE)

    await crs.apply_bias(a_catalog(-1e6, 0.0, +1e6))

    assert await crs.get_amplitude(
        channel=FOREIGN_CHANNEL, module=MODULE
    ) == pytest.approx(FOREIGN_AMPLITUDE)
    assert await crs.get_frequency(
        channel=FOREIGN_CHANNEL, module=MODULE
    ) == pytest.approx(50e6)


@pytest.mark.asyncio
async def test_an_nco_out_of_reach_is_reset_to_the_catalogs_midpoint(crs_mock):
    crs = crs_mock
    await crs.resolve()
    await crs.set_nco_frequency(on_grid(2e9), module=MODULE)

    catalog = a_catalog(-1e6, +3e6)
    await crs.apply_bias(catalog)

    nco = await crs.get_nco_frequency(module=MODULE)
    assert nco == pytest.approx(on_grid(CENTRE_HZ + 1e6))
    for r in catalog:
        assert await crs.get_frequency(
            channel=r.channel, module=MODULE
        ) == pytest.approx(r.bias.frequency_hz - nco)


@pytest.mark.asyncio
async def test_the_nco_it_chooses_is_on_the_tone_grid(crs_mock):
    """The grid binds the offset that gets programmed, not the absolute
    frequency, so an off-grid NCO would put every tone off-grid however
    carefully the bias point was quantized."""
    crs = crs_mock
    await crs.resolve()
    await crs.set_nco_frequency(on_grid(2e9), module=MODULE)

    # Offsets whose midpoint falls between two grid steps.
    catalog = a_catalog(-1e6, +1e6 + BASE_FREQUENCY / 3)
    await crs.apply_bias(catalog)

    nco = await crs.get_nco_frequency(module=MODULE)
    assert nco == pytest.approx(on_grid(nco))
    for r in catalog:
        offset = await crs.get_frequency(channel=r.channel, module=MODULE)
        assert offset == pytest.approx(on_grid(offset))


@pytest.mark.asyncio
async def test_an_nco_off_grid_by_less_than_the_dds_can_resolve_is_on_grid(crs_mock):
    """On-grid is decided at the DDS's own resolution. A remainder below one
    FREQ_QUANTUM is arithmetic, not a frequency the board could hold, and
    resetting the NCO over it would move every tone on the module for nothing."""
    crs = crs_mock
    await crs.resolve()
    barely_off = SETTLED_NCO_HZ + FREQ_QUANTUM / 2
    await crs.set_nco_frequency(barely_off, module=MODULE)

    await crs.apply_bias(a_catalog(-1e6, +1e6))

    assert await crs.get_nco_frequency(module=MODULE) == pytest.approx(barely_off)


@pytest.mark.asyncio
async def test_an_off_grid_nco_is_reset_even_though_it_reaches(crs_mock):
    crs = crs_mock
    await crs.resolve()
    off_grid = SETTLED_NCO_HZ + BASE_FREQUENCY / 3
    await crs.set_nco_frequency(off_grid, module=MODULE)

    await crs.apply_bias(a_catalog(-1e6, +1e6))

    nco = await crs.get_nco_frequency(module=MODULE)
    assert nco != pytest.approx(off_grid)
    assert nco == pytest.approx(on_grid(CENTRE_HZ))


@pytest.mark.asyncio
async def test_a_forbidden_reset_raises_and_moves_nothing(crs_mock):
    crs = crs_mock
    await crs.resolve()
    await crs.set_nco_frequency(on_grid(2e9), module=MODULE)

    catalog = a_catalog(-1e6, +1e6)
    with pytest.raises(ValueError, match="allow_nco_reset=False"):
        await crs.apply_bias(catalog, allow_nco_reset=False)

    assert await crs.get_nco_frequency(module=MODULE) == pytest.approx(on_grid(2e9))
    await assert_nothing_applied(crs, catalog)


@pytest.mark.asyncio
async def test_a_forbidden_reset_refuses_an_off_grid_nco_too(crs_mock):
    """It reaches every tone, but the offsets computed from it would be
    off-grid — and with the reset forbidden there is nothing to be done."""
    crs = crs_mock
    await crs.resolve()
    off_grid = SETTLED_NCO_HZ + BASE_FREQUENCY / 3
    await crs.set_nco_frequency(off_grid, module=MODULE)

    catalog = a_catalog(-1e6, +1e6)
    with pytest.raises(ValueError, match="off the tone grid"):
        await crs.apply_bias(catalog, allow_nco_reset=False)

    assert await crs.get_nco_frequency(module=MODULE) == pytest.approx(off_grid)
    await assert_nothing_applied(crs, catalog)


@pytest.mark.asyncio
async def test_a_catalog_wider_than_one_nco_raises(crs_mock):
    """No NCO carries these together, so this is a catalog to be rebuilt
    rather than a call to be retried with different arguments."""
    crs = crs_mock
    await crs.resolve()
    await crs.set_nco_frequency(SETTLED_NCO_HZ, module=MODULE)

    catalog = a_catalog(0.0, ALLOWED_NCO_BANDWIDTH_HZ + 1e6)
    with pytest.raises(ValueError, match="one NCO reaches"):
        await crs.apply_bias(catalog)

    assert await crs.get_nco_frequency(module=MODULE) == pytest.approx(SETTLED_NCO_HZ)
    await assert_nothing_applied(crs, catalog)


@pytest.mark.asyncio
async def test_a_report_passed_whole_says_so(crs_mock):
    """`apply_bias(report)` instead of `apply_bias(report.catalog)` is the
    mistake this call site invites, so it gets a sentence rather than a
    KeyError from somewhere inside."""
    crs = crs_mock
    await crs.resolve()

    catalog = a_catalog(-1e6, +1e6)
    report = BiasReport(catalog=catalog, findings=[])
    with pytest.raises(TypeError, match="report.catalog"):
        await crs.apply_bias(report)


@pytest.mark.asyncio
async def test_an_empty_catalog_raises(crs_mock):
    crs = crs_mock
    await crs.resolve()

    with pytest.raises(ValueError, match="empty"):
        await crs.apply_bias(ResonatorCatalog([], module=MODULE))
