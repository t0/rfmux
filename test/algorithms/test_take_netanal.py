"""What a network analysis hands back.

Two tiers. The portable half exercises :func:`pack_netanal` and the guards that
keep a netanal out of the sweep readers — a module's output is shaped the same
as ``multisweep``'s down to ``results`` and then is not, and that is the one
confusion the shared container makes possible.

The acquisition half runs the driver against a MockCRS. It is there for the
properties the packing cannot show on its own: that the frequency grid comes
back whole, once each and in the direction that was swept, across however many
NCO settings it took to measure. Chunks used to overlap by a point so their
phases could be stitched together, so "npoints in, npoints out" is a claim worth
a test.
"""

import asyncio
import copy

import numpy as np
import pytest

from rfmux.core.resonators import ResonatorCatalog
from rfmux.tuning import (
    AmplitudeSchedule,
    find_resonances_in_netanal,
    fit_sweeps,
    netanal_trace,
)
from rfmux.tuning.sweep_results import (
    RESULTS_SCHEMA_VERSION,
    collect_amplitude_iterations_for,
    pack_multisweep,
    pack_netanal,
)

MODULE = 1


def a_netanal(npoints=8, module=MODULE, sweep_direction="upward"):
    """One module's netanal, packed the way the driver packs it."""
    frequencies = np.linspace(1.0e9, 1.5e9, npoints)
    if sweep_direction == "downward":
        frequencies = frequencies[::-1]
    iq = np.ones(npoints, dtype=complex)
    return pack_netanal(
        {
            "frequencies": frequencies,
            "iq_counts": iq,
            "iq_volts": iq * 1e-7,
            "sweep_amplitude": 1e-3,
            "sweep_direction": sweep_direction,
        },
        module_id=f"crs0000_rmod{module}",
        module=module,
        amp=1e-3,
        fmin=1.0e9,
        fmax=1.5e9,
        npoints=npoints,
        nsamps=10,
        max_chans=1023,
        max_span=500e6,
        rotate_phase_to_0=True,
        sweep_direction=sweep_direction,
        requested_module=module,
    )


# ─── the packed shape ─────────────────────────────────────────────────────────


class TestPacking:
    pytestmark = pytest.mark.portable

    def test_it_is_keyed_by_module(self):
        assert list(a_netanal()) == ["crs0000_rmod1"]

    def test_the_output_says_what_made_it(self):
        module_netanal = a_netanal()["crs0000_rmod1"]

        # A literal, not the constant: bumping the version should mean editing
        # a test, because it is a claim about what readers of older files need.
        assert module_netanal["schema_version"] == 7
        assert module_netanal["schema_version"] == RESULTS_SCHEMA_VERSION
        assert module_netanal["measurement"] == "netanal"
        assert module_netanal["module"] == 1

    def test_call_params_records_the_call_verbatim(self):
        params = a_netanal()["crs0000_rmod1"]["call_params"]

        assert params == {
            "amp": 1e-3,
            "fmin": 1.0e9,
            "fmax": 1.5e9,
            "npoints": 8,
            "nsamps": 10,
            "max_chans": 1023,
            "max_span": 500e6,
            "rotate_phase_to_0": True,
            "sweep_direction": "upward",
            "module": 1,
        }

    def test_the_trace_is_the_results(self):
        """No iteration and no direction above it: a netanal measures once."""
        module_netanal = a_netanal()["crs0000_rmod1"]

        assert netanal_trace(module_netanal) is module_netanal["results"]

    def test_a_direction_is_one_direction(self):
        """Both directions is two calls, so a sequence is refused rather than
        having its first element quietly measured."""
        with pytest.raises(TypeError, match="two calls"):
            a_netanal(sweep_direction=["upward", "downward"])

    def test_an_unknown_direction_is_refused(self):
        with pytest.raises(ValueError, match="Invalid sweep_direction"):
            a_netanal(sweep_direction="sideways")

    def test_the_trace_carries_counts_volts_and_what_it_was_probed_at(self):
        trace = netanal_trace(a_netanal()["crs0000_rmod1"])

        assert set(trace) == {
            "frequencies",
            "iq_counts",
            "iq_volts",
            "sweep_amplitude",
            "sweep_direction",
        }
        assert trace["sweep_amplitude"] == 1e-3
        assert trace["sweep_direction"] == "upward"

    def test_no_phase_array(self):
        """Phase is np.angle(iq_counts) at the point of use, as for a sweep."""
        assert "phase_degrees" not in netanal_trace(a_netanal()["crs0000_rmod1"])


# ─── the guards ───────────────────────────────────────────────────────────────


def a_sweep_output():
    """One module's multisweep, for the confusion worth guarding against."""
    section = {
        "channel": 1,
        "frequencies": np.linspace(1.0e9 - 5e4, 1.0e9 + 5e4, 8),
        "iq_counts": np.ones(8, dtype=complex),
        "iq_volts": np.ones(8, dtype=complex) * 1e-7,
        "original_center_frequency": 1.0e9,
        "sweep_direction": "upward",
        "sweep_amplitude": 1e-3,
    }
    return pack_multisweep(
        {0: {"upward": {"R0001": section}}},
        module_id="crs0000_rmod1",
        module=MODULE,
        amp_schedule=AmplitudeSchedule(1e-3),
        directions=("upward",),
        span_hz=1e5,
        npoints_per_sweep=8,
        nsamps=10,
        catalog=ResonatorCatalog.from_frequencies(
            [1.0e9], module=MODULE, amplitude=1e-3
        ),
    )["crs0000_rmod1"]


class TestGuards:
    pytestmark = pytest.mark.portable

    def test_the_sweep_readers_refuse_a_netanal(self):
        """They would otherwise walk 'frequencies' as if it were a resonator
        name and hand back an array dressed as a sweep — silently."""
        module_netanal = a_netanal()["crs0000_rmod1"]

        with pytest.raises(TypeError, match="not a sweep"):
            collect_amplitude_iterations_for(module_netanal, "R0001")

    def test_the_fitter_refuses_a_netanal(self):
        module_netanal = a_netanal()["crs0000_rmod1"]

        with pytest.raises(TypeError, match="not a sweep"):
            fit_sweeps(module_netanal)

    def test_netanal_trace_refuses_a_sweep(self):
        with pytest.raises(TypeError, match="not a netanal"):
            netanal_trace(a_sweep_output())

    def test_netanal_trace_refuses_the_container(self):
        """Named for what it holds, and for the index that gets past it."""
        with pytest.raises(TypeError, match=r"whole netanal.*netanal\['crs"):
            netanal_trace(a_netanal())

    def test_the_sweep_readers_still_read_a_sweep(self):
        """The guard is on 'measurement', so it must not catch what it is for."""
        collected = collect_amplitude_iterations_for(a_sweep_output(), "R0001")

        assert list(collected) == [0]


# ─── the driver, against a simulated board ────────────────────────────────────


@pytest.fixture(scope="module")
def mock_crs():
    """One board for the class below — spinning one up costs ~20 s."""
    from rfmux.mock.helpers import create_mock_crs

    loop = asyncio.new_event_loop()
    crs = loop.run_until_complete(create_mock_crs(
        module=MODULE,
        config={
            "num_resonances": 2,
            "freq_start": 1.05e9,
            "freq_end": 1.45e9,
            "resonator_random_seed": 7,
            "auto_bias_kids": False,
        },
        verbose=False,
    ))
    yield loop, crs
    loop.close()


@pytest.mark.slow_acquisition
class TestTheDriver:
    # Three NCO chunks over the band below, so the seams are in the data.
    FMIN, FMAX, NPOINTS, MAX_SPAN = 1.0e9, 1.6e9, 600, 250e6

    @pytest.fixture(scope="class")
    @classmethod
    def netanal(cls, mock_crs):
        """One sweep for the whole class: it is the same measurement each time."""
        loop, crs = mock_crs
        return crs, loop.run_until_complete(crs.take_netanal(
            amp=1e-3, fmin=cls.FMIN, fmax=cls.FMAX, npoints=cls.NPOINTS,
            nsamps=1, max_chans=128, max_span=cls.MAX_SPAN,
            module=MODULE, save=False,
        ))

    def test_it_comes_back_keyed_by_the_module_it_swept(self, netanal):
        crs, result = netanal

        assert list(result) == [crs.module[MODULE].index()]
        assert result[crs.module[MODULE].index()]["module"] == MODULE

    def test_every_frequency_asked_for_is_measured_exactly_once(self, netanal):
        """The property the NCO stitch used to break: chunks overlapped by one
        frequency, which was measured twice and then dropped."""
        crs, result = netanal
        frequencies = netanal_trace(result[crs.module[MODULE].index()])["frequencies"]

        assert len(frequencies) == self.NPOINTS
        assert len(np.unique(frequencies)) == self.NPOINTS

    def test_the_trace_is_sorted_by_frequency(self, netanal):
        """Not the interleaved order the comb takes them in."""
        crs, result = netanal
        frequencies = netanal_trace(result[crs.module[MODULE].index()])["frequencies"]

        assert np.all(np.diff(frequencies) > 0)

    def test_it_spans_the_band_it_was_asked_for(self, netanal):
        crs, result = netanal
        frequencies = netanal_trace(result[crs.module[MODULE].index()])["frequencies"]

        # Each tone is dithered by tens of Hz off the grid, so the ends land
        # near the requested limits rather than on them.
        assert frequencies[0] == pytest.approx(self.FMIN, abs=1e3)
        assert frequencies[-1] == pytest.approx(self.FMAX, abs=1e3)

    def test_volts_and_counts_describe_the_same_measurement(self, netanal):
        crs, result = netanal
        trace = netanal_trace(result[crs.module[MODULE].index()])

        assert trace["iq_volts"].shape == trace["iq_counts"].shape
        assert np.all(np.isfinite(trace["iq_volts"]))
        # One scale factor for the whole trace, not a per-point correction.
        ratios = trace["iq_volts"] / trace["iq_counts"]
        assert np.allclose(ratios, ratios[0])

    def test_call_params_records_what_it_was_called_with(self, netanal):
        crs, result = netanal
        params = result[crs.module[MODULE].index()]["call_params"]

        assert params["fmin"] == self.FMIN
        assert params["fmax"] == self.FMAX
        assert params["npoints"] == self.NPOINTS
        assert params["max_span"] == self.MAX_SPAN
        assert params["amp"] == 1e-3
        assert params["sweep_direction"] == "upward"
        assert params["module"] == MODULE


@pytest.mark.slow_acquisition
class TestDownward(TestTheDriver):
    """The same band, measured from the top down.

    Inherits every claim above, so a downward netanal has to satisfy them too —
    bar the three that name the direction or the ends of the band, overridden
    below. The chunks are visited in reverse and the tones inside them
    programmed in reverse, so what this pins is that reversing the traversal
    still measures the whole grid, once each, and hands the finder something it
    can read.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def netanal(cls, mock_crs):
        loop, crs = mock_crs
        return crs, loop.run_until_complete(crs.take_netanal(
            amp=1e-3, fmin=cls.FMIN, fmax=cls.FMAX, npoints=cls.NPOINTS,
            nsamps=1, max_chans=128, max_span=cls.MAX_SPAN,
            sweep_direction="downward", module=MODULE, save=False,
        ))

    def test_the_trace_is_sorted_by_frequency(self, netanal):
        """Descending: the order it was measured in, as for a downward sweep."""
        crs, result = netanal
        trace = netanal_trace(result[crs.module[MODULE].index()])

        assert np.all(np.diff(trace["frequencies"]) < 0)
        assert trace["sweep_direction"] == "downward"

    def test_it_spans_the_band_it_was_asked_for(self, netanal):
        crs, result = netanal
        frequencies = netanal_trace(result[crs.module[MODULE].index()])["frequencies"]

        # Highest first, and dithered by tens of Hz off the grid at each end.
        assert frequencies[0] == pytest.approx(self.FMAX, abs=1e3)
        assert frequencies[-1] == pytest.approx(self.FMIN, abs=1e3)

    def test_call_params_records_what_it_was_called_with(self, netanal):
        crs, result = netanal
        params = result[crs.module[MODULE].index()]["call_params"]

        assert params["sweep_direction"] == "downward"

    def test_the_resonance_finder_reads_it(self, netanal):
        """A descending trace is flipped on the way into the finder, which wants
        ascending frequencies, so the search carries the grid it really searched
        and candidate.index indexes that."""
        crs, result = netanal
        # A copy: the finder writes its search into the netanal, and this
        # fixture is shared with the tests above.
        module_netanal = copy.deepcopy(result[crs.module[MODULE].index()])
        frequencies = netanal_trace(module_netanal)["frequencies"]

        search = find_resonances_in_netanal(module_netanal, save=False)

        assert np.array_equal(search.frequencies_hz, frequencies[::-1])
        assert np.all(np.diff(search.frequencies_hz) > 0)
