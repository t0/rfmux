"""Demo axes must state the units of the displayed measurement."""

import ast
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
import pytest

from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.core.transferfunctions import VOLTS_PER_ROC
from rfmux.tuning import (
    BiasReport, bifurcated_by_derivative, collect_amplitude_iterations_for,
    normalized_arc_speed, hysteresis_separation,
)
from rfmux.tuning.bias import BiasFinding, BifurcationCheck

pytestmark = pytest.mark.portable
DEMOS = Path(__file__).parents[2] / "rfmux/reference-notebooks/Demos"


@pytest.fixture
def plotters(monkeypatch):
    monkeypatch.syspath_prepend(str(DEMOS))
    monkeypatch.setattr(plt, "show", lambda: None)
    modules = {}
    for name in ("multisweep", "bias", "netanal", "noise"):
        spec = importlib.util.spec_from_file_location(
            name, DEMOS / f"example_plotting_{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    yield SimpleNamespace(**modules)
    plt.close("all")


def measurement(scale: float = 0.0) -> dict:
    # A 50-ohm system with voltage transmission 0.5, 0.25, 0.5.
    full_scale_peak_volts = np.sqrt(2 * 50 * 1e-3 * 10 ** (scale / 10))
    results = {}
    for step, amplitude in enumerate((0.01, 0.1)):
        volts = full_scale_peak_volts * amplitude * np.array([0.5, 0.25, 0.5])
        results[step] = {"upward": {"R1": {
            "frequencies": np.array([599.99e6, 600e6, 600.01e6]),
            "original_center_frequency": 600e6,
            "sweep_amplitude": amplitude,
            "iq_counts": volts.astype(complex) / VOLTS_PER_ROC,
            "iq_volts": volts.astype(complex),
        }}}
    return {"results": results, "dac_scale_dbm": scale}


def measurement_with_catalog(
    *, frequency_hz: float = 600.003e6, amplitude: float = 0.01,
) -> dict:
    block = measurement()
    catalog = ResonatorCatalog([
        Resonator(
            "R1", 1,
            BiasPoint(
                frequency_hz, amplitude, bias_frequency_quantized=False,
            ),
        ),
    ], module=1)
    block["call_params"] = {"catalog": catalog.to_dict()}
    return block


@pytest.mark.parametrize("scale", [0.0, -12.0])
def test_multisweep_transmission_uses_drive_power(plotters, scale):
    plotters.multisweep.plot_magnitude_panels(measurement(scale))
    panel = plt.gcf().axes[0]
    for line in panel.lines:
        np.testing.assert_allclose(line.get_ydata(), [-6.0206, -12.0412, -6.0206], atol=1e-4)
    assert panel.get_ylabel() == "|S21| [dB, drive-referenced]"


def test_multisweep_received_power_without_normalization(plotters):
    plotters.multisweep.plot_magnitude_panels(measurement(), normalize=False)
    panel = plt.gcf().axes[0]
    for line, drive_dbm in zip(panel.lines, [-40, -20]):
        np.testing.assert_allclose(
            line.get_ydata(), np.array([-6.0206, -12.0412, -6.0206]) + drive_dbm,
            atol=1e-4,
        )
    assert panel.get_ylabel() == "received power [dBm]"


def test_multisweep_missing_scale_requires_received_power_view(plotters):
    block = measurement()
    block["dac_scale_dbm"] = None
    with pytest.raises(ValueError, match="normalize=False"):
        plotters.multisweep.plot_magnitude_panels(block)
    plotters.multisweep.plot_magnitude_panels(block, normalize=False)
    assert plt.gcf().axes[0].get_ylabel() == "received power [dBm]"


def test_multisweep_zero_drive_cannot_be_a_reference(plotters):
    block = measurement()
    block["results"][0]["upward"]["R1"]["sweep_amplitude"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        plotters.multisweep.plot_magnitude_panels(block)


def test_multisweep_magnitude_marks_catalog_bias_point_and_amplitude(plotters):
    block = measurement_with_catalog(amplitude=0.01 * (1 + 5e-7))
    plotters.multisweep.plot_magnitude_panels(block)
    panel = plt.gcf().axes[0]

    sweep_lines = [line for line in panel.lines if len(line.get_xdata()) == 3]
    bias_lines = [line for line in panel.lines if len(line.get_xdata()) == 2]
    assert [line.get_linewidth() for line in sweep_lines] == [3.5, 1.5]
    assert [line.get_zorder() for line in sweep_lines] == [2, 1]
    assert len(bias_lines) == 1
    np.testing.assert_allclose(bias_lines[0].get_xdata(), [3, 3])
    assert bias_lines[0].get_linestyle() == "--"


def test_multisweep_bias_amplitude_requires_a_close_match(plotters):
    plotters.multisweep.plot_magnitude_panels(
        measurement_with_catalog(amplitude=0.01 * (1 + 2e-6)),
    )
    panel = plt.gcf().axes[0]
    sweep_lines = [line for line in panel.lines if len(line.get_xdata()) == 3]
    assert [line.get_linewidth() for line in sweep_lines] == [1.5, 1.5]


def test_multisweep_bias_overlay_can_be_disabled(plotters):
    plotters.multisweep.plot_magnitude_panels(
        measurement_with_catalog(), overlay_bias=False,
    )
    panel = plt.gcf().axes[0]
    assert len(panel.lines) == 2
    assert [line.get_linewidth() for line in panel.lines] == [1.5, 1.5]


def biased_measurement() -> dict:
    block = measurement()
    finding = BiasFinding(
        name="R1", iteration=0, amplitude=0.01, frequency_hz=600e6,
        dI_df=0.0, dQ_df=0.0, bifurcated_at=0.1,
        checks={
            0: BifurcationCheck("hysteresis", False, {"separation": 0.2}, 0.5),
            1: BifurcationCheck("hysteresis", True, {"separation": 0.8}, 0.5),
        },
    )
    block["bias_report"] = BiasReport(
        catalog=ResonatorCatalog([], module=1), findings=[finding],
        settings={"direction": "upward", "amplitude_method": "hysteresis"},
    ).to_dict()
    return block


def test_bias_magnitude_is_received_power(plotters):
    plotters.bias.plot_bias_points(biased_measurement())
    panel = plt.gcf().axes[0]
    np.testing.assert_allclose(
        panel.lines[0].get_ydata(), [-46.0206, -52.0412, -46.0206], atol=1e-4,
    )
    np.testing.assert_allclose(panel.lines[1].get_xdata(), [0, 0])
    assert panel.get_ylabel() == "received power [dBm]"


@pytest.mark.parametrize("noise_gate_factor", [0.0, 50.0])
def test_bifurcation_plot_matches_periscope_quantity(plotters, noise_gate_factor):
    block = biased_measurement()
    block["bias_report"]["settings"].update(
        spike_prominence_factor=0.3, noise_gate_factor=noise_gate_factor,
    )
    frequencies = 600e6 + np.arange(8) * 1000
    iq = np.array([0, 1, 3, 4, 4.4, 4.7, 8, 9]) + 1j * np.array(
        [0, 0.4, 1, 2, 2.4, 2.6, 3, 4],
    )
    # Three steps, including one beyond the report's recorded checks.
    for step in range(3):
        block["results"][step] = {}
        for direction in ("upward", "downward"):
            order = slice(None) if direction == "upward" else slice(None, None, -1)
            block["results"][step][direction] = {"R1": {
                "frequencies": frequencies[order], "iq_counts": iq[order] * (step + 1),
                "original_center_frequency": 600e6,
                "sweep_amplitude": 0.01 * (step + 1),
            }}
    plotters.bias.plot_bifurcation_checks(block)
    panel = plt.gcf().axes[0]
    traces = [line for line in panel.lines if len(line.get_xdata()) == 6]
    assert len(traces) == 6
    for line, (step, direction) in zip(traces, (
        (step, direction) for step in range(3) for direction in ("upward", "downward")
    )):
        entry = block["results"][step][direction]["R1"]
        midpoints, speed = normalized_arc_speed(entry)
        threshold = bifurcated_by_derivative(
            {direction: entry}, spike_prominence_factor=0.3,
            noise_gate_factor=noise_gate_factor,
        ).threshold
        np.testing.assert_allclose(line.get_xdata(),
                                   (0.5 * (midpoints[:-1] + midpoints[1:]) - 600e6) / 1e3)
        np.testing.assert_allclose(line.get_ydata(), np.diff(speed) / threshold)
        assert line.get_linestyle() == ("-" if direction == "upward" else "--")
    assert traces[0].get_linewidth() > traces[2].get_linewidth()
    bars = [line.get_ydata()[0] for line in panel.lines if len(line.get_xdata()) == 2]
    assert 1 in bars and -1 in bars
    assert panel.get_ylabel() == "IQ-speed change / threshold"


def test_bifurcation_plot_skips_unusable_traces(plotters):
    # Flat Q and fewer than four samples cannot provide derivative thresholds.
    plotters.bias.plot_bifurcation_checks(biased_measurement())
    panel = plt.gcf().axes[0]
    assert any(text.get_text() == "no usable derivative traces" for text in panel.texts)
    np.testing.assert_allclose([line.get_ydata()[0] for line in panel.lines], [1, -1])


@pytest.mark.parametrize("function", ["plot_bias_points", "plot_bifurcation_checks", "plot_hysteresis_checks"])
def test_bias_plot_requires_embedded_report(plotters, function):
    with pytest.raises(ValueError, match="run find_bias_points"):
        getattr(plotters.bias, function)(measurement())


@pytest.mark.parametrize("function", ["plot_bias_points", "plot_bifurcation_checks", "plot_hysteresis_checks"])
def test_bias_plot_rejects_separate_report(plotters, function):
    block = biased_measurement()
    report = BiasReport.from_dict(block["bias_report"])
    with pytest.raises(TypeError, match="positional argument"):
        getattr(plotters.bias, function)(report, block)


def test_notebook_bias_plot_reads_embedded_report(plotters):
    namespace = {"np": np, "plt": plt, "BiasReport": BiasReport, "LogNorm": LogNorm,
                 "AMPLITUDE_CMAP": plotters.bias.AMPLITUDE_CMAP,
                 "collect_amplitude_iterations_for": collect_amplitude_iterations_for}
    source = (DEMOS / "bias_finding.md").read_text()
    for cell in re.findall(r"```python\n(.*?)```", source, re.S):
        if not any(f"def {name}(" in cell for name in (
            "amplitude_colours", "plot_bias_points_on_sweeps",
        )):
            continue
        tree = ast.parse(cell)
        tree.body = [node for node in tree.body if isinstance(
            node, (ast.FunctionDef, ast.Import, ast.ImportFrom),
        )]
        exec(compile(tree, str(DEMOS / "bias_finding.md"), "exec"), namespace)
    namespace["plot_bias_points_on_sweeps"](biased_measurement())
    panel = plt.gcf().axes[0]
    np.testing.assert_allclose(panel.lines[2].get_ydata(), [-52.0412], atol=1e-4)


def test_normalized_iq_retains_count_units(plotters):
    plotters.multisweep.plot_iq_panels(measurement())
    panel = plt.gcf().axes[0]
    assert panel.get_xlabel() == "I [counts / DAC amplitude]"
    assert panel.get_ylabel() == "Q [counts / DAC amplitude]"


def netanal_measurement(scale: float = 0.0, step: int = 0) -> dict:
    block = measurement(scale)
    block["measurement"] = "netanal"
    block["results"] = block["results"][step]["upward"]["R1"]
    return block


def test_netanal_transmission_uses_each_modules_drive_power(plotters):
    modules = {
        "crs0001_rmod1": netanal_measurement(),
        "crs0001_rmod2": netanal_measurement(-12.0, 1),
    }
    plotters.netanal.plot_netanal(modules)
    for number in plt.get_fignums():
        panel = plt.figure(number).axes[0]
        np.testing.assert_allclose(
            panel.lines[0].get_ydata(), [-6.0206, -12.0412, -6.0206], atol=1e-4,
        )
        assert panel.get_ylabel() == "|S21| [dB, drive-referenced]"


def test_netanal_received_power_needs_no_dac_scale(plotters):
    block = netanal_measurement()
    block["dac_scale_dbm"] = None
    plotters.netanal.plot_netanal(block, phase=False, normalize=False)
    panel = plt.gcf().axes[0]
    np.testing.assert_allclose(
        panel.lines[0].get_ydata(), [-46.0206, -52.0412, -46.0206], atol=1e-4,
    )
    assert panel.get_ylabel() == "received power [dBm]"


def test_netanal_missing_scale_requires_received_power_view(plotters):
    block = netanal_measurement()
    block["dac_scale_dbm"] = None
    with pytest.raises(ValueError, match="normalize=False"):
        plotters.netanal.plot_netanal(block)


def test_netanal_zero_drive_cannot_be_a_reference(plotters):
    block = netanal_measurement()
    block["results"]["sweep_amplitude"] = 0
    with pytest.raises(ValueError, match="positive"):
        plotters.netanal.plot_netanal(block)


def test_netanal_explicit_count_reference_is_preserved(plotters):
    block = netanal_measurement()
    reference = abs(block["results"]["iq_counts"][0])
    plotters.netanal.plot_netanal(block, reference=reference)
    panel = plt.gcf().axes[0]
    np.testing.assert_allclose(
        panel.lines[0].get_ydata(), [0, -6.0206, 0], atol=1e-4,
    )
    assert panel.get_ylabel() == "|S21| [dB, reference-normalized]"


def test_netanal_phase_is_unchanged_by_power_normalization(plotters):
    block = netanal_measurement()
    block["results"]["iq_counts"] *= np.exp(1j * np.deg2rad([30, -60, 90]))
    plotters.netanal.plot_netanal(block)
    np.testing.assert_allclose(plt.gcf().axes[1].lines[0].get_ydata(), [30, -60, 90])


def test_netanal_notebook_transmission_matches_module(plotters):
    block = netanal_measurement(-12.0, 1)
    trace = block["results"]
    namespace = {
        "np": np, "plt": plt, "module_netanal_outputs": block,
        "netanal_measured": trace, "netanal_iq_counts": trace["iq_counts"],
        "netanal_frequencies": trace["frequencies"],
    }
    path = DEMOS / "network_analysis_find_resonances.md"
    cell = next(cell for cell in re.findall(r"```python\n(.*?)```", path.read_text(), re.S)
                if "magnitude_panel.plot(" in cell)
    exec(compile(cell, str(path), "exec"), namespace)
    panel = plt.gcf().axes[0]
    np.testing.assert_allclose(
        panel.lines[0].get_ydata(), [-6.0206, -12.0412, -6.0206], atol=1e-4,
    )
    assert panel.get_ylabel() == "|S21| [dB, drive-referenced]"


@pytest.mark.parametrize("notebook,function,argument", [
    ("multisweep", "plot_amplitude_iterations", "R1"),
    ("multisweep", "plot_sections_at_iteration", 0),
    ("fitting_resonators", "plot_amplitude_iterations", "R1"),
    ("fitting_resonators", "plot_sections_at_iteration", 0),
    ("bias_finding", "plot_amplitude_steps", ["R1"]),
])
def test_notebook_transmission_matches_module(plotters, notebook, function, argument):
    # Run the shipped plotting definitions without acquiring a new measurement.
    namespace = {
        "np": np, "plt": plt, "LinearSegmentedColormap": LinearSegmentedColormap,
        "LogNorm": LogNorm,
        "collect_amplitude_iterations_for": collect_amplitude_iterations_for,
    }
    source = (DEMOS / f"{notebook}.md").read_text()
    for match in re.finditer(r"```python\n(.*?)```", source, re.S):
        cell = match[1]
        if not any(f"def {name}(" in cell for name in (function, "amplitude_colours")):
            continue
        tree = ast.parse(cell)
        ast.increment_lineno(tree, source[:match.start(1)].count("\n"))
        tree.body = [node for node in tree.body if (
            isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom))
            or isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "AMPLITUDE_CMAP"
                for target in node.targets
            )
        )]
        exec(compile(tree, str(DEMOS / f"{notebook}.md"), "exec"), namespace)
    namespace[function](measurement(), argument)
    panel = plt.gcf().axes[0]
    for line in panel.lines:
        if len(line.get_ydata()):
            np.testing.assert_allclose(
                line.get_ydata(), [-6.0206, -12.0412, -6.0206], atol=1e-4,
            )
    assert panel.get_ylabel() == "|S21| [dB, drive-referenced]"


@pytest.mark.parametrize("compare", ["magnitude", "iq"])
@pytest.mark.parametrize("limit", [0.1, 0.0])
def test_hysteresis_plot_uses_saved_settings(plotters, compare, limit):
    block = biased_measurement()
    block["bias_report"]["settings"].update(compare=compare, max_discrepancy=limit)
    for pair in block["results"].values():
        up = pair["upward"]["R1"]
        pair["downward"] = {"R1": dict(up, iq_counts=up["iq_counts"] * 0.8)}
    plotters.bias.plot_hysteresis_checks(block)
    panel = plt.gcf().axes[0]
    assert len(panel.lines) == 3
    for line, pair in zip(panel.lines, block["results"].values()):
        entries = {direction: sections["R1"] for direction, sections in pair.items()}
        frequencies, separation = hysteresis_separation(entries, compare=compare)
        np.testing.assert_allclose(line.get_xdata(), (frequencies - 600e6) / 1e3)
        np.testing.assert_allclose(line.get_ydata(), separation / (limit or 1))
    np.testing.assert_allclose(panel.lines[-1].get_ydata(), [1, 1] if limit else [0, 0])
    assert panel.lines[0].get_linewidth() > panel.lines[1].get_linewidth()


def test_hysteresis_plot_explains_missing_pairs(plotters):
    plotters.bias.plot_hysteresis_checks(biased_measurement())
    panel = plt.gcf().axes[0]
    assert any(text.get_text() == "no usable up/down pairs" for text in panel.texts)
    assert len(panel.lines) == 1  # the threshold remains visible


def noise_measurement() -> dict:
    sweep = dict(measurement()["results"][0]["upward"]["R1"], sweep_direction="upward")
    catalog = ResonatorCatalog([
        Resonator("R1", 7, BiasPoint(600e6, .01, bias_sweep={
            key: sweep[key] for key in BiasPoint.BIAS_SWEEP_KEYS})),
    ], module=1)
    axes = dict(freq_iq=np.arange(5.), freq_dsb=np.arange(-2., 3.))
    data = dict(iq_counts=np.array([2+3j, 3+4j, 4+5j]),
                psd_i=np.arange(5.) - 100, psd_q=np.arange(5.) - 90,
                psd_dual_sideband=np.arange(5.) - 80)
    return dict(measurement="noise", module=1,
                call_params=dict(catalog=catalog.to_dict()),
                results=dict(
                    acquisition=dict(iq_units="adc_counts", spectrum_units="dBm/Hz",
                                     slow_sample_rate_hz=2.),
                    slow=axes,
                    resonators={"R1": dict(channel=7, amplitude=.01,
                                           tone_frequency_hz=600e6, slow=data,
                                           pfb=dict(data, **axes, time_s=np.arange(3.) / 100))}))


@pytest.mark.parametrize("units", ["volts", "counts"])
@pytest.mark.parametrize("explicit", [True, False])
def test_noise_iq_overlay_uses_same_units_and_measured_tone(plotters, units, explicit):
    block = noise_measurement()
    sweeps = dict(measurement(), measurement="multisweep", module=1) if explicit else None
    figures = plotters.noise.plot_iq_panels(block, units=units, sweeps=sweeps)
    panel = figures[0].axes[0]
    factor = VOLTS_PER_ROC if units == "volts" else 1.
    sweep = measurement()["results"][0]["upward"]["R1"]
    np.testing.assert_allclose(panel.lines[0].get_xdata(), sweep["iq_counts"].real * factor)
    np.testing.assert_allclose(panel.collections[0].get_offsets(),
                               np.array([[2, 3], [3, 4], [4, 5]]) * factor)
    np.testing.assert_allclose(panel.lines[-1].get_xdata(), sweep["iq_counts"][1].real * factor)
    assert panel.get_xlabel() == f"I [{units}]"


def test_noise_iq_refuses_nearest_sweep_at_wrong_drive(plotters):
    block = noise_measurement()
    block["results"]["resonators"]["R1"]["amplitude"] = .03
    sweeps = dict(measurement(), measurement="multisweep", module=1)
    with pytest.raises(ValueError, match="amplitudes must match"):
        plotters.noise.plot_iq_panels(block, sweeps=sweeps)


@pytest.mark.parametrize("stream", ["slow", "pfb"])
@pytest.mark.parametrize("demean", [False, True])
@pytest.mark.parametrize("units", ["volts", "counts"])
def test_noise_timestream_units_and_spacing(plotters, stream, demean, units):
    block = noise_measurement()
    figures = plotters.noise.plot_timestreams(block, stream=stream, demean=demean, units=units)
    panel = figures[0].axes[0]
    factor = VOLTS_PER_ROC if units == "volts" else 1.
    expected = np.array([2., 3., 4.])
    if demean:
        expected -= expected.mean()
    np.testing.assert_allclose(panel.lines[0].get_ydata(), expected * factor, atol=1e-12 * factor)
    np.testing.assert_allclose(panel.lines[0].get_xdata(),
                               np.arange(3.) / (2 if stream == "slow" else 100))
    assert units in panel.get_ylabel()
    np.testing.assert_array_equal(block["results"]["resonators"]["R1"][stream]["iq_counts"],
                                  [2+3j, 3+4j, 4+5j])


@pytest.mark.parametrize("stream", ["slow", "pfb"])
@pytest.mark.parametrize("dual_sideband", [False, True])
@pytest.mark.parametrize("units", ["dBm/Hz", "dBc/Hz"])
def test_noise_psd_preserves_signed_offsets_units_and_saved_bins(plotters, stream, dual_sideband, units):
    block = noise_measurement()
    block["results"]["acquisition"]["spectrum_units"] = units
    figures = plotters.noise.plot_psds(block, stream=stream, dual_sideband=dual_sideband)
    panel = figures[0].axes[0]
    expected_x = [-2, 2] if dual_sideband else [2, 3, 4]
    expected_y = [-80, -76] if dual_sideband else [-98, -97, -96]
    valid = np.isfinite(panel.lines[0].get_ydata())
    np.testing.assert_array_equal(panel.lines[0].get_xdata()[valid], expected_x)
    np.testing.assert_array_equal(panel.lines[0].get_ydata()[valid], expected_y)
    if dual_sideband:
        assert not valid[2]  # The plotted line must not bridge the omitted carrier.
    assert panel.get_ylabel() == f"PSD [{units}]"
    assert panel.get_xscale() == ("symlog" if dual_sideband else "log")
    assert len(block["results"]["resonators"]["R1"][stream]["psd_i"]) == 5


def test_noise_plots_explain_missing_pfb(plotters):
    block = noise_measurement()
    del block["results"]["resonators"]["R1"]["pfb"]
    with pytest.raises(ValueError, match="no pfb capture"):
        plotters.noise.plot_psds(block, stream="pfb")


def test_noise_iq_explicit_channels_need_supplied_sweep(plotters):
    block = noise_measurement()
    block["call_params"]["catalog"] = None
    with pytest.raises(ValueError, match="supply sweeps="):
        plotters.noise.plot_iq_panels(block)


def test_noise_psd_explains_capture_with_only_carrier_bin(plotters):
    block = noise_measurement()
    block["results"]["slow"]["freq_dsb"] = np.array([0.])
    block["results"]["resonators"]["R1"]["slow"]["psd_dual_sideband"] = np.array([0.])
    figures = plotters.noise.plot_psds(block, dual_sideband=True)
    assert "no bins beyond" in figures[0].axes[0].texts[0].get_text()
