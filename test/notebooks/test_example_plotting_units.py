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

from rfmux.core.transferfunctions import VOLTS_PER_ROC
from rfmux.tuning import collect_amplitude_iterations_for

pytestmark = pytest.mark.portable
DEMOS = Path(__file__).parents[2] / "rfmux/reference-notebooks/Demos"


@pytest.fixture
def plotters(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    modules = {}
    for name in ("multisweep", "bias", "netanal"):
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


def test_bias_magnitude_is_received_power(plotters):
    report = SimpleNamespace(findings=[SimpleNamespace(
        name="R1", iteration=0, amplitude=0.01, frequency_hz=600e6,
        good=True, bifurcated_at=None,
    )])
    plotters.bias.plot_bias_points(report, measurement())
    panel = plt.gcf().axes[0]
    np.testing.assert_allclose(
        panel.lines[0].get_ydata(), [-46.0206, -52.0412, -46.0206], atol=1e-4,
    )
    assert panel.get_ylabel() == "received power [dBm]"


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
