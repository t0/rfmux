---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: rfmux-tuning
    language: python
    name: python3
---

# Finding bias points

Bias finding chooses an operating amplitude and frequency for each resonator,
then estimates calibration at that point. The usual sequence is:

1. Sweep several amplitudes and choose one below bifurcation.
2. Choose a frequency within that sweep.
3. Measure IQ slopes to convert later data into frequency shifts.

The functions below return a new `ResonatorCatalog`. They preserve the input
catalog and measured arrays, and add a `bias_report` to the sweep result.
You can rerun the analysis with different settings and compare the catalogs.

| Task | Module |
|---|---|
| Find bias points | `rfmux.tuning.bias` |
| Measure amplitude sweeps | `rfmux.algorithms.measurement.multisweep` |
| Manage resonators and bias points | `rfmux.core.resonators` |
| Save and load measurements | `rfmux.tuning.store` |

We’ll generate fresh mock data with seed 42, using the same four-resonator setup
as `fitting_resonators.md`. No saved measurement file is needed.
See `multisweep.md` for sweep options and `network_analysis_find_resonances.md`
for building a catalog from a network analysis.

## How to use this document

This is a runnable Jupytext notebook. Select a code cell and press **Shift+Enter**.

- Run cells from top to bottom. Later cells use variables defined earlier.
  Use *Kernel → Restart Kernel and Run All Cells* to start again.
- The markdown file stores no outputs. Run a cell to see its results.
- Feel free to change the sweep and bias-finding settings and rerun the cells to explore them. The shipped
  copy is read-only; use *File → Save Notebook As…* to keep your changes.
- In Periscope's JupyterLab, double-click this file. In another JupyterLab
  session, use *Open With → Notebook*.
- VS Code opens this file as text. With a Jupytext extension, use *Open Paired
  Notebook* (the command name may vary). If pairing fails, check that the
  extension's Python environment has Jupytext installed. You can also run
  `jupytext --sync <this file>.md` in an environment with Jupytext. The paired
  `.ipynb` is a local, gitignored copy; the markdown is kept in version control.

The kernel must use the environment where this checkout of rfmux is installed.
Check the interpreter and package paths:

```python
import sys
import rfmux

print(sys.executable)
print(rfmux.__file__)
```

```python
%matplotlib inline

from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from rfmux.core.resonators import BiasPoint, ResonatorCatalog
from rfmux.core.transferfunctions import BASE_FREQUENCY
from rfmux.tuning import AmplitudeSchedule

MODULE = 1
```

## 1. Generate a mock multisweep

Let’s create four pre-biased simulated resonators and read their tone frequencies
into a catalog. Give two resonators different bias amplitudes, then sweep 0.5,
1, 2, 4, and 8 times each bias amplitude in both frequency directions.

The mock evaluates frequency points independently. It can show changes with drive,
but does not reproduce physical hysteresis. These examples demonstrate the
analysis and its flags; they do not establish safe operating amplitudes for a real array.

The sweep below saves a fresh measurement file in the configured data directory.
For real data, replace this setup with your own session, catalog, and multisweep.

```python
# Use the same array settings as the fitting notebook.
mock_config = {
    "num_resonances": 4,
    "freq_start": 0.6e9,
    "freq_end": 0.9e9,
    "resonator_random_seed": 42,
    "auto_bias_kids": True,
    "bias_amplitude": 0.001,
    "pulse_mode": "none",
    "tls_noise_enabled": False,
    "nqp_noise_std_factor": 0.001

}

session = rfmux.load_session("""
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
""")
crs = session.query(rfmux.CRS).one()
await crs.resolve()
resonator_count, _ = await crs.generate_resonators(mock_config)

# Hardware frequencies are relative to the NCO; the catalog uses absolute Hz.
nco_frequency = await crs.get_nco_frequency(module=MODULE)
bias_frequencies = [
    nco_frequency + await crs.get_frequency(channel=channel, module=MODULE)
    for channel in range(1, resonator_count + 1)
]
catalog = ResonatorCatalog.from_frequencies(
    bias_frequencies, module=MODULE, amplitude=0.001,
)
first_resonator, second_resonator, third_resonator, *_ = catalog.names()
catalog[second_resonator].update_bias_point(amplitude=0.002)
catalog[third_resonator].update_bias_point(amplitude=0.0005)

# The 70 kHz span covers shifts with drive; 101 points give 700 Hz spacing.
multi_amplitude_ms = await crs.multisweep(
    catalog,
    span_hz=70e3,
    npoints_per_sweep=101,
    nsamps=10,
    amp=AmplitudeSchedule.multiplicative(0.5, 8.0, 5),
    sweep_direction=("upward", "downward"),
    save=True,
    label="mock_bias_finding",
)

# Select one module, then inspect the step → direction → resonator structure.
multi_amplitude_module_results = multi_amplitude_ms[crs.module[MODULE].index()]
print(f"schema_version:  {multi_amplitude_module_results['schema_version']}")
print(f"measurement:     {multi_amplitude_module_results['measurement']}")
print(f"module:          {multi_amplitude_module_results['module']}")
print(f"amplitude steps: {list(multi_amplitude_module_results['results'])}")
print(f"directions:      {list(multi_amplitude_module_results['results'][0])}")
print(f"resonators:      {list(multi_amplitude_module_results['results'][0]['upward'])}")
```

Each saved module result has `file_metadata`, including its path and creation
information. Inspect the new file’s metadata here:

```python
for key, value in multi_amplitude_module_results["file_metadata"].items():
    print(f"{key:<18} {value}")
```

The saved path lets analysis update the measurement file without a separate
filename argument.

### Read the catalog snapshot

Multisweep stores its input catalog as a dictionary in `call_params`.
`find_bias_points()` uses that snapshot, so no separate catalog argument is needed.
Sweeps made from a frequency list also store a generated catalog here.

```python
swept_catalog = ResonatorCatalog.from_dict(
    multi_amplitude_module_results["call_params"]["catalog"]
)

print(swept_catalog)
```

### Inspect the sweeps

First, read the amplitudes directly from each step and direction. Then plot the
traces with colour for amplitude, solid lines for upward sweeps, and dashed
lines for downward sweeps. Dividing by the
drive amplitude makes their shapes easier to compare.

```python

for step, by_direction in multi_amplitude_module_results["results"].items():
    for direction, sections in by_direction.items():
        amplitudes = {name: section["sweep_amplitude"]
                      for name, section in sections.items()}
        print(f"step {step}, {direction}: {amplitudes}")


AMPLITUDE_CMAP = LinearSegmentedColormap.from_list(
    "gnuplot_truncated", plt.cm.gnuplot(np.linspace(0.0, 0.9, 256))
)


def amplitude_colours(amplitudes):
    """Map amplitudes to logarithmic colours and a colourbar."""
    low, high = min(amplitudes), max(amplitudes)
    if high > low:
        norm = LogNorm(vmin=low, vmax=high)
        colours = [AMPLITUDE_CMAP(norm(a)) for a in amplitudes]
    else:
        norm = LogNorm(vmin=low * 0.9, vmax=low * 1.1)
        colours = [AMPLITUDE_CMAP(0.5)] * len(amplitudes)
    return colours, plt.cm.ScalarMappable(norm=norm, cmap=AMPLITUDE_CMAP)


def plot_amplitude_steps(results, resonator_names, directions=None):
    """Every amplitude step of each resonator, one panel per resonator."""
    fig, axes = plt.subplots(
        1, len(resonator_names), figsize=(3.1 * len(resonator_names), 3.0),
        constrained_layout=True, squeeze=False,
    )
    panels = axes[0]

    styles = {"upward": "-", "downward": "--"}

    for panel, name in zip(panels, resonator_names):
        iterations = {
            step: {direction: sections[name]
                   for direction, sections in by_direction.items()}
            for step, by_direction in results["results"].items()
        }
        amplitudes = [next(iter(e.values()))["sweep_amplitude"] for e in iterations.values()]
        # Use one scale for the whole array, even with different bias amplitudes.
        all_amplitudes = [section["sweep_amplitude"]
                          for by_direction in results["results"].values()
                          for sections in by_direction.values() for section in sections.values()]
        _, mappable = amplitude_colours(all_amplitudes)
        colours = [mappable.to_rgba(amplitude) for amplitude in amplitudes]

        for (entry, colour) in zip(iterations.values(), colours):
            for direction, sweep in entry.items():
                if directions is not None and direction not in directions:
                    continue
                # Normalize by drive amplitude to compare trace shapes.
                iq = sweep["iq_counts"] / sweep["sweep_amplitude"]
                panel.plot(((sweep["frequencies"] - sweep["original_center_frequency"]) / 1e3), 20 * np.log10(np.abs(iq)),
                           styles[direction],
                        lw=1.0, color=colour)

        panel.set_title(name, fontsize=10)
        panel.set_xlabel("offset from sweep centre [kHz]", fontsize=8)
        panel.tick_params(labelsize=8)

    # One legend entry per direction, shared by every amplitude step.
    for sweep_direction, style in (("upward", "-"), ("downward", "--")):
        if any(sweep_direction in by_direction for by_direction in results["results"].values()):
            panels[0].plot([], [], color="0.3", ls=style, label=sweep_direction)
    panels[0].legend(fontsize=7)

    panels[0].set_ylabel("|S21| / drive [dB]", fontsize=8)
    fig.colorbar(mappable, ax=list(panels), label="drive amplitude")
    plt.show()


resonator_names = list(multi_amplitude_module_results["results"][0]["upward"])
plot_amplitude_steps(multi_amplitude_module_results, resonator_names)
```

## 2. Choose a bias amplitude

A common choice is the highest measured amplitude below the first detected
bifurcation. Higher drive can improve the signal relative to additive readout
noise, but driving beyond bifurcation can make the response unsuitable.

`find_bias_amplitude()` works on one resonator’s sweeps. It checks amplitudes
from low to high and selects the step below the first bifurcated step.

| Argument | Default | Meaning |
|---|---|---|
| `iterations` | required | One resonator’s `{step: {direction: section}}` dictionary |
| `method` | `"both"` | `"derivative"`, `"hysteresis"`, or either test combined |
| `spike_prominence_factor` | `0.5` | Required spike prominence relative to arc-speed range |
| `noise_gate_factor` | `50.0` | Required spike prominence relative to noise |
| `max_discrepancy` | `0.1` | Hysteresis separation threshold |
| `compare` | `"magnitude"` | Compare directions in magnitude or `"iq"` |

Each method uses the settings relevant to its test. Start with one resonator,
selected by catalog order rather than a hard-coded name:

```python
from rfmux.tuning import find_bias_amplitude

resonator_iterations = {
            step: {direction: sections[first_resonator]
                   for direction, sections in by_direction.items()}
            for step, by_direction in multi_amplitude_module_results["results"].items()
        }
amplitude_choice = find_bias_amplitude(resonator_iterations, method="derivative")

print(f"iteration:              {amplitude_choice.iteration}")
print(f"amplitude:              {amplitude_choice.amplitude}")
print(f"bifurcated_at:          {amplitude_choice.bifurcated_at}")
print(f"is_bifurcated_at_bias:  {amplitude_choice.is_bifurcated_at_bias}")
```

Read `iteration` and `amplitude` for the selected step, and `bifurcated_at` for
the first detected bifurcation. `None` means none was detected.

`checks` explains each tested step. The search stops at the first detected
bifurcation, so later steps may not appear.

```python
for iteration, check in amplitude_choice.checks.items():
    # `metric` is a dict: one entry per quantity the method examined, named for
    # what it is. Printing it whole rather than picking an entry out keeps this
    # loop working whichever method produced the checks.
    numbers = "  ".join(
        f"{key}={value:.3e}" if isinstance(value, float) else f"{key}={value}"
        for key, value in check.metric.items()
    )
    print(f"step {iteration}: bifurcated={check.bifurcated!s:5}  {numbers}  "
          f"threshold={check.threshold:.3e}  ({check.method})")



```

Two cases need a closer look:

- If no step bifurcates, the highest amplitude is selected. The measurement has
  not established an upper limit. This can occur with the mock data here.
- If the lowest amplitude already bifurcates, it is selected and
  `is_bifurcated_at_bias=True`. A real measurement would need lower amplitudes.

### Detecting bifurcation by looking for jumps in the derivatives

`normalized_arc_speed()` measures IQ movement per hertz after scaling I and Q
by their respective ranges. It returns `(frequencies, speed)` on the midpoints
between samples, with one fewer point than the sweep.

The detector looks for an adjacent positive and negative spike in the
point-to-point change of that speed. Plot that change for both directions:

```python
from rfmux.tuning import normalized_arc_speed


def plot_derivative_test(results, names):
    """Plot changes in arc speed for each step and direction."""
    fig, axes = plt.subplots(
        1, len(names), figsize=(3.1 * len(names), 3.2),
        constrained_layout=True, squeeze=False, sharey=True,
    )
    panels = axes[0]

    for panel, name in zip(panels, names):
        iterations = {
            step: {direction: sections[name]
                   for direction, sections in by_direction.items()}
            for step, by_direction in results["results"].items()
        }
        all_amplitudes = [section["sweep_amplitude"]
                          for by_direction in results["results"].values()
                          for sections in by_direction.values() for section in sections.values()]
        _, mappable = amplitude_colours(all_amplitudes)
        for entries in iterations.values():
            for direction, sweep in entries.items():
                frequencies, speed = normalized_arc_speed(sweep)
                # A difference belongs between the two samples it uses.
                midpoints = 0.5 * (frequencies[:-1] + frequencies[1:])
                panel.plot((midpoints - sweep["original_center_frequency"]) / 1e3,
                           np.diff(speed), lw=1.0,
                           ls="-" if direction == "upward" else "--",
                           color=mappable.to_rgba(sweep["sweep_amplitude"]))

        panel.set_title(name, fontsize=10)
        panel.set_xlabel("offset from sweep centre [kHz]", fontsize=8)
        # Symmetric log, so the quiet steps are not a flat line beside the loud
        # ones — the spikes here are two orders of magnitude apart.
        panel.set_yscale("symlog", linthresh=1e-5)
        panel.tick_params(labelsize=8)

    # One legend entry per direction, shared by every amplitude step.
    for sweep_direction, style in (("upward", "-"), ("downward", "--")):
        if any(sweep_direction in by_direction for by_direction in results["results"].values()):
            panels[0].plot([], [], color="0.3", ls=style, label=sweep_direction)
    panels[0].legend(fontsize=7)

    panels[0].set_ylabel("point-to-point change in arc speed", fontsize=8)
    fig.colorbar(mappable, ax=list(panels), label="drive amplitude")
    fig.suptitle("What the derivative test looks at", fontsize=11)
    plt.show()


plot_derivative_test(multi_amplitude_module_results, resonator_names)


```

A sharp jump can produce a positive spike followed by a negative one.
`bifurcated_by_derivative()` checks every supplied direction and marks the step
bifurcated if any direction passes its tests.

| Argument | Default | Meaning |
|---|---|---|
| `entries` | required | One resonator at one step: `{direction: section}` |
| `spike_prominence_factor` | `0.5` | Required prominence as a fraction of arc-speed range |
| `noise_gate_factor` | `50.0` | Required prominence in multiples of the noise floor; `0.0` disables it |

Larger factors make the test less sensitive.

```python
from rfmux.tuning import bifurcated_by_derivative

```

#### setting thresholds for what is considered bifurcated

The map below tests a range of prominence factors at every amplitude step.
Black means bifurcation was detected. Blue tint shows where the noise threshold
is higher than the prominence threshold, with a band for each direction.
The vertical line marks the default prominence factor, 0.5.

Start with noise gating disabled:

```python

def plot_bifurcation_verdict_map(results, noise_gate_factor=50.0, title=None):
    """Show the derivative verdict as prominence and noise thresholds change."""
    steps = results["results"]
    first_step = next(iter(steps.values()))
    names = list(next(iter(first_step.values())))
    factors = np.linspace(0.02, 1.0, 80)
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 2.2 * len(names)),
                             constrained_layout=True, squeeze=False)
    for panel, name in zip(axes[:, 0], names):
        verdicts = []
        # Each row is one amplitude step; each column is a prominence factor.
        for row, (step, by_direction) in enumerate(steps.items()):
            entries = {direction: sections[name]
                       for direction, sections in by_direction.items()}
            verdicts.append([
                bifurcated_by_derivative(
                    entries, spike_prominence_factor=factor,
                    noise_gate_factor=noise_gate_factor,
                ).bifurcated
                for factor in factors
            ])
            for band, (direction, entry) in enumerate(entries.items()):
                # With prominence disabled, threshold is the noise gate alone.
                noise_check = bifurcated_by_derivative(
                    {direction: entry}, spike_prominence_factor=0.0,
                    noise_gate_factor=noise_gate_factor,
                )
                _, speed = normalized_arc_speed(entry)
                span = np.ptp(speed)
                crossing = noise_check.threshold / span if span > 0 else 0.0
                if crossing > factors[0]:
                    bottom = row - 0.5 + band / len(entries)
                    panel.fill_betweenx(
                        [bottom, bottom + 1 / len(entries)], factors[0],
                        min(crossing, factors[-1]), color="tab:blue", alpha=0.25,
                        zorder=3,
                    )
        panel.imshow(verdicts, aspect="auto", origin="lower", cmap="binary",
                     vmin=0, vmax=1, interpolation="nearest",
                     extent=(factors[0], factors[-1], -0.5, len(steps) - 0.5))
        panel.axvline(0.5, color="tab:orange", lw=1.5)
        panel.set_yticks(range(len(steps)), labels=list(steps))
        panel.set_ylabel("amplitude step")
        panel.set_title(name)
    axes[-1, 0].set_xlabel("spike prominence factor")
    fig.suptitle(title or "Derivative verdicts")
    plt.show()


plot_bifurcation_verdict_map(
    multi_amplitude_module_results,
    noise_gate_factor=0.0,
    title="Only looking at spike prominence without considering noise",
)
```

A prominence-only test can mistake noise for a jump. Add the default noise gate
and compare the maps on the same data:

```python
plot_bifurcation_verdict_map(
    multi_amplitude_module_results,
    title="With the noise gate activated at its default value",
)
```

Blue regions show where noise gating controls the threshold. A high gate can
suppress real jumps as well as noise. Try a smaller value and inspect which
verdicts change:

```python
plot_bifurcation_verdict_map(
    multi_amplitude_module_results,
    noise_gate_factor=20,
    title="Smaller noise gate",
)
```


### Inspect the threshold on individual traces

```python
def plot_prominence_bar(results, names,
                        spike_prominence_factor=0.5,
                        noise_gate_factor=50.0):
    """The bar each verdict was read off, on the steps that settled it.
    """
    fig, axes = plt.subplots(
        1, len(names), figsize=(3.3 * len(names), 3.6),
        constrained_layout=True, squeeze=False, sharey=True,
    )
    panels = axes[0]

    for panel, name in zip(panels, names):
        iterations = {
            step: {direction: sections[name]
                   for direction, sections in by_direction.items()}
            for step, by_direction in results["results"].items()
        }
        choice = find_bias_amplitude(iterations, method="derivative",
            spike_prominence_factor=spike_prominence_factor,
            noise_gate_factor=noise_gate_factor)

        steps = [(choice.iteration, "0.35", "chosen")]
        bifurcated_at = next(
            (i for i, c in choice.checks.items() if c.bifurcated), None
        )
        if bifurcated_at is not None:
            steps.append((bifurcated_at, "tab:red", "bifurcated"))

        for iteration, colour, role in steps:
            for direction, entry in iterations[iteration].items():
                frequencies, speed = normalized_arc_speed(entry)
                midpoints = 0.5 * (frequencies[:-1] + frequencies[1:])

                check = bifurcated_by_derivative(
                    {direction: entry},
                    spike_prominence_factor=spike_prominence_factor,
                    noise_gate_factor=noise_gate_factor,
                )
                # Read the actual threshold: the larger of the two tests’ bars.
                bar = check.threshold

                up = check.metric["positive_spike_prominence"] / check.threshold
                down = check.metric["negative_spike_prominence"] / check.threshold
                panel.plot(((midpoints - entry["original_center_frequency"]) / 1e3), np.diff(speed),
                           lw=1.0, color=colour,
                           ls="-" if direction == "upward" else "--",
                           label=f"step {iteration} ({role})\n"
                                 f"{direction} only: {up:.2f} up, {down:.2f} down\n"
                                 f"{direction} only: adjacent = "
                                 f"{check.metric['adjacency']}")
                panel.axhline(bar, color=colour, ls="--", lw=1.0)
                panel.axhline(-bar, color=colour, ls="--", lw=1.0)

        panel.set_title(name, fontsize=10)
        panel.set_xlabel("offset from sweep centre [kHz]", fontsize=8)
        panel.set_yscale("symlog", linthresh=1e-5)
        panel.tick_params(labelsize=8)
        panel.legend(fontsize=7)

    panels[0].set_ylabel("point-to-point change in arc speed", fontsize=8)
    fig.suptitle("The prominence bar, and the steps each verdict was read off",
                 fontsize=11)
    plt.show()


plot_prominence_bar(multi_amplitude_module_results, resonator_names)
```

Grey traces show the selected step. Red traces show the first detected
bifurcation, if there is one. Both directions are shown; either can trigger the
step’s verdict. Dashed horizontal lines mark each trace’s threshold.

`BifurcationCheck` records the evidence:

| Field | Meaning |
|---|---|
| `method` | Test used |
| `bifurcated` | Verdict |
| `metric` | Measured quantities |
| `threshold` | For derivatives, the larger of the prominence and noise thresholds |
| `parts` | Component checks when tests or directions are combined |

For derivative checks, `metric` includes `positive_spike_prominence`,
`negative_spike_prominence`, and `adjacency`. Both prominences must reach the
threshold, and the positive spike must be followed within two samples by a
qualifying negative spike.

Use the plots to inspect sensitivity. A detector response alone is not proof of
physical bifurcation, especially with independently evaluated mock traces.

### Hysteresis detection

`bifurcated_by_hysteresis()` compares upward and downward sweeps at one amplitude.
It requires both directions.

| Argument | Default | Meaning |
|---|---|---|
| `entries` | required | `{direction: section}` with both directions |
| `max_discrepancy` | `0.1` | Maximum allowed separation |
| `compare` | `"magnitude"` | Compare magnitude in dip depths, or `"iq"` in loop radii |

The check reports the largest normalized separation in `metric["max_separation"]`.

```python
from rfmux.tuning import bifurcated_by_hysteresis

```

```python
def plot_magnitude_hysteresis(results, names, max_discrepancy=0.1):
    """The two directions' |S21| at the loudest step and their difference.
    """
    fig, axes = plt.subplots(
        2, len(names), figsize=(3.2 * len(names), 5.4),
        constrained_layout=True, squeeze=False, sharex="col",
    )

    for column, name in enumerate(names):
        iterations = {
            step: {direction: sections[name]
                   for direction, sections in by_direction.items()}
            for step, by_direction in results["results"].items()
        }
        amplitude_step = iterations[max(iterations)]
        top, bottom = axes[0][column], axes[1][column]

        # Both directions on one ascending frequency grid — downward sweeps
        # arrive high-to-low, and the difference below needs them side by side.
        traces = {}
        for direction, style in (("upward", "-"), ("downward", "--")):
            entry = amplitude_step[direction]
            order = np.argsort(entry["frequencies"])
            frequencies = entry["frequencies"][order]
            traces[direction] = (frequencies, np.abs(entry["iq_counts"])[order])
            top.plot(((frequencies - entry["original_center_frequency"]) / 1e3), traces[direction][1],
                     style, lw=1.2, label=direction)

        (f_up, up), (f_down, down) = traces["upward"], traces["downward"]
        difference = np.abs(up - np.interp(f_up, f_down, down)) / np.ptp(up)
        bottom.plot(((f_up - amplitude_step["upward"]["original_center_frequency"]) / 1e3), difference,
                    lw=1.2, color="C3")
        bottom.axhline(max_discrepancy, ls=":", color="0.4")

        check = bifurcated_by_hysteresis(amplitude_step, compare="magnitude",
                                         max_discrepancy=max_discrepancy)
        top.set_title(f"{name}\nbifurcated={check.bifurcated}, "
                      f"separation={check.metric['max_separation']:.3f}",
                      fontsize=9)
        bottom.set_yscale("log")
        bottom.set_xlabel("offset from sweep centre [kHz]", fontsize=8)
        for panel in (top, bottom):
            panel.tick_params(labelsize=7)

    axes[0][0].set_ylabel("|S21| [counts]", fontsize=8)
    axes[0][0].legend(fontsize=8)
    axes[1][0].set_ylabel("separation [dip depths]", fontsize=8)
    fig.suptitle("The magnitude comparison, at the loudest drive", fontsize=11)
    plt.show()


plot_magnitude_hysteresis(multi_amplitude_module_results, resonator_names,
                          max_discrepancy=0.1)
```

The lower panels compare separation with the threshold. Mock sweeps do not have
physical sweep-history dependence; any detected difference needs that context.
On a real array, a jump at the same frequency in both directions can also escape
this test while still triggering derivative detection.

### Combine both tests

`bifurcated_by_either()` marks a step bifurcated if either test does.
It takes `{direction: section}` with both directions and accepts the same
`spike_prominence_factor=0.5`, `noise_gate_factor=50.0`,
`max_discrepancy=0.1`, and `compare="magnitude"` settings.
This is the default `method="both"` in the amplitude finder.

## 3. Choose a bias frequency

Once an amplitude is selected, choose a frequency within that sweep.
`find_bias_frequency(entry, method="iq_derivative")` reads one section:
one resonator, one amplitude, and one direction.

| Method | Selects |
|---|---|
| `"iq_derivative"` | Maximum `\|dI/df + j·dQ/df\|`, where IQ changes fastest |
| `"minimum"` | Minimum `\|S21\|`, the bottom of the dip |

Both return a frequency on the measured grid. Here we use the upward sweep.

```python
from rfmux.tuning import find_bias_frequency, iq_arc_speed

chosen_sweep = resonator_iterations[amplitude_choice.iteration]["upward"]
chosen_sweep_centre = chosen_sweep["original_center_frequency"]

for method in ("iq_derivative", "minimum"):
    frequency = find_bias_frequency(chosen_sweep, method=method)
    print(f"{method:<14} {frequency/1e6:.6f} MHz "
          f"({(frequency - chosen_sweep_centre)/1e3:+.2f} kHz from the sweep centre)")
```

Plot `iq_arc_speed()` to see the quantity the derivative method maximizes.
The vertical lines show the two frequency choices:

```python
frequencies, speed = iq_arc_speed(chosen_sweep)

bias_frequency_by_derivative = find_bias_frequency(chosen_sweep)
bias_frequency_by_minimum = find_bias_frequency(chosen_sweep, method="minimum")

fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True,
                         constrained_layout=True)

axes[0].plot(((chosen_sweep["frequencies"] - chosen_sweep["original_center_frequency"]) / 1e3),
             20 * np.log10(np.abs(chosen_sweep["iq_counts"])),
             ".-", lw=1.0, ms=3, color="0.2")
axes[0].set_ylabel("|S21| [dB]")

axes[1].plot((frequencies - chosen_sweep_centre) / 1e3, speed,
             ".-", lw=1.0, ms=3, color="0.2")
axes[1].set_ylabel("|dI/df + j dQ/df|  [counts/Hz]")
axes[1].set_xlabel("offset from sweep centre [kHz]")

for panel in axes:
    panel.axvline((bias_frequency_by_derivative - chosen_sweep_centre) / 1e3,
                  color="tab:red", lw=1.2, label="iq_derivative")
    panel.axvline((bias_frequency_by_minimum - chosen_sweep_centre) / 1e3,
                  color="tab:blue", ls="--", lw=1.2, label="minimum")
    panel.axvline(0.0, color="0.7", lw=1.0, label="sweep centre")

axes[0].legend(fontsize=8)
fig.suptitle(f"{first_resonator} at chosen bias amplitude {amplitude_choice.amplitude}: bias frequency selection",
             fontsize=11)
plt.show()
```

The methods may choose different samples. Compare them across the array,
using each resonator’s selected amplitude and the upward sweep:

```python
def plot_frequency_methods(results, names, direction="upward"):
    """Compare frequency choices on each resonator’s selected sweep."""
    fig, axes = plt.subplots(
        1, len(names), figsize=(3.2 * len(names), 3.4),
        constrained_layout=True, squeeze=False,
    )
    panels = axes[0]

    for panel, name in zip(panels, names):
        iterations = {
            step: {direction: sections[name]
                   for direction, sections in by_direction.items()}
            for step, by_direction in results["results"].items()
        }
        choice = find_bias_amplitude(iterations)
        entry = iterations[choice.iteration][direction]

        frequencies, speed = iq_arc_speed(entry)
        panel.plot(((frequencies - entry["original_center_frequency"]) / 1e3), speed, ".-", lw=1.0, ms=3,
                   color="0.2")

        for method, colour, style in (("iq_derivative", "tab:red", "-"),
                                      ("minimum", "tab:blue", "--")):
            frequency = find_bias_frequency(entry, method=method)
            panel.axvline(((frequency - entry["original_center_frequency"]) / 1e3), color=colour, ls=style,
                          lw=1.2, label=method)
        panel.axvline(0.0, color="0.7", lw=1.0, label="sweep centre")

        panel.set_title(f"{name}\nstep {choice.iteration}, "
                        f"amp {entry['sweep_amplitude']:.4f}", fontsize=9)
        panel.set_xlabel("offset from sweep centre [kHz]", fontsize=8)
        panel.tick_params(labelsize=7)

    panels[0].set_ylabel("|dI/df + j dQ/df|  [counts/Hz]", fontsize=8)
    panels[0].legend(fontsize=7)
    fig.suptitle("Where each method puts the tone, at each resonator's chosen "
                 "amplitude", fontsize=11)
    plt.show()


plot_frequency_methods(multi_amplitude_module_results, resonator_names)
```

Inspect peak width as well as the selected frequency. A broad maximum gives
more room for drift; a narrow peak can lose sensitivity after a small shift.
Agreement between methods does not by itself show how stable a bias point will be.

## 4. Build a calibrated BiasPoint

A `BiasPoint` stores at least a frequency and amplitude.

### Round to the tone grid

Bias frequencies are rounded to multiples of `BASE_FREQUENCY`. The grid avoids
in-band intermodulation products and records the frequency the hardware will use.

```python
bias_point = BiasPoint(
    frequency_hz=bias_frequency_by_derivative,
    amplitude=amplitude_choice.amplitude,
)

print(f"asked for  {bias_frequency_by_derivative:.3f} Hz")
print(f"stored     {bias_point.frequency_hz:.3f} Hz")
print(f"difference "
      f"{bias_point.frequency_hz - bias_frequency_by_derivative:+.3f} Hz "
      f"(aligned to a grid of {BASE_FREQUENCY:.3f} Hz)")
```

### Estimate IQ slopes at the bias frequency

Evaluate calibration at the rounded frequency. `iq_derivatives_at(entry,
frequency_hz)` uses splines through the chosen section’s voltage data to estimate
`dI_df` and `dQ_df`, including between measured points.

```python
from rfmux.tuning import iq_derivatives_at

dI_df, dQ_df = iq_derivatives_at(chosen_sweep, bias_point.frequency_hz)

print(f"dI_df  {dI_df:+.4e} V/Hz")
print(f"dQ_df  {dQ_df:+.4e} V/Hz")

print(bias_point)
```

These slopes describe volts of I and Q per hertz. Draw local tangents to see
how they relate to the measured sweep:

```python
bias_frequency = bias_point.frequency_hz

window = 800.0   # Hz either side, for drawing the tangent
tangent_frequencies = np.linspace(bias_frequency - window,
                                  bias_frequency + window, 2)

chosen_sweep_volts = chosen_sweep["iq_volts"]
i_at_bias = np.interp(bias_frequency, chosen_sweep["frequencies"],
                      chosen_sweep_volts.real)
q_at_bias = np.interp(bias_frequency, chosen_sweep["frequencies"],
                      chosen_sweep_volts.imag)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)

for panel, (label, measured, at_bias, slope) in zip(axes, [
    ("I", chosen_sweep_volts.real, i_at_bias, dI_df),
    ("Q", chosen_sweep_volts.imag, q_at_bias, dQ_df),
]):
    panel.plot(((chosen_sweep["frequencies"] - chosen_sweep["original_center_frequency"]) / 1e3), measured, ".-", lw=1.0, ms=3,
               color="0.2", label="measured")
    panel.plot((tangent_frequencies - chosen_sweep_centre) / 1e3,
               at_bias + slope * (tangent_frequencies - bias_frequency),
               lw=2.0, color="tab:red", label=f"d{label}/df = {slope:+.2e} V/Hz")
    panel.plot((bias_frequency - chosen_sweep_centre) / 1e3, at_bias, "o",
               color="tab:red", ms=6)
    panel.set_xlabel("offset from sweep centre [kHz]", fontsize=8)
    panel.set_ylabel(f"{label} [V]", fontsize=8)
    panel.set_xlim((bias_frequency - chosen_sweep_centre) / 1e3 - 10,
                   (bias_frequency - chosen_sweep_centre) / 1e3 + 10)
    # Set the vertical range from the measured trace, not the tangent.
    margin = 0.1 * np.ptp(measured)
    panel.set_ylim(measured.min() - margin, measured.max() + margin)
    panel.legend(fontsize=8)

fig.suptitle("The slopes used in the calibration", fontsize=11)
plt.show()
```

### Convert to frequency-shift units

`BiasPoint` computes:

    df_calibration = 1 / (dI_df + j·dQ_df)    # Hz/V

This calibration belongs to the chosen frequency and amplitude. Changing either
requires new calibration. A `BiasPoint` is immutable, so use `replace()` to create
a calibrated copy:

```python
bias_point = replace(bias_point, dI_df=dI_df, dQ_df=dQ_df)

print(bias_point)
print(f"\ndf_calibration   {bias_point.df_calibration}")
print(f"|df_calibration| {abs(bias_point.df_calibration)/1e6:.3f} MHz/V")
print(f"so 1 µV along the arrow above is "
      f"{abs(bias_point.df_calibration) * 1e-6:.2f} Hz of resonance movement")
```

## 5. Find all bias points in one call

`find_bias_points()` runs amplitude selection, frequency selection, and calibration
for every resonator in the sweep’s catalog snapshot.

| Argument | Default | Meaning |
|---|---|---|
| `sweeps` | required | One module’s multisweep output |
| `amplitude_method` | `"both"` | Bifurcation test; `"both"` and `"hysteresis"` require both directions |
| `frequency_method` | `"iq_derivative"` | Frequency-selection method |
| `direction` | `None` | Direction for frequency and calibration; prefers upward |
| `spike_prominence_factor` | `0.5` | Derivative prominence threshold |
| `noise_gate_factor` | `50.0` | Derivative noise threshold |
| `max_discrepancy` | `0.1` | Hysteresis threshold |
| `compare` | `"magnitude"` | Hysteresis comparison quantity |
| `max_distance_hz` | `None` | Maximum accepted offset from sweep centre; farther choices are flagged and kept at the centre |
| `save` | `None` | Follow autosave settings; save the sweeps with their new report |
| `label` | `None` | Label when creating a new measurement file |

We use `save=False` while comparing settings. The report still enters the result
in memory. Omit that option to follow autosave settings and update the fresh
measurement file from section 1.

```python
from rfmux.tuning import find_bias_points

bias_report = find_bias_points(multi_amplitude_module_results, save=False)

print(bias_report)
print(bias_report.catalog)

```

The returned `bias_report.catalog` is a new catalog. Its resonators come from
the snapshot in `call_params`. To analyze a subset, sweep a catalog containing
that subset.

```python
print(bias_report.catalog)
```

```python
print(f"{'name':<7}{'amplitude':>22}{'bias frequency':>28}")
for after in bias_report.catalog:
    before = swept_catalog[after.name]  # Match by identity, even if frequency order changes.
    print(f"{after.name:<7}"
          f"{before.bias.amplitude:>10.4f} → {after.bias.amplitude:<9.4f}"
          f"{before.bias.frequency_hz/1e6:>13.6f} → "
          f"{after.bias.frequency_hz/1e6:.6f} MHz"
          f"  ({(after.bias.frequency_hz - before.bias.frequency_hz)/1e3:+.2f} kHz)")
```

The original catalog and measured arrays stay unchanged. You can compare bias
choices from several analysis settings:

```python
print(f"the catalog we started from is still: {swept_catalog[first_resonator].bias}")

```

The latest report is also stored under `bias_report` as a plain dictionary.
Rebuild the report object with `BiasReport.from_dict()`:

```python
from rfmux.tuning import BiasReport

print(BiasReport.from_dict(multi_amplitude_module_results["bias_report"]))

```

Rerunning replaces the report in memory. `save` controls whether the measurement
file is updated too.

### Inspect the report

Each resonator has a `BiasFinding` with its selected step, amplitude, frequency,
calibration, and any reason for flagging the result:

```python
finding = bias_report[first_resonator]

print(f"name           {finding.name}")
print(f"iteration      {finding.iteration}")
print(f"amplitude      {finding.amplitude}")
print(f"bifurcated_at  {finding.bifurcated_at}")
print(f"frequency_hz   {finding.frequency_hz}")
print(f"dI_df, dQ_df   {finding.dI_df:.4e}, {finding.dQ_df:.4e}")
print(f"good           {finding.good}")
print(f"flagged_because {finding.flagged_because}")
print(f"\nchecks         {list(finding.checks)}")
```

```python
print(f"{'name':<7}{'step':>6}{'amplitude':>12}{'bif at':>10}"
      f"{'bias freq [MHz]':>18}{'|df_cal| [MHz/V]':>19}")
for f in bias_report.findings:
    df_calibration = bias_report.catalog[f.name].bias.df_calibration
    # No detected bifurcation has no numeric amplitude to format.
    bifurcated_at = "—" if f.bifurcated_at is None else f"{f.bifurcated_at:.4f}"
    print(f"{f.name:<7}{f.iteration:>6}{f.amplitude:>12.4f}"
          f"{bifurcated_at:>10}{f.frequency_hz/1e6:>18.6f}"
          f"{abs(df_calibration)/1e6:>19.3f}")
```

Review `bias_report.good` and `bias_report.flagged` before applying a catalog.
A flag explains an unresolved limit or fallback, even when a bias point was returned.

```python
print(f"biased:  {len(bias_report)}")
print(f"good:    {len(bias_report.good)}")
print(f"flagged: {len(bias_report.flagged)}")
```

Now compare the hysteresis-only result. With this simulator, a lack of detected
hysteresis is expected and may leave the highest amplitude selected as a fallback:

```python
print(find_bias_points(multi_amplitude_module_results,
                       amplitude_method="hysteresis", save=False))
```

Read `flagged_because` for each flagged finding. If no bifurcation was detected,
the selected amplitude is the highest one measured, not an established limit.
For real arrays, inspect the traces and decide whether another amplitude sweep is needed.

### Plot the selected bias points

Show all steps and directions, with the selected step in bold and the bias
frequency marked on the direction used for calibration. These traces use raw
magnitude so the marker shows the measured operating point.

```python
def plot_bias_points_on_sweeps(results, report, direction="upward"):
    """Plot all sweeps and mark the selected operating points."""
    names = [finding.name for finding in report.findings]
    fig, axes = plt.subplots(
        1, len(names), figsize=(3.2 * len(names), 3.5),
        constrained_layout=True, squeeze=False,
    )
    panels = axes[0]

    for panel, finding in zip(panels, report.findings):
        iterations = {
            step: {direction: sections[finding.name]
                   for direction, sections in by_direction.items()}
            for step, by_direction in results["results"].items()
        }
        all_amplitudes = [section["sweep_amplitude"]
                          for by_direction in results["results"].values()
                          for sections in by_direction.values() for section in sections.values()]
        _, mappable = amplitude_colours(all_amplitudes)
        for iteration, entries in iterations.items():
            for sweep_direction, sweep in entries.items():
                chosen = iteration == finding.iteration
                panel.plot((sweep["frequencies"] - sweep["original_center_frequency"]) / 1e3,
                           20 * np.log10(np.abs(sweep["iq_counts"])),
                           lw=2.0 if chosen else 0.8,
                           ls="-" if sweep_direction == "upward" else "--",
                           alpha=1.0 if chosen else 0.55,
                           color=mappable.to_rgba(sweep["sweep_amplitude"]))

        chosen_sweep = iterations[finding.iteration][direction]
        depth_at_bias = np.interp(
            finding.frequency_hz, chosen_sweep["frequencies"],
            20 * np.log10(np.abs(chosen_sweep["iq_counts"])),
        )
        # Ringed in the flag's colour, so a bias point that is a fallback rather
        # than a finding is visible here and not only in the printed report.
        panel.plot(((finding.frequency_hz - chosen_sweep["original_center_frequency"]) / 1e3), depth_at_bias,
                   "o", ms=9, zorder=4, color="white",
                   mec="tab:red" if finding.good else "darkorange", mew=2.0)

        panel.set_title(
            f"{finding.name}{'' if finding.good else '  (flagged)'}\n"
            f"amp {finding.amplitude:.4f}",
            fontsize=9,
        )
        panel.set_xlabel("offset from sweep centre [kHz]", fontsize=8)
        panel.tick_params(labelsize=7)

    # One legend entry per direction, shared by every amplitude step.
    for sweep_direction, style in (("upward", "-"), ("downward", "--")):
        if any(sweep_direction in by_direction for by_direction in results["results"].values()):
            panels[0].plot([], [], color="0.3", ls=style, label=sweep_direction)
    panels[0].legend(fontsize=7)

    panels[0].set_ylabel("|S21| [dB]", fontsize=8)
    fig.colorbar(mappable, ax=list(panels), label="drive amplitude")
    fig.suptitle("The bias points for each resonator",
                 fontsize=11)
    plt.show()


plot_bias_points_on_sweeps(multi_amplitude_module_results, bias_report)
```

## 6. Apply the bias points

After reviewing the report, apply its catalog with:

    await crs.apply_bias(bias_report.catalog)

This returns nothing. The catalog’s frequencies must fit within one NCO bandwidth.
The call is shown for reference; the cells above only measure and analyze sweeps.

## 7. Next steps

- **IQ rotation:** the bias point has a field for it, but this analysis does not
  measure rotation from a timestream.
- **Fitted resonance frequency:** `fit_sweeps()` produces `fr`, but it is not yet
  a bias-frequency method here. See `fitting_resonators.md`.
- **Threshold selection:** inspect verdict maps and traces from your own array.
  The mock demonstrates the calculations and fallbacks, but cannot establish
  physical hysteresis thresholds or operating limits.

