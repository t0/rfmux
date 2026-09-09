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

# Fitting resonators

Fitting estimates resonator parameters from multisweep traces. Run it after a
measurement, or on data loaded from disk. You can refit the same traces with
different settings without taking another sweep.

rfmux provides three independent models:

| Model | Fits | Returns |
|---|---|---|
| `skewed` | `\|S21\|` | `fr`, `Qr`, `Qc`, `Qi` |
| `nonlinear` | Complex `S21`, after removing readout gain | Resonator parameters and nonlinearity `a` |
| `circle` | The IQ loop | Circle centre and radius |

| Task | Module |
|---|---|
| Fit sweeps | `rfmux.tuning.fits` |
| Measure sweeps | `rfmux.algorithms.measurement.multisweep` |
| Define amplitude schedules | `rfmux.tuning.multisweep_amplitudes` |
| Manage resonators | `rfmux.core.resonators` |

We’ll start with a simulated array whose bias points are already set. See
`network_analysis_find_resonances.md` and `multisweep.md` for the preceding steps.

## How to use this document

This is a runnable Jupytext notebook. Select a code cell and press **Shift+Enter**.

- Run cells from top to bottom. Later cells use variables defined earlier.
  Use *Kernel → Restart Kernel and Run All Cells* to start again.
- The markdown file stores no outputs. Run a cell to see its results.
- Feel free to change the sweep and fit settings and rerun the cells to explore them. The shipped
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

import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import rfmux
from rfmux.core.resonators import ResonatorCatalog
from rfmux.tuning import AmplitudeSchedule

MODULE = 1
```

## 1. Start with a tuned array

We’ll use four simulated LEKIDs with a fixed random seed. `auto_bias_kids=True`
places a tone at each resonator’s transmission minimum. Reading those frequencies
back gives us starting points without running a network analysis.

For real hardware, replace the simulation and catalog setup cells with your
board session and a catalog you built or loaded:

    session = rfmux.load_session('!HardwareMap [ !CRS { serial: "0042" } ]')
    crs = session.query(rfmux.CRS).one()
    await crs.resolve()
    catalog = ResonatorCatalog.from_csv(..., module=1)

```python
MOCK_CONFIG = {
    "num_resonances": 4,
    "freq_start": 0.6e9,
    "freq_end": 0.9e9,
    "resonator_random_seed": 42,   # same array every run
    "auto_bias_kids": True,        # the simulator tunes itself, so we can skip ahead
    "bias_amplitude": 0.001,
}

session = rfmux.load_session("""
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
""")
crs = session.query(rfmux.CRS).one()
await crs.resolve()

resonator_count, _ = await crs.generate_resonators(MOCK_CONFIG)

# Where the simulator put its own tones: one channel per resonator, and
# get_frequency reports relative to the NCO.
nco_frequency = await crs.get_nco_frequency(module=MODULE)
bias_frequencies = []
for channel in range(1, resonator_count + 1):
    bias_frequencies.append(
        nco_frequency + await crs.get_frequency(channel=channel, module=MODULE)
    )

print(f"{resonator_count} simulated resonators, biased at:")
for frequency in bias_frequencies:
    print(f"  {frequency/1e6:.4f} MHz")
```

Give two resonators different bias amplitudes. This lets us show how to select
a sweep at each resonator’s operating amplitude in section 6.

```python
catalog = ResonatorCatalog.from_frequencies(
    bias_frequencies,
    module=MODULE,
    amplitude=0.001,
)

# Each resonator gets a short made-up name — BOTA, KOZR — drawn fresh each run,
# so read the ones this notebook follows off the catalog rather than typing them
# in. catalog.names() is in frequency order, lowest first.
first_resonator, second_resonator, third_resonator, fourth_resonator, *_ = (
    catalog.names()
)

catalog[second_resonator].update_bias_point(amplitude=0.001 * 2)
catalog[third_resonator].update_bias_point(amplitude=0.001 / 2)

print(catalog)
for resonator in catalog:
    print(f"  {resonator.name}  ch {resonator.channel}  "
          f"{resonator.bias.frequency_hz/1e6:.4f} MHz  "
          f"amp {resonator.bias.amplitude:.5f}")
```

## 2. Measure sweeps to fit

Sweep four resonators at five amplitudes in both directions: 40 traces in total.
Pass an `AmplitudeSchedule` as `amp`, as described in `multisweep.md`.

This multiplicative schedule uses 0.5, 1, 2, 4, and 8 times each resonator’s bias
amplitude. Step 1 is therefore the bias-amplitude step for every resonator.

Choose a span wide enough to include the dip and baseline on both sides, while
sampling the dip with several points. The nonlinear fitter works best with a span
of about `6 * fr / Qr`. The skewed fitter’s default frequency bound is 37.5% of
the span from the centre, so allow room for the resonance to shift with drive.

We use a 200 kHz span here to cover those shifts. Later, a 40 kHz sweep gives
more detail in the IQ plots.

```python
amplitude_schedule = AmplitudeSchedule.multiplicative(0.5, 8.0, 5)
print(amplitude_schedule)

for step in amplitude_schedule.steps(catalog):
    print(step)

multi_amplitude_ms = await crs.multisweep(
    catalog,
    span_hz=200e3,
    npoints_per_sweep=100,
    nsamps=10,
    amp=amplitude_schedule,
    sweep_direction=("upward", "downward"),
)

# A sweep comes back keyed by module
# fit_sweeps takes one module's output at a time, so we index into it and
# everything below is about this module.
multi_amplitude_results = multi_amplitude_ms[crs.module[MODULE].index()]

print(f"\nmodules:         {list(multi_amplitude_ms)}")
print(f"amplitude steps: {list(multi_amplitude_results['results'])}")
print(f"directions:      {list(multi_amplitude_results['results'][0])}")
print(f"resonators:      {list(multi_amplitude_results['results'][0]['upward'])}")
```

### Inspect the traces

A quick plot helps catch measurement problems before fitting. Each panel below
shows one resonator at step 1, its bias amplitude.

Colour shows amplitude; solid and dashed lines show sweep direction. Divide IQ
by each section’s drive amplitude to compare shapes. The plotters read the
`results[step][direction][name]` dictionaries directly.

```python
# Omit the pale end of gnuplot so traces remain visible on white.
AMPLITUDE_CMAP = LinearSegmentedColormap.from_list(
    "gnuplot_truncated", plt.cm.gnuplot(np.linspace(0.0, 0.9, 256))
)


def amplitude_colours(amplitudes):
    """Map amplitudes to log-scaled colours and a colourbar."""
    lo, hi = min(amplitudes), max(amplitudes)
    if hi > lo:
        norm = LogNorm(vmin=lo, vmax=hi)
        colours = [AMPLITUDE_CMAP(norm(a)) for a in amplitudes]
    else:
        # One amplitude, or several identical ones: nothing to grade.
        norm = LogNorm(vmin=lo * 0.9, vmax=lo * 1.1)
        colours = [AMPLITUDE_CMAP(0.5)] * len(amplitudes)
    return colours, plt.cm.ScalarMappable(norm=norm, cmap=AMPLITUDE_CMAP)


def plot_sections_at_iteration(results, iteration, ncols=4):
    """Plot every direction at one step, with one panel per resonator."""
    by_direction = results["results"][iteration]
    # The same resonators occur in each direction. Use the first direction
    # to get panel names; this also works for downward-only measurements.
    first_direction = next(iter(by_direction))
    sections = by_direction[first_direction]
    amplitudes = [
        section["sweep_amplitude"]
        for direction_sections in by_direction.values()
        for section in direction_sections.values()
    ]
    _, mappable = amplitude_colours(amplitudes)
    styles = {"upward": "-", "downward": "--"}

    nrows = -(-len(sections) // ncols)   # ceiling division, no import needed
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.4 * ncols, 2.5 * nrows),
        constrained_layout=True, squeeze=False,
    )
    panels = axes.ravel()

    for panel, name in zip(panels, sections):
        # Read this resonator's section separately for each direction.
        for direction, direction_sections in by_direction.items():
            section = direction_sections[name]
            amplitude = section["sweep_amplitude"]
            colour = mappable.to_rgba(amplitude)
            offset_khz = (
                section["frequencies"] - section["original_center_frequency"]
            ) / 1e3
            iq = section["iq_counts"] / amplitude
            panel.plot(offset_khz, 20 * np.log10(np.abs(iq)), lw=1.0,
                       color=colour, ls=styles[direction], label=direction)
        panel.set_title(f"{name}\n{sections[name]['sweep_amplitude']:.5f}", fontsize=8)
        panel.tick_params(labelsize=7)

    panels[0].legend(fontsize=7)

    # Axis labels only on the outer edge, and hide any panel left over when the
    # section count does not fill the grid.
    for panel in panels[len(sections):]:
        panel.set_visible(False)
    for panel in axes[-1, :]:
        if panel.get_visible():
            panel.set_xlabel("offset [kHz]", fontsize=8)
    for panel in axes[:, 0]:
        panel.set_ylabel("|S21| / drive [dB]", fontsize=8)

    fig.colorbar(mappable, ax=axes, label="sweep amplitude")
    fig.suptitle(f"all {len(sections)} sweep sections at amplitude step {iteration}")
    plt.show()


# Step 1 uses each resonator’s bias amplitude; colours show those amplitudes.
plot_sections_at_iteration(multi_amplitude_results, 1)
```

Now follow one resonator across all five amplitudes. Section 7 will use fitted
parameters to describe these changes in resonance frequency and shape.

```python
def plot_amplitude_iterations(results, name):
    """Plot every amplitude step and available direction for one resonator."""
    # Keep the step → direction → resonator structure visible as we read it.
    steps = results["results"]
    amplitudes = [
        sections[name]["sweep_amplitude"]
        for by_direction in steps.values()
        for sections in by_direction.values()
    ]
    colours, mappable = amplitude_colours(amplitudes)
    colours = iter(colours)  # One colour per trace, in the same order as above.
    styles = {"upward": "-", "downward": "--"}
    shown_directions = set()

    fig, (ax_mag, ax_iq) = plt.subplots(
        1, 2, figsize=(11, 4), constrained_layout=True
    )
    for step, by_direction in steps.items():
        for direction, sections in by_direction.items():
            section = sections[name]
            colour = next(colours)
            offset_khz = (
                section["frequencies"] - section["original_center_frequency"]
            ) / 1e3
            # Normalize by drive amplitude to compare shapes.
            iq = section["iq_counts"] / section["sweep_amplitude"]

            # Label each direction once, even when it appears at several steps.
            label = direction if direction not in shown_directions else None
            ax_mag.plot(offset_khz, 20 * np.log10(np.abs(iq)), lw=1.0,
                        color=colour, ls=styles[direction], label=label)
            ax_iq.plot(iq.real, iq.imag, lw=1.0,
                       color=colour, ls=styles[direction])
            shown_directions.add(direction)

    ax_mag.set_xlabel("offset [kHz]")
    ax_mag.set_ylabel("|S21| / drive [dB]")
    ax_mag.legend(title="frequency direction", fontsize=8)
    ax_iq.set_xlabel("I / drive")
    ax_iq.set_ylabel("Q / drive")
    ax_iq.set_aspect("equal", "datalim")
    fig.colorbar(mappable, ax=(ax_mag, ax_iq), label="sweep amplitude")
    fig.suptitle(f"{name}, {len(steps)} amplitude steps")
    plt.show()


plot_amplitude_iterations(multi_amplitude_results, first_resonator)
```

Take a second sweep at the bias amplitudes, using 40 kHz and 201 points.
The 200 Hz spacing makes the IQ loop easier to see. It uses the same result
structure, with one amplitude step and one direction.

```python
fine_multisweep = (await crs.multisweep(
    catalog,
    span_hz=40e3,
    npoints_per_sweep=201,
    nsamps=10,
))[crs.module[MODULE].index()]


print(f"{len(fine_multisweep['results'][0]['upward'])} sweeps, "
      f"{40e3 / (201 - 1):.0f} Hz between points")
```

Plot the fine sweep in IQ and magnitude. Compare its IQ loops with the wider
sweep above: the closer frequency spacing traces each loop in more detail.

```python
def plot_ms(sections, keys, title):
    """A set of sweep sections: the IQ loop above, the magnitude below."""
    fig, axes = plt.subplots(2, len(keys), figsize=(3.0 * len(keys), 5.5))
    for column, key in enumerate(keys):
        s = sections[key]
        centre = s["original_center_frequency"]
        offset_khz = (s["frequencies"] - centre) / 1e3

        axes[0, column].plot(s["iq_counts"].real, s["iq_counts"].imag, lw=0.9)
        axes[0, column].set_aspect("equal", "datalim")
        axes[0, column].set_title(f"{key}\n{centre/1e6:.3f} MHz", fontsize=9)

        axes[1, column].plot(offset_khz, 20 * np.log10(np.abs(s["iq_counts"])), lw=0.9)
        axes[1, column].set_xlabel("offset [kHz]", fontsize=8)

    axes[0, 0].set_ylabel("Q")
    axes[1, 0].set_ylabel("|S21| [dB]")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


fine_sections = fine_multisweep['results'][0]['upward']
plot_ms(fine_sections, list(fine_sections),
        f"fine multisweep, {40e3/1e3:.0f} kHz span at the bias amplitudes")
```

Before fitting, inspect one section’s keys and array shapes:

```python
sweep_section = multi_amplitude_results["results"][0]["upward"][first_resonator]

# Read each field directly; print array shapes instead of all samples.
for key, value in sweep_section.items():
    if isinstance(value, np.ndarray):
        print(f"{key:<28} ndarray{value.shape} {value.dtype}")
    elif isinstance(value, dict):
        print(f"{key:<28} dict, keys {list(value)}")
    else:
        print(f"{key:<28} {value!r}")
```

## 3. Fit the data

`fit_sweeps()` takes one module’s output. It fits the selected sections and adds
results under each section’s `fits` key. The return value is a `FitReport`,
which records successes, failures, and settings.

```python
from rfmux.tuning import fit_sweeps

fit_report = fit_sweeps(multi_amplitude_results)

print(fit_report)
```

### Fit options

The default call fits all three models to all 40 traces. Use these keyword
arguments to select less data:

| Argument | Default | Selects |
|---|---|---|
| `models` | `("skewed", "nonlinear", "circle")` | Models to run |
| `names` | `None` | Resonator or section names |
| `iterations` | `None` | Amplitude steps |
| `directions` | `None` | Frequency directions |

Each selection accepts one value or an iterable. `None` selects all.
Section 6 also shows selection by each resonator’s bias amplitude.

| Setting | Default | Model | Meaning |
|---|---|---|---|
| `approx_Qr` | `10000.0` | skewed | Initial Qr estimate |
| `normalize` | `True` | skewed | Divide the trace by its last point; models use these normalized units |
| `fr_limit_hz` | `None` | skewed | Maximum fr offset from sweep centre; default is 37.5% of the span |
| `fit_nonlinearity` | `True` | nonlinear | Fit `a`; otherwise hold it at zero |
| `n_extrema_points` | `5` | nonlinear | Points averaged at each end to estimate gain |
| `max_residual` | `0.1` | nonlinear | Reject fits above this residual |

The circle fit has no tuning settings.

| Argument | Default | Meaning |
|---|---|---|
| `max_workers` | `None` | Thread count; default is `min(4, cpu_count)`. Each sweep is one job |
| `progress_callback` | `None` | Called with `(completed, total)` after each sweep |
| `save` | `None` | Follow autosave configuration; `False` skips saving |
| `label` | `None` | Label for a newly saved file |

The report counts each model fit separately. Its `settings` dictionary records
fit settings and module information. Selection is recorded by the report’s fit
entries. Keep the report if you need to reproduce the fitting settings.

```python
print(f"total fits    {len(fit_report)}")
print(f"fitted        {len(fit_report.fitted)}")
print(f"failed        {len(fit_report.failed)}")
print(f"skewed only   {len(fit_report.for_model('skewed'))}")

print("\nsettings:")
for key, value in fit_report.settings.items():
    print(f"  {key:<18} {value!r}")
```

## 4. Inspect the fitted results

The section now has a `fits` dictionary alongside its measurement data:

```python
fitted_sweep_section = multi_amplitude_results["results"][0]["upward"][first_resonator]

# Read each field directly; print array shapes instead of all samples.
for key, value in fitted_sweep_section.items():
    if isinstance(value, np.ndarray):
        print(f"{key:<28} ndarray{value.shape} {value.dtype}")
    elif isinstance(value, dict):
        print(f"{key:<28} dict, keys {list(value)}")
    else:
        print(f"{key:<28} {value!r}")
```

`fits` is keyed by model name. Each model stores its own parameters and status:

```python
for model, fit in fitted_sweep_section["fits"].items():
    print(f"{model}:")
    for key, value in fit.items():
        if isinstance(value, dict):
            shown = ", ".join(f"{k}={v:.4g}" for k, v in value.items())
            print(f"  {key:<16} {{{shown}}}")
        else:
            print(f"  {key:<16} {value!r}")
    print()
```

The full path from one module’s output to a fit is:

```text
multi_amplitude_results
├── schema_version
├── module
├── call_params                             what the driver was asked for
└── results
    └── 0                                   amplitude step, numbered as measured
        └── "upward"                        sweep direction
            └── "BOTA"                      resonator (or "S0001…" for a bare
                │                            frequency list)
                ├── channel                 ╮
                ├── frequencies             │
                ├── iq_counts               │ what multisweep measured,
                ├── iq_volts                │ untouched by the fitters
                ├── original_center_frequency
                ├── sweep_direction         │
                ├── sweep_amplitude         ╯
                └── fits                    ← added by the fitters
                    ├── "skewed"    → params, errors, failed_because
                    ├── "nonlinear" → params, errors, residual, gain,
                    │                 failed_because
                    └── "circle"    → center, radius, failed_because
```

Fitters update `fits` and leave the measured arrays unchanged. Rerunning one
model replaces that model’s result and keeps the others.
`failed_because=None` means the fit passed; otherwise it explains the failure.

Model curves are computed from stored parameters when needed:

| Function or expression | Result |
|---|---|
| `skewed_model_magnitude(section)` | Skewed model magnitude |
| `nonlinear_model_iq(section)` | Nonlinear model in complex counts |
| `section["iq_counts"] / section["fits"]["nonlinear"]["gain"]` | Gain-corrected IQ |
| `section["iq_counts"] - section["fits"]["circle"]["center"]` | Centred IQ |

## 5. Compare fits with measurements

Use points for measured samples and lines for model curves. To draw a smooth
model, copy the section dictionary and replace its frequency array with a finer
grid. The stored fit parameters stay the same.

With `normalize=True`, compare the skewed model with
`np.abs(iq_counts / iq_counts[-1])`. The plots zoom around fitted `fr`, although
the fit uses the full sweep span. Both measured directions are shown.

```python
from rfmux.tuning import (
    nonlinear_model_iq,
    skewed_model_magnitude,
)


def plot_skewed_fits(results, iteration=0, linewidths=6):
    """Every resonator at one amplitude step, with its skewed fit over it."""
    by_direction = results["results"][iteration]
    first_direction = next(iter(by_direction))
    sections = by_direction[first_direction]
    styles = {"upward": "-", "downward": "--"}
    markers = {"upward": ".", "downward": "x"}

    fig, axes = plt.subplots(
        1, len(sections), figsize=(3.1 * len(sections), 3.2),
        constrained_layout=True, squeeze=False,
    )
    for panel, name in zip(axes[0], sections):
        limits = []
        for direction, direction_sections in by_direction.items():
            sweep_section = direction_sections[name]
            offset_khz = (
                sweep_section["frequencies"] - sweep_section["original_center_frequency"]
            ) / 1e3
            normalized = np.abs(
                sweep_section["iq_counts"] / sweep_section["iq_counts"][-1]
            )

            panel.plot(offset_khz, 20 * np.log10(normalized), lw=0, marker=markers[direction],
                       ms=2.5, color="0.45", label=f"{direction} measured")

            skewed_fit = sweep_section["fits"]["skewed"]
            if skewed_fit["failed_because"] is None:
                params = skewed_fit["params"]
                # Copy the section with a denser frequency axis for model evaluation.
                model_frequencies = np.linspace(
                    sweep_section["frequencies"][0], sweep_section["frequencies"][-1],
                    25 * len(sweep_section["frequencies"]),
                )
                model_section = {**sweep_section, "frequencies": model_frequencies}
                model = skewed_model_magnitude(model_section)
                model_offset_khz = (
                    model_frequencies - sweep_section["original_center_frequency"]
                ) / 1e3
                panel.plot(model_offset_khz, 20 * np.log10(model), lw=1.4,
                           color="crimson", ls=styles[direction], label=f"{direction} fit")
                panel.set_title(
                    f"{name}\nQr {params['Qr']:.3g}   Qi {params['Qi']:.3g}",
                    fontsize=9,
                )
                # Zoom to a few linewidths around the fitted resonance. fr / Qr is
                # the linewidth, and fr itself is not the middle of the sweep.
                centre_khz = (
                    params["fr"] - sweep_section["original_center_frequency"]
                ) / 1e3
                half_width_khz = linewidths * params["fr"] / params["Qr"] / 1e3
                limits.extend([centre_khz - half_width_khz, centre_khz + half_width_khz])
            else:
                panel.set_title(f"{name}\nno fit", fontsize=9)

        if limits:
            panel.set_xlim(min(limits), max(limits))
        panel.set_xlabel("offset [kHz]", fontsize=8)
        panel.tick_params(labelsize=7)

    axes[0, 0].set_ylabel("|S21| / off-resonance [dB]", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"skewed Lorentzian fits, amplitude step {iteration}")
    plt.show()


plot_skewed_fits(multi_amplitude_results)
```

For this simulated array, compare fitted `Qi` with the internal Q values
printed during setup. This is a useful check on the fit.

Next, fit the finer sweep. The nonlinear model uses complex IQ data, so we’ll
inspect it in the IQ plane as well as in magnitude.

```python
fit_sweeps(fine_multisweep)

print(f"{first_resonator} fits: "
      f"{list(fine_multisweep['results'][0]['upward'][first_resonator]['fits'])}")
```

The middle panel shows the data used by the nonlinear fitter. Divide
`iq_counts` by the complex gain stored in `fits["nonlinear"]["gain"]`.

```python
def plot_nonlinear_fit(sections, name=None):
    """One resonator's nonlinear fit: measured and model, in IQ and in magnitude."""
    name = next(iter(sections)) if name is None else name
    sweep_section = sections[name]
    nonlinear_fit = sweep_section["fits"]["nonlinear"]

    if nonlinear_fit["failed_because"] is not None:
        print(f"{name}: {nonlinear_fit['failed_because']}")
        return

    measured = sweep_section["iq_counts"]
    corrected = measured / nonlinear_fit["gain"]  # Remove the fitted readout gain.
    offset_khz = (
        sweep_section["frequencies"] - sweep_section["original_center_frequency"]
    ) / 1e3

    # The model on a finer axis than the measurement: in the IQ plane a coarse
    # one would cut the loop into chords, and it is the loop we are looking at.
    # Copy the section with a denser frequency axis for model evaluation.
    model_frequencies = np.linspace(
        sweep_section["frequencies"][0], sweep_section["frequencies"][-1],
        25 * len(sweep_section["frequencies"]),
    )
    model_section = {**sweep_section, "frequencies": model_frequencies}
    model = nonlinear_model_iq(model_section)
    model_offset_khz = (
        model_frequencies - sweep_section["original_center_frequency"]
    ) / 1e3

    fig, (ax_iq, ax_corrected, ax_mag) = plt.subplots(
        1, 3, figsize=(12, 3.8), constrained_layout=True
    )

    ax_iq.plot(measured.real, measured.imag, lw=0, marker=".", ms=3,
               color="0.45", label="measured")
    ax_iq.plot(model.real, model.imag, lw=1.4, color="teal", label="model")
    ax_iq.set_xlabel("I [counts]")
    ax_iq.set_ylabel("Q [counts]")
    ax_iq.set_aspect("equal", "datalim")
    ax_iq.legend(fontsize=8)
    ax_iq.set_title("IQ plane", fontsize=9)

    ax_corrected.plot(corrected.real, corrected.imag, lw=0, marker=".", ms=3,
                      color="0.45")
    ax_corrected.set_xlabel("I / gain")
    ax_corrected.set_ylabel("Q / gain")
    ax_corrected.set_aspect("equal", "datalim")
    ax_corrected.set_title("what the fitter saw\n(gain divided out)", fontsize=9)

    ax_mag.plot(offset_khz, 20 * np.log10(np.abs(measured)), lw=0, marker=".",
                ms=2.5, color="0.45")
    ax_mag.plot(model_offset_khz, 20 * np.log10(np.abs(model)), lw=1.4,
                color="teal")
    ax_mag.set_xlabel("offset [kHz]")
    ax_mag.set_ylabel("|S21| [dB]")
    ax_mag.set_title("magnitude", fontsize=9)

    params = nonlinear_fit["params"]
    fig.suptitle(
        f"{name} nonlinear fit — fr {params['fr']/1e6:.4f} MHz, "
        f"Qr {params['Qr']:.3g}, a {params['a']:.3f}, "
        f"residual {nonlinear_fit['residual']:.2e}"
    )
    plt.show()


plot_nonlinear_fit(fine_multisweep['results'][0]['upward'])
```

The nonlinearity parameter `a` is zero for a linear resonator; bifurcation is
expected near `a ≈ 0.77`. At low drive, the fitted value should be near zero.

The circle fit stores a centre and radius. Subtract the centre from `iq_counts`
to place the loop around the origin before interpreting phase around the loop.

```python
def plot_circle_fit(sections, name=None):
    """The fitted circle, and the loop it recentres."""
    name = next(iter(sections)) if name is None else name
    sweep_section = sections[name]
    circle_fit = sweep_section["fits"]["circle"]

    if circle_fit["failed_because"] is not None:
        print(f"{name}: {circle_fit['failed_because']}")
        return

    measured = sweep_section["iq_counts"]
    centre, radius = circle_fit["center"], circle_fit["radius"]
    angles = np.linspace(0, 2 * np.pi, 361)

    fig, (ax_measured, ax_centred) = plt.subplots(
        1, 2, figsize=(9, 4.2), constrained_layout=True
    )

    ax_measured.plot(measured.real, measured.imag, lw=0, marker=".", ms=3,
                     color="0.45", label="measured")
    ax_measured.plot(centre.real + radius * np.cos(angles),
                     centre.imag + radius * np.sin(angles),
                     lw=1.2, color="darkorange", label="fitted circle")
    ax_measured.plot(centre.real, centre.imag, marker="+", ms=12,
                     color="darkorange", label="centre")
    ax_measured.set_xlabel("I [counts]")
    ax_measured.set_ylabel("Q [counts]")
    ax_measured.set_aspect("equal", "datalim")
    ax_measured.legend(fontsize=8)
    ax_measured.set_title("as measured", fontsize=9)

    recentred = measured - centre  # Shift the fitted centre to the origin.
    ax_centred.plot(recentred.real, recentred.imag, lw=0, marker=".", ms=3,
                    color="0.45")
    ax_centred.axhline(0, lw=0.6, color="0.8")
    ax_centred.axvline(0, lw=0.6, color="0.8")
    ax_centred.set_xlabel("I − centre")
    ax_centred.set_ylabel("Q − centre")
    ax_centred.set_aspect("equal", "datalim")
    ax_centred.set_title("IQ minus fitted centre", fontsize=9)

    fig.suptitle(f"{name} circle fit — radius {radius:.4g} counts")
    plt.show()


plot_circle_fit(fine_multisweep['results'][0]['upward'])
```

## 6. Select sweeps to fit

By default, `fit_sweeps()` fits all sections. For a larger array, select the
resonators, steps, directions, or models you need.

Copy the results and remove their fits so we can see what each selection adds:

```python
unfitted_results = copy.deepcopy(multi_amplitude_results)
for by_direction in unfitted_results["results"].values():
    for sections in by_direction.values():
        for sweep_section in sections.values():
            sweep_section.pop("fits")


# The copy now has no fits; the original results still have theirs.
print(unfitted_results["results"][0]["upward"][first_resonator].keys())
```

### Select names, steps, and directions

`names`, `iterations`, and `directions` accept one value or an iterable.
`None` selects everything. A string such as `names="BOTA"` selects one name.

```python
one_trace_report = fit_sweeps(
    unfitted_results,
    names=first_resonator,
    iterations=0,
    directions="upward",
)

print(one_trace_report)
# Print the paths of sections that now contain fits.
for step, by_direction in unfitted_results["results"].items():
    for direction, sections in by_direction.items():
        for name, section in sections.items():
            if "fits" in section:
                print(step, direction, name, list(section["fits"]))
```

### Select models

`models` accepts any subset of the three models. The nonlinear fit is the most
expensive: it fits seven parameters to complex data and may try three times.
The skewed fit uses five parameters and magnitude data; the circle fit is a linear solve.

For Q values at moderate or low drive, try the skewed model first. Running another
model adds its results while keeping the existing models’ fits:

```python
fit_sweeps(unfitted_results, names=second_resonator, iterations=0, models=("skewed",))
sweep_section = unfitted_results["results"][0]["upward"][second_resonator]
print(f"after skewed:            {list(sweep_section['fits'])}")

fit_sweeps(unfitted_results, names=second_resonator, iterations=0, models=("circle",))
print(f"after circle:            {list(sweep_section['fits'])}  ← skewed kept")
```

### Fit at each resonator’s bias amplitude

`fit_sweeps_at_bias_amplitude()` selects the nearest measured amplitude for each
resonator. By default, it reads bias amplitudes from the catalog snapshot in
`call_params`. See `bias_finding.md` for choosing operating amplitudes.

```python
from rfmux.tuning import fit_sweeps_at_bias_amplitude

at_bias_report = fit_sweeps_at_bias_amplitude(
    unfitted_results,
    directions="upward",
    models=("skewed",),
)

print(at_bias_report)
print()
for fit in at_bias_report.fits:
    measured_at = unfitted_results["results"][fit.iteration][fit.direction][fit.name]["sweep_amplitude"]
    print(f"{fit.name}  biased at {catalog[fit.name].bias.amplitude:.5f}  "
          f"→ step {fit.iteration}, measured at {measured_at:.5f}")
```

Every resonator selects step 1 here because that step multiplies its own bias
amplitude by 1. The step index is shared, but the amplitudes differ.

A fixed requested amplitude can select a different step for each resonator.
Print their amplitude ranges, then try 0.002:

```python
print(f"{'':<8}" + "".join(f"{s:>10}" for s in unfitted_results["results"]))
for resonator in catalog:
    row = [
        unfitted_results["results"][step]["upward"][resonator.name]["sweep_amplitude"]
        for step in unfitted_results["results"]
    ]
    print(f"{resonator.name:<8}" + "".join(f"{a:>10.5f}" for a in row))

print()
fixed_report = fit_sweeps_at_bias_amplitude(
    unfitted_results,
    amplitude=0.002,
    directions="upward",
    models=("circle",),
)
for fit in fixed_report.fits:
    measured_at = unfitted_results["results"][fit.iteration][fit.direction][fit.name]["sweep_amplitude"]
    print(f"0.00200 for {fit.name}  → step {fit.iteration} "
          f"(actually {measured_at:.5f})")
```

Matching selects the nearest amplitude, even if it is far from the request.
Check the selected section’s `sweep_amplitude` when closeness matters.

## 7. Follow fitted parameters across amplitudes

Fits let us compare how frequency and Q change with drive. First, overlay the
skewed model on each trace for one resonator. Colours show amplitude, markers
show measured data, and line styles distinguish sweep directions.

The measured points are about 2 kHz apart and may miss the dip minimum.
A fitted curve can extend below them; inspect the fit quality before treating
that depth as a reliable estimate.

```python
def plot_fitted_traces(results, name, linewidths=8):
    """One resonator at every amplitude, each trace with its skewed fit over it."""
    # Keep direction with each section so both sweeps can be drawn.
    traces = [
        (direction, sections[name])
        for by_direction in results["results"].values()
        for direction, sections in by_direction.items()
    ]
    amplitudes = [section["sweep_amplitude"] for direction, section in traces]
    colours, mappable = amplitude_colours(amplitudes)
    styles = {"upward": "-", "downward": "--"}
    markers = {"upward": ".", "downward": "x"}

    fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    fitted_centres_khz, widest_khz = [], 0.0

    for (direction, sweep_section), colour in zip(traces, colours):
        offset_khz = (
            sweep_section["frequencies"] - sweep_section["original_center_frequency"]
        ) / 1e3
        normalized = np.abs(
            sweep_section["iq_counts"] / sweep_section["iq_counts"][-1]
        )
        ax.plot(offset_khz, 20 * np.log10(normalized), lw=0, marker=markers[direction], ms=2,
                color=colour, alpha=0.6)

        skewed_fit = sweep_section["fits"]["skewed"]
        if skewed_fit["failed_because"] is None:
            params = skewed_fit["params"]
            # Copy the section with a denser frequency axis for model evaluation.
            model_frequencies = np.linspace(
                sweep_section["frequencies"][0], sweep_section["frequencies"][-1],
                25 * len(sweep_section["frequencies"]),
            )
            model_section = {**sweep_section, "frequencies": model_frequencies}
            model = skewed_model_magnitude(model_section)
            ax.plot((model_frequencies
                     - sweep_section["original_center_frequency"]) / 1e3,
                    20 * np.log10(model), lw=1.3, color=colour, ls=styles[direction])
            fitted_centres_khz.append(
                (params["fr"] - sweep_section["original_center_frequency"]) / 1e3
            )
            widest_khz = max(widest_khz, params["fr"] / params["Qr"] / 1e3)

    # Wide enough to hold every step's resonance, plus a few linewidths of the
    # broadest one. Since the drive pulls fr down as the amplitude increases,
    # this window is not centred on the sweep centre.
    if fitted_centres_khz:
        pad = linewidths * widest_khz
        ax.set_xlim(min(fitted_centres_khz) - pad, max(fitted_centres_khz) + pad)

    ax.set_xlabel("offset [kHz]")
    ax.set_ylabel("|S21| / off-resonance [dB]")
    fig.colorbar(mappable, ax=ax, label="sweep amplitude")
    for direction in dict.fromkeys(direction for direction, section in traces):
        ax.plot([], [], color="0.3", ls=styles[direction],
                marker=markers[direction], label=direction)
    ax.legend(fontsize=8)
    fig.suptitle(f"{name}: points measured, lines fitted")
    plt.show()


plot_fitted_traces(multi_amplitude_results, first_resonator)
```

Read each parameter from the section’s `fits` dictionary. The plot below uses
`np.nan` for failed fits, leaving a gap in the curve. Each resonator has one colour,
with a separate line style for each direction.

```python
def plot_fitted_parameters_vs_amplitude(results, model="skewed"):
    """Plot parameter curves for every resonator and available direction."""
    steps = results["results"]
    # Use all measured directions, preserving their measurement order.
    directions = list(dict.fromkeys(
        direction for by_direction in steps.values() for direction in by_direction
    ))
    names = list(dict.fromkeys(
        name for by_direction in steps.values()
        for sections in by_direction.values() for name in sections
    ))
    styles = {"upward": "-", "downward": "--"}
    panels = [
        ("fr", "fr − fr(lowest drive) [kHz]", 1e-3),
        ("Qr", "Qr", 1.0),
        ("Qc", "Qc", 1.0),
        ("Qi", "Qi", 1.0),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4),
                             constrained_layout=True)
    for panel, (parameter, label, scale) in zip(axes, panels):
        for index, name in enumerate(names):
            for direction in directions:
                amplitudes, values = [], []
                for by_direction in steps.values():
                    # A step may contain just one direction.
                    if direction not in by_direction:
                        continue
                    section = by_direction[direction][name]
                    fit = section["fits"][model]
                    amplitudes.append(section["sweep_amplitude"])
                    # Failed fits leave gaps rather than usable-looking values.
                    values.append(fit["params"][parameter]
                                  if fit["failed_because"] is None else np.nan)
                amplitudes = np.array(amplitudes)
                values = np.array(values, dtype=float)
                if parameter == "fr":
                    # Compare shifts from the lowest measured drive.
                    values = values - values[np.argmin(amplitudes)]
                panel.plot(amplitudes, values * scale, marker="o", ms=4,
                           lw=1.2, color=f"C{index % 10}", ls=styles[direction],
                           label=f"{name} {direction}")
        panel.set_xscale("log")
        panel.set_xlabel("sweep amplitude")
        panel.set_ylabel(label, fontsize=9)
        panel.tick_params(labelsize=8)

    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[3].set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{model} fit parameters against drive amplitude")
    plt.show()


plot_fitted_parameters_vs_amplitude(multi_amplitude_results)
```

Compare the skewed and nonlinear estimates as a cross-check. Agreement on `Qr`
is useful evidence; disagreement is a reason to inspect the traces and fit quality.
The table below compares upward sweeps.

```python
print(f"{'':<8}{'amplitude':>12}{'skewed Qr':>12}{'nonlinear Qr':>14}"
      f"{'a':>8}{'residual':>11}")
for name in (first_resonator, fourth_resonator):
    for by_direction in multi_amplitude_results["results"].values():
        sweep_section = by_direction["upward"][name]
        skewed_fit = sweep_section["fits"]["skewed"]
        nonlinear_fit = sweep_section["fits"]["nonlinear"]
        skewed_qr = skewed_fit["params"]["Qr"] if skewed_fit["params"] else float("nan")
        nonlinear_params = nonlinear_fit["params"] or {}
        print(f"{name:<8}{sweep_section['sweep_amplitude']:>12.5f}"
              f"{skewed_qr:>12.4g}"
              f"{nonlinear_params.get('Qr', float('nan')):>14.4g}"
              f"{nonlinear_params.get('a', float('nan')):>8.3f}"
              f"{nonlinear_fit['residual']:>11.1e}")
    print()
```

## 8. Inspect failed fits

A failed fit records a reason in `failed_because`, both in the section and in the
report, so other traces can still be fitted.

Set a very small `max_residual` to demonstrate rejection of nonlinear fits
whose residual exceeds the threshold:

```python
fussy_results = copy.deepcopy(multi_amplitude_results)

fussy_report = fit_sweeps(
    fussy_results,
    iterations=0,
    directions="upward",
    models=("nonlinear",),
    max_residual=1e-9,
)

print(fussy_report)
```

A fit rejected for a high residual keeps its parameters. You can inspect what
it found and why it was rejected:

```python
rejected_fit = (
    fussy_results["results"][0]["upward"][first_resonator]["fits"]["nonlinear"]
)
print(f"failed_because  {rejected_fit['failed_because']}")
print(f"params          fr {rejected_fit['params']['fr']/1e6:.4f} MHz, "
      f"Qr {rejected_fit['params']['Qr']:.4g}")
print(f"residual        {rejected_fit['residual']:.2e}")
```

A fit that did not converge has `params=None` and a failure explanation.

## 9. Saving and next steps

- **Save fitted data:** `fit_sweeps()` and `fit_sweeps_at_bias_amplitude()` use
  the configured autosave setting. They save the modified sweeps, updating the
  original file if one exists. Pass `save=False` to skip saving, or `label=`
  when creating a new file. The `fits` dictionaries are included.
- **Keep fit settings:** settings belong to the report, not each section.
  Use `fit_report.to_dict()` to retain the report separately from the sweeps.
- **Calibrate frequency shifts:** current df calibration uses IQ derivatives
  measured at the bias point. These examples do not derive it from a fit.
  See `bias_finding.md`.
- **Choose operating amplitudes:** `rfmux.tuning.find_bias_points` uses the
  amplitude sweeps to select bias points and return an updated catalog.
- **Use fitting in a GUI:** `progress_callback(completed, total)` can drive a
  progress bar. The original notebook describes moving Periscope’s inline
  fitting to a separate action on completed sweeps.

