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

# Multisweep

`crs.multisweep()` measures narrow sweeps around several frequencies in parallel.
Each sweep section uses one hardware channel. After a network analysis locates
resonators, multisweep measures them in more detail.

This notebook starts with known resonance frequencies. You can supply them in
two ways:

| Input | Arguments | Sweep section keys |
|---|---|---|
| A `ResonatorCatalog` | `catalog` | Resonator names, such as `"BOTA"` |
| A frequency list | `center_frequencies=` and `amp=` | Section names, such as `"S0001"` |

A catalog stores each resonator's frequency, probe amplitude, hardware channel,
and module. Both inputs produce the same output structure, shown in section 2.

| Task | Module |
|---|---|
| Run a sweep | `rfmux.algorithms.measurement.multisweep` (`crs.multisweep`) |
| Manage resonators | `rfmux.core.resonators` |
| Find resonances | `rfmux.tuning.find_resonances` |

See `network_analysis_find_resonances.md` to build a catalog from a network
analysis, and `resonator_catalogs.md` for catalog details.

By default, multisweep runs one upward sweep at each resonator's bias amplitude.
Pass an `AmplitudeSchedule` to `amp` to sweep several amplitudes. Pass both
frequency directions to `sweep_direction` to measure each step twice.
Sections 4–7 cover these options.

## How to use this document

This is a runnable Jupytext notebook. Select a code cell and press **Shift+Enter**.

- Run cells from top to bottom. Later cells use variables defined earlier.
  Use *Kernel → Restart Kernel and Run All Cells* to start again.
- The markdown file stores no outputs. Run a cell to see its results.
- Feel free to change the sweep settings and rerun the cells to explore them. The shipped
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

import numpy as np
import matplotlib.pyplot as plt

import rfmux
from rfmux.core.resonators import ResonatorCatalog

MODULE = 1

# The band the simulated array lives in.
FMIN, FMAX = 0.6e9, 1.0e9


```

## 1. Start with a simulated board and catalog

Let’s start with ten simulated, pre-tuned MKIDs. A fixed random seed reproduces the same
array. We read the existing bias frequencies to build a `ResonatorCatalog`.
This starts at the same stage as the end of `network_analysis_find_resonances.md`.

For real hardware, replace the next cell with your board session and catalog:

    session = rfmux.load_session('!HardwareMap [ !CRS { serial: "0042" } ]')
    crs = session.query(rfmux.CRS).one()
    await crs.resolve()
    catalog = ResonatorCatalog.from_csv(...)  # or from_frequencies(...)

Multisweep overwrites the frequency and amplitude of each channel it uses,
then silences those channels when it finishes. Other channels stay unchanged.
To silence the whole module first, use `await crs.clear_channels(module=MODULE)`.

```python
from rfmux.mock.helpers import create_mock_crs

MOCK_CONFIG = {
    "num_resonances": 10,
    "freq_start": FMIN,
    "freq_end": FMAX,
    "resonator_random_seed": 42,  # same array every run
    "auto_bias_kids": True,       # the simulator tunes itself, so we can skip ahead
    "bias_amplitude": 0.001,
}

crs = await create_mock_crs(module=MODULE, config=MOCK_CONFIG, verbose=False)

# Where the simulator parked its own tones: one channel per resonator, and
# get_frequency reports relative to the NCO.
nco_frequency = await crs.get_nco_frequency(module=MODULE)
bias_frequencies = [
    nco_frequency + await crs.get_frequency(channel=channel, module=MODULE)
    for channel in range(1, MOCK_CONFIG["num_resonances"] + 1)
]

# Sort by frequency and assign channels 1..N with the given bias amplitude.
# Names are generated each run. Use catalog.names() to get a list of the names. The names
# themselves do not encode order. See resonator_catalogs.md for naming options.
catalog = ResonatorCatalog.from_frequencies(
    bias_frequencies,
    module=MODULE,
    amplitude=0.001,
)

print(catalog)

# Select three resonators by catalog order.
first_resonator, second_resonator, third_resonator = catalog.names()[:3]
print(f"\nlooking at {first_resonator}, {second_resonator} and {third_resonator}")
```

## 2. Run a multisweep using a resonator catalog

The catalog supplies each resonator's:

- Sweep centre: `bias.frequency_hz`
- Probe amplitude: `bias.amplitude`
- Hardware channel: `channel`
- Module

Specify the sweep span and resolution in the call. Multisweep returns a dictionary
and does not modify the catalog. `call_params` records the input catalog.

```python
ms = await crs.multisweep(
    catalog,
    span_hz=75e3,
    npoints_per_sweep=101,
    nsamps=10,
)

print(f"keyed by module: {list(ms)}")
```

### What comes back

The outer dictionary is keyed by module identifier, `crs.module[MODULE].index()`.
This call has one entry, since we only swept one module. A call across four modules would have four.
Each entry contains that module's data and measurement settings.

```python
module_sweeps = ms[crs.module[MODULE].index()]

print(f"module output  {list(module_sweeps)}")
print(f"module         {module_sweeps['module']}")
print(f"span_hz        {module_sweeps['call_params']['span_hz']}")
print(f"amplitude steps {list(module_sweeps['results'])}")
print(f"directions     {list(module_sweeps['results'][0])}")
```

`results` is keyed by amplitude step, frequency direction, then section name.
This sweep has one amplitude step, numbered `0`:

    ms[module_index]["results"][step][direction][name]

The same structure holds for multiple amplitudes and directions.

```python
sweep_sections = module_sweeps["results"][0]["upward"]

print(f"{len(sweep_sections)} sweep sections, keyed by resonator name: "
      f"{list(sweep_sections)[:4]} …")
```

Each section contains measurement arrays and information about the sweep.
Print the keys, array shapes, and other values for one resonator:

```python
entry = sweep_sections[first_resonator]
for key, value in entry.items():
    if isinstance(value, np.ndarray):
        print(f"{key:<30} ndarray{value.shape} {value.dtype}")
    else:
        print(f"{key:<30} {value!r}")
```

Plot the first four sections in the IQ plane and as magnitude versus frequency:

```python
def plot_ms(sections, keys, title):
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

plot_ms(sweep_sections, list(sweep_sections)[:4], "example multisweep")
```

### Override the amplitude

Pass a number as `amp` to use one amplitude for all resonators. Pass a
`{name: amplitude}` mapping to set each separately; it must include every resonator.

The catalog stays unchanged. Each result section records the amplitude used
in `sweep_amplitude`.

```python
ms_louder = await crs.multisweep(
    catalog,
    span_hz=75e3,
    npoints_per_sweep=101,
    nsamps=10,
    amp=0.001 * 2,
)

print(f"catalog bias amplitude   {catalog[first_resonator].bias.amplitude}")
print(f"swept at (default)       {sweep_sections[first_resonator]['sweep_amplitude']}")
print(f"swept at (override)      {ms_louder[crs.module[MODULE].index()]['results'][0]['upward'][first_resonator]['sweep_amplitude']}")
print(f"catalog after the sweep  {catalog[first_resonator].bias.amplitude}  ← unchanged")

# call_params stores the requested amplitude as the schedule base.
# Each section records the amplitude used in sweep_amplitude.
print(f"\ncall_params amp (default)   "
      f"{ms[crs.module[MODULE].index()]['call_params']['amp_schedule']['base']}")
print(f"call_params amp (override)  "
      f"{ms_louder[crs.module[MODULE].index()]['call_params']['amp_schedule']['base']}")
```

Or, using a per-resonator amplitude mapping:

```python
per_resonator_amplitude_mapping = {r.name: r.bias.amplitude for r in catalog}
per_resonator_amplitude_mapping[first_resonator] = 0.001 * 4
per_resonator_amplitude_mapping[second_resonator] = 0.001 / 2

mixed_amplitude_ms = await crs.multisweep(
    catalog,
    span_hz=75e3,
    npoints_per_sweep=101,
    nsamps=10,
    amp=per_resonator_amplitude_mapping,
)

mixed_amplitude_sections = mixed_amplitude_ms[crs.module[MODULE].index()]["results"][0]["upward"]

for name in list(mixed_amplitude_sections)[:4]:
    print(f"{name}  swept at {mixed_amplitude_sections[name]['sweep_amplitude']:.5f}")
```

With a catalog, a positional amplitude list is rejected. Use resonator names
to associate amplitudes with resonators explicitly.

```python
try:
    await crs.multisweep(
        catalog,
        span_hz=75e3,
        npoints_per_sweep=101,
        amp=[0.001] * len(catalog),
    )
except TypeError as e:
    print(f"TypeError: {e}")
```

## 3. No catalog? Multisweep using a plain list of frequencies instead

You can supply frequencies without a catalog:

- `amp` is required: a number, a list, or a `{section_name: amplitude}` mapping.
- `module` is required.
- By default, the sweep sections will be named `S0001`, `S0002`, etc., in input order. Pass `names` to
  use your own names in the same order as the frequencies.

```python
section_center_frequencies = [1.005e9, 1.015e9, 1.025e9]   

no_catalog_ms = await crs.multisweep(
    center_frequencies=section_center_frequencies,
    amp=0.001,
    span_hz=75e3,
    npoints_per_sweep=101,
    nsamps=10,
    module=MODULE,
)

no_catalog_sections = no_catalog_ms[crs.module[MODULE].index()]["results"][0]["upward"]

print(f"keys: {list(no_catalog_sections)}")
for section_name, s in no_catalog_sections.items():
    print(f"{section_name}  ch {s['channel']}  "
          f"{s['original_center_frequency']/1e6:.3f} MHz  "
          f"amp {s['sweep_amplitude']}")
```

These frequencies are off-resonance in the simulated array, so the traces are flat.

```python
plot_ms(no_catalog_sections, list(no_catalog_sections),
        "multisweep done using a plain frequency list")
```

### Pass a list of amplitudes

Supply one amplitude per frequency, in the same order as `center_frequencies`.

```python
per_section_amplitude_ms = await crs.multisweep(
    center_frequencies=section_center_frequencies,
    amp=[0.001, 0.001 * 2, 0.001 * 4],
    span_hz=75e3,
    npoints_per_sweep=101,
    nsamps=10,
    module=MODULE,
)

for section_name, s in per_section_amplitude_ms[crs.module[MODULE].index()]["results"][0]["upward"].items():
    print(f"{section_name}  {s['original_center_frequency']/1e6:.3f} MHz  "
          f"amp {s['sweep_amplitude']:.5f}")
```

### Name the sections

Pass `names` in the same order as `center_frequencies`.

```python
section_names = ["below_band", "in_band", "above_band"]

named_section_ms = await crs.multisweep(
    center_frequencies=section_center_frequencies,
    names=section_names,
    amp={"below_band": 0.001, "in_band": 0.001 * 2,
         "above_band": 0.001},
    span_hz=75e3,
    npoints_per_sweep=101,
    nsamps=10,
    module=MODULE,
)

for section_name, s in named_section_ms[crs.module[MODULE].index()]["results"][0]["upward"].items():
    print(f"{section_name:<12} ch {s['channel']}  "
          f"{s['original_center_frequency']/1e6:.3f} MHz  "
          f"amp {s['sweep_amplitude']:.5f}")
```

<!-- #region -->
## 4. Iterate over amplitudes

Pass an `AmplitudeSchedule` as the `amp` argument to iteratively multisweep over several amplitues a single call.


| Task | Module |
|---|---|
| Run sweeps | `rfmux.algorithms.measurement.multisweep` |
| Define amplitude schedules | `rfmux.tuning.multisweep_amplitudes` |
| Read results | `rfmux.tuning.sweep_results` |

The `AmplitudeSchedule` is a helper class that orchestrates the amplitude iteration.
A schedule defines a base amplitude and a sequence of amplitude steps for each
resonator. The base usually comes from `bias.amplitude` in the catalog.
Steps are numbered from 0 in measurement order. Each step can be swept in
one or both frequency directions.

- `AmplitudeSchedule()`: one step at each resonator's bias amplitude.
- `AmplitudeSchedule(0.005)`: one step at 0.005 for every resonator.
- `AmplitudeSchedule.ramp(0.001, 0.005, 3)`: three steps from 0.001 to 0.005,
  shared by all resonators.
- `AmplitudeSchedule.multiplicative(0.5, 2.0, 3)`: three steps from 0.5 to 2
  times each resonator's base amplitude.
- `AmplitudeSchedule.explicit([1e-4, 3e-4, 2e-3])`: use these amplitudes in order.

The first two options give the same results as a single sweep without a schedule.
`ramp` and `multiplicative` use logarithmic spacing by default. Pass
`spacing="linear"` for linear spacing.
<!-- #endregion -->

```python
from rfmux.tuning import AmplitudeSchedule

amplitude_schedule = AmplitudeSchedule.multiplicative(0.5, 4.0, 4)
print(amplitude_schedule)
print(amplitude_schedule.steps)  # configured multipliers

for step in amplitude_schedule.resolve_steps(catalog):
    print(step)
```

### Absolute and relative steps

Relative steps multiply each resonator's base amplitude. Absolute steps use
the specified amplitudes for all resonators.

Change two catalog amplitudes to show the difference:

```python
catalog[second_resonator].update_bias_point(amplitude=0.001 * 4)
catalog[third_resonator].update_bias_point(amplitude=0.001 / 2)

for r in list(catalog)[:4]:
    print(f"{r.name}  bias amplitude {r.bias.amplitude:.5f}")

relative = AmplitudeSchedule.multiplicative(1.0, 2.0, 2)
absolute = AmplitudeSchedule.ramp(1e-3, 2e-3, 2)

for label, schedule in [("multiplicative (relative)", relative), ("ramp (absolute)", absolute)]:
    print(f"\n{label}:  {schedule}")
    for step in schedule.resolve_steps(catalog):
        shown = {n: f"{a:.5f}" for n, a in list(step.amplitudes.items())[:4]}
        print(f"  step {step.step}  {shown}")
```

## 5. Check a schedule

A quick look at the schedule can save a long measurement. `describe()` reports
sweep counts and amplitude ranges. `validate()` returns `(severity, message)` pairs.

```python
described = amplitude_schedule.describe(catalog, n_directions=2)

for key in ("nsteps", "n_directions", "n_sweeps", "n_sections",
            "amplitude_min", "amplitude_max", "spacing"):
    print(f"{key:<18} {described[key]}")

print("\nper resonator (min, max):")
for name, (lo, hi) in list(described["amplitude_range_by_name"].items())[:4]:
    print(f"  {name}  {lo:.5f} → {hi:.5f}")
```

Amplitudes use normalized DAC units and must be between 0 and 1.
`validate()` reports values above full scale:

```python
for severity, message in amplitude_schedule.validate(catalog, n_directions=2):
    print(f"{severity:>7}: {message}")

print()
too_loud = AmplitudeSchedule.multiplicative(1.0, 500.0, 3)
for severity, message in too_loud.validate(catalog):
    print(f"{severity:>7}: {message}")
```

## 6. Run the amplitude schedule

Pass the schedule as `amp`. The callback below prints progress after each sweep.
The output has the same structure as a single sweep.

```python
def report(record):
    amplitudes = record["amplitudes"]
    print(f"  [{record['completed']}/{record['total']}] "
          f"step {record['step']} {record['direction']:<8} "
          f"{first_resonator} at {amplitudes[first_resonator]:.5f}")

multi_amplitude_ms = await crs.multisweep(
    catalog,
    span_hz=75e3,
    npoints_per_sweep=101,
    nsamps=10,
    amp=amplitude_schedule,
    sweep_callback=report,
)

print(f"\nkeyed by module index: {list(multi_amplitude_ms)}")
```

The module's `results` now contains several amplitude steps:

```python
multi_amplitude_module_results = multi_amplitude_ms[crs.module[MODULE].index()]

print(f"full module output    {list(multi_amplitude_module_results)}")
print(f"amplitude steps       {list(multi_amplitude_module_results['results'])}")
print(f"directions            {list(multi_amplitude_module_results['results'][0])}")
```

Each step contains a dictionary of directions. Each direction contains a
dictionary of sections, keyed by name. Note that this is the same format as when
we only multiswept a single amplitude.

`call_params` records the requested settings, including `amp_schedule`.
A scalar `amp` is stored as a one-step schedule.

```python
first_sweep_iteration_sections = multi_amplitude_ms[crs.module[MODULE].index()]["results"][0]["upward"]
print(f"step 0, upward: {list(first_sweep_iteration_sections)[:4]} …")
print(f"{first_resonator} swept at "
      f"{first_sweep_iteration_sections[first_resonator]['sweep_amplitude']:.5f}")

print(f"\ncall_params: {list(multi_amplitude_module_results['call_params'])}")
print(f"schedule as stored: {multi_amplitude_module_results['call_params']['amp_schedule']}")
```

The amplitude used is stored in each section's `sweep_amplitude` field.

## 7. Read and plot the results

Access the data directly through `module_results["results"][step][direction][name]`.
Let’s follow those keys through a few examples. Each starts with one module’s
output, `multi_amplitude_module_results`.

### Get one resonator across every amplitude

```python
# results maps amplitude steps to their measured directions.
for step, by_direction in multi_amplitude_module_results["results"].items():
    # Each direction maps resonator names to sweep sections.
    for direction, sections in by_direction.items():
        section = sections[first_resonator]
        print(f"step {step}  {direction}  {section['sweep_amplitude']:.5f}")
```

### Plot one resonator across amplitudes

The plot below reads sections directly from `results`. Colours show amplitude
on a logarithmic scale; line styles show direction. Both directions are plotted
when present. Divide IQ by the sweep amplitude to compare trace shapes.

```python
from matplotlib.colors import LinearSegmentedColormap, LogNorm

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


plot_amplitude_iterations(multi_amplitude_module_results, first_resonator)
```

### Get amplitudes at one step

```python
# Select amplitude step 2, then read each direction's sections.
by_direction = multi_amplitude_module_results["results"][2]
for direction, sections in by_direction.items():
    for name, section in list(sections.items())[:4]:
        print(f"{name}  {direction}  {section['sweep_amplitude']:.5f}")
```

Plot all sections at one step, with one panel per resonator.
Multiplicative steps can give each resonator a different amplitude and colour.
Absolute ramp steps give all resonators the same amplitude and colour.
Each panel includes all available directions, using solid and dashed lines.

```python
def plot_sections_at_iteration(results, iteration, ncols=5):
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


plot_sections_at_iteration(multi_amplitude_module_results, 2)
```

### Find the sweep nearest an amplitude

`find_iteration_matching_amplitude()` returns the matching sections as
`{direction: section}`, plus the step number.

```python
from rfmux.tuning import find_iteration_matching_amplitude

for name in (first_resonator, second_resonator, third_resonator):
    bias = catalog[name].bias.amplitude
    at_bias, step = find_iteration_matching_amplitude(
        multi_amplitude_module_results, name, amplitude=bias
    )
    print(f"{name}  bias {bias:.5f}  → step {step}, "
          f"swept at {at_bias['upward']['sweep_amplitude']:.5f}")
```

A fixed amplitude can match a different step for each resonator because their
base amplitudes differ:

```python
print(f"{'':<8}" + "".join(f"{s:>10}" for s in multi_amplitude_module_results["results"]))
for name in (first_resonator, second_resonator, third_resonator):
    amplitudes = [
        # This measurement used upward sweeps. Select that direction and name.
        by_direction["upward"][name]["sweep_amplitude"]
        for by_direction in multi_amplitude_module_results["results"].values()
    ]
    print(f"{name:<8}" + "".join(f"{a:>10.5f}" for a in amplitudes))

print()
for name in (first_resonator, second_resonator, third_resonator):
    matched, step = find_iteration_matching_amplitude(
        multi_amplitude_module_results, name, 0.002
    )
    got = matched["upward"]["sweep_amplitude"]
    print(f"0.00200 for {name}  → step {step}  (actually {got:.5f})")
```

The match is the nearest available amplitude, even if it is far from the request.
Here, only the second resonator reaches 0.016. The others return their highest
available amplitude:

```python
for name in (first_resonator, second_resonator, third_resonator):
    matched, step = find_iteration_matching_amplitude(
        multi_amplitude_module_results, name, 0.016
    )
    got = matched["upward"]["sweep_amplitude"]
    print(f"0.01600 for {name}  → step {step}  (actually {got:.5f})")
```

Check the returned section's `sweep_amplitude` when the match needs to be close.

### Sweep in both directions

Pass one direction as a string, or both as a sequence. At each amplitude step,
multisweep measures the directions in the order supplied before moving to the
next amplitude.

```python
both_ways = await crs.multisweep(
    catalog,
    span_hz=75e3,
    npoints_per_sweep=101,
    nsamps=10,
    amp=AmplitudeSchedule.multiplicative(1.0, 2.0, 2),
    sweep_direction=("upward", "downward"),
)

both_ways_module_results = both_ways[crs.module[MODULE].index()]

for step, by_direction in both_ways_module_results["results"].items():
    for direction, sections in by_direction.items():
        print(f"step {step}  {direction:<9} "
              f"{first_resonator} at "
              f"{sections[first_resonator]['sweep_amplitude']:.5f}")
```

Plot amplitude as colour and direction as line style. The simulated traces
overlap. Real detectors can show different traces in the two directions when
driven into bifurcation.

```python
# The same plotters include both directions automatically.
plot_amplitude_iterations(both_ways_module_results, first_resonator)
plot_sections_at_iteration(both_ways_module_results, 0)
```

The result contains only the requested directions. A downward-only sweep uses
the same dictionary structure:

```python
one_way = await crs.multisweep(
    catalog,
    span_hz=75e3,
    npoints_per_sweep=101,
    nsamps=10,
    amp=AmplitudeSchedule.explicit([0.001]),
    sweep_direction="downward",
)

print(f"directions present: "
      f"{list(one_way[crs.module[MODULE].index()]['results'][0])}")
```

### Sweep a frequency list at several amplitudes

Without a catalog, the schedule must supply its own amplitudes. `ramp` and
`explicit` do this directly. A `multiplicative` schedule needs an explicit `base`.
Use these sweeps to explore probe amplitudes before tuning.

```python
untuned_results = await crs.multisweep(
    center_frequencies=section_center_frequencies,
    module=MODULE,
    span_hz=75e3,
    npoints_per_sweep=101,
    nsamps=10,
    amp=AmplitudeSchedule.ramp(0.001, 0.001 * 4, 3),
)

untuned_module_results = untuned_results[crs.module[MODULE].index()]

for step, by_direction in untuned_module_results["results"].items():
    amplitude = by_direction["upward"]["S0001"]["sweep_amplitude"]
    print(f"step {step}  every section at {amplitude:.5f}")

# Frequency-list results use section names as keys. These off-resonance
# traces show the three amplitudes.
plot_amplitude_iterations(untuned_module_results, "S0001")

try:
    await crs.multisweep(
        center_frequencies=section_center_frequencies,
        module=MODULE,
        span_hz=75e3,
        npoints_per_sweep=101,
        amp=AmplitudeSchedule.multiplicative(0.5, 2.0, 3),   # relative to what?
    )
except ValueError as e:
    print(f"\nValueError: {e}")
```

## 8. Next steps and saving

- **Choose an operating amplitude:** `rfmux.tuning.find_bias_points` finds
  bifurcation in an amplitude sequence and returns a new catalog biased one
  step below it. See `bias_finding.md`.
- **Fit resonators:** `rfmux.tuning.fit_sweeps` stores model results under `fits`
  in each fitted sweep section. See `fitting_resonators.md`. Writing fits back
  to the catalog is still to come.
- **Save data:** multisweep saves results to `~/rfmux_data/ipy_session_<today>/`
  by default. The result records the path under `file_metadata`. Pass
  `save=False` to skip saving, or `label="cooldown3"` to label the file.
  See `rfmux.tuning.store` for output directory settings.

Multisweep silences only the channels it swept. Other tones remain active.

