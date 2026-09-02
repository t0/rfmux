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

`crs.multisweep()` measures a narrow, high-resolution sweep around each of many
frequencies at once — one hardware channel per frequency, all of them swept in
parallel. In a typical array characterization / tuning flow, it is the step after 
a network analysis: netanal finds roughly where the
resonators are, multisweep looks at each one closely enough to characterise it.

This notebook starts with the resonances already located, so that it is about
multisweep rather than about finding them.

There are two ways to
tell multisweep what to sweep, and they are identical once the measurement
starts:

| You have | You pass | Sweep sections keyed by |
|---|---|---|
| Already done a netanal and used `find_resonances` to make a `ResonatorCatalog` | a `ResonatorCatalog` | resonator name (`"R0001"`) |
| A list of frequencies | `center_frequencies=` + `amp=` | section name (`"S0001"`) |

Those names key the sweep sections, which sit a few levels inside what the call
returns — section 2 unpacks the shape, and it is the same shape for every sweep
in this notebook.

Unless you are starting completely from scratch, or looking at hardware that doesn't
have resonances, you will likely pass a catalog. A catalog already knows
each resonator's frequency, its probe amplitude and its hardware channel. 

| Piece | Module |
|---|---|
| The sweep | `rfmux.algorithms.measurement.multisweep` (`crs.multisweep`) |
| The array bookkeeping | `rfmux.core.resonators` |
| Finding resonances first | `rfmux.tuning.find_resonances` |

Seeding a catalog from a network analysis is the subject of
`network_analyses_find_resonances_make_resonator_catalog.md`. If you haven't done
that yet and are unfamiliar with the workflow, start there, then come back here.

One `multisweep` call is always *one* sweep, at one amplitude per resonator, in
one direction. Iterating the same array over several amplitudes is a layer on
top, `crs.multiamp_multisweep()`, and it is the subject of sections 4 to 7.

## How to use this document

**This is a runnable notebook, not a web page.** Every grey block below is a live
code cell: put the cursor in it and press **Shift+Enter** to execute it.

- **Run the cells in order, top to bottom.** Later cells use variables the
  earlier ones defined, so skipping ahead fails with a `NameError`. *Kernel →
  Restart Kernel and Run All Cells* starts clean.
- **The outputs you see are the ones you just produced.** This file is stored as
  jupytext markdown, which keeps no saved outputs, so a cell is blank until you
  run it. Nothing here can show you a stale number from someone else's run.
- **Editing is encouraged.** Change the span, the number of points, the
  amplitudes, and re-run — that is what this document is for. The shipped copy
  is read-only, so *File → Save Notebook As…* to keep your changes.
- **How you open it depends on your editor.** This file is jupytext markdown,
  not `.ipynb`. In the JupyterLab session Periscope launches it opens as a
  notebook on double-click; in a JupyterLab you started yourself, right-click →
  *Open With* → *Notebook*. **In VS Code it opens as plain text**, so pair it
  instead: with a jupytext extension installed, right-click → *Open Paired
  Notebook* (the exact wording varies by extension) creates an `.ipynb` beside
  this file and keeps the two in step — run and edit the notebook, and your
  changes flow back into the markdown. If that command does nothing, the
  extension could not find jupytext: it runs whichever interpreter VS Code
  resolved, which is often the base environment rather than the one rfmux is
  installed in. Install jupytext there, point the extension at the right
  interpreter, or skip the extension and run `jupytext --sync <this file>.md`
  from a shell that has it. The `.ipynb` is a local working copy and is
  gitignored; the markdown is the version that is kept, reviewed and tested.
- **Check which kernel you are running.** rfmux has to be importable from the
  interpreter the notebook uses, and if you have more than one checkout, it must
  be the environment installed against *this* one. This says which copy you
  actually got:

  ```python
  import sys, rfmux; print(sys.executable); print(rfmux.__file__)
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

PROBE_AMPLITUDE = 0.001   # normalized DAC units

# One sweep's worth of settings, shared by every call below so the comparisons
# are like for like.
SPAN_HZ = 100e3           # total width of each individual sweep
NPOINTS_PER_SWEEP = 101   # points measured across that width
NSAMPS = 10               # averages per point
```

## 1. A simulated board, and a catalog

Ten simulated LEKIDs on a fixed random seed, told to tune themselves, with the
tones they parked read back into a `ResonatorCatalog`. That is where
`network_analyses_find_resonances_make_resonator_catalog.md` leaves off, so this
notebook picks up from there and is about multisweep rather than about finding
resonances. (The simulator biases at the S21 dip rather than the nominal
resonance — the two differ by up to ~1 MHz — so reading its tones back gives
roughly the frequencies a real tuning run would have found.)

To run against real hardware, replace this one cell with a session on your board
and a catalog you built or loaded. Everything after it is unchanged:

    session = rfmux.load_session('!HardwareMap [ !CRS { serial: "0042" } ]')
    crs = session.query(rfmux.CRS).one()
    await crs.resolve()
    catalog = ResonatorCatalog.from_csv(...)      # or from_frequencies(...)

Note that multisweep takes over the channels it sweeps — one per resonator,
overwriting their frequency and amplitude, and zeroing them again when it
finishes. Other channels on the module are left exactly as they were, so a tone
you have parked by hand survives the call. The flip side is that multisweep does
not guarantee a quiet module: if something else is live and would intermodulate
with the sweep, clear it first with `await crs.clear_channels(module=MODULE)`.

```python
from rfmux.mock.helpers import create_mock_crs

MOCK_CONFIG = {
    "num_resonances": 10,
    "freq_start": FMIN,
    "freq_end": FMAX,
    "resonator_random_seed": 42,  # same array every run
    "auto_bias_kids": True,       # the simulator tunes itself, so we can skip ahead
    "bias_amplitude": PROBE_AMPLITUDE,
}

crs = await create_mock_crs(module=MODULE, config=MOCK_CONFIG, verbose=False)

# Where the simulator parked its own tones: one channel per resonator, and
# get_frequency reports relative to the NCO.
nco_frequency = await crs.get_nco_frequency(module=MODULE)
bias_frequencies = [
    nco_frequency + await crs.get_frequency(channel=channel, module=MODULE)
    for channel in range(1, MOCK_CONFIG["num_resonances"] + 1)
]

# from_frequencies sorts by frequency, numbers the resonators R0001… in that
# order, assigns channels 1..N, and parks every bias point at PROBE_AMPLITUDE.
catalog = ResonatorCatalog.from_frequencies(
    bias_frequencies,
    module=MODULE,
    amplitude=PROBE_AMPLITUDE,
)

print(catalog)
```

## 2. Do a multisweep using the resonator catalog

The catalog carries everything multisweep needs (the centre frequencies of
each sweep section, the amplitudes to sweep them at, etc), so the call says almost nothing
beyond specifying the sweep bandwidth and resolution:

- each resonator's **sweep centre** is its `bias.frequency_hz`
- each resonator's **probe amplitude** is its `bias.amplitude`
- each resonator's **hardware channel** is its `channel`

The catalog even knows its own module, so `module=` is optional here — pass it
only if you want the mismatch checked.

**Multisweep does not modify the catalog.** 

Everything a multisweep produces comes back in the returned dict.

```python
ms = await crs.multisweep(
    catalog,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
)

print(f"keyed by module: {list(ms)}")
```

### What comes back

The outermost key is a **module identifier** — the board's serial and the readout
module, as `crs.module[MODULE].index()` spells it. One key here, because this
call swept one module; a call that swept four would have four, and the loop you
write over them is the same either way.

Inside is one **envelope** per module, holding the sweep sections and enough
provenance to say what produced them. Pickle it a year from now and it still
knows its module, its span, its point count and the catalog it came from:

```python
module_sweeps = ms[crs.module[MODULE].index()]

print(f"envelope       {list(module_sweeps)}")
print(f"module         {module_sweeps['module']}")
print(f"span_hz        {module_sweeps['call_params']['span_hz']}")
print(f"amplitude steps {list(module_sweeps['results'])}")
print(f"directions     {list(module_sweeps['results'][0])}")
```

`results` is keyed by **amplitude step**, then by **direction**, then by section
name. A single `multisweep` is one step in one direction, so it has exactly one
of each — step `0`, and whichever direction you asked for. That is not a padded
slot: one call really is one amplitude at one direction, and saying so means
`multiamp_multisweep` (sections 4–7) returns the same shape with more steps in
it. Nothing downstream has to ask which of the two produced a result.

```python
sweep_sections = module_sweeps["results"][0]["upward"]

print(f"{len(sweep_sections)} sweep sections, keyed by resonator name: "
      f"{list(sweep_sections)[:4]} …")
```

Reaching for the same four levels every time gets old, so name it once and reuse
it — this is the one below, and every later cell uses it:

```python
def sections_of(sweeps, module=MODULE, step=0, direction="upward"):
    """The {name: entry} sweep sections of one sweep, from what a macro returned.

    Analysis functions take *one module's* envelope, never the whole dict, so
    stepping into the module you mean is deliberate rather than guessed at.
    """
    return sweeps[crs.module[module].index()]["results"][step][direction]
```

Each entry holds the sweep section data itself plus the bookkeeping needed to know what it
is:

```python
entry = sweep_sections["R0001"]
for key, value in entry.items():
    if isinstance(value, np.ndarray):
        print(f"{key:<30} ndarray{value.shape} {value.dtype}")
    else:
        print(f"{key:<30} {value!r}")
```

A look at the first four, in the IQ plane and in magnitude:

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

### Overriding the amplitude

By default, multisweep uses each resonator's bias amplitude found in the catalog.
You can also override these using the `amp` argument. Pass a number to override all
 of them for this one call, or a
`{name: amplitude}` mapping to set them individually. Note that the mapping has to name
every resonator.

The catalog is left alone either way; the amplitude actually used is reported
per resonator as `sweep_amplitude`.

```python
ms_louder = await crs.multisweep(
    catalog,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    amp=PROBE_AMPLITUDE * 2,
)

print(f"catalog bias amplitude   {catalog['R0001'].bias.amplitude}")
print(f"swept at (default)       {sections_of(ms)['R0001']['sweep_amplitude']}")
print(f"swept at (override)      {sections_of(ms_louder)['R0001']['sweep_amplitude']}")
print(f"catalog after the sweep  {catalog['R0001'].bias.amplitude}  ← unchanged")

# call_params records the amp you asked for, verbatim — a number here, None when
# you let the catalog decide. What each resonator was *actually* probed at is
# sweep_amplitude on its own entry, so nothing has to be stored twice.
print(f"\ncall_params['amp'] (default)   "
      f"{ms[crs.module[MODULE].index()]['call_params']['amp']}")
print(f"call_params['amp'] (override)  "
      f"{ms_louder[crs.module[MODULE].index()]['call_params']['amp']}")
```

Or, using a per-resonator amplitude mapping:

```python
per_resonator_amplitude_mapping = {r.name: r.bias.amplitude for r in catalog}
per_resonator_amplitude_mapping["R0001"] = PROBE_AMPLITUDE * 4
per_resonator_amplitude_mapping["R0002"] = PROBE_AMPLITUDE / 2

mixed_amplitude_ms = await crs.multisweep(
    catalog,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    amp=per_resonator_amplitude_mapping,
)

mixed_amplitude_sections = sections_of(mixed_amplitude_ms)

for name in list(mixed_amplitude_sections)[:4]:
    print(f"{name}  swept at {mixed_amplitude_sections[name]['sweep_amplitude']:.5f}")
```

Note that a positional *list* of amplitudes is refused if provided alongside a catalog: it
would silently depend on the catalog ordering, which is not meaningful since the resonators
are in arbitrary order within the catalog.

```python
try:
    await crs.multisweep(
        catalog,
        span_hz=SPAN_HZ,
        npoints_per_sweep=NPOINTS_PER_SWEEP,
        amp=[PROBE_AMPLITUDE] * len(catalog),
    )
except TypeError as e:
    print(f"TypeError: {e}")
```

## 3. No catalog? Multisweep using a plain list of frequencies

No catalog required. For when you have a few frequencies from somewhere and you want to look at them.

Two differences from the catalog version:

- **`amp` is required.** Can be a single value, a list, or a dict mapping `{section_name: amplitude}`.
- **`module` is required.** 
- **The sweep sections are named `S0001…`** — S for section — in the order you passed
  the frequencies. You can also pass `names`, as a list of sweep section names in the same order as
  the frequency list

```python
section_center_frequencies = [1.005e9, 1.015e9, 1.025e9]   

no_catalog_ms = await crs.multisweep(
    center_frequencies=section_center_frequencies,
    amp=PROBE_AMPLITUDE,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    module=MODULE,
)

no_catalog_sections = sections_of(no_catalog_ms)

print(f"keys: {list(no_catalog_sections)}")
for section_name, s in no_catalog_sections.items():
    print(f"{section_name}  ch {s['channel']}  "
          f"{s['original_center_frequency']/1e6:.3f} MHz  "
          f"amp {s['sweep_amplitude']}")
```

(This data is "measured" off-resonance, so the S21's are flat.)

```python
plot_ms(no_catalog_sections, list(no_catalog_sections),
        "multisweep done using a plain frequency list")
```

#### No-catalog operation: passing a list of amplitudes

`amp` may also be a list, one value per frequency, in the same order as
`center_frequencies`. 

```python
per_section_amplitude_ms = await crs.multisweep(
    center_frequencies=section_center_frequencies,
    amp=[PROBE_AMPLITUDE, PROBE_AMPLITUDE * 2, PROBE_AMPLITUDE * 4],
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    module=MODULE,
)

for section_name, s in sections_of(per_section_amplitude_ms).items():
    print(f"{section_name}  {s['original_center_frequency']/1e6:.3f} MHz  "
          f"amp {s['sweep_amplitude']:.5f}")
```

#### No-catalog operation: Naming the sections yourself

`S0001…` is the default. Pass `names` as a list in the same
order as the section center frequencies, if you want to call
them something special.

```python
section_names = ["below_band", "in_band", "above_band"]

named_section_ms = await crs.multisweep(
    center_frequencies=section_center_frequencies,
    names=section_names,
    amp={"below_band": PROBE_AMPLITUDE, "in_band": PROBE_AMPLITUDE * 2,
         "above_band": PROBE_AMPLITUDE},
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    module=MODULE,
)

for section_name, s in sections_of(named_section_ms).items():
    print(f"{section_name:<12} ch {s['channel']}  "
          f"{s['original_center_frequency']/1e6:.3f} MHz  "
          f"amp {s['sweep_amplitude']:.5f}")
```

<!-- #region -->

## 4. Iterating over amplitudes

rfmux provides various ways of iteratively running `multisweep` at different amplitudes. 
Note that every tone's amplitude can be different, so this allows quite a bit of freedom.

The iteration options live external to the core `multisweep` measurement. Caller functions
parse your intended iteration options and iteratively call `multisweep`, then package up 
the outputs into a format that replicates the core `multisweep` outputs, in a (hopefully) 
sensible way.

| Piece | Module |
|---|---|
| The driver that iteratively calls `multisweep`  | `rfmux.algorithms.measurement.multiamp_multisweep` (`crs.multiamp_multisweep`) |
| Coordinating the amplitudes | `rfmux.tuning.multisweep_amplitudes` |
| Reading the results back out | `rfmux.tuning.sweep_results` |

The amplitude iteration options are specified using a `AmplitudeSchedule` object. An amplitude
schedule has two key components for every resonator in the catalog: the **base** amplitude, 
and the **ladder** of amplitude steps that that resonator will be `multiswept` over.
The **base** amplitude is generally the `bias_amplitude` for that resonator, as found in the catalog.
The **steps** in the ladder are then generated based on the type of iteration you want. 
Each step is one amplitude. Steps are numbered from 0 in the order
they are measured, and each one can be swept up to twice — once per frequency direction.

- No iteration (just do one sweep at everyone's bias amplitude): `AmplitudeSchedule()`
- No iteration, but use the same amplitude for every resonator: `AmplitudeSchedule(0.005)`

Note that the above two options have no iteration. You could just call multisweep directly; the outputs
from using the iterative callers will be identical.

Some iterative options:

- Step all the resonators' amplitudes from value A to value B, each of them getting an identical 
amplitude at each step: `AmplitudeSchedule.ramp(0.001, 0.005, 3)` 
- Scale the resonators' amplitudes by a set of multiplicative factors, so they each go from e.g. 
0.5x bias amplitude to 2x bias amplitude, whatever their bias amplitudes may be : `AmplitudeSchedule.multiplicative(0.5, 2, 3)`
- Scale some specified base amplitude by a set of multiplicative factors, so all resonators are swept at the
same amplitudes : `AmplitudeSchedule.multiplicative(0.5, 2, 3, base=0.001)`
- Whatever you want: `AmplitudeSchedule.explicit([1e-4, 3e-4, 2e-3])`


On the iterative options, the spacing between iteration steps is logarithmic by default. Pass
`spacing="linear"` for evenly spaced amplitudes instead.

<!-- #endregion -->

```python
from rfmux.tuning import AmplitudeSchedule

amplitude_schedule = AmplitudeSchedule.multiplicative(0.5, 4.0, 4)
print(amplitude_schedule)

for step in amplitude_schedule.steps(catalog):
    print(step)
```

### absolute vs relative amplitude steps



**Relative** steps multiply each resonator's own amplitude. **Absolute**
steps ignore the catalog's amplitudes entirely and apply the same thing to
all resonators.

```python
catalog["R0002"].set_bias(amplitude=PROBE_AMPLITUDE * 4)
catalog["R0003"].set_bias(amplitude=PROBE_AMPLITUDE / 2)

for r in list(catalog)[:4]:
    print(f"{r.name}  bias amplitude {r.bias.amplitude:.5f}")

relative = AmplitudeSchedule.multiplicative(1.0, 2.0, 2)
absolute = AmplitudeSchedule.ramp(1e-3, 2e-3, 2)

for label, schedule in [("multiplicative (relative)", relative), ("ramp (absolute)", absolute)]:
    print(f"\n{label}:  {schedule}")
    for step in schedule.steps(catalog):
        shown = {n: f"{a:.5f}" for n, a in list(step.amplitudes.items())[:4]}
        print(f"  step {step.step}  {shown}")
```

## 5. Checking a schedule before you spend an hour on it

Iterating over amplitudes is a slow measurement, so it is worth checking what the
algorithm is going to do before starting the actual iteration. `describe()` gives the derived numbers, and `validate()` returns
`(severity, message)` pairs.

```python
described = amplitude_schedule.describe(catalog, n_directions=2)

for key in ("nsteps", "n_directions", "n_sweeps", "n_sections",
            "amplitude_min", "amplitude_max", "spacing"):
    print(f"{key:<18} {described[key]}")

print("\nper resonator (min, max):")
for name, (lo, hi) in list(described["amplitude_range_by_name"].items())[:4]:
    print(f"  {name}  {lo:.5f} → {hi:.5f}")
```

`validate()` checks whether a schedule contains any amplitude values that are 
greater than the DAC full-scale amplitude (normalized units > 1).
All amplitudes are in normalized DAC units, and must be between 0 and 1.

```python
for severity, message in amplitude_schedule.validate(catalog, n_directions=2):
    print(f"{severity:>7}: {message}")

print()
too_loud = AmplitudeSchedule.multiplicative(1.0, 500.0, 3)
for severity, message in too_loud.validate(catalog):
    print(f"{severity:>7}: {message}")
```

## 6. Running the amplitude iteration

The call looks like `multisweep`'s, plus `amp_schedule` and `directions`. Note that
`amp_schedule` replaces `amp`.

`sweep_callback` fires once per completed sweep, which is what to hook a
progress bar or a live plot to. It also hands back intermediate sweeps,
so that in case something fails, not all the data is lost.

```python
def report(record):
    amplitudes = record["amplitudes"]
    print(f"  [{record['completed']}/{record['total']}] "
          f"step {record['step']} {record['direction']:<8} "
          f"R0001 at {amplitudes['R0001']:.5f}")

multiamp_results = await crs.multiamp_multisweep(
    catalog,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    amp_schedule=amplitude_schedule,
    sweep_callback=report,
)

print(f"\nkeyed by module: {list(multiamp_results)}")
```

Which is the same shape section 2 unpacked — module identifier, then an envelope,
then `results` by amplitude step and direction. The only difference is that there
is now more than one step in it:

```python
multiamp_module_results = multiamp_results[crs.module[MODULE].index()]

print(f"envelope        {list(multiamp_module_results)}")
print(f"amplitude steps {list(multiamp_module_results['results'])}")
print(f"directions      {list(multiamp_module_results['results'][0])}")
```

- `results` is keyed by **amplitude step**, numbered in the order measured, and
  each step holds one entry per **direction** swept and nothing else. Under a
  direction is the sweep sections of one `multisweep`.
- `call_params` records what the driver was asked for — including the schedule,
  so a saved result can say what produced it.

```python
first = sections_of(multiamp_results)
print(f"step 0, upward: {list(first)[:4]} …")
print(f"R0001 swept at {first['R0001']['sweep_amplitude']:.5f}")

print(f"\ncall_params: {list(multiamp_module_results['call_params'])}")
print(f"schedule as stored: {multiamp_module_results['call_params']['amp_schedule']}")
```

Note what is *not* in the call_params: no step-level copy of the amplitudes. These are
documented within each sweep section's entry in the iterated multisweep results.

`sections_of` works on this unchanged, because a ladder and a single sweep nest
identically — pass `step=` to pick a rung out.

## 7. Reading the results back

There are also some convenience functions for extracting the data in various
arrangements. They all take **one module's envelope** — the thing
`multiamp_results[crs.module[MODULE].index()]` gave us above — rather than the
whole dict.

```python
from rfmux.tuning import (
    collect_amplitude_iterations_for,
    find_iteration_matching_amplitude,
    get_amplitudes_at_iteration,
)
```

Hand one the whole dict instead and it says so, and names the modules it found,
rather than walking a container as though it were an envelope:

```python
try:
    collect_amplitude_iterations_for(multiamp_results, "R0001")
except TypeError as e:
    print(f"TypeError: {e}")
```

### Get one resonator across every amplitude

```python
iterations = collect_amplitude_iterations_for(multiamp_module_results, "R0001")

for iteration, by_direction in iterations.items():
    section = by_direction["upward"]
    print(f"iteration {iteration}  {section['sweep_amplitude']:.5f}")
```

Which is the shape a plot wants. Colouring the traces by amplitude makes the
progression readable without a ten-entry legend — and the same helper is reused
by every plot below, so one colourbar means one thing throughout:

```python
from matplotlib.colors import LogNorm

# gnuplot runs black → purple → red → orange → yellow, so it stays saturated
# from end to end and every trace reads against a white background.
AMPLITUDE_CMAP = plt.cm.gnuplot


def amplitude_colours(amplitudes):
    """One colour per amplitude, plus the mappable a colourbar needs.

    Log-scaled, because an amplitude schedule is log-spaced by default and a
    linear scale would bunch every low rung into one shade.
    """
    lo, hi = min(amplitudes), max(amplitudes)
    if hi > lo:
        norm = LogNorm(vmin=lo, vmax=hi)
        colours = [AMPLITUDE_CMAP(norm(a)) for a in amplitudes]
    else:
        # One amplitude, or several identical ones: nothing to grade.
        norm = LogNorm(vmin=lo * 0.9, vmax=lo * 1.1)
        colours = [AMPLITUDE_CMAP(0.5)] * len(amplitudes)
    return colours, plt.cm.ScalarMappable(norm=norm, cmap=AMPLITUDE_CMAP)


def plot_amplitude_iterations(results, name, direction="upward"):
    """One sweep section, at every amplitude it was measured at."""
    iterations = collect_amplitude_iterations_for(results, name)
    sections = [by_direction[direction] for by_direction in iterations.values()]
    amplitudes = [s["sweep_amplitude"] for s in sections]
    colours, mappable = amplitude_colours(amplitudes)

    fig, (ax_mag, ax_iq) = plt.subplots(
        1, 2, figsize=(11, 4), constrained_layout=True
    )
    for section, colour in zip(sections, colours):
        offset_khz = (
            section["frequencies"] - section["original_center_frequency"]
        ) / 1e3
        # Divide out the drive, so the shapes can be compared rather than just
        # the one that was loudest sitting on top.
        iq = section["iq_counts"] / section["sweep_amplitude"]

        ax_mag.plot(offset_khz, 20 * np.log10(np.abs(iq)), lw=1.0, color=colour)
        ax_iq.plot(iq.real, iq.imag, lw=1.0, color=colour)

    ax_mag.set_xlabel("offset [kHz]")
    ax_mag.set_ylabel("|S21| / drive [dB]")
    ax_iq.set_xlabel("I / drive")
    ax_iq.set_ylabel("Q / drive")
    ax_iq.set_aspect("equal", "datalim")
    fig.colorbar(mappable, ax=(ax_mag, ax_iq), label="sweep amplitude")
    fig.suptitle(f"{name}, swept {direction} at {len(sections)} amplitudes")
    plt.show()


plot_amplitude_iterations(multiamp_module_results, "R0001")
```

### Get every resonator's data at a particular amplitude step

```python
for name, amplitude in list(
    get_amplitudes_at_iteration(multiamp_module_results, 2).items()
)[:4]:
    print(f"{name}  {amplitude:.5f}")
```

Plotted, that is the whole array as one amplitude step saw it — a panel per
sweep section, since they sit at different frequencies and have different
depths, so overlaying them would compare nothing. With *multiplicative* steps
every section is at its own amplitude, so the panels take a spread of colours;
with *ramp* steps they would all be one colour, because they were all probed at
the same amplitude:

```python
def plot_sections_at_iteration(results, iteration, direction="upward", ncols=5):
    """Every sweep section of one amplitude step, one panel each.

    A panel apiece rather than one crowded axes: the sections sit at different
    frequencies and have different depths, so overlaying them compares nothing.
    """
    sections = results["results"][iteration][direction]
    amplitudes = get_amplitudes_at_iteration(results, iteration)
    colours, mappable = amplitude_colours([amplitudes[n] for n in sections])

    nrows = -(-len(sections) // ncols)   # ceiling division, no import needed
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.4 * ncols, 2.5 * nrows),
        constrained_layout=True, squeeze=False,
    )
    panels = axes.ravel()

    for panel, (name, section), colour in zip(panels, sections.items(), colours):
        offset_khz = (
            section["frequencies"] - section["original_center_frequency"]
        ) / 1e3
        iq = section["iq_counts"] / section["sweep_amplitude"]

        panel.plot(offset_khz, 20 * np.log10(np.abs(iq)), lw=1.0, color=colour)
        panel.set_title(f"{name}\n{amplitudes[name]:.5f}", fontsize=8)
        panel.tick_params(labelsize=7)

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


plot_sections_at_iteration(multiamp_module_results, 2)
```

### determine which amplitude step used a particular amplitude


```python
for name in ("R0001", "R0002", "R0003"):
    at_bias = find_iteration_matching_amplitude(multiamp_module_results, name)
    bias = catalog[name].bias.amplitude
    print(f"{name}  bias {bias:.5f}  → step {at_bias}")
```

Ask for a *fixed* amplitude instead and the three part company, which is why the
function needs a name at all. `R0001`, `R0002` and `R0003` are walking different
ranges, so the same amplitude sits at a different rung of each:

```python
print(f"{'':<8}" + "".join(f"{s:>10}" for s in multiamp_module_results["results"]))
for name in ("R0001", "R0002", "R0003"):
    amplitudes = [
        by_direction["upward"]["sweep_amplitude"]
        for by_direction in collect_amplitude_iterations_for(
            multiamp_module_results, name
        ).values()
    ]
    print(f"{name:<8}" + "".join(f"{a:>10.5f}" for a in amplitudes))

print()
for name in ("R0001", "R0002", "R0003"):
    step = find_iteration_matching_amplitude(multiamp_module_results, name, 0.002)
    got = get_amplitudes_at_iteration(multiamp_module_results, step)[name]
    print(f"0.00200 for {name}  → step {step}  (actually {got:.5f})")
```

Note that the matching is on *nearest*, not exact.

The corollary is that there is *always* a nearest, so the function answers even
when nothing is remotely close.

For example, if you for an amplitude only `R0002` ever reaches
and the other two return their top rung regardless:

```python
for name in ("R0001", "R0002", "R0003"):
    step = find_iteration_matching_amplitude(multiamp_module_results, name, 0.016)
    got = get_amplitudes_at_iteration(multiamp_module_results, step)[name]
    print(f"0.01600 for {name}  → step {step}  (actually {got:.5f})")
```

So if the match has to be good, check it as shown above.

### sweeping in both directions

Tell multisweep which frequency direction to sweep in with the `directions` parameter.
The directions will be measured in the order they are provided.

Each amplitude's
up-and-down pair is measured together and the amplitude marches monotonically.

```python
both_ways = await crs.multiamp_multisweep(
    catalog,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    amp_schedule=AmplitudeSchedule.multiplicative(1.0, 2.0, 2),
    directions=("upward", "downward"),
)

both_ways_module_results = both_ways[crs.module[MODULE].index()]

for step, by_direction in both_ways_module_results["results"].items():
    for direction, sections in by_direction.items():
        print(f"step {step}  {direction:<9} "
              f"R0001 at {sections['R0001']['sweep_amplitude']:.5f}")
```

Both directions of one section, on one pair of axes — amplitude as colour,
direction as line style. On a simulated array the two directions lie on top of
each other; on real detectors driven hard enough to bifurcate, they part
company, and that gap is the thing you are looking for:

```python
def plot_both_directions(results, name):
    iterations = collect_amplitude_iterations_for(results, name)
    amplitudes = [
        next(iter(by_direction.values()))["sweep_amplitude"]
        for by_direction in iterations.values()
    ]
    colours, mappable = amplitude_colours(amplitudes)
    styles = {"upward": "-", "downward": "--"}

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for by_direction, colour in zip(iterations.values(), colours):
        for direction, section in by_direction.items():
            offset_khz = (
                section["frequencies"] - section["original_center_frequency"]
            ) / 1e3
            iq = section["iq_counts"] / section["sweep_amplitude"]
            ax.plot(offset_khz, 20 * np.log10(np.abs(iq)), lw=1.0, color=colour,
                    ls=styles.get(direction, ":"))

    # One legend entry per direction, rather than one per trace.
    for direction, style in styles.items():
        ax.plot([], [], color="0.3", ls=style, label=direction)

    ax.set_xlabel("offset [kHz]")
    ax.set_ylabel("|S21| / drive [dB]")
    ax.legend(fontsize=8)
    fig.colorbar(mappable, ax=ax, label="sweep amplitude")
    fig.suptitle(f"{name}, both frequency directions")
    plt.show()


plot_both_directions(both_ways_module_results, "R0001")
```

A step swept once and a step swept twice have the same shape — the directions
present are simply the ones you asked for:

```python
one_way = await crs.multiamp_multisweep(
    catalog,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    amp_schedule=AmplitudeSchedule.explicit([PROBE_AMPLITUDE]),
    directions=("downward",),
)

print(f"directions present: "
      f"{list(one_way[crs.module[MODULE].index()]['results'][0])}")
```

### A frequency list at several amplitudes

The bare-frequency form works here too — this is how you find a sensible probe
amplitude *before* anything is tuned. There is no bias amplitude to scale, so
the schedule has to carry its own: `ramp` and `explicit` do by construction,
while `multiplicative` would need an explicit `base`.

```python
untuned_results = await crs.multiamp_multisweep(
    center_frequencies=section_center_frequencies,
    module=MODULE,
    span_hz=SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
    amp_schedule=AmplitudeSchedule.ramp(PROBE_AMPLITUDE, PROBE_AMPLITUDE * 4, 3),
)

untuned_module_results = untuned_results[crs.module[MODULE].index()]

for step, by_direction in untuned_module_results["results"].items():
    amplitude = by_direction["upward"]["S0001"]["sweep_amplitude"]
    print(f"step {step}  every section at {amplitude:.5f}")

# The readers work on this exactly as they do on a catalog's results — the only
# difference is the key. These sections are off-resonance, so the traces are
# flat; what the plot shows is the three amplitudes, not a resonance.
plot_amplitude_iterations(untuned_module_results, "S0001")

try:
    await crs.multiamp_multisweep(
        center_frequencies=section_center_frequencies,
        module=MODULE,
        span_hz=SPAN_HZ,
        npoints_per_sweep=NPOINTS_PER_SWEEP,
        amp_schedule=AmplitudeSchedule.multiplicative(0.5, 2.0, 3),   # relative to what?
    )
except ValueError as e:
    print(f"\nValueError: {e}")
```

## 8. What is not here yet

- **Choosing the operating amplitude.** Iterating over amplitudes gives you the
  data to see where each detector bifurcates; deciding which step to bias at, and
  writing that back into the catalog, is `find_bias_points` and is not ported yet.
- **Fitting.** Not missing any more — `rfmux.tuning.fit_sweeps` takes what
  `multiamp_multisweep` returned and writes each model's answers into the sweep
  entry it fitted, under `fits`. `fitting_resonators.md` is the notebook.
  Writing the results back into the *catalog* is still to come.
- **Saving to disk.** `pickle.dump` on the returned dict works today — it is
  plain builtins and ndarrays throughout — but a proper `store.py` with a file
  layout is still to come.

One cleanup note: multisweep silences the channels it swept, but only those. If
you parked tones on this module by hand, they are still live.
