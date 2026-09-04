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

A multisweep gives you one sweep trace per resonator. Fitting takes each of
those traces and estimates the parameters of the resonator that produced it.

Fitting is a separate step that you run yourself on multisweep data that
already exists. By default, the sweep measurement does not fit anything as it goes.
 This means you can re-fit the same data with
different parameters as many times as you like, and you can fit data loaded from
disk exactly as you would fit a sweep you just took.

rfmux currently provides three models. They are independent of each other, so
you can run any combination of them:

| Model | Fitted to | Gives you |
|---|---|---|
| `skewed` | `\|S21\|` | `fr`, `Qr`, `Qc`, `Qi` |
| `nonlinear` | the complex `S21`, with the readout gain divided out | the same, plus `a`, which indicates how close the drive has pushed the resonator towards bifurcation |
| `circle` | the IQ loop | a centre and a radius, which you may or may not want to subtract before looking at IQ data |

| Piece | Module |
|---|---|
| The fitters, and the entry points used below | `rfmux.tuning.fits` |
| The sweep being fitted | `rfmux.algorithms.measurement.multisweep` (`crs.multisweep`) |
| Iterating the sweep over amplitudes | `rfmux.algorithms.measurement.multiamp_multisweep` |
| The array bookkeeping | `rfmux.core.resonators` |

This notebook starts from an array that has already been tuned, i.e. a
`ResonatorCatalog` whose bias points are set, so that it can get on with the
fitting. Getting to that point is covered in two other notebooks: run a network
analysis and find the resonances in
`network_analyses_find_resonances_make_resonator_catalog.md`, then sweep them in
`multisweep.md`. If the tuning workflow is unfamiliar, you will probably want to
read those first.

## How to use this document

**This is a runnable notebook, not a web page.** Every grey block below is a live
code cell: put the cursor in it and press **Shift+Enter** to execute it.

- **Run the cells in order, top to bottom.** Later cells use variables the
  earlier ones defined, so skipping ahead fails with a `NameError`. *Kernel →
  Restart Kernel and Run All Cells* starts clean.
- **The outputs you see are the ones you just produced.** This file is stored as
  jupytext markdown, which keeps no saved outputs, so a cell is blank until you
  run it. Nothing here can show you a stale number from someone else's run.
- **Editing is encouraged.** Change the amplitudes, the span, the models you ask
  for, and re-run — that is what this document is for. The shipped copy is
  read-only, so *File → Save Notebook As…* to keep your changes.
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

import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import rfmux
from rfmux.core.resonators import ResonatorCatalog
from rfmux.tuning import AmplitudeSchedule

MODULE = 1

PROBE_AMPLITUDE = 0.001   # normalized DAC units — where this array is biased

# The sweep span and point spacing matter more for fitting than they do for
# plotting, and there are a few things pulling against each other:
#
# - The fitters want the dip AND some off-resonance baseline either side of it,
#   which is what they use to estimate the readout gain and the dip depth. The
#   nonlinear fitter works best with a span of about 6 * fr / Qr.
# - They also want the dip itself sampled by more than a couple of points. A
#   very wide span will still fit, but the resonance ends up as a handful of
#   points in the middle of a straight line, and the IQ loop is barely traced
#   out at all.
# - The dip also needs to stay within the fit's bound on fr (37.5% of the span,
#   by default) at every amplitude in the ladder, since driving a resonator
#   harder pulls its resonance down in frequency.
#
# So this notebook uses two spans, and section 2 measures both: a wide one for
# the amplitude ladder, where the resonance moves around, and a narrow one for
# the IQ plots, where we want the loop well sampled.
LADDER_SPAN_HZ = 200e3
FINE_SPAN_HZ = 40e3
NPOINTS_PER_SWEEP = 201
NSAMPS = 10
```

## 1. An array that is already tuned

Four simulated LEKIDs, on a fixed random seed so that this notebook produces the
same array and the same numbers every time it runs.

Normally you would find the resonances with a network analysis and build a
catalog from them, but we can take a shortcut here. Setting
`auto_bias_kids: True` asks the simulator to park a tone on each of its own
resonators, at the actual S21 transmission minimum.
 Reading those tone frequencies back gives us roughly the
frequencies a real tuning workflow would have found.

To run against real hardware instead, replace this one cell with a session on
your board and a catalog you built or loaded. Everything after it is unchanged:

    session = rfmux.load_session('!HardwareMap [ !CRS { serial: "0042" } ]')
    crs = session.query(rfmux.CRS).one()
    await crs.resolve()

```python
MOCK_CONFIG = {
    "num_resonances": 4,
    "freq_start": 0.6e9,
    "freq_end": 0.9e9,
    "resonator_random_seed": 42,   # same array every run
    "auto_bias_kids": True,        # the simulator tunes itself, so we can skip ahead
    "bias_amplitude": PROBE_AMPLITUDE,
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

<!-- #region -->


We then move two of the bias amplitudes off the default, just so that they are
not all the same. Real arrays generally end up with a different bias amplitude
per detector, and it makes section 6 more interesting: that section is about
picking out the sweep taken at each resonator's own operating point, which is
not much of a question if they all share one.
<!-- #endregion -->

```python
catalog = ResonatorCatalog.from_frequencies(
    bias_frequencies,
    module=MODULE,
    amplitude=PROBE_AMPLITUDE,
)

catalog["R0002"].set_bias(amplitude=PROBE_AMPLITUDE * 2)
catalog["R0003"].set_bias(amplitude=PROBE_AMPLITUDE / 2)

print(catalog)
for resonator in catalog:
    print(f"  {resonator.name}  ch {resonator.channel}  "
          f"{resonator.bias.frequency_hz/1e6:.4f} MHz  "
          f"amp {resonator.bias.amplitude:.5f}")
```

## 2. Something to fit

Now we need some multisweep data. For example, let's look at the
array swept at five amplitudes, in both frequency directions, which gives 40
traces from four resonators.

This is one `multiamp_multisweep` call.
`multisweep.md` covers this call in detail; here it is just the input to the
fitting. 

Note that the amplitude schedule below is *multiplicative*, so each resonator's steps are
multiples of its own bias amplitude — half of it, then one, two, four and eight
times. That means step 1 (the second step, since they are zero-indexed)
is the step where each resonator is actually biased.

```python
amplitude_schedule = AmplitudeSchedule.multiplicative(0.5, 8.0, 5)
print(amplitude_schedule)

for step in amplitude_schedule.steps(catalog):
    print(step)

multiamp_ms = await crs.multiamp_multisweep(
    catalog,
    span_hz=LADDER_SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP//2,
    nsamps=NSAMPS,
    amp_schedule=amplitude_schedule,
    directions=("upward", "downward"),
)

# A sweep comes back keyed by module
# fit_sweeps takes one module's output at a time, so we index into it and
# everything below is about this module.
multiamp_results = multiamp_ms[crs.module[MODULE].index()]

print(f"\nmodules:         {list(multiamp_ms)}")
print(f"amplitude steps: {list(multiamp_results['results'])}")
print(f"directions:      {list(multiamp_results['results'][0])}")
print(f"resonators:      {list(multiamp_results['results'][0]['upward'])}")
```

### Looking at the traces before fitting them

Printing the keys says what shape the data is; it does not say whether the
measurement is any good. Worth a look before spending time fitting it.

These are the plotters from `multisweep.md`, reproduced here so this notebook
runs on its own: `plot_sections_at_iteration` for the whole array at one
amplitude step, `plot_amplitude_iterations` for one resonator up the whole
ladder, and `plot_ms` for a plain set of sweep sections. `amplitude_colours`
underpins the first two, and section 7 re-uses it for the fitted versions of the
same plots.

Every sweep section is at a different amplitude to its neighbours here, so
colour means amplitude throughout, and the traces are divided by their own drive
so that their shapes can be compared rather than the loudest simply sitting on
top.

```python
from rfmux.tuning import (
    collect_amplitude_iterations_for,
    get_amplitudes_at_iteration,
)

# gnuplot runs black → purple → red → orange → yellow, so it stays saturated
# from end to end and every trace reads against a white background.
AMPLITUDE_CMAP = plt.cm.gnuplot


def amplitude_colours(amplitudes):
    """One colour per amplitude, plus the mappable a colourbar needs.

    Log-scaled, because an amplitude schedule is log-spaced by default and a
    linear scale would bunch every low rung into one shade.
    """
    low, high = min(amplitudes), max(amplitudes)
    if high > low:
        norm = LogNorm(vmin=low, vmax=high)
        colours = [AMPLITUDE_CMAP(norm(a)) for a in amplitudes]
    else:
        # One amplitude, or several identical ones: nothing to grade.
        norm = LogNorm(vmin=low * 0.9, vmax=low * 1.1)
        colours = [AMPLITUDE_CMAP(0.5)] * len(amplitudes)
    return colours, plt.cm.ScalarMappable(norm=norm, cmap=AMPLITUDE_CMAP)


def plot_sections_at_iteration(results, iteration, direction="upward", ncols=4):
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

    for panel, (name, sweep_section), colour in zip(panels, sections.items(), colours):
        offset_khz = (
            sweep_section["frequencies"] - sweep_section["original_center_frequency"]
        ) / 1e3
        iq = sweep_section["iq_counts"] / sweep_section["sweep_amplitude"]

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


# Step 1 is the factor-of-1.0 rung, so this is the array as it sits at its own
# bias amplitudes — four different amplitudes, hence four different colours.
plot_sections_at_iteration(multiamp_results, 1)
```

And one resonator across the whole ladder. This is the measurement section 7
fits: the resonance moving down in frequency and going shallow as the drive
comes up is exactly what the fitted parameters are going to report as a number.

```python
def plot_amplitude_iterations(results, name, direction="upward"):
    """One sweep section, at every amplitude it was measured at."""
    iterations = collect_amplitude_iterations_for(results, name)
    sections = [by_direction[direction] for by_direction in iterations.values()]
    amplitudes = [s["sweep_amplitude"] for s in sections]
    colours, mappable = amplitude_colours(amplitudes)

    fig, (ax_mag, ax_iq) = plt.subplots(
        1, 2, figsize=(11, 4), constrained_layout=True
    )
    for sweep_section, colour in zip(sections, colours):
        offset_khz = (
            sweep_section["frequencies"] - sweep_section["original_center_frequency"]
        ) / 1e3
        # Divide out the drive, so the shapes can be compared rather than just
        # the one that was loudest sitting on top.
        iq = sweep_section["iq_counts"] / sweep_section["sweep_amplitude"]

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


plot_amplitude_iterations(multiamp_results, "R0001")
```

<!-- #region -->
We will also take one plain `multisweep` over the narrow span, at the bias
amplitudes only. The fitting does not need this, but it makes the IQ plots in
section 5 much easier to read: 40 kHz over 201 points is a point every 200 Hz,
which is a few points per linewidth on this array. The sweep that iterated over amplitudes used a fairly coarse
point spacing - this is generally fine for fits, but doesn't look as nice.


Note that a single `multisweep` returns the same shape as the iterative one above, just with a single amplitude iteration
in a single direction.
<!-- #endregion -->

```python
fine_multisweep = (await crs.multisweep(
    catalog,
    span_hz=FINE_SPAN_HZ,
    npoints_per_sweep=NPOINTS_PER_SWEEP,
    nsamps=NSAMPS,
))[crs.module[MODULE].index()]


def sections_of(results, step=0, direction="upward"):
    """The {name: entry} sweep sections of one step, out of one module's result."""
    return results["results"][step][direction]


print(f"{len(sections_of(fine_multisweep))} sweeps, "
      f"{FINE_SPAN_HZ / (NPOINTS_PER_SWEEP - 1):.0f} Hz between points")
```

Plotted with `plot_ms`, the plainest of the three: a set of sweep sections, in
the IQ plane and in magnitude, with no amplitude colouring because a single
`multisweep` is one amplitude per resonator. Compare the loops here against the
IQ panel of the ladder plot above — same resonators, a fifth of the span, and
the loop actually traced out rather than cut across.

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


fine_sections = sections_of(fine_multisweep)
plot_ms(fine_sections, list(fine_sections),
        f"fine multisweep, {FINE_SPAN_HZ/1e3:.0f} kHz span at the bias amplitudes")
```

Before fitting anything, here is what one sweep section entry holds. Seven keys:
the measured data, plus the bookkeeping that says what was measured and how.

```python
def show_sweep_section(sweep_section, indent=""):
    """Print an entry's keys, with ndarrays shown as shapes rather than dumped."""
    for key, value in sweep_section.items():
        if isinstance(value, np.ndarray):
            print(f"{indent}{key:<28} ndarray{value.shape} {value.dtype}")
        elif isinstance(value, dict):
            print(f"{indent}{key:<28} dict, keys {list(value)}")
        else:
            print(f"{indent}{key:<28} {value!r}")


show_sweep_section(sections_of(multiamp_results)["R0001"])
```

## 3. Fitting the data

`rfmux.tuning.fit_sweeps` does the whole thing in one call. It walks the results dictionary,
fits every sweep it finds in there, and writes each model's results back into
the sweep entry it fitted, so that the fit parameters end up stored alongside
the data they came from.

**Note that it does not return a copy of your data.** What comes back is a report
describing what it did, which is handy when you are fitting a few thousand
traces and want to know how it went.

```python
from rfmux.tuning import fit_sweeps

fit_report = fit_sweeps(multiamp_results)

print(fit_report)
```

### What you can ask fit_sweeps for

That call took every default, which is why it fitted all three models to all 40
traces. In practice you will often want less than that, so here is the whole set
of arguments in one place. They are all keyword-only.

The first four choose **which sweeps get fitted**. Each takes a single value or
an iterable of them, and `None` — the default — means all of them:

| Argument | Default | Selects |
|---|---|---|
| `models` | `("skewed", "nonlinear", "circle")` | which of the three models to run. Only want Q values? `models=("skewed",)` and you have skipped the expensive one |
| `names` | `None` | which resonators, or which sections for a bare frequency list |
| `iterations` | `None` | which amplitude steps |
| `directions` | `None` | `"upward"`, `"downward"`, or both |

Section 6 is where these get used in earnest, including the case they cannot
express: fitting each resonator at *its own* bias amplitude, which is a
different amplitude step per resonator.

The rest control **how each fit is done**. The circle fit takes none of them —
it is a linear solve with nothing to tune:

| Argument | Default | Model | Does |
|---|---|---|---|
| `approx_Qr` | `10000.0` | skewed | the initial guess for `Qr`. Worth setting if your array is far from the default and the fits are missing |
| `normalize` | `True` | skewed | divide each trace by its last point before fitting, so `A` comes out near 1 and the model is in units of the off-resonance level. This changes the units `rfmux.tuning.skewed_model_magnitude` returns — section 5 |
| `fr_limit_hz` | `None` | skewed | how far `fr` is allowed to move from the sweep centre. `None` means 37.5% of the span, which stops the fit wandering onto a neighbouring resonator that leaked into the edge |
| `fit_nonlinearity` | `True` | nonlinear | fit `a`, or hold it at zero and fit a linear resonator on the same seven-parameter machinery |
| `n_extrema_points` | `5` | nonlinear | how many points at each end of the sweep are averaged to estimate the readout gain |
| `max_residual` | `0.1` | nonlinear | the ceiling above which a converged fit is reported as a bad one. Section 8 abuses this to make fits fail on purpose |

And two about **how it runs**, which change nothing about the answers:

| Argument | Default | Does |
|---|---|---|
| `max_workers` | `None` | threads to fit on. One sweep is one job, so all of a sweep's models run on the same thread. `None` uses `min(4, cpu_count)` |
| `progress_callback` | `None` | called `(completed, total)` after each sweep, where *total* counts sweeps rather than fits. For driving a progress bar |

The report counts each model separately. It also carries a `settings` dict
recording what the fitters were asked for — the second table above, the
arguments that change the answers. These are not stored on the individual
entries, since they would be the same values repeated on every resonator, so the
report is where to look if you want to know how a given set of fits was
produced. The selection arguments are not in there: which sweeps you picked is
already visible in which entries came back fitted.

```python
print(f"total fits    {len(fit_report)}")
print(f"fitted        {len(fit_report.fitted)}")
print(f"failed        {len(fit_report.failed)}")
print(f"skewed only   {len(fit_report.for_model('skewed'))}")

print("\nsettings:")
for key, value in fit_report.settings.items():
    print(f"  {key:<18} {value!r}")
```

## 4. How the fitters modify the results dictionary

Here is the same sweep section entry we looked at in section 2, now with one
extra key on it:

```python
fitted_sweep_section = sections_of(multiamp_results)["R0001"]

show_sweep_section(fitted_sweep_section)
```

That `fits` subdict is keyed by model name, and each model gets its own subdict
under it, containing that model's own results:

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

Putting that together, here is the whole layout, from the top of a
`multiamp_multisweep` result down to a single fitted trace:

```text
multiamp_results
├── schema_version
├── module
├── call_params                             what the driver was asked for
└── results
    └── 0                                   amplitude step, numbered as measured
        └── "upward"                        sweep direction
            └── "R0001"                     resonator (or "S0001…" for a bare
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

A few things worth noting about this layout.

**The measured data is left alone.** The fitters only ever add the `fits` key,
so re-running a fit with different parameters is safe, and running one model
does not disturb another model's results.

**`failed_because` is either `None` or a short explanation.** If a fit does not
work, the reason is recorded on the entry rather than warned about somewhere
you may not be looking. Section 8 goes into this.

**Quantities that can be recomputed are not stored.** You will notice there are
no model curves in there, no gain-corrected trace and no re-centred IQ loop.
Each of those is just a function of the stored parameters and the arrays the
entry already carries, so storing them as well would mostly be a way for a saved
file to end up internally inconsistent. There are four reader functions that
recompute them for you instead, and section 5 demonstrates all four:

| Reader | Recomputes |
|---|---|
| `skewed_model_magnitude(entry)` | the `\|S21\|` predicted by the skewed fit |
| `nonlinear_model_iq(entry)` | the complex trace predicted by the nonlinear fit, in counts |
| `gain_corrected_iq(entry)` | `iq_counts` with the estimated readout gain divided out |
| `centered_iq(entry)` | `iq_counts` with the fitted circle centre subtracted |

## 5. Plotting the fits against the data

Below is each of the three models drawn over the trace it was fitted to, using
the readers from the table above.

A fit is a continuous model, and the measurement is a coarse sampling of it, so
the two want drawing differently: points for what was measured, a smooth line
for what was fitted. The readers evaluate on the entry's own `frequencies`,
which is what you want for a residual but makes the model look as jagged as the
data. A sweep entry is a plain dict, though, so handing a reader a copy with a
denser frequency axis gets the model's real shape out of it — the fit
parameters it reads are the stored ones either way. `model_on_a_finer_grid`
below does that, and every model curve in this notebook goes through it.

It is worth watching what this shows. The amplitude steps were measured at
2 kHz between points, which is coarse enough that the dip is a handful of
samples, and the fitted curve still runs through them.

Starting with the skewed Lorentzian, in magnitude. One thing to watch out for:
`rfmux.tuning.skewed_model_magnitude` returns the model in whatever units the
fit worked in.
With the default `normalize=True`, the fitter divides the trace by its last
point before fitting, so the model should be compared against
`np.abs(iq_counts / iq_counts[-1])` rather than the raw magnitude.

The x-axis here is zoomed to a few linewidths either side of the fitted `fr`.
The fit itself used the whole 200 kHz span, but there is not much to see out at
the edges.

```python
from rfmux.tuning import (
    centered_iq,
    gain_corrected_iq,
    nonlinear_model_iq,
    skewed_model_magnitude,
)


def model_on_a_finer_grid(reader, sweep_section, oversample=25):
    """(frequencies, model) from a reader, on a denser axis than was measured.

    Returns the frequencies too, since they are no longer the entry's own and
    the model has to be plotted against them.
    """
    frequencies = np.linspace(
        sweep_section["frequencies"][0],
        sweep_section["frequencies"][-1],
        oversample * len(sweep_section["frequencies"]),
    )
    return frequencies, reader({**sweep_section, "frequencies": frequencies})


def plot_skewed_fits(results, iteration=0, direction="upward", linewidths=6):
    """Every resonator at one amplitude step, with its skewed fit over it."""
    sections = sections_of(results, iteration, direction)

    fig, axes = plt.subplots(
        1, len(sections), figsize=(3.1 * len(sections), 3.2),
        constrained_layout=True, squeeze=False,
    )
    for panel, (name, sweep_section) in zip(axes[0], sections.items()):
        offset_khz = (
            sweep_section["frequencies"] - sweep_section["original_center_frequency"]
        ) / 1e3
        normalized = np.abs(
            sweep_section["iq_counts"] / sweep_section["iq_counts"][-1]
        )

        panel.plot(offset_khz, 20 * np.log10(normalized), lw=0, marker=".",
                   ms=2.5, color="0.45", label="measured")

        skewed_fit = sweep_section["fits"]["skewed"]
        if skewed_fit["failed_because"] is None:
            params = skewed_fit["params"]
            model_frequencies, model = model_on_a_finer_grid(
                skewed_model_magnitude, sweep_section
            )
            model_offset_khz = (
                model_frequencies - sweep_section["original_center_frequency"]
            ) / 1e3
            panel.plot(model_offset_khz, 20 * np.log10(model), lw=1.4,
                       color="crimson", label="skewed fit")
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
            panel.set_xlim(centre_khz - half_width_khz, centre_khz + half_width_khz)
        else:
            panel.set_title(f"{name}\nno fit", fontsize=9)

        panel.set_xlabel("offset [kHz]", fontsize=8)
        panel.tick_params(labelsize=7)

    axes[0, 0].set_ylabel("|S21| / off-resonance [dB]", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"skewed Lorentzian fits, amplitude step {iteration} {direction}")
    plt.show()


plot_skewed_fits(multiamp_results)
```

Since we are working with a simulated array, there is a sanity check available
here for free. Scroll back to the simulator's output in section 1: it printed a
`Q value` for each resonator as it generated them. Those are internal Qs, so
they should be roughly comparable to the fitted `Qi` values above — a quick way
to convince yourself the fitter is doing something sensible.

Next, the nonlinear model. This one is fitted to the complex trace rather than
just the magnitude, so the IQ plane is where you can see what it is doing. Let's apply
this one to the more finely sampled sweep.

```python
fit_sweeps(fine_multisweep)

print(f"R0001 fits: {list(sections_of(fine_multisweep)['R0001']['fits'])}")
```

The middle panel below shows what the fitter actually worked with:
`rfmux.tuning.gain_corrected_iq` is `iq_counts` divided by the single complex
gain value the
fit estimated and stored.

```python
def plot_nonlinear_fit(sections, name="R0001"):
    """One resonator's nonlinear fit: measured and model, in IQ and in magnitude."""
    sweep_section = sections[name]
    nonlinear_fit = sweep_section["fits"]["nonlinear"]

    if nonlinear_fit["failed_because"] is not None:
        print(f"{name}: {nonlinear_fit['failed_because']}")
        return

    measured = sweep_section["iq_counts"]
    corrected = gain_corrected_iq(sweep_section)
    offset_khz = (
        sweep_section["frequencies"] - sweep_section["original_center_frequency"]
    ) / 1e3

    # The model on a finer axis than the measurement: in the IQ plane a coarse
    # one would cut the loop into chords, and it is the loop we are looking at.
    model_frequencies, model = model_on_a_finer_grid(
        nonlinear_model_iq, sweep_section
    )
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


plot_nonlinear_fit(sections_of(fine_multisweep))
```

`a` is the nonlinearity parameter: 0 corresponds to a linear resonator, and
bifurcation is expected around `a ≈ 0.77`. At this array's bias amplitude it
comes out near zero.

Finally, the circle fit, which stores just a centre and a radius.
`rfmux.tuning.centered_iq`
uses the centre to shift the IQ loop so that it sits around the origin. You
will need to do this before talking about the "phase" direction for a resonator,
 since a phase measured about the
origin of the raw data reflects the readout chain much more then the resonator.

```python
def plot_circle_fit(sections, name="R0001"):
    """The fitted circle, and the loop it recentres."""
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

    recentred = centered_iq(sweep_section)
    ax_centred.plot(recentred.real, recentred.imag, lw=0, marker=".", ms=3,
                    color="0.45")
    ax_centred.axhline(0, lw=0.6, color="0.8")
    ax_centred.axvline(0, lw=0.6, color="0.8")
    ax_centred.set_xlabel("I − centre")
    ax_centred.set_ylabel("Q − centre")
    ax_centred.set_aspect("equal", "datalim")
    ax_centred.set_title("centered_iq(entry)", fontsize=9)

    fig.suptitle(f"{name} circle fit — radius {radius:.4g} counts")
    plt.show()


plot_circle_fit(sections_of(fine_multisweep))
```

## 6. Choosing which sweeps to fit

`fit_sweeps(results)` on its own fits everything it can find. That is fine here,
but on a 1,000-resonator array swept at eight amplitudes in two directions it
would be 16,000 traces, and you often want rather less than that.

For this section we will work on a copy of the iterated multiple with the `fits` stripped
back off it, so that you can see which entries each selection actually touched:

```python
unfitted_results = copy.deepcopy(multiamp_results)
for by_direction in unfitted_results["results"].values():
    for sections in by_direction.values():
        for sweep_section in sections.values():
            sweep_section.pop("fits")


def which_are_fitted(results):
    """Every (step, direction, name) that has a fits subdict."""
    return [
        (step, direction, name)
        for step, by_direction in results["results"].items()
        for direction, sections in by_direction.items()
        for name, sweep_section in sections.items()
        if "fits" in sweep_section
    ]


print(f"fitted so far: {which_are_fitted(unfitted_results)}")
```

### By name, step and direction

`names`, `iterations` and `directions` each take either a single value or an
iterable of them, and `None` (the default) means all of them. Note that a bare
string counts as one name rather than a sequence of characters, so
`names="R0001"` does what it looks like.

```python
one_trace_report = fit_sweeps(
    unfitted_results,
    names="R0001",
    iterations=0,
    directions="upward",
)

print(one_trace_report)
print(f"\nfitted now: {which_are_fitted(unfitted_results)}")
```

### By model

`models` takes any subset of the three. `nonlinear` is by far the most expensive
of them: it is a seven-parameter fit to the complex data, run up to three times
per trace, where `skewed` is a five-parameter fit to the magnitude and `circle`
is just a linear solve. If you only want Q values, and you are operating at moderate or
low readout amplitude, it is worth leaving the
nonlinear model out.

Since running one model leaves the other models' results alone, you can fit
`skewed` across the whole multiamp multisweep and then run `nonlinear` only where you
actually need it:

```python
fit_sweeps(unfitted_results, names="R0002", iterations=0, models=("skewed",))
sweep_section = unfitted_results["results"][0]["upward"]["R0002"]
print(f"after skewed:            {list(sweep_section['fits'])}")

fit_sweeps(unfitted_results, names="R0002", iterations=0, models=("circle",))
print(f"after circle:            {list(sweep_section['fits'])}  ← skewed kept")
```

### At the amplitude each resonator is biased at

This is a common thing to want after running a multiamp multisweep measurement. The multiple amplitude steps were
measured in order to find a sensible operating amplitude (`bias_finding.md` covers how that one gets chosen), and it is the operating
amplitude itself that you now want fitted.
`rfmux.tuning.fit_sweeps_at_bias_amplitude` works out that
step for each resonator individually, reading each one's bias amplitude from the
catalog snapshot that `multiamp_multisweep` recorded in `call_params`.

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
    measured_at = get_amplitudes_at_iteration(unfitted_results, fit.iteration)[fit.name]
    print(f"{fit.name}  biased at {catalog[fit.name].bias.amplitude:.5f}  "
          f"→ step {fit.iteration}, measured at {measured_at:.5f}")
```

Every resonator came back on step 1 here. That is expected for this particular
amplitude schedule: because it is multiplicative, its steps are the same set of factors
applied to whatever each resonator was biased at, so the factor-of-1.0 step has
the same index for all of them. The *amplitudes* at that step differ, since
these are different resonators at different bias points, but the step number
does not.

Asking for a fixed amplitude instead is where the per-resonator lookup starts to
matter. Each resonator is walking its own range of amplitudes, so the same
number sits at a different step of each. This is also why it can't be expressed
as an `iterations=` argument, which would have to be one step for everybody:

```python
print(f"{'':<8}" + "".join(f"{s:>10}" for s in unfitted_results["results"]))
for resonator in catalog:
    row = [
        get_amplitudes_at_iteration(unfitted_results, step)[resonator.name]
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
    measured_at = get_amplitudes_at_iteration(unfitted_results, fit.iteration)[fit.name]
    print(f"0.00200 for {fit.name}  → step {fit.iteration} "
          f"(actually {measured_at:.5f})")
```

Note that the matching is on *nearest*, not exact. There is always a nearest
step, so it will give you an answer even when nothing in the ladder is
particularly close to what you asked for. If the match needs to be a good one,
check it against `rfmux.tuning.get_amplitudes_at_iteration`, as above.

## 7. Fitted parameters across all amplitude steps

Fitting a whole ladder is mostly worth doing so that you can look at how the
parameters move with drive. Driving a resonator harder tends to pull its
resonance down in frequency and degrade its Q, and the fits condense those 40
traces into a handful of curves showing that.

The first plot is `plot_amplitude_iterations` from section 2 again — same
resonator, same colour scale — with each trace's skewed fit drawn over it, and
zoomed in to where the resonance actually moved to.

Notice that the fitted curves go deeper than the measured points do. That is not
the fit overshooting: at 2 kHz between points, no sample lands on the true
bottom of a dip this narrow, and the model is telling you where the bottom
actually is. It is the clearest illustration in this notebook of why fitting
beats reading numbers off the trace.

```python
def plot_fitted_traces(results, name="R0001", direction="upward", linewidths=8):
    """One resonator at every amplitude, each trace with its skewed fit over it."""
    iterations = collect_amplitude_iterations_for(results, name)
    sections = [by_direction[direction] for by_direction in iterations.values()]
    amplitudes = [s["sweep_amplitude"] for s in sections]
    colours, mappable = amplitude_colours(amplitudes)

    fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    fitted_centres_khz, widest_khz = [], 0.0

    for sweep_section, colour in zip(sections, colours):
        offset_khz = (
            sweep_section["frequencies"] - sweep_section["original_center_frequency"]
        ) / 1e3
        normalized = np.abs(
            sweep_section["iq_counts"] / sweep_section["iq_counts"][-1]
        )
        ax.plot(offset_khz, 20 * np.log10(normalized), lw=0, marker=".", ms=2,
                color=colour, alpha=0.6)

        skewed_fit = sweep_section["fits"]["skewed"]
        if skewed_fit["failed_because"] is None:
            params = skewed_fit["params"]
            model_frequencies, model = model_on_a_finer_grid(
                skewed_model_magnitude, sweep_section
            )
            ax.plot((model_frequencies
                     - sweep_section["original_center_frequency"]) / 1e3,
                    20 * np.log10(model), lw=1.3, color=colour)
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
    fig.suptitle(f"{name}: points measured, lines fitted")
    plt.show()


plot_fitted_traces(multiamp_results)
```

Pulling the fitted parameters out is just a walk over the same nesting. The
helper below returns `None` wherever a fit failed, so that a failure leaves a
gap in the curve instead of a misleading point:

```python
def fitted_parameter(results, name, parameter, model="skewed", direction="upward"):
    """(amplitudes, values) for one fitted parameter across the ladder."""
    iterations = collect_amplitude_iterations_for(results, name)
    amplitudes, values = [], []
    for by_direction in iterations.values():
        sweep_section = by_direction[direction]
        fit = sweep_section["fits"][model]
        amplitudes.append(sweep_section["sweep_amplitude"])
        values.append(
            fit["params"][parameter] if fit["failed_because"] is None else None
        )
    return np.array(amplitudes), np.array(values, dtype=float)


def plot_fitted_parameters_vs_amplitude(results, model="skewed", direction="upward"):
    """fr shift, Qr, Qc and Qi against drive, one line per resonator."""
    names = list(sections_of(results, direction=direction))
    panels = [
        ("fr", "fr − fr(lowest drive) [kHz]", 1e-3),
        ("Qr", "Qr", 1.0),
        ("Qc", "Qc", 1.0),
        ("Qi", "Qi", 1.0),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4),
                             constrained_layout=True)
    for panel, (parameter, label, scale) in zip(axes, panels):
        for name in names:
            amplitudes, values = fitted_parameter(
                results, name, parameter, model=model, direction=direction
            )
            if parameter == "fr":
                # Absolute fr differs by hundreds of MHz between resonators, so
                # plot the shift and the four curves land on one axis.
                values = values - values[0]
            panel.plot(amplitudes, values * scale, marker="o", ms=4, lw=1.2,
                       label=name)
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


plot_fitted_parameters_vs_amplitude(multiamp_results)
```

Because the two fitters are independent, and fitting different things (one the
magnitude, the other the whole complex trace), comparing them is a reasonable
cross-check. Where they agree on `Qr` you can be fairly confident in the number;
where they disagree, it is worth a closer look at the trace to see which of them
is struggling.

```python
print(f"{'':<8}{'amplitude':>12}{'skewed Qr':>12}{'nonlinear Qr':>14}"
      f"{'a':>8}{'residual':>11}")
for name in ("R0001", "R0004"):
    for by_direction in collect_amplitude_iterations_for(multiamp_results, name).values():
        sweep_section = by_direction["upward"]
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

## 8. When a fit fails

If you fit a thousand resonators, some of them are not going to fit. `fit_sweeps`
treats that as a result rather than an error, so it does not raise: the reason
is recorded as `failed_because` on the entry, and repeated on the report.

The easiest way to see this is to force it. `max_residual` is the ceiling the
nonlinear fit uses to decide whether a converged fit is a good one; setting it
absurdly tight means every fit converges and every one is then rejected:

```python
fussy_results = copy.deepcopy(multiamp_results)

fussy_report = fit_sweeps(
    fussy_results,
    iterations=0,
    directions="upward",
    models=("nonlinear",),
    max_residual=1e-9,
)

print(fussy_report)
```

Note that the parameters are still there after the rejection. When a fit
converges on something that does not describe the data well, it is usually more
useful to be able to see what it converged to than to have it thrown away.

```python
rejected_fit = fussy_results["results"][0]["upward"]["R0001"]["fits"]["nonlinear"]
print(f"failed_because  {rejected_fit['failed_because']}")
print(f"params          fr {rejected_fit['params']['fr']/1e6:.4f} MHz, "
      f"Qr {rejected_fit['params']['Qr']:.4g}")
print(f"residual        {rejected_fit['residual']:.2e}")
```

A fit that never converged at all gets `params: None`, along with a note about
what stopped it. 


## 10. What is not here yet

A few related things that this layer does not currently do:

- **df calibration off a fit.** Converting a timestream into a frequency shift
  needs a Hz-per-volt scale. There is one today, but it comes from the IQ
  derivatives measured at the bias point rather than from a fit — see
  `bias_finding.md`. The fitters produce the parameters a fitted version would
  need; nothing here computes one.
- **Choosing the operating amplitude, and writing it back to the catalog.**
  Not missing any more: section 7 gives you the data behind that decision, and
  `rfmux.tuning.find_bias_points` makes it. `bias_finding.md` is the notebook.
- **Fitting from Periscope.** The GUI currently fits inline during its own
  multisweep, using the deprecated functions in
  `rfmux.algorithms.measurement.fitting`. The plan is for this to become a
  button that calls `fit_sweeps` on a finished sweep; `fit_sweeps` already
  accepts a `progress_callback(completed, total)` for driving a progress bar.
- **Saving to disk.** `pickle.dump` on the fitted result does work today, since
  the whole thing is plain builtins and ndarrays including the `fits` subdicts.
  A proper `store.py` with a defined file layout, and somewhere sensible to
  record the fit settings from the report, is still to come.
