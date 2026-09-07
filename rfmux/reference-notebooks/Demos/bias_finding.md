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

Bias finding for a resonance involves determining both the ideal amplitude to
operate it at, and the ideal frequency at that amplitude to operate it at.
This generally involves measuring frequency sweeps of the resonator at different amplitudes,
and then making some decisions.

Once we have a range of frequency sweeps in hand, the process flow is generally:
- choose the amplitude
- choose the frequency
- take some calibration data at the chosen amplitude and frequency (e.g. for
converting data to df units later on)

There are lots of ways to identify these optimal points. rfmux provides some options,
which are covered in this document. The inputs for the rfmux methods are amplitude-iterated
multisweep data, along with a `ResonatorCatalog`.

The bias finding routines produce a **new** `ResonatorCatalog`. The catalog you
swept is not modified, and neither are the measured sweeps, so you can run the
analysis twice with different settings and compare the two resulting catalogs,
and the catalog you started from is still the catalog you started from.

The one thing that *does* change is the sweep result you hand in: the report is
written into it under `bias_report`. An operating point then travels with the
amplitude steps it was derived from, and saving updates that measurement's own
file rather than leaving a second one beside it.

| Piece | Module |
|---|---|
| The bias finding functions used below | `rfmux.tuning.bias` |
| The multi-amplitude multisweep data used to inform the bias finding routines | `rfmux.algorithms.measurement.multiamp_multisweep` |
| The `ResonatorCatalog` and related array bookkeeping | `rfmux.core.resonators` |
| Writing measurements to disk and reading them back | `rfmux.tuning.store` |

<!-- | Reading a sweep result back | `rfmux.tuning.sweep_results` | -->

This notebook starts from a multisweep that was measured earlier and saved to
disk, so it can get straight to the analysis. How that measurement is set up and
run is covered in `multisweep.md`, and how you get to a catalog in the first
place in `network_analysis_find_resonances.md`.

## How to use this document

**This is a runnable notebook, not a web page.** Every grey block below is a live
code cell: put the cursor in it and press **Shift+Enter** to execute it.

- **Run the cells in order, top to bottom.** Later cells use variables the
  earlier ones defined, so skipping ahead fails with a `NameError`. *Kernel →
  Restart Kernel and Run All Cells* starts clean.
- **The outputs you see are the ones you just produced.** This file is stored as
  jupytext markdown, which keeps no saved outputs, so a cell is blank until you
  run it. Nothing here can show you a stale number from someone else's run.
- **Editing is encouraged.** Change the spike factors, the methods, the
  discrepancy threshold, and re-run — that is what this document is for. The
  shipped copy is read-only, so *File → Save Notebook As…* to keep your changes.
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

from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import rfmux
from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.core.transferfunctions import BASE_FREQUENCY
from rfmux.tuning import store

# The module the sweep below came off. Nothing in this notebook needs a board,
# so this is only here for the sketches of board-side calls in sections 5 and 6.
MODULE = 2

# The recorded sweep this notebook analyses, found through the package rather
# than by a relative path, so it works whichever directory the kernel started
# in. Matched by pattern rather than named outright, because `store` puts the
# date and time of the writing into every filename it makes — your own sweeps
# land in `store.output_directory()` under names of exactly this shape.
DEMOS = Path(rfmux.__file__).parent / "reference-notebooks" / "Demos"
MULTISWEEP_PKL = max(DEMOS.glob("multiamp_multisweep_*_demo_biasfind1.pkl"))

print(MULTISWEEP_PKL)
```

## 1. Start with a previously-measured multisweep

Bias finding is purely analysis: it takes sweeps that already exist and works on
them. So rather than spend the first few minutes of this notebook taking a
measurement, we load a multi-amplitude sweep that was taken once and saved.

**This one is real data**, off a real array on a real board — not mock mode.
That matters more here than in the other notebooks: bias finding is entirely
about recognizing what a resonator does when you drive it too hard, and the
simulator's resonators do not bifurcate the way physical ones do. In particular
mock mode computes each sweep point independently, so its upward and downward
traces come out identical up to noise, and the hysteresis test in section 2
would have nothing to find.

The file is nothing special otherwise: it is an ordinary measurement file, the
kind `multiamp_multisweep` writes for itself, holding five resonators swept over
five amplitude steps from 0.0008 to 0.008, in both directions. Getting it was one
call, which saved itself into `store.output_directory()` on the way out, and
everything below this section is unchanged by the fact that it happened
yesterday rather than in the cell above:

    multiamp_ms = await crs.multiamp_multisweep(
        catalog,
        span_hz=75e3,
        npoints_per_sweep=101,
        nsamps=10,
        amp_schedule=AmplitudeSchedule.multiplicative(0.8, 8.0, 5),
        directions=("upward", "downward"),
    )

`store.load` is `pickle.load` plus one correction: the path the file recorded
about itself when it was written is replaced with where the file has actually
turned out to be. Demo data that shipped inside a package, or a sweep copied off
the acquisition machine — as this one was — can then still save itself back to
the file *you* opened rather than to a path on a computer you may not even be
on.

```python
multiamp_ms = store.load(MULTISWEEP_PKL)

# A sweep comes back keyed by module identifier, and every function below takes
# one module's value out of it. Ours only has the one.
print(f"modules: {list(multiamp_ms)}")

multiamp_module_results = multiamp_ms[list(multiamp_ms)[0]]

print(f"schema_version:  {multiamp_module_results['schema_version']}")
print(f"measurement:     {multiamp_module_results['measurement']}")
print(f"module:          {multiamp_module_results['module']}")
print(f"amplitude steps: {list(multiamp_module_results['results'])}")
print(f"directions:      {list(multiamp_module_results['results'][0])}")
print(f"resonators:      {list(multiamp_module_results['results'][0]['upward'])}")
```

Every saved measurement also carries a `file_metadata` block saying what it is,
when it was taken, what wrote it, and where it lives. It is stamped inside each
module's output rather than at the top of the file, so you reach it wherever
you happen to be already working:

```python
for key, value in multiamp_module_results["file_metadata"].items():
    print(f"{key:<18} {value}")
```

That `path` is what lets an analysis write its results back into the measurement
it read, without anyone having to carry a filename around — which is exactly
what `find_bias_points` does in section 5.

### The catalog that was swept

A sweep records the catalog it was given, under `call_params`, as a plain dict.
That is worth knowing about for two reasons: it is how you get the array
bookkeeping back out of a file weeks later, and it is what
`rfmux.tuning.find_bias_points` falls back on when you do not hand it a catalog
yourself.

```python
swept_catalog = ResonatorCatalog.from_dict(
    multiamp_module_results["call_params"]["catalog"]
)

print(swept_catalog)
```

<!-- #region -->
The array contains five resonators, which currently all have bias amplitudes listed as 0.001, which is the amplitude
that the resonance finding netanal was performed at.


### Take a look at the data

Here we demonstrate extracting and plotting the multiamp multisweep data. We'll draft the plotting
functions by hand as an exercise, but canned example
plotting functions can also be found under `Demos/example_plotting_{...}.py`, for the various
topics covered in these notebooks.

<!-- #endregion -->

```python

from rfmux.tuning import (
    collect_amplitude_iterations_for,
    get_amplitudes_at_iteration,
)

for iteration in multiamp_module_results["results"]:
    amplitudes = get_amplitudes_at_iteration(multiamp_module_results, iteration)
    print(f"step {iteration}: {amplitudes}")


AMPLITUDE_CMAP = LinearSegmentedColormap.from_list(
    "gnuplot_truncated", plt.cm.gnuplot(np.linspace(0.0, 0.9, 256))
)


def amplitude_colours(amplitudes):
    """One colour per amplitude, plus the mappable a colourbar needs.

    Log-scaled, because this schedule doubles at every step and a linear scale
    would bunch the quiet steps into one shade.
    """
    low, high = min(amplitudes), max(amplitudes)
    if high > low:
        norm = LogNorm(vmin=low, vmax=high)
        colours = [AMPLITUDE_CMAP(norm(a)) for a in amplitudes]
    else:
        norm = LogNorm(vmin=low * 0.9, vmax=low * 1.1)
        colours = [AMPLITUDE_CMAP(0.5)] * len(amplitudes)
    return colours, plt.cm.ScalarMappable(norm=norm, cmap=AMPLITUDE_CMAP)


def offset_khz(entry, frequencies=None):
    """Frequencies as kHz either side of where the sweep was centred.

    Pass *frequencies* to convert a grid other than the entry's own — the
    midpoint grid a point-to-point difference lands on, for instance.
    """
    if frequencies is None:
        frequencies = entry["frequencies"]
    return (frequencies - entry["original_center_frequency"]) / 1e3


def panels_per_resonator(names, width=3.1, height=3.0, **kwargs):
    """One panel per resonator, in a single row, and the figure holding them.

    Every plot below this point is per-resonator: a detector that is run on the
    whole array is worth seeing on the whole array, because what you are looking
    for is the one panel that does not look like the others.
    """
    fig, axes = plt.subplots(
        1, len(names), figsize=(width * len(names), height),
        constrained_layout=True, squeeze=False, **kwargs
    )
    return fig, axes[0]


def plot_amplitude_steps(results, resonator_names, directions=["upward", 'downward']):
    """Every amplitude step of each resonator, one panel per resonator."""
    fig, panels = panels_per_resonator(resonator_names)

    linestyles = ['--', '-']

    for panel, name in zip(panels, resonator_names):
        iterations = collect_amplitude_iterations_for(results, name)
        amplitudes = [e[directions[0]]["sweep_amplitude"] for e in iterations.values()]
        colours, mappable = amplitude_colours(amplitudes)

        for (entry, colour) in zip(iterations.values(), colours):
            for d, direction in enumerate(directions):
                sweep = entry[direction]
                # Divided by its own drive, so the traces can be compared by shape
                # rather than the loudest simply sitting on top of the others.
                iq = sweep["iq_counts"] / sweep["sweep_amplitude"]
                panel.plot(offset_khz(sweep), 20 * np.log10(np.abs(iq)),
                           linestyles[d],
                        lw=1.0, color=colour)

        panel.set_title(name, fontsize=10)
        panel.set_xlabel("offset from sweep centre [kHz]", fontsize=8)
        panel.tick_params(labelsize=8)

    panels[0].set_ylabel("|S21| / drive [dB]", fontsize=8)
    fig.colorbar(mappable, ax=list(panels), label="drive amplitude")
    plt.show()


resonator_names = list(multiamp_module_results["results"][0]["upward"])
plot_amplitude_steps(multiamp_module_results, resonator_names)
```

<!-- #region -->
This is a reasonably suitable measurement to use for our bias finding. Every
resonator has been swept at a low enough amplitude that it does not appear to be
perturbed by the readout current, and most of them have also been swept at a high
enough amplitude that they are clearly bifurcated. That brackets a reasonable bias
amplitude somewhere between the two end points.



## 2. Choosing the bias amplitude

Generally the idea is to use as much readout amplitude as the resonator will tolerate before
becoming seriously bifurcated, because using higher readout amplitudes raises the detector
signal above additive system noise sources, such as from the LNA.

To try to identify the best amplitude to use, we look at the sweeps to find the lowest-amplitude sweep which 
is bifurcated, and then select one amplitude step below that (the highest amplitude sweep
which is **not** bifurcated).

We will try two different methods to identify whether a sweep is bifurcated:
`derivative` and `hysteresis`. They look for different evidence of the same
thing, and they do not miss the same resonators, so the default is a third
option — `both` — which runs the two of them and calls a sweep bifurcated if
either one says so.

<!-- #endregion -->

rfmux provides `rfmux.tuning.find_bias_amplitude` to do this process on one
resonator at a time. Its arguments:

| Argument | Default | Does |
|---|---|---|
| `iterations` | required | multiamplitude multisweep measurements of a resonator in the usual form: `{iteration: {direction: entry}}`. This can be extracted using the convenience wrapper `rfmux.tuning.collect_amplitude_iterations_for` |
| `method` | `"both"` | which bifurcation detection method to apply. Options are: `"derivative"` (reads the shape of a single trace and looks for jumps), `"hysteresis"` (compares the two sweep directions against each other to see when they diverge), and `"both"` (runs the two and takes a step as bifurcated if either says so) |
| `spike_prominence_factor` | `0.5` | `"derivative"` and `"both"`: how far a spike has to stand out from its surroundings to count as a jump, as a multiple of the arc speed's range. Larger is less sensitive |
| `max_discrepancy` | `0.1` | `"hysteresis"` and `"both"`: how far the upward and downward traces may part company, in the units `compare` measures in, before the step is called bifurcated |
| `compare` | `"magnitude"` | `"hysteresis"` and `"both"`: which plane the two directions are compared in — `"magnitude"` for their `\|S21\|` against frequency, `"iq"` for their distance on the IQ plane |

`spike_prominence_factor`, `max_discrepancy` and `compare` are handed straight
down to whichever `method` was selected, so passing them all is harmless — the
test that has no use for a knob never sees it. `"both"` is the one method that
reads all three, since it runs both tests.

It needs to be called on measurements of one resonator at a time, so below we
demonstrate calling it on the first resonator in the array:

```python
from rfmux.tuning import collect_amplitude_iterations_for, find_bias_amplitude

iterations_of_BRUL = collect_amplitude_iterations_for(
    multiamp_module_results, "BRUL"
)
amplitude_choice = find_bias_amplitude(iterations_of_BRUL, method="derivative")

print(f"iteration:              {amplitude_choice.iteration}")
print(f"amplitude:              {amplitude_choice.amplitude}")
print(f"bifurcated_at:          {amplitude_choice.bifurcated_at}")
print(f"is_bifurcated_at_bias:  {amplitude_choice.is_bifurcated_at_bias}")
```

So: bifurcation was first seen at 0.008, and the amplitude below it — 0.0045,
step 3 — is where this resonator should sit. 

This also returns some of the checks that were done on the data, in an attempt to make it easier to 
troubleshoot why decisions were made. This is under `amplitude_choice.checks`.
Note that the search stops
at the first bifurcated step, so if there were further amplitude steps in the multisweep data,
they will not have any entries under checks.

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


### TODO plot each amplitude for this resonator, and label it with these checks rather than printing them
```

<!-- #region -->
Some quasi-failure modes:

- **If nothing bifurcates**, the loudest step is chosen. Sweep again and include higher
amplitudes. 
- **If the quietest step already bifurcates**, there is nothing below it to go
  back to. The quietest step is chosen,
  and `amplitude_choice.is_bifurcated_at_bias` is `True` . The schedule started too high.



### Two methods for detecting bifurcation: #1: `"derivative"`

This method tries to the detect the jump in the derivatives that is the hallmark of
a discontinuity from bifurcation. It computes `rfmux.tuning.normalized_arc_speed`:
how far the IQ 
trace moves per hertz from one sweep point to
the next. A smooth resonance gives a smooth bump in that quantity. A jump gives a spike.
The arc speed first divides I and Q each by their own range, which allows a single
threshold value to be meaningful on resonators of various depths.

`normalized_arc_speed` is exported so you can plot exactly what the test looked
at rather than a re-derivation of it. It takes one sweep entry and nothing else,
and hands back `(frequencies, speed)` — one point shorter than the sweep, on the
midpoints of the point pairs, because that is where a difference between two
points belongs.


<!-- #endregion -->

```python
from rfmux.tuning import normalized_arc_speed


def plot_derivative_test(results, names, direction="upward"):
    """What the derivative test differentiates, every resonator, every step.

    The arc speed itself is not drawn. The test does not read it directly — it
    differentiates it once more and looks for spikes in *that*, so the
    point-to-point change is the quantity a threshold means something against.
    """
    fig, panels = panels_per_resonator(names, height=3.2, sharey=True)

    for panel, name in zip(panels, names):
        iterations = collect_amplitude_iterations_for(results, name)
        amplitudes = [e[direction]["sweep_amplitude"] for e in iterations.values()]
        colours, mappable = amplitude_colours(amplitudes)

        for entry, colour in zip(iterations.values(), colours):
            sweep = entry[direction]
            frequencies, speed = normalized_arc_speed(sweep)
            # The difference between two points belongs between them, so its
            # x-axis is the midpoints of the arc speed's own grid.
            midpoints = 0.5 * (frequencies[:-1] + frequencies[1:])
            panel.plot(offset_khz(sweep, midpoints), np.diff(speed),
                       lw=1.0, color=colour)

        panel.set_title(name, fontsize=10)
        panel.set_xlabel("offset from sweep centre [kHz]", fontsize=8)
        # Symmetric log, so the quiet steps are not a flat line beside the loud
        # ones — the spikes here are two orders of magnitude apart.
        panel.set_yscale("symlog", linthresh=1e-5)
        panel.tick_params(labelsize=8)

    panels[0].set_ylabel("point-to-point change in arc speed", fontsize=8)
    fig.colorbar(mappable, ax=list(panels), label="drive amplitude")
    fig.suptitle("What the derivative test looks at", fontsize=11)
    plt.show()


plot_derivative_test(multiamp_module_results, resonator_names)

### TODO show the line for the threshold on these plots
```

<!-- #region -->
This is the quantity the test actually reads — effectively the second derivative
of I and Q with frequency. At low amplitudes the change from point to point is
small and featureless. At the loudest step, four of the five panels grow a sharp
positive spike with a negative spike immediately after it: the trace jumping onto
the other state and dropping back off it again.

**Two spikes, adjacent, first positive then negative** is the pattern
`rfmux.tuning.bifurcated_by_derivative` cues off of.


`find_bias_amplitude` calls `bifurcated_by_derivative` on each amplitude step, 
in both directions (if present). The amplitude step is counted as bifurcated if either
direction triggers the threshold. You can also call it yourself, on one step at
a time, which is how you work out what the factor should be. Its whole
argument list:

| Argument | Default | Does |
|---|---|---|
| `entries` | required | one amplitude step, `{direction: entry}` — one value out of what `collect_amplitude_iterations_for` returns. Every direction present is tested, and the step counts as bifurcated if any of them says so: a bifurcated resonator jumps whichever way the sweep runs, so needing both to agree would only lose the one that happened to catch it |
| `spike_prominence_factor` | `0.5` | a spike must stand out from its surroundings by more than this factor times the full range of the arc speed. It multiplies, so larger asks for a bigger spike: larger is less sensitive |

So the default of `0.5` asks a spike to stand a full half of the arc speed's
range out of its own neighbourhood. The bar is set relative to the sweep
itself, which allows a single number to apply to multiple resonators and measurements.


<!-- #endregion -->

```python
from rfmux.tuning import bifurcated_by_derivative

for iteration in (3, 4):
    check = bifurcated_by_derivative(iterations_of_BRUL[iteration])
    print(f"step {iteration}: {check}")
```

Here is that threshold drawn on the data, computed the way the detector computes it,
for every resonator — on the two steps that decided its answer:

```python
SPIKE_PROMINENCE_FACTOR = 0.5


def plot_prominence_bar(results, names, direction="upward", spike_prominence_factor=SPIKE_PROMINENCE_FACTOR):
    """The bar each verdict was read off, on the steps that settled it.

    Two steps per panel: the one chosen as the bias amplitude, and the first one
    called bifurcated. Each carries its own bar, because the bar is a fraction of
    that sweep's own arc speed range rather than one number for the array — which
    is what makes a single `spike_prominence_factor` portable between resonators.

    One bar, drawn positive. Both spikes are held to it — the up-spike and the
    down-spike after it — so the same line mirrored is what the trough below is
    judged against. Even then it is a guide rather than the literal comparison:
    the detector measures a spike's *prominence*, its height above its own
    neighbourhood, while the trace here is drawn from zero. The margins in the
    legend are the comparison, and there are two of them because there are two
    spikes.

    A resonator that never bifurcated has no second step to draw, and the empty
    half of its panel is the finding.
    """
    fig, panels = panels_per_resonator(names, width=3.3, height=3.6, sharey=True)

    for panel, name in zip(panels, names):
        iterations = collect_amplitude_iterations_for(results, name)
        choice = find_bias_amplitude(iterations, method = "derivative",
            spike_prominence_factor = spike_prominence_factor )

        steps = [(choice.iteration, "0.35", "chosen")]
        bifurcated_at = next(
            (i for i, c in choice.checks.items() if c.bifurcated), None
        )
        if bifurcated_at is not None:
            steps.append((bifurcated_at, "tab:red", "bifurcated"))

        for iteration, colour, role in steps:
            entry = iterations[iteration][direction]
            frequencies, speed = normalized_arc_speed(entry)
            midpoints = 0.5 * (frequencies[:-1] + frequencies[1:])

            # Exactly what bifurcated_by_derivative computes before find_peaks.
            # The factor this was called with, not the module default — the
            # labels come from a search run at it, so bars drawn at anything
            # else would put margins under 1.0 beside the word "bifurcated".
            bar = spike_prominence_factor * (speed.max() - speed.min())
            check = bifurcated_by_derivative(
                {direction: entry},
                spike_prominence_factor=spike_prominence_factor,
            )

            up = check.metric["positive_spike_prominence"] / check.threshold
            down = check.metric["negative_spike_prominence"] / check.threshold
            panel.plot(offset_khz(entry, midpoints), np.diff(speed),
                       lw=1.0, color=colour,
                       label=f"step {iteration} ({role})\n"
                             f"margin = {up:.2f} up, {down:.2f} down\n"
                             f"adjacent = {check.metric['adjacency']}")
            panel.axhline(bar, color=colour, ls="--", lw=1.0)
            panel.axhline(-bar, color=colour, ls="--", lw=1.0)
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


plot_prominence_bar(multiamp_module_results, resonator_names)
```

<!-- #region -->
The amplitude step that is chosen for the bias (grey) does have a visible spike but does not clear its
dashed bar. This is classified as "not bifurcated" by the search. The red traces have been
classified as bifurcated.

Note that `MELL` has no red trace, because it was not found to be bifurcated even at the
highest amplitude used in the sweep. As a result, its bias amplitude has simply been chosen
to be the highest amplitude used, for lack of a better option.
However, it does appear quite strongly driven
at the highest amplitude, and so perhaps we should adjust our threshold.

 

Note: `rfmux.tuning.BifurcationCheck` also reports the numbers it compared as well
as its verdict based on them, to facilitate troubleshooting. These include:

| Field | Is |
|---|---|
| `method` | which test produced this: `"derivative"`, `"hysteresis"` or `"both"` |
| `bifurcated` | the verdict |
| `metric` | a dict, one entry per quantity the method examined — see below |
| `threshold` | the single bar those quantities were held to. For `"derivative"`: `spike_prominence_factor` times the arc speed's range |
| `parts` | empty, unless this check combined several tests — `"both"` keeps each test's own check in here, raw numbers and own threshold |

`metric` is a dict rather than one number because the verdict is not one
comparison. `"derivative"` asks three things, and reports all three:

| Key | Is |
|---|---|
| `positive_spike_prominence` | how far the tallest up-spike stands out of its own neighbourhood. `0.0` if there is no up-spike at all |
| `negative_spike_prominence` | the same for the tallest down-spike |
| `adjacency` | whether the spikes that cleared the bar sat next to each other, up first. A condition, with no threshold of its own |

The verdict is `True` when both prominences clear `threshold` **and**
`adjacency` — so these three entries tell you which condition decided it. 


If we now re-run the bias amplitude identification with a slightly more sensitive threshold:


<!-- #endregion -->

```python
plot_prominence_bar(multiamp_module_results, resonator_names, spike_prominence_factor=0.45)
```

Now `MELL` also is flagged as bifurcated.

In general, you may have to fiddle with the various thresholds and other bias finding settings for a given array.


<!-- #region -->


### Bifurcation detection method #2 `"hysteresis"`

The other test, `rfmux.tuning.bifurcated_by_hysteresis`, looks for the amplitude at which 
the upward and downward frequency sweeps *begin* to differ.

We can do the comparison on either the complex IQ plane, or on magnitude vs frequency.
Generally the magnitude vs frequency test tends to be more robust, and so it is the default.

| Argument | Default | Does |
|---|---|---|
| `entries` | required | one amplitude step, `{direction: entry}`, where `"upward"` and `"downward"` are required |
| `max_discrepancy` | `0.1` | how far apart the two traces may be, in whatever units `compare` measures in, before the step is called bifurcated |
| `compare` | `"magnitude"` | the quantity being compared: `"iq"` for the distance on the IQ plane in loop radii, and `"magnitude"` for the difference in `\|S21\|` in dip depths |

The bifurcation detector reports `metric["max_separation"]`: the
**largest separation between the two traces, in units of the trace's own
scale**.
<!-- #endregion -->

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
        iterations = collect_amplitude_iterations_for(results, name)
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
            top.plot(offset_khz(entry, frequencies), traces[direction][1],
                     style, lw=1.2, label=direction)

        (f_up, up), (f_down, down) = traces["upward"], traces["downward"]
        difference = np.abs(up - np.interp(f_up, f_down, down)) / np.ptp(up)
        bottom.plot(offset_khz(amplitude_step["upward"], f_up), difference,
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


plot_magnitude_hysteresis(multiamp_module_results, resonator_names,
                          max_discrepancy=0.1)
```

The lower row shows how close each measurement is to the chosen threshold (dashed line).
Interestingly, although MELL and LALM both look over-driven by eye in the |S21| vs freq
data, they jump at almost the same point in both directions, and thus don't trigger this
method of bifurcation detection.

This emphasizes the importance of using multiple methods to attempt to identify bifurcation
and a good bias amplitude. Combining the hysteresis method with the derivative method,
and later on by fitting to the chosen bias points, we should be able to get a decent result.

<!-- #region -->


### Bifurcation detection method #3: `"both"`

To increase our odds of detecting resonance bifurcation, the default method is
`rfmux.tuning.bifurcated_by_either`. It runs the two tests
on every amplitude step and calls the step bifurcated if **either** of them
says so. 

It compares the two directions, so like `"hysteresis"` you need to provide both.

| Argument | Default | Does |
|---|---|---|
| `entries` | required | one amplitude step, `{direction: entry}`, with both `"upward"` and `"downward"` present |
| `spike_prominence_factor` | `0.5` | handed to `rfmux.tuning.bifurcated_by_derivative` |
| `max_discrepancy` | `0.1` | handed to `rfmux.tuning.bifurcated_by_hysteresis` |
| `compare` | `"magnitude"` | handed to `rfmux.tuning.bifurcated_by_hysteresis` |


<!-- #endregion -->

<!-- #region -->


## 3. Selecting a bias frequency

Once the bias amplitude has been chosen, we can choose the bias frequency to use
at that amplitude.

`rfmux.tuning.find_bias_frequency` takes two arguments:

| Argument | Default | Does |
|---|---|---|
| `entry` | required | **one** sweep, as `multisweep` returns it — a single direction of a single amplitude step, not the `{direction: entry}` mapping the bifurcation tests take.  |
| `method` | `"iq_derivative"` | how to decide where in that trace to put the tone |

There are two methods available for deciding at what frequency to bias, within the given sweep trace:

| `method` | Puts the tone at | Because |
|---|---|---|
| `"iq_derivative"` (the default) | maximum `\|dI/df + j·dQ/df\|` | where the IQ trace moves fastest per hertz, so a small shift in the resonance makes the largest signal  |
| `"minimum"` | minimum `\|S21\|` | the bottom of the dip. Survives traces the derivative method finds noisy |

**Note** that both methods return a point on the measured grid. 
<!-- #endregion -->

```python
from rfmux.tuning import find_bias_frequency, iq_arc_speed

chosen_sweep = iterations_of_BRUL[amplitude_choice.iteration]["upward"]
chosen_sweep_centre = chosen_sweep["original_center_frequency"]

for method in ("iq_derivative", "minimum"):
    frequency = find_bias_frequency(chosen_sweep, method=method)
    print(f"{method:<14} {frequency/1e6:.6f} MHz "
          f"({(frequency - chosen_sweep_centre)/1e3:+.2f} kHz from the sweep centre)")
```

`rfmux.tuning.iq_arc_speed` reads back the quantity the default method maximizes
— it takes a sweep entry and nothing else — so you can see what it picked and
why:

```python
frequencies, speed = iq_arc_speed(chosen_sweep)

bias_frequency_by_derivative = find_bias_frequency(chosen_sweep)
bias_frequency_by_minimum = find_bias_frequency(chosen_sweep, method="minimum")

fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True,
                         constrained_layout=True)

axes[0].plot(offset_khz(chosen_sweep),
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
fig.suptitle(f"BRUL at chosen bias amplitude {amplitude_choice.amplitude}: bias frequency selection",
             fontsize=11)
plt.show()
```

The two answers will generally be close but not identical.

Across the array, each resonator read at *its own* chosen amplitude — which for
`MELL` is the loudest step rather than one below a limit, so its trace is the
distorted one:

```python
def plot_frequency_methods(results, names, direction="upward"):
    """Where each method puts the tone, for every resonator.

    Drawn on the arc speed rather than on |S21|, because that is the quantity the
    default method maximizes: the red line should sit on the peak of the trace
    beneath it, and if it does not, that is the thing to chase.
    """
    fig, panels = panels_per_resonator(names, width=3.2, height=3.4)

    for panel, name in zip(panels, names):
        iterations = collect_amplitude_iterations_for(results, name)
        choice = find_bias_amplitude(iterations)
        entry = iterations[choice.iteration][direction]

        frequencies, speed = iq_arc_speed(entry)
        panel.plot(offset_khz(entry, frequencies), speed, ".-", lw=1.0, ms=3,
                   color="0.2")

        for method, colour, style in (("iq_derivative", "tab:red", "-"),
                                      ("minimum", "tab:blue", "--")):
            frequency = find_bias_frequency(entry, method=method)
            panel.axvline(offset_khz(entry, frequency), color=colour, ls=style,
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


plot_frequency_methods(multiamp_module_results, resonator_names)
```

The two methods land within a point or two of each other on all five, so on this
array the choice between them barely matters. What does vary is the shape they are
choosing from. `BRUL` and `LALM` have broad, rounded peaks: the maximum is a
region, and putting the tone a point either side of it costs almost nothing.

`MELL`'s peak is four times taller than anyone else's and only a couple of points
wide — the arc speed at a near-discontinuity, which is what its trace at the
loudest drive has become. Both methods correctly find the top of it. The trouble
is that the top of a spike that narrow is a fragile place to sit: the resonance
only has to drift slightly for the tone to be somewhere much less sensitive.
Nothing in `find_bias_frequency` will tell you that, because it answered the
question it was asked.


<!-- #region -->


## 4. Making a `BiasPoint` for each `Resonator` in the `ResonatorCatalog`

Each resonator in the catalog must have a 
`rfmux.core.resonators.BiasPoint`, which comprises (at a minimum) a
frequency and an amplitude.


### base frequency quantization

For intermodulation distortion reasons that are outside the scope of this demo, 
we only synthesize bias tones at integer multiples of a base frequency. This
quantization is applied automatically when adding bias points to the catalog.
<!-- #endregion -->

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

<!-- #region -->
**Note that the number that is stored in the Bias Point is the true frequency that the board will output.**

### Measure I and Q derivatives with frequency to apply a conversion to df units

Finally, we measure the IQ derivatives **at
the quantized bias frequency**, so that the calibration is accurate at the true
operational frequency. 

`rfmux.tuning.iq_derivatives_at` extracts these derivatives based on the multisweep
entry at the chosen bias amplitude.


| Argument | Default | Does |
|---|---|---|
| `entry` | required | the same single multisweep trace `find_bias_frequency` reads |
| `frequency_hz` | required | where along that trace to evaluate the slopes. Since the true bias frequency may not be a measured point in the sweep, it uses splines to interpolate |
<!-- #endregion -->

```python
from rfmux.tuning import iq_derivatives_at

dI_df, dQ_df = iq_derivatives_at(chosen_sweep, bias_point.frequency_hz)

print(f"dI_df  {dI_df:+.4e} V/Hz")
print(f"dQ_df  {dQ_df:+.4e} V/Hz")

print(bias_point)
```

These are slopes: how many volts of I, and of Q, you get per hertz the resonance
moves. Here they are drawn as tangents on the measured trace:

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
    panel.plot(offset_khz(chosen_sweep), measured, ".-", lw=1.0, ms=3,
               color="0.2", label="measured")
    panel.plot((tangent_frequencies - chosen_sweep_centre) / 1e3,
               at_bias + slope * (tangent_frequencies - bias_frequency),
               lw=2.0, color="tab:red", label=f"d{label}/df = {slope:+.2e} V/Hz")
    panel.plot((bias_frequency - chosen_sweep_centre) / 1e3, at_bias, "o",
               color="tab:red", ms=6)
    panel.set_xlabel("offset from sweep centre [kHz]", fontsize=8)
    panel.set_ylabel(f"{label} [V]", fontsize=8)
    panel.set_xlim(-15, 10)
    # The tangent is steep enough to leave the plot in under a linewidth, which
    # is the point of it — but the measured trace is what should set the scale.
    margin = 0.1 * np.ptp(measured)
    panel.set_ylim(measured.min() - margin, measured.max() + margin)
    panel.legend(fontsize=8)

fig.suptitle("The slopes used in the calibration", fontsize=11)
plt.show()
```

The same movement for every resonator, each on its own loop at its own chosen
amplitude and frequency:

```python
ARRAY_MOVEMENT_HZ = 500.0


def plot_calibration_arrows(results, names, direction="upward",
                            movement_hz=ARRAY_MOVEMENT_HZ):
    """The measured slopes, as an arrow on each resonator's own IQ loop.

    Repeats what sections 2 to 4 did by hand, for every resonator: choose the
    amplitude, choose the frequency on that step, quantize it, then read the
    slopes at the frequency the board will actually output.

    Each loop is drawn in units of its own radius rather than in volts. A loop
    measured at a louder drive is simply a bigger loop, and that is a fact about
    the drive rather than about the resonator — dividing it out is what makes the
    arrows comparable from panel to panel.
    """
    fig, panels = panels_per_resonator(names, width=3.0, height=3.4)

    for panel, name in zip(panels, names):
        iterations = collect_amplitude_iterations_for(results, name)
        choice = find_bias_amplitude(iterations)
        entry = iterations[choice.iteration][direction]

        bias_frequency = BiasPoint(frequency_hz=find_bias_frequency(entry),
                                   amplitude=choice.amplitude).frequency_hz
        dI, dQ = iq_derivatives_at(entry, bias_frequency)

        volts = entry["iq_volts"]
        centre = complex(np.mean([volts.real.min(), volts.real.max()]),
                         np.mean([volts.imag.min(), volts.imag.max()]))
        radius = 0.5 * max(np.ptp(volts.real), np.ptp(volts.imag))

        loop = (volts - centre) / radius
        at_bias = complex(
            np.interp(bias_frequency, entry["frequencies"], volts.real),
            np.interp(bias_frequency, entry["frequencies"], volts.imag),
        )
        at_bias = (at_bias - centre) / radius

        panel.plot(loop.real, loop.imag, ".-", lw=1.0, ms=2, color="0.2")
        panel.plot(at_bias.real, at_bias.imag, "o", color="tab:red", ms=6)

        tip = at_bias + complex(dI, dQ) * movement_hz / radius
        panel.plot(tip.real, tip.imag, ".", alpha=0)  # keeps the arrow in frame
        panel.annotate("", xytext=(at_bias.real, at_bias.imag),
                       xy=(tip.real, tip.imag),
                       arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0))

        calibration = abs(BiasPoint(frequency_hz=bias_frequency,
                                    amplitude=choice.amplitude,
                                    dI_df=dI, dQ_df=dQ).df_calibration)
        panel.set_title(f"{name}\n|df_cal| = {calibration/1e6:.3f} MHz/V",
                        fontsize=9)
        panel.set_xlabel("I  [loop radii]", fontsize=8)
        panel.set_aspect("equal")
        panel.tick_params(labelsize=7)

    panels[0].set_ylabel("Q  [loop radii]", fontsize=8)
    fig.suptitle(f"{movement_hz:.0f} Hz of movement, at each resonator's bias "
                 f"point", fontsize=11)
    plt.show()


plot_calibration_arrows(multiamp_module_results, resonator_names)
```

Four of the arrows are short and one is not. `MELL` responds with several times
the reading per hertz that its neighbours do — its `|df_calibration|` is four
times smaller than the nearest of them, and smaller means more volts per hertz of
resonance movement.

That is not the array's best detector. It is the resonator biased at 0.008,
reading its slope off the near-vertical section of a trace that is on the point of
jumping. The number is a correct measurement of that trace, and the trace only has
that slope for a kilohertz or two either side; move the resonance past the jump
and none of it applies. A large `df_calibration` measured at a flagged bias point
is a reason to look at the flag, not a reason to be pleased.

`LALM`, at the other end, has the shortest arrow and the largest
`|df_calibration|` — an ordinary, well-behaved operating point on a broad
resonance, which is what most of these should look like.

### `df_calibration`

The `BiasPoint` saves the derivatives and computes:

```text
df_calibration = 1 / (dI_df + j·dQ_df)     Hz/V
```

This factor will be used later to transform voltage data
for this resonator **at this bias point** into df units.

**Note how, clearly, if we update either the bias frequency or the bias
amplitude, this calibration data will no longer be valid! And thus we would
need to remeasure it. This is why we enforce that changing either of those
parameters requires generating a whole new `BiasPoint`.**

This bookkeeping lock is currently also applied to the df calibration numbers, so 
to manually update them for our existing bias_point we must in fact generate a new bias point,
which will use our existing one and add our new df calibration numbers to it:

```python
bias_point = replace(bias_point, dI_df=dI_df, dQ_df=dQ_df)

print(bias_point)
print(f"\ndf_calibration   {bias_point.df_calibration}")
print(f"|df_calibration| {abs(bias_point.df_calibration)/1e6:.3f} MHz/V")
print(f"so 1 µV along the arrow above is "
      f"{abs(bias_point.df_calibration) * 1e-6:.2f} Hz of resonance movement")
```

<!-- #region -->


## 5. All-in-one: `find_bias_points`

In the above sections we stepped through the bias finding routines manually to explore
how they worked. This is a useful exercise, and to determine whether the values you're using
for a particular array, it may be helpful to work through them manually again later.

However, we also provide a Jesus-take-the-wheel function:
`rfmux.tuning.find_bias_points`, which
runs the same bias finding steps over every resonator in a catalog and assembles the
result.


| Argument | Default | Does |
|---|---|---|
| `sweeps` | required | **one module's** `multiamp_multisweep` outputs, i.e. `multiamp_ms[crs.module[MODULE].index()]` |
| `catalog` | `None` | the resonators to bias, which must match the catalog used to make the above multisweeps. `None` uses the one recorded in the sweep's `call_params`, which is the usual case. |
| `amplitude_method` | `"both"` | which bifurcation test the amplitude search uses — section 2. The default and `"hysteresis"` require the sweeps to have been taken in both directions; `"derivative"` is the one that reads a single trace. |
| `frequency_method` | `"iq_derivative"` | what method to use to determine what frequency to bias at — section 3 |
| `direction` | `None` | which sweep direction to measure the bias frequency and the calibration on. `None` prefers `"upward"`. |
| `spike_prominence_factor` | `0.5` | passed to `rfmux.tuning.bifurcated_by_derivative` — section 2 |
| `max_discrepancy` | `0.1` | passed to `rfmux.tuning.bifurcated_by_hysteresis` — section 2 |
| `compare` | `"magnitude"` | passed to `rfmux.tuning.bifurcated_by_hysteresis` — section 2 |
| `max_distance_hz` | `None` | how far from the sweep centre a resonance may come out before the bias frequency is rejected. Past this, the tone is left where the sweep was centred and the finding is flagged. Useful for handling densely packed arrays or collisions. |
| `save` | `None` | write the sweeps — which now carry the report — back to the file they came from. `None` does whatever `rfmux.tuning.store.autosave_enabled()` says, which is on unless you turned it off. Sweeps that have never been in a file get a new one |
| `label` | `None` | your name for that file, used only when these sweeps are being written for the first time. A re-save keeps the name the file already has |

The last three go to whichever test `amplitude_method` selected, and
`"both"` — running both tests — is the one that reads all three.

We pass `save=False` below for one reason that has nothing to do with bias
finding: this notebook's ladder is the demo file that ships inside the rfmux
package, and re-saving it would edit the copy every other reader gets. Your own
sweeps came out of your own measurement, so leave `save` alone and the report
lands in the file beside the data it describes.
<!-- #endregion -->

```python
from rfmux.tuning import find_bias_points

bias_report = find_bias_points(multiamp_module_results, save=False)

print(bias_report)
print(bias_report.catalog)

```

No catalog was passed, so it used the one recorded in the sweep's `call_params`
— the array that was swept, which is nearly always the array you want to bias.
Pass `catalog=` to override that.

**note that `bias_report.catalog`** is a new `ResonatorCatalog`, not the one the function started with from `call_params`:

```python
print(bias_report.catalog)
```

```python
print(f"{'name':<7}{'amplitude':>22}{'bias frequency':>28}")
for before, after in zip(swept_catalog, bias_report.catalog):
    print(f"{after.name:<7}"
          f"{before.bias.amplitude:>10.4f} → {after.bias.amplitude:<9.4f}"
          f"{before.bias.frequency_hz/1e6:>13.6f} → "
          f"{after.bias.frequency_hz/1e6:.6f} MHz"
          f"  ({(after.bias.frequency_hz - before.bias.frequency_hz)/1e3:+.2f} kHz)")
```


The catalog that went in is untouched, as are the measured sweeps. This allows the
analysis to be re-run with different settings on the same data as many times as
you like:

```python
print(f"the catalog we started from is still: {swept_catalog['BRUL'].bias}")

```

The sweep result itself is a different matter: the report goes into it under
`bias_report`, so that an operating point travels with the amplitude steps it
was derived from. It is stored as plain builtins rather than as the class, which
is what keeps the file readable by anything with `pickle` and outlives a rename
of `BiasReport`. Re-running replaces it, the same way re-running a fit replaces
that fit:

```python
from rfmux.tuning import BiasReport

print(BiasReport.from_dict(multiamp_module_results["bias_report"]))

```

That happens whether or not you save. `save=` is only the question of whether
the file on disk is brought up to date to match — and had we left it alone here,
this would have rewritten the `multiamp_multisweep_*_demo_biasfind1.pkl` the
notebook loaded, in place, report and all. That is the point of it: the ladder
and the operating point read off it stay one file.

### The report

The function also returns one
`rfmux.tuning.BiasFinding` per resonator, which says how that resonator's bias
point was
arrived at:

```python
finding_for_BRUL = bias_report["BRUL"]

print(f"name           {finding_for_BRUL.name}")
print(f"iteration      {finding_for_BRUL.iteration}")
print(f"amplitude      {finding_for_BRUL.amplitude}")
print(f"bifurcated_at  {finding_for_BRUL.bifurcated_at}")
print(f"frequency_hz   {finding_for_BRUL.frequency_hz}")
print(f"dI_df, dQ_df   {finding_for_BRUL.dI_df:.4e}, {finding_for_BRUL.dQ_df:.4e}")
print(f"good           {finding_for_BRUL.good}")
print(f"flagged_because {finding_for_BRUL.flagged_because}")
print(f"\nchecks         {list(finding_for_BRUL.checks)}")
```

```python
print(f"{'name':<7}{'step':>6}{'amplitude':>12}{'bif at':>10}"
      f"{'bias freq [MHz]':>18}{'|df_cal| [MHz/V]':>19}")
for f in bias_report.findings:
    df_calibration = bias_report.catalog[f.name].bias.df_calibration
    # `bifurcated_at` is None when no step bifurcated, which is a resonator to
    # read the flags on rather than a number to format — MELL is the one here.
    bifurcated_at = "—" if f.bifurcated_at is None else f"{f.bifurcated_at:.4f}"
    print(f"{f.name:<7}{f.iteration:>6}{f.amplitude:>12.4f}"
          f"{bifurcated_at:>10}{f.frequency_hz/1e6:>18.6f}"
          f"{abs(df_calibration)/1e6:>19.3f}")
```

`bias_report.flagged` and `bias_report.good` split the findings by whether the
answer is something the amplitude steps established or a default that was
fallen back to. It's a good idea to read this list before trusting and applying the biases.

```python
print(f"biased:  {len(bias_report)}")
print(f"good:    {len(bias_report.good)}")
print(f"flagged: {len(bias_report.flagged)}")
```

Four are good and one is flagged: `MELL`, the resonator whose up-spike came 1.5%
under the threshold at the loudest drive back in section 2. Its bias amplitude is 0.008 because
that is the loudest step measured, not because anything established 0.008 as its
limit — and `bifurcated_at` is `None` in the table above for exactly that reason.
Nothing else in the report distinguishes it from a resonator that was genuinely
measured, which is what the flag is for.

That is a bias point you can use, incidentally. It is just one you should decide
to use, having read that it is a floor rather than a finding — the right response
being another ladder that goes louder.

Running the hysteresis detector on its own flags `LALM` as well, since that test
saw nothing on it at any drive either — and it is the default's derivative half
that caught it above:

```python
print(find_bias_points(multiamp_module_results,
                       amplitude_method="hysteresis", save=False))
```

`MELL` is flagged whichever way it is run, which is the honest answer: no
combination of tests can find a limit in a ladder that never reached one.

Each flagged finding carries the sentence in `flagged_because`, so what you read
here is per resonator and specific — not a bit that says something went wrong
somewhere.

### The whole answer, on the whole measurement

Everything the report decided, drawn on the sweeps it decided it from. One panel
per resonator: the full ladder colour-coded by drive, the chosen step picked out
in bold, and the chosen frequency marked on it.

```python
def plot_bias_points_on_sweeps(results, report, direction="upward"):
    """Every sweep in the file, with the operating point read off it.

    The chosen amplitude is the trace drawn in bold; the quieter and louder steps
    stay thin behind it. The marker is the bias frequency, at the amplitude it was
    measured at — the pair is the answer, and neither half means much alone.

    Traces are *not* divided by their drive here, unlike the overview in section
    1. The point of this plot is where the tone ends up on the sweep it was
    chosen from, so the sweeps are left as they were measured.
    """
    names = [finding.name for finding in report.findings]
    fig, panels = panels_per_resonator(names, width=3.2, height=3.5)

    for panel, finding in zip(panels, report.findings):
        iterations = collect_amplitude_iterations_for(results, finding.name)
        amplitudes = [e[direction]["sweep_amplitude"] for e in iterations.values()]
        colours, mappable = amplitude_colours(amplitudes)

        for iteration, colour in zip(iterations, colours):
            sweep = iterations[iteration][direction]
            chosen = iteration == finding.iteration
            panel.plot(offset_khz(sweep),
                       20 * np.log10(np.abs(sweep["iq_counts"])),
                       lw=2.0 if chosen else 0.8,
                       alpha=1.0 if chosen else 0.55,
                       color=colour, zorder=3 if chosen else 2)

        chosen_sweep = iterations[finding.iteration][direction]
        depth_at_bias = np.interp(
            finding.frequency_hz, chosen_sweep["frequencies"],
            20 * np.log10(np.abs(chosen_sweep["iq_counts"])),
        )
        # Ringed in the flag's colour, so a bias point that is a fallback rather
        # than a finding is visible here and not only in the printed report.
        panel.plot(offset_khz(chosen_sweep, finding.frequency_hz), depth_at_bias,
                   "o", ms=9, zorder=4, color="white",
                   mec="tab:red" if finding.good else "darkorange", mew=2.0)

        panel.set_title(
            f"{finding.name}{'' if finding.good else '  (flagged)'}\n"
            f"amp {finding.amplitude:.4f}, "
            f"{offset_khz(chosen_sweep, finding.frequency_hz):+.2f} kHz",
            fontsize=9,
        )
        panel.set_xlabel("offset from sweep centre [kHz]", fontsize=8)
        panel.tick_params(labelsize=7)

    panels[0].set_ylabel("|S21| [dB]", fontsize=8)
    fig.colorbar(mappable, ax=list(panels), label="drive amplitude")
    fig.suptitle("The chosen bias point, on the ladder it was chosen from",
                 fontsize=11)
    plt.show()


plot_bias_points_on_sweeps(multiamp_module_results, bias_report)
```

This is the figure to keep. Everything the notebook worked out by hand is in it,
per resonator, on the data it came from: which step was chosen out of the ladder,
where on that step the tone goes, and whether the answer was measured or fallen
back to.

Four panels show the same thing — the second-loudest step in bold, with the tone a
little way down the steep flank of it and below the sweep centre, because a harder
drive has already pulled the resonance down. That is what a bias point is supposed
to look like.

`MELL`'s panel is titled `(flagged)`, its marker ringed in orange rather than red,
and its bold trace is the loudest in the ladder rather than one step below a
limit. The marker sits on a nearly vertical edge. Nothing about that is a
malfunction — every routine in this notebook did what it was asked — but it is a
tone placed on the side of a cliff, and one glance at this figure says so where
five printed tables did not.

<!-- #region -->
## 6. Applying the bias points

Once you have a catalog that you are happy with, you can set those tones on the array
using:

    await crs.apply_bias(bias_report.catalog)

This function returns nothing.

However, it will fail if the catalog contains resonators whose frequencies
span more than one NCO bandwidth. 



## 7. What is not here yet

- **IQ rotation.** `BiasPoint` has a field for it, and bias finding leaves it
  alone: the angle comes from a timestream rather than from a sweep, so it is
  not this step's to measure.
- **The fitted `fr` as a bias-frequency method.** `rfmux.tuning.fit_sweeps`
  already produces
  `fr` for every sweep, and `fitting_resonators.md` covers it. Wiring it in as a
  third `frequency_method` is a small job; the methods take a whole sweep entry
  rather than two arrays precisely so that one of them can read the entry's
  `fits`.
- **Thresholds that have met more than one array.** `spike_prominence_factor` is
  the GUI's bar restated as a multiplication, and `max_discrepancy` is a round
  number picked to sit between the quiet steps and the jumped ones on the array
  above. There they get four of the five resonators, with
  a margin of about a factor of two either side — which is not much to spend on a
  different array with a different noise floor, and `MELL` is what running out of
  it looks like. Read `metric` and `threshold` across your own amplitude steps,
  the way section 2 does, before trusting the defaults on them. A ladder that
  reaches high enough for every resonator to bifurcate is the other half of the
  answer, and the cheaper half.




<!-- #endregion -->


