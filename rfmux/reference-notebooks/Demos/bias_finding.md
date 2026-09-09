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
| The multi-amplitude multisweep data used to inform the bias finding routines | `rfmux.algorithms.measurement.multisweep` |
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
# land in `store.session_directory()` under names of exactly this shape.
DEMOS = Path(rfmux.__file__).parent / "reference-notebooks" / "Demos"
MULTISWEEP_PKL = max(DEMOS.glob("multisweep_*_demo_biasfind1.pkl"))

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
kind `multisweep` writes for itself, holding five resonators swept over
five amplitude steps from 0.0008 to 0.008, in both directions. Getting it was one
call, which saved itself into `store.session_directory()` on the way out, and
everything below this section is unchanged by the fact that it happened
yesterday rather than in the cell above:

    multi_amplitude_ms = await crs.multisweep(
        catalog,
        span_hz=75e3,
        npoints_per_sweep=101,
        nsamps=10,
        amp=AmplitudeSchedule.multiplicative(0.8, 8.0, 5),
        sweep_direction=("upward", "downward"),
    )

`store.load` is `pickle.load` plus one correction: the path the file recorded
about itself when it was written is replaced with where the file has actually
turned out to be. Demo data that shipped inside a package, or a sweep copied off
the acquisition machine — as this one was — can then still save itself back to
the file *you* opened rather than to a path on a computer you may not even be
on.

```python
multi_amplitude_ms = store.load(MULTISWEEP_PKL)

# A sweep comes back keyed by module identifier, and every function below takes
# one module's value out of it. Ours only has the one.
print(f"modules: {list(multi_amplitude_ms)}")

multi_amplitude_module_results = multi_amplitude_ms[list(multi_amplitude_ms)[0]]

print(f"schema_version:  {multi_amplitude_module_results['schema_version']}")
print(f"measurement:     {multi_amplitude_module_results['measurement']}")
print(f"module:          {multi_amplitude_module_results['module']}")
print(f"amplitude steps: {list(multi_amplitude_module_results['results'])}")
print(f"directions:      {list(multi_amplitude_module_results['results'][0])}")
print(f"resonators:      {list(multi_amplitude_module_results['results'][0]['upward'])}")
```

Every saved measurement also carries a `file_metadata` block saying what it is,
when it was taken, what wrote it, and where it lives. It is stamped inside each
module's output rather than at the top of the file, so you reach it wherever
you happen to be already working:

```python
for key, value in multi_amplitude_module_results["file_metadata"].items():
    print(f"{key:<18} {value}")
```

That `path` is what lets an analysis write its results back into the measurement
it read, without anyone having to carry a filename around — which is exactly
what `find_bias_points` does in section 5.

### The catalog that was swept

A sweep records the catalog it was given, under `call_params`, as a plain dict.
That is worth knowing about for two reasons: it is how you get the array
bookkeeping back out of a file weeks later, and it is the array
`rfmux.tuning.find_bias_points` biases — the function takes no catalog of its
own, because the sweep already carries the one it was taken from. A sweep of
bare `center_frequencies` records one too: `multisweep` generates a catalog
from the list, names the sections in it, and puts it here, so bias finding
works the same way on an array you have not tuned yet.

```python
swept_catalog = ResonatorCatalog.from_dict(
    multi_amplitude_module_results["call_params"]["catalog"]
)

print(swept_catalog)
```

<!-- #region -->
The array contains five resonators, which currently all have bias amplitudes listed as 0.001, which is the amplitude
that the resonance finding netanal was performed at.


### Take a look at the data

Here we demonstrate extracting and plotting the multi-amplitude multisweep data. We'll draft the plotting
functions by hand as an exercise, but canned example
plotting functions can also be found under `Demos/example_plotting_{...}.py`, for the various
topics covered in these notebooks.

<!-- #endregion -->

```python

from rfmux.tuning import (
    collect_amplitude_iterations_for,
    get_amplitudes_at_iteration,
)

for iteration in multi_amplitude_module_results["results"]:
    amplitudes = get_amplitudes_at_iteration(multi_amplitude_module_results, iteration)
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


resonator_names = list(multi_amplitude_module_results["results"][0]["upward"])
plot_amplitude_steps(multi_amplitude_module_results, resonator_names)
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
| `iterations` | required | multi-amplitude multisweep measurements of a resonator in the usual form: `{iteration: {direction: entry}}`. This can be extracted using the convenience wrapper `rfmux.tuning.collect_amplitude_iterations_for` |
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
    multi_amplitude_module_results, "BRUL"
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


plot_derivative_test(multi_amplitude_module_results, resonator_names)


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
| `noise_gate_factor` | `50.0` | and it must *also* stand this many noise floors out of the trace's own scatter. Larger is less sensitive; `0.0` switches the gate off |



<!-- #endregion -->

```python
from rfmux.tuning import bifurcated_by_derivative

```

### Examining the spike_prominence_factor and noise_gate_factor

If we run the bifurcation detection just looking at the spike prominence, without
any noise gating:

```python

import sys

if str(DEMOS) not in sys.path:
    sys.path.insert(0, str(DEMOS))

import example_plotting_bias as biasplots

biasplots.plot_bifurcation_verdict_map(
    multi_amplitude_module_results,
    noise_gate_factor=0.0,
    title="Only looking at spike prominence without considering noise",
)
```

Ideally the bifurcation detection should only trigger on a truly bifurcated trace,
so what we are seeing here is that noise spikes on the lower amplitude are being caught instead.
The finder provides a way to suppress these, by adding a noise gating factor,
which says that a spike's prominence must also be N times larger than the median
noise spread on a given sweep.


Below we make the same plot on the same data, but turning on the noise gate to its default value:

```python
biasplots.plot_bifurcation_verdict_map(
    multi_amplitude_module_results,
    title="With the noise gate activated at its default value",
)
```

One black bar per panel, in the top row, every one of them crossing the default
factor — and no black anywhere below it, at any factor on the axis.

The blue tint is the diagnostic for the noise gate: it covers the part of each
row where the noise gate is preventing the spikes from reading as bifurcated.

If the loudest row of one of your resonators were tinted all the way across, the
gate would be too high for that array and would be suppressing real
bifurcation. In that case, lower `noise_gate_factor`.



```python
biasplots.plot_bifurcation_verdict_map(
    multi_amplitude_module_results,
    noise_gate_factor=20,
    title="Smaller noise gate",
)
```

```python

```

##### The spike prominence threshold bar illustrated:




```python
SPIKE_PROMINENCE_FACTOR = 0.5
NOISE_GATE_FACTOR = 50.0


def plot_prominence_bar(results, names, direction="upward",
                        spike_prominence_factor=SPIKE_PROMINENCE_FACTOR,
                        noise_gate_factor=NOISE_GATE_FACTOR):
    """The bar each verdict was read off, on the steps that settled it.
    """
    fig, panels = panels_per_resonator(names, width=3.3, height=3.6, sharey=True)

    for panel, name in zip(panels, names):
        iterations = collect_amplitude_iterations_for(results, name)
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
            entry = iterations[iteration][direction]
            frequencies, speed = normalized_arc_speed(entry)
            midpoints = 0.5 * (frequencies[:-1] + frequencies[1:])

            check = bifurcated_by_derivative(
                {direction: entry},
                spike_prominence_factor=spike_prominence_factor,
                noise_gate_factor=noise_gate_factor,
            )
            # The bar the detector actually used, read off the check rather
            # than recomputed: it is the higher of the span bar and the noise
            # gate, and which one that is varies from step to step. Drawing a
            # re-derivation of one of them would put the line in the wrong
            # place exactly where the gate is what decided.
            bar = check.threshold

            up = check.metric["positive_spike_prominence"] / check.threshold
            down = check.metric["negative_spike_prominence"] / check.threshold
            panel.plot(offset_khz(entry, midpoints), np.diff(speed),
                       lw=1.0, color=colour,
                       label=f"step {iteration} ({role})\n"
                             f"{direction} only: {up:.2f} up, {down:.2f} down\n"
                             f"{direction} only: adjacent = "
                             f"{check.metric['adjacency']}")
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


plot_prominence_bar(multi_amplitude_module_results, resonator_names)
```

<!-- #region -->
The amplitude step chosen for the bias (grey) has a visible spike but does not
clear its dashed bar, so the search calls it not bifurcated. The red trace is the
step above it, the one that does clear the bar — and every resonator has one,
including `MELL`.



`MELL` is worth a second look. Its red step is labelled bifurcated, but the
numbers printed beside it — which are the *upward* sweep, the one drawn — say
`0.98 up` and `adjacent = False`. However, in this plotter we are printing the 
metric from the upward sweep only. Either the upward or downward can trigger the bifurcation
finder.

The numbers in the margins are various metrics that `rfmux.tuning.BifurcationCheck` reports along with its decisions,
 to facilitate troubleshooting. These include:

| Field | Is |
|---|---|
| `method` | which test produced this: `"derivative"`, `"hysteresis"` or `"both"` |
| `bifurcated` | the verdict |
| `metric` | a dict, one entry per quantity the method examined — see below |
| `threshold` | for `"derivative"`: the higher of `spike_prominence_factor` times the arc speed's range and `noise_gate_factor` times the trace's noise floor |
| `parts` | empty, unless this check combined several tests — `"both"` keeps each test's own check in here, raw numbers and own threshold |

`metric` is a dict rather than one number because the verdict is not one
comparison. `"derivative"` asks three things, and reports all three:

| Key | Is |
|---|---|
| `positive_spike_prominence` | how far the tallest up-spike stands out of its own neighbourhood. `0.0` if there is no up-spike at all |
| `negative_spike_prominence` | the same for the tallest down-spike |
| `adjacency` | whether a spike that cleared the bar was followed within two samples by a down-spike that also cleared it.|

The sweep is considered bifurcated when both prominences clear `threshold` **and**
`adjacency` — so these three entries tell you which condition decided it. To
find out *which* of the two bars was the binding one, run the check again with
`noise_gate_factor=0.0` and compare the thresholds, or read it off the tint in
the verdict map.

In general, you may have to fiddle with the various thresholds and other bias
finding settings for a given array. The verdict map is the quickest way to see
whether a change helped: a setting is right when every resonator shows one black
bar, at the loudest step, with your factor well inside it.


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


plot_magnitude_hysteresis(multi_amplitude_module_results, resonator_names,
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
| `noise_gate_factor` | `50.0` | handed to `rfmux.tuning.bifurcated_by_derivative` |
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

Across the array, each resonator read at *its own* chosen amplitude — which on
this measurement is step 3 for all five, since every one of them bifurcated at
the loudest step and nowhere below it:

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


plot_frequency_methods(multi_amplitude_module_results, resonator_names)
```

The two methods land within a single sweep point of each other on all five — the
grid spacing here is 750 Hz — so on this array the choice between them barely
matters. What does vary is the shape they are choosing from, and that is worth
looking at even when the answers agree.

`MELL` and `LALM` have broad, rounded peaks, 17 and 15 points across at half
height. The maximum is a region rather than a point: putting the tone a couple of
samples either side of it costs almost nothing, and the two methods landing on
the same sample is not telling you much.

`TOEL` is the other extreme, a peak two and a half times taller than `MELL`'s and
four points wide. Both methods find the top of it. The trouble is that the top of
a peak that narrow is a fragile place to sit: the resonance only has to drift by
a couple of kilohertz for the tone to be somewhere much less responsive. Nothing
in `find_bias_frequency` will tell you that, because it answered the question it
was asked — which is why section 6 comes back to how far the array may move
before the operating point has to be found again.


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

<!-- #region -->


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
<!-- #endregion -->

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
| `sweeps` | required | **one module's** `multisweep` outputs, i.e. `multi_amplitude_ms[crs.module[MODULE].index()]`. This is the whole input — the resonators to bias are the catalog recorded in its `call_params` |
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
finding: this notebook's schedule is the demo file that ships inside the rfmux
package, and re-saving it would edit the copy every other reader gets. Your own
sweeps came out of your own measurement, so leave `save` alone and the report
lands in the file beside the data it describes.
<!-- #endregion -->

```python
from rfmux.tuning import find_bias_points

bias_report = find_bias_points(multi_amplitude_module_results, save=False)

print(bias_report)
print(bias_report.catalog)

```

There was no array to pass: the resonators are the ones recorded in the sweep's
`call_params`, which are the ones the sweeps were taken from. That is the whole
reason there is no `catalog=` argument — a catalog handed in beside the sweeps
could only agree with them or disagree, and a bias point measured against
sweeps of a different array is the one thing it cannot survive. To bias part of
an array, take the subset out of the catalog and sweep it.

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

print(BiasReport.from_dict(multi_amplitude_module_results["bias_report"]))

```

That happens whether or not you save. `save=` is only the question of whether
the file on disk is brought up to date to match — and had we left it alone here,
this would have rewritten the `multisweep_*_demo_biasfind1.pkl` the
notebook loaded, in place, report and all. That is the point of it: the schedule
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
    # read the flags on rather than a number to format. Every resonator here
    # has one under the default method; run it with amplitude_method=
    # "hysteresis" below and two of them come back as None.
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

All five are good under the default method: every one of them bifurcated at the
loudest step and at none below it, so every bias amplitude is a step the schedule
established rather than a floor it ran into.

That is worth not taking for granted, because it is easy to arrange a report
where it is not true. Run the hysteresis detector on its own and two resonators
come back flagged:

```python
print(find_bias_points(multi_amplitude_module_results,
                       amplitude_method="hysteresis", save=False))
```

`MELL` and `LALM` are the two that jump at nearly the same frequency in both
directions, from section 2 — invisible to a test that works by comparing the
directions, and caught by the derivative half of the default. Their
`bifurcated_at` is `None` and their amplitude is 0.008 because that is the
loudest step measured, not because anything established 0.008 as a limit.

Those are bias points you can use, incidentally. They are just ones you should
decide to use, having read that they are a floor rather than a finding — the
right response being another schedule that goes louder.

Each flagged finding carries the sentence in `flagged_because`, so what you read
here is per resonator and specific — not a bit that says something went wrong
somewhere.

### The whole answer, on the whole measurement

Everything the report decided, drawn on the sweeps it decided it from. One panel
per resonator: the full schedule colour-coded by drive, the chosen step picked out
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
            f"amp {finding.amplitude:.4f}",
            fontsize=9,
        )
        panel.set_xlabel("offset from sweep centre [kHz]", fontsize=8)
        panel.tick_params(labelsize=7)

    panels[0].set_ylabel("|S21| [dB]", fontsize=8)
    fig.colorbar(mappable, ax=list(panels), label="drive amplitude")
    fig.suptitle("The bias points for each resonator",
                 fontsize=11)
    plt.show()


plot_bias_points_on_sweeps(multi_amplitude_module_results, bias_report)
```

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
  the GUI's bar restated as a multiplication; `noise_gate_factor` was set where
  the two populations separate most cleanly on the array above; `max_discrepancy`
  is a round number picked to sit between the quiet steps and the jumped ones.
  They get all five resonators here, but the margin is not the same for each of
  them — `MELL`'s black bar stops at 0.56 where `TOEL`'s runs to 0.92 — and a
  different array with a different noise floor will not distribute itself the
  same way. Draw the verdict map on your own sweeps, the way section 2 does,
  before trusting any of the defaults on them. A schedule that reaches high enough
  for every resonator to bifurcate is the other half of the answer, and the
  cheaper half.




<!-- #endregion -->


