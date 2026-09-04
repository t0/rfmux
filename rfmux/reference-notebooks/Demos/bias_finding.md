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

The bias finding routines produce a **new** `ResonatorCatalog`. Nothing you pass in is
modified: not the catalog you swept, not the sweep data. That means you can run
the analysis twice with different settings and compare the two resulting catalogs, and it
means the catalog you started from is still the catalog you started from.

| Piece | Module |
|---|---|
| The bias finding functions used below | `rfmux.tuning.bias` |
| The multi-amplitude multisweep data used to inform the bias finding routines | `rfmux.algorithms.measurement.multiamp_multisweep` |
| The `ResonatorCatalog` and related array bookkeeping | `rfmux.core.resonators` |

<!-- | Reading a sweep result back | `rfmux.tuning.sweep_results` | -->

This notebook starts from a multisweep that was measured earlier and saved to
disk, so it can get straight to the analysis. How that measurement is set up and
run is covered in `multisweep.md`, and how you get to a catalog in the first
place in `network_analyses_find_resonances_make_resonator_catalog.md`.

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

import copy
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import rfmux
from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.core.transferfunctions import BASE_FREQUENCY

MODULE = 1

# The recorded sweep this notebook analyses, found through the package rather
# than by a relative path, so it works whichever directory the kernel started
# in.
DEMOS = Path(rfmux.__file__).parent / "reference-notebooks" / "Demos"
MULTISWEEP_PKL = DEMOS / "bias_finding_multisweep.pkl"

print(MULTISWEEP_PKL)
```

## 1. Start with a previously-measured multisweep

Bias finding is purely analysis: it takes sweeps that already exist and works on them.
 So rather than spend the first two minutes of
this notebook using mock mode to simulate a new multisweep measurement,
 we load a multi-amplitude sweep that was
measured once and saved.

The file holds the pickled output of `crs.multiamp_multisweep`: four simulated resonators, 
six amplitude steps a
factor of two apart, using both sweep directions. The script that produced it is
`make_bias_finding_multisweep.py`, next to this notebook, and it is worth a look
if you want to see the measurement that is being stood in for. On a real board
the same data comes from one call, and everything below this section is
unchanged:

    multiamp_ms = await crs.multiamp_multisweep(
        catalog,
        span_hz=60e3,
        npoints_per_sweep=201,
        nsamps=10,
        amp_schedule=AmplitudeSchedule.multiplicative(1.0, 32.0, 6),
        directions=("upward", "downward"),
    )



```python
with MULTISWEEP_PKL.open("rb") as f:
    multiamp_ms = pickle.load(f)

# A sweep comes back keyed by module identifier, and every function below takes
# one module's value out of it. Ours only has the one.
print(f"modules: {list(multiamp_ms)}")

multiamp_module_results = multiamp_ms[list(multiamp_ms)[0]]

print(f"schema_version:  {multiamp_module_results['schema_version']}")
print(f"module:          {multiamp_module_results['module']}")
print(f"amplitude steps: {list(multiamp_module_results['results'])}")
print(f"directions:      {list(multiamp_module_results['results'][0])}")
print(f"resonators:      {list(multiamp_module_results['results'][0]['upward'])}")
```

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

Four resonators, all with bias amplitudes listed as 0.001 in normalized DAC units — the amplitude the
array was found and first swept at. 

### Take a look at the data

Here we demonstrate extracting and plotting the multiamp multisweep data. We'll draft the plotting
functions by hand as an exercise, but canned example
plotting functions can also be found under `Demos/example_plotting_{...}.py`, for the various
topics covered in these notebooks.


```python

from rfmux.tuning import (
    collect_amplitude_iterations_for,
    get_amplitudes_at_iteration,
)

for iteration in multiamp_module_results["results"]:
    amplitudes = get_amplitudes_at_iteration(multiamp_module_results, iteration)
    print(f"step {iteration}: {amplitudes}")

AMPLITUDE_CMAP = plt.cm.gnuplot


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


def offset_khz(entry):
    """A sweep's frequencies as kHz either side of where it was centred."""
    return (entry["frequencies"] - entry["original_center_frequency"]) / 1e3


def plot_amplitude_steps(results, resonator_names, direction="upward"):
    """Every amplitude step of each resonator, one panel per resonator."""
    fig, axes = plt.subplots(
        1, len(resonator_names), figsize=(3.1 * len(resonator_names), 3.0),
        constrained_layout=True, squeeze=False,
    )

    for panel, name in zip(axes[0], resonator_names):
        iterations = collect_amplitude_iterations_for(results, name)
        amplitudes = [e[direction]["sweep_amplitude"] for e in iterations.values()]
        colours, mappable = amplitude_colours(amplitudes)

        for (entry, colour) in zip(iterations.values(), colours):
            sweep = entry[direction]
            # Divided by its own drive, so the traces can be compared by shape
            # rather than the loudest simply sitting on top of the others.
            iq = sweep["iq_counts"] / sweep["sweep_amplitude"]
            panel.plot(offset_khz(sweep), 20 * np.log10(np.abs(iq)),
                       lw=1.0, color=colour)

        panel.set_title(name, fontsize=10)
        panel.set_xlabel("offset from sweep centre [kHz]", fontsize=8)
        panel.tick_params(labelsize=8)

    axes[0][0].set_ylabel("|S21| / drive [dB]", fontsize=8)
    fig.colorbar(mappable, ax=axes[0], label="drive amplitude")
    plt.show()


resonator_names = list(multiamp_module_results["results"][0]["upward"])
plot_amplitude_steps(multiamp_module_results, resonator_names)
```

This is a reasonably suitable measurement to use for our bias finding. Each 
resonator has been swept at a low enough amplitude that it does not appear to be
perturbed by the readout current, and has also been swept at a high enough amplitude
that it is clearly bifurcated. This means that a reasonable bias amplitude is bracketed
somewhere between these two end points.

## 2. Choosing the bias amplitude

Generally the idea is to use as much readout amplitude as the resonator will tolerate before
becoming seriously bifurcated, because using higher readout amplitudes raises the detector
signal above additive system noise sources, such as from the LNA.

To try to identify the best amplitude to use, we look at the sweeps to find the lowest-amplitude sweep which 
is bifurcated, and then select one amplitude step below that (the highest amplitude sweep
which is **not** bifurcated).



rfmux provides `rfmux.tuning.find_bias_amplitude` to do this process on one
resonator at a time. Its arguments:

| Argument | Default | Does |
|---|---|---|
| `iterations` | required | multiamplitude multisweep measurements of a resonator in the usual form: `{iteration: {direction: entry}}`. This can be extracted using the convenience wrapper `rfmux.tuning.collect_amplitude_iterations_for` |
| `method` | `"derivative"` | which bifurcation detection method to apply. Options are: `"derivative"` (reads the shape of a single trace and looks for jumps) and `"hysteresis"` (compares the two sweep directions against each other to see when they diverge) |
| `spike_prominence_factor` | `2.0` | `"derivative"` method only: how far a spike has to stand out from its surroundings to count as a jump. Larger is less sensitive |
| `spike_height_factor` | `3.0` | `"derivative"` method only: how tall a spike has to be to count as a jump. Larger is less sensitive |
| `max_discrepancy` | `0.25` | `"hysteresis"` method only: how far the upward and downward traces may part company, in units of the IQ loop's radius, before the step is called bifurcated |

`iterations` is positional; everything after it is keyword-only. The two spike
factors and `max_discrepancy` are handed straight down to whichever detector
`method` selected, so passing all of them is harmless — the test that has no use
for a knob never sees it.

Called with nothing but the sweeps, on the first resonator:

```python
from rfmux.tuning import collect_amplitude_iterations_for, find_bias_amplitude

iterations_of_R0001 = collect_amplitude_iterations_for(
    multiamp_module_results, "R0001"
)
amplitude_choice = find_bias_amplitude(iterations_of_R0001)

print(f"iteration:      {amplitude_choice.iteration}")
print(f"amplitude:      {amplitude_choice.amplitude}")
print(f"bifurcated_at:  {amplitude_choice.bifurcated_at}")
print(f"is_bifurcated:  {amplitude_choice.is_bifurcated}")
```

So: bifurcation was first seen at 0.004, and the amplitude below it — 0.002,
step 1 — is where this resonator should sit. 

`checks` holds the verdict on each step it actually examined. The search stops
at the first bifurcated step, so the steps above it were never looked at and
have nothing to report:

```python
for iteration, check in amplitude_choice.checks.items():
    print(f"step {iteration}: bifurcated={check.bifurcated!s:5}  "
          f"metric={check.metric:.3e}  threshold={check.threshold:.3e}  "
          f"({check.method})")
```

<!-- #region -->
Some quasi-failure modes:

- **If nothing bifurcates**, the loudest step is chosen. Sweep again and include higher
amplitudes.
- **If the quietest step already bifurcates**, there is nothing below it to go
  back to. The quietest step is chosen,
  and `amplitude_choice.is_bifurcated` is `True` . The schedule started too high.



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

def plot_arc_speed(results, name, iterations_to_show, direction="upward"):
    """The normalized arc speed, and its point-to-point change, over a few
    amplitude steps."""
    collected = collect_amplitude_iterations_for(results, name)
    amplitudes = [collected[i][direction]["sweep_amplitude"] for i in iterations_to_show]
    colours, _ = amplitude_colours(amplitudes)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True,
                             constrained_layout=True)

    for iteration, colour in zip(iterations_to_show, colours):
        entry = collected[iteration][direction]
        frequencies, speed = normalized_arc_speed(entry)
        centre = entry["original_center_frequency"]
        label = f"step {iteration}, amp {entry['sweep_amplitude']:.4f}"

        axes[0].plot((frequencies - centre) / 1e3, speed, lw=1.0,
                     color=colour, label=label)
        # The point-to-point change is what the spikes are looked for in. It sits
        # between the points above, so its x-axis is their midpoints.
        midpoints = 0.5 * (frequencies[:-1] + frequencies[1:])
        axes[1].plot((midpoints - centre) / 1e3, np.diff(speed), lw=1.0,
                     color=colour)

    axes[0].set_ylabel("normalized arc speed [1/Hz]")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("point-to-point change")
    # Symmetric log, so the quiet steps are not a flat line beside the loud
    # ones — the spikes here are two orders of magnitude apart.
    axes[1].set_yscale("symlog", linthresh=1e-5)
    axes[1].set_xlabel("offset from sweep centre [kHz]")
    fig.suptitle(f"{name}: what the derivative test looks at", fontsize=11)
    plt.show()


plot_arc_speed(multiamp_module_results, "R0001", [0, 1, 2, 3])
```

The bifurcation test uses the bottom panel, which shows the point-to-point change 
in the arc speed -- effectively the second derivative of I and Q with frequency.
At low amplitudes the change
from point to point is small. At step 2 — the one that was called
bifurcated — there is a sharp positive spike with a negative spike immediately
after it: the trace jumping onto the other state and dropping back off it again.

**Two spikes, adjacent, first positive then negative** is the pattern
`rfmux.tuning.bifurcated_by_derivative` cues off of.

`find_bias_amplitude` calls `bifurcated_by_derivative` on each amplitude step, 
in both directions (if present). The amplitude step is counted as bifurcated if either
direction triggers the threshold. You can also call it yourself, on one step at
a time, which is how you work out what the two factors should be. Its whole
argument list:

| Argument | Default | Does |
|---|---|---|
| `entries` | required | one amplitude step, `{direction: entry}` — one value out of what `collect_amplitude_iterations_for` returns. Every direction present is tested, and the step counts as bifurcated if any of them says so: a bifurcated resonator jumps whichever way the sweep runs, so needing both to agree would only lose the one that happened to catch it |
| `spike_prominence_factor` | `2.0` | a spike must stand out from its surroundings by more than the full range of the arc speed divided by this. Larger is less sensitive |
| `spike_height_factor` | `3.0` | a spike must be taller than this many standard deviations of the point-to-point change. Larger is less sensitive |

A spike has to clear **both** bars, and both are set relative to the sweep
itself, which is what makes them portable from one resonator to another. On the
last clean step and the first bifurcated one, at those defaults:

```python
from rfmux.tuning import bifurcated_by_derivative

for iteration in (1, 2):
    check = bifurcated_by_derivative(iterations_of_R0001[iteration])
    print(f"step {iteration}: {check}")
```

Here are both bars drawn on the data, computed the way the detector computes
them, for the same two steps:

```python
SPIKE_PROMINENCE_FACTOR = 2.0
SPIKE_HEIGHT_FACTOR = 3.0

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)

for panel, iteration in zip(axes, (1, 2)):
    entry = iterations_of_R0001[iteration]["upward"]
    frequencies, speed = normalized_arc_speed(entry)
    jumps = np.diff(speed)
    midpoints = 0.5 * (frequencies[:-1] + frequencies[1:])
    step_centre = entry["original_center_frequency"]

    # Exactly what bifurcated_by_derivative computes before calling find_peaks.
    prominence = (speed.max() - speed.min()) / SPIKE_PROMINENCE_FACTOR
    height = SPIKE_HEIGHT_FACTOR * np.std(jumps)

    panel.plot((midpoints - step_centre) / 1e3, jumps, lw=1.0, color="0.2")
    for bar, colour, label in ((height, "tab:orange", "height"),
                               (prominence, "tab:red", "prominence")):
        panel.axhline(bar, color=colour, ls="--", lw=1.0,
                      label=f"{label} bar {bar:.2e}")
        panel.axhline(-bar, color=colour, ls="--", lw=1.0)

    check = bifurcated_by_derivative({"upward": entry})
    panel.set_title(f"step {iteration}, amp {entry['sweep_amplitude']:.4f}"
                    f"\nbifurcated={check.bifurcated}, "
                    f"metric/threshold = {check.metric / check.threshold:.2f}",
                    fontsize=9)
    panel.set_xlabel("offset from sweep centre [kHz]", fontsize=8)
    panel.legend(fontsize=8)

axes[0].set_ylabel("point-to-point change in arc speed", fontsize=8)
plt.show()
```

That is the whole story of why step 1 is called clean. It plainly has a spike,
and that spike is well clear of the height bar — but not of the prominence bar,
because next to the broad hump the arc speed makes on the way up, it is not a
big enough departure from its surroundings. By step 2 the spike is several times
the prominence bar. On this array the prominence bar is the binding one at every
amplitude, which is worth knowing before you reach for `spike_height_factor`.

Every `rfmux.tuning.BifurcationCheck` reports the numbers it compared, not just its verdict,
because these factors are knobs you have to set against your own array:

| Field | Is |
|---|---|
| `method` | which test produced this, `"derivative"` or `"hysteresis"` |
| `bifurcated` | the verdict |
| `metric` | the largest positive jump seen, for `"derivative"` |
| `threshold` | the bar it had to clear: the larger of the two above, since it has to clear both |

`metric` above `threshold` is necessary but not sufficient, because the up-spike
still has to be followed by a down-spike. Watching the ratio across the
amplitude steps is the practical way to choose the factors — filled markers
below are the steps that were actually called bifurcated:

```python
fig, panel = plt.subplots(figsize=(6.5, 3.4), constrained_layout=True)

for name in resonator_names:
    iterations_of_this_resonator = collect_amplitude_iterations_for(
        multiamp_module_results, name
    )
    amplitudes, ratios, verdicts = [], [], []
    for entry in iterations_of_this_resonator.values():
        check = bifurcated_by_derivative(entry)
        amplitudes.append(entry["upward"]["sweep_amplitude"])
        ratios.append(check.metric / check.threshold)
        verdicts.append(check.bifurcated)

    line, = panel.plot(amplitudes, ratios, "o-", lw=1.0, ms=5,
                       mfc="white", label=name)
    panel.plot([a for a, v in zip(amplitudes, verdicts) if v],
               [r for r, v in zip(ratios, verdicts) if v],
               "o", ms=5, color=line.get_color())

panel.axhline(1.0, color="0.6", ls="--", lw=1.0)
panel.set_xscale("log")
panel.set_xlabel("drive amplitude")
panel.set_ylabel("metric / threshold")
panel.set_title("How far over the bar each amplitude step is", fontsize=10)
panel.legend(fontsize=8)
plt.show()
```

All four cross the line together between 0.002 and 0.004, which is the drive at
which this simulated array starts jumping. Note the margin either side of the
crossing: about 0.5 below and about 1.9 above, so a factor of two rather than
the orders of magnitude you might hope for. And R0003 at the quietest amplitude
sits just over the line without being called bifurcated, held back only by the
adjacency rule. Neither of the two factors has been calibrated against real
hardware, so read your own array before trusting the defaults on it.

### How bifurcation is decided: `"hysteresis"`

The other test, `rfmux.tuning.bifurcated_by_hysteresis`, does not need to know
what a jump looks like. A resonator below
bifurcation does not care which direction it was swept in: the upward and
downward traces lie on top of each other. Above it, the resonator jumps at a
different frequency going up than it does coming down, so the two traces part
company in between. The amplitude at which they *begin* to differ is the
amplitude at which bifurcation set in.

Two arguments, and one of them is the data:

| Argument | Default | Does |
|---|---|---|
| `entries` | required | one amplitude step, `{direction: entry}`, same as the derivative test takes. Both `"upward"` and `"downward"` are required here rather than optional — this test *is* the comparison, so a step missing one of them raises |
| `max_discrepancy` | `0.25` | how far apart the two traces may be, in units of the IQ loop's radius, before the step is called bifurcated. This is the one number in the module with no history behind it — a starting point rather than a measured value |

That needs a sweep taken in both directions — which this one is — and it needs
the hardware to have some memory of where it was, which is where the simulator
lets us down. It computes each sweep point independently, so its upward and
downward traces are identical up to noise, at every drive:

```python
from rfmux.tuning import bifurcated_by_hysteresis

for iteration, entry in iterations_of_R0001.items():
    check = bifurcated_by_hysteresis(entry)
    print(f"step {iteration}, amp {entry['upward']['sweep_amplitude']:.4f}: "
          f"bifurcated={check.bifurcated!s:5}  metric={check.metric:.3f}  "
          f"threshold={check.threshold}")
```

Every step sits at a few percent, which is the noise floor of this measurement
rather than a resonator doing anything, and none of them come near the default
threshold of 0.25. On this data the hysteresis search therefore finds no
bifurcation at all and picks the loudest step — the correct answer to the
question it was asked, and the wrong operating point.

The metric is the **largest separation between the two traces, in units of the
IQ loop's own radius**, so it means the same thing for a deep resonator and a
shallow one. To see what firing looks like, we can displace one direction by a
known amount. This is imposed rather than simulated — real hysteresis appears
over the band between the two jump points, not across the whole sweep — but it
shows what the number is measuring. Expect the metric to come back a little over
a third, since the traces were not exactly on top of each other to begin with:

```python
pretend_hysteretic_step = copy.deepcopy(iterations_of_R0001[1])
up = pretend_hysteretic_step["upward"]["iq_counts"]
loop_radius = np.max(np.abs(up - up.mean()))

# Push the downward trace a third of a loop radius away from the upward one,
# over a band of frequencies near the resonance. Reversed on the way in,
# because a downward sweep's points are stored high frequency first.
displacement = np.zeros_like(up)
displacement[95:115] = loop_radius / 3
pretend_hysteretic_step["downward"]["iq_counts"] = (
    pretend_hysteretic_step["downward"]["iq_counts"] + displacement[::-1]
)

print(f"as measured:  {bifurcated_by_hysteresis(iterations_of_R0001[1])}")
print(f"displaced:    {bifurcated_by_hysteresis(pretend_hysteretic_step)}")
```

```python
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)

for panel, (title, amplitude_step) in zip(
    axes,
    [("as measured", iterations_of_R0001[1]),
     ("displaced by hand", pretend_hysteretic_step)],
):
    for direction, style in (("upward", "-"), ("downward", "--")):
        iq = amplitude_step[direction]["iq_counts"]
        panel.plot(iq.real, iq.imag, style, lw=1.2, label=direction)
    panel.set_title(title, fontsize=10)
    panel.set_xlabel("I [counts]", fontsize=8)
    panel.set_aspect("equal")
    panel.legend(fontsize=8)

axes[0].set_ylabel("Q [counts]", fontsize=8)
fig.suptitle("The hysteresis test compares these two", fontsize=11)
plt.show()
```

`max_discrepancy` is the threshold, in those loop radii, and it is the one
number in this module with no history behind it — 0.25 is a starting point, not
a measured value. The way to set it is to read `metric` across the amplitude
steps of a resonator you know bifurcates, which is why every check reports it.

Which test to use, then:

| | `"derivative"` | `"hysteresis"` |
|---|---|---|
| Needs | one direction | both directions, so twice the measurement |
| Looks for | the shape of a jump in one trace | disagreement between two traces |
| Fooled by | a sweep too coarse to resolve the resonance, or one with no resonance in it — both look like a jump | its own noise floor, if `max_discrepancy` is set anywhere near it. And it sees nothing at all if the jump is not hysteretic |
| Tuned with | `spike_prominence_factor`, `spike_height_factor` | `max_discrepancy` |

## 3. Choosing the frequency

With the amplitude settled, the tone still has to go somewhere inside that
sweep. The sweep centre is where we *looked* — it came from the catalog, which
was seeded from a network analysis at some other drive — and the resonance is
wherever it turned out to be, which the plots in section 1 showed can be tens of
kilohertz away.

`rfmux.tuning.find_bias_frequency` is what answers that. Two arguments:

| Argument | Default | Does |
|---|---|---|
| `entry` | required | **one** sweep, as `multisweep` returns it — a single direction of a single amplitude step, not the `{direction: entry}` mapping the bifurcation tests take. A frequency has to come off one trace |
| `method` | `"iq_derivative"` | where in that trace to put the tone, from the two below |

| `method` | Puts the tone at | Because |
|---|---|---|
| `"iq_derivative"` (the default) | maximum `\|dI/df + j·dQ/df\|` | that is where the IQ trace moves fastest per hertz, so a small shift in the resonance makes the largest signal — the point you want to sit on |
| `"minimum"` | minimum `\|S21\|` | the bottom of the dip. Says nothing about responsivity, but it survives traces the derivative method finds noisy |

Both return a point of the measured grid. Neither needs a fit to have converged.
The one trace to use is the chosen amplitude step, in whichever direction we
prefer to read:

```python
from rfmux.tuning import find_bias_frequency, iq_arc_speed

chosen_sweep = iterations_of_R0001[amplitude_choice.iteration]["upward"]
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
fig.suptitle(f"R0001 at step {amplitude_choice.iteration}: where the tone goes",
             fontsize=11)
plt.show()
```

The two answers are close but not identical, and which of them is *right*
depends on what you want from the tone. The arc-speed peak is the responsivity
argument; the dip bottom is the simpler thing to defend when the trace is ugly.

### When the answer is not plausible

`find_bias_frequency` always answers, and the answer is always a point of the
trace. Whether it is a *believable* point is a separate question, and it needs
two things this function does not have: the sweep centre, and how far from it
you are willing to believe. So the judgement lives in `find_bias_points`, under
`max_distance_hz` — section 5.

It is worth seeing why. R0004's resonance is pulled off the bottom of its own
sweep as the drive goes up:

```python
iterations_of_R0004 = collect_amplitude_iterations_for(
    multiamp_module_results, "R0004"
)

quietest_step_frequencies = iterations_of_R0004[0]["upward"]["frequencies"]
print(f"the sweep runs to ±"
      f"{np.ptp(quietest_step_frequencies)/2e3:.0f} kHz "
      f"either side of its centre\n")

for iteration, amplitude_step in iterations_of_R0004.items():
    entry = amplitude_step["upward"]
    answer = find_bias_frequency(entry) - entry["original_center_frequency"]
    print(f"step {iteration}, amp {entry['sweep_amplitude']:.4f}: "
          f"{answer/1e3:+7.2f} kHz")
```

Steps 0 to 3 march steadily downwards, which is the resonance being pulled. By
step 5 the answer is out at −26.7 kHz, all but on the edge — a resonance that
has essentially left the span, and a bias frequency you would want to know
about before trusting it.

Step 4 is the honest caveat about any distance test. Its answer jumps back to
*+5.4 kHz*, on the wrong side of the centre, because by then the resonance is
gone from the span and the largest arc speed is simply the worst noise sample in
a flat trace. A distance test cannot catch that one: the noise landed near the
middle, so the answer is implausible without being far away. What fixes step 4
is a wider span, or re-centring the sweep between amplitude steps — a
measurement decision rather than an analysis one.

What happens to an answer that fails the distance test is that the tone is left
where the sweep was centred — the frequency it already had — and the finding is
flagged. Moving the tone onto a peak we do not believe would be worse than not
moving it at all. The calibration is then measured at the centre too, so a bias
point never carries derivatives read somewhere the tone is not.

## 4. The calibration at that point

An amplitude and a frequency are enough to make a
`rfmux.core.resonators.BiasPoint`, which is the
type the catalog holds one of per resonator. Building it is also the moment the
frequency is quantized:

```python
quantized_bias_point = BiasPoint(
    frequency_hz=bias_frequency_by_derivative,
    amplitude=amplitude_choice.amplitude,
)

print(f"asked for  {bias_frequency_by_derivative:.3f} Hz")
print(f"stored     {quantized_bias_point.frequency_hz:.3f} Hz")
print(f"difference "
      f"{quantized_bias_point.frequency_hz - bias_frequency_by_derivative:+.3f} Hz, "
      f"on a grid of {BASE_FREQUENCY:.3f} Hz")
```

Bias frequencies land on the hardware tone grid as they are set, so what you
read back is what the board will actually play, and no later reader has to
wonder which of the two numbers it is holding.

That matters here, because the last step measures the two IQ derivatives **at
that quantized frequency** rather than at the peak we found: the calibration
should describe the tone that will be there, not one half a grid step away.
`rfmux.tuning.iq_derivatives_at` does the measuring, off `iq_volts` and nothing
else — the units are the entire point, and counts per hertz would be a number of
the right magnitude and the wrong meaning. Both of its arguments are positional,
and it has no options:

| Argument | Default | Does |
|---|---|---|
| `entry` | required | one sweep, the same single trace `find_bias_frequency` reads. It must carry `iq_volts`, or this raises rather than quietly answering in counts |
| `frequency_hz` | required | where along that trace to evaluate the slopes. Normally the bias frequency *after* quantization, so the calibration belongs to the tone that will actually be played |

```python
from rfmux.tuning import iq_derivatives_at

dI_df, dQ_df = iq_derivatives_at(chosen_sweep, quantized_bias_point.frequency_hz)

print(f"dI_df  {dI_df:+.4e} V/Hz")
print(f"dQ_df  {dQ_df:+.4e} V/Hz")
```

These are slopes: how many volts of I, and of Q, you get per hertz the resonance
moves. Here they are drawn as tangents on the measured trace:

```python
bias_frequency = quantized_bias_point.frequency_hz

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

fig.suptitle("The slopes that become the calibration", fontsize=11)
plt.show()
```

And on the IQ loop, where the two together are the direction and speed the trace
is travelling at the bias point:

```python
MOVEMENT_HZ = 200.0

fig, panel = plt.subplots(figsize=(4.8, 4.6), constrained_layout=True)

panel.plot(chosen_sweep_volts.real, chosen_sweep_volts.imag,
           ".-", lw=1.0, ms=3, color="0.2")
panel.plot(i_at_bias, q_at_bias, "o", color="tab:red", ms=7, label="bias point")

# The two derivatives together, as an arrow: where the tone's reading goes if
# the resonance moves by MOVEMENT_HZ, to scale against the loop.
tip = (i_at_bias + dI_df * MOVEMENT_HZ, q_at_bias + dQ_df * MOVEMENT_HZ)
panel.plot(*tip, ".", alpha=0)   # so the arrow stays inside the axes
panel.annotate("", xytext=(i_at_bias, q_at_bias), xy=tip,
               arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0))

panel.set_xlabel("I [V]")
panel.set_ylabel("Q [V]")
panel.set_aspect("equal")
panel.set_title(f"{MOVEMENT_HZ:.0f} Hz of movement, at the bias point", fontsize=10)
panel.legend(fontsize=8)
plt.show()
```

### `df_calibration`

The two slopes are stored, and the thing you actually use is derived from them:

```text
df_calibration = 1 / (dI_df + j·dQ_df)     Hz/V
```

which is the factor that turns a measured voltage excursion into a frequency
shift — the whole point of the exercise, and what df display units are computed
through. It is a property on `BiasPoint` rather than a stored field, so it
cannot disagree with the derivatives it comes from:

```python
bias_point = replace(quantized_bias_point, dI_df=dI_df, dQ_df=dQ_df)

print(bias_point)
print(f"\ndf_calibration   {bias_point.df_calibration}")
print(f"|df_calibration| {abs(bias_point.df_calibration)/1e6:.3f} MHz/V")
print(f"so 1 µV along the arrow above is "
      f"{abs(bias_point.df_calibration) * 1e-6:.2f} Hz of resonance movement")
```

That `dataclasses.replace` is worth a second look, because it is doing something
the type
insists on: **a `BiasPoint` is frozen, and its frequency and its calibration are
one fact.** You cannot set the slopes on an existing one, you build a new one
carrying both — which is why bias finding measures the calibration in the same
step that chooses the frequency, rather than leaving it for later.

The same rule going the other way: move the tone, and the calibration does not
come along, because a slope measured at the old frequency does not describe the
new one. `Resonator.set_bias` is the chokepoint that enforces it:

```python
retuned_resonator = Resonator(name="R0001", channel=1, bias=bias_point)
print(f"before: df_calibration = {retuned_resonator.bias.df_calibration}")

retuned_resonator.set_bias(frequency_hz=bias_point.frequency_hz + 10e3)
print(f"after:  df_calibration = {retuned_resonator.bias.df_calibration}")
```

A stale calibration is not something you have to remember to avoid here; it is
not representable.

## 5. All three steps at once: `find_bias_points`

Sections 2 to 4 are one resonator at a time, deliberately, because that is how
you work out whether the analysis is doing something sensible.
`rfmux.tuning.find_bias_points`
runs the same three steps over every resonator in a catalog and assembles the
result.

Its arguments are the union of the three steps' arguments, plus one of its own.
Nothing here is new except `catalog`, `direction` and `max_distance_hz` — the
rest are the knobs from sections 2 and 3, passed through to the function that
uses them:

| Argument | Default | Does |
|---|---|---|
| `sweeps` | required | **one module's** value out of what `multiamp_multisweep` returned, `multiamp_ms[crs.module[MODULE].index()]`. The whole container, keyed by module, is refused: a report is about one module, and which one is your choice to make |
| `catalog` | `None` | the resonators to bias, and the source of everything the new catalog keeps unchanged — names, channels, the module, the separation rule. `None` takes the snapshot recorded in the sweep's `call_params`, which is the usual case: you are biasing the array you swept |
| `amplitude_method` | `"derivative"` | which bifurcation test the amplitude search uses — section 2. `"hysteresis"` requires the sweeps to have been taken in both directions, and is refused up front if they were not |
| `frequency_method` | `"iq_derivative"` | where in the chosen sweep the tone goes — section 3 |
| `direction` | `None` | which direction's sweep to measure the bias frequency and the calibration on, since those have to come off a single trace. `None` prefers `"upward"`. The amplitude search is unaffected: a detector sees every direction of its own step |
| `spike_prominence_factor` | `2.0` | passed to `rfmux.tuning.bifurcated_by_derivative` — section 2 |
| `spike_height_factor` | `3.0` | passed to `bifurcated_by_derivative` — section 2 |
| `max_discrepancy` | `0.25` | passed to `rfmux.tuning.bifurcated_by_hysteresis` — section 2 |
| `max_distance_hz` | `None` | how far from the sweep centre a resonance may come out before the answer is disbelieved. Past this, the tone is left where the sweep was centred and the finding is flagged — section 3. `None` believes anything, which is everything the trace could offer: the answer is always a point of the trace, so this only means something when it is tighter than the span |

`sweeps` is positional; everything after it is keyword-only except `catalog`,
which can be given either way.

Called with nothing but one module's sweeps, which is the whole of it for a
measurement that came out well:

```python
from rfmux.tuning import find_bias_points

bias_report = find_bias_points(multiamp_module_results)

print(bias_report)
```

No catalog was passed, so it used the one recorded in the sweep's `call_params`
— the array that was swept, which is nearly always the array you want to bias.
Pass `catalog=` to override that.

**The answer is `bias_report.catalog`**, a new `ResonatorCatalog`:

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

Every one of them moved down in frequency and up in amplitude: the array was
biased too quietly, and at the higher drive the resonances sit lower.

The catalog that went in is untouched — as are the sweeps, which is why the
analysis can be re-run with different settings on the same data as many times as
you like:

```python
print(f"the catalog we started from is still: {swept_catalog['R0001'].bias}")
print(f"\nit is a different object:            "
      f"{bias_report.catalog['R0001'] is not swept_catalog['R0001']}")
print(f"names and channels are the same:     "
      f"{[r.name for r in bias_report.catalog] == [r.name for r in swept_catalog]}, "
      f"{[r.channel for r in bias_report.catalog] == [r.channel for r in swept_catalog]}")
```

### The report

`bias_report.catalog` is the answer; the findings are the working. There is one
`rfmux.tuning.BiasFinding` per resonator, and it says how that resonator's bias
point was
arrived at:

```python
finding_for_R0001 = bias_report["R0001"]

print(f"name           {finding_for_R0001.name}")
print(f"iteration      {finding_for_R0001.iteration}")
print(f"amplitude      {finding_for_R0001.amplitude}")
print(f"bifurcated_at  {finding_for_R0001.bifurcated_at}")
print(f"frequency_hz   {finding_for_R0001.frequency_hz}")
print(f"dI_df, dQ_df   {finding_for_R0001.dI_df:.4e}, {finding_for_R0001.dQ_df:.4e}")
print(f"good           {finding_for_R0001.good}")
print(f"flagged_because {finding_for_R0001.flagged_because}")
print(f"\nchecks         {list(finding_for_R0001.checks)}")
```

```python
print(f"{'name':<7}{'step':>6}{'amplitude':>12}{'bif at':>10}"
      f"{'bias freq [MHz]':>18}{'|df_cal| [MHz/V]':>19}")
for f in bias_report.findings:
    df_calibration = bias_report.catalog[f.name].bias.df_calibration
    print(f"{f.name:<7}{f.iteration:>6}{f.amplitude:>12.4f}"
          f"{f.bifurcated_at:>10.4f}{f.frequency_hz/1e6:>18.6f}"
          f"{abs(df_calibration)/1e6:>19.3f}")
```

`bias_report.flagged` and `bias_report.good` split the findings by whether the
answer is something the amplitude steps established or a default that was
fallen back to — the next section is about that. It is the list to read before
applying anything:

```python
print(f"biased:  {len(bias_report)}")
print(f"good:    {len(bias_report.good)}")
print(f"flagged: {len(bias_report.flagged)}")
```

And a `settings` dict recording what the analysis was asked for. These are not
copied onto the individual bias points, since they would be the same values
repeated on every resonator:

```python
for key, value in bias_report.settings.items():
    print(f"  {key:<24} {value!r}")
```

### When the answer is a default rather than a measurement

**Every resonator in the catalog comes back with a bias point.** There is no
unbiased outcome to check for: the sweeps were taken from this catalog, so
every resonator has the data the three steps need, and the three steps always
produce an answer. (A catalog holding a resonator these sweeps do not cover, or
a sweep with no `iq_volts`, means the two arguments did not come from the same
measurement — that raises, rather than turning into a per-resonator result.)

What does happen is that an answer comes out as a **default**. There are three,
and `flagged_because` says which:

| Flagged when | Meaning |
|---|---|
| the quietest amplitude measured was already bifurcated | there was nothing below it to fall back to, so the operating point is the best of a bad set. Start the schedule lower |
| nothing bifurcated at all | this is simply the loudest amplitude measured, not a limit that was found. Extend the schedule higher |
| the bias frequency landed further than `max_distance_hz` from the sweep centre | usually a neighbour in the span, or a resonance pulled out of it — section 3 |

Those bias points are perfectly usable, and they are the best the measurement
supports. They are just not what the analysis set out to find, which is the
distinction the flag draws. The first two are the ones you meet in practice, and
both are a statement about the amplitude schedule rather than about the
resonator:

```python
nothing_bifurcated_report = find_bias_points(
    multiamp_module_results, spike_height_factor=1e6
)
print("nothing looks bifurcated at all:")
print(nothing_bifurcated_report)

print()
print(f"still biased: "
      f"{len(nothing_bifurcated_report.good) + len(nothing_bifurcated_report.flagged)}"
      f" of {len(nothing_bifurcated_report)}, at "
      f"{nothing_bifurcated_report['R0001'].amplitude} (the loudest measured)")
```

That run turned the spike threshold up until nothing could clear it, which is
the same situation as an amplitude schedule that never drove the array hard
enough. Note that every resonator still has a bias point, and every one is
flagged.

Only the first concern is reported, worst first, so a resonator whose sweeps
never bifurcated does not also get told its tone is a little off centre.

### Swapping out one of the three steps

Swapping a method swaps a step, and everything else stays as it was. Two runs
over the same sweeps, differing only in how the frequency was chosen:

```python
minimum_frequency_report = find_bias_points(
    multiamp_module_results, frequency_method="minimum"
)

print(f"{'name':<7}{'iq_derivative':>18}{'minimum':>16}{'difference':>14}")
for a, b in zip(bias_report.findings, minimum_frequency_report.findings):
    print(f"{a.name:<7}{a.frequency_hz/1e6:>18.6f}{b.frequency_hz/1e6:>16.6f}"
          f"{(b.frequency_hz - a.frequency_hz):>13.0f} Hz")
```

One to three tone-grid steps apart, on this data. Which of them is right depends
on what you want from the tone, and on a noisier trace they can part company by
a good deal more than that.

And the same comparison for the amplitude, which on this data shows the
simulator's lack of hysteresis rather than anything about the two methods:

```python
hysteresis_amplitude_report = find_bias_points(
    multiamp_module_results, amplitude_method="hysteresis"
)

print(f"{'name':<7}{'derivative':>14}{'hysteresis':>14}")
for a, b in zip(bias_report.findings, hysteresis_amplitude_report.findings):
    print(f"{a.name:<7}{a.amplitude:>14.4f}{b.amplitude:>14.4f}")

print(f"\n{hysteresis_amplitude_report}")
```

Every resonator still comes back biased, and every one of them is flagged — the
hysteresis run found no bifurcation anywhere, so all four answers are the
loudest amplitude measured rather than a limit that was found. That is exactly
what the flag is for: the numbers look like an answer, and the report tells you
they are a fallback.

## 6. What is not here yet

- **Applying the bias.** `bias_report.catalog` is a catalog with the operating points
  in it; programming those tones onto the board is `apply_bias`, which has not
  been ported to take a catalog yet.
- **IQ rotation.** `BiasPoint` has a field for it, and bias finding leaves it
  alone: the angle comes from a timestream rather than from a sweep, so it is
  not this step's to measure.
- **The fitted `fr` as a bias-frequency method.** `rfmux.tuning.fit_sweeps`
  already produces
  `fr` for every sweep, and `fitting_resonators.md` covers it. Wiring it in as a
  third `frequency_method` is a small job; the methods take a whole sweep entry
  rather than two arrays precisely so that one of them can read the entry's
  `fits`.
- **Thresholds that have met a real array.** The two spike factors are inherited
  from the GUI and `max_discrepancy` was picked to be roughly right; neither has
  been calibrated against hardware that is known to bifurcate. Read `metric` and
  `threshold` across your own amplitude steps before trusting the defaults.
- **Saving the result.** Not missing any more: `find_bias_points` writes the
  whole report — the biased catalog and the working behind it — to
  `~/rfmux_data/ipy_session_<today>/bias_report_*.pkl`, and
  `BiasReport.from_dict(rfmux.tuning.store.load(path))` reads it back. Pass
  `save=False` to skip it. What is still to come is one *folder* per tuning run
  that ties a catalog to the sweeps and settings that produced it; today each
  measurement is its own file, found by its name and its `file_metadata`.
