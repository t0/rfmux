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

from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
print(f"module:          {multiamp_module_results['module']}")
print(f"amplitude steps: {list(multiamp_module_results['results'])}")
print(f"directions:      {list(multiamp_module_results['results'][0])}")
print(f"resonators:      {list(multiamp_module_results['results'][0]['upward'])}")
```

Every saved measurement also carries a `file_metadata` block saying what it is,
when it was taken, what wrote it, and where it lives. It is stamped inside each
module's envelope rather than at the top of the file, so you reach it wherever
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

Five resonators, all with bias amplitudes listed as 0.001 in normalized DAC units — the amplitude the
array was found and first swept at. The amplitude steps below are *relative* to
that, which is why the ladder runs 0.0008 to 0.008 rather than 0.8 to 8.

### Take a look at the data

Here we demonstrate extracting and plotting the multiamp multisweep data. We'll draft the plotting
functions by hand as an exercise, but canned example
plotting functions can also be found under `Demos/example_plotting_{...}.py`, for the various
topics covered in these notebooks.


```python

```

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
| `spike_prominence_factor` | `0.5` | `"derivative"` method only: how far a spike has to stand out from its surroundings to count as a jump, as a multiple of the arc speed's range. Larger is less sensitive |
| `max_discrepancy` | `0.25` | `"hysteresis"` method only: how far the upward and downward traces may part company, in units of the IQ loop's radius, before the step is called bifurcated |

`iterations` is positional; everything after it is keyword-only.
`spike_prominence_factor` and `max_discrepancy` are handed straight down to
whichever detector `method` selected, so passing both is harmless — the test
that has no use for a knob never sees it.

Called with nothing but the sweeps, on the first resonator:

```python
from rfmux.tuning import collect_amplitude_iterations_for, find_bias_amplitude

iterations_of_R0001 = collect_amplitude_iterations_for(
    multiamp_module_results, "R0001"
)
amplitude_choice = find_bias_amplitude(iterations_of_R0001)

print(f"iteration:              {amplitude_choice.iteration}")
print(f"amplitude:              {amplitude_choice.amplitude}")
print(f"bifurcated_at:          {amplitude_choice.bifurcated_at}")
print(f"is_bifurcated_at_bias:  {amplitude_choice.is_bifurcated_at_bias}")
```

So: bifurcation was first seen at 0.008, and the amplitude below it — 0.0045,
step 3 — is where this resonator should sit. 

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


plot_arc_speed(multiamp_module_results, "R0001", [0, 1, 2, 3, 4])
```

The bifurcation test uses the bottom panel, which shows the point-to-point change 
in the arc speed -- effectively the second derivative of I and Q with frequency.
At low amplitudes the change
from point to point is small. At step 4 — the one that was called
bifurcated — there is a sharp positive spike with a negative spike immediately
after it: the trace jumping onto the other state and dropping back off it again.

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
itself, which is what makes one number portable from one resonator to another.

If you are coming from the Periscope GUI, note that this is the same bar under
the same argument name but **not the same number**: the GUI *divided* the range
by a `spike_prominence_factor` of `2.0`, so turning its knob up made the test
more sensitive. Here the factor multiplies, which is what a factor does, and
`0.5` is the reciprocal that lands on the identical threshold. On the last
clean step and the first bifurcated one, at that default:

```python
from rfmux.tuning import bifurcated_by_derivative

for iteration in (3, 4):
    check = bifurcated_by_derivative(iterations_of_R0001[iteration])
    print(f"step {iteration}: {check}")
```

Here is that bar drawn on the data, computed the way the detector computes it,
for the same two steps:

```python
SPIKE_PROMINENCE_FACTOR = 0.5

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)

for panel, iteration in zip(axes, (3, 4)):
    entry = iterations_of_R0001[iteration]["upward"]
    frequencies, speed = normalized_arc_speed(entry)
    jumps = np.diff(speed)
    midpoints = 0.5 * (frequencies[:-1] + frequencies[1:])
    step_centre = entry["original_center_frequency"]

    # Exactly what bifurcated_by_derivative computes before calling find_peaks.
    prominence_threshold = SPIKE_PROMINENCE_FACTOR * (speed.max() - speed.min())

    panel.plot((midpoints - step_centre) / 1e3, jumps, lw=1.0, color="0.2")
    for sign in (1, -1):
        panel.axhline(sign * prominence_threshold, color="tab:red", ls="--",
                      lw=1.0,
                      label=f"prominence bar {prominence_threshold:.2e}"
                      if sign > 0 else None)

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

Step 3 clearly has a spike, but it's too small to meet the threshold we set with
the `spike_prominence_factor`, so it does not flag as bifurcated.
 On step 4 the spike is much larger, and exceeds the threshold set.
 
`rfmux.tuning.BifurcationCheck` reports the numbers it compared as well
as its verdict based on them, to facilitate troubleshooting

| Field | Is |
|---|---|
| `method` | which test produced this, `"derivative"` or `"hysteresis"` |
| `bifurcated` | the verdict |
| `metric` | the largest positive jump seen, for `"derivative"` |
| `threshold` | the bar a spike had to clear, for `"derivative"`: `spike_prominence_factor` times the arc speed's range |



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
panel.set_ylabel("measured spike height / threshold spike height")
panel.set_title("How far over the threshold each amplitude step is", fontsize=10)
panel.legend(fontsize=8)
plt.show()
```

<!-- #region -->
All five cross the line together between 0.0045 and 0.008, which is the drive at
which this array starts jumping. Note the margin either side of the crossing:
under 0.9 below and 1.6 or more above, so roughly a factor of two rather than
the orders of magnitude you might hope for. The factor has not been calibrated
across arrays, so read your own before trusting the default on it — this is one
array on one cooldown, and the plot above is exactly how you would read it.

### Bifurcation detection method #2 `"hysteresis"`

The other test, `rfmux.tuning.bifurcated_by_hysteresis`, looks for the amplitude at which 
the upward and downward frequency sweeps *begin* to differ.



| Argument | Default | Does |
|---|---|---|
| `entries` | required | one amplitude step, `{direction: entry}`, where `"upward"` and `"downward"` are required |
| `max_discrepancy` | `0.25` | how far apart the two traces may be, in units of the IQ loop's radius, before the step is called bifurcated |

The metric is the **largest separation between the two traces, in units of the
IQ loop's own radius**, so it means the same thing for a deep resonator and a
shallow one.
<!-- #endregion -->

```python
from rfmux.tuning import bifurcated_by_hysteresis

for iteration, entry in iterations_of_R0001.items():
    check = bifurcated_by_hysteresis(entry)
    print(f"step {iteration}, amp {entry['upward']['sweep_amplitude']:.4f}: "
          f"bifurcated={check.bifurcated!s:5}  metric={check.metric:.3f}  "
          f"threshold={check.threshold}")
```

The four quiet steps sit at a percent or less, which is the noise floor of this
measurement rather than a resonator doing anything. Then the loudest step jumps
to well over a full loop radius — two orders of magnitude of separation between
"these are the same trace" and "these are not", which is a far more comfortable
margin than the derivative test's factor of two.

Here are the two traces the test compares, at the last clean step and at the one
that fired:

```python
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)

for panel, iteration in zip(axes, (3, 4)):
    amplitude_step = iterations_of_R0001[iteration]
    for direction, style in (("upward", "-"), ("downward", "--")):
        iq = amplitude_step[direction]["iq_counts"]
        panel.plot(iq.real, iq.imag, style, lw=1.2, label=direction)

    check = bifurcated_by_hysteresis(amplitude_step)
    panel.set_title(f"step {iteration}, amp "
                    f"{amplitude_step['upward']['sweep_amplitude']:.4f}\n"
                    f"bifurcated={check.bifurcated}, "
                    f"metric={check.metric:.3f}", fontsize=9)
    panel.set_xlabel("I [counts]", fontsize=8)
    panel.set_aspect("equal")
    panel.legend(fontsize=8)

axes[0].set_ylabel("Q [counts]", fontsize=8)
fig.suptitle("The hysteresis test compares these two", fontsize=11)
plt.show()
```

On the left the two directions lie on top of each other. On the right they do
not: the sweep jumps off the resonance at a different frequency depending on
which way it is walking, so the two traces enclose the region between the jump
points. That is the physical signature of bifurcation, and it is the thing mock
mode cannot produce — the simulator evaluates each sweep point independently, so
its two directions are the same trace twice at every drive.

The two tests do not have to agree, and here they do not, quite:

```python
print(f"{'name':<8}{'derivative':>22}{'hysteresis':>22}")
for name in resonator_names:
    iterations = collect_amplitude_iterations_for(multiamp_module_results, name)
    by_derivative = find_bias_amplitude(iterations)
    by_hysteresis = find_bias_amplitude(iterations, method="hysteresis")
    print(f"{name:<8}"
          f"{by_derivative.amplitude:>15.4f} (step {by_derivative.iteration})"
          f"{by_hysteresis.amplitude:>15.4f} (step {by_hysteresis.iteration})")
```

Three of the five land in the same place. On R0004 and R0005 the hysteresis test
sees nothing at any amplitude, so it has no limit to step back from and returns
the loudest step it was given — the correct answer to the question it was asked,
and the wrong operating point. `find_bias_points` flags exactly that case, which
is section 5's business; the point here is that the two detectors read different
evidence and a resonator can show one and not the other.

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
fig.suptitle(f"R0001 at chosen bias amplitude {amplitude_choice.amplitude}: bias frequency selection",
             fontsize=11)
plt.show()
```

The two answers will generally be close but not identical.

Note also that this is a frequency **for one amplitude**, not for the resonator.
Driving a KID harder pulls its resonance down, so the answer moves as you climb
the ladder — which is why the bias frequency has to be read off the step you
actually chose, and why moving the tone invalidates the calibration measured at
the old one. Over this schedule R0004 walks the better part of 20 kHz:

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
| `amplitude_method` | `"derivative"` | which bifurcation test the amplitude search uses — section 2. `"hysteresis"` requires the sweeps to have been taken in both directions. |
| `frequency_method` | `"iq_derivative"` | what method to use to determine what frequency to bias at — section 3 |
| `direction` | `None` | which sweep direction to measure the bias frequency and the calibration on. `None` prefers `"upward"`. |
| `spike_prominence_factor` | `0.5` | passed to `rfmux.tuning.bifurcated_by_derivative` — section 2 |
| `max_discrepancy` | `0.25` | passed to `rfmux.tuning.bifurcated_by_hysteresis` — section 2 |
| `max_distance_hz` | `None` | how far from the sweep centre a resonance may come out before the bias frequency is rejected. Past this, the tone is left where the sweep was centred and the finding is flagged. Useful for handling densely packed arrays or collisions. |
| `save` | `None` | write the sweeps — which now carry the report — back to the file they came from. `None` does whatever `rfmux.tuning.store.autosave_enabled()` says, which is on unless you turned it off. Sweeps that have never been in a file get a new one |
| `label` | `None` | your name for that file, used only when these sweeps are being written for the first time. A re-save keeps the name the file already has |

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
print(f"the catalog we started from is still: {swept_catalog['R0001'].bias}")

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
this would have rewritten the `multiamp_multisweep_*_bias_finding.pkl` the
notebook loaded, in place, report and all. That is the point of it: the ladder
and the operating point read off it stay one file.

### The report

The function also returns one
`rfmux.tuning.BiasFinding` per resonator, which says how that resonator's bias
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
fallen back to. It's a good idea to read this list before trusting and applying the biases.

```python
print(f"biased:  {len(bias_report)}")
print(f"good:    {len(bias_report.good)}")
print(f"flagged: {len(bias_report.flagged)}")
```

Nothing is flagged here, because every resonator on this array bifurcated inside
the schedule and had a step to fall back to. The hysteresis run from section 2 is
where this array does produce flags — R0004 and R0005 showed that detector
nothing at any drive, so it returned the loudest step rather than a limit it had
found, and the report says so rather than leaving you to notice:

```python
print(find_bias_points(multiamp_module_results,
                       amplitude_method="hysteresis", save=False))
```

Each flagged finding carries the sentence in `flagged_because`, so what you read
here is per resonator and specific — not a bit that says something went wrong
somewhere.


## 6. Applying the bias points

`bias_report.catalog` is a catalog with the operating points in it. Playing
those tones is one call, and it is the first thing in this notebook that needs a
board:

    await crs.apply_bias(bias_report.catalog)

Read `bias_report.flagged` before you run it. A flagged bias point is still the
best the measurement supports, but it is a default rather than something the
amplitude steps established, and applying one is a decision rather than a
formality.

Nothing comes back. `apply_bias` is an *operation* — it lives in
`rfmux.algorithms.operation` rather than `.measurement` — so it either put the
board into the state you asked for or it raised saying why it could not. Each
resonator gets its bias frequency and its bias amplitude on its own channel.
Channels the catalog does not name are left exactly as they were, which means a
tone you parked by hand survives the call and also that applying a bias does not
leave the module otherwise quiet. Call `await crs.clear_channels(module=MODULE)`
first if you need it to be.

The one thing it may change beyond those channels is the module's NCO. Every
channel frequency is programmed as an offset from it, so the NCO has to reach
every tone in the catalog — and it has to sit on the tone grid, or the offsets
computed from it are off-grid however carefully each bias point was quantized.
If the NCO in place satisfies both, it is left alone; otherwise it is reset to
the quantized midpoint of the catalog's frequencies. `allow_nco_reset=False`
turns that into an error instead, for a run where something else owns the NCO:

    await crs.apply_bias(bias_report.catalog, allow_nco_reset=False)

A catalog whose bias frequencies span more than `ALLOWED_NCO_BANDWIDTH_HZ`
raises whatever you pass, because a module plays one NCO at a time and those
tones cannot be on the air together. That one is a catalog to be split, not a
call to be retried.

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
  the GUI's bar restated as a multiplication, and `max_discrepancy` was picked to
  be roughly right. Both get the right answer on the array above, which is one
  array on one cooldown — and the derivative test got it with a margin of about a
  factor of two either side, which is not much to spend on a different array with
  a different noise floor. Read `metric` and `threshold` across your own
  amplitude steps, the way section 2 does, before trusting the defaults on them.




