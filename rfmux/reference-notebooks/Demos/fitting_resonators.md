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

The nonlinear model uses each trace's `sweep_direction` to select the stable
branch through bifurcation. When calling `fit_nonlinear_iq` directly, pass
`sweep_direction="upward"` or `"downward"`; if omitted, it uses the input
frequency order before sorting. Model reconstruction uses the same direction.
This assumes a sweep entering from outside the bistable region. A trace that
starts inside it can depend on its prepared state, which direction alone does
not describe; inspect the residual and accept that some fits fail.

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
Measurements use `rfmux.tuning.store` and are saved by default. Fit functions
add their results to the measurement in place and update its saved file.

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

import rfmux
from rfmux.core.resonators import ResonatorCatalog
from rfmux.tuning import AmplitudeSchedule, store
import example_plotting_multisweep as msplots
import example_plotting_fits as fitplots

MODULE = 1

OUTPUT_DIR = store.session_directory()
print(f"results: {OUTPUT_DIR}")
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
more detail in the IQ plots. This notebook enables autosave to demonstrate
keeping fitted measurements on disk and checking that comparison fits leave
them unchanged. `store.session_directory()` shows the destination.

```python
store.set_autosave(True)
print(f"Saving measurements to: {store.session_directory()}")
amplitude_schedule = AmplitudeSchedule.multiplicative(0.5, 8.0, 5)
print(amplitude_schedule)

for step in amplitude_schedule.resolve_steps(catalog):
    print(step)

multi_amplitude_ms = await crs.multisweep(
    catalog,
    span_hz=200e3,
    npoints_per_sweep=100,
    nsamps=10,
    amp=amplitude_schedule,
    sweep_direction=("upward", "downward"),
    label="multi_amplitude_ms",
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

A quick plot helps catch measurement problems before fitting. Start with the
bias-amplitude step: each resonator has its own amplitude at step 1. Colour
shows amplitude and line style shows sweep direction. The magnitude plot
subtracts each trace's drive power; the IQ plot divides counts by its drive
amplitude so traces can be compared across steps.

```python
msplots.plot_magnitude_panels(
    multi_amplitude_results, iterations=1, ncols=4, batchlen=None,
)
```

Follow one resonator across all five amplitudes in magnitude and IQ.

```python
msplots.plot_magnitude_panels(
    multi_amplitude_results, names=first_resonator, batchlen=None,
)
msplots.plot_iq_panels(
    multi_amplitude_results, names=first_resonator, batchlen=None,
)
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
    label="fine_multisweep",
))[crs.module[MODULE].index()]

print(f"{len(fine_multisweep['results'][0]['upward'])} sweeps, "
      f"{40e3 / (201 - 1):.0f} Hz between points")
```

Compare the fine sweep's IQ loops and magnitude with the wider sweep above.
The closer frequency spacing traces each loop in more detail. With one step,
there is no amplitude ladder to compare, so show received power and raw IQ.

```python
msplots.plot_iq_panels(
    fine_multisweep, normalize=False, ncols=4, batchlen=None,
)
msplots.plot_magnitude_panels(
    fine_multisweep, normalize=False, ncols=4, batchlen=None,
)
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
results under each section’s `fits` key. The returned dictionary contains
`schema_version`, `fits`, and `settings`. Each fit row records its coordinates,
model, and failure reason (`None` on success).

```python
from rfmux.tuning import fit_sweeps

fit_report = fit_sweeps(multi_amplitude_results)

print(f"{len(fit_report['fits'])} model fits completed")
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
| `fit_nonlinearity` | `True` | nonlinear | Fit `a`; otherwise constrain it near zero |
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
fits = fit_report["fits"]
print(f"total fits    {len(fits)}")
print(f"fitted        {sum(f['failed_because'] is None for f in fits)}")
print(f"failed        {sum(f['failed_because'] is not None for f in fits)}")
print(f"skewed only   {sum(f['model'] == 'skewed' for f in fits)}")

print("\nsettings:")
for key, value in fit_report["settings"].items():
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
                ├── sweep_amplitude         │
                ├── sweep_amplitude_dbm     ╯
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

The fit plotter uses points for measured samples and lines for model curves.
It evaluates the stored model on a finer frequency grid. The magnitude panels
normalize both models and data to the last measured point. Plot each sweep
direction separately; a failed fit still shows its data and failure reason.

```python
for direction in ("upward", "downward"):
    fitplots.plot_fit_panels(
        multi_amplitude_results, model="skewed", iterations=0,
        direction=direction, ncols=4, batchlen=None,
    )
```

Next, fit the finer sweep. The nonlinear model uses complex IQ data; its fit
panel compares the model with the measured magnitude. The circle panel draws
the fitted circle around the measured IQ loop.

```python
fit_sweeps(fine_multisweep)

print(f"{first_resonator} fits: "
      f"{list(fine_multisweep['results'][0]['upward'][first_resonator]['fits'])}")
fitplots.plot_fit_panels(
    fine_multisweep, model="nonlinear", names=first_resonator,
)
fitplots.plot_fit_panels(
    fine_multisweep, model="circle", names=first_resonator,
)
```

The nonlinear fit stores a complex readout gain. Divide the measured IQ by
that gain to inspect the data in the coordinates used by the fitter.

```python
section = fine_multisweep["results"][0]["upward"][first_resonator]
fit = section["fits"]["nonlinear"]
if fit["failed_because"] is None:
    corrected = section["iq_counts"] / fit["gain"]
    plt.plot(corrected.real, corrected.imag, ".")
    plt.gca().set_aspect("equal", "datalim")
    plt.xlabel("I / gain")
    plt.ylabel("Q / gain")
    plt.title(f"{first_resonator}: gain-corrected IQ")
    plt.show()
else:
    print(fit["failed_because"])
```

The nonlinearity parameter `a` is zero for a linear resonator; bifurcation is
expected near `a ≈ 0.77`. At low drive, the fitted value should be near zero.

The circle fit stores a centre and radius. Subtract the centre from `iq_counts`
to place the loop around the origin before interpreting phase around the loop.

```python
fit = section["fits"]["circle"]
if fit["failed_because"] is None:
    centred = section["iq_counts"] - fit["center"]
    plt.plot(centred.real, centred.imag, ".")
    plt.gca().set_aspect("equal", "datalim")
    plt.xlabel("I − centre [counts]")
    plt.ylabel("Q − centre [counts]")
    plt.title(f"{first_resonator}: centred IQ")
    plt.show()
else:
    print(fit["failed_because"])
```

## 6. Select sweeps to fit

By default, `fit_sweeps()` fits all sections. For a larger array, select the
resonators, steps, directions, or models you need.

Copy the results and remove their fits so we can see what each selection adds.
Use `save=False` for these comparisons: a deep copy still carries the original
file path in `file_metadata`, so autosaving it would overwrite that measurement.

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
    unfitted_results, save=False,
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
fit_sweeps(
    unfitted_results, names=second_resonator, iterations=0,
    models=("skewed",), save=False,
)
sweep_section = unfitted_results["results"][0]["upward"][second_resonator]
print(f"after skewed:            {list(sweep_section['fits'])}")

fit_sweeps(
    unfitted_results, names=second_resonator, iterations=0,
    models=("circle",), save=False,
)
print(f"after circle:            {list(sweep_section['fits'])}  ← skewed kept")
```

### Fit at each resonator’s bias amplitude

`fit_sweeps_at_bias_amplitude()` selects the nearest measured amplitude for each
resonator. By default, it reads bias amplitudes from the catalog snapshot in
`call_params`. Adding a `bias_report` does not replace that snapshot. To fit
an amplitude selected by a new report, call `fit_sweeps()` with
`names=finding.name` and `iterations=finding.iteration` for each finding.
See `bias_finding.md` for choosing operating amplitudes.

```python
from rfmux.tuning import fit_sweeps_at_bias_amplitude

at_bias_report = fit_sweeps_at_bias_amplitude(
    unfitted_results, save=False,
    directions="upward",
    models=("skewed",),
)

print(at_bias_report)
print()
for fit in at_bias_report["fits"]:
    measured_at = unfitted_results["results"][fit['iteration']][fit['direction']][fit['name']]["sweep_amplitude"]
    print(f"{fit['name']}  biased at {catalog[fit['name']].bias.amplitude:.5f}  "
          f"→ step {fit['iteration']}, measured at {measured_at:.5f}")
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
    unfitted_results, save=False,
    amplitude=0.002,
    directions="upward",
    models=("circle",),
)
for fit in fixed_report["fits"]:
    measured_at = unfitted_results["results"][fit['iteration']][fit['direction']][fit['name']]["sweep_amplitude"]
    print(f"0.00200 for {fit['name']}  → step {fit['iteration']} "
          f"(actually {measured_at:.5f})")
```

Matching selects the nearest amplitude, even if it is far from the request.
Check the selected section’s `sweep_amplitude` when closeness matters.

## 7. Follow fitted parameters across amplitudes

Fits let us compare how frequency and Q change with drive. Overlay the
skewed model on each amplitude for one resonator. Colour shows amplitude,
points show measured data, and lines show the fitted model. Plot the two sweep
directions separately.

The measured points are about 2 kHz apart and may miss the dip minimum.
A fitted curve can extend below them; inspect fit quality before treating
that depth as a reliable estimate.

```python
for direction in ("upward", "downward"):
    fitplots.plot_fit_panels(
        multi_amplitude_results, model="skewed", names=first_resonator,
        direction=direction,
    )
```

Plot the fitted parameter curves against drive amplitude. Failed fits leave
gaps. Frequency shifts are relative to the lowest drive with a usable fit.

```python
for direction in ("upward", "downward"):
    fitplots.plot_fitted_parameters(
        multi_amplitude_results, model="skewed", direction=direction,
        batchlen=None,
    )
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
    fussy_results, save=False,
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

The comparison fits must leave the saved measurement unchanged. Reload its
first sweep’s fits and compare it with the original result:

```python
saved_path = store.saved_path(multi_amplitude_results)
saved_results = store.load(saved_path)[crs.module[MODULE].index()]
np.testing.assert_equal(
    saved_results["results"][0]["upward"][first_resonator]["fits"],
    multi_amplitude_results["results"][0]["upward"][first_resonator]["fits"],
)
print("Comparison fits left the saved measurement unchanged.")
```

## 9. Saving and next steps

- **Save fitted data:** `fit_sweeps()` and `fit_sweeps_at_bias_amplitude()` use
  the configured autosave setting. They save the modified sweeps, updating the
  original file if one exists. Pass `save=False` to skip saving, or `label=`
  when creating a new file. The `fits` dictionaries are included. To keep a
  separate analysis variant, use `store.save(copy_of_sweeps, "multisweep",
  new=True)` before fitting it; `directory=` alone does not change a saved path.
- **Keep fit settings:** settings belong to the report, not each section.
  Save `fit_report` directly to retain the report separately from the sweeps.
  Each row in `fit_report["fits"]` records the sweep coordinates, model, and
  `failed_because` (`None` on success). No class conversion is needed.
- **Calibrate frequency shifts:** current df calibration uses IQ derivatives
  measured at the bias point. These examples do not derive it from a fit.
  See `bias_finding.md`.
- **Choose operating amplitudes:** `rfmux.tuning.find_bias_points` uses the
  amplitude sweeps to select bias points and return an updated catalog.
- **Use fitting in a GUI:** `progress_callback(completed, total)` can drive a
  progress bar while the headless fitter analyzes completed sweeps.
