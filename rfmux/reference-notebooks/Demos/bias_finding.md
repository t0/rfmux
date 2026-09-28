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

In day-to-day testing, bias finding is a short, repeatable part of a larger
experiment: load or build a resonator catalog, take some multisweeps, inspect and refine
the operating points, then apply the bias points now stored in the catalog to the array,
and continue with noise or other
measurements. Later, perhaps repeat a short sweep to check for drift and update calibration as needed.

This workbook demonstrates that sequence with three measurements: an initial first pass multisweep
to take a quick look at the array response, a second multisweep over a wide range of amplitudes that reaches bifurcation, and a finer
search over a smaller amplitude range. We plot each one and base our decisions for the next
measurement based on the outcomes of the first. We pass a resonator catalog (see `resonator_catalogs.md` for more
info on these) to each measurement, refine it based on the measurement, and then 
pass it on to the next one. 

The second half of the notebook looks at how the bias finding functions work in a bit more detail, using the data collected during the example workflow.

Run this workbook from top to bottom. The example uses a
simulated board and a simulated array ("mock" mode).
See [opening reference notebooks](../README.md#opening-them) for setup and
saving your own editable copy.

## 1. Exploratory multisweep

We begin with a
120 kHz frequency span around the known locations of our resonances, using a low sweep amplitude, 0.001 of DAC full scale. This gives an initial view of the resonances and refines our knowledge of their locations in frequency space.


```python
%matplotlib inline


import rfmux
from rfmux.core.resonators import ResonatorCatalog
from rfmux.tuning import AmplitudeSchedule, find_bias_points, store
import example_plotting_multisweep as msplots
import example_plotting_bias as biasplots

MODULE = 1
print(f"results: {store.session_directory()}")
```

```python
mock_config = {
    "num_resonances": 4,
    "freq_start": 0.6e9,
    "freq_end": 0.9e9,
    "resonator_random_seed": 42,
    "auto_bias_kids": True,
    "bias_amplitude": 0.001,
    "pulse_mode": "none",
    "tls_noise_enabled": False,
    "nqp_noise_std_factor": 0.001,
}

session = rfmux.load_session("""
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
""")
crs = session.query(rfmux.CRS).one()
await crs.resolve()
resonator_count, _ = await crs.generate_resonators(mock_config)

# Board tone frequencies are relative to the NCO; catalogs use absolute Hz.
nco_frequency_hz = await crs.get_nco_frequency(module=MODULE)
resonator_frequencies_hz = [
    nco_frequency_hz + await crs.get_frequency(channel=channel, module=MODULE)
    for channel in range(1, resonator_count + 1)
]
initial_catalog = ResonatorCatalog.from_frequencies(
    resonator_frequencies_hz, module=MODULE, amplitude=0.001,
)
module_id = crs.module[MODULE].index()

coarse_multisweep = await crs.multisweep(
    initial_catalog,
    span_hz=120e3,
    npoints_per_sweep=101,
    nsamps=10,
    amp=0.001,
    sweep_direction=("upward"),
    save=True,
    label="initial_wide_quick",
)
coarse_multisweep_module_outputs = coarse_multisweep[module_id]
```

**Note**: Measurements autosave by default, following the store settings. Bias analysis adds
its report to the measurement and, when saving is enabled, updates that file.
Pass `save=False` to measurement algorithms or processing functions to disable this.

## 1b. Inspect the initial sweep

You can of course open and plot the dictionary outputs from multisweep as you wish; here we use the example plotters for convenience.

```python
msplots.plot_magnitude_panels(coarse_multisweep_module_outputs, ncols=4)
```

The resonators look reasonable: they are well centered in the multisweep bandwidths and as expected, none of them is bifurcated.

We now run the bias finder on this measurement. With only one amplitude available in a measurement, the bias finder must return that amplitude as the "ideal" bias amplitude for each resonator. While it is unlikely this amplitude will be the one we end up using for any of these resonators, it provides a starting point. The bias finder always chooses the amplitude first, and then looks for the best bias frequency within the sweep taken at that amplitude. So in this case, we are really only using it for its frequency selection.

Also, note that because the bias finder is a separate analysis step to the multisweep measurement, you can run the bias finder on a given measurement multiple times, rather than needing to re-measure each time.

```python
coarse_bias_report = find_bias_points(coarse_multisweep_module_outputs)

msplots.plot_magnitude_panels(coarse_multisweep_module_outputs, ncols=4)
```

The bias finder has identified a frequency in each of the resonators, and has added a bias report (identical to the one it returned) to the multiweep. The contents of this new catalog are now indicated in the plots. These look like reasonable choices for the bias frequencies, so we will extract this new catalog from the bias report and feed it into the next multisweep.


## 2. Iteratively multisweep over a broad range of amplitudes

For the next sweep, we will use a smaller 70 kHz span, and five
amplitudes running from 1 to 4 times the starting amplitude. Hopefully
these amplitudes will span the bifurcation point for all resonators.

```python
coarse_bias_catalog = coarse_bias_report.catalog

bifurcation_multisweep = await crs.multisweep(
    coarse_bias_catalog,
    span_hz=70e3,
    npoints_per_sweep=101,
    nsamps=10,
    amp=AmplitudeSchedule.multiplicative(1.0, 4.0, 5),
    sweep_direction=("upward", "downward"),
    label="bias_amplitude_coarse_search",
)

```

```python
bifurcation_multisweep_module_outputs = bifurcation_multisweep[module_id]
bifurcation_bias_report = find_bias_points(bifurcation_multisweep_module_outputs)

msplots.plot_magnitude_panels(bifurcation_multisweep_module_outputs, ncols=2)
```

This looks successful: each resonator is bifurcated at the higher amplitudes, and very far from bifurcation at the lower amplitudes. The bias finder identifies the bias amplitude by looking for bifurcation, then selecting the sweep amplitude which was one index below the bifurcation amplitude. So here it has identified one of the middle sweeps for each resonance.

We can take a closer look at the sweep at the chosen amplitude:

```python
biasplots.plot_bias_points(bifurcation_multisweep_module_outputs, ncols=2)
```

These all look reasonable, but we can probably refine things a bit more.
So like before, we extract the updated catalog from the bias report, and feed that into the next multisweep.

Note that if you wanted instead to work with the original catalog, it has been
preserved unchanged under the `call_params` key inside the multisweep.

## 3. Sweep again with a smaller range of amplitudes around the selected points

We also reduce the frequency span to 50 kHz and increase the sample count from
101 to 151 frequency points. This resolves the response more closely around the
updated frequencies, but is slower, so we chose to not start off with these finer 
settings, to improve efficiency.

```python
bifurcation_bias_catalog = bifurcation_bias_report.catalog

refined_multisweep = await crs.multisweep(
    bifurcation_bias_catalog,
    span_hz=50e3,
    npoints_per_sweep=151,
    nsamps=10,
    amp=AmplitudeSchedule.multiplicative(1.0, 2**0.5, 6),
    sweep_direction=("upward", "downward"),
    label="bias_amplitude_refinement",
)

```

```python
refined_multisweep_module_outputs = refined_multisweep[module_id]
refined_bias_report = find_bias_points(refined_multisweep_module_outputs)

msplots.plot_magnitude_panels(refined_multisweep_module_outputs, ncols=2, xlim_khz=(-10, 10))
```

Looks good. Let's inspect the sweep at the chosen bias amplitude:

```python
biasplots.plot_bias_points(refined_multisweep_module_outputs, ncols=2)

```

You could now fit these sweeps to extract the nonlinearity parameter `a` for each resonance at its chosen bias point; the diagnostics at the end of this workbook do that for every refined sweep. See `fitting_resonators.md` for more info on the built-in fitting options.

If you want, feed the catalog into another round of multisweeps to refine the bias point further. If you are happy with the bias points that have been found, we're ready to apply these biases to the array.



## 4. Done - bias the array!



```python
catalog = refined_bias_report.catalog

catalog_path = await crs.apply_bias(catalog)
print(f"applied catalog saved to {catalog_path}")
```

<!-- #region -->
This programs the tones and, by default, saves the catalog as a pickled dictionary,
returning the file's path. It does not make any decisions or changes, simply sets
the chosen tone frequencies and amplitudes.

The catalog already
carries the voltage slopes and source sweep used for calibration. It extracted these
from the measured sweep at the chosen bias amplitude, at the chosen bias frequency.
No separate
calibration measurement is needed for this workflow.

You're now ready to start streaming detector data! :tada:



## 5. 10 hours later...

Maybe you went for a long lunch, maybe the ADR cycled... now you want to resume 
measurements.


If the system is in nominally the same or a quite simiar state to where you were before, it's probably a good idea to check the state of the biases, but probably not necessary to start the bias finding procedure from scratch. In that case, a follow-up multisweep can check frequency drift and refresh calibration at the
chosen amplitudes. 

If you still have the catalog loaded in your ipython session, you can simply pass that
to the multisweep. If not, you can load it from the saved catalog pkl file that `apply_bias` 
generated (or extract it from the last multisweep you ran before applying the bias).

<!-- #endregion -->

```python
# load the saved resonator catalog from when we applied the biases
catalog = ResonatorCatalog.from_dict(store.load(catalog_path))

# run a new multisweep using it, to check on the state of the array
confirmation_multisweep = await crs.multisweep(
    catalog, span_hz=70e3, npoints_per_sweep=151,
    sweep_direction=("upward",), label="bias_confirmation",
)

```

```python
confirmation_multisweep_module_outputs = confirmation_multisweep[module_id]
confirmation_bias_report = find_bias_points(confirmation_multisweep_module_outputs)

msplots.plot_magnitude_panels(confirmation_multisweep_module_outputs, ncols=4)

biasplots.plot_bias_points(confirmation_multisweep_module_outputs, ncols=2)
```

Looks good!

```python

```

<!-- #region -->


# More info on how the bias finder operates 




These comparisons reuse the refined measurement without collecting new data.
Pick one resonator to keep the diagnostic plots readable. The example selects
one flagged resonator if available. To choose another, use a name from
`refined_bias_report.catalog.names()`.
<!-- #endregion -->

```python


resonator_name = (
    refined_bias_report.flagged[0].name
    if refined_bias_report.flagged else refined_bias_report.catalog.names()[0]
)
biasplots.plot_bias_points(
    refined_multisweep_module_outputs,
    names=resonator_name, projection="iq",
)
```

The ring marks the measured sample nearest the chosen bias frequency on the IQ
loop. For several resonators, pass a list to `names`.

### Bifurcation detection: derivative method

The default method looks for a jump in the IQ trace. I and Q are each divided by
their range over the sweep, and the distance between neighbouring points divided
by the frequency step gives the arc speed: how fast the sweep moves around the
IQ loop. A bifurcated sweep snaps from one branch to the other between two
frequency points, so its arc speed shoots up and straight back down. In the
point-to-point change of arc speed, that shows as a positive spike followed
within one or two points by a negative one. A resonance that has not bifurcated
also speeds up and slows down, but gradually, so its rise and fall are spread
over many points.

Both spikes must clear a threshold, which is the larger of two bars:

* **the prominence bar**, `spike_prominence_factor` times the full range of the
  arc speed over the sweep. A jump has to be a large share of everything the
  sweep does, however strongly it is driven.
* **the noise bar**, `noise_gate_factor` times the noise in the arc-speed
  changes, estimated from their median absolute deviation so the jump itself
  hardly moves it. On a sweep with little signal, the range is set by noise too,
  and a fraction of it is easy for a noise spike to clear; this bar rules that
  out.

The sweep is called bifurcated if any direction has a qualifying pair.

Both factors are parameters of `find_bias_points`: `spike_prominence_factor`
(default 0.5) and `noise_gate_factor` (default 50; 0 disables the noise bar).

For example, we can examine the effects of changing the noise threshold on one of our previously measured
multisweeps. First, the default:


```python

biasplots.plot_bifurcation_checks(
    refined_multisweep_module_outputs,
    names=resonator_name,
)
```

The selected amplitude is bold. The ±1 lines show the derivative test's effective
threshold; its legend identifies whether prominence or noise sets that threshold.
The detector also checks spike prominence and adjacency, so a line crossing
alone is not a bifurcation verdict.

### Cross-check: the fitted nonlinearity `a`

The nonlinear resonator model measures how far the drive has bent a resonance
with one parameter, `a`: zero for a linear resonator, rising in proportion to
drive power, and bifurcating at `a = 4√3/9 ≈ 0.77` (`BIFURCATION_A`). Fitting
every sweep of the refined multisweep gives a measure of how close each step
came to bifurcation that does not depend on the derivative test, so the two
can be compared step by step.

```python
import matplotlib.pyplot as plt
from rfmux.tuning import collect_fit_params, fit_sweeps
from rfmux.tuning.fits import BIFURCATION_A

fit_sweeps(refined_multisweep_module_outputs, models=("nonlinear",))
nonlinear_fits = [
    row for row in collect_fit_params(refined_multisweep_module_outputs, "nonlinear")
    if row["failed_because"] is None
]

for finding in refined_bias_report.findings:
    print(finding.name)
    for row in nonlinear_fits:
        if row["name"] != finding.name or row["direction"] != "upward":
            continue
        step, a = row["iteration"], row["params"]["a"]
        check = finding.checks.get(step)
        verdict = ("not examined" if check is None
                   else "bifurcated" if check.bifurcated else "not bifurcated")
        chosen = "  <- bias" if step == finding.iteration else ""
        print(f"  step {step}  amp {row['amplitude']:.5f}  a = {a:.3f} "
              f"({a / BIFURCATION_A:.0%} of bifurcation)  derivative: {verdict}{chosen}")
```

The table reads the upward fits; the downward ones agree closely until the
sweep is bistable. The plot shows both, against drive amplitude. An open circle
marks the step chosen for the bias point, a cross the first step the derivative
test called bifurcated.

```python
fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
for colour, finding in zip(plt.cm.tab10.colors, refined_bias_report.findings):
    for direction, linestyle in (("upward", "-"), ("downward", "--")):
        rows = [row for row in nonlinear_fits
                if row["name"] == finding.name and row["direction"] == direction]
        ax.plot([row["amplitude"] for row in rows],
                [row["params"]["a"] for row in rows],
                marker=".", ls=linestyle, color=colour,
                label=finding.name if direction == "upward" else None)
        for row in rows:
            if direction != "upward":
                continue
            check = finding.checks.get(row["iteration"])
            if row["iteration"] == finding.iteration:
                ax.plot(row["amplitude"], row["params"]["a"], "o", ms=14,
                        mfc="none", mew=2, color=colour)
            if check is not None and check.bifurcated:
                ax.plot(row["amplitude"], row["params"]["a"], "x", ms=12,
                        mew=2.5, color=colour)
ax.axhline(BIFURCATION_A, color="0.3", lw=1.5, ls=":",
           label=f"BIFURCATION_A = {BIFURCATION_A:.3f}")
ax.set_xlabel("drive amp. [norm.]")
ax.set_ylabel("fitted nonlinearity a")
ax.set_title("Refined multisweep: fitted a, upward solid, downward dashed")
ax.legend()
plt.show()
```

If the derivative test is well tuned, its first bifurcated step lands as `a`
reaches `BIFURCATION_A`, and the chosen bias step sits just below it. A first
verdict at `a` well below 0.77 means the test calls the jump early and gives up
drive; one well above means it misses the jump. The fitter bounds `a` at 0.9,
so a fit pinned there is above bifurcation and says no more than that.

To explore sensitivity, change one setting and rerun on the same sweeps.
Larger `spike_prominence_factor` or `noise_gate_factor` values make detection
less sensitive. Here we lower the noise gate from its default of 50 to 20:

```python
# optionally duplicate the existing multisweep to avoid overwriting its bias info, just in case
import copy
noise_gate_demo_ms_output = copy.deepcopy(refined_multisweep_module_outputs)

# run the bias finder again, with a different noise threshold factor
noise_gate_bias_report = find_bias_points(
    noise_gate_demo_ms_output,
    noise_gate_factor=20.0, save=True,
)
biasplots.plot_bifurcation_checks(
    noise_gate_demo_ms_output,
    names=resonator_name,
)
biasplots.plot_bias_points(noise_gate_demo_ms_output, names=resonator_name)
```

Copying the module dictionary lets this comparison store a separate report
while sharing the unchanged sweep arrays. `save=False` leaves the saved
measurement unchanged. Compare the chosen point and diagnostic with the
originals above. Changing the threshold need not change the selected point.

With **both sweep directions**, you can instead compare up/down separation:

```python
if {"upward", "downward"} <= set(refined_multisweep_module_outputs["results"][0]):
    hysteresis_multisweep_module_outputs = dict(refined_multisweep_module_outputs)
    find_bias_points(
        hysteresis_multisweep_module_outputs,
        amplitude_method="hysteresis", save=False,
    )
    biasplots.plot_hysteresis_checks(
        hysteresis_multisweep_module_outputs,
        names=resonator_name,
    )
    biasplots.plot_bias_points(
        hysteresis_multisweep_module_outputs,
        names=resonator_name,
    )
```

Values above 1 exceed the allowed separation; the selected amplitude is bold.
The two methods can disagree: a jump can be visible in each sweep even when
the upward and downward traces nearly coincide. Hysteresis alone may then
select the highest tested amplitude and flag it. This comparison leaves the
refined catalog unchanged.

`amplitude_method="both"` detects bifurcation when **either** the derivative
or hysteresis test fires. Both `"hysteresis"` and `"both"` need upward and
downward sweeps; `"derivative"` also works with just one direction.

### Compare maximum IQ sensitivity with the magnitude minimum

`frequency_method="iq_derivative"` selects the largest IQ change per hertz;
`"minimum"` selects the magnitude dip's minimum. Plot the IQ sensitivity of the
selected sweep, then compare the two operating points:

```python
selected_iteration = refined_bias_report[resonator_name].iteration
measured_directions = refined_multisweep_module_outputs["results"][selected_iteration]
sweep_direction = "upward" if "upward" in measured_directions else "downward"
biasplots.plot_arc_speed_panels(
    refined_multisweep_module_outputs, names=resonator_name,
    iterations=selected_iteration, direction=sweep_direction,
)

minimum_multisweep_module_outputs = dict(refined_multisweep_module_outputs)
find_bias_points(
    minimum_multisweep_module_outputs,
    frequency_method="minimum", save=False,
)
biasplots.plot_bias_points(
    refined_multisweep_module_outputs,
    names=resonator_name, title="Maximum IQ sensitivity",
)
biasplots.plot_bias_points(
    minimum_multisweep_module_outputs,
    names=resonator_name, title="Magnitude minimum",
)
```

The sensitivity maximum can be away from the dip minimum. Look at its width as
well as its height: a narrow feature leaves less room for frequency drift.

These are the main controls to explore; use `help(find_bias_points)` for the
full argument reference.

| Setting | Default | When to change it |
|---|---|---|
| `amplitude_method` | `"derivative"` | Compare jump detection with `"hysteresis"` or `"both"` |
| `frequency_method` | `"iq_derivative"` | Try the dip `"minimum"` instead |
| `noise_gate_factor` | `50.0` | Adjust rejection of noise-like jumps |
| `spike_prominence_factor` | `0.5` | Adjust the required jump prominence |
| `max_discrepancy` | `0.1` | Adjust allowed up/down separation for hysteresis |
| `max_distance_hz` | `None` | Limit the selected frequency's distance from the sweep centre |

A frequency beyond `max_distance_hz` falls back to the sweep centre and is
flagged (the report records the first concern when several apply). Review the
trace for a neighbouring resonance or a dip outside the window before changing
the limit or remeasuring. The limit is optional; there is no default bound.
