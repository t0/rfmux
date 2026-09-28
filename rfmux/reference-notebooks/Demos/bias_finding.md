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

These look pretty good, but we can get a cross check by fitting each one.

```python
import matplotlib.pyplot as plt
import example_plotting_fits as fitplots
from rfmux.tuning import fit_section

fig, axes = plt.subplots(1, len(refined_bias_report.findings),
                         figsize=(5 * len(refined_bias_report.findings), 4),
                         constrained_layout=True, squeeze=False)
for panel, finding in zip(axes.flat, refined_bias_report.findings):
    entry = refined_multisweep_module_outputs["results"][finding.iteration]["upward"][finding.name]
    fit_section(entry, models=("nonlinear",))
    failure = fitplots.draw_measured_and_model(
        panel, entry, "nonlinear", "magnitude", "0.45", "crimson", oversample=25,
    )
    panel.axvline((finding.frequency_hz - entry["original_center_frequency"]) / 1e3,
                  color="royalblue", label="bias frequency")
    fit = entry["fits"]["nonlinear"]
    detail = failure or f"a = {fit['params']['a']:.3f}"
    panel.set_title(f"{finding.name}: step {finding.iteration}, {detail}")
    panel.set_xlabel("frequency from sweep centre [kHz]")
    panel.set_ylabel("magnitude [dB, normalized]")
    panel.set_xlim(-10, 10)
axes.flat[0].legend()
plt.show()
```

The grey points are the measured sweep, the red curve is the nonlinear fit,
and the blue line is the selected bias frequency. The fitted nonlinearity
parameter `a` gives another check on how strongly driven the selected sweep is;
the panel shows a failure reason if the fit does not pass. Let's inspect the
chosen bias points on the measured sweeps:

```python
biasplots.plot_bias_points(refined_multisweep_module_outputs, ncols=2)

```

See `fitting_resonators.md` for more on the built-in fitting options.

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

TODO: the above is still finding that most of these resonators are bifurcated at the bias points, even though on the previous multisweep they werent! Why?




```python

```

<!-- #region -->


# More info on how the bias finder operates 


Here we are picking one resonator to keep the diagnostic plots readable. To choose another, use a name from
`refined_bias_report.catalog.names()`.
<!-- #endregion -->

```python


resonator_name = refined_bias_report.catalog.names()[0]

biasplots.plot_bias_points(
    refined_multisweep_module_outputs,
    names=resonator_name, projection="iq",
)
```

The ring marks the measured sample nearest the chosen bias frequency on the IQ
loop. For several resonators, pass a list to `names`.

### Bifurcation detection: derivative method

The default test looks for a sudden jump in the IQ trace. It scales I and Q by
their range over the sweep, then measures how far the trace moves between each
pair of frequency points. A jump produces a sharp increase in that movement,
followed within two points by a sharp decrease. Smooth changes in a resonance
are more spread out. Either sweep direction can reveal a jump.

The increase and decrease must both stand out from the rest of the sweep. The
test uses the stricter of two limits:

* `spike_prominence_factor` sets the required jump relative to the range of
  movement across the sweep.
* `noise_gate_factor` sets the required jump relative to the background
  variation. This keeps noise in a weak sweep from looking like bifurcation.

Both factors are parameters of `find_bias_points`: `spike_prominence_factor`
(default 0.5) and `noise_gate_factor` (default 50; 0 disables the noise bar).

First, inspect the default threshold on the refined measurement:


```python

biasplots.plot_bifurcation_checks(
    refined_multisweep_module_outputs,
    names=resonator_name,
    xlim_khz=(-5,5)
)
```

The selected amplitude is bold. The ±1 lines show the derivative test's effective
threshold; its legend identifies whether prominence or noise sets that threshold.
The detector also checks spike prominence and adjacency, so a line crossing
alone is not a bifurcation verdict.





To make the noise gate easier to see, regenerate the mock array with 50 times
the earlier quasiparticle noise and average two samples per frequency instead
of ten. The random seed and resonance range stay the same, so we can use the
earlier coarse catalog for a similar amplitude ladder. This is a separate
measurement after the applied-bias example above.

```python
noisy_mock_config = {**mock_config, "nqp_noise_enabled": True,
                     "nqp_noise_std_factor": 0.08}
await crs.generate_resonators(noisy_mock_config)

noisy_multisweep = await crs.multisweep(
    coarse_bias_catalog,
    span_hz=70e3,
    npoints_per_sweep=101,
    nsamps=10,
    amp=AmplitudeSchedule.multiplicative(1.0, 4.0, 5),
    sweep_direction=("upward", "downward"),
    label="bias_noise_gate_demo",
)
noisy_multisweep_module_outputs = noisy_multisweep[module_id]
msplots.plot_magnitude_panels(noisy_multisweep_module_outputs, ncols=2)
```

Run the bias finder with the noise gate disabled, with its default factor
of 50, and with a stronger factor of 150. Larger factors make detection less
sensitive. Analyze copies of the same measured sweeps so a change in the result
comes from the gate setting, not from a new noise realization.

```python
noise_gate_results = {}
noise_gate_reports = {}
for factor in (0.0, 50.0, 150.0):
    measured = dict(noisy_multisweep_module_outputs)
    report = find_bias_points(measured, noise_gate_factor=factor, save=False)
    noise_gate_results[factor] = measured
    noise_gate_reports[factor] = report
    print(f"noise gate factor {factor:g}:")
    for finding in report.findings:
        print(f"  {finding.name}: step {finding.iteration}, "
              f"amplitude {finding.amplitude:.5f}")

# Focus the plots on a resonator whose chosen step changed, if there is one.
noise_gate_name = next(
    (finding.name for finding in noise_gate_reports[0.0].findings
     if finding.iteration != noise_gate_reports[150.0][finding.name].iteration),
    noise_gate_reports[0.0].findings[0].name,
)
for factor in (0.0, 150.0):
    measured = noise_gate_results[factor]
    biasplots.plot_bifurcation_checks(measured, names=noise_gate_name,
                                     xlim_khz=(-5, 5),
                                     title=f"Noise gate factor {factor:g}")
    biasplots.plot_bias_points(measured, names=noise_gate_name,
                              xlim_khz=(-10, 10),
                              title=f"Noise gate factor {factor:g}")
```

<!-- #region -->
Compare the printed amplitude steps and the threshold lines. On a noisy sweep,
the gate can reject noise-induced spike pairs that pass the prominence
threshold alone; the selected step can then move to a higher amplitude. The
exact result varies with each noise realization.


### Hysteresis method for bifurcation detection

If your multisweep contains **both sweep directions**, you can instead compare the up/down sweeps and look for when they start to become notably different due to the readout current hysteresis:
<!-- #endregion -->

```python
if {"upward", "downward"} <= set(bifurcation_multisweep_module_outputs["results"][0]):
    hysteresis_multisweep_module_outputs = dict(bifurcation_multisweep_module_outputs)
    find_bias_points(
        hysteresis_multisweep_module_outputs,
        amplitude_method="hysteresis", save=False,
    )
    biasplots.plot_hysteresis_checks(
        hysteresis_multisweep_module_outputs,
        names=resonator_name,
    )
    msplots.plot_magnitude_panels(bifurcation_multisweep_module_outputs, names=resonator_name,
                                   xlim_khz=(-10, 10))
```

Values above 1 are classified as bifurcation.

`amplitude_method="both"` detects bifurcation when **either** the derivative
or hysteresis test fires. Both `"hysteresis"` and `"both"` need upward and
downward sweeps; `"derivative"` also works with just one direction.




## For more info...

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
