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

For the next sweep, we will use a smaller 50 kHz span, and five
amplitudes running from 1 to 4 times the starting amplitude. Hopefully
these amplitudes will span the bifurcation point for all resonators.

```python
coarse_bias_catalog = coarse_bias_report.catalog

bifurcation_multisweep = await crs.multisweep(
    coarse_bias_catalog,
    span_hz=50e3,
    npoints_per_sweep=151,
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

We also reduce the frequency span to 30 kHz and increase the sample count from
151 to 251 frequency points. This resolves the response more closely around the
updated frequencies, but is slower, so we chose to not start off with these finer
settings, to improve efficiency.

```python
bifurcation_bias_catalog = bifurcation_bias_report.catalog

refined_multisweep = await crs.multisweep(
    bifurcation_bias_catalog,
    span_hz=30e3,
    npoints_per_sweep=251,
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
from rfmux.tuning import fit_sweeps

fit_sweeps(refined_multisweep_module_outputs, models=("nonlinear",),
           directions="upward")

fig, axes = plt.subplots(1, len(refined_bias_report.findings),
                         figsize=(5 * len(refined_bias_report.findings), 4),
                         constrained_layout=True, squeeze=False)
for panel, finding in zip(axes.flat, refined_bias_report.findings):
    entry = refined_multisweep_module_outputs["results"][finding.iteration]["upward"][finding.name]
    fit_results = entry['fits']['nonlinear']
    failure = fitplots.draw_measured_and_model(
        panel, entry, "nonlinear", "magnitude", "0.55", "firebrick", 25,
    )

    panel.axvline((finding.frequency_hz - entry["original_center_frequency"]) / 1e3,
                  color="royalblue", label="bias frequency")
    a = (fit_results.get("params") or {}).get("a")
    detail = failure or (f"a = {a:.3f}" if a is not None else "no fit parameter a")
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
    catalog, span_hz=50e3, npoints_per_sweep=251,
    sweep_direction=("upward",), label="bias_confirmation",
)

```

```python
confirmation_multisweep_module_outputs = confirmation_multisweep[module_id]
confirmation_bias_report = find_bias_points(confirmation_multisweep_module_outputs)

msplots.plot_magnitude_panels(confirmation_multisweep_module_outputs, ncols=4)
biasplots.plot_bias_points(confirmation_multisweep_module_outputs, ncols=2)
```

<!-- #region -->

# More info on how the bias finder operates

<!-- #endregion -->

<!-- #region -->


### Bifurcation detection: derivative method

For each sweep direction, the derivative method sorts samples into ascending
frequency and scales I and Q by their respective peak-to-peak ranges, $R_I$ and
$R_Q$. It then calculates the normalized IQ distance between consecutive samples,
divided by their frequency separation:

$$
s_i = \frac{\sqrt{(\Delta I_i/R_I)^2 + (\Delta Q_i/R_Q)^2}}{\Delta f_i}.
$$

A discontinuity produces a narrow peak in this IQ speed: speed rises abruptly
as the sweep crosses the jump, then falls afterward. To find that rise and fall,
the detector takes successive changes in speed:

$$
d_i = s_{i+1} - s_i.
$$

This difference is **not divided by frequency spacing again**. Both $s$ and $d$
have units of inverse Hz; $d$ is a change between speed samples, rather than a
second derivative with respect to frequency.

The detector finds peaks in $d$ and troughs by finding peaks in $-d$. An eligible
pair is a peak followed by a trough **one or two indices later**. Each member's
prominence measures how far it stands out from its surrounding base, not its
height or depth relative to zero. The pair's strength is its weaker prominence:

$$
P_{\mathrm{pair}} = \min(P_+, P_-).
$$

The detector selects the strongest eligible pair in that sweep:

$$
P_{\mathrm{best}} = \max_{\text{eligible pairs}} P_{\mathrm{pair}}.
$$

**The two thresholds** set the minimum acceptable pair strength. Let $p$ be
`spike_prominence_factor` and $g$ be `noise_gate_factor`:

| Control | Default | Threshold | Intended role |
|---|---|---|---|
| `spike_prominence_factor` ($p$) | 0.5 | $T_{\mathrm{shape}} = p\,(\max s - \min s)$ | Require an abrupt change substantial relative to the sweep's speed range |
| `noise_gate_factor` ($g$) | 50 | $T_{\mathrm{noise}} = g\,\hat\sigma_d$ | Require the pair to stand well above background variation |

The background scatter estimate is:

$$
\hat\sigma_d = 1.4826\,\mathrm{median}\!\left(\left|d - \mathrm{median}(d)\right|\right).
$$

This estimate limits the influence of a few large spikes. It includes real
resonance curvature as well as measurement noise, so a factor of 50 is **not a
calibrated 50-sigma detection significance**.

An eligible pair triggers detection when its strength reaches **both** thresholds:

$$
P_{\mathrm{best}} \geq \max(T_{\mathrm{shape}}, T_{\mathrm{noise}}).
$$

If no eligible pair exists, there is no detection, even when both thresholds
are zero. The diagnostic plot shows zero strength in that case.

Raise either factor to demand stronger evidence; lower it to admit weaker pairs.
Only the larger threshold controls the decision, so lowering the smaller one
has no effect while the larger one stays unchanged. Setting `noise_gate_factor`
to zero disables the noise requirement.

Either sweep direction can trigger detection. The amplitude search selects the
measured amplitude below the first detected bifurcation. If the lowest amplitude
already bifurcates, it selects that amplitude and flags it; if none bifurcate,
it selects the highest measured amplitude and flags that no bifurcation was
observed.

<!-- #endregion -->

```python
biasplots.plot_bifurcation_checks(
    refined_multisweep_module_outputs)
```

Here we are plotting the quantities examined by the bifurcation detector at each
sweep amplitude. Blue circles show pair strength,
orange triangles the shape threshold, and green squares the noise threshold,
all in inverse Hz. A pair triggers when its strength quantity matches or exceeds both threshold curves.
The vertical dotted line marks the selected
amplitude.

To inspect where the strongest pair occurs on the frequency sweep, use the raw
speed-change plot. Filled circles mark a detected pair; open circles mark a
pair below threshold. There are no horizontal threshold lines because prominence
is measured from a surrounding base, not from zero.

```python
biasplots.plot_arc_speed_panels(
    refined_multisweep_module_outputs,
    quantity="spikes", xlim_khz=(-5, 5),
    spike_prominence_factor=0.5, noise_gate_factor=50.0,
)
```

The same information is available numerically. The report contains only the
amplitude steps examined before the search stopped:

```python
finding = next(f for f in refined_bias_report.findings)
for step, check in finding.checks.items():
    derivative = check.parts.get("derivative", check)
    details = derivative.diagnostics
    print(f"step {step}, {details['direction']}: "
          f"pair={derivative.metric['pair_strength']:.3g}, "
          f"shape={details['shape_threshold']:.3g}, "
          f"noise={details['noise_threshold']:.3g} 1/Hz; "
          f"detected={derivative.bifurcated}")
```

#### an example extra-noisy array

To make the effects of the noise gate easier to see, we can increase the additive noise level
in the mock resonator array, and average five samples per
frequency measurement instead of ten in the multisweeps.

```python
noisy_mock_config = {**mock_config, "udp_noise_level": 100.0}
await crs.generate_resonators(noisy_mock_config)

noisy_multisweep = await crs.multisweep(
    coarse_bias_catalog,
    span_hz=50e3,
    npoints_per_sweep=151,
    nsamps=5,
    amp=AmplitudeSchedule.multiplicative(1.0, 4.0, 5),
    sweep_direction=("upward", "downward"),
    label="bias_noise_gate_demo",
)
noisy_multisweep_module_outputs = noisy_multisweep[module_id]
msplots.plot_magnitude_panels(noisy_multisweep_module_outputs, ncols=2)

fit_sweeps(noisy_multisweep_module_outputs, models=("nonlinear",))
fitplots.plot_fitted_parameters(
    noisy_multisweep_module_outputs, model="nonlinear", parameters="a",
)
```

Run the bias finder on the same measurement with the noise gate disabled, with its default factor
of 50, and with a stronger factor of 150. Larger factors make bifurcation detection less
sensitive. The fitted nonlinearity parameter `a` above offers another view of
the amplitude ladder.

```python
noise_gate_factor = 0  # disable the noise gate entirely

report = find_bias_points(noisy_multisweep_module_outputs, noise_gate_factor=noise_gate_factor, save=False)
biasplots.plot_bifurcation_checks(noisy_multisweep_module_outputs,
                                    title=f"Noise gate factor {noise_gate_factor:g}")
msplots.plot_magnitude_panels(noisy_multisweep_module_outputs, ncols=4)
biasplots.plot_bias_points(noisy_multisweep_module_outputs,
                            xlim_khz=(-15, 15),
                            title=f"Noise gate factor {noise_gate_factor:g}")
```

With the gate disabled, the green noise threshold is zero. Noise can produce
strong pairs even at low amplitudes. Compare the blue pair strengths with the
orange shape threshold and inspect the measured traces before trusting the bias.

Turning the noise gate on to its default setting:

```python
noise_gate_factor = 50

report = find_bias_points(noisy_multisweep_module_outputs, noise_gate_factor=noise_gate_factor, save=False)
biasplots.plot_bifurcation_checks(noisy_multisweep_module_outputs,
                                    title=f"Noise gate factor {noise_gate_factor:g}")
msplots.plot_magnitude_panels(noisy_multisweep_module_outputs, ncols=4)
biasplots.plot_bias_points(noisy_multisweep_module_outputs,
                            xlim_khz=(-15, 15),
                            title=f"Noise gate factor {noise_gate_factor:g}")
```

At the default gate setting, the pair strengths and shape thresholds stay the
same; the green noise threshold rises. Detection requires a blue point to reach
both thresholds. Compare the selected amplitude with its measured sweep.

For completeness, below we run this again, but using a very high noise gate factor:

```python
noise_gate_factor = 150

report = find_bias_points(noisy_multisweep_module_outputs, noise_gate_factor=noise_gate_factor, save=False)
biasplots.plot_bifurcation_checks(noisy_multisweep_module_outputs,
                                    title=f"Noise gate factor {noise_gate_factor:g}")
msplots.plot_magnitude_panels(noisy_multisweep_module_outputs, ncols=4)
biasplots.plot_bias_points(noisy_multisweep_module_outputs,
                            xlim_khz=(-15, 15),
                            title=f"Noise gate factor {noise_gate_factor:g}")
```

At a high gate setting, even strong jumps can fall below the threshold. Use the
diagnostic plots to check how the chosen factor changes the classification and
selected bias amplitudes in this noise realization.

<!-- #region -->



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
        hysteresis_multisweep_module_outputs
    )
    msplots.plot_magnitude_panels(bifurcation_multisweep_module_outputs,
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
| `spike_prominence_factor` | `0.5` | Adjust required pair strength relative to the speed range |
| `max_discrepancy` | `0.1` | Adjust allowed up/down separation for hysteresis |
| `max_distance_hz` | `None` | Limit the selected frequency's distance from the sweep centre |
