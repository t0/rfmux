# Headless tuning branch: release notes

Branch `tuning_headless_revamp`, October 2026.

Resonator tuning runs from a script or a notebook as well as from Periscope:
finding resonances, sweeping them over a schedule of readout powers, fitting,
choosing bias points, programming them and measuring noise. Periscope calls
the same functions. A `ResonatorCatalog` names each resonator and carries its
channel, frequency and amplitude from one step to the next, to the board, to
the saved files and to pulse capture. Every step saves its result through
`rfmux.tuning.store`, into the session folder when Periscope has one open.

[simplified_tuning_flow.md](../../rfmux/reference-notebooks/Demos/simplified_tuning_flow.md) runs the whole
flow on the simulator from top to bottom. Each step below has a notebook of
its own.

## The tuning flow

### Find resonances

```python
netanal = await crs.take_netanal(module=1, amp=0.001, fmin=1.0e9, fmax=1.1e9)
search = find_resonances_in_netanal(netanal[crs.module[1].index()])
catalog = search.to_catalog(module=1, amplitude=0.004)
```

The catalog gives each resonator a generated name (`BOTA`), a channel in
frequency order and a frequency on the hardware tone grid.
`numbered_names` gives `R0001…` instead, and a catalog can be built from a
list of frequencies with `ResonatorCatalog.from_frequencies`. See
[network_analysis_find_resonances.md](../../rfmux/reference-notebooks/Demos/network_analysis_find_resonances.md)
and [resonator_catalogs.md](../../rfmux/reference-notebooks/Demos/resonator_catalogs.md).

### Sweep over readout powers

```python
sweeps = await crs.multisweep(
    catalog, amp=AmplitudeSchedule.multiplicative(0.5, 8.0, 5),
    sweep_direction=("upward", "downward"))
```

`AmplitudeSchedule.multiplicative` scales each resonator's own amplitude,
`ramp` gives every resonator the same amplitudes, and `explicit` takes a
list. One call measures the whole schedule, in both directions if asked, and
saves one file. See [multisweep.md](../../rfmux/reference-notebooks/Demos/multisweep.md).

### Amplitudes in dBm

```python
await crs.take_netanal(module=1, amp=-60, amp_units=crs.UNITS.DBM)
await crs.multisweep(catalog, amp=-55, amp_units=crs.UNITS.DBM)
await crs.multisweep(catalog, amp=AmplitudeSchedule.ramp(-60, -40, 5, units="dbm"))
```

`amp_units` and the `units` of a schedule take `"normalized"` or `"dbm"`.
`"normalized"` is the default. The case of the letters is not important.

The algorithm reads the DAC scale of the module from the board when the
measurement starts. It uses this DAC scale to convert a dBm amplitude. Thus
the power is the power that you ask for, at all DAC scales.

The spacing of a ramp applies to the amplitude. `"log"` gives equal dB steps.
`"linear"` gives equal amplitude steps.

The file records the schedule as you wrote it. Each sweep records its
normalized amplitude and its power.

### Fit

```python
fit_sweeps(sweeps[crs.module[1].index()], models=("skewed", "nonlinear"))
```

Each fit is written into its sweep, under `entry["fits"][model]`.

### Choose and apply bias points

```python
report = find_bias_points(sweeps[crs.module[1].index()])
bias_path = await crs.apply_bias(report.catalog)
```

`find_bias_points` chooses each resonator's readout power from where it
bifurcates, judged by the IQ derivative, by up/down hysteresis, or by either
(`amplitude_method="both"`), and its frequency where the IQ trace moves
fastest per hertz. `report.flagged` lists the
resonators whose choice needs a look, with the reason. `apply_bias` programs
the catalog and saves a `bias_catalog` file with the NCO and DAC scale
setting read back from the board; `load_bias_catalog(bias_path)` reads it.
See [bias_finding.md](../../rfmux/reference-notebooks/Demos/bias_finding.md).

### Measure noise

```python
noise = await crs.measure_noise(report.catalog)
```

Slow-stream timestreams and spectra for every biased resonator, and the
fast PFB stream when `pfb_samples` is set. With a df calibration the
spectra are also in Hz²/Hz. See
[noise_measurement.md](../../rfmux/reference-notebooks/Demos/noise_measurement.md).

### Calibrate tones that are already playing

```python
catalog = await crs.empirically_calibrate_df(module=1)
```

One short sweep around every biased tone of the module, at its own
amplitude, gives each tone its IQ slope where it sits and so its df
calibration. The tones are put back and the catalog is saved as a
`bias_catalog` file. `rfmux record` reads the session's newest one.

## Periscope

The network analysis panel's **Find Resonances** marks the resonances it
found, and **Take Multisweep** sweeps them. The multisweep window runs the
rest of the flow from its toolbar: **Run Fit**, **Find Bias**, **Apply
Bias** and **Get Noise Spectrum**, each with a settings panel beside it, and
**Edit catalog** and **Re-run Multisweep**.

The network analysis and multisweep dialogs take amplitudes in normalized
units or in dBm. A selector next to the fields sets the units. Each field
shows its value in the other units. When you change the units, the powers
do not change.

The two dialogs show "DAC full scale at output" as a field. You can set it
from −70.64 to +5.34 dBm, which is the range that the board accepts (6.4 µA
to 40.5 mA), after the balun loss. The recommended value is −0.75 dBm at the
output (20 mA). A changed value applies to all the
tones of the module when the measurement starts. The dialogs warn when the
tones can clip, or when they are so small that quantization noise can
increase. `take_netanal` and `multisweep` warn when the tones can clip.
`rfmux.core.dac_scale.comb_drive_issues` gives the same checks to a script.

Its tabs:

- **Magnitude Sweeps** and **IQ Circles**: every resonator at every
  readout power, coloured by power, with the chosen bias point marked.
- **Detector Digest**: one resonator's sweeps, loops and bias point, with
  its fit values. Double-click a subplot on any grid to open it here.
- **Fit Results** and **Fit Histograms**: the model over the measurement,
  and the fit parameters across the array.
- **Bias: derivative**, **Bias: frequency** and **Bias: hysteresis**: what
  `find_bias_points` looked at for each resonator.
- **Collisions**: sweeps that hold two resonances closer than a threshold,
  with **Remove names and re-sweep**.

Double-click a `bias_catalog` in the session browser to see each channel's
bias in the tuning window. If **Apply Bias** finds that the module's DAC
scale changed since the sweep, it recomputes the amplitudes to keep each
resonator's measured power and says so in a dismissable window.

## Simulator

`rfmux.mock.standard_array()` starts the simulator with eight resonators
between 1.00 and 1.10 GHz, the same eight every run, biased by the
simulator, and returns `(crs, catalog)`. The tuning tests start from it.

The simulator biases its resonators at `BIAS_DBM`, -56.75 dBm at the
output.

## Migrating

Every saved tuning format changed, and the bias functions were replaced.
Code that used `bias_kids`, `measure_df_calibrations` or the
`rfmux.algorithms.measurement.fitting` functions needs the changes below.

### Results are keyed by module, then by name

Every measurement returns `{module_id: block}`. A multisweep block holds its
sweeps under `results[step][direction][name]`: the readout-power step, the
sweep direction, and the resonator's name in the catalog.

```python
sweeps = await crs.multisweep(catalog, amp=AmplitudeSchedule.ramp(0.002, 0.032, 5),
                              sweep_direction=("upward", "downward"))
block = sweeps[crs.module[1].index()]
entry = block["results"][0]["upward"]["TEMO"]   # frequencies, iq_volts, ...
```

`results_by_detector` and detector-index keys are gone.

### Biasing: `bias_kids` becomes `find_bias_points` and `apply_bias`

Before:

```python
from rfmux.algorithms.measurement.bias_kids import bias_kids
bias = await bias_kids(crs=crs, multisweep_results=sweeps, module=1)
```

After:

```python
from rfmux.tuning import find_bias_points
report = find_bias_points(sweeps[crs.module[1].index()])
bias_path = await crs.apply_bias(report.catalog)
```

Bias finding reads a multisweep taken over a schedule of readout powers. It returns
a catalog: each resonator's bias frequency, amplitude, power in dBm, dI/df and
dQ/df, the sweep it was read from with that sweep's fits, and any concern
about the choice (`flagged_kind`, `flagged_because`). The ADC phase is not
optimized.

`apply_bias` saves the applied catalog as `bias_catalog_<date>_<time>.pkl`,
with the NCO and DAC scale setting read back from the board. Read it with
`rfmux.tuning.load_bias_catalog(path)`.

### Calibrating tones already playing: `measure_df_calibrations`

```python
from rfmux.tuning import tuning_rows
catalog = await crs.empirically_calibrate_df(module=1)
rows = tuning_rows(catalog)          # {channel: row}, as measure_df_calibrations gave
```

| `measure_df_calibrations` | `empirically_calibrate_df` |
| --- | --- |
| `channels` | every biased channel of the module |
| `span_hz` | `span_hz` |
| `resolution_hz` | `npoints_per_sweep` (span / resolution + 1) |
| `n_samples` | `nsamps` |
| `progress(done, total)` | `progress_callback(module, percent)` |
| returns `{channel: row}` | returns a catalog; `tuning_rows(catalog)` gives the rows |

It puts the tones back at their frequencies and moves the NCO onto the tone
grid at the middle of the tones.
It saves a `bias_catalog` file unless `save=False`.

### Fitting and finding resonances

| Before | After |
| --- | --- |
| `fitting.fit_skewed_multisweep(...)` | `rfmux.tuning.fit_sweeps(block, models=("skewed",))` |
| `fitting_nonlinear.fit_nonlinear_iq_multisweep(...)` | `rfmux.tuning.fit_sweeps(block, models=("nonlinear",))` |
| `fitting.find_resonances(frequencies, iq, ...)` | `rfmux.tuning.find_resonances_in_netanal(netanal_block, ...)`, then `.to_catalog(module=..., amplitude=...)` |
| `min_resonance_separation_hz` | `min_separation_hz` |
| `fitting.identify_bifurcation(iq)` | `rfmux.tuning.bifurcated_by_derivative({"upward": entry})` |
| `fitting_nonlinear.nonlinear_iq`, `get_y_nonlinear`, `fit_nonlinear_iq`, `BIFURCATION_A` | the same names in `rfmux.tuning.fits` |

`fit_sweeps` writes each fit into its sweep, under `entry["fits"][model]`.

### Pulse capture tuning

A capture's tuning is `{channel: row}` from `rfmux.tuning.tuning_rows(catalog)`.
`rfmux record` reads the session's newest `bias_catalog`, listed in the
session folder by `apply_bias`. Pass `--bias <file>` to name one. It does not
read the bias pickles `bias_kids` wrote.

### Powers include the balun loss

Reported powers are the DAC scale setting less the 1.75 dB balun loss
(`rfmux.core.transferfunctions.BALUN_LOSS_DB`). Powers in files saved by
earlier versions used a 1.5 dB offset on main and none on this branch.
The mock's default bias, `BIAS_DBM`, is -56.75 dBm at the output: the same
amplitude as before.

### Old files

Periscope opens network analysis, multisweep and noise files saved in the
current format. A file Periscope saved in an earlier version does not open,
and Periscope says so. Pulse capture files from earlier versions open with
their df calibrations, and their Tuning window shows each channel's sweep.
