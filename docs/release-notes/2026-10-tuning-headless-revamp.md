# Headless tuning branch: release notes

Branch `tuning_headless_revamp`, October 2026.

Resonator tuning runs headlessly from `rfmux.tuning`, and Periscope calls the
same functions. A resonator catalog carries each resonator's bias from bias
finding to the board, to the saved files and to pulse capture.

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
