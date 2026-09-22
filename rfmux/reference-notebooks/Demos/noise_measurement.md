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

# Noise from a biased array

Start with three biased mock resonators, check the operating points with a
multisweep, and measure slow-stream and optional PFB noise. Then reopen the
saved files and plot IQ clouds on their bias sweeps, timestreams, and spectra.
The analysis cells need only those files, not a running board.

Run top to bottom in the environment where this checkout is installed.
This is a Jupytext workbook; the Markdown is the source and the paired
`.ipynb` is a local copy. See `multisweep.md` and `bias_finding.md` for the
tuning steps, and `simplified_tuning_flow.md` for the complete tuning pipeline.

## 1. A biased mock array

The seed fixes the array, not every noise sample. Pulses and TLS noise are
disabled; a small quasiparticle-noise term remains in the slow-stream model.
We read the programmed frequencies and amplitudes into a named catalog.
The simulator supplies demonstration biases; this is not a bias optimizer.

```python
%matplotlib inline

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rfmux
from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.mock.config import apply_overrides
from rfmux.streamer import find_streamer_conflict
from rfmux.tuning import store

DEMO_DIR = Path(rfmux.__file__).resolve().parent / "reference-notebooks" / "Demos"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))
import example_plotting_multisweep as msplots
import example_plotting_noise as noiseplots

MODULE = 1
OUTPUT_DIR = Path(os.environ.get(
    "RFMUX_DEMO_OUTPUT", Path(tempfile.gettempdir()) / "rfmux_noise_demo"))
store.set_output_directory(OUTPUT_DIR)

session = rfmux.load_session('''
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
''')
crs = session.query(rfmux.CRS).one()
await crs.resolve()
count, _ = await crs.generate_resonators(apply_overrides({
    "num_resonances": 3, "freq_start": 601e6, "freq_end": 608e6,
    "C_variation": 0.0001, "resonator_random_seed": 42,
    "auto_bias_kids": True, "bias_amplitude": 0.003,
    "pulse_mode": "none", "tls_noise_enabled": False,
    "nqp_noise_std_factor": 0.001, "T": 0.23,
}))
nco = await crs.get_nco_frequency(module=MODULE)
resonators = []
for channel in range(1, count + 1):
    frequency = nco + await crs.get_frequency(channel=channel, module=MODULE)
    amplitude = await crs.get_amplitude(channel=channel, module=MODULE)
    resonators.append(Resonator(
        name=f"KID{channel:02d}", channel=channel,
        bias=BiasPoint(frequency_hz=frequency, amplitude=amplitude)))
catalog = ResonatorCatalog(resonators, module=MODULE)
module_id = crs.module[MODULE].index()
print(catalog)
```

## 2. Verify the biases with a multisweep

Take one sweep at each current bias amplitude. The dashed frequency marker
in the magnitude plots lets us inspect where the bias sits relative to the
resonance. This inspection does not automatically accept or change a bias.
Multisweep silences the channels it used, so reapply the catalog afterwards.

```python
verification = await crs.multisweep(
    catalog, span_hz=80e3, npoints_per_sweep=201, nsamps=10,
    sweep_direction="upward", save=True, label="noise_bias_check")
sweep_path = store.saved_path(verification)
msplots.plot_magnitude_panels(verification[module_id], directions="upward")
await crs.apply_bias(catalog)
print(f"bias-check file: {sweep_path}")
```

## 3. Acquire and save noise

`measure_noise` measures configured tones; it never chooses or applies
biases. `decimation=None` preserves the current slow-stream configuration. A
different explicit decimation selects this module and changes the packet width;
that configuration remains in effect.

The measurement routine captures slow data from all requested channels
simultaneously, then, when `pfb_samples is not None`, collects 2.44 MS/s PFB
data from one channel at a time. The slow and PFB captures, and each channel's
PFB capture, are sequential rather than synchronized. The PFB UDP sender must
be disabled while the RPC captures run.

**Mock-mode PFB data is synthetic uniform noise**, not a resonator-noise
prediction, although the slow stream does use the resonator model. On real hardware,
use your existing slow stream and a valid timestamp source; omit the mock
sender management below. It stops only a sender this cell started, including
on exceptions and cancellation.

```python
NOISE_PARAMS = dict(
    num_samples=2_000, nsegments=8, reference="absolute",
    spectrum_cutoff=0.9, pfb_samples=20_000, pfb_nsegments=5,
)

def report_progress(event):
    print(f"{event['completed']}/{event['total']}: {event['stream']}"
          f" channel={event['channel']}")

started_sender = False
try:
    conflict = find_streamer_conflict()
    if conflict:
        raise RuntimeError(f"Cannot start a second mock stream: {conflict}")
    started_sender = await crs.start_udp_streaming()
    noise = await crs.measure_noise(
        catalog, **NOISE_PARAMS, progress_callback=report_progress,
        save=True, label="biased_array_noise")
finally:
    if started_sender:
        await crs.stop_udp_streaming()

noise_path = store.saved_path(noise)
print(f"noise file: {noise_path}")
```

`num_samples / sample_rate` is the nominal capture duration; segment length
sets spectral resolution. These short settings make the example quick to run,
not a low-frequency noise characterization. The callback counts completed
captures, not fractional elapsed time. A failed capture is not saved as a
completed measurement.

## 4. Reopen and inspect the products

Both files are `{module_id: block}` containers. Each block carries its schema,
measurement type, module, DAC scale, requested parameters and file metadata.
The noise block records actual tone readbacks separately from the catalog
snapshot. Missing readbacks remain `None`.

```python
saved_noise = store.load(noise_path)
saved_sweeps = store.load(sweep_path)
block = saved_noise[module_id]
sweep_block = saved_sweeps[module_id]
results = block["results"]
settings = results["info"]
restored_catalog = ResonatorCatalog.from_dict(block["call_params"]["catalog"])

print("block fields:", list(block))
print("measurement:", block["measurement"], "schema:", block["schema_version"])
print("requested:", block["call_params"])
print("acquired:", settings)
print("first packet timestamp:", results["shared_slow"]["timestamps"][0])
for name, record in results["resonators"].items():
    print(name, "channel", record["channel"],
          "bias [Hz]", record["bias_frequency_hz"],
          "DAC fraction", record["bias_amplitude"],
          "power [dBm]", record["bias_amplitude_dbm"])
    for stream in ("slow", "pfb"):
        key = f"{stream}_data"
        if key in record:
            data = record[key]
            iq = data[f"iq_{settings['iq_units']}"]
            print(" ", stream, "IQ", iq.shape, iq.dtype,
                  settings["iq_units"], "I PSD", data["psd_i"].shape)
print("nominal slow duration [s]:",
      block["call_params"]["num_samples"] / settings["slow_sample_rate_hz"])
```

Slow timestamps, `freq_iq` and `freq_dsb` are shared under
`results.shared_slow`. The nominal PFB time axis is under
`results.shared_pfb`; PFB spectral axes stay in each resonator's `pfb_data`
because the channel-dependent droop correction can give them different spans.
Each named resonator has `slow_data` and, when requested, `pfb_data`, containing
complex `iq_volts` (absolute) or `iq_counts` (relative) and the three spectra `psd_i`, `psd_q`, and
`psd_dual_sideband`. Slow and PFB time origins are independent.

Time-domain values retain the helpers' units: volts for absolute reference,
counts for relative reference. The plotters convert either representation to
the requested units using `VOLTS_PER_ROC`; they also read older `adc_counts`
records. All frequency axes are in Hz. Absolute spectra are dBm/Hz. With
`reference="relative"`, spectra are dBc/Hz except the carrier bins, stored
as dBc; `carrier_bin_units` records that exception. All returned bins stay in
the file, including the carrier neighborhood.
`bias_amplitude` is the normalized DAC fraction and `bias_amplitude_dbm` is its
power using the measured module DAC scale; it is `None` when either readback is
unavailable. These describe the drive independently of the received spectrum.

`results.info` records the settings and unit declarations needed to
interpret the arrays: the resolved `decimation`, whether it was changed, the
NCO and slow/PFB sample rates, the effective PFB segmentation/correction
settings, the absolute/relative reference, and the IQ, spectrum and
carrier-bin units. Requested arguments remain separately under `call_params`.

| Field | Meaning and source |
| --- | --- |
| `decimation` | Effective board stage: the existing stage when the call omitted `decimation`, otherwise the validated requested stage. |
| `decimation_changed` | Whether that effective stage differed from the stage read before acquisition. |
| `nco_frequency_hz` | Module NCO read from the board before capture; it may be `None` if unavailable. |
| `slow_sample_rate_hz` | Rate calculated by `decimation_to_sampling(decimation)`. |
| `pfb_sample_rate_hz` | Fixed `PFB_SAMPLING_FREQ` when PFB capture was requested, otherwise `None`. |
| `pfb_nsegments` | Effective PFB segment count: the explicit value, or slow `nsegments` when omitted; `None` without PFB capture. |
| `pfb_binlim_hz` | The ±1 MHz limit about the PFB bin center: frequencies outside it are discarded before droop correction. `None` without PFB capture. |
| `pfb_trim` | `False` retains the full corrected dual-sideband span. `True` would further crop it to equal bin counts on either side of zero (with one endpoint excluded). I/Q spectra are clipped independently to the common sideband span. `None` without PFB capture. |
| `pfb_reset_nco` | `False` for a PFB capture because the routine preserves the programmed NCO; otherwise `None`. |
| `reference` | Validated `absolute` or `relative` value requested by the caller and passed to both spectral helpers. |
| `iq_units` | `volts` for absolute reference (`iq_volts`), `counts` for relative reference (`iq_counts`), matching the helpers without rescaling. |
| `spectrum_units` | `dBm/Hz` for absolute spectra or `dBc/Hz` for relative spectra. |
| `carrier_bin_units` | `dBm/Hz` in absolute mode; `dBc` in relative mode because the helper multiplies its density by the FFT bin width. |
| `carrier_bin_rule` | Identifies the carrier as the bin nearest zero frequency in each spectrum. |

Carrier handling in both helpers uses the complex dual-sideband DC bin as the
common I/Q reference. In absolute mode every bin remains a density in dBm/Hz,
including DC. In relative mode the reference is the DC density multiplied by
`sample_rate / segment_length`; every spectrum is divided by that reference.
The helpers also multiply the I, Q and dual-sideband DC entries themselves by
that bin width, so those entries are powers in dBc while other entries remain
dBc/Hz. Dual-sideband DC is therefore 0 dBc; I and Q DC show their respective
fractions of the combined carrier power.

This single-bin estimate is not the full Hann-windowed carrier power. For a
constant `3+4j` volt signal, 4,096 samples and one segment, the expected received
power is 0.25 W under the helpers' peak-voltage/50-ohm convention. The slow
helper estimates 0.166667 W and the PFB helper (tone at bin center) 0.166626 W:
about 1.76 dB low, making relative noise densities about 1.76 dB high. The slow
helper uses a periodic Hann window and PFB uses a symmetric Hann window, which
accounts for the small difference. Absolute PSD normalization does not have
this single-bin integration error; carrier power requires integration over
its window-broadened peak. A future correction should keep every spectral bin
in density units and report a separate, window-corrected carrier power. The
current helpers and their mixed-unit carrier entries are preserved here.

## 5. Overlay noise on the bias sweep

Both traces below are in volts, without per-trace normalization. The gray curve
is the verification sweep at the measured drive amplitude, the cloud contains
slow IQ samples, and markers show the noise mean and the sweep interpolated at
the measured tone frequency. The plotter refuses a mismatched drive amplitude.

Here we supply the saved verification block. For a noise file acquired from a
`find_bias_points` catalog, `plot_iq_panels(block)` can instead read the stored
`bias_sweep` from its catalog snapshot, as in the simplified tuning flow.

```python
noiseplots.plot_iq_panels(block, sweeps=sweep_block)
```

## 6. Timestreams and PSDs

Mean subtraction below reveals fluctuations without changing saved IQ.
Time axes use nominal sample spacing, not reconstructed packet timestamps.
Pass `units="counts"` for readout counts, or `demean=False` to retain DC.
Use `names="KID01"` or a list to select resonators in any of the plotters.

```python
noiseplots.plot_timestreams(block, demean=True)
if block["call_params"]["pfb_samples"] is not None:
    noiseplots.plot_timestreams(
        block, stream="pfb", demean=True,
        title="PFB mock RPC: synthetic uniform noise")
```

I/Q spectra use positive offsets on a log axis. Dual-sideband spectra retain
signed offsets on a symmetric log axis; they are not folded or doubled.
The plotters omit the carrier and its adjacent bins to reduce the displayed
Hann-window carrier contribution. They neither rebin nor recalibrate spectra,
and do not alter the saved arrays. Plot slow and PFB separately because their
bandwidths and measurement times differ.

```python
noiseplots.plot_psds(block)
noiseplots.plot_psds(block, dual_sideband=True)
if block["call_params"]["pfb_samples"] is not None:
    noiseplots.plot_psds(block, stream="pfb",
                         title="PFB mock RPC: synthetic uniform noise")
```

The mock remains biased and this workbook's UDP sender is stopped. The saved
files are sufficient to rerun sections 4–6 in another session. For detector
pulses and triggered records, continue with `pulse_capture.md`.
