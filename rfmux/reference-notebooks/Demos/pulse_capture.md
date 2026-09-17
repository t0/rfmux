---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Pulse Capture

End-to-end pulse capture from the Python API: configure the streamers, detect
pulses live, save them, and analyze them, without opening Periscope.

This is the code path Periscope's *Pulse Capture* panel drives. The panel
builds a `PulseCaptureSession`, feeds it from its own packet receiver and from
`run_pfb_source`, and draws the callbacks. This notebook builds the same
session, feeds it with the source functions, and prints or plots.

`rfmux.pulse_capture` re-exports every class and function of its submodules,
so one import line covers them. The table gives the module each lives in.

| Piece | Module |
|---|---|
| Streamer setup and link budget | `rfmux.algorithms.measurement.streamer_config` |
| Detection engine (ring buffer, triggering) | `rfmux.pulse_capture.detection` |
| Live capture orchestration | `rfmux.pulse_capture.capture_session` |
| Concurrent slow+fast with matching | `rfmux.pulse_capture.capture_session` |
| Packet sources that feed a session | `rfmux.pulse_capture.sources` |
| Per-pulse metrics (SNR, derived τ) | `rfmux.pulse_capture.analysis` |
| Streaming HDF5 persistence | `rfmux.pulse_capture.hdf5` |

## How to use this document

Run the cells in order; later ones use variables the earlier ones defined.
Sections 1 and 2 are the exception: run only the one option (1A, 1B or 2)
that fits.

This format saves no outputs, so every number you see comes from your own run.
The shipped copy is read-only: *File → Save Notebook As…* to keep changes.

Captures are written to `OUTPUT_DIR`, printed by the next cell. Section 7 reads
them back, and Periscope can open them in review mode.

```python
%matplotlib inline

import asyncio
import os
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import rfmux
from rfmux.pulse_capture import (
    DualPulseCaptureSession, PulseCaptureConfig, PulseCaptureSession,
    PulseHDF5Reader,
    run_dual_source, run_pfb_source, run_slow_source,
)
from rfmux.core.transferfunctions import (
    PFB_SAMPLING_FREQ, decimation_to_sampling,
)
from rfmux.algorithms.measurement.streamer_config import (
    StreamerConfig, describe, validate,
)

# Reference notebooks are provisioned read-only, so captures go to a
# scratch directory; override it with RFMUX_DEMO_OUTPUT.
OUTPUT_DIR = Path(os.environ.get(
    "RFMUX_DEMO_OUTPUT", Path(tempfile.gettempdir()) / "rfmux_pulse_capture"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODULE = 1
CHANNELS = [1, 2]

crs = None          # set by whichever cell in section 1 or 2 you run
host = "127.0.0.1"  # where the streamers send
IS_MOCK = False     # True only if THIS notebook created the simulation

print(f"capture files → {OUTPUT_DIR}")
```

## 1. Connect

Everything below needs a CRS. **Run exactly one** of the three options:

| | When to use it | Where |
|---|---|---|
| **A. Periscope is running** | Periscope launched this Jupyter session and is driving a board or a simulation | below |
| **B. Starting from scratch with real hardware** | You have a CRS and this notebook is being viewed separately from Periscope | below |
| **C. Start a new simulated environment** | No Periscope GUI instance already, and nothing already running | section 2 |

### A. Attach to the CRS Periscope is driving

Use this when Periscope is driving a board, real or simulated, and you want
that one rather than your own.

Periscope sets `RFMUX_CRS_HOSTNAME` when it launches this notebook, which is how
the cell finds the board with no configuration from you.

Attaching matters most if you have already configured Periscope in mock mode.
A second `create_mock_crs()` gives you a *second, unrelated* simulation,
whose detectors are not the ones Periscope is showing you.

```python
HOSTNAME = os.environ.get("RFMUX_CRS_HOSTNAME")   # or paste "127.0.0.1:43431"
SERIAL = os.environ.get("RFMUX_CRS_SERIAL", "0000")

if HOSTNAME:
    s = rfmux.load_session(
        f'!HardwareMap [ !CRS {{ serial: "{SERIAL}", '
        f'hostname: "{HOSTNAME}" }} ]')
    crs = s.query(rfmux.CRS).one()
    await crs.resolve()
    # The streamer sockets join on the board's address, not the default
    # loopback set above -- for a real board 127.0.0.1 is the wrong
    # interface and the capture receives nothing, silently.
    host = HOSTNAME.split(":")[0]
    print(f"attached to CRS {SERIAL} at {HOSTNAME}")
else:
    print("Nothing advertised a board. Set HOSTNAME above, or use option B "
          "(your own board) or section 2 (simulation).")
```

### B. Your own board, no Periscope GUI

```python
# SERIAL = "0042"
# s = rfmux.load_session(f'!HardwareMap [ !CRS {{ serial: "{SERIAL}" }} ]')
# crs = s.query(rfmux.CRS).one()
# await crs.resolve()
# await crs.set_timestamp_port(crs.TIMESTAMP_PORT.TEST)
# host = crs.tuber_hostname          # where the streamers send
# print(f"connected to CRS {SERIAL} at {host}")
```


## 2. Mock mode configuration

**Skip this section if section 1 gave you a CRS.** The cell below does
nothing in that case.

With no hardware, this cell creates a simulated CRS with two biased resonators
and periodic pulses. `auto_bias_kids` sweeps each resonator at
`bias_amplitude` (-55 dBm by default) and biases it at the S21 minimum. That
gives biased resonators, not a df calibration; section 7 measures one.

Pulse heights are drawn uniformly between `pulse_random_amp_min` and
`pulse_random_amp_max`, so the amplitude histogram in section 7 shows a
distribution rather than a single spike.

Three noise sources are on:

- **White readout noise** (`udp_noise_level`): the flat floor.
- **Quasiparticle number fluctuations** (`nqp_noise_enabled`):
  generation-recombination noise in the resonator.
- **TLS 1/f frequency noise** (`tls_noise_enabled`): the resonant frequency
  wanders with a `1/f**alpha` spectrum, so the baseline moves slowly. The
  trigger in section 4 tracks it.

> **This cell refuses to run if something is already streaming.** Two
> simulations send to the same UDP port and a receiver gets both interleaved,
> with no error. The message says which case you are in. If Periscope is in
> mock mode, attach to its simulation with option 1A.

```python
from rfmux.mock.config import bias_amplitude_from_dbm

MOCK_CONFIG = {
    "num_resonances": 2,
    "resonator_random_seed": 42,
    "auto_bias_kids": True,        # bias the detectors
    "bias_amplitude": bias_amplitude_from_dbm(-55.0),   # tone power for the
                                   # sweep and the bias (the dialog shows dBm)

    # ── Noise (these are the shipped defaults, spelled out) ─────
    "udp_noise_level": 11.0,       # white readout noise (ADC counts), from a board
    "nqp_noise_enabled": True,     # quasiparticle generation-recombination
    "nqp_noise_std_factor": 0.01,  # 1% of base quasiparticle density
    "tls_noise_enabled": True,     # TLS 1/f frequency wander
    "tls_fractional_rms": 1e-7,    # RMS of df/f
    "tls_alpha": 1.0,              # PSD ~ 1/f**alpha
    "tls_corner_hz": 100.0,        # upper corner; law spans 3 decades below

    # ── Pulses to detect ────────────────────────────────────────
    "pulse_mode": "periodic",
    "pulse_period": 0.05,          # one every 50 ms
    "pulse_tau_rise": 1e-6,
    "pulse_tau_decay": 1e-3,       # 1 ms decay constant
    "pulse_random_amp_mode": "uniform",   # spread of pulse heights,
    "pulse_random_amp_min": 1.5,          # not one repeated event
    "pulse_random_amp_max": 3.0,
}

if crs is not None:
    print("already connected: skip this cell")
else:
    from rfmux.streamer import find_streamer_conflict

    if os.environ.get("RFMUX_CRS_HOSTNAME"):
        raise RuntimeError(
            "Periscope launched this notebook and is already driving a CRS.\n"
            "Run option 1A above to attach to that one. Creating a second "
            "simulation here would stream to the same UDP port as Periscope's, "
            "and every reader would see the two interleaved.")

    conflict = find_streamer_conflict()
    if conflict:
        raise RuntimeError(
            f"Something is already using the streamer port: {conflict}.\n"
            "A second simulation would send to that same port, and a reader "
            "would get both streams interleaved with nothing to say so.\n"
            "Attach to what is running with option 1A, or stop it, then re-run "
            "this cell.")

    from rfmux.mock.helpers import create_mock_crs
    crs = await create_mock_crs(module=MODULE, config=MOCK_CONFIG,
                                verbose=False)
    IS_MOCK = True
    host = "127.0.0.1"
    print(f"simulated CRS ready: {MOCK_CONFIG['num_resonances']} resonators, "
          f"1/f on (df/f = {MOCK_CONFIG['tls_fractional_rms']:.0e} rms)")
```

### Confirm the connection

The rest of the notebook uses `crs`; this fails early if section 1 did not
set it.

```python
if crs is None:
    raise RuntimeError(
        "No CRS. Run option 1A (attach), 1B (your board), or the cell above "
        "(simulate one) before continuing.")

print(f"CRS       {crs.tuber_hostname}")
print(f"streamers {host}")
print(f"module {MODULE}, channels {CHANNELS}")
print("simulation created by this notebook" if IS_MOCK
      else "pre-existing board: this notebook will not tear it down")
```

## 3. Configure the streamers

Two streams carry data off the board:

- the **slow** readout stream: 38 kHz at stage 0 down to 596 Hz at stage 6,
  up to 1024 channels per module, port 9876;
- the **fast** PFB stream: 2.44 MHz, up to 4 channels of one module, port 9877.

Choose the stream and decimation from the pulse: aim for **10 or more samples
across one decay constant**. Fewer and the decay cannot be fitted; many more
and you are oversampling.

`validate()` reports the hardware rules (long packets need stage ≥ 3, the 1 GbE
budget, OS receive-buffer advice) as `(severity, message)` pairs. `describe()`
returns the derived rates and link budget.

> If you attached to Periscope's CRS in section 1, remember the streamer is a
> shared resource: changing the decimation here changes it for Periscope's plots
> too, and this call also switches the PFB streamer off. Pass
> `pfb_channels=None` to leave it as it is.

```python
PULSE_TAU_S = 1e-3          # expected decay constant

needed_fs = 10.0 / PULSE_TAU_S
dec = next(d for d in range(6, -1, -1) if decimation_to_sampling(d) >= needed_fs)
cfg = StreamerConfig(dec_stage=dec, short_packets=(dec < 3), modules=[MODULE])

print(f"τ = {PULSE_TAU_S*1e3:.1f} ms → need ≥ {needed_fs:.0f} Hz "
      f"→ stage {dec} ({decimation_to_sampling(dec):.0f} Hz)")

# Check the link budget BEFORE touching the board
budget = describe(cfg)
print(f"{budget['channels_per_module']} ch/module × {budget['n_modules']} "
      f"module(s) at {budget['sample_rate_hz']:.0f} Hz "
      f"= {budget['total_mbps']:.0f} Mbps of 1 GbE")
for severity, message in validate(cfg):
    print(f"  [{severity}] {message}")

info = await crs.configure_streamer(cfg.dec_stage, short=cfg.short_packets,
                                    modules=cfg.modules)
info
```

```python
if IS_MOCK:
    # The simulated stream, already running, needs a moment to settle
    # after a rate change.
    await asyncio.sleep(2.0)
```

### Choosing channels

Periscope's **Channels** field and this notebook take the same strings, because
they call the same parser. Single channels, inclusive ranges, or a mix:

```python
from rfmux.algorithms.measurement.channel_selection import parse_channel_spec

for spec in ("1,2", "2-19", "1,5-8,20"):
    print(f"{spec!r:12} -> {parse_channel_spec(spec)}")
```

For `all` (or `*`) the parser returns `None`: resolve it with
`get_biased_channels`, which reads every channel's amplitude in one batched
round trip. Pass `max_channels` as the packet width you are streaming; a
channel above it is in no packet.

```python
print(f"{'all'!r:12} -> {parse_channel_spec('all')}   (resolve against the board)")

# The packet width the streamer is configured for (128 short, 1024 long).
biased = await crs.get_biased_channels(
    MODULE, max_channels=budget["channels_per_module"])
print(f"biased on module {MODULE}: {len(biased)} channel(s)")
print(f"  {biased[:12]}{' ...' if len(biased) > 12 else ''}")
```

`CHANNELS` is a list of ints. Set it from a spec, from `get_biased_channels`,
or by hand.

## 4. Choose the detection parameters

`PulseCaptureConfig` holds every user-facing parameter in **physical units**
(σ and milliseconds). It converts them to samples for the stream rate you hand
it, so one configuration works unchanged from 596 Hz to 2.44 MHz.

How pulse detection works:

- **A capture opens on either axis and closes when both have settled.** The
  axes are df and dissipation for a channel with a calibration (section 7),
  I and Q otherwise. It opens when either leaves ±`threshold_sigma`. It
  closes when both are back inside ±`end_sigma` of the baseline, or of the
  level the pulse rose from. They must stay there for the longest of
  `min_end_samples`, a tenth of the time above threshold, and
  `post_pulse_ms`. `end_sigma` must sit below `threshold_sigma`.
- **Triggers are confirmed.** `trigger_samples` consecutive samples must clear
  the threshold. Left at 0 it is derived from the stream rate to hold
  accidental triggers under `max_accidental_per_min`: 1 sample at 596 Hz, 2 on
  the PFB stream.
- **A trigger needs a fast rise.** The deviation must have grown by more than
  `threshold_sigma` jump-σ within the last `edge_lookback` samples. That is a
  difference of raw samples, so baseline drift cancels out of it: only a fast
  rise triggers.
- **The baseline is a rolling median** over `baseline_window` samples, which
  ignores pulses as long as they are a minority of the window. The noise σ is
  the samples' scatter about a block-median baseline, three hard-stop lengths
  per block, so a slow drift is not counted as noise.
- **`max_pulse_ms` sizes the pulse scale.** The ring buffer is 1.5× it, the
  hard stop 1.2×, the edge lookback a tenth. Estimate it generously: a pulse
  that outlasts the buffer loses its rising edge.
- **`noise_train_ms` is the 1/f window** (5 s): the record the noise σ is
  fitted from and the span of the rolling baseline, long compared with any
  pulse and with the 1/f knee. Below 2 s the baseline refresh costs real time
  at many channels.
- **A capture that never ends is cut off** at the hard stop and flagged
  `truncated`.
- **The saved record has set margins.** `pre_pulse_ms` is kept before the
  trigger and `post_pulse_ms` after the pulse settled, 5 ms each by default.
- **Pulses on different channels can be one event.** With
  `coincidence_window_ms` set, every pulse on any channel that triggers
  within that window of an event's first trigger belongs to the event. The
  pulses stay stored under their channels and the event lists them.
  `dump_all_channels=True` saves, with each event, the same span of every
  channel that did not trigger.
- **Noise samples are taken at random.** With `noise_capture_interval_s`
  set, every channel is saved over one window at random moments, whether or
  not a pulse is present, for the statistics of the noise. The waits between
  samples are normally distributed about the interval, a quarter of it wide.
  A sample is as long as a typical pulse record, the median of the latest
  200 saved, and is saved as an event of kind `"noise"`. Until five records
  have been saved it is `pre_pulse_ms + max_pulse_ms + post_pulse_ms` long.
- **Piled-up pulses are split.** A fresh rise on the tail of a pulse, after
  it was seen decaying, starts a new one. The rise is judged against the
  larger of the trained jump-σ and the scatter measured inside the capture,
  and must hold for `trigger_samples` consecutive samples, so neither noise
  that grows with the pulse nor one stray sample splits it. The second
  fragment is dated at its rise and keeps the first's pre-pulse level as
  its end reference. Both fragments carry the `pileup` flag: templates skip
  them, histograms keep them.

![Anatomy of one capture window](pulse_capture_anatomy.png)

The figure is the engine's own output on a synthetic pulse and a piled-up
pair, at the defaults with `max_pulse_ms=50`. The shaded region is what gets
saved: from `pre_pulse_ms` before the trigger (5 ms by default), so the
record keeps pre-trigger baseline, to `post_pulse_ms` (5 ms) after the sample
the pulse settled on inside the end band. The end confirmation runs at least
that long and at least `min_end_samples`. The rest of the confirmation only
verifies that the pulse stayed there, and is not saved. `duration_ms` runs
from the trigger to the settled sample. The drop below `threshold_sigma` is
kept as a mark and feeds the fit-free decay constant.

`describe()` reports everything derived at a given rate, and `validate()` catches
inconsistent settings before you spend a capture on them.

```python
capture_config = PulseCaptureConfig(
    threshold_sigma=5.0,    # trigger when I or Q leaves ±5σ
    end_sigma=1.5,          # close when BOTH are back inside ±1.5σ
    min_pulse_ms=0.2,       # glitch filter: drop anything shorter
    max_pulse_ms=50.0,      # longest recordable pulse; sizes the buffer
    pre_pulse_ms=5.0,       # saved before the trigger
    post_pulse_ms=5.0,      # saved after the pulse settled
    noise_train_ms=1000.0,  # the 1/f window: noise fit and rolling baseline
                            # (5 s by default; 1 s keeps this demo short)
    enable_pileup=True,     # split piled-up events on a sharp re-rise
    coincidence_window_ms=0.0,     # 0: no events (section 7 turns these on)
    dump_all_channels=False,       # save the channels that did not trigger
    noise_capture_interval_s=0.0,  # 0: no noise samples
)

fs = decimation_to_sampling(cfg.dec_stage)
for severity, message in capture_config.validate(fs):
    print(f"  [{severity}] {message}")

d = capture_config.describe(fs, n_channels=len(CHANNELS))
print(f"\nAt {d['sample_rate_hz']:.0f} Hz:")
print(f"  ring buffer     {d['buf_samples']:,} samples "
      f"({d['buf_mb_total']:.2f} MB total, "
      f"{d['max_recordable_ms']:.0f} ms max)")
print(f"  noise training  {d['noise_samples']:,} samples "
      f"({d['noise_train_actual_ms']:.0f} ms)")
print(f"  baseline median {d['baseline_window']:,} samples "
      f"({d['baseline_window_ms']:.0f} ms)")
print(f"  trigger confirm {d['trigger_samples']} sample(s) → "
      f"{d['accidental_per_min']:.2e} accidentals/min/channel")
print(f"  edge lookback   {d['edge_lookback']} samples "
      f"({d['edge_lookback_ms']:.1f} ms)")
print(f"  capture limit   {d['max_capture_ms']:.0f} ms")
```

The confirmation length follows the rate, which is what makes one config
portable across streams:

```python
for rate, label in [(596.0, "slow, stage 6"), (fs, f"slow, stage {dec}"),
                    (PFB_SAMPLING_FREQ, "fast (PFB)")]:
    dd = capture_config.describe(rate)
    print(f"{label:<18} {rate:>10,.0f} Hz → confirm "
          f"{dd['trigger_samples']} sample(s), "
          f"{dd['accidental_per_min']:.2e} accidentals/min")
```

## 5. One-shot capture

`trigger_capture` runs a session for `time_run` seconds of sample time and
returns every pulse in memory. Pass `hdf5_path=` to write the capture file as
well. For a capture too long to hold in memory, drive a session directly
(section 6).

`streamer_mode` is `"slow"`, `"fast"` (PFB, ≤ 4 channels) or `"both"`. Noise
training runs first and is *not* charged against `time_run`.

```python
res = await crs.trigger_capture(
    channel=CHANNELS, module=MODULE,
    streamer_mode="slow",
    time_run=2.0,               # seconds of SAMPLE time, not wall clock
    threshold_sigma=5.0,
    end_sigma=1.5,
)

for ch in res.channels:
    n = res.noise[ch]
    print(f"ch{ch}: {len(res.pulses[ch])} pulses, "
          f"I={n.mean_I:.3g}±{n.std_I:.3g} V")
res
```

No calibration was passed, so the samples are in volts on the I and Q axes;
section 7 shows what a calibration changes.

Each pulse comes with its metrics already computed. `res.summaries[ch][idx]` is
`pulse_summary()` output.

```python
ch = CHANNELS[0]
idx = min(res.pulses[ch])
pulse = res.pulses[ch][idx]

print({k: (round(v, 4) if isinstance(v, float) else v)
       for k, v in res.summaries[ch][idx].items()})

t_ms = (pulse["Time"] - pulse["Time"][0]) * 1e3
plt.figure(figsize=(8, 3))
plt.plot(t_ms, pulse["Amp_I"], label="I")
plt.plot(t_ms, pulse["Amp_Q"], label="Q")
plt.xlabel("time (ms)"); plt.ylabel("V")
plt.title("One captured pulse"); plt.legend(); plt.show()
```

The derived τ is not a fit. It uses two points on the falling edge, the peak
and the moment the envelope falls back through the trigger threshold:

$$\tau = \frac{t_{\rm thr} - t_{\rm peak}}{\ln(\mathrm{SNR}_{\rm peak} / \sigma_{\rm thr})}$$

The ratio of amplitudes cancels the unknown event energy, so a detector with
one decay time gives one τ at every energy. It is a live cross-check, not a
precision measurement: the discrete crossing sample lands slightly below the
true crossing, so it runs a few percent low.

## 6. Live capture with streaming persistence

`PulseCaptureSession` is what the panel runs. It trains on noise, then
detects, and as each pulse closes it appends to HDF5, updates the histograms
and stacks a trigger-aligned template. Memory stays flat however long you
capture.

Feed it from `run_slow_source` / `run_pfb_source`, or from any sample source
of your own through `feed_sample(channel, I, Q, t)`.

`on_pulse` is the callback the GUI uses to update its display. Here it prints.

```python
def on_pulse(channel, pulse_idx, summary, waveform):
    if pulse_idx <= 3:       # keep the output short
        print(f"  ch{channel} #{pulse_idx}: {summary['snr']:.0f}σ, "
              f"{summary['duration_ms']:.2f} ms, τ={summary['tau_ms']:.2f} ms")

capture_session = PulseCaptureSession(
    channels=CHANNELS, module=MODULE, streamer_mode="slow",
    sample_rate=fs, hdf5_path=str(OUTPUT_DIR / "pulse_capture_demo.h5"),
    on_pulse=on_pulse,
    **capture_config.session_kwargs(fs),
)
capture_session.start()
covered = await run_slow_source(capture_session, host, module=MODULE,
                               duration_s=3.0)
capture_session.stop()

print(f"\n{capture_session.total_pulses} pulses over {covered:.2f} s of sample time")
print(f"rolling baseline median over {capture_session.baseline_window:,} samples "
      f"({capture_session.baseline_window / fs:.2f} s)")
```

`duration_s` counts **sample time** accumulated from packet timestamps, not wall
clock, so a capture covers the span you asked for even if packets arrive late.
Pass `should_stop=<callable>` instead to run until you decide to stop.

## 7. Read the capture back

The file holds every pulse waveform, the running histograms, the templates, and
the noise statistics and capture parameters as attributes. Periscope opens these
same files in review mode (double-click in the Session Browser).

```python
reader = PulseHDF5Reader(OUTPUT_DIR / "pulse_capture_demo.h5")
print("channels:", reader.channels)
print("threshold_sigma:", reader.metadata["threshold_sigma"],
      "| sample rate:", f"{reader.metadata['sample_rate_slow']:.0f} Hz")

for ch in reader.channels:
    n = reader.noise_stats(ch)
    print(f"  ch{ch}: {reader.pulse_count(ch)} pulses, "
          f"noise I={n.mean_I:.3g}±{n.std_I:.3g}, "
          f"Q={n.mean_Q:.3g}±{n.std_Q:.3g} {reader.stored_units(ch)}")

# Per-pulse metadata without loading any waveforms.  trigger_utc is
# decoded from the packet timestamps, not from the host clock.
for meta in list(reader.iter_pulse_metadata(reader.channels[0]))[:5]:
    print(f"  #{meta['pulse_idx']:04d} snr={meta.get('snr', 0):.0f}σ "
          f"dur={meta.get('duration_s', 0)*1e3:.2f} ms "
          f"τ={meta.get('tau_s', float('nan'))*1e3:.2f} ms "
          f"at {meta.get('trigger_utc', '?')}")
```

Histograms accumulate as the capture runs and expand their ranges automatically
when a pulse falls outside the current binning, so you never have to guess the
scale in advance. Keys are `<metric>_edges`, `<metric>_bins` (centers) and
`<metric>_counts_ch<N>`.

```python
hist = reader.get_histograms()
fig, axes = plt.subplots(1, 4, figsize=(17, 3))
for ax, metric, xlabel in zip(
        axes, ["snr", "amplitude_i", "amplitude_q", "tau_ms"],
        ["peak deviation (σ)", "peak along df (V)", "peak along dissipation (V)",
         "derived τ (ms)"]):
    edges = hist.get(f"{metric}_edges")
    for ch in reader.channels:
        counts = hist.get(f"{metric}_counts_ch{ch}")
        if edges is None or counts is None:
            continue
        centers = 0.5 * (edges[:-1] + edges[1:])
        keep = counts > 0
        ax.bar(centers[keep], counts[keep],
               width=(edges[1] - edges[0]), alpha=0.6, label=f"ch{ch}")
    ax.set_xlabel(xlabel); ax.set_ylabel("count"); ax.legend()
plt.tight_layout(); plt.show()
```

### Trigger-aligned template

Every pulse is stacked on its **trigger crossing**, not on the start of its
window: the pre-pulse time is a setting, and a pulse that arrives before
the ring buffer has filled gets less of it. Either would smear the stack.
The mean beats the noise down as 1/√N; the shaded band is the per-bin RMS
spread, which separates pulse-to-pulse variation from measurement noise.

```python
tmpl = reader.get_templates()
plt.figure(figsize=(8, 3.5))
for ch in reader.channels:
    t = tmpl.get(f"time_s_ch{ch}")
    mean = tmpl.get(f"template_I_ch{ch}")
    resid = tmpl.get(f"residual_I_ch{ch}")
    if t is None or mean is None:
        continue
    n = int(np.nanmax(tmpl[f"counts_ch{ch}"]))
    plt.plot(t * 1e3, mean, label=f"ch{ch} template (n={n})")
    if resid is not None:
        plt.fill_between(t * 1e3, mean - resid, mean + resid, alpha=0.25)
plt.axvline(0, color="k", lw=0.8, ls=":")
plt.xlabel("time from trigger (ms)"); plt.ylabel("I (V)")
plt.title("Trigger-aligned template"); plt.legend(); plt.show()
```

### Calibrated amplitudes

Samples are stored in physical units, not ADC counts: volts, or hertz for a
channel rotated into the frequency basis.

The calibration comes from `bias_kids`, whose entry per detector carries a
complex `df_calibration`. Its magnitude is hertz per volt; its phase is minus
the angle of the frequency direction in the (I, Q) plane, so multiplying by
it turns that direction onto the real axis. The capture takes the whole
entry, the tuning record: `tuning_rows` keys the entries by readout channel,
and the session stores each channel's row (bias frequency, amplitude, sweep,
fit parameters, calibration) in the file with its pulses:

    from rfmux.algorithms.measurement.df_calibration import tuning_rows

    bias_results = await bias_kids(crs=crs, multisweep_results=...,
                                   module=MODULE)
    tuning = tuning_rows(bias_results, await crs.get_nco_frequency(module=MODULE))

    capture_session = PulseCaptureSession(..., tuning=tuning)

A calibrated channel is rotated before thresholding by default
(`trigger_basis="df"`), so a pulse lands on one axis instead of being split
between two by an angle nothing controls. A channel without a calibration
stays on the quadratures, and in volts. Pass `trigger_basis="iq"` to threshold
the raw quadratures even where a calibration exists.

`auto_bias_kids` biases the resonators but does not produce a calibration, so
a simulated array has none yet. `measure_df_calibrations` is that measurement
on its own: a narrow sweep around each bias point, every channel stepping
together, with a resonance fitted to the sweep and differentiated at the bias
point. This is the estimate `bias_kids` keeps as `df_calibration_fit`; its
`df_calibration` is measured by stepping each tone. It uses only ordinary CRS
calls (`get_frequency`, `set_frequency`, `get_samples`), so it runs against a
board too. On hardware take the calibration from `bias_kids` instead of
sweeping a tuned array again. With no channel list it measures every channel
the module reports as biased, which is what Periscope does at startup in mock
mode. It returns rows of the same shape: the calibration, the sweep it was
read from and the resonance fitted to that sweep.

```python
tuning = await crs.measure_df_calibrations(module=MODULE)
for ch, row in sorted(tuning.items()):
    cal = row["df_calibration"]
    print(f"  ch{ch}: {abs(cal):.3g} Hz per volt, "
          f"{np.degrees(np.angle(cal)):+.1f} deg")
```

A capture with this tuning triggers in the frequency basis and stores each
calibrated channel in hertz. This one also records events and noise samples,
and captures a third channel that has no detector on it. That channel never
triggers, so `dump_all_channels` saves it with every event:

```python
from dataclasses import replace

reader.close()

EVENT_CHANNELS = CHANNELS + [3]
event_config = replace(
    capture_config,
    coincidence_window_ms=2.0,      # the simulated pulses arrive together
    dump_all_channels=True,
    noise_capture_interval_s=0.25,  # about eight noise samples in 2 s
)
CAPTURE_FILE = OUTPUT_DIR / "pulse_capture_calibrated.h5"
res = await crs.trigger_capture(
    channel=EVENT_CHANNELS, module=MODULE, streamer_mode="slow", time_run=2.0,
    config=event_config, tuning=tuning, hdf5_path=str(CAPTURE_FILE),
)
for ch in res.channels:
    peaks = [s["peak_amp"] for s in res.summaries[ch].values()]
    units = "Hz" if ch in tuning else "V"
    print(f"ch{ch}: {len(peaks)} pulses"
          + (f", mean peak {np.mean(peaks):.4g} {units}" if peaks else ""))
kinds = [e["kind"] for e in res.events]
print(f"{kinds.count('pulses')} events, {kinds.count('noise')} noise samples")
```

Every file records the units per channel, the counts-to-volts constant and
the tuning row under each channel's `tuning` group, with `df_calibration`
among its attributes. Dual files (section 9) carry the same:

```python
with PulseHDF5Reader(CAPTURE_FILE) as r:
    print(f"trigger basis: {r.trigger_basis()}   "
          f"volts per count: {r.volts_per_count():.4g}")
    for ch in r.channels:
        units = r.stored_units(ch)
        cal = r.df_calibration(ch)
        first = next(r.iter_pulse_metadata(ch), {})
        peak = first.get("peak_amp", float("nan"))
        note = "uncalibrated" if cal is None else f"|cal| = {abs(cal):.3g} Hz/V"
        print(f"  ch{ch}: peak {peak:.4g} {units}   ({note}), "
              f"tuning fields {sorted(r.tuning(ch))}")
```

### Everything in the file

The file is plain HDF5, so `h5py` alone reads it. This prints the whole tree:
groups, datasets with their shapes, and how many attributes each carries.

```python
import h5py

def show(name, item):
    depth = name.count("/")
    # One pulse and one event stand for the rest.
    if "/pulse_" in name and "pulse_000001" not in name:
        return
    if name.startswith("events/event_") and "event_000001" not in name:
        return
    kind = (f"dataset {item.shape} {item.dtype}"
            if isinstance(item, h5py.Dataset)
            else f"group, {len(item.attrs)} attributes")
    print("  " * depth + f"{name.split('/')[-1]:<22} {kind}")

with h5py.File(CAPTURE_FILE, "r") as f:
    f.visititems(show)
    print("\nmetadata:")
    for key, value in sorted(f["metadata"].attrs.items()):
        print(f"  {key:<26} {value}")
    print("\none pulse's attributes:")
    for key, value in sorted(f["channel_1/pulse_000001"].attrs.items()):
        print(f"  {key:<26} {value}")
```

Each `channel_<n>` group holds that channel's noise statistics as attributes,
its `noise_training` record, its `tuning` row and one `pulse_<k>` group per
pulse. `events` lists the events, `histograms` and `templates` hold what the
capture accumulated, and `metadata` the capture parameters.
`PulseHDF5Reader` reads the same things without the paths:

```python
reader = PulseHDF5Reader(CAPTURE_FILE)
ch = CHANNELS[0]
PULSE = 3        # the first pulse of a capture can arrive before the
                 # buffer holds a full pre-pulse span
pulse = reader.get_pulse(ch, PULSE)
print("a pulse:", sorted(pulse))
print("noise training record:", reader.noise_training(ch).shape,
      reader.noise_training(ch).dtype)
print("tuning row:", sorted(reader.tuning(ch)))
print("events:", reader.event_count)
```

### A pulse as Periscope draws it

The Pulse View draws the two stored axes against time with the noise bands
the trigger used and the marks of its decisions. The record carries all of
them.

```python
ns = reader.noise_stats(ch)
units = reader.stored_units(ch)
t_ms = (pulse["Time"] - pulse["trigger_time"]) * 1e3
names = ("df", "dissipation") if units == "Hz" else ("I", "Q")

fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
for ax, key, name, mean, std in (
        (axes[0], "Amp_I", names[0], ns.mean_I, ns.std_I),
        (axes[1], "Amp_Q", names[1], ns.mean_Q, ns.std_Q)):
    ax.plot(t_ms, pulse[key], lw=1.2)
    ax.axhline(mean, color="0.5", lw=0.8)
    for k, style in ((pulse["threshold_sigma"], "--"),
                     (pulse["end_sigma"], ":")):
        ax.axhline(mean + k * std, color="0.5", lw=0.8, ls=style)
        ax.axhline(mean - k * std, color="0.5", lw=0.8, ls=style)
    for mark, color in (("trigger_time", "tab:orange"),
                        ("below_threshold_time", "tab:red"),
                        ("settled_time", "tab:cyan")):
        if mark in pulse:
            ax.axvline((pulse[mark] - pulse["trigger_time"]) * 1e3,
                       color=color, lw=1)
    ax.set_ylabel(f"{name} ({units})")
axes[1].set_xlabel("time from trigger (ms)")
axes[0].set_title(f"channel {ch}, pulse {PULSE}: trigger (orange), back "
                  "below threshold (red), settled (cyan)")
plt.tight_layout(); plt.show()
```

The IQ Plane draws the pulse over the sweep the channel was tuned with.
`tuning_sweep` returns that sweep in counts, turned to sit under the samples,
with the bias point. `display_transform` gives the factor that takes stored
samples, or counts, into the view. Here the view is volts on the I and Q
axes:

```python
from rfmux.pulse_capture.analysis import (
    baseline_level, display_transform, tuning_sweep)

row = reader.tuning(ch)
cal = reader.df_calibration(ch)
f_sweep, sweep_counts, bias_point = tuning_sweep(row)

pulse_to_volts, _ = display_transform(cal, "df", units, "iq", "V")
counts_to_volts, _ = display_transform(cal, "iq", "counts", "iq", "V")
z = (pulse["Amp_I"] + 1j * pulse["Amp_Q"]) * pulse_to_volts
sweep = sweep_counts * counts_to_volts

plt.figure(figsize=(5.5, 5.5))
plt.plot(sweep.real, sweep.imag, color="0.6", lw=1, label="tuning sweep")
plt.plot((bias_point * counts_to_volts).real,
         (bias_point * counts_to_volts).imag, "k+", ms=12, label="bias point")
plt.scatter(z.real, z.imag, c=t_ms, s=10, cmap="viridis", label="pulse")
plt.colorbar(label="time from trigger (ms)")
plt.xlabel("I (V)"); plt.ylabel("Q (V)"); plt.axis("equal")
plt.title(f"channel {ch}: pulse over its sweep"); plt.legend(); plt.show()
```

The pulse leaves the bias point along the sweep and relaxes back to it. The
calibration is the sweep's slope at the bias point, so it is exact for small
shifts. A pulse this large travels part of the way around the resonance
circle. The part of its path that curves away from that slope reads as
dissipation in the Pulse View above.

### Changing units

A channel's samples are stored once, in hertz along df and dissipation when it
has a calibration. Every other view is one complex multiplication: a rotation
by the calibration's angle and a scale. With `cal` in hertz per volt and `vpc`
volts per count:

| view | factor on stored hertz |
|---|---|
| df and dissipation (Hz) | `1` |
| I and Q (V) | `1 / cal` |
| I and Q (counts) | `1 / (vpc * cal)` |

`display_transform(cal, stored_basis, stored_units, view_basis, view_units)`
returns the factor for any pair of views, and `apply_iq_conversion` applies
one to two real arrays.

```python
from rfmux.core.transferfunctions import apply_iq_conversion

vpc = reader.volts_per_count()
stored = pulse["Amp_I"] + 1j * pulse["Amp_Q"]          # df + j dissipation, Hz

to_volts, _ = display_transform(cal, "df", "Hz", "iq", "V")
to_counts, _ = display_transform(cal, "df", "Hz", "iq", "counts")
print(f"to volts : x {to_volts:.4g}   = 1/cal           : {1 / cal:.4g}")
print(f"to counts: x {to_counts:.4g}   = 1/(vpc * cal)   : {1 / (vpc * cal):.4g}")
print(f"a rotation by {np.degrees(np.angle(to_volts)):+.1f} degrees, "
      f"a scale of {abs(to_volts):.4g} V per Hz")

i_volts, q_volts = apply_iq_conversion(pulse["Amp_I"], pulse["Amp_Q"], to_volts)

# And back: volts on I and Q into hertz along df and dissipation.
back, _ = display_transform(cal, "iq", "V", "df", "Hz")
df_hz, diss_hz = apply_iq_conversion(i_volts, q_volts, back)
print("round trip exact:", np.allclose(df_hz, pulse["Amp_I"])
      and np.allclose(diss_hz, pulse["Amp_Q"]))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
axes[0].plot(t_ms, pulse["Amp_I"], label="df")
axes[0].plot(t_ms, pulse["Amp_Q"], label="dissipation")
axes[0].set_ylabel("Hz"); axes[0].set_title("as stored")
axes[1].plot(t_ms, i_volts, label="I")
axes[1].plot(t_ms, q_volts, label="Q")
axes[1].set_ylabel("V"); axes[1].set_title("the same pulse on I and Q")
for ax in axes:
    ax.set_xlabel("time from trigger (ms)"); ax.legend()
plt.tight_layout(); plt.show()
```

The projection matters for noise too. A channel is usually noisier along df
than along dissipation, and on the I and Q axes each axis sees a mix of the
two.

### Pileup, coincident pulses and noise samples

Every pulse records how it ended. `pileup` marks both fragments of a split
pair and `truncated` a capture cut off at the hard stop:

```python
for c in reader.channels:
    metas = list(reader.iter_pulse_metadata(c))
    print(f"ch{c}: {len(metas)} pulses, "
          f"{sum(m['pileup'] for m in metas)} pileup, "
          f"{sum(m['truncated'] for m in metas)} truncated")
```

An event lists the pulses that make it up. Its `kind` is `"pulses"` or
`"noise"`, its `members` name each pulse by channel and index, and `dumped`
names the channels saved with it without a trigger:

```python
print(f"coincidence window: "
      f"{reader.metadata['coincidence_window_s'] * 1e3:g} ms\n")
events = list(reader.iter_events())
# The first few, and the first noise samples further on.
for event in events[:5] + [e for e in events[5:] if e["kind"] == "noise"][:2]:
    members = [(m["channel"], m["pulse_idx"]) for m in event["members"]]
    if event["kind"] == "noise":
        t0, t1 = event["window"]
        print(f"  event {event['event_idx']:>3}  noise sample   "
              f"channels {event['dumped']} over {(t1 - t0) * 1e3:.1f} ms, "
              f"pulses inside it: {members}")
        continue
    spread_ms = (event["members"][-1]["trigger_time"]
                 - event["trigger_time"]) * 1e3
    what = "coincident" if len({c for c, _ in members}) > 1 else "one channel"
    print(f"  event {event['event_idx']:>3}  {what:<14} pulses {members}, "
          f"{spread_ms:.2f} ms apart, saved without a trigger: "
          f"{event['dumped']}")
```

`get_event` returns the saved samples as well. This draws one coincident
event, each channel about its own baseline. `baseline_level` gives the level
a pulse triggered from, and for a channel that did not trigger the median of
its samples before the event:

```python
coincident = next(e for e in reader.iter_events()
                  if e["kind"] == "pulses" and len(e["members"]) > 1)
event = reader.get_event(coincident["event_idx"])
t0 = event["trigger_time"]

plt.figure(figsize=(9, 3.5))
for m in event["members"]:
    wf = reader.get_pulse(m["channel"], m["pulse_idx"])
    plt.plot((wf["Time"] - t0) * 1e3,
             wf["Amp_I"] - baseline_level(wf, "I", t0),
             label=f"ch{m['channel']} pulse {m['pulse_idx']}")
for c, wf in event["dump"].items():
    plt.plot((wf["Time"] - t0) * 1e3,
             wf["Amp_I"] - baseline_level(wf, "I", t0),
             ls=":", lw=1, label=f"ch{c}, no trigger")
plt.xlabel("time from the event's first trigger (ms)")
plt.ylabel("first stored axis, about its baseline")
plt.title(f"event {event['event_idx']}"); plt.legend(); plt.show()
```

The channels are in their own stored units here: hertz for the calibrated
ones, volts for the channel without a detector.

A noise sample holds every channel over the same window, taken whether or not
anything triggered. Its `members` are the pulses that happened to fall inside
it. The samples with none are the ones to measure the noise from. The
simulator pulses every 50 ms, so few of these samples are free of a pulse.
A real capture is mostly quiet.

```python
samples = [reader.get_event(e["event_idx"]) for e in reader.iter_events()
           if e["kind"] == "noise"]
quiet = [s for s in samples if not s["members"]]
print(f"{len(samples)} noise samples, {len(quiet)} without a pulse in them")
for c in reader.channels:
    trained = reader.noise_stats(c).std_I
    if quiet:
        sigma = np.mean([np.std(s["dump"][c]["Amp_I"]) for s in quiet])
        print(f"  ch{c}: sigma over the quiet samples {sigma:.4g} "
              f"{reader.stored_units(c)}, trained sigma {trained:.4g}")
    else:
        print(f"  ch{c}: trained sigma {trained:.4g} {reader.stored_units(c)}")
```

### A channel's sweep and fit

The `tuning` row is shaped like the `bias_kids` entry for the channel, so it
holds the sweep, the fitted resonance and the bias point as well as the
calibration. Scalars and fit parameters come back as values, the sweep as
arrays:

```python
row = reader.tuning(ch)
fit = row.get("nonlinear_fit_params") or {}
print(f"bias frequency {row['bias_frequency'] / 1e6:.6f} MHz, "
      f"df calibration {abs(row['df_calibration']):.4g} Hz/V "
      f"({row['df_calibration_source']})")
for key in ("fr", "Qr", "amp", "phi", "a"):
    if key in fit:
        print(f"  {key:<4} {fit[key]:.6g}")

f_mhz = np.asarray(row["frequencies"]) / 1e6
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
axes[0].plot(f_mhz, np.abs(row["iq_complex"]), ".", ms=4, label="sweep")
if "nonlinear_model_iq" in row:
    axes[0].plot(f_mhz, np.abs(row["nonlinear_model_iq"]), lw=1, label="fit")
axes[0].axvline(row["bias_frequency"] / 1e6, color="k", lw=0.8, ls=":",
                label="bias")
axes[0].set_xlabel("frequency (MHz)"); axes[0].set_ylabel("|S21| (counts)")
axes[0].legend()
axes[1].plot(np.real(row["iq_complex"]), np.imag(row["iq_complex"]), ".", ms=4)
axes[1].set_xlabel("I (counts)"); axes[1].set_ylabel("Q (counts)")
axes[1].axis("equal")
fig.suptitle(f"channel {ch}: the sweep its calibration was read from")
plt.tight_layout(); plt.show()

reader.close()
```

## 8. Fast (PFB) capture

The fast streamer carries up to **4 channels of one module** at 2.44 MHz,
128× the slow stream here. Enable it through `configure_streamer` and turn it
off in a `finally`.

The same `PulseCaptureConfig` is reused: `session_kwargs(PFB_SAMPLING_FREQ)`
re-derives the buffer, training length and confirmation count for the new rate.
`duration_s` includes the noise training, so this short demo trains for
100 ms. The simulator generates the PFB stream slower than real time.

```python
from dataclasses import replace

fast_config = replace(capture_config, noise_train_ms=100.0)

await crs.configure_streamer(cfg.dec_stage, short=cfg.short_packets,
                             modules=[MODULE],
                             pfb_channels=CHANNELS, pfb_module=MODULE)
try:
    fast_session = PulseCaptureSession(
        channels=CHANNELS, module=MODULE, streamer_mode="fast",
        sample_rate=PFB_SAMPLING_FREQ,
        hdf5_path=str(OUTPUT_DIR / "pulse_capture_fast.h5"),
        **fast_config.session_kwargs(PFB_SAMPLING_FREQ),
    )
    fast_session.start()
    covered = await run_pfb_source(fast_session, host, CHANNELS,
                                   module=MODULE, duration_s=0.4)
    fast_session.stop()
    print(f"{fast_session.total_pulses} pulses over {covered*1e3:.0f} ms")
finally:
    await crs.configure_streamer(cfg.dec_stage, short=cfg.short_packets,
                                 modules=[MODULE], pfb_channels=[])
```

## 9. Both streams at once, with matched pairs

`DualPulseCaptureSession` runs one detection engine per stream, each with its
own noise training, and matches pulses by trigger time. Two triggers pair when
they fall within three slow samples of each other. A trigger with no partner
is held until the longest capture could have closed, plus 50 ms, then released
as a one-sided pair. `run_dual_source` drives both sockets; whichever side
finishes first stops the other.

Every pair stores both streams over one common interval, the union of the two
saved records, recorded as `window_t0`/`window_t1`: the same event at both
rates over the same interval. A one-sided pair gets the other stream's window
when that buffer still covers it; after `pair_window_wait_s` (3 s) the pair is
written without it. Metrics are computed from each stream's own triggered
samples.

The slow stream's timestamps are late by its decimation filter's group delay.
The session shifts the slow clock back by that delay before matching and
writing; the shift is in the file as `slow_time_offset_s`. Add it back if you
compare against raw packet stamps.

```python
pairs = []

dual = DualPulseCaptureSession(
    channels=CHANNELS, module=MODULE,
    slow_rate=fs, fast_rate=PFB_SAMPLING_FREQ,
    config=capture_config,
    hdf5_path=str(OUTPUT_DIR / "pulse_capture_dual.h5"),
    on_pair=lambda p: pairs.append(p),
)
dual.start()

await crs.configure_streamer(cfg.dec_stage, short=cfg.short_packets,
                             modules=[MODULE],
                             pfb_channels=CHANNELS, pfb_module=MODULE)
try:
    slow_elapsed, fast_elapsed = await run_dual_source(
        dual, host, CHANNELS, module=MODULE, duration_s=2.0)
finally:
    await crs.configure_streamer(cfg.dec_stage, short=cfg.short_packets,
                                 modules=[MODULE], pfb_channels=[])
dual.stop()

stats = dual.stats()
print(f"slow {stats['slow']['total_pulses']} pulses over {slow_elapsed:.2f} s")
print(f"fast {stats['fast']['total_pulses']} pulses over {fast_elapsed:.2f} s")
print(f"matched {stats['pairs_matched']}, unmatched {stats['pairs_unmatched']}")
```

One pair, plotted: the same event sampled two ways on a shared time axis.

```python
two_sided = [p for p in pairs
             if p.get("slow_idx") is not None and p.get("fast_idx") is not None]
print(f"{len(two_sided)} of {len(pairs)} pairs have both streams")

if two_sided:
    p = two_sided[len(two_sided) // 2]
    plt.figure(figsize=(9, 3.5))
    for key, label in (("slow_tod", f"slow ({fs/1e3:.0f} kHz)"),
                       ("fast_tod", f"fast ({PFB_SAMPLING_FREQ/1e6:.2f} MHz)")):
        tod = p.get(key)
        if tod is None:
            continue
        t = np.asarray(tod["Time"], dtype=float)
        plt.plot((t - np.nanmin(t)) * 1e3, tod["Amp_I"],
                 marker="." if key == "slow_tod" else None,
                 ms=4, lw=1, label=label)
    plt.xlabel("time (ms)"); plt.ylabel("I (V)")
    plt.title(f"ch{p['channel']} pair #{p['pair_idx']}: "
              f"trigger offset {(p.get('time_offset') or 0.0)*1e6:+.0f} µs")
    plt.legend(); plt.show()
```

Every pair stores that common span from both streams, thousands of samples at
2.44 MHz, so check the file size before a long dual capture.

The dual file keeps the two streams in separate groups plus a match table.
`PulseHDF5Reader` reports `dual=True` and takes a `stream=` argument:

```python
with PulseHDF5Reader(OUTPUT_DIR / "pulse_capture_dual.h5") as r:
    print("dual layout:", r.dual, "| streams:", r.streams)
    for ch in r.channels:
        print(f"  ch{ch}: slow={r.pulse_count(ch, stream='slow')}, "
              f"fast={r.pulse_count(ch, stream='fast')}, "
              f"pairs={r.pair_count(ch)}")
```

## 10. Where this maps in Periscope

The panel sets the same objects this notebook does:

| Periscope control | API equivalent |
|---|---|
| **Streamer…** dialog | `StreamerConfig` + `crs.configure_streamer(...)` |
| **Channels** field: `1,2`, `2-19`, `1,5-8,20` | `parse_channel_spec(...)` |
| **Channels** field: `all` / `*` | `crs.get_biased_channels(module)` |
| **Settings…** dialog | `PulseCaptureConfig` fields |
| **Thresh σ** / **End σ** / **Pileup** | `threshold_sigma`, `end_sigma`, `enable_pileup` |
| Mode: slow / fast / both | `run_slow_source` / `run_pfb_source` / `run_dual_source` |
| **▶ Start** | `capture_session.start()` + a source coroutine |
| (how the GUI feeds the session) | `SlowIngest`, the class `run_slow_source` uses too |
| **⟳ Re-estimate Noise** | `capture_session.re_estimate_noise()` |
| Live pulse / histogram / template plots | `on_pulse`, `on_histograms`, `on_templates` callbacks |
| **Plot** field: `1,2,4`, `1-5`, `*` | `plot_groups(...)`, `combine_histograms(...)`, `combine_templates(...)` in `rfmux.pulse_capture.analysis` |
| **Units** control | `display_transform(...)` |
| Output `.h5` + Session Browser review | `hdf5_path=` + `PulseHDF5Reader` |

`pulse_capture_flow.py` in this folder runs the same sequence as a plain
script, to copy from or to run against MOCK:

    python pulse_capture_flow.py MOCK      # simulated CRS
    python pulse_capture_flow.py 0042      # real board

```python
# Only tear down a streamer this notebook started. If section 1 attached to
# Periscope's CRS, stopping it here would kill Periscope's own live plots.
if IS_MOCK:
    await crs.stop_udp_streaming()
    print("simulated streamer stopped")
else:
    print("left the streamer running: it is not ours to stop")
```
