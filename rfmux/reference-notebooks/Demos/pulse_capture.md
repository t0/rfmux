---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Pulse Capture

End-to-end pulse capture from the Python API: configure the streamers, detect
pulses live, persist them, and analyse them — **without opening Periscope**.

Everything here is the same code path Periscope's *Pulse Capture* panel drives.
The panel builds a `PulseCaptureSession`, feeds it from the streamers, and draws
the callbacks; this notebook builds the same session, feeds it with the same
source functions, and prints or plots instead. There is no capability in the GUI
that is unavailable here.

| Piece | Module |
|---|---|
| Streamer setup + link-budget math | `rfmux.algorithms.measurement.streamer_config` |
| Detection engine (ring buffer, triggering) | `rfmux.algorithms.measurement.pulse_detection` |
| Live capture orchestration | `rfmux.algorithms.measurement.pulse_capture_session` |
| Concurrent slow+fast with matching | `rfmux.algorithms.measurement.pulse_capture_dual` |
| Packet sources that feed a session | `rfmux.algorithms.measurement.pulse_sources` |
| Per-pulse metrics (SNR, derived τ) | `rfmux.algorithms.measurement.pulse_analysis` |
| Streaming HDF5 persistence | `rfmux.algorithms.measurement.pulse_hdf5` |

## How to use this document

**This is a runnable notebook, not a web page.** Every grey block below is a live
code cell: put the cursor in it and press **Shift+Enter** to execute it. The
output — numbers, tables, plots — appears underneath the cell as it runs.

- **Run the cells in order, top to bottom.** Later cells use variables the
  earlier ones defined, so skipping ahead will fail with a `NameError`. If you
  lose your place, *Kernel → Restart Kernel and Run All Cells* starts clean.
- **The outputs you see are the ones you just produced.** This file is stored as
  jupytext markdown, which keeps no saved outputs, so a cell is blank until you
  run it. Nothing here can show you a stale number from someone else's run.
- **Editing is encouraged.** Change a threshold, a channel list, a capture
  length, and re-run — that is what this document is for. The shipped copy is
  read-only, so *File → Save Notebook As…* to keep your changes.
- **Section 1 is the one exception to running everything.** It offers three ways
  to get a CRS — attach to a running one, use your own board, or simulate one —
  and you run only the one that fits. Everything after it is identical whichever
  you chose.
- **Captures are written to disk** in `OUTPUT_DIR` (printed by the next cell).
  Section 7 reads them back, and Periscope can open them in review mode.

```python
%matplotlib inline

import asyncio
import os
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import rfmux
from rfmux.algorithms.measurement.pulse_analysis import counts_to_hz_scale
from rfmux.algorithms.measurement.pulse_capture_dual import (
    DualPulseCaptureSession,
)
from rfmux.algorithms.measurement.pulse_capture_session import (
    PulseCaptureConfig, PulseCaptureSession,
)
from rfmux.algorithms.measurement.pulse_hdf5 import PulseHDF5Reader
from rfmux.algorithms.measurement.pulse_sources import (
    run_dual_source, run_pfb_source, run_slow_source,
)
from rfmux.algorithms.measurement.streamer_config import (
    PFB_SAMPLE_RATE, StreamerConfig, describe, slow_sample_rate, validate,
)

# Captures are written here. Reference notebooks are provisioned to a
# read-only directory, so writing next to the notebook would fail for anyone
# who opened it from Periscope — default to a writable scratch directory and
# say where it is. Override with RFMUX_DEMO_OUTPUT.
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
| **A. Attach to a running board** | Periscope launched this notebook, or you know the board's address | below |
| **B. Your own board** | You have hardware and its serial | below |
| **C. Simulate one** | No hardware, and nothing already running | section 2 |

On real hardware the detectors must already be biased, since pulse detection
works on the deviation of a parked carrier. Run `simplified_tuning_flow.py` (in
this folder) first, or use the Periscope tuning panels.

### A. Attach to a board that is already running

Use this when Periscope is driving a board — real or simulated — and you want to
work with *that* one rather than starting your own.

Periscope sets `RFMUX_CRS_HOSTNAME` when it launches this notebook, which is how
the cell finds the board with no configuration from you. It is not magic and not
required: paste an address into `HOSTNAME` and this works from any kernel.

Attaching matters most in mock mode. A simulated CRS is created, not discovered
— its RPC port is assigned by the OS at startup — so there is no address to look
up, and a second `create_mock_crs()` gives you a *second, unrelated* simulation
streaming to the same UDP port as the first. A receiver then sees both
interleaved, with no error anywhere.

```python
HOSTNAME = os.environ.get("RFMUX_CRS_HOSTNAME")   # or paste "127.0.0.1:43431"
SERIAL = os.environ.get("RFMUX_CRS_SERIAL", "0000")

if HOSTNAME:
    _s = rfmux.load_session(
        f'!HardwareMap [ !CRS {{ serial: "{SERIAL}", '
        f'hostname: "{HOSTNAME}" }} ]')
    crs = _s.query(rfmux.CRS).one()
    await crs.resolve()
    host = HOSTNAME.split(":")[0]
    print(f"attached to CRS {SERIAL} at {HOSTNAME}")
else:
    print("Nothing advertised a board. Set HOSTNAME above, or use option B "
          "(your own board) or section 2 (simulation).")
```

### B. Your own board

```python
# SERIAL = "0042"
# _s = rfmux.load_session(f'!HardwareMap [ !CRS {{ serial: "{SERIAL}" }} ]')
# crs = _s.query(rfmux.CRS).one()
# await crs.resolve()
# host = crs.tuber_hostname
# print(f"connected to CRS {SERIAL} at {host}")
```

## 2. Mock mode configuration

**Skip this section entirely if section 1 already gave you a CRS** — the cell
below no-ops in that case.

If you have no hardware, this stands up a simulated CRS: a couple of resonators
with carriers parked on them, and periodic quasiparticle pulses to detect.

Pulse heights are drawn uniformly between `pulse_random_amp_min` and
`..._max` rather than repeating one value, so the amplitude histogram in
section 7 shows a distribution instead of a single spike — which is the whole
point of plotting one.

The noise is deliberately not idealised either, because a detector that only
works on white noise is not worth testing:

- **White readout noise** (`udp_noise_level`) — the flat floor the σ thresholds
  are measured against.
- **Quasiparticle number fluctuations** (`nqp_noise_enabled`) — physical
  generation–recombination noise in the resonator itself.
- **TLS 1/f frequency noise** (`tls_noise_enabled`) — two-level systems in the
  substrate make the resonant frequency wander with a `1/f**alpha` spectrum.
  This is the interesting one: it is exactly what a naive fixed-baseline
  detector triggers on endlessly. It is what the rolling-median baseline and the
  edge gate in section 4 exist to survive, so leaving it off would let this
  notebook demonstrate a detector that cannot handle a real detector.

At these settings the quasiparticle term dominates the slow-stream noise floor:
σ ≈ 8.6 counts at stage 1, against ≈ 1 count when it is set ten times lower.
That is worth knowing, because it means the σ your thresholds are quoted in is a
*physical* quantity here, not a readout artefact — and it is what makes the
pulse SNRs below look like real detector numbers rather than the enormous ones a
noiseless simulation produces.

The 1/f on top of it is correlated rather than white, so it moves the baseline
instead of scattering it: about 2.8 σ of wander over a three-second capture,
against 2.2 σ with TLS off. The detector absorbs that without complaint — same
59 pulses either way, no over-long windows.

**Which is the point, and also the trap.** Turn the quasiparticle noise back
down and the same absolute drift becomes 14 σ, and roughly one window in seven
stays open until the hard stop, because a pulse riding a drifting baseline is
slow to fall back inside `end_sigma`. Those windows still measure the right τ —
the decay is derived from the peak and the threshold crossing, not the window
length — but their `duration_ms` describes the baseline, not the event. That
failure mode is why the rolling median and edge gate exist, and it is exactly
the kind of thing that stays invisible in simulation and then bites on
hardware.

> ⚠️ **This cell refuses to run if something is already streaming.** Two
> simulations send to the same UDP port, so a receiver gets both interleaved —
> no exception, no dropped packets, just samples from two unrelated detectors in
> one trace. Rather than leave that to be noticed later, the cell checks and
> stops, and the message says which case you are in. If Periscope is in mock
> mode, attach to *its* simulation with option 1A instead.

```python
MOCK_CONFIG = {
    "num_resonances": 2,
    "resonator_random_seed": 42,
    "auto_bias_kids": True,        # park carriers on the resonators
    "bias_amplitude": 0.001,

    # ── Noise (these are the shipped defaults, spelled out) ─────
    "udp_noise_level": 10.0,       # white readout noise (ADC counts)
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
    print("already connected — skip this cell")
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
            f"Something is already using the streamer port — {conflict}.\n"
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

### Ready?

Whichever route you took, this is the checkpoint — it confirms what the rest of
the notebook will be talking to.

```python
if crs is None:
    raise RuntimeError(
        "No CRS. Run option 1A (attach), 1B (your board), or the cell above "
        "(simulate one) before continuing.")

print(f"CRS       {crs.tuber_hostname}")
print(f"streamers {host}")
print(f"module {MODULE}, channels {CHANNELS}")
print("simulation created by this notebook" if IS_MOCK
      else "pre-existing board — this notebook will not tear it down")
```

## 3. Configure the streamers

Two streams carry samples off the board:

- the **slow** readout stream, decimated in stages from ~38 kHz down to ~596 Hz,
  carrying up to 1024 channels per module (port 9876);
- the **fast** PFB stream at ~1.22 MHz, limited to 4 channels of one module
  (port 9877).

The decimation is a physics choice, not a taste one: you want roughly **10 or
more samples across one pulse decay constant**, or the decay is too sparsely
sampled to fit. Below that the pulse is a spike; far above it you are paying
bandwidth for nothing and risking dropped packets.

`validate()` reports the hardware rules (long packets need stage ≥ 3, the 1 GbE
budget, OS receive-buffer advice) as `(severity, message)` pairs. `describe()`
returns the derived rates and link budget.

> If you attached to Periscope's CRS in section 1, remember the streamer is a
> shared resource: changing the decimation here changes it for Periscope's plots
> too, exactly as the *Streamer…* dialog would. That is usually what you want —
> one board, one configuration — but it is not a private setting.

```python
PULSE_TAU_S = 1e-3          # expected decay constant

needed_fs = 10.0 / PULSE_TAU_S
dec = next(d for d in range(6, -1, -1) if slow_sample_rate(d) >= needed_fs)
cfg = StreamerConfig(dec_stage=dec, short_packets=(dec < 3), modules=[MODULE])

print(f"τ = {PULSE_TAU_S*1e3:.1f} ms → need ≥ {needed_fs:.0f} Hz "
      f"→ stage {dec} ({slow_sample_rate(dec):.0f} Hz)")

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
    # The simulated streamer needs a moment to fill after a rate change.
    await crs.start_udp_streaming()
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

`all` (or `*`) is deliberately *not* a list — it means "ask the board", which
only the board can answer, so the parser returns `None` and leaves the question
to `get_biased_channels`. That reads every channel's amplitude in one batched
round trip rather than one RPC per channel, and drops any channel the packet
cannot carry:

```python
print(f"{'all'!r:12} -> {parse_channel_spec('all')}   (resolve against the board)")

# 128 is the short-packet width: a channel above it is in no packet, however
# it is biased.  Pass the width you are actually streaming.
biased = await crs.get_biased_channels(MODULE, max_channels=128)
print(f"biased on module {MODULE}: {len(biased)} channel(s)")
print(f"  {biased[:12]}{' ...' if len(biased) > 12 else ''}")
```

Whatever you choose, `CHANNELS` below is just a list of ints — set it from a
spec, from `get_biased_channels`, or by hand.

## 4. Choose the detection parameters

`PulseCaptureConfig` holds every user-facing parameter in **physical units**
(σ and milliseconds). It converts them to samples for whatever stream rate you
hand it, which is what lets one configuration work unchanged across a 2000×
span of rates from the decimated slow stream to the PFB stream.

How the detector works:

- **Triggering is two-sided with hysteresis.** A capture opens when *either* I or
  Q leaves the ±`threshold_sigma` noise band, and closes only when *both* have
  returned inside ±`end_sigma`. `end_sigma` must sit below `threshold_sigma`, or
  captures would end the instant they began.
- **Triggers must be confirmed.** `trigger_samples` consecutive samples have to
  clear the threshold. Left at 0 it is *derived from the stream rate* to hold
  accidental triggers under `max_accidental_per_min`: at 596 Hz one sample is
  ample evidence, at 1.22 MHz it is not.
- **An edge gate suppresses drift.** A real pulse rises; 1/f wander does not.
  The detector also demands a rise of `threshold_sigma` jump-σ within
  `edge_lookback` samples, with the baseline cancelled — so slow drift cannot
  trigger even when the baseline estimate lags behind it.
- **The baseline is a rolling median**, re-estimated continuously over
  `baseline_window` samples. A median rather than a mean because it ignores
  pulses as long as they stay a minority of the window.
- **`max_pulse_ms` is the primary control.** It sizes the ring buffer (×1.5 for
  pre-trigger margin and the end-confirmation tail) and sets the floor under the
  baseline window and the noise training length. Estimate it *generously*: a
  capture that outlasts the ring silently loses its own rising edge.
- **Captures cannot wedge.** One that never satisfies its end condition is
  force-closed at a hard stop of 1.2 × `max_pulse_ms` and flagged `truncated`.

![Anatomy of one capture window](pulse_capture_anatomy.png)

The shaded region is what gets saved. It opens `margin_fraction` of the window
before the trigger — 10% by default, too small to draw legibly here — so the
record keeps some pre-trigger baseline to measure against rather than starting
exactly at the crossing.

Where it *closes* is `save_to_end_confirmed`, on by default: the window runs to
the end-of-pulse confirmation, keeping the whole decay tail. Those samples are
already in the ring, so this costs disk rather than acquisition. Turn it off and
the window stops a `margin_fraction` tail past the below-threshold instant
instead — shorter files, and window length becomes a property of the pulse rather
than of the baseline, since how long the end confirmation takes depends on where
the baseline was wandering. Measured on the mock at 19 kHz with a 1 ms decay,
identical injected pulses gave a median window of 5.56 ms with the tail and
3.93 ms without.

Worth turning off for PFB captures, where windows already carry ~64× the samples,
and at high count rates, where longer windows overlap and more events get flagged
as pileup.

Note what does *not* change: `duration_ms` is measured from the threshold
crossings, not from the length of the saved window, so it reports the pulse
either way — 3.09 ms in both runs above.

`describe()` reports everything derived at a given rate, and `validate()` catches
inconsistent settings before you spend a capture on them.

```python
capture_config = PulseCaptureConfig(
    threshold_sigma=5.0,    # trigger when I or Q leaves ±5σ
    end_sigma=1.5,          # close when BOTH are back inside ±1.5σ
    min_pulse_ms=0.2,       # glitch filter: drop anything shorter
    max_pulse_ms=50.0,      # longest recordable pulse — sizes the ring
    noise_train_ms=50.0,    # 0 would derive this from max_pulse_ms
    enable_pileup=True,     # split piled-up events on a sharp re-rise
)

fs = slow_sample_rate(cfg.dec_stage)
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
print(f"  hard stop       {d['max_capture_ms']:.0f} ms")
```

Note how the confirmation length changes on its own with the rate — this is the
mechanism that makes one config portable across streams:

```python
for rate, label in [(596.0, "slow, stage 6"), (fs, f"slow, stage {dec}"),
                    (PFB_SAMPLE_RATE, "fast (PFB)")]:
    dd = capture_config.describe(rate)
    print(f"{label:<18} {rate:>10,.0f} Hz → confirm "
          f"{dd['trigger_samples']} sample(s), "
          f"{dd['accidental_per_min']:.2e} accidentals/min")
```

## 5. One-shot capture

`trigger_capture` is the simplest entry point — one call, no session lifecycle,
everything handed back in memory. Underneath it is a thin caller over exactly
the machinery the rest of this notebook uses, so the two are never out of step.

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
          f"I={n.mean_I:.0f}±{n.std_I:.2f}")
res
```

Because it is a session underneath, `hdf5_path=` makes the one-shot call write a
real capture file too — pulses, histograms and templates, openable in Periscope.
Use a session directly (next section) when a capture is long enough that holding
every pulse in memory stops being reasonable.

Each pulse comes with its metrics already computed. `res.summaries[ch][idx]` is
`pulse_summary()` output — the single source of truth for per-pulse scalars,
which the HDF5 writer, the histograms and the GUI all derive from, so they can
never disagree.

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
plt.xlabel("time (ms)"); plt.ylabel("counts")
plt.title("One captured pulse"); plt.legend(); plt.show()
```

The derived τ is worth understanding, because it is not a fit. It uses two
well-measured points on the falling edge — the peak, and the moment the envelope
falls back through the trigger threshold:

$$\tau = \frac{t_{\rm thr} - t_{\rm peak}}{\ln(\mathrm{SNR}_{\rm peak} / \sigma_{\rm thr})}$$

Taking the *ratio* of amplitudes cancels the unknown event energy, so for a
detector with one decay time every energy line collapses onto a single τ. It is
a live cross-check, not a precision measurement: the discrete crossing sample
lands slightly below the true crossing, so it runs a few percent low.

## 6. Live capture with streaming persistence

`PulseCaptureSession` is what the panel actually runs. It trains on noise, then
detects, and as each pulse closes it appends to HDF5, updates running histograms
and stacks a trigger-aligned template — all incrementally, so memory stays flat
no matter how long you capture.

You feed it from `run_slow_source` / `run_pfb_source`, or from any sample source
of your own: the session's interface is just `feed_sample(channel, I, Q, t)`.

`on_pulse` is the callback the GUI uses to update its display. Here it prints.

```python
def on_pulse(channel, pulse_idx, summary, waveform):
    if pulse_idx <= 3:       # keep the output short
        print(f"  ch{channel} #{pulse_idx}: {summary['snr']:.0f}σ, "
              f"{summary['duration_ms']:.2f} ms, τ={summary['tau_ms']:.2f} ms")

session = PulseCaptureSession(
    channels=CHANNELS, module=MODULE, streamer_mode="slow",
    sample_rate=fs, hdf5_path=str(OUTPUT_DIR / "pulse_capture_demo.h5"),
    on_pulse=on_pulse,
    **capture_config.session_kwargs(fs),
)
session.start()
covered = await run_slow_source(session, host, module=MODULE, duration_s=3.0)
session.stop()

print(f"\n{session.total_pulses} pulses over {covered:.2f} s of sample time")
print(f"rolling baseline median over {session.baseline_window:,} samples "
      f"({session.baseline_window / fs:.2f} s)")
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
          f"noise I={n.mean_I:.0f}±{n.std_I:.2f}, Q={n.mean_Q:.0f}±{n.std_Q:.2f}")

# Per-pulse metadata without loading any waveforms
for meta in list(reader.iter_pulse_metadata(reader.channels[0]))[:5]:
    print(f"  #{meta['pulse_idx']:04d} snr={meta.get('snr', 0):.0f}σ "
          f"dur={meta.get('duration_s', 0)*1e3:.2f} ms "
          f"τ={meta.get('tau_s', float('nan'))*1e3:.2f} ms")
```

Histograms accumulate as the capture runs and expand their ranges automatically
when a pulse falls outside the current binning, so you never have to guess the
scale in advance. Keys are `<metric>_edges`, `<metric>_bins` (centers) and
`<metric>_counts_ch<N>`.

```python
hist = reader.get_histograms()
fig, axes = plt.subplots(1, 3, figsize=(13, 3))
for ax, metric, xlabel in zip(
        axes, ["snr", "amplitude", "tau_ms"],
        ["peak deviation (σ)", "amplitude (counts)", "derived τ (ms)"]):
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
window — window starts carry a pre-margin that varies with pulse length, so
stacking on them would smear the template. The mean beats the noise down as
1/√N; the shaded band is the per-bin RMS spread, which is what separates real
pulse-to-pulse variation from measurement noise.

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
plt.xlabel("time from trigger (ms)"); plt.ylabel("I (counts)")
plt.title("Trigger-aligned template"); plt.legend(); plt.show()
```

### Calibrated amplitudes

Waveforms are stored in raw ADC counts, which are only comparable across
channels once calibrated. `counts_to_hz_scale` turns them into Δf, and is the
same conversion behind the panel's counts/Hz selector.

The calibration itself comes from `bias_kids`, which returns a `df_calibration`
(Hz per radian) per detector. Hand those to the session and they are written
into the capture file alongside the pulses:

    bias_results = await bias_kids(crs=crs, multisweep_results=...,
                                   module=MODULE)
    df_cals = {ch: d["df_calibration"] for ch, d in bias_results.items()
               if "df_calibration" in d}

    session = PulseCaptureSession(..., df_calibrations=df_cals)

This notebook's mock detectors were parked by `auto_bias_kids`, which skips that
sweep-and-fit step, so the channels below are uncalibrated and
`counts_to_hz_scale` returns `None`. Treat `None` as "show counts" rather than
substituting 1.0 — unscaled counts mislabelled as Hz are worse than no
calibration at all.

```python
for ch in reader.channels:
    scale = counts_to_hz_scale(reader.df_calibration(ch))
    if scale is None:
        print(f"  ch{ch}: uncalibrated — amplitudes stay in counts")
    else:
        peak = reader.get_pulse_metadata(ch, 1).get("peak_amp", float("nan"))
        print(f"  ch{ch}: {peak:.0f} counts → {peak * scale:.1f} Hz")

reader.close()
```

## 8. Fast (PFB) capture

The fast streamer carries up to **4 channels of one module** at ~1.22 MHz —
about 64× the slow stream here, enough to resolve a rise time the slow stream
sees as a single sample. Enable it through `configure_streamer`, and always tear
it down in a `finally` so a failed capture doesn't leave it running.

The same `PulseCaptureConfig` is reused: `session_kwargs(PFB_SAMPLE_RATE)`
re-derives the buffer, training length and confirmation count for the new rate.

```python
await crs.configure_streamer(cfg.dec_stage, short=cfg.short_packets,
                             modules=[MODULE],
                             pfb_channels=CHANNELS, pfb_module=MODULE)
try:
    fast_session = PulseCaptureSession(
        channels=CHANNELS, module=MODULE, streamer_mode="fast",
        sample_rate=PFB_SAMPLE_RATE,
        hdf5_path=str(OUTPUT_DIR / "pulse_capture_fast.h5"),
        **capture_config.session_kwargs(PFB_SAMPLE_RATE),
    )
    fast_session.start()
    covered = await run_pfb_source(fast_session, host, CHANNELS,
                                   duration_s=0.25)
    fast_session.stop()
    print(f"{fast_session.total_pulses} pulses over {covered*1e3:.0f} ms")
finally:
    await crs.configure_streamer(cfg.dec_stage, short=cfg.short_packets,
                                 modules=[MODULE], pfb_channels=[])
```

## 9. Both streams at once, with matched pairs

`DualPulseCaptureSession` runs two independent detectors — one per stream, each
with its own noise training and rate-appropriate parameters — and matches their
pulses by trigger time. `run_dual_source` drives both sockets concurrently;
whichever side finishes first stops the other, since a matcher fed by one stream
alone just accumulates one-sided pulses.

Every matched pair also carries a **union time window**: the widest interval
spanned by either trigger, extracted from *both* ring buffers. So a pair gives
you the same event at 19 kHz and at 1.22 MHz over a common span — which is what
makes cross-stream comparison meaningful. Metrics stay computed on each stream's
own triggered core, so the union windows never contaminate the statistics.

```python
pairs = []

dual = DualPulseCaptureSession(
    channels=CHANNELS, module=MODULE,
    slow_rate=fs, fast_rate=PFB_SAMPLE_RATE,
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

Plotting one pair shows the point of the exercise — the same event, sampled two
ways, on a shared time axis:

```python
two_sided = [p for p in pairs
             if p.get("slow_idx") is not None and p.get("fast_idx") is not None]
print(f"{len(two_sided)} of {len(pairs)} pairs have both streams")

if two_sided:
    p = two_sided[len(two_sided) // 2]
    plt.figure(figsize=(9, 3.5))
    for key, label in (("slow_tod", f"slow ({fs/1e3:.0f} kHz)"),
                       ("fast_tod", f"fast ({PFB_SAMPLE_RATE/1e6:.2f} MHz)")):
        tod = p.get(key)
        if tod is None:
            continue
        t = np.asarray(tod["Time"], dtype=float)
        plt.plot((t - np.nanmin(t)) * 1e3, tod["Amp_I"],
                 marker="." if key == "slow_tod" else None,
                 ms=4, lw=1, label=label)
    plt.xlabel("time (ms)"); plt.ylabel("I (counts)")
    plt.title(f"ch{p['channel']} pair #{p['pair_idx']} — "
              f"trigger offset {p['time_offset']*1e6:+.0f} µs")
    plt.legend(); plt.show()
```

Budget for the file size: every pair stores its union window from *both* rings,
and at 1.22 MHz that window is thousands of samples. The capture above averages
roughly 200 kB per pair — about 16 MB for two seconds at this (very high) mock
pulse rate. Real detector rates are far lower, but on a long run the union
windows, not the triggered cores, are what fills the disk.

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

If you also use the GUI, the correspondence is exact — the panel sets the same
objects this notebook does:

| Periscope control | API equivalent |
|---|---|
| **Streamer…** dialog | `StreamerConfig` + `crs.configure_streamer(...)` |
| **Channels** field: `1,2`, `2-19`, `1,5-8,20` | `parse_channel_spec(...)` |
| **Channels** field: `all` / `*` | `crs.get_biased_channels(module)` |
| **Settings…** dialog | `PulseCaptureConfig` fields |
| Threshold σ / End σ / Pileup | `threshold_sigma`, `end_sigma`, `enable_pileup` |
| Mode: slow / fast / both | `run_slow_source` / `run_pfb_source` / `run_dual_source` |
| **▶ Start** | `session.start()` + a source coroutine |
| (how the GUI tap feeds the session) | `SlowBlockAccumulator` — the same class `run_slow_source` uses, so the GUI and this notebook ingest identically |
| **⟳ Re-estimate Noise** | `session.re_estimate_noise()` |
| Live pulse / histogram / template plots | `on_pulse`, `on_histograms`, `on_templates` callbacks |
| counts / Hz selector | `counts_to_hz_scale(df_calibration)` |
| Output `.h5` + Session Browser review | `hdf5_path=` + `PulseHDF5Reader` |

For an unattended acquisition run with no notebook at all, see
`pulse_capture_flow.py` in this folder:

    python pulse_capture_flow.py MOCK      # simulated CRS
    python pulse_capture_flow.py 0042      # real board

```python
# Only tear down a streamer this notebook started. If section 1 attached to
# Periscope's CRS, stopping it here would kill Periscope's own live plots.
if IS_MOCK:
    await crs.stop_udp_streaming()
    print("simulated streamer stopped")
else:
    print("left the streamer running — it is not ours to stop")
```
