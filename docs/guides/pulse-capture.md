# Pulse Capture

rfmux can trigger on transient events in a detector timestream as the board
streams, record each pulse to HDF5 with its summary statistics, and show
them as they arrive. It runs in Periscope, from a script, and against the
simulated board in mock-mode.

This guide shows what the feature does and how to use it from the Periscope GUI.
For the headless version, with every step as a runnable cell, open the
[Pulse Capture notebook](../../rfmux/reference-notebooks/Demos/pulse_capture.md).

## Pulses, not timestreams

![Anatomy of one capture window](images/capture-window-anatomy.png)

A capture estimates the noise on each channel first, then triggers when a
sample leaves `threshold_sigma` faster than the baseline 1/f drift.
It closes when both axes are back inside `end_sigma` of the baseline.
The saved window starts `pre_pulse_ms` before the trigger, so the rising edge
is kept, and ends `post_pulse_ms` after the pulse settled; the rest of the
confirmation that follows only verifies that it stayed there. A capture still
open at 1.2 times `max_pulse_ms` plus `post_pulse_ms` is closed there and
flagged `truncated`. Two pulses that
overlap are split when the signal rises sharply again on the tail of the
first, and both fragments are flagged `pileup`. The figure above is pulled
from the output from a mock-mode run. All of the annotated metadata for the pulse
also exist within the saved HDF5 output.

Each pulse also carries its signal-to-noise, peak amplitudes, duration, derived
decay constant and trigger time in UTC, decoded from the packet timestamps.
The file is written as the capture runs, so it can be run indefinitely without
RAM constraints, and will be preserved if the capture is interrupted.

## Capture in Periscope

![Pulse Capture panel reviewing a capture file](images/pulse-capture-panel-review.png)

1. Start Periscope on a board or the simulator:

   ```bash
   periscope 0042        # a board, by serial
   periscope MOCK        # the simulated board
   ```

2. Press **Pulse Capture** in the main toolbar. The panel docks in the
   window.
3. Set **Mode** (slow, fast, or both), **Channels** (`1,2`, `2-19`, or `all`
   for every biased channel) and **Module**.
4. Set **Thresh σ** and **End σ**.
5. **Settings** holds the rest of the individual settings, each channel's
   among them; see
   [Configuring the pulse capture engine](#configuring-the-pulse-capture-engine).
   **Load Config…** takes them all from a saved trigger config instead.
6. To configure the data-stream used for the capture, press **Streamer**,
   which provides access to the PFB and decimated streamer settings; see
   [Selecting the stream](#selecting-the-stream).
7. Choose the output file with **…**, then press **▶ Start** to run.

The left pane lists every pulse with its length, signal-to-noise and trigger
time. **Pulse View** shows whatever displayed unit, (I,Q) or (df,diss), against
a common time axis with vertical annotations for each of the pulse detection
parameters (trigger; drop below threshold; settled point). 
Left and Right move through the pulses, Home and End jump to the first and
last, Space cycles the tabs, and Ctrl+E exports the trigger config. **⟳ Re-estimate
Noise** retrains the baseline without stopping.

**IQ Plane** draws the same pulse as points in the plane, grey before the
trigger and one hue darkening with time after it. Under it lies the sweep
the channel was tuned with and, always on top, its bias point as a star.
The pulse's baseline sits on the bias point when the tuning describes the
channel. In the df view the whole picture is rotated so a frequency shift
runs along the horizontal df axis. In both mode a selector picks which
stream's record of the pair is drawn, slow or fast.

**Units** switches the pulse view, IQ plane, histograms and templates
between counts, volts and df in hertz. df units require a df calibration
(below).

## Selecting the stream

![Streamer Configuration dialog](images/streamer-configuration-dialog.png)

**Mode** on the panel picks which data-stream the pulse detection uses:
`slow` for the ordinary decimated readout stream, `fast` for the raw PFB stream,
and `both` for a dual-stream capture (see
[Fast and dual-stream captures](#fast-and-dual-stream-captures)).
**Streamer** opens the Streamer Configuration dialog, the same one that is
accessible via the main window. **OK** applies it to the board at once.

- **Current stream** is what the board streams now, read when the dialog
  opens.
- **Decimation stage** sets the slow sample rate (table below). Aim for ten
  or more samples across one decay constant of your pulses.
- **Packet format**: short packets carry 128 channels per module, long
  packets 1024. Below stage 3 only short packets fit the 1GbE link, and the box
  is locked on.
- **Modules** lists the modules the slow stream carries, as `1,2` or `1-4`.
- **Enable fast (PFB) streamer** turns the 2.44 MHz stream on for up to
  four **PFB channels** of one **PFB module**. Unchecked, **OK** turns it
  off. While it runs, `get_pfb_samples` is unavailable.

| Stage | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Slow sample rate | 38.1 kHz | 19.1 kHz | 9.5 kHz | 4.8 kHz | 2.4 kHz | 1.2 kHz | 596 Hz |

The rows below the settings report what they give: the slow sample rate and
its Nyquist frequency, the channels per module, and the link budget in Mbps
against the 1 GbE port. The banner lists anything wrong. An error disables
**OK**. Examples of invalid configurations are: long packets below stage 3, 
more than four PFB channels, or a total bandwidth over 1000 Mbps.

In dual capture mode it is permitted for only a subset of channels to have 
the fast PFB data, but if none of the captured channels are being streamed,
the capture stops.

## Configuring the pulse capture engine

![Pulse Capture Settings dialog with the Advanced group open](images/pulse-capture-settings-dialog.png)

**Thresh σ** (5.0), **End σ** (1.5) and **Pileup** (on) sit on the panel's
toolbar. The **Settings** dialog includes:

- **Stream** is the stream and sample rate the capture will read. The
  derived values below are computed for it.
- **Threshold σ** is how significant an event must be. Both trigger tests
  use it: a sample must leave the baseline by this many σ within a narrow window.
  The second test is a difference of raw samples, so baseline drift cannot fake it.
- **Max pulse (ms)** (50) is the longest pulse you expect. It sizes the
  pulse-scale quantities: the ring buffer at 1.5 times it, the hard stop at
  1.2, and the edge lookback at a tenth. Estimate it generously. A pulse
  that outlasts the buffer loses its rising edge.
- **Pre-pulse time (ms)** (5) and **Post-pulse time (ms)** (5) are how much
  is saved before the trigger and after the pulse settled. The ring buffer
  grows by both and the hard stop by the post-pulse time. The capture is
  released once the post-pulse time has arrived, so the channel cannot
  trigger again inside it, and a pulse arriving there is a pileup.
- **Coincidence window (ms)** (off) groups pulses into events: every pulse,
  on any channel, that triggers within this of an event's first trigger.
  Measuring from the first trigger bounds an event at the window, so a
  steady rate of unrelated pulses cannot chain into one. In both mode a
  channel's share of an event is its pair, whichever of the two streams
  triggered.
- **Save every channel with each event** (off) also saves, with each event,
  the same span of every channel that did not trigger, from both streams
  in both mode. Only a capture can
  do this, since those samples are gone once the ring buffer moves on. With
  the coincidence window off, each pulse is then an event of its own, unless
  two channels trigger on the same sample. The ring buffer grows by the
  window and two blocks of samples (at least 0.1 s) so the span is still
  there when the event closes. Noise samples grow it the same way.
- **Noise sample every (s)** (off) takes noise samples for the statistics
  of the noise: every channel over one window, at random moments, whether
  or not a pulse is present. The waits between them are normally distributed
  about this many seconds, a quarter of it wide. A sample is as long as a
  typical pulse record: the median of the latest 200 saved. Until five have
  been saved it is the pre-pulse time plus the max pulse plus the post-pulse
  time.
  Each is an event tagged as a noise sample.
- **1/f window (ms)** (5000) is the record the noise σ is fitted from and
  the span of the rolling baseline. It has to be long compared with any
  pulse and with the 1/f knee, so it is seconds whatever the pulse length.
  Below 2 s the banner warns: the baseline is then refreshed so often that
  a capture of many channels falls behind the stream.

**Advanced** opens the rest. The defaults suit most captures.

- **Trigger confirmation (samples)**: consecutive samples that must clear
  the threshold. `auto` picks the fewest that keep accidental triggers
  under one per minute per channel at this rate: 1 at 596 Hz, 2 on the PFB
  stream (based on an assumption of white noise).
- **End σ**: a capture ends once both axes are back inside this band.
  It must sit below **Threshold σ**.
- **End confirmation floor (samples)** (10): the fewest in-band samples
  that confirm the end. The count grows to a tenth of the time above
  threshold for a long pulse, and to the post-pulse time when that is
  longer. It counts down while the signal is out of band, so one noisy
  sample does not restart it.
- **Min pulse (ms)** (0): pulses shorter than this are dropped as
  glitches. 0 turns the filter off.
- **Split piled-up events** (on): a fresh rise on the tail of a pulse
  starts a new one. The rise is judged against the larger of the trained
  noise and the scatter inside the capture, and must hold for as many
  samples as a trigger needs at this rate, so neither noise that grows
  with the pulse nor one stray sample splits it. Both fragments are
  flagged `pileup`. Templates skip them, histograms keep them.
- **Trigger basis**: `df/dissipation (rotated)` triggers in the frequency
  basis on every channel with a df calibration; a channel without one
  triggers on I and Q. `I/Q (quadratures)` triggers on the raw quadratures
  everywhere. A trigger in the df basis is likely to be more sensitive than
  a trigger in an arbitrary (un-rotated) (I,Q) basis.

A text box below dervies the relevant timescales and expectations for the selected
parameters.

### Per-channel settings

The **Settings** dialog lists the capture's channels in a table, one row
each:

- **Trigger** (checked): unchecked records the channel without triggering
  on it. Its samples are saved with every event, over the event's span,
  and with every noise sample. Such a channel turns events on by itself,
  so each pulse on another channel is then an event.
- **Threshold σ** and **End σ**: the channel's own values. Blank takes the
  capture's. A channel's End σ must sit below its threshold.

Each pulse records the `threshold_sigma` and `end_sigma` its channel ran
with.

### Save and load a trigger config

**Export Config** saves the trigger configuration, with the channels, module
and mode, as `trigger_config_<HHMMSS>.h5` in the session folder (without
a session, the output file's folder or your home folder). The file is
a capture file's `metadata` group and nothing else, and it appears in the
Session Browser under the Pulse Capture filter.

**Load Config…** reads a trigger config file, or any capture file, and sets
the panel from it, ready for a new capture. Every capture records the
config it ran with, so an earlier run can be repeated as it was.
Double-clicking a trigger config file in the Session Browser opens a panel
set from it. The `rfmux record` dialog has the same table, **Export
Config…** and **Load Config…** on its Pulse capture tab, and reads and
writes the same files (see the [100G captures guide](100g-captures.md)).

## Histograms and templates

![Histogram tab](images/pulse-capture-panel-histograms.png)

The **Histograms** tab accumulates signal-to-noise, peak amplitude, duration
and decay constant over every pulse, live, with ranges that expand as pulses
arrive. Peak amplitude is one histogram per axis on shared bins, overlaid
in the channel colour, the first filled and the second hatched, with a key
naming the two. A calibrated channel keeps two pairs, frequency and
dissipation in hertz and I and Q in volts, and the units selector picks
the pair that matches the view, so the toggle changes frame here as it
does on the other tabs.

The **Template** tab stacks the pulses trigger-aligned and shows the
mean with its residual scatter. In both mode each tab has its own stream
selector, slow or fast.

The **Plot** field on both tabs takes the same language as **Channels**:
`1,2,4` draws those channels, `1-5` combines five channels into one
histogram or template, and `*` combines them all. Combined templates are
weighted by each channel's pulse count.

## Review a finished capture

Captures written into the session folder appear in the **Session Browser**
under the Pulse Capture filter. Double-click one to open it in review mode:
the capture parameters load into locked controls, and the pulse list,
histograms and templates come from the file. A capture still running opens
its live panel instead. Note -- it can take some time to load a large .hdf5
record and re-generate the template and histograms.

A capture taken after Bias KIDs carries each channel's tuning: the sweep at
the chosen amplitude, the fit, the bias point and the df calibration. The
pulse list shows it as a **Tuning** item, one per module. Double-click it to
browse those sweeps in a multisweep window, one sweep per resonator, with
the detector digest a double-click away as usual. The window reads nothing
from the board and changes nothing on it.

The file also keeps the tail of each channel's noise training record, the
last five max-pulse lengths of the samples the noise statistics were
fitted to, in the channel's stored units (the `noise_training` dataset of
its group, one per stream in both mode). The
pulse list shows it as a **noise training** row under each channel, one
per stream in both mode, live and in review; double-click it to see that
channel's record with its baselines and bands.

## Coincident events

Set a **Coincidence window** in Settings and the capture records events:
every pulse, on any channel, that triggers within the window of an event's
first trigger. **Save every channel with each event** adds the same span
of the channels that did not trigger. A run across modules groups across
them.

**Group by** above the pulse list chooses **Channels** or **Events**. An
event lists its pulses, and a **no trigger** row for every channel saved
with it. In both mode the rows are pairs. A channel counts once whether it
triggered on the slow stream, the fast stream or both. An event can hold a
fast-only pulse on one channel and a slow-only pulse on another.
The pulses are the same either way: they are stored once, under their
channels, and the events index them. A capture that recorded no events
can still be grouped by events, live or in review. The grouping uses the
pulses' trigger times and the window in Settings. The channels that did
not trigger are there only if the capture saved them.

The line under the status names the most active channel. It counts the
pulses that shared an event with another channel, and those that came
alone. Its tooltip ranks the channels. Until the first pulse it shows the
noise each channel trained to. The pulse and event views name the
frequency a channel is biased at, when the capture carries its tuning.

Double-click an event to draw its channels together. Time is measured
from the event's first trigger and each channel is drawn about its own
baseline. Channels that did not trigger are thin dotted traces. In both mode
**Event shows** picks the slow samples (points), the fast ones (lines) or
both. Double-click a pulse
under it for that pulse alone, and a **no trigger** row for that
channel's samples over the event's window: it fills the Pulse View and
the IQ Plane the way a pulse does, against the channel's noise bands,
with both streams in both mode. **Prev** and **Next** step through events,
and **Follow latest** shows the newest event's pulses as it closes,
without the channels that did not trigger. An event closes one hard stop
after its window ends, when no pulse that belongs to it can still be open.
It appears in the list that long after its first trigger.

A **noise sample** is listed among the events with the UTC time it was
taken at and every channel beneath it. A pulse that happened to fall inside
it is listed too; the sample was taken regardless. Double-click it to draw
all its channels, or one of its channels for that channel alone. Follow
latest passes over noise samples. With the coincidence window off and noise
samples on, the only events are the noise samples.

From a script the events are on the result and in the file:

```python
config = PulseCaptureConfig(coincidence_window_ms=2.0, dump_all_channels=True)
result = await crs.trigger_capture(channel=[1, 2, 3], module=1,
                                   config=config, hdf5_path="capture.h5")
for event in result.events:
    print(event["event_idx"], [m["channel"] for m in event["members"]],
          sorted(event["dump"]))

from rfmux.pulse_capture import PulseHDF5Reader, events_of
with PulseHDF5Reader("capture.h5") as r:
    event = r.get_event(1)            # members, window, dumped channels
    pulse = r.get_pulse(event["members"][0]["channel"],
                        event["members"][0]["pulse_idx"])
    regrouped = events_of(r, window_s=0.010)   # any file, any window
```

`event["kind"]` is `"pulses"` or `"noise"`; a noise sample's `dump` holds
every channel, and its `members` the pulses inside its window, often
none. `PulseCaptureConfig(noise_capture_interval_s=30)` asks for them.

`result.events` has the file's shape, with `slow_tod` and `fast_tod` under
each dumped channel in both mode. A slow capture that `rfmux record` merges
with its 100G recording becomes a both-mode file: its events carry over, and
each dumped channel gains the recording over the event's window. The layout
of `events/` is under [File layout](#file-layout).

## Fast and dual-stream captures

The slow stream runs at 596 Hz at the default decimation stage and 38 kHz
at stage 0. For faster pulses, **Mode: fast** captures from the polyphase
filterbank stream at 2.44 MHz per channel, one, two or four channels at
once. **Mode: both** runs a slow and a fast capture together and matches
pulses between them into pairs: the slow stream gives a long clean baseline,
the fast stream resolves the rise. The pair view draws the fast trace over
the slow one with the pair's trigger offset.

A fast capture is a lot of data does require a performant system to keep up
with the full 1 GbE link. The status line turns amber and then
red if the processing falls behind, with the cause and the remedy in its
tooltip. Raise `net.core.rmem_max` before a long fast capture (see
[Networking Guide](networking.md)).

## Trigger in the frequency basis

A pulse moves the resonance frequency, so in the IQ plane it lies along one
direction set by the bias point. With a df calibration in a channel's
tuning row from `bias_kids`, the capture rotates the channel onto that
direction, triggers on it, and stores the samples in hertz. Without one, a
channel triggers and stores in volts on the I and Q axes. The file keeps the
whole row (bias frequency, amplitude, sweep, fit parameters, calibration)
with the channel's pulses. In Periscope the tuning comes from the Bias KIDs
step of the tuning flow; in mock mode Periscope measures a calibration for
every simulated detector at startup.

The screenshots in this guide are frequency-basis captures: the axes are df
and dissipation, and the amplitudes are in hertz.

## File layout

A capture file is plain HDF5. `PulseHDF5Reader` reads it without the paths;
`h5py`, `h5ls` and HDFView read it with them. Times are packet seconds of
day. Numbered groups are zero-padded to six digits.

```
metadata/                      attributes only
channel_<n>/                   one per channel
  noise_training               the tail of the noise training record
  tuning/                      the channel's tuning row, when it has one
  pulse_<k>/
    Amp_I, Amp_Q, Time         the saved window, in the channel's stored units
events/                        when the capture recorded events
  event_<k>/
    members, trigger_times
    pulses/                    a soft link to each member
    dump/channel_<n>/          Amp_I, Amp_Q, Time
histograms/
templates/
```

A run across modules nests each channel as `module_<m>/channel_<n>`, here
and under `dump/`, and a `members` row then starts with the module.

**`metadata`** holds `streamer_mode`, `module`, `channels`, the sample rate
(`sample_rate_slow` or `sample_rate_fast`), `stored_units`, `trigger_basis`,
`volts_per_count`, `slow_time_offset_s`, `capture_start` and `capture_end`.
`trigger_config` is the whole `PulseCaptureConfig` as JSON, per-channel
settings included; `read_trigger_config` loads it.
`time_origin_epoch` and `time_origin_utc` are midnight of the packet clock's
day. A capture configured through `PulseCaptureConfig` records its times in
milliseconds (`pre_pulse_ms`, `post_pulse_ms`, `min_pulse_ms`,
`max_pulse_ms`, `noise_train_ms`) beside the sample counts the engine ran
with (`pre_samples`, `post_samples`, `min_pulse_samples`, `trigger_samples`,
`baseline_window`, `edge_lookback`, `max_capture_samples`), with
`threshold_sigma`, `end_sigma`, `min_end_samples` and `enable_pileup`. With
events on it records `coincidence_window_s` and `dump_all_channels`. With
noise samples on it records `noise_capture_interval_s` and
`noise_capture_window_s`, the length of a noise sample until five pulse
records have been saved. Each sample's own span is on its event.

**A channel group** carries `pulse_count`, `stored_units` and the trained
noise: `noise_mean_I`, `noise_mean_Q`, `noise_std_I`, `noise_std_Q`, and
`noise_jump_std_I` and `noise_jump_std_Q` (0 when not measured). `tuning/`
holds the row's scalars as attributes (`df_calibration` among them), its
arrays as datasets, and its mappings as JSON attributes named in
`json_fields`.

**A pulse group** carries, beside its samples:

- `timestamp` (the first saved sample), `n_samples`, `pileup`, `truncated`;
- the trigger: `trigger_time`, `trigger_index`, `trigger_quad`,
  `trigger_baseline_I/Q`, `trigger_sigma_I/Q`, and `trigger_epoch` and
  `trigger_utc` once the packet clock's day is known;
- the end: `below_threshold_index/time` (the drop below threshold, which
  feeds the decay constant), `settled_index/time` (absent after a hard
  stop), `end_index/time`, `end_baseline_I/Q`, `end_confirm_samples`,
  `end_confirm_target`;
- the summary: `peak_I`, `peak_Q`, `peak_amp`, `snr`, `peak_snr_I`,
  `peak_snr_Q`, `duration_s`, `tau_s`, and the `threshold_sigma` and
  `end_sigma` in force.

**An event group** carries `kind` (`"pulses"` or `"noise"`), `trigger_time`
(the first trigger, or the moment a noise sample was taken), `trigger_epoch`
and `trigger_utc`, and the saved span `window_t0` and `window_t1`. A
`members` row is a channel and a pulse index. `pulses/` links to the same
pulses, so a generic tool opens an event and finds them:

    h5ls --follow-symlinks -r capture.h5/events/event_000001/pulses

`dump/channel_<n>` holds one channel saved without a trigger, a dumped
channel. A noise sample dumps every channel, and its `members` are often
empty. The reader works from `members`; the links are for browsing.

**`histograms`** holds, per metric (`amplitude_i`, `amplitude_q`, `snr`,
`duration_ms`, `tau_ms`, and for calibrated channels `amplitude_raw_i` and
`amplitude_raw_q` in volts), `<metric>_edges`, `<metric>_bins` and one
`<metric>_counts_ch<n>` per channel. **`templates`** holds, per channel,
`time_s_`, `template_I_`, `template_Q_`, `residual_I_`, `residual_Q_` and
`counts_` on one time grid, and the scalars `n_pulses_`, `pre_samples_` and
`post_samples_`. Across modules the suffix is `m<m>ch<n>`.

**A both-mode file**, and a file `rfmux record` merged with its 100G
recording, has `layout = "dual"` and `streamer_mode = "both"` in its
metadata, both sample rates, `fast_channels`, and the sample counts once per
stream: `pre_samples_slow`, `pre_samples_fast` and so on. A merged file has
the `_slow` counts only, since no engine ran on the recording, and its pairs
carry the recording's window (`fast_tod_*`) with no `fast_idx`.

```
slow/channel_<n>/ ...          each as a channel group above
fast/channel_<n>/ ...
matched/channel_<n>/pair_<k>/  slow_idx, fast_idx (-1 for a one-sided pair),
                               time_offset, window_t0, window_t1, and
                               slow_tod_* and fast_tod_* over that window
events/event_<k>/              a member's index is a pair; pulses/ links to
                               the pairs; dump/channel_<n>/slow/ and fast/
histograms/slow/, histograms/fast/
templates/slow/,  templates/fast/
```

## From a script

One call captures and writes the file:

```python
from rfmux.pulse_capture import PulseCaptureConfig

result = await crs.trigger_capture(
    channel=[1, 2],
    module=1,
    streamer_mode="slow",
    time_run=15.0,
    config=PulseCaptureConfig(threshold_sigma=5.0, end_sigma=1.5),
    hdf5_path="capture.h5",
)
```

A trigger config saved from Periscope, or any earlier capture, supplies the
config and the channels:

```python
from rfmux.pulse_capture import read_trigger_config

config, setup = read_trigger_config("trigger_config_142501.h5")
result = await crs.trigger_capture(
    channel=setup["channels"], module=setup["module"],
    streamer_mode=setup["streamer_mode"], time_run=15.0,
    config=config, hdf5_path="capture.h5")
```

Everything else, from configuring the streamers and choosing thresholds to
live sessions, fast and dual captures, reading a file back and calibrated
amplitudes, is in the
[Pulse Capture notebook](../../rfmux/reference-notebooks/Demos/pulse_capture.md),
section by section, against a board or the mock mode simulator. Open it from the
Jupyter panel Periscope launches, or in JupyterLab with Open With → Notebook.
[`pulse_capture_flow.py`](../../rfmux/reference-notebooks/Demos/pulse_capture_flow.py)
beside it runs the same sequence unattended.

## Try it without a board

`periscope MOCK` runs the same GUI on the simulated board, which injects
pulses of a configurable shape and rate. The **Mock Configuration** dialog
sets the pulse period and the rise and decay constants; a pulse change
applies without rebuilding the array. The simulator generates both
streams in one process, so fast and dual captures run slower than real time
there; slow-stream captures keep up.

## Where to go next

- [Pulse Capture notebook](../../rfmux/reference-notebooks/Demos/pulse_capture.md):
  the headless how-to, cell by cell.
- [What changed on the branch](../release-notes/2026-09-pulse-capture-branch.md):
  what else changed and what to change when upgrading.
- [Networking Guide](networking.md): UDP buffer sizing for long
  captures.
