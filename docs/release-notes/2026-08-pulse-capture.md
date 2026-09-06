# Pulse Capture

rfmux can trigger on transient events in a detector timestream as the board
streams, record each pulse to HDF5 with its summary statistics, and show
them as they arrive. It runs in Periscope, from a script, and against the
simulated board.

This note shows what the feature does and how to drive it from Periscope.
For the headless version, with every step as a runnable cell, open the
[Pulse Capture notebook](../../rfmux/reference-notebooks/Demos/pulse_capture.md).

## Pulses, not timestreams

![Anatomy of one capture window](images/capture-window-anatomy.png)

A capture estimates the noise on each channel first, then triggers when a
sample leaves `threshold_sigma` and rises faster than the baseline drifts.
It closes when the signal is back inside `end_sigma`. The saved window
starts before the trigger, so the rising edge is kept, and ends a short
margin after the signal drops back below threshold. A capture still open at
1.2 times `max_pulse_ms` is closed, and flagged `truncated` if the signal had
not yet come back below threshold. Two pulses that overlap are split when
the deviation rises sharply again.

Each pulse carries its signal-to-noise, peak amplitude, duration, derived
decay constant and trigger time in UTC, decoded from the packet timestamps.
The file is written as the capture runs, so an interrupted run keeps what it
saw.

## Capture in Periscope

![Pulse Capture panel reviewing a capture file](images/pulse-capture-panel-review.png)

1. Start Periscope on a board or the simulator:

   ```bash
   periscope 0042        # a board, by serial
   periscope MOCK        # the simulated board
   ```

2. Press **Pulse Capture** in the main toolbar. The panel docks in the
   window.
3. Set **Mode** (slow, fast or both), **Channels** (`1,2`, `2-19`, or `all`
   for every biased channel) and **Module**.
4. Set **Thresh σ** and **End σ**. **Settings…** holds the rest: the
   longest pulse you expect, the margin saved around each pulse, the
   trigger basis.
5. For fast or both mode, press **Streamer…** and configure the PFB
   streamer for the channels you will capture. The capture reads what the
   board streams and never changes it; if the streamed channels do not match
   the capture, the panel says so and stops.
6. Choose the output file with **…**, then press **▶ Start**.

The left pane lists every pulse with its length, signal-to-noise and trigger
time. **Pulse View** stacks the two axes against a common time axis with
marks for the trigger, the return below threshold and the end confirmation.
Left and Right move through the pulses, Home and End jump to the first and
last, Space cycles the tabs, and Ctrl+E exports the list. **⟳ Re-estimate
Noise** retrains the baseline without stopping.

**Units** switches the pulse view, histograms and templates between counts,
volts and df in hertz. Hertz needs a calibrated channel (below). In both
mode the pair view shows each stream in the units it was stored in.

## Histograms and templates

![Histogram tab](images/pulse-capture-panel-histograms.png)

The **Histograms** tab accumulates signal-to-noise, peak amplitude, duration
and decay constant over every pulse, live, with ranges that expand as pulses
arrive. The **Template** tab stacks the pulses trigger-aligned and shows the
mean with its residual scatter.

The **Plot** field on both tabs takes the same language as **Channels**:
`1,2,4` draws those channels, `1-5` combines five channels into one
histogram or template, and `*` combines them all. Combined templates are
weighted by each channel's pulse count.

## Review a finished capture

Captures written into the session folder appear in the **Session Browser**
under the Pulse Capture filter. Double-click one to open it in review mode:
the capture parameters load into locked controls, and the pulse list,
histograms and templates come from the file. A capture still running opens
its live panel instead.

## Fast and dual-stream captures

The slow stream runs at 596 Hz at the default decimation stage and 38 kHz
at stage 0. For faster pulses, **Mode: fast** captures from the polyphase
filterbank stream at 2.44 MHz per channel, one, two or four channels at
once. **Mode: both** runs a slow and a fast capture together and matches
pulses between them into pairs: the slow stream gives a long clean baseline,
the fast stream resolves the rise. The pair view draws the fast trace over
the slow one with the pair's trigger offset.

The **Streamer…** dialog shows the data rate each configuration puts on the
link against the 1 GbE budget and refuses one that does not fit. A fast
capture is a lot of data, so the status line turns amber and then red as
the fast stream falls behind, with the cause and the remedy in its tooltip.
Raise `net.core.rmem_max` before a long fast capture; the
[Networking Guide](../guides/networking.md) has the numbers.

## Trigger in the frequency basis

A pulse moves the resonance frequency, so in the IQ plane it lies along one
direction set by the bias point. With a df calibration from `bias_kids`,
the capture rotates each channel onto that direction, triggers on it, and
stores the samples in hertz. Without one, a channel triggers and stores in
volts on the I and Q axes. In Periscope the calibration comes from the Bias
KIDs step of the tuning flow; in mock mode Periscope measures one for every
simulated detector at startup.

The screenshots in this note are frequency-basis captures: the axes are df
and dissipation, and the amplitudes are in hertz.

## From a script

One call captures and writes the file:

```python
from rfmux.pulse_capture import PulseCaptureConfig

result = await crs.trigger_capture(
    channel=[1, 2],
    module=1,
    streamer_mode="slow",
    time_run=15.0,
    config=PulseCaptureConfig(threshold_sigma=5.0, end_sigma=1.0),
    hdf5_path="capture.h5",
)
```

Everything else, from configuring the streamers and choosing thresholds to
live sessions, fast and dual captures, reading a file back and calibrated
amplitudes, is in the
[Pulse Capture notebook](../../rfmux/reference-notebooks/Demos/pulse_capture.md),
section by section, against a board or the simulator. Open it from the
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
- [What changed on the branch](2026-09-pulse-capture-branch.md): what
  else changed and what to change when upgrading.
- [Networking Guide](../guides/networking.md): UDP buffer sizing for long
  captures.
