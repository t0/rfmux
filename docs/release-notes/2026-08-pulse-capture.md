# Pulse Capture

This release adds pulse capture to rfmux: detecting transient events in a
detector timestream, recording them to HDF5, and reviewing them as they
arrive. It works headlessly from a script, interactively from Periscope, and
offline against the mock board.

Previously the only way to look at a pulse was to capture a raw timestream and
find it yourself afterwards. Now the board streams, rfmux triggers on the
stream, and you get a file of individual pulses with their summary statistics
already computed.

## Capture pulses in one call

`trigger_capture` is a macro on the CRS object, so it is available on any
board you have resolved:

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

for ch in result.channels:
    print(ch, len(result.pulses[ch]), "pulses")
```

The call estimates the noise on each channel first, then triggers at
`threshold_sigma` above that baseline. Nothing needs to be known in advance
about the pulse height.

`result` carries the pulses, their summaries and the noise statistics. The
same content is written to `hdf5_path` as the capture runs, so an interrupted
run keeps whatever it had already seen.

## How a pulse is detected

![Anatomy of one capture window](images/capture-window-anatomy.png)

A capture opens when either quadrature crosses `threshold_sigma`. It closes
when both quadratures come back inside `end_sigma` and stay there. The window
that gets saved starts before the trigger, so the rising edge is not clipped,
and by default it runs to that confirmation, so the decay tail is kept. Set
`save_to_end_confirmed=False` to stop it a short margin after the signal drops
back below threshold instead.

Two settings bound the result. `max_pulse_ms` sets the longest pulse you
expect; a window still open at 1.2 times that is closed anyway and flagged
truncated. `min_pulse_ms` discards anything too short to be real.

## See them live in Periscope

The Pulse Capture panel runs the same engine and draws each pulse as it is
detected. Open Periscope, add the panel, set the channels and thresholds in
the toolbar, and press Start.

![Pulse Capture panel reviewing a capture file](images/pulse-capture-panel-review.png)

The left pane lists every pulse with its length and signal-to-noise. The plot
stacks I and Q against a common time axis. Markers show the trigger sample,
where the signal fell back below threshold, and where the end was confirmed.

The panel also opens a finished file. Point it at an `.h5` and it enters
review mode, which is what the screenshot above shows. Captures written into
a session folder appear in the Session Browser and open from there.

### Histograms while you capture

![Histogram tab](images/pulse-capture-panel-histograms.png)

Signal-to-noise, peak amplitude, duration and derived decay constant are
accumulated over every pulse and updated live. Ranges expand as new pulses
arrive, so there is nothing to configure up front. Amplitudes can be shown in
raw counts or converted to Hz of frequency shift.

## Read the file back

```python
from rfmux.pulse_capture import PulseHDF5Reader

reader = PulseHDF5Reader("capture.h5")

for ch in reader.channels:
    print(ch, reader.pulse_count(ch), "pulses,", reader.noise_stats(ch))

    for meta in reader.iter_pulse_metadata(ch):
        print(meta["pulse_idx"], meta["snr"], meta["duration_s"])
```

`iter_pulse_metadata` reads only the summary fields, so it stays fast on a
file with tens of thousands of pulses. `get_pulse` loads one waveform when you
want the samples themselves.

## Capture the fast stream

The slow stream tops out at 38 kHz. For pulses faster than that, capture from
the polyphase filterbank instead:

```python
result = await crs.trigger_capture(
    channel=[1], module=1, streamer_mode="fast",
    time_run=0.25, hdf5_path="fast.h5",
)
```

The PFB stream runs at 2.44 MHz per channel, roughly sixty times the fastest
slow-stream rate. Four channels at once is the firmware's limit, not a
software one. It is also a lot of data, so a quarter of a second is usually
enough.

## Both streams at once

`streamer_mode="both"` runs a slow and a fast capture together and matches
pulses between them:

```python
result = await crs.trigger_capture(
    channel=[1], module=1, streamer_mode="both",
    time_run=2.0, hdf5_path="dual.h5",
)

print(len(result.pairs), "matched pairs")
```

Each stream is also reachable on its own as `result.slow` and `result.fast`.
The matching is the reason to run both. The slow stream gives a long clean
baseline, and the fast stream resolves the rise. The pairs tie the two views
of one event together.

## Check the link budget before you configure

Streaming two PFB channels alongside four modules of slow data does not fit in
1 GbE. `streamer_config` works out whether a configuration fits:

```python
from rfmux.algorithms.measurement.streamer_config import StreamerConfig, validate

cfg = StreamerConfig(dec_stage=0, pfb_channels=[1, 2], pfb_module=1)

for severity, message in validate(cfg):
    print(severity, message)
```

Periscope exposes the same check through the Streamer Configuration dialog,
reachable from the Pulse Capture toolbar.

## Try it without a board

Everything above works against the mock. The mock injects pulses of a
configurable shape and rate, so you can develop against it:

```python
from rfmux.mock.helpers import create_mock_crs

crs = await create_mock_crs(module=1, config={
    "num_resonances": 2,
    "resonator_random_seed": 42,
    "auto_bias_kids": True,
    "bias_amplitude": 0.001,
    "pulse_mode": "periodic",
    "pulse_period": 0.25,
    "pulse_tau_rise": 5e-3,
    "pulse_tau_decay": 25e-3,
    "pulse_amplitude": 2.0,
})
```

Match the decay constant to the rate you are sampling at. The default
decimation gives 596 Hz, so a sample every 1.68 ms, and the 25 ms decay above
is about fifteen samples long. A decay shorter than the sample period lands
inside one sample and is detected but not resolved, which looks like a single
spike rather than a pulse.

`periscope --mock` gives the same simulation behind the GUI. The screenshots
in this document were taken from a mock capture.

The mock generates both streams in one process, so fast and dual captures run
slower than real time: two PFB channels is 4.9 M complex samples a second to
synthesise. Slow-stream captures keep up. This affects only the simulation.

The mock also gained TLS noise this release. Resonators now carry 1/f
frequency wander with a configurable amplitude and corner, which is what a
real detector looks like at low frequency. It is set from the same config
dictionary and from the Mock Configuration dialog.

## Behaviour changes

**The PFB stream is 2.44 MHz, not 1.22 MHz.** An earlier constant in this
branch described the PFB rate as 1.22 MHz. That number is the Nyquist
frequency, and also the bin spacing, but not the sample rate. Anything that
divided a PFB sample count by it got twice the true elapsed time, and the
link budget under-counted PFB load by half. Both constants now exist and say
what they are:

```python
from rfmux.core.transferfunctions import PFB_SAMPLING_FREQ, PFB_NYQUIST_FREQ

PFB_SAMPLING_FREQ   # 2441406.25 Hz, the per-channel complex sample rate
PFB_NYQUIST_FREQ    # 1220703.125 Hz, the single-sided bandwidth
```

If you carried the old number into your own code, durations computed from PFB
sample counts were wrong by a factor of two.

**`rfmux.streamer.PFB_SAMPLE_RATE` is gone.** It held the same incorrect
value and was never used inside the package. Use `PFB_SAMPLING_FREQ` from
`rfmux.core.transferfunctions`.

## Where to go next

- [Pulse Capture notebook](../../rfmux/reference-notebooks/Demos/pulse_capture.md)
  is the full explanation, with runnable cells. Open it from the Jupyter panel
  Periscope launches, or right-click it in JupyterLab and choose Open With →
  Notebook.
- [`pulse_capture_flow.py`](../../rfmux/reference-notebooks/Demos/pulse_capture_flow.py)
  is the same sequence as a plain script. Read it as a reference for your own
  code, or run it against `MOCK` to check a change end to end.
- [Networking Guide](../guides/networking.md) covers the UDP buffer sizing
  that long captures need.
