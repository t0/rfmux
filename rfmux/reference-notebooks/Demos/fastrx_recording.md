---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Reading a fastrx recording

A fastrx recording is the 100G channel stream written to disk: every packet,
2.44 million records per second, each record one sample of a module's first
channels with the board's IRIG stamp. This notebook opens one offline, reads
its times and samples, and puts it beside a pulse capture.

Making a recording is the 100G captures guide (`docs/guides/100g-captures.md`):
`rfmux record`, or `fastrx.PacketWriter` by hand. Reading one needs no board
and no daemon.

| Piece | Module |
|---|---|
| `Recording`: the time index and the samples | `rfmux.pulse_capture.overlay` |
| `pulse_overlay`, `merge_fastrx`: a recording beside a capture | `rfmux.pulse_capture.overlay` |
| `RecordingFile`, `write_recording`: the file with numpy alone | `rfmux.pulse_capture.recording_file` |
| `PacketFile`: the file through the fastrx extension | `rfmux.fastrx` |
| The on-disk format | `rfmux/streamer/include/fastrx.h` |

The fastrx extension builds on Linux only. `Recording` reads a file through it
where it is built and through `RecordingFile` elsewhere, so this notebook runs
on any machine that has the file.

## How to use this document

Run the cells in order. Set `RFMUX_FASTRX_FILE` to a recording of your own, or
leave it unset and section 1 writes a small one with known content. This
format saves no outputs, so every number you see comes from your own run. The
shipped copy is read-only: *File → Save Notebook As…* to keep changes.

```python
%matplotlib inline

import os
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from rfmux.core.transferfunctions import (
    PFB_SAMPLING_FREQ, VOLTS_PER_ROC, decimated_stream_delay_s,
    decimation_to_sampling)
from rfmux.pulse_capture import (
    PulseCaptureConfig, PulseCaptureSession, PulseHDF5Reader)
from rfmux.pulse_capture.overlay import (
    COUNTS_PER_LSB, Recording, merge_fastrx, pulse_overlay)
from rfmux.pulse_capture.recording_file import write_recording

# Reference notebooks are provisioned read-only, so files go to a scratch
# directory; override it with RFMUX_DEMO_OUTPUT.
OUTPUT_DIR = Path(os.environ.get(
    "RFMUX_DEMO_OUTPUT", Path(tempfile.gettempdir()) / "rfmux_fastrx_recording"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("files →", OUTPUT_DIR)
```

## 1. A recording to read

With `RFMUX_FASTRX_FILE` set, that file is read and `CHANNEL` and `MODULE`
name what to look at in it.

Without it, this cell writes a recording with `write_recording`: 50 ms of two
modules at the channel stream's rate, eight channels each. Module 1's channel 5
carries one pulse. The recording also has the three things a real one can
have: a few stamps that are not disciplined, a gap in the sequence numbers, and
a stretch where the transmitter dropped a pipeline.

```python
CHANNEL, MODULE = 5, 1
T_EVENT = 43002.0          # seconds of day
AMP, TAU = 4000.0, 0.005   # counts, seconds


def event(t):
    """The pulse, in ADC counts, on the PFB clock."""
    return np.where(t >= T_EVENT, AMP * np.exp(-(t - T_EVENT) / TAU), 0.0)


FASTRX_FILE = os.environ.get("RFMUX_FASTRX_FILE")
SYNTHETIC = FASTRX_FILE is None
if SYNTHETIC:
    rng = np.random.default_rng(1)
    stamps = T_EVENT + np.arange(-0.010, 0.040, 1.0 / PFB_SAMPLING_FREQ)
    modules = (1, 2)
    # The writer keeps every module the stream carries, interleaved: one
    # record per module per stamp.
    seconds = np.repeat(stamps, len(modules))
    module = np.tile(modules, len(stamps))
    iq = rng.normal(0.0, 30.0, (len(seconds), 8, 2))
    iq[module == MODULE, CHANNEL - 1, 0] += event(stamps)

    seq = np.repeat(np.arange(len(stamps)), len(modules))
    seq[seq >= 30_000] += 7                  # seven packets lost, per module
    seconds[2000:2006] = np.nan              # stamps that are not disciplined
    snapshot = np.ones(len(seconds), dtype=np.uint8)
    snapshot[100_000:100_400] = 0            # pipe 1 dropped out
    iq[100_000:100_400] = 0.0                # and the writer zero-filled it

    FASTRX_FILE = write_recording(
        OUTPUT_DIR / "demo.fastrx", seconds, np.round(iq), module=module,
        seq=seq, pipe_snapshot=snapshot, sample_trunc=0)
print(FASTRX_FILE, f"{Path(FASTRX_FILE).stat().st_size / 1e6:.1f} MB")
```

## 2. Open it

`Recording` maps the file. Nothing is read until it is asked for, so opening a
recording of hundreds of gigabytes costs the same as opening this one.

```python
rec = Recording(FASTRX_FILE)
print("read through      ", type(rec.file).__module__, type(rec.file).__name__)
print("records           ", f"{rec.num_packets:,}")
print("channels per record", rec.channels, "(the module's channels 1 to this)")
print("first record's module", rec.module + 1)
print("sample_trunc      ", rec.sample_trunc,
      f"→ {rec.counts_per_lsb:g} ADC counts per unit")
print("first disciplined stamp", rec.t_first, "s of day")
```

`sample_trunc` says which 16 bits of the 24-bit sample the board sent: 0 is
LOW (bits 15:0, exact while the signal is under 32768 counts), 1 is MID (bits
19:4, counts/16) and 2 is HIGH (bits 23:8, counts/256). `Recording` scales by
`COUNTS_PER_LSB`, so its samples are ADC counts whichever was sent, less the
bits the truncation dropped.

```python
print(COUNTS_PER_LSB)
```

## 3. What a record holds

Each record is the packet's wire header followed by the I and Q of channels 1
to `rec.channels` as int16. `rec.file` is the reader underneath, and its
accessors return views of the mapped file, not copies:

```python
headers = rec.file.headers()
print(headers.dtype.names)
first = headers[0]
print({name: first[name].item() for name in
       ("seq", "pipe_snapshot", "sample_trunc", "module", "serial")})
print("stamp fields:", first["ts"].dtype.names)
print("iq:", rec.file.iq().dtype, rec.file.iq().shape, "(records, channels, I and Q)")
```

The wire counts modules from 0. A recording holds every module the channel
stream carried, and each module counts its own packets:

```python
wire_module, per_module = np.unique(headers["module"], return_counts=True)
for m, n in zip(wire_module, per_module):
    print(f"module {m + 1}: {n:,} records")
```

So a query that names a module keeps that module's records. Without one,
`channel()` and `window()` return every module's records in file order, which
is what you want only for a recording of one module.

## 4. Time

A stamp is the board's IRIG time. `seconds()` gives seconds of day, the axis
pulse capture files and parser dirfiles use, and NaN where the stamp was not
disciplined. It touches only the records asked for. A recording that crosses
midnight is unwrapped onto one rising axis.

```python
print(rec.seconds(0, 6))
stamp = rec.file.ts()[0]
print(f"day {stamp['d']} of year 20{stamp['y']:02d}")

t = rec.seconds()
print(f"{np.count_nonzero(np.isnan(t)):,} of {len(t):,} stamps are not disciplined")
print(f"spans {np.nanmax(t) - np.nanmin(t):.4f} s")
```

`index_at` finds the record for a time with a bisect, about twenty probes of
the file however long it is:

```python
i = rec.index_at(rec.t_first + 0.010)
print(i, rec.seconds(i, i + 2))
```

## 5. Samples

`window(t0, t1, channel, module=...)` is the usual way in: one channel over a
span of time, with the record indices it covered and two counts of trouble
inside it.

```python
t0 = (T_EVENT - 0.002) if SYNTHETIC else rec.t_first
w = rec.window(t0, t0 + 0.020, CHANNEL, module=MODULE)
print(f"records {w.start:,} to {w.stop:,}: {len(w.samples):,} samples of "
      f"module {w.module}")
print("sequence gaps:", w.seq_gaps, "  records with the pipe dropped out:", w.dropouts)

volts = w.samples * VOLTS_PER_ROC
ms = (w.times - t0) * 1e3
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
axes[0].plot(ms, volts.real * 1e6, lw=0.6, label="I")
axes[0].plot(ms, volts.imag * 1e6, lw=0.6, label="Q")
axes[0].set_ylabel("µV"); axes[0].legend(loc="upper right")
axes[1].plot(ms, np.abs(volts) * 1e6, lw=0.6, color="k")
axes[1].set_ylabel("|I + jQ| (µV)"); axes[1].set_xlabel("time (ms)")
axes[0].set_title(f"module {MODULE} channel {CHANNEL}, from the recording")
plt.tight_layout(); plt.show()
```

`samples` are complex ADC counts, the unit the 1G streams report, and
`VOLTS_PER_ROC` takes counts to volts.

`seq_gaps` counts places where the sequence number did not follow on: packets
the network or the daemon lost. `dropouts` counts records whose pipeline the
transmitter was not sending, which the writer fills with zeros so that the
record layout stays fixed. Both are zero in a clean window. Over the whole
recording:

```python
whole = rec.window(np.nanmin(t), np.nanmax(t), CHANNEL, module=MODULE)
print("sequence gaps:", whole.seq_gaps, "  dropped-out records:", whole.dropouts)

seq = rec.file.seq()[headers["module"] == MODULE - 1].astype(np.int64)
at = np.flatnonzero(np.diff(seq) != 1)
for k in at[:5]:
    print(f"  after record {k:,} of the module: {seq[k + 1] - seq[k] - 1} packets missing")
```

`channel(channel, start, stop, module=...)` reads by record index instead of
by time, for a stretch already located:

```python
z = rec.channel(CHANNEL, w.start, w.start + 8, module=MODULE)
print(z)
```

## 6. A spectrum

The channel stream is unfiltered beyond the PFB, so a quiet stretch gives the
noise to 1.22 MHz:

```python
quiet = rec.window(rec.t_first, rec.t_first + 0.008, CHANNEL, module=MODULE)
f, psd = welch(quiet.samples.real * VOLTS_PER_ROC, fs=PFB_SAMPLING_FREQ,
               nperseg=4096)
plt.figure(figsize=(9, 3.5))
plt.loglog(f[1:], np.sqrt(psd[1:]) * 1e9)
plt.xlabel("frequency (Hz)"); plt.ylabel("I noise (nV/√Hz)")
plt.title(f"module {MODULE} channel {CHANNEL}, {len(quiet.samples):,} samples")
plt.tight_layout(); plt.show()
```

## 7. Beside a pulse capture

A slow-stream pulse capture records where the pulses are, at the decimated
rate. The recording holds the same moments at 2.44 MHz. All the streams carry
the board's IRIG time, so the two line up by time alone.

With your own recording, set `CAPTURE_FILE` to the capture taken alongside it.
Here a capture of the same event is made by feeding a `PulseCaptureSession` by
hand at 596 Hz. The board stamps the decimated stream late by its CIC group
delay; the session takes that out as it writes, and the file records it as
`slow_time_offset_s`.

```python
CAPTURE_FILE = os.environ.get("RFMUX_CAPTURE_FILE")
if CAPTURE_FILE is None:
    fs = decimation_to_sampling(6)
    late = decimated_stream_delay_s(6)
    CAPTURE_FILE = OUTPUT_DIR / "demo_capture.h5"
    config = PulseCaptureConfig(max_pulse_ms=30.0, noise_train_ms=300.0)
    session = PulseCaptureSession(channels=[CHANNEL], module=MODULE,
                                  sample_rate=fs, hdf5_path=CAPTURE_FILE,
                                  **config.session_kwargs(fs))
    session.start()
    rng = np.random.default_rng(5)
    ts = (T_EVENT - 2.0) + np.arange(int(2.5 * fs)) / fs
    session.feed_block(CHANNEL, event(ts) + rng.normal(0, 5, len(ts)),
                       rng.normal(0, 5, len(ts)), ts + late)
    session.stop()

with PulseHDF5Reader(CAPTURE_FILE) as reader:
    print(f"channel {CHANNEL}: {reader.pulse_count(CHANNEL)} pulse(s), stored in "
          f"{reader.stored_units(CHANNEL)}")
    ov = pulse_overlay(reader, rec, channel=CHANNEL, pulse_idx=1, pad_s=0.005)
```

`pulse_overlay` reads the pulse's window out of the recording, plus `pad_s`
either side, and converts it to the units the capture stored: volts here, or
hertz along the frequency direction for a channel captured with its df
calibration. `rfmux fastrx overlay` draws this figure from the command line.

```python
t_ref = ov.pulse["times"][0]
plt.figure(figsize=(9, 3.8))
plt.plot((ov.fastrx["times"] - t_ref) * 1e3, ov.fastrx["I"], lw=0.5,
         color="0.6", label="recording, 2.44 MHz")
plt.plot((ov.pulse["times"] - t_ref) * 1e3, ov.pulse["I"], "o-", ms=4,
         label=f"capture, {ov.stream} stream")
plt.xlabel("time from the pulse's first saved sample (ms)")
plt.ylabel(f"I ({ov.units})")
plt.title(f"channel {ov.channel} pulse {ov.pulse_idx}: sequence gaps "
          f"{ov.seq_gaps}, dropped-out records {ov.dropouts} in the window")
plt.legend(); plt.tight_layout(); plt.show()
```

## 8. Merging the recording into the capture

`merge_fastrx` adds the recording to a slow capture as its fast stream. Every
slow pulse becomes a pair that names the slow pulse (`slow_idx`) and carries the
recording over its window (`fast_tod`). The result is a both-mode file: `PulseHDF5Reader` and Periscope's review mode
read it like a capture taken in both mode. `rfmux record` does this after a
run and names the file `_100G`.

```python
merged = merge_fastrx(CAPTURE_FILE, FASTRX_FILE,
                      out=OUTPUT_DIR / "demo_capture_100G.h5")
with PulseHDF5Reader(merged) as reader:
    print("both-mode file:", reader.dual)
    pair = reader.get_match(CHANNEL, 1)
    slow = reader.get_pulse(CHANNEL, pair["slow_idx"], "slow")
    print(f"pair 1: slow pulse {pair['slow_idx']} with {len(slow['Time'])} "
          f"samples, and {len(pair['fast_tod']['Time']):,} from the recording "
          f"over {(pair['window'][1] - pair['window'][0]) * 1e3:.1f} ms")
print(f"periscope --review {merged}")
```

## Where to go next

- `docs/guides/100g-captures.md`: taking a recording, the viewer and its
  options, and the parser dirfile beside both.
- `Demos/pulse_capture.md`: pulse capture, and a walk through the capture
  file this notebook merged into.
- `rfmux/streamer/include/fastrx.h`: the on-disk format, field by field.
