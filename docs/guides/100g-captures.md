# 100G captures

The channel stream (fastrx, 100G) carries every channel of a module at the
PFB rate; the 1G paths carry the decimated slow stream (the parser, and
pulse capture) and at most four PFB channels. All three are timestamped by the
board's IRIG clock, so a pulse recorded on the slow stream can be looked up
in a fastrx recording and drawn over it. `rfmux record` takes the three
together for the same stretch, into one session folder, merging the 100G 
and 1G records for pulses, or generating a time-base-browsable record of
aligned fast and slow data. This guide is how to use it: the dialog, the 
command line, what the run does, and the viewer.

Needs: Linux with the fastrx extension built (clang, libxdp, libbpf and
liburing at install time), a 100G NIC on the channel-stream network, and
pygetdata for the parser dirfile (`uv pip install -e .[dirfile]`, with
libgetdata on the system).

## 1. Before the first run

**The receiver daemon.** fastrxd owns the 100G NIC and hands packets to
every client. Start it once and leave it running. Started without
privilege it prints the command to use:

```bash
rfmux fastrxd
sudo <path it prints>/fastrxd -i <100G interface>
```

**The channel streamer.** Each module's 100G streamer is enabled on the CRS
separately, and switching on the slow streamer does not switch it on. The
recorder can do it for you (the check box or `--channel-streamer` below);
by hand it is one call per module:

```python
await crs.set_channel_streamer(channels=128, module=<module>, sample_trunc="LOW")
```

`channels` is a multiple of 16. `sample_trunc` picks 16 of the 24 bits of
each sample, which is in ADC counts: `"LOW"` keeps bits 15:0 and is exact
while the signal stays within ±32767 counts, `"MID"` keeps bits 19:4
(counts/16) and `"HIGH"` bits 23:8 (counts/256), each dropping the finer
bits. With DC levels of a few thousand counts and noise of a few hundred,
HIGH leaves about one bit of noise; LOW or MID keeps it. The viewer scales
every truncation back to counts. `rfmux fastrx hud --module <module>` shows
whether packets are flowing. For typical MKID applications the LOW truncation
is ideal, since most of the dynamic range is consumed by having many channels
that are each relatively small individual signals.

**A session with a bias export.** The recorder needs to know about which channels
are biased, and their tuning information in order to issue a Pulse Capture
session in df units. It does this by reading from the session's newest Bias KIDs 
export for each module, and stores each channel's tuning (the sweep at the chosen 
amplitude, the fit, the bias point, the df calibration) with its pulses.

Tune in Periscope with a session active, or name channel ranges by hand to bypass this.
If bypassed the df and tuning fields will be missing, and Pulse Capture triggering will
be on the I and Q bases.

## 2. Recording with the dialog

Run `rfmux record` with no options and a dialog asks for everything. Its
choices are remembered between runs, and its Record button stays off
while the status line at the bottom lists what is missing.

The **Run** tab:

- **CRS serial** and, when the board is not `rfmux<NNNN>.local`, its
  **Hostname**.
- **Modules**: `1`, or `2,3` for one RF line fed by several modules.
- **Session**: an existing session folder, the newest under the default
  path filled in, or a new `session_YYYYMMDD_HHMMSS` folder under a
  directory of your choice.
- **Channels**: the biased channels of each module's newest bias export
  in that session, with the export named and its channel count shown, or
  ranges you type: `1-88` on every module, or `2:1-114,3:1-96` naming the
  modules itself.
- **Duration**, in seconds, after the capture's noise training.
- **Products**: the pulse capture of the slow stream, the parser dirfile
  with its 1G interface (found from the board's address by default), and
  the fastrx recording with its 100G interface. The interfaces are listed
  with their negotiated rates, those under 100 Gb/s for the parser and the
  100 Gb/s ones for fastrx, a lone 100 Gb/s interface filled in. The
  dialog checks for a running fastrxd on that interface and, when there
  is none, shows the command that starts it, with a button to copy it and
  one to check again. Below that it reports the disk free in the session
  folder against what the recording needs.
- **Turn the channel streamer on for these modules**, with the sample
  bits beside it. Off, the recorder only reads the board, and a module
  whose channel stream is off is refused before anything is written. On,
  it calls `set_channel_streamer` for every module recorded, channels 1 to
  the highest rounded up to a multiple of 16, and lets the stream flow for
  a second before checking it.
- **Merge the recording into the pulse file after the run.**
- **After the run**: Periscope in review mode on the pulse file, the
  overlay viewer on the channel with the most pulses, or nothing.

The **Pulse capture** tab holds the capture settings: threshold and end
sigma, the pulse length limits, the noise-training span and the trigger
basis. They validate as you type.

## 3. Recording from the command line

Every choice of the dialog is an option; `rfmux record --help` lists them
with their defaults. `--serial` and `--duration` are required. One module,
into an existing session, channels from its newest bias export:

```bash
rfmux record --serial <NNNN> --module <module> --duration 20 \
    --session ~/data/session_20260909_153654
```

One RF line over modules 2 and 3, the channel streamer turned on for both
first, into a new session folder under `~/data`:

```bash
rfmux record --serial <NNNN> --module 2 --module 3 --duration 20 \
    --session-dir ~/data --channel-streamer --sample-trunc LOW
```

Channel ranges instead of the bias export: `--channels 1-88` applies the
same ranges to every module, `--channels 2:1-114,3:1-96` names the modules
itself; `--bias <file>` names a bias export instead of the newest.
`--no-capture`, `--no-parser` and `--no-fastrx` leave a product out.
`--parser-interface` names the 1G interface when the board's address does
not find it; `--fastrx-interface` names the 100G NIC when several fastrxd
run. The capture settings are `--threshold-sigma`, `--end-sigma`,
`--min-pulse-ms`, `--max-pulse-ms`, `--noise-train-ms` and
`--trigger-basis`.

## 4. What a run does

The parser is brought up first (its process takes a few seconds to
import), then the capture starts. The capture spends its noise-training
span (5 s by default) before it detects anything, so the fastrx writer
starts when that span ends and runs for the duration: the capture and the
recording cover the same stretch, and the dirfile that stretch plus the
training span. Without the capture the recording starts as soon as the
parser is up. A run across modules reads the slow stream of every module
through one source, gives the parser one `-c MODULE:RANGE` per module,
and records every module the channel stream carries; a module that is not
streaming stops the run before the recording window opens.

The products, sharing one time stamp, named `module2` for one module and
`modules2+3` for a run across modules:

- `pulse_module<M>_HHMMSS.h5`, the slow-stream pulse capture, with each
  channel's tuning under its `tuning` group. A file across modules keys
  its channels by (module, channel).
- `parser_module<M>_HHMMSS.dirfile/serial_<NNNN>`, the parser's dirfile of
  the same channels, and a `.log` with its drop statistics.
- `fastrx_module<M>_HHMMSS.fastrx`, the channel-stream recording of
  channels 1 to the highest of them.

All three are listed in the session's metadata, so Periscope's session
browser shows them.

After the run the command lists the channels that triggered with their
pulse counts, merges the recording into the pulse file as its fast stream
(`--no-merge-fastrx` leaves the file slow-only; `rfmux fastrx merge
<pulse.h5> <run.fastrx>` does it later) and opens Periscope in review
mode on the pulse file, in its session folder. The command exits 1 after
a run that warned: a capture that ended before its noise training was
done, no channel-stream packets, a disk too small for the recording, a
parser that wrote nothing, or a recording that could not be merged.

## 5. Reviewing in Periscope

The merged file is a both-mode file: the Pulse Capture panel shows the
recording's samples under every pulse, and names the module beside each
channel for a run across modules. Any capture file opens that way with
`periscope --review <pulse.h5>`, offline, or by double-clicking it in the
session browser.

The pulse list carries a **Tuning** item, one per module, for a capture
taken after Bias KIDs. Double-click it to browse the sweeps the channels
were tuned with in a multisweep window, one sweep per resonator with its
probe amplitude in dBm, the detector digest a double-click away. That
window reads nothing from the board and changes nothing on it.

## 6. The viewer

A capture file's pulses over the recording, with the parser's samples on
the same axis. Every trace is in the capture's stored units: volts, or
hertz along the frequency direction for a channel captured with its df
calibration, which rotates and scales the recording and the parser trace
the same way.

```bash
rfmux fastrx overlay pulse_module2_143012.h5 /data/run.fastrx \
    --channel 5 --pulse 1 --dirfile ~/data/run.dirfile/serial_<NNNN> --pad 5
```

Press n and p to step through pulses. `--channel 2:5` names module 2's
channel 5 of a capture across modules, whose recording holds every
module's records; the viewer and the merge read the channel's module
alone. `--pad` shows that many milliseconds of recording either side of
the pulse window; `--save fig.png` writes the figure instead of opening a
window. The title reports gaps or dropouts in the recording window and,
for a both-mode capture, the lag at which the recording best matches the
fast-stream pulse.

A window of the parser dirfile over the recording, in counts, without a
capture file:

```bash
rfmux fastrx overlay-dirfile ~/data/run.dirfile/serial_<NNNN> /data/run.fastrx \
    --module 2 --channel 5 --t0 51612.250 --t1 51612.275
```

`--t0` and `--t1` are seconds of day on the PFB clock; a capture file's
pulse attribute `trigger_time`, or any value of the dirfile's timebase, is
on that axis.

## 7. The pieces by hand

The recorder is `record_streams` in
`rfmux.algorithms.measurement.record_streams`, and each product can be
taken on its own.

The parser, on the interface that receives the board's 1G traffic, writes
the module and channels of interest to a dirfile; it stops on Ctrl-C:

```bash
rfmux parser -i <1G interface> -d ~/data/run.dirfile -c <module>:<channels> --drop-stats
```

It writes one subdirfile per board, `~/data/run.dirfile/serial_<NNNN>`,
which is the path the viewer takes. Its `m<MM>_timebase` is the packet
stamp in seconds of day on the PFB clock, and `m<MM>_dec_stage` records
the decimation stage per frame.

A slow-stream pulse capture comes from Periscope or `crs.trigger_capture`
(see the pulse capture guide); its file lands in the session export folder
as `pulse_module<M>_<HHMMSS>.h5`.

A fastrx recording keeps channels 1 to `channels` of every packet, 2.44 M
records per second: 128 channels is about 1.5 GB per second on disk, all
1024 about 10 GB. Leaving the `with` block finalizes the file; `overruns`
counts records the disk was too slow to take.

```python
from rfmux import fastrx

with fastrx.PacketWriter("/data/run.fastrx", channels=128) as w:
    w.wait(timeout=20)                       # seconds to record
    print("packets", w.packets, "overruns", w.overruns, "dropouts", w.dropouts)
```

The viewer's pieces are `Recording`, `pulse_overlay` and `dirfile_window`
in `rfmux.pulse_capture.overlay`:

```python
from rfmux.pulse_capture.hdf5 import PulseHDF5Reader
from rfmux.pulse_capture.overlay import Recording, pulse_overlay

rec = Recording("/data/run.fastrx")
with PulseHDF5Reader("pulse_module2_143012.h5") as r:
    ov = pulse_overlay(r, rec, channel=5, pulse_idx=1,
                       dirfile="~/data/run.dirfile/serial_0156")
# ov.pulse, ov.dirfile, ov.fastrx: dicts of times, I, Q in ov.units
```
