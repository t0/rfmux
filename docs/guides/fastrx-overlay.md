# Overlaying 100G and 1G data

The channel stream (fastrx, 100G) carries every channel of a module at the
PFB rate; the 1G paths carry the decimated slow stream (the parser, and
pulse capture) and at most four PFB channels. All three are stamped by the
board's IRIG clock, so a pulse recorded on the slow stream can be looked up
in a fastrx recording and drawn over it. This guide runs one such
comparison end to end: a parser dirfile, a pulse capture and a fastrx
recording of the same stretch, with one command or by hand, then the
viewer.

The board stamps the decimated stream late by its CIC group delay (5 ms at
stage 6, 117 µs at stage 1). Pulse-capture files and parser dirfiles are
written with that delay taken out, so nothing here needs to know the stage.
Files written before that correction are shifted by the viewer from their
slow rate.

Needs: Linux with the fastrx extension built (clang, libxdp, libbpf and
liburing at install time), a 100G NIC on the channel-stream network, and
pygetdata for the parser dirfile (`uv pip install -e .[dirfile]`, with
libgetdata on the system).

## 1. One command

With the streamers configured (the slow stream at its decimation stage,
the channel streamer on for the module) and fastrxd running, one command
records all three products of a module for the same stretch, into one
session folder:

```bash
rfmux record --serial <NNNN> --module <module> --duration 20 \
    --session ~/data/session_20260909_153654
```

Run with no options, `rfmux record` opens a dialog with the same
choices on one page, remembered between runs: board, session folder,
channels, duration, the three products with their interfaces, what to
open afterwards, and the pulse capture settings folded under their
own heading. It checks for a running fastrxd on the 100G interface and,
when there is none, shows the command that starts it and waits.

It reads the board and never configures it. The session is a Periscope
session folder: the channels and their df calibrations come from the
newest bias export in it (`--channels 1-88` and `--bias <file>`
override), and the products are listed in its metadata so the session
browser shows them. Without `--session` a new `session_YYYYMMDD_HHMMSS`
folder is made under `--session-dir`. The products, sharing one time
stamp:

- `pulse_module<M>_HHMMSS.h5`, a slow-stream pulse capture
  (`--threshold-sigma`, `--end-sigma`, `--min-pulse-ms`, `--max-pulse-ms`,
  `--noise-train-ms`, `--trigger-basis` are the capture's settings)
- `parser_module<M>_HHMMSS.dirfile/serial_<NNNN>`, the parser's dirfile
  of the same channels, and a `.log` with its drop statistics
- `fastrx_module<M>_HHMMSS.fastrx`, the channel-stream recording of the
  pipes those channels are on

After the run the command lists the channels that triggered with their
pulse counts, merges the recording into the pulse file as its fast
stream (`--no-merge-fastrx` leaves the file slow-only) and opens
Periscope in review mode on the pulse file, in its session folder
(`--show overlay` opens the viewer of section 3 on the channel with the
most pulses instead; `--show none` opens nothing). The merged file is a
both-mode file: Periscope's pulse capture panel shows the recording's
samples under every pulse, and `rfmux fastrx merge <pulse.h5>
<run.fastrx>` does the same for a run recorded without it. Any capture
file opens that way with `periscope --review <pulse.h5>`.

The parser is brought up first (its process takes a few seconds to
import), then the capture starts. It spends its noise-training span
(5 s by default) before it detects anything, so the fastrx writer starts
when that span ends and runs for `--duration`: the capture and the
recording cover the same stretch, and the dirfile that stretch plus the
training span. `--no-capture`, `--no-parser` and `--no-fastrx` leave a
product out; without the capture the recording starts as soon as the
parser is up. `--parser-interface` names the 1G
interface when the board's address does not find it; `--fastrx-interface`
names the 100G NIC when several fastrxd run. The command exits 1 after a
run with a warning: no channel-stream packets (the channel streamer is
off), a disk too small for the recording, or a parser that wrote nothing.

## 2. By hand

### The 1G side

Configure the streamer as usual (decimation stage, short or long packets)
and start a slow-stream pulse capture in Periscope or with
`crs.trigger_capture`. Its file lands in the session export folder as
`pulse_module<M>_<HHMMSS>.h5`.

Start the parser on the interface that receives the board's 1G traffic,
writing the module and channels of interest to a dirfile:

```bash
rfmux parser -i <1G interface> -d ~/data/run.dirfile -c <module>:<channels> --drop-stats
```

It writes one subdirfile per board, `~/data/run.dirfile/serial_<NNNN>`,
which is the path the viewer takes. Its `m<MM>_timebase` is the packet
stamp in seconds of day on the PFB clock, and `m<MM>_dec_stage` records the
decimation stage per frame.

### The 100G side

Run the receiver daemon on the 100G NIC and leave it running. Started
without privilege it prints the command to use:

```bash
rfmux fastrxd
sudo <path it prints>/fastrxd -i <100G interface>
```

Enable the channel streamer for the module and record. Channels 1 to 128
are pipe 1, 129 to 256 pipe 2, and so on (channel c is column (c-1) % 128
of pipe (c-1) // 128 + 1). One pipe is 2.44 M records per second, about
1.5 GB per second on disk.

```python
import asyncio
import rfmux
from rfmux import fastrx

s = rfmux.load_session('!HardwareMap [ !CRS { serial: "<NNNN>", hostname: "rfmux<NNNN>.local" } ]')
d = s.query(rfmux.CRS).one()

async def enable():
    await d.resolve()
    await d.set_channel_streamer(channels=128, module=<module>, sample_trunc="LOW")

asyncio.run(enable())

with fastrx.PacketWriter("/data/run.fastrx", pipes=[1]) as w:
    w.wait(timeout=20)                       # seconds to record
    print("packets", w.packets, "overruns", w.overruns, "dropouts", w.dropouts)
```

`sample_trunc` picks 16 of the 24 bits of each sample, which is in ADC
counts: `"LOW"` keeps bits 15:0 and is exact while the signal stays within
±32767 counts, `"MID"` keeps bits 19:4 (counts/16) and `"HIGH"` bits 23:8
(counts/256), each dropping the finer bits. With DC levels of a few
thousand counts and noise of a few hundred, HIGH leaves about one bit of
noise; LOW or MID keeps it. The viewer scales every truncation back to
counts. `overruns` counts records the disk was too slow to take. To check
that packets are flowing before recording, `rfmux fastrx hud --pipe 1`.

Leaving the `with` block finalizes the file. The parser stops on Ctrl-C and
the pulse capture in Periscope; the daemon can stay up.

## 3. The viewer

A capture file's pulses over the recording, with the parser's samples on
the same axis. Every trace is in the capture's stored units: volts, or
hertz along the frequency direction for a channel captured with its df
calibration, which rotates and scales the recording and the parser trace
the same way.

```bash
rfmux fastrx overlay pulse_module2_143012.h5 /data/run.fastrx \
    --channel 5 --pulse 1 --dirfile ~/data/run.dirfile/serial_<NNNN> --pad 5
```

Press n and p to step through pulses. `--pad` shows that many milliseconds
of recording either side of the pulse window; `--save fig.png` writes the
figure instead of opening a window. The title reports gaps or dropouts in
the recording window and, for a both-mode capture, the lag at which the
recording best matches the fast-stream pulse.

A window of the parser dirfile over the recording, in counts, without a
capture file:

```bash
rfmux fastrx overlay-dirfile ~/data/run.dirfile/serial_<NNNN> /data/run.fastrx \
    --module 2 --channel 5 --t0 51612.250 --t1 51612.275
```

`--t0` and `--t1` are seconds of day on the PFB clock; a capture file's
pulse attribute `trigger_time`, or any value of the dirfile's timebase, is
on that axis.

From Python the recording is `record_streams` in
`rfmux.algorithms.measurement.record_streams`, and the viewer's pieces
are `Recording`, `pulse_overlay` and `dirfile_window` in
`rfmux.pulse_capture.overlay`:

```python
from rfmux.pulse_capture.hdf5 import PulseHDF5Reader
from rfmux.pulse_capture.overlay import Recording, pulse_overlay

rec = Recording("/data/run.fastrx")
with PulseHDF5Reader("pulse_module2_143012.h5") as r:
    ov = pulse_overlay(r, rec, channel=5, pulse_idx=1,
                       dirfile="~/data/run.dirfile/serial_0156")
# ov.pulse, ov.dirfile, ov.fastrx: dicts of times, I, Q in ov.units
```
