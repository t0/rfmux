"""Packets and timestamps for tests, built as the board builds them."""
import numpy as np

from rfmux import streamer
from rfmux.streamer import Timestamp, TimestampSource


def stamp(seconds: float, *, y: int = 26, d: int = 245,
          recent: bool = True) -> Timestamp:
    """A packet Timestamp at *seconds* of day on day *d* of year *y*."""
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = int(seconds % 60)
    ss = int((seconds % 1) * streamer.SS_PER_SECOND)
    return Timestamp(y=y, d=d, h=h, m=m, s=s, ss=ss, c=0, sbs=0,
                     source=TimestampSource.TEST, recent=recent)


def readout_packet(seq: int, *, t_s: float = 43200.0, module: int = 1,
                   version: int = 5, serial: int = 156, fir_stage: int = 6,
                   raw=None, recent: bool = True, y: int = 26,
                   d: int = 245) -> streamer.ReadoutPacket:
    """A readout packet for *module* (1-indexed) with sequence *seq*,
    stamped at *t_s*; *raw* gives the interleaved int32 I/Q samples,
    zeros otherwise."""
    pkt = streamer.ReadoutPacket(magic=streamer.STREAMER_MAGIC,
                                 version=version, serial=serial,
                                 num_modules=1, flags=0,
                                 fir_stage=fir_stage, module=module - 1,
                                 seq=seq)
    if raw is None:
        pkt[:] = np.zeros(len(pkt), dtype=complex)
    else:
        pkt.raw_samples = raw
    pkt.ts = stamp(t_s, y=y, d=d, recent=recent)
    return pkt


def pfb_packet(module0: int, t_s: float = 43200.0, seq: int = 0,
               slots=(1,), values=None, num_samples: int = 100,
               recent: bool = True) -> streamer.PFBPacket:
    """A PFB packet whose slots carry *slots*' channels (1-indexed),
    interleaved; *values* gives one constant per slot.  *module0* is
    the module as the wire carries it, 0-indexed."""
    pkt = streamer.PFBPacket()
    pkt.magic = streamer.PFB_PACKET_MAGIC
    pkt.module = module0
    pkt.seq = seq
    pkt.mode = {1: 0, 2: 1, 4: 2}[len(slots)]
    for i, ch in enumerate(slots):
        setattr(pkt, f"slot{i + 1}", ch - 1)
    pkt.num_samples = num_samples
    data = np.zeros(num_samples, dtype=complex)
    for i, v in enumerate(values or ()):
        data[i::len(slots)] = v
    pkt[:] = data
    pkt.ts = stamp(t_s, recent=recent)
    return pkt


def pfb_datagram(*args, **kwargs) -> bytes:
    return bytes(pfb_packet(*args, **kwargs))
