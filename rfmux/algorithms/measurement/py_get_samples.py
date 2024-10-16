"""
py_get_samples: an experimental pure-Python, client-side implementation of the
get_samples() call.

CONSIDERATIONS:
    - Packet loss due to network congestion or mis-configuration is more likely
      to show up with a client-side get_samples() than a CRS-side
      get_samples(), because the CRS is tapping packets off its own interface
      and not at the far end of a potentially unreliable network.

    - Binary packing and geometry are exposed in Python here - which makes it
      harder to change. Given how stable the packet format has been, I don't
      find this a compelling counter-argument to a Python get_samples()
      alternative - but it's worth remembering.

TO DO:
    - Overrange / overvoltage flags present in get_samples are not decoded here
"""

from ...core.schema import CRS
from ...core.hardware_map import macro
from ...tuber.codecs import TuberResult

import struct
import array
import enum
import socket
import contextlib
import numpy as np

from dataclasses import dataclass, asdict

STREAMER_PORT = 9876
STREAMER_HOST = "239.192.0.2"
STREAMER_LEN = 8240
STREAMER_MAGIC = 0x5344494B
STREAMER_VERSION = 5
NUM_CHANNELS = 1024


class TimestampPort(str, enum.Enum):
    BACKPLANE = "BACKPLANE"
    TEST = "TEST"
    SMA = "SMA"
    GND = "GND"


@dataclass
class Timestamp:
    y: np.int32
    d: np.int32
    h: np.int32
    m: np.int32
    s: np.int32
    ss: np.int32
    c: np.int32
    sbs: np.int32

    source: TimestampPort
    recent: bool

    @classmethod
    def from_bytes(self, data):
        args = struct.unpack("<8I", data)
        ts = Timestamp(*args, recent=False, source=TimestampPort.GND)

        # decode source
        match (ts.c >> 29) & 0x3:
            case 0:
                ts.source = TimestampPort.BACKPLANE
            case 1:
                ts.source = TimestampPort.TEST
            case 2:
                ts.source = TimestampPort.SMA
            case 3:
                ts.source = TimestampPort.GND
            case _:
                raise RuntimeError("Unexpected timestamp source!")

        # decode recent
        ts.recent = bool(ts.c & 0x80000000)

        # Mask off these bits from the "c" field
        ts.c &= 0x1FFFFFFF

        return ts


@dataclass
class DfmuxPacket:
    magic: np.uint32
    version: np.uint16
    serial: np.uint16

    num_modules: np.uint8
    block: np.uint8
    fir_stage: np.uint8
    module: np.uint8

    seq: np.uint32

    s: array.array

    ts: Timestamp

    @classmethod
    def from_bytes(self, data):
        assert (
            len(data) == STREAMER_LEN
        ), f"Packet had unexpected size {len(data)} != {STREAMER_LEN}!"

        header = struct.Struct("<IHHBBBBI")
        header_args = header.unpack(data[: header.size])

        body = array.array("i")
        bodysize = NUM_CHANNELS * 2 * body.itemsize
        body.frombytes(data[header.size : header.size + bodysize])

        ts = Timestamp.from_bytes(data[header.size + bodysize :])

        return DfmuxPacket(*header_args, s=body, ts=ts)


@macro(CRS, register=True)
async def py_get_samples(crs, num_samples, channel=None, module=None):

    # "module" is given a default value due to syntax restrictions - but we
    # require it to be set and sane.
    assert module in crs.modules.module, "Unspecified or invalid module!"

    if channel is not None:
        assert 1 <= channel <= NUM_CHANNELS, f"Invalid channel {channel}!"

    packets = []

    with contextlib.closing(
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    ) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((STREAMER_HOST, STREAMER_PORT))

        # Use source-specific multicast (SSM) to collect packets from only this CRS board
        crs_ip = socket.inet_aton(socket.gethostbyname(crs.tuber_hostname))
        mc_ip = socket.inet_aton(STREAMER_HOST)
        mreq_source = struct.pack("4sI4s", mc_ip, socket.INADDR_ANY, crs_ip)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_SOURCE_MEMBERSHIP, mreq_source)

        # FIXME: CAUSALITY DELAY

        while len(packets) < num_samples:
            p = DfmuxPacket.from_bytes(sock.recv(STREAMER_LEN))
            if p.serial != int(crs.serial):
                raise RuntimeError(
                    "Packet serial number {p.serial} didn't match CRS serial number {crs.serial}!"
                )

            # In c++ code, this filtering is done kernel-side via BPF. It's
            # probably more efficient that way - we could do the same here too.
            if p.module != module - 1:
                print(f"Punted unwanted module {p.module+1}")
                continue

            # Sanity checks
            if p.magic != STREAMER_MAGIC:
                raise RuntimeError(
                    "Invalid packet magic! {p.magic} != {STREAMER_MAGIC}"
                )

            if p.version != STREAMER_VERSION:
                raise RuntimeError(
                    "Invalid packet magic! {p.magic} != {STREAMER_MAGIC}"
                )

            # Check for contiguity
            with np.errstate(over="ignore"):
                if packets and packets[-1].seq + 1 != p.seq:
                    raise RuntimeError(
                        f"Discontinuous packet capture! Index {len(packets)}, sequence {packets[-1].seq} -> {p.seq}"
                    )

            packets.append(p)

    # We've collected the right amount of data and need to construct a result
    # that matches expectations.
    results = dict(ts=[TuberResult(asdict(p.ts)) for p in packets])

    if channel is None:
        results["i"] = []
        results["q"] = []
        for c in range(NUM_CHANNELS):
            results["i"].append([p.s[2 * c] / 256 for p in packets])
            results["q"].append([p.s[2 * c + 1] / 256 for p in packets])
    else:
        results["i"] = [p.s[2 * (channel - 1)] / 256 for p in packets]
        results["q"] = [p.s[2 * (channel - 1) + 1] / 256 for p in packets]

    return TuberResult(results)
