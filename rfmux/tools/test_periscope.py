#!/usr/bin/env python3
"""
test_periscope.py
=================

A script to emulate real UDP packets for Periscope, with two modes:

1) **FourModuleLiteThread (default)**:
   - Emulates 4 modules × 1024 channels each = 4096 channels total.
   - Only the first 16 channels in each module get a "fancy" signal (pink noise or
     sine waves), the rest are zero, keeping CPU usage low.
   - We send 4 DfmuxPackets each iteration, one per module, every ~1/f_s seconds.

2) **KidsRasterThread (--kid-raster)**:
   - Emulates 1 module with 1024 channels, but only the first 16 hold a "KIDs raster"
     style signal.  Those 16 are treated as a 4×4 grid that "scans" over a source.
   - Q channel: DC offset, pink noise, white noise, plus an inverse-distance
     response from the beam center.  I channel: ~10% fraction of that same response.
   - The rest channels [16..1023] are zero.
   - Sends 1 DfmuxPacket per iteration.  Sleep ~1/f_s each frame.

**Command‑Line Usage**:
    python test_periscope.py 127.0.0.1 --port 9876 -m 1 -c 1-4096 -d 6 [--kid-raster]

The decimation stage -d sets sampling freq:
    f_s(dec) = (625e6 /256/64) / 2^dec
Then each "frame" is sent at ~f_s, ensuring we produce ~f_s frames/s.

After launching, Periscope receives the real 8,240-byte packets.  The layout matches
the streamer.py from_bytes() logic: 16-byte header + 8192 bytes of channel data + 32
bytes of timestamp = 8,240.

"""

import argparse
import array
import datetime as dt
import math
import random
import socket
import struct
import threading
import time
from typing import Dict, Tuple

import numpy as np

# Local imports
import rfmux.streamer as streamer
import rfmux.tools.periscope as ps
from rfmux.streamer import (
    STREAMER_MAGIC,
    LONG_PACKET_VERSION,
    LONG_PACKET_SIZE,
    LONG_PACKET_CHANNELS,
    SHORT_PACKET_VERSION,
    SHORT_PACKET_SIZE,
    SHORT_PACKET_CHANNELS,
    SS_PER_SECOND,
    TimestampPort,
    Timestamp,
    DfmuxPacket,
)

from rfmux import load_session, CRS

###############################################################################
# Extend DfmuxPacket with a to_bytes() method
###############################################################################
def dfmuxpacket_to_bytes(self: DfmuxPacket) -> bytes:
    """
    Serialize a DfmuxPacket into 8240 bytes, matching streamer.py 'from_bytes()'.
    Layout:
      - 16-byte header (<IHHBBBBI)
      - 8192-byte channel data (NUM_CHANNELS*2 int32)
      - 32-byte timestamp (<8I)
    """
    c_masked = self.ts.c & 0x1FFFFFFF
    # encode source
    source_map = {
        TimestampPort.BACKPLANE: 0,
        TimestampPort.TEST:      1,
        TimestampPort.SMA:       2,
        TimestampPort.GND:       3,
    }
    source_val = source_map.get(self.ts.source, 0)
    c_masked |= (source_val << 29)
    # if recent => bit31
    if self.ts.recent:
        c_masked |= 0x80000000

    # 1) Header
    hdr_struct = struct.Struct("<IHHBBBBI")
    hdr = hdr_struct.pack(
        self.magic,
        self.version,
        self.serial,
        self.num_modules,
        self.block,
        self.fir_stage,
        self.module,
        self.seq,
    )

    # 2) Channel data
    body_bytes = self.s.tobytes()
    if len(body_bytes) not in {LONG_PACKET_CHANNELS*2*4, SHORT_PACKET_CHANNELS*2*4}:
        raise ValueError(f"Channel data must be num_channels*2*4 bytes.")
    
    # 3) Timestamp
    ts_struct = struct.Struct("<8I")
    ts_data = ts_struct.pack(
        self.ts.y,
        self.ts.d,
        self.ts.h,
        self.ts.m,
        self.ts.s,
        self.ts.ss,
        c_masked,
        self.ts.sbs,
    )

    packet = hdr + body_bytes + ts_data
    if len(packet) not in {LONG_PACKET_SIZE, SHORT_PACKET_SIZE}:
        raise ValueError(f"Packet length mismatch: {len(packet)} != {STREAMER_LEN}")
    return packet

setattr(DfmuxPacket, "to_bytes", dfmuxpacket_to_bytes)

###############################################################################
# Pink Noise
###############################################################################
class PinkNoise:
    """
    Simple 1/f noise generator (Voss‑McCartney).
    """
    def __init__(self, size=200000, seed=None):
        self._size = size
        self._rng = random.Random(seed)
        self._levels = [0.0]*int(math.ceil(math.log2(size)))
        self._buf = np.zeros(size, dtype=np.float32)
        self._fill()
        # normalize
        mx = np.abs(self._buf).max()
        if mx>0:
            self._buf/=mx
        self._idx=0

    def _fill(self):
        running_sum=0.0
        for i in range(self._size):
            bit=(i & -i).bit_length()-1
            if bit>=0:
                self._levels[bit] = self._rng.uniform(-1,1)
            running_sum = sum(self._levels)
            self._buf[i] = running_sum

    def next(self)->float:
        val = float(self._buf[self._idx])
        self._idx = (self._idx+1)%self._size
        return val


###############################################################################
# FourModuleLiteThread
#   Emulates 4 modules x 1024 channels each, only first 16 channels get signals
###############################################################################
class FourModuleLiteThread(threading.Thread):
    def __init__(self, host: str, port: int, dec: int):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.dec = dec

        # sampling freq
        base = 625e6/(256*64)  # ~38147.46
        self.fs = base/(2**dec)
        self.dt = 1.0/self.fs

        self.num_modules=4
        # We'll store param for the first 16 channels in each module
        self.params = {}
        self.pink_i = {}
        self.pink_q = {}

        # init
        for mod in range(self.num_modules):
            for ch in range(16):
                self._init_channel(mod,ch)

        self.seq_list = [0]*self.num_modules

        self.sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4_000_000)
        self._running=True

    def stop(self):
        self._running=False
        self.sock.close()

    def run(self):
        while self._running:
            t0=time.perf_counter()
            # send 4 packets
            for mod in range(self.num_modules):
                pkt = self._make_packet(mod)
                raw=pkt.to_bytes()
                try:
                    self.sock.sendto(raw,(self.host,self.port))
                except OSError as e:
                    print(f"sendto err {e}")

            elapsed = time.perf_counter()-t0
            leftover = self.dt-elapsed
            if leftover>0:
                time.sleep(leftover)

    def _init_channel(self, mod:int,ch:int):
        rng = random.Random(mod*100000 + ch*101)
        if ch==0:
            # pink + random DC≥100
            mag=rng.uniform(100,800)
            ph =rng.uniform(0,2*math.pi)
            dc_i = mag*math.cos(ph)
            dc_q = mag*math.sin(ph)
            p_i = PinkNoise(seed=mod*999 + ch)
            p_q = PinkNoise(seed=mod*999 + ch+5555)
            self.params[(mod,ch)] = dict(
                mode="pink",
                dc_i=dc_i,
                dc_q=dc_q,
            )
            self.pink_i[(mod,ch)] = p_i
            self.pink_q[(mod,ch)] = p_q
        else:
            # 1..15 => 1..2 waves + noise + DC
            n_waves = rng.randint(1,2)
            waves=[]
            for _ in range(n_waves):
                amp=rng.uniform(500,3000)
                freq=rng.uniform(-300,300)
                ph0=rng.uniform(0,2*math.pi)
                waves.append((amp,freq,ph0))
            mag=rng.uniform(100,800)
            ph =rng.uniform(0,2*math.pi)
            dc_i = mag*math.cos(ph)
            dc_q = mag*math.sin(ph)
            noise_amp=rng.uniform(0,150)
            self.params[(mod,ch)] = dict(
                mode="sine",
                waves=waves,
                dc_i=dc_i,
                dc_q=dc_q,
                noise_amp=noise_amp
            )

    def _make_packet(self,mod)->DfmuxPacket:
        seq = self.seq_list[mod]
        self.seq_list[mod]+=1

        arr = array.array("i",[0]*(LONG_PACKET_CHANNELS*2))

        t_abs = time.time()
        for ch in range(16):
            param = self.params[(mod,ch)]
            if param["mode"]=="pink":
                i_val = param["dc_i"] + self.pink_i[(mod,ch)].next()*1000
                q_val = param["dc_q"] + self.pink_q[(mod,ch)].next()*1000
            else:
                i_val=param["dc_i"]
                q_val=param["dc_q"]
                for (amp,freq,ph0) in param["waves"]:
                    angle = 2*math.pi*freq*t_abs + ph0
                    i_val += amp*math.cos(angle)
                    q_val += amp*math.sin(angle)
                # noise
                rng = random.Random(mod*123456 + ch*89 + seq)
                i_val += rng.gauss(0,param["noise_amp"])
                q_val += rng.gauss(0,param["noise_amp"])

            arr[ch*2+0] = int(i_val)
            arr[ch*2+1] = int(q_val)

        # timestamp
        dt_utc = dt.datetime.fromtimestamp(t_abs)
        y = dt_utc.year%100
        d = dt_utc.timetuple().tm_yday
        h,m,s= dt_utc.hour, dt_utc.minute, dt_utc.second
        ss = int(dt_utc.microsecond * streamer.SS_PER_SECOND /1e6)

        ts = Timestamp(
            y=y,
            d=d,
            h=h,
            m=m,
            s=s,
            ss=ss,
            c=0,
            sbs=0,
            source=TimestampPort.GND,
            recent=True
        )

        pkt=DfmuxPacket(
            magic=streamer.STREAMER_MAGIC,
            version=streamer.LONG_PACKET_VERSION,
            serial=0,
            num_modules=1,
            block=0,
            fir_stage=0,
            module=mod,
            seq=seq,
            s=arr,
            ts=ts,
        )
        return pkt

###############################################################################
# KidsRasterThread
#   Emulates 1 module x 1024 channels, but only first 16 are a 4x4 "KIDs" array
#   scanning over a source.  Q channel has main response ~ 1/d^2 + pink/noise/DC
###############################################################################
class KidsRasterThread(threading.Thread):
    def __init__(self, host: str, port: int, dec: int):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.dec = dec

        base = 625e6/(256*64)
        self.fs = base/(2**dec)
        self.dt = 1.0/self.fs

        # 1 module => 1024 channels, first 16 are 4x4, rest 0
        self.num_modules=1
        self.seq=0

        # pink noise for each ch in [0..15], for Q channel
        self.pinkQ = {}
        self.pinkI = {}
        self.dcQ   = {}
        self.dcI   = {}
        self.noise_amp = {}
        # init a 4x4
        for ch in range(16):
            rng = random.Random(ch*9999 + 34567)
            # Pink
            self.pinkQ[ch] = PinkNoise(seed=ch+1)
            self.pinkI[ch] = PinkNoise(seed=ch+10000)
            # DC
            mag=rng.uniform(50,200)
            ph = rng.uniform(0,2*math.pi)
            self.dcQ[ch] = mag*math.cos(ph)
            self.dcI[ch] = mag*math.sin(ph)*0.1  # I is ~10%
            # white noise amplitude
            self.noise_amp[ch] = rng.uniform(0.0, 50.0)

        # We'll do a "beam center" that moves across [0..3,0..3] in a slow cycle
        # => We'll maintain a float x_center, y_center that increments over time
        self.x_center = 0.0
        self.y_center = 0.0
        self._center_speed=0.05  # moves 0.05 per iteration => slow drift

        # Socket
        self.sock= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4_000_000)
        self._running=True

    def stop(self):
        self._running=False
        self.sock.close()

    def run(self):
        while self._running:
            t0=time.perf_counter()
            pkt= self._make_packet()
            raw= pkt.to_bytes()
            try:
                self.sock.sendto(raw,(self.host,self.port))
            except OSError as e:
                print(f"send error {e}")

            # update center
            self.x_center += self._center_speed
            self.y_center += self._center_speed * 0.6
            # wrap to remain in [0..3]
            if self.x_center>3.5: self.x_center=0.0
            if self.y_center>3.5: self.y_center=0.0

            elapsed= time.perf_counter()-t0
            leftover= self.dt-elapsed
            if leftover>0:
                time.sleep(leftover)

    def _make_packet(self)->DfmuxPacket:
        arr=array.array("i",[0]*(LONG_PACKET_CHANNELS*2))
        # fill first 16 => 4x4
        t_abs= time.time()
        for ch in range(16):
            # convert ch => x,y
            x= ch%4
            y= ch//4
            # distance from beam center
            dx= x-self.x_center
            dy= y-self.y_center
            dist= math.sqrt(dx*dx + dy*dy)+0.3
            # "KIDs response" => Q ~ DCQ + pinkQ + 1/dist^2 + white noise
            # I is ~10% plus same noise
            base_q= self.dcQ[ch] + self.pinkQ[ch].next()*50.0
            base_i= self.dcI[ch] + self.pinkI[ch].next()*5.0

            # 1/dist^2
            # Avoid huge => clamp dist>0.05
            if dist<0.05: dist=0.05
            response= 5000.0/(dist*dist)
            # add to Q primarily, 10% to I
            q_val= base_q + response
            i_val= base_i + 0.1*response

            # white noise
            rng= random.Random(ch*123456 + int(self.seq))
            q_val+= rng.gauss(0,self.noise_amp[ch])
            i_val+= rng.gauss(0,self.noise_amp[ch]*0.1)

            arr[ch*2+0] = int(i_val)
            arr[ch*2+1] = int(q_val)

        self.seq+=1

        # timestamp
        dt_utc = dt.datetime.fromtimestamp(t_abs)
        y = dt_utc.year%100
        d = dt_utc.timetuple().tm_yday
        h,m,s= dt_utc.hour, dt_utc.minute, dt_utc.second
        ss = int(dt_utc.microsecond * streamer.SS_PER_SECOND /1e6)

        ts = Timestamp(
            y=y,d=d,h=h,m=m,s=s,ss=ss,
            c=0,sbs=0,
            source=TimestampPort.GND,
            recent=True
        )

        pkt= DfmuxPacket(
            magic=streamer.STREAMER_MAGIC,
            version=streamer.LONG_PACKET_VERSION,
            serial=0,
            num_modules=1,
            block=0,
            fir_stage=0,
            module=0,
            seq=self.seq,
            s=arr,
            ts=ts,
        )
        return pkt


###############################################################################
# main
###############################################################################
async def main():
    ap = argparse.ArgumentParser(
        description="Emulate Periscope data with either a 4-module-lite or a KIDs raster mode."
    )
    ap.add_argument("hostname", default="127.0.0.1", help="IP for sending real UDP, e.g. 127.0.0.1")
    ap.add_argument("--port", type=int, default=9876, help="UDP port to send to")
    ap.add_argument("-m","--module",type=int,default=1,help="passed to Periscope, not used in the emu")
    ap.add_argument("-c","--channels",default="1-4096",help="passed to Periscope if you like")
    ap.add_argument("-d","--dec",type=int,default=6,choices=range(1,7),
                    help="decimation => sampling freq => 1/f_s")
    ap.add_argument("--kid-raster",action="store_true",
                    help="If set, run the 'KidsRasterThread' (1 module, 16-ch scanning). Otherwise do 4-module lite.")
    args = ap.parse_args()

    args.hostname = '127.0.0.1'

    s = load_session(f"""
!HardwareMap
- !CRS {{ hostname: {args.hostname} }}
""")
    # Access the CRS object  
    crs = s.query(CRS).one() 

    # Launch Periscope in non-blocking
    viewer, qt_app = await crs.raise_periscope(
        module=args.module,
        channels=args.channels,
        fps=30.0,
        blocking=False,
    ) 

    if args.kid_raster:
        # KIDs Raster approach: 1 module, 1024ch, only first 16 used
        inj= KidsRasterThread(host=args.hostname, port=args.port, dec=args.dec)
    else:
        # 4 modules, 1024ch each, only first 16 per module computed => 4096 total
        inj= FourModuleLiteThread(host=args.hostname, port=args.port, dec=args.dec)

    inj.start()
    try:
        qt_app.exec()
    finally:
        inj.stop()
        inj.join()


if __name__=="__main__":
    import asyncio
    asyncio.run(main())      # starts & closes the loop
