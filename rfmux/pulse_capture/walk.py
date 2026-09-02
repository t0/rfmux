"""The engine's per-sample state machine over a block.

``walk`` is :meth:`PulseCapture.process_sample` transcribed to operate
on arrays and scalars only: the rings and the baseline reservoir as
their doubled buffers with pointer and count, the channel state packed
into two small arrays, the engine's parameters as scalars.  It walks a
block until something Python must handle -- a capture ending, a hard
stop, a pileup split -- and returns there.  process_sample stays as
written and is the definition the tests hold this to.

Two things never happen inside: a baseline refresh (the driver hands
the one sample a refresh falls on to process_sample), and a save.

Compiled with numba and released from the GIL: while one engine walks,
the receiver, the other stream's engine, and the GUI keep running.
``walk.py_func`` is the same function uncompiled.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

# ── Reasons a walk returns ──────────────────────────────────────────
DONE = 0            # walked to *stop*
END = 1             # end confirmed on the sample at the returned index
HARD_STOP = 2       # hard stop on that sample
SPLIT = 3           # pileup split on that sample: save, then re-arm

# ── Packed state layout ─────────────────────────────────────────────
# Integers.  trig_abs and active_duration use -1 for None; the two
# quadrature markers use 0 for "", 1 for "I", 2 for "Q".
CAPTURING, END_PTR, TRIG_ABS, FIRE_ABS, RUN_QUAD, TRIG_QUAD, PILEUP_CHILD, \
    CH_N, RETRIG, ACTIVE_DUR, ABOVE_RUN, RUN_START, EPOCH, DECIM_N, \
    SINCE_REFRESH = range(15)
N_INT = 15
# Floats.
ANCHOR_I, ANCHOR_Q, TMEAN_I, TMEAN_Q, TSTD_I, TSTD_Q, NEAR_I, NEAR_Q = range(8)
N_FLT = 8

_SQRT2 = math.sqrt(2.0)


@njit(nogil=True, cache=True)
def _median3(a, b, c):
    """The middle of three, as sorted()[1] would give."""
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
        if a > b:
            a, b = b, a
    return b


@njit(nogil=True, cache=True)
def walk(I, Q, T, start, stop,
         rI, rQ, rT, rptr, rcount, rN,
         bI, bQ, bptr, bcount, bN, bl_decim, baseline_on,
         mean_I, mean_Q, std_I, std_Q, jump_std_I, jump_std_Q,
         thr, end_sigma, trigger_samples, edge_lookback, min_end,
         margin, max_capture, enable_pileup, freeze,
         si, sf, out):
    """Walk samples start..stop-1.  Writes the rings and the state in
    place; ``out`` receives (index, reason, rptr, rcount, bptr, bcount).
    """
    js_I = jump_std_I if jump_std_I > 0 else _SQRT2 * std_I
    js_Q = jump_std_Q if jump_std_Q > 0 else _SQRT2 * std_Q
    sI = max(std_I, 1e-30)
    sQ = max(std_Q, 1e-30)
    jI = max(js_I, 1e-30)
    jQ = max(js_Q, 1e-30)
    reason = DONE
    k = start
    while k < stop:
        i_val = I[k]
        q_val = Q[k]
        # ring feed
        rI[rptr] = i_val
        rI[rptr + rN] = i_val
        rQ[rptr] = q_val
        rQ[rptr + rN] = q_val
        rT[rptr] = T[k]
        rT[rptr + rN] = T[k]
        rptr = (rptr + 1) % rN
        if rcount < rN:
            rcount += 1
        si[CH_N] += 1
        ch_n = si[CH_N]
        # rolling-baseline reservoir (a refresh never falls in here)
        if baseline_on:
            si[DECIM_N] += 1
            if si[DECIM_N] >= bl_decim:
                si[DECIM_N] = 0
                bI[bptr] = i_val
                bI[bptr + bN] = i_val
                bQ[bptr] = q_val
                bQ[bptr + bN] = q_val
                bptr = (bptr + 1) % bN
                if bcount < bN:
                    bcount += 1
                si[SINCE_REFRESH] += 1
        # deviations
        raw_I = abs(i_val - mean_I)
        raw_Q = abs(q_val - mean_Q)
        dev_I = raw_I / sI
        dev_Q = raw_Q / sQ
        max_dev = dev_I if dev_I > dev_Q else dev_Q
        # trigger confirmation
        if max_dev > thr:
            if si[ABOVE_RUN] == 0:
                si[RUN_START] = ch_n
                si[RUN_QUAD] = 1 if dev_I >= dev_Q else 2
            si[ABOVE_RUN] += 1
        else:
            si[ABOVE_RUN] = 0
        eligible = si[ABOVE_RUN] >= trigger_samples
        capturing = si[CAPTURING] != 0
        # edge detector
        edge_ok = False
        have_taps = False
        t0_I = t1_I = t2_I = 0.0
        t0_Q = t1_Q = t2_Q = 0.0
        n_taps = 0
        if edge_lookback > 0 and eligible and not capturing:
            lmax = edge_lookback
            if rcount - 1 < lmax:
                lmax = rcount - 1
            if ch_n - si[EPOCH] - 1 < lmax:
                lmax = ch_n - si[EPOCH] - 1
            if lmax >= 1:
                for tap in (lmax, lmax // 2, lmax // 4):
                    if tap >= 1:
                        idx = (rptr - 1 - tap) % rN
                        if n_taps == 0:
                            t0_I, t0_Q = rI[idx], rQ[idx]
                        elif n_taps == 1:
                            t1_I, t1_Q = rI[idx], rQ[idx]
                        else:
                            t2_I, t2_Q = rI[idx], rQ[idx]
                        n_taps += 1
                have_taps = True
                if n_taps == 3:
                    ref_I = _median3(abs(t0_I - mean_I), abs(t1_I - mean_I),
                                     abs(t2_I - mean_I))
                    ref_Q = _median3(abs(t0_Q - mean_Q), abs(t1_Q - mean_Q),
                                     abs(t2_Q - mean_Q))
                elif n_taps == 2:
                    # sorted()[1] of two is the larger
                    a = abs(t0_I - mean_I)
                    b = abs(t1_I - mean_I)
                    ref_I = a if a > b else b
                    a = abs(t0_Q - mean_Q)
                    b = abs(t1_Q - mean_Q)
                    ref_Q = a if a > b else b
                else:
                    ref_I = abs(t0_I - mean_I)
                    ref_Q = abs(t0_Q - mean_Q)
                edge_ok = ((raw_I - ref_I) / jI > thr
                           or (raw_Q - ref_Q) / jQ > thr)
        # trigger
        if edge_lookback > 0:
            trigger_ok = eligible and edge_ok
        else:
            trigger_ok = si[ABOVE_RUN] == trigger_samples
        if not capturing and not freeze and trigger_ok:
            si[CAPTURING] = 1
            si[END_PTR] = 0
            si[FIRE_ABS] = ch_n
            sf[TMEAN_I] = mean_I
            sf[TMEAN_Q] = mean_Q
            sf[TSTD_I] = std_I
            sf[TSTD_Q] = std_Q
            si[TRIG_QUAD] = si[RUN_QUAD]
            if have_taps:
                if n_taps == 3:
                    sf[ANCHOR_I] = _median3(t0_I, t1_I, t2_I)
                    sf[ANCHOR_Q] = _median3(t0_Q, t1_Q, t2_Q)
                elif n_taps == 2:
                    sf[ANCHOR_I] = t0_I if t0_I > t1_I else t1_I
                    sf[ANCHOR_Q] = t0_Q if t0_Q > t1_Q else t1_Q
                else:
                    sf[ANCHOR_I] = t0_I
                    sf[ANCHOR_Q] = t0_Q
            else:
                sf[ANCHOR_I] = mean_I
                sf[ANCHOR_Q] = mean_Q
            si[TRIG_ABS] = si[RUN_START]
            if edge_lookback > 0 and ch_n - edge_lookback > si[TRIG_ABS]:
                si[TRIG_ABS] = ch_n - edge_lookback
            capturing = True
        # end condition & pileup
        if capturing:
            trig_abs = si[TRIG_ABS] if si[TRIG_ABS] >= 0 else ch_n
            since_trig = ch_n - trig_abs
            since_fire = ch_n - si[FIRE_ABS]
            decaying_now = False
            rising_above_self = False
            have_near = False
            if enable_pileup and edge_lookback > 0 and since_fire >= 1:
                span = edge_lookback
                if since_fire < span:
                    span = since_fire
                if rcount - 1 < span:
                    span = rcount - 1
                if ch_n - si[EPOCH] - 1 < span:
                    span = ch_n - si[EPOCH] - 1
                if span >= 1:
                    hi_I = 0.0
                    hi_Q = 0.0
                    for tap in (span, span // 2, span // 4):
                        if tap >= 1:
                            idx = (rptr - 1 - tap) % rN
                            d = abs(rI[idx] - mean_I)
                            if d > hi_I:
                                hi_I = d
                            d = abs(rQ[idx] - mean_Q)
                            if d > hi_Q:
                                hi_Q = d
                    decaying_now = ((raw_I - hi_I) / jI < -thr
                                    or (raw_Q - hi_Q) / jQ < -thr)
                    near = edge_lookback // 4
                    if span < near:
                        near = span
                    if near < 1:
                        near = 1
                    idx = (rptr - 1 - near) % rN
                    sf[NEAR_I] = rI[idx]
                    sf[NEAR_Q] = rQ[idx]
                    have_near = True
                    rising_above_self = (
                        (raw_I - abs(sf[NEAR_I] - mean_I)) / jI > thr
                        or (raw_Q - abs(sf[NEAR_Q] - mean_Q)) / jQ > thr)
            returned = (abs(i_val - sf[ANCHOR_I]) < end_sigma * sI
                        and abs(q_val - sf[ANCHOR_Q]) < end_sigma * sQ)
            if max_dev < thr or returned:
                si[RETRIG] = 1
                if si[ACTIVE_DUR] < 0:
                    si[ACTIVE_DUR] = since_trig
            elif decaying_now and si[RETRIG] == 0 and since_fire > min_end:
                si[RETRIG] = 1
            if (enable_pileup and edge_lookback > 0 and si[RETRIG] != 0
                    and eligible and rising_above_self):
                # The split: Python saves this capture, then re-arms the
                # next one with these values.
                reason = SPLIT
                if not have_near:
                    sf[NEAR_I] = math.nan
                    sf[NEAR_Q] = math.nan
                break
            if returned or (dev_I < end_sigma and dev_Q < end_sigma):
                si[END_PTR] += 1
                if si[ACTIVE_DUR] < 0:
                    si[ACTIVE_DUR] = since_trig
            elif si[END_PTR] > 0:
                si[END_PTR] -= 1
            ref_duration = si[ACTIVE_DUR] if si[ACTIVE_DUR] > 0 else since_trig
            adaptive_end = int(margin * ref_duration)
            if adaptive_end < min_end:
                adaptive_end = min_end
            if si[END_PTR] > adaptive_end:
                reason = END
                break
            if max_capture > 0 and since_trig >= max_capture:
                reason = HARD_STOP
                break
        k += 1
    out[0] = k
    out[1] = reason
    out[2] = rptr
    out[3] = rcount
    out[4] = bptr
    out[5] = bcount


def warm_up() -> None:
    """Compile, or load from cache, before the first block arrives."""
    z = np.zeros(1, dtype=np.float64)
    ring = np.zeros(2, dtype=np.float64)
    walk(z, z, z, 0, 0, ring, ring, ring, 0, 0, 1, ring, ring, 0, 0, 1, 1,
         True, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 5.0, 2.0, 2, 1, 1, 0.5, 0,
         True, False, np.zeros(N_INT, dtype=np.int64),
         np.zeros(N_FLT, dtype=np.float64), np.zeros(6, dtype=np.int64))
