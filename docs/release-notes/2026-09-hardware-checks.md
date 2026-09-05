# Hardware checks for the pulse-capture branch

What the branch assumes about a real CRS that only a board can confirm,
each with the measurement that settles it and what to change if it comes
out the other way. Run these against one board streaming to one host, with
Periscope closed so nothing else holds the streamer ports.

Settled already, by inspection of a board: `crs.get_samples` and
`py_get_samples` agree in counts, so the C++ receiver's single division by
256 leaves ADC counts on the slow stream; all voltages in rfmux are peak
amplitudes, not rms.

## 1. The count scale on the fast stream

The branch now applies one division by 256 (the receiver's) to both
streams and to the simulator, so a stream sample and a `get_samples` or
`get_pfb_samples` sample carry the same counts.

- Bias one tone, leave it on. Read the channel's mean through
  `get_pfb_samples` in counts and through the PFB stream
  (`PacketQueue.pop_pfb_batch`, or `np.array(pkt)` on a `PFBPacket`).
- Expect the two means to agree; a ratio of 256 either way means the fast
  stream needs its own scale constant in `rfmux/pulse_capture/sources.py`
  and the simulator's PFB emitter must follow.
- Repeat on the slow stream (`get_samples` against `np.array(pkt)` on a
  `ReadoutPacket`) to record the number, even though it was settled by
  inspection.
- With Periscope on the same tone in Real Units, expect the IQ plot's level
  and the Noise Spectrum panel's dBm to agree with `get_samples` converted
  through `convert_roc_to_volts` and `convert_roc_to_dbm`. Periscope's main
  window divided by 256 a second time until this branch, so a display that
  matches the old level rather than `get_samples` says the receiver is not
  yielding counts and the fixes need reversing.

## 2. The fast-to-slow gain

The dual capture compares pulse heights across the two streams and the
simulator packs both at one scale.

- With the tone from check 1 on, record both means at the same time.
- Expect them equal within the noise. A fixed ratio becomes a named
  constant applied on the fast source, and the simulator's PFB emitter
  must reproduce it.

## 3. The df calibration reads a frequency step

The calibration is measured in hertz per volt from `get_samples` counts
and applied to stream samples; a wrong count scale shows up here first.

- Run `multisweep` then `bias_kids` on a few resonators. Step one biased
  tone by 300 Hz with `set_frequency`, read the channel's mean before and
  after through the slow stream, and multiply the difference in volts by the
  stored `df_calibration`.
- Expect about +300 Hz along the real axis and near zero on the imaginary
  one. A factor of 256 points back to check 1; a rotation points to check 4.
- `test/algorithms/test_multisweep_then_bias.py` does this on the
  simulator; the same sequence is the hardware-tier test to add.

## 4. The ADC phase sign

`bias_kids` turns the calibration by minus the phase it programs, assuming
the board rotates samples by plus the phase, as the simulator does.

- With one tone biased, read the channel's mean, set the ADC phase to +90
  degrees with `set_phase(..., target=ADC)`, read again.
- Expect the complex mean to have turned by +90 degrees (multiplied by
  `exp(+j pi/2)`). If it turned the other way, the sign in `bias_kids`
  (the `exp(-1j * radians(phase))` on the fit calibration and the PCA's
  `90 - theta`) flips, and the phase-optimisation test's expectation with it.

## 5. The CIC group delay on slow-stream timestamps

In dual mode the slow clock is pulled back by the CIC group delay so pairs
match; the sign was pinned against a simulator that stamps its slow packets
late.

- Capture in `streamer_mode="both"` with pulses present (a heater pulse or
  a tone step) and look at `pairs[i]["time_offset"]`.
- Expect offsets centred near zero. A consistent offset of about twice the
  group delay means the sign is wrong; one delay means the board already
  stamps at the filter centroid and the shift should be removed for boards.

## 6. The DAC scale to tone power offset

Periscope subtracts 1.5 dB from `get_dac_scale` before every dBm label,
the simulator subtracts nothing, and the QC test uses 2.56 dB.

- Set one tone at a known normalized amplitude, measure its power at the
  DAC output with a spectrum analyser, and compare with
  `get_dac_scale() + 20 log10(amplitude)`.
- Record the offset as one named constant; Periscope, the simulator and
  the netanal legends all move to it. Check also that `set_dac_scale`
  changes the measured power by the amount set.

## 7. What `get_pfb_samples` returns by default

The simulator returns normalized values; the PFB spectrum macro multiplies
by the counts-to-volts constant, which is right only for counts.

- Call `get_pfb_samples` with no units argument on one tone and compare its
  scale with `get_samples` on the same tone.
- Expect counts. If not, `py_get_pfb_samples` must request counts
  explicitly, and the simulator's stub should return them.

## 8. Stale timestamps on the fast stream

`py_run_pfb_streamer` and the pulse-capture sources treat a packet whose
timestamp is not marked recent as unusable, and the runner raises after five
seconds of them.

- With no timestamp source set, start a fast capture; expect the
  `set_timestamp_port` error within about five seconds, not a hang.
- Set the port, wait two seconds, capture again; expect it to run.

## 9. Two readers on one stream

The mock's competing-receiver warning fires only for a unicast stream.

- With Periscope running on the board, start a headless slow capture on the
  same host. Expect both to receive every packet (multicast), and no
  competing-receiver warning in Periscope.

## 10. The readout noise floor, for the simulator

The simulator's white readout noise (`udp_noise_level`, 0.04 counts per
slow sample) preserves the signal-to-noise it had before the count scale
was corrected; neither it nor the earlier value came from a board.

- With no tone on a channel, record the slow stream's sigma in counts at
  stage 6 (`np.std` of `get_samples(10000)` or of the stream) and the PFB
  stream's sigma on the same channel.
- Set `udp_noise_level` in `rfmux/mock/config.py` to the slow-stream
  figure. Expect the PFB figure to be about 64 times larger at stage 6
  (root of the decimation ratio); a different ratio means the simulator's
  PFB sigma rule in `_emit_pfb_frame` needs the measured ratio instead.
- Check the pulse-capture demos and `test/pulse_capture/` still detect at
  their thresholds with the new floor; `threshold_sigma` in the tests was
  tuned to the provisional one.

## What to record

For each check: board serial, firmware version, decimation stage, the two
numbers compared, and the ratio or angle. The hardware tier
(`pytest --tier=hardware --serial <serial>`) is where checks 1 to 4 become
tests once the numbers are in hand; check 10 sets a simulator default.
