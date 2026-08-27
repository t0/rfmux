---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# test_py_get_samples

These tests demonstrate (and validate) scaling and conversion expectations for py_get_samples. They are kept here as a notebook in order to make the computations and results easier to visualize and explain.

```python
import sys
sys.path.append('../../')
import rfmux

# We're going to need the spectrum helper py_get_samples uses.  It used to
# live in py_get_samples as _compute_spectrum; it now sits in
# rfmux.core.transferfunctions and takes dec_stage instead of (fsamp, stage).
from rfmux.core.transferfunctions import spectrum_from_slow_tod

import matplotlib.pyplot as plt
import numpy as np
import pytest
import warnings

fir_stage = 6
fsamp = 625e6/(256*64*2**fir_stage)
```

## DC signal in I only

```python
with warnings.catch_warnings(action='ignore'):  # don't complain about divide-by-0
    x = spectrum_from_slow_tod(np.ones(1000),
                               np.zeros(1000),
                               fir_stage,
                               scaling='ps')

(freq_dsb, psd_dsb, psd_i, psd_q, freq_iq) = (
    x["freq_dsb"],
    x["psd_dual_sideband"],
    x["psd_i"],
    x["psd_q"],
    x["freq_iq"],
)

plt.figure()
plt.plot(np.fft.fftshift(freq_dsb),
         np.fft.fftshift(psd_dsb), label="DSB")
plt.plot(freq_iq, psd_i, label="I")
plt.plot(freq_iq, psd_q, label="Q") # probably doesn't show
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amp (dBc)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Q had better be negligible
assert all(psd_q < -300)

# I and DSB spectra are only allowed energy at DC.  The single-sideband
# channels get floored to ~-3000 dB, but the dual-sideband spectrum carries
# genuine float64 FFT roundoff at ~-300 dB, so it is held to -200 dB — still
# 20 orders of magnitude below the 0 dB carrier at DC.
assert all((psd_i < -300) | (np.abs(freq_iq) < 1))
assert all((psd_dsb < -200) | (np.abs(freq_dsb) < 1))

# magnitude 1 -> expect 0 dB
assert max(psd_i) == pytest.approx(0, abs=1)
assert max(psd_dsb) == pytest.approx(0, abs=1)
```

## DC signal in Q only

```python
with warnings.catch_warnings(action='ignore'):  # don't complain about divide-by-0
    x = spectrum_from_slow_tod(np.zeros(1000),
                               np.ones(1000),
                               fir_stage,
                               scaling='ps')

(freq_dsb, psd_dsb, psd_i, psd_q, freq_iq) = (
    x["freq_dsb"],
    x["psd_dual_sideband"],
    x["psd_i"],
    x["psd_q"],
    x["freq_iq"],
)


plt.figure()
plt.plot(np.fft.fftshift(freq_dsb),
         np.fft.fftshift(psd_dsb), label="DSB")
plt.plot(freq_iq, psd_i, label="I")
plt.plot(freq_iq, psd_q, label="Q") # probably doesn't show
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amp (dBc)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# I had better be negligible
assert all(psd_i < -300)

# Q and DSB spectra are only allowed energy at DC (see the -200 dB note above)
assert all((psd_q < -300) | (np.abs(freq_iq) < 1))
assert all((psd_dsb < -200) | (np.abs(freq_dsb) < 1))

# magnitude 1 -> expect 0 dB
assert max(psd_q) == pytest.approx(0, abs=1)
assert max(psd_dsb) == pytest.approx(0, abs=1)
```

```python
# Complex +Nyquist/2 signal
x = spectrum_from_slow_tod(np.array([1, 0, -1, 0]*250),
                           np.array([0, 1, 0, -1]*250),
                           fir_stage,
                           scaling='ps', reference='absolute')

(freq_dsb, psd_dsb, psd_i, psd_q, freq_iq) = (
    x["freq_dsb"],
    x["psd_dual_sideband"],
    x["psd_i"],
    x["psd_q"],
    x["freq_iq"],
)

plt.figure()
plt.plot(np.fft.fftshift(freq_dsb),
         np.fft.fftshift(psd_dsb), label="DSB")
plt.plot(freq_iq, psd_i, label="I")
plt.plot(freq_iq, psd_q, label="Q") # probably doesn't show
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amp (dBc)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# The complex PSD is only allowed to have "big" signal in a narrow neighbourhood around Nyquist/4.
assert all(psd_dsb[np.abs(freq_dsb - fsamp/4) > 5] < -200)

# Maximum magnitude should compensate for CIC droop - so needs to be > 0dB here
assert max(np.abs(psd_dsb) > 0)
```

```python

```
