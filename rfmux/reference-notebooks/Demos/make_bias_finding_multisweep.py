#!/usr/bin/env python
"""Record the multisweep that ``bias_finding.md`` reads.

The notebook is about analysing a multi-amplitude sweep, not about taking one,
and taking one against the simulator costs a couple of minutes. So it is
measured once, here, and pickled next to the notebook.

Run this to regenerate the file — after a change to ``multisweep``'s output
shape, say, or to sweep a different array::

    python make_bias_finding_multisweep.py

It writes ``bias_finding_multisweep.pkl``, holding exactly what
``crs.multiamp_multisweep`` returned: a dict keyed by module identifier, with
the catalog that was swept recorded inside it under ``call_params``. Nothing is
reshaped on the way to disk, so what the notebook loads is what a macro hands
you.

The array is a fixed random seed, so re-running this produces the same four
resonators — but the sweeps themselves carry the simulator's noise, which is
not seeded. Regenerating changes every number in the notebook's output by a
little, which is what re-measuring does.
"""

import asyncio
import pickle
from pathlib import Path

import rfmux
from rfmux.core.resonators import ResonatorCatalog
from rfmux.tuning import AmplitudeSchedule

MODULE = 1
HERE = Path(__file__).parent
OUT = HERE / "bias_finding_multisweep.pkl"

PROBE_AMPLITUDE = 0.001  # normalized DAC units — where the array starts out

# Four resonators is enough for a plot grid and small enough to keep the file
# in the low hundreds of kilobytes.
MOCK_CONFIG = {
    "num_resonances": 4,
    "freq_start": 0.6e9,
    "freq_end": 0.9e9,
    "resonator_random_seed": 42,
    "auto_bias_kids": True,  # the simulator parks a tone on each resonance
    "bias_amplitude": PROBE_AMPLITUDE,
}

# The span has to hold the resonance at every amplitude step: driving one of
# these harder pulls it down by about 15 kHz over the range below, so a 60 kHz
# span still has it well inside. 201 points over that is ~300 Hz apart, which
# puts several points across a linewidth — the bifurcation test needs the
# resonance resolved, or a dip crossed in two samples looks like a jump.
SPAN_HZ = 60e3
NPOINTS_PER_SWEEP = 201
NSAMPS = 10

# Six amplitude steps, each twice the last. The first is where the array is
# biased now.
AMPLITUDE_SCHEDULE = AmplitudeSchedule.multiplicative(1.0, 32.0, 6)


async def main():
    session = rfmux.load_session(
        """
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
"""
    )
    crs = session.query(rfmux.CRS).one()
    await crs.resolve()

    resonator_count, _ = await crs.generate_resonators(MOCK_CONFIG)

    # Where the simulator put its own tones. get_frequency reports relative to
    # the NCO, so add it back.
    nco_frequency = await crs.get_nco_frequency(module=MODULE)
    bias_frequencies = [
        nco_frequency + await crs.get_frequency(channel=channel, module=MODULE)
        for channel in range(1, resonator_count + 1)
    ]

    catalog = ResonatorCatalog.from_frequencies(
        bias_frequencies, module=MODULE, amplitude=PROBE_AMPLITUDE
    )
    print(catalog)

    sweeps = await crs.multiamp_multisweep(
        catalog,
        span_hz=SPAN_HZ,
        npoints_per_sweep=NPOINTS_PER_SWEEP,
        nsamps=NSAMPS,
        amp_schedule=AMPLITUDE_SCHEDULE,
        directions=("upward", "downward"),
    )

    with OUT.open("wb") as f:
        pickle.dump(sweeps, f)

    module_sweeps = sweeps[crs.module[MODULE].index()]
    print(
        f"\nwrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)\n"
        f"  modules:         {list(sweeps)}\n"
        f"  amplitude steps: {list(module_sweeps['results'])}\n"
        f"  directions:      {list(module_sweeps['results'][0])}\n"
        f"  resonators:      {list(module_sweeps['results'][0]['upward'])}"
    )


if __name__ == "__main__":
    asyncio.run(main())
