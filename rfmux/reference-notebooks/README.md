# Reference notebooks

These ship with the `rfmux` package and are provisioned **read-only**, so save
your own copies elsewhere (*File → Save Notebook As…*) before editing.

## Opening them

They are jupytext markdown rather than `.ipynb`. In the Jupyter session
Periscope launches they open as notebooks on double-click. In a JupyterLab you
started yourself, right-click → *Open With* → *Notebook*, or set
*Settings → Document Manager → Default Viewers* to `markdown: Jupytext
Notebook`. To convert one instead:

```bash
jupytext -o pulse_capture.ipynb pulse_capture.md
```

## Where to start

The first four take the tuning flow a step at a time, in this order:

- **`Demos/network_analysis_find_resonances.md`** — sweep a band, find the dips,
  and seed the resonator catalog everything downstream passes around.
- **`Demos/resonator_catalogs.md`** — the catalog on its own: building one by
  hand, reading and amending it, the invariants, and the CSV and dictionary
  round trips. It picks up from a saved network analysis, so it needs no
  hardware.
- **`Demos/multisweep.md`** — look at each resonance closely: one narrow sweep
  per resonator, all of them in parallel, and then the same array over a ladder
  of probe amplitudes.
- **`Demos/fitting_resonators.md`** — turn those sweeps into numbers. The three
  resonator models, where their results land in the results dictionary, and what
  the fitted parameters do as you drive a detector harder.

Then:

- **`Demos/simplified_tuning_flow.md`** — the whole chain end to end: sweep,
  find and fit the resonators, park the carriers, measure the noise.
- **`Demos/pulse_capture.md`** — detect and record detector pulses, with
  streaming HDF5, histograms and matched slow+fast capture.

`simplified_tuning_flow` and `pulse_capture` have an unattended `.py`
counterpart beside them for cron jobs and smoke tests; the notebook is the
documentation, the script is the runner. Both notebooks and
`simplified_tuning_flow.py` run in the acquisition tier; `pulse_capture_flow.py`
does not, so run it by hand after changing its notebook.

## Connecting

```python
import rfmux
```

```python
s = rfmux.load_session('!HardwareMap [ !CRS { serial: "0033" } ]') # Replace with your board serial
crs = s.query(rfmux.CRS).one()
await crs.resolve()
await crs.set_timestamp_port(crs.TIMESTAMP_PORT.TEST)  # the fast stream needs a timestamp source
```

No hardware? Every demo above stands up a simulated CRS instead — see their
mock-mode sections.
