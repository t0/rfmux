"""Plot a saved noise module block without a CRS connection.

    block = store.load(path)[module_id]
    plot_iq_panels(block)  # calibration sweeps from the catalog snapshot
    plot_iq_panels(block, sweeps=verification[module_id])
    plot_timestreams(block, units="counts")
    plot_psds(block, stream="pfb", dual_sideband=True)

Each function returns its figures and displays them. Selections and carrier
omission affect plots only. Slow time axes use nominal sample spacing; actual
packet timestamps remain in the saved block. PFB times are relative to each
sequential RPC capture and must not be aligned with slow or other channels.
"""

from collections.abc import Sequence
from typing import Iterator

import matplotlib.pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator
import numpy as np

from rfmux.core.resonators import ResonatorCatalog
from rfmux.core.transferfunctions import VOLTS_PER_ROC
from rfmux.tuning import find_iteration_matching_amplitude

from example_plotting_multisweep import (
    BIAS_AMPLITUDE_RTOL, PLOT_STYLE, _batches, _panel_grid, _titled, square_axes,
)

__all__ = ["plot_iq_panels", "plot_timestreams", "plot_psds"]

IQ_COLORS = ("#3366CC", "#CC6633")


def _records(block: dict, names: Sequence[str] | str | None, stream: str) -> dict:
    if block.get("measurement") != "noise":
        raise ValueError("Pass one noise module block: noise[module_id].")
    if stream not in ("slow", "pfb"):
        raise ValueError("stream must be 'slow' or 'pfb'.")
    if block["results"]["acquisition"]["iq_units"] != "adc_counts":
        raise ValueError("Expected noise IQ in adc_counts.")
    records = block["results"]["resonators"]
    names = list(records) if names is None else [names] if isinstance(names, str) else names
    selected = {name: records[name] for name in names}
    if not selected:
        raise ValueError("Select at least one resonator.")
    for name, record in selected.items():
        if stream not in record:
            raise ValueError(f"{name} has no {stream} capture.")
    return selected


def _factor(units: str) -> float:
    if units not in ("volts", "counts"):
        raise ValueError("units must be 'volts' or 'counts'.")
    return VOLTS_PER_ROC if units == "volts" else 1.


def _panels(records: dict, title: str) -> Iterator[tuple]:
    for batch in _batches(list(records), 12):
        with plt.rc_context(PLOT_STYLE):
            fig, _, axes = _panel_grid(len(batch), min(3, len(batch)), (5., 4.5))
        _titled(fig, title)
        for ax, name in zip(axes, batch):
            ax.set_title(f"{name} · channel {records[name]['channel']}", fontsize=12)
        yield fig, axes, batch


def _bias_sweep(block: dict, name: str, record: dict, sweeps: dict | None,
                direction: str, catalog: ResonatorCatalog | None) -> dict:
    if sweeps is not None:
        if sweeps.get("measurement") != "multisweep" or sweeps["module"] != block["module"]:
            raise ValueError("sweeps must be a multisweep block for the same module.")
        if record["amplitude"] is None:
            raise ValueError(f"{name} has no measured amplitude for sweep matching.")
        by_direction, _ = find_iteration_matching_amplitude(sweeps, name, record["amplitude"])
        sweep = by_direction[direction]
    else:
        if catalog is None:
            raise ValueError("No catalog snapshot; supply sweeps= for the IQ overlay.")
        sweep = catalog[name].bias.bias_sweep
        if sweep is None:
            raise ValueError(f"{name} has no stored bias sweep; supply sweeps=.")
    amplitude = sweep.get("sweep_amplitude")
    if (amplitude is None or record["amplitude"] is None or not np.isclose(
            amplitude, record["amplitude"], rtol=BIAS_AMPLITUDE_RTOL, atol=0)):
        raise ValueError(f"{name}: sweep and measured tone amplitudes must match.")
    return sweep


def plot_iq_panels(
    block: dict, *, sweeps: dict | None = None, names: Sequence[str] | str | None = None,
    stream: str = "slow", units: str = "volts", direction: str = "upward",
    title: str | None = None,
) -> list[plt.Figure]:
    """Overlay noise on a matching-drive sweep and mark its measured tone frequency."""
    records = _records(block, names, stream)
    factor = _factor(units)
    snapshot = block["call_params"].get("catalog")
    catalog = ResonatorCatalog.from_dict(snapshot) if sweeps is None and snapshot else None
    traces = {name: _bias_sweep(block, name, record, sweeps, direction, catalog)
              for name, record in records.items()}
    figures = []
    for fig, axes, batch in _panels(records, title or f"{stream.upper()} noise on bias sweeps"):
        for ax, name in zip(axes, batch):
            record, sweep = records[name], traces[name]
            iq = np.asarray(record[stream]["iq_counts"]) * factor
            curve = np.asarray(sweep["iq_volts"]) * factor / VOLTS_PER_ROC
            ax.plot(curve.real, curve.imag, color="0.4", lw=1, label="bias sweep")
            ax.scatter(iq.real, iq.imag, s=4, alpha=.25, color=IQ_COLORS[0],
                       label="noise samples", rasterized=True)
            ax.plot(iq.real.mean(), iq.imag.mean(), "+", color="C3", ms=10,
                    label="noise mean")
            frequencies = np.asarray(sweep["frequencies"])
            tone = record["tone_frequency_hz"]
            if tone is not None and frequencies.min() <= tone <= frequencies.max():
                order = np.argsort(frequencies)
                point = np.interp(tone, frequencies[order], curve[order])
                ax.plot(point.real, point.imag, "x", color="black", ms=8,
                        label="sweep at measured tone")
            ax.set(xlabel=f"I [{units}]", ylabel=f"Q [{units}]")
            square_axes(ax)
            ax.legend(fontsize=8)
        figures.append(fig)
        plt.show()
    return figures


def plot_timestreams(
    block: dict, *, names: Sequence[str] | str | None = None, stream: str = "slow",
    units: str = "volts", demean: bool = False, title: str | None = None,
) -> list[plt.Figure]:
    """Plot I/Q against nominal elapsed time; optionally subtract each component's mean."""
    records = _records(block, names, stream)
    factor = _factor(units)
    figures = []
    for fig, axes, batch in _panels(records, title or f"{stream.upper()} timestreams"):
        for ax, name in zip(axes, batch):
            data = records[name][stream]
            iq = np.asarray(data["iq_counts"])
            if demean:
                iq = iq - iq.mean()
            iq = iq * factor
            time = (np.asarray(data["time_s"]) if stream == "pfb" else
                    np.arange(len(iq)) / block["results"]["acquisition"]["slow_sample_rate_hz"])
            for values, label, color in zip((iq.real, iq.imag), ("I", "Q"), IQ_COLORS):
                ax.plot(time, values, color=color, lw=.7, label=label)
            ax.set(xlabel="nominal elapsed time [s]",
                   ylabel=f"{'mean-subtracted ' if demean else ''}I/Q [{units}]")
            ax.legend(fontsize=9)
        figures.append(fig)
        plt.show()
    return figures


def plot_psds(
    block: dict, *, names: Sequence[str] | str | None = None, stream: str = "slow",
    dual_sideband: bool = False, title: str | None = None,
) -> list[plt.Figure]:
    """Plot saved PSDs, omitting the carrier and adjacent bins only for display.

    I/Q uses positive frequencies on a log axis. DSB retains signed frequencies
    on a symmetric log axis. No binning, folding or spectral recalibration.
    """
    records = _records(block, names, stream)
    units = block["results"]["acquisition"]["spectrum_units"]
    figures = []
    kind = "dual-sideband" if dual_sideband else "I/Q"
    for fig, axes, batch in _panels(records, title or f"{stream.upper()} {kind} noise spectra"):
        for ax, name in zip(axes, batch):
            data = records[name][stream]
            shared = block["results"]["slow"] if stream == "slow" else data
            freq = np.asarray(shared["freq_dsb" if dual_sideband else "freq_iq"])
            if not len(freq):
                raise ValueError(f"{name} has an empty spectral frequency axis.")
            carrier = int(np.argmin(abs(freq)))
            keep = abs(np.arange(len(freq)) - carrier) > 1
            if dual_sideband:
                ax.plot(freq, np.where(keep, data["psd_dual_sideband"], np.nan), lw=.8)
                spacing = np.min(np.diff(np.unique(freq))) if len(freq) > 1 else 1.
                ax.set_xscale("symlog", linthresh=spacing)
                locator = SymmetricalLogLocator(base=10, linthresh=spacing)
                locator.set_params(numticks=5)
                ax.set_xticks([tick for tick in locator.tick_values(freq.min(), freq.max())
                               if tick == 0 or abs(tick) >= spacing])
                ax.tick_params(axis="x", labelsize=12)
            else:
                keep &= freq > 0
                for component, color in zip(("i", "q"), IQ_COLORS):
                    ax.semilogx(freq[keep], np.asarray(data[f"psd_{component}"])[keep],
                                color=color, label=component.upper(), lw=.8)
                ax.legend(fontsize=9)
            if not np.any(keep):
                ax.text(.5, .5, "no bins beyond carrier neighborhood",
                        ha="center", transform=ax.transAxes, fontsize=10)
            ax.set(xlabel="frequency offset [Hz]", ylabel=f"PSD [{units}]")
        figures.append(fig)
        plt.show()
    return figures
