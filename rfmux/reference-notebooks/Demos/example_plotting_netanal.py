#!/usr/bin/env python3
"""Plot network analyses, resonance searches and candidate measurements.

    from rfmux.tuning import find_resonances_in_netanal
    import example_plotting_netanal as naplots

    naplots.plot_netanal(netanal)
    found = find_resonances_in_netanal(netanal[crs.module[1].index()])
    naplots.plot_resonance_search(found)
    naplots.plot_candidate_details(found)

Plotters accept a single module's result or results keyed by module ID.
Candidate panels show sampled dips with the finder's depth and width markers.
Style is applied per figure; ``batchlen=None`` puts all candidates in one
candidate-detail figure per module.
"""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from rfmux.core.transferfunctions import convert_dacunits_to_dbm, convert_roc_to_dbm
from rfmux.tuning import magnitude_db, netanal_trace

__all__ = [
    "BATCH_SIZE",
    "PLOT_STYLE",
    "FOUND_COLOUR",
    "REJECTED_COLOUR",
    "panels_per_row",
    "labelled_traces",
    "plot_netanal",
    "plot_resonance_search",
    "plot_candidate_details",
]


# Applied per figure through plt.rc_context.
PLOT_STYLE = {
    "font.size": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 14,
    "axes.grid": True,
    # Show absolute tick values.
    "axes.formatter.useoffset": False,
    "axes.formatter.use_mathtext": True,

}

# Candidates per figure.
BATCH_SIZE = 50

FOUND_COLOUR = "red"
REJECTED_COLOUR = "darkorange"


def panels_per_row(count, few=5, many=7):
    """Choose a column count from the number of panels."""
    if count > 30:
        return many
    if count < 10:
        return count
    return few


def labelled_traces(result, what):
    """Wrap a single result with an empty label, or copy results keyed by module ID."""
    if isinstance(result, dict) and "results" not in result and all(
        isinstance(key, str) for key in result
    ):
        return dict(result)
    return {"": result}


def _batches(items, batchlen):
    """*items* cut into chunks of at most *batchlen*. Falsy means one chunk."""
    if not batchlen or batchlen >= len(items):
        return [items]
    return [items[start:start + batchlen] for start in range(0, len(items), batchlen)]


def _titled(fig, text):
    """Wrap the figure title and reserve space above the panel titles."""
    # Wrapped to roughly what the figure is wide enough to hold at the title's
    # type size: a one-panel figure is only a few inches across, and an
    # unwrapped title simply runs off both ends of it.
    columns = max(24, int(fig.get_figwidth() / 0.16))
    lines = textwrap.wrap(text, columns) or [text]
    band = (0.3 + 0.32 * len(lines)) / fig.get_figheight()
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - band))
    fig.suptitle("\n".join(lines), y=1 - band / 2, va="center")


def _netanal_arrays(module_netanal):
    """Read one module's frequency and complex readout-count arrays."""
    try:
        trace = netanal_trace(module_netanal)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"Expected one module's netanal — what take_netanal returned, "
            f"indexed by module. For several modules pass the whole dict it "
            f"returned; this unpacks it. ({e})"
        ) from None
    return (
        np.asarray(trace["frequencies"]),
        np.asarray(trace["iq_counts"]),
    )


def plot_netanal(
    netanal, phase=True, reference=None, figsize=(14.0, 8.0), title=None,
    *, normalize: bool = True,
):
    """|S21| and phase against frequency, one figure per trace measured.

    Args:
        netanal: what ``crs.take_netanal()`` returned — the dict keyed by
            module identifier, or one module's output out of it. A figure is
            drawn for each module in it.
        phase: draw the phase in a second panel under the magnitude.
        reference: optional magnitude in readout counts that maps to 0 dB.
            Overrides drive normalization when supplied. None uses the saved
            drive power, rather than an off-resonance baseline.
        figsize: ``(width, height)`` of the whole figure, in inches.
        title: overrides the figure title.
        normalize: subtract drive power from received power in dBm, using
            each module's saved ``dac_scale_dbm`` and the trace's
            ``sweep_amplitude``. False shows received power in dBm and needs
            no DAC scale. An explicit reference takes precedence.

    Raises:
        TypeError: if handed something that is not a netanal result.
        ValueError: if drive normalization lacks a finite DAC scale or a
            finite, positive drive amplitude.
    """
    for label, module_netanal in labelled_traces(netanal, "sweep").items():
        frequencies, iq = _netanal_arrays(module_netanal)
        if reference is not None:
            magnitude = magnitude_db(iq, reference)
            ylabel = "|S21| [dB, reference-normalized]"
        else:
            magnitude = convert_roc_to_dbm(np.abs(iq))
            ylabel = "received power [dBm]"
            if normalize:
                dac_scale = module_netanal.get("dac_scale_dbm")
                if dac_scale is None or not np.isfinite(dac_scale):
                    raise ValueError(
                        "Drive-referenced dB requires a finite dac_scale_dbm; "
                        "use normalize=False to plot received power in dBm."
                    )
                drive = netanal_trace(module_netanal).get("sweep_amplitude")
                if drive is None or not np.isfinite(drive) or drive <= 0:
                    raise ValueError(
                        "Drive normalization requires a finite, positive amplitude."
                    )
                magnitude -= convert_dacunits_to_dbm(drive, dac_scale)
                ylabel = "|S21| [dB, drive-referenced]"

        with plt.rc_context(PLOT_STYLE):
            nrows = 2 if phase else 1
            fig, axes = plt.subplots(
                nrows, 1, figsize=figsize, sharex=True,
                constrained_layout=True, squeeze=False,
            )
            panels = axes[:, 0]

            panels[0].plot(frequencies / 1e6, magnitude, lw=1.0)
            panels[0].set_ylabel(ylabel)

            if nrows == 2:
                panels[1].plot(frequencies / 1e6, np.degrees(np.angle(iq)), lw=1.0)
                panels[1].set_ylabel("phase [deg]")

            panels[-1].set_xlabel("frequency [MHz]")

            # min/max, not the ends: a downward netanal comes back descending,
            # in the order it was measured, and the band is still the band.
            span = f"{frequencies.min() / 1e6:.1f}–{frequencies.max() / 1e6:.1f} MHz"
            _titled(fig, title if title is not None else (
                f"network analysis{f' — {label}' if label else ''}: "
                f"{len(frequencies)} points over {span}"
            ))
            plt.show()


def plot_resonance_search(
    search, figsize=(16.0, 6.0), title=None, mark_rejected=True
):
    """Plot the searched trace with accepted candidates circled and rejected ones marked.

    Args:
        search: a :class:`~rfmux.tuning.ResonanceSearch`, or a dict of them
            keyed by module identifier if you searched several modules and kept
            them that way. A figure is drawn for each.
        figsize: ``(width, height)`` of the whole figure, in inches.
        title: overrides the figure title.
        mark_rejected: draw the rejected candidates as well as the accepted.
    """
    for label, found in labelled_traces(search, "search").items():
        with plt.rc_context(PLOT_STYLE):
            fig, panel = plt.subplots(figsize=figsize, constrained_layout=True)

            panel.plot(
                found.frequencies_hz / 1e6, found.magnitude_db, lw=0.9, zorder=1,
            )
            if found.candidates:
                indices = [c.index for c in found.candidates]
                panel.scatter(
                    found.frequencies_hz[indices] / 1e6,
                    found.magnitude_db[indices],
                    s=200, facecolor="none", edgecolor=FOUND_COLOUR, lw=2, zorder=3,
                    label=f"found ({len(found.candidates)})",
                )
            if mark_rejected:
                for candidate in found.rejected:
                    panel.axvline(
                        candidate.frequency_hz / 1e6,
                        color=REJECTED_COLOUR, ls="--", lw=1.5, alpha=0.8, zorder=2,
                    )
                if found.rejected:
                    # A proxy handle: one legend entry for the whole set rather
                    # than one per line.
                    panel.add_line(Line2D(
                        [], [], color=REJECTED_COLOUR, ls="--", lw=1.5,
                        label=f"rejected ({len(found.rejected)})",
                    ))

            panel.set_xlabel("frequency [MHz]")
            panel.set_ylabel("|S21| [dB, norm.]")
            if found.candidates or (mark_rejected and found.rejected):
                panel.legend(loc="lower right")

            _titled(fig, title if title is not None else (
                f"resonance search{f' — {label}' if label else ''}: "
                f"{len(found)} found, {len(found.rejected)} rejected"
            ))
            plt.show()


def plot_candidate_details(
    search,
    include_rejected=False,
    span_widths=4.0,
    ncols=None,
    panel_size=(6.0, 5.0),
    title=None,
    batchlen=BATCH_SIZE,
):
    """Plot sampled dips with measured depth and half-depth width markers.

    Markers use the resonance finder's measurements. They are not fitted Q values.

    Args:
        search: a :class:`~rfmux.tuning.ResonanceSearch`, or a dict of them
            keyed by module identifier.
        include_rejected: show rejected candidates and their rejection reasons.
        span_widths: how many measured widths either side of the candidate to
            show. Widened automatically when that would be too few samples to
            look at.
        ncols: panels per row, or ``None`` to let :func:`panels_per_row` pick.
        panel_size: ``(width, height)`` of one panel, in inches.
        title: overrides the figure title. The batch marker is still appended.
        batchlen: candidates per figure; None uses one figure.
    """
    for label, found in labelled_traces(search, "search").items():
        shown = list(found.candidates)
        if include_rejected:
            # In frequency order, so a rejected candidate sits next to the
            # accepted ones it was competing with rather than after all of them.
            shown = sorted(shown + list(found.rejected), key=lambda c: c.frequency_hz)
        if not shown:
            raise ValueError(
                f"This search{f' ({label})' if label else ''} has no candidates "
                f"to draw. It rejected {len(found.rejected)}"
                + (", which include_rejected=True would show."
                   if found.rejected else ", and found nothing.")
            )

        spacing = float(np.mean(np.diff(found.frequencies_hz)))
        batches = _batches(shown, batchlen)
        # One column count for every figure, taken from a full batch, so that
        # a short final batch is drawn at the same width as the ones before it.
        columns = ncols if ncols is not None else panels_per_row(len(batches[0]))
        columns = min(columns, len(batches[0]))

        for batch_number, batch in enumerate(batches, start=1):
            _draw_candidate_batch(
                batch, found, spacing, span_widths, columns, panel_size,
                _batch_title(
                    title, label, len(shown), len(found), found, batch_number,
                    len(batches),
                ),
            )


def _batch_title(title, label, shown, accepted, found, batch_number, batch_count):
    """What goes above the figure, plus which batch of how many it is."""
    if title is None:
        title = (
            f"candidates{f' — {label}' if label else ''}: {shown} drawn, "
            f"{accepted} accepted, {len(found.rejected)} rejected"
        )
    if batch_count > 1:
        return f"{title}  [batch {batch_number} of {batch_count}]"
    return title


def _draw_candidate_batch(
    batch, found, spacing, span_widths, columns, panel_size, title
):
    """One figure, holding one batch of candidates."""
    with plt.rc_context(PLOT_STYLE):
        nrows = -(-len(batch) // columns)  # ceiling division, no import needed
        fig, axes = plt.subplots(
            nrows, columns,
            figsize=(panel_size[0] * columns, panel_size[1] * nrows),
            constrained_layout=True, squeeze=False,
        )
        panels = axes.ravel()
        for spare in panels[len(batch):]:
            spare.set_visible(False)

        for panel, candidate in zip(panels, batch):
            rejected = candidate.rejected_because is not None
            colour = REJECTED_COLOUR if rejected else FOUND_COLOUR

            # A window a few widths wide, but never so few samples that there
            # is nothing to look at — an unresolved dip is exactly the case
            # this plot exists to show.
            half = max(int(np.ceil(span_widths * candidate.width_hz / spacing)), 8)
            low = max(candidate.index - half, 0)
            high = min(candidate.index + half + 1, len(found.frequencies_hz))
            panel.plot(
                (found.frequencies_hz[low:high] - candidate.frequency_hz) / 1e3,
                found.magnitude_db[low:high],
                ".-", ms=8, lw=1.2,
            )

            # Depth and width as the finder measured them. The width bar is
            # drawn centred; find_peaks' two crossings are usually a little
            # asymmetric.
            floor = found.magnitude_db[candidate.index]
            panel.vlines(0.0, floor, floor + candidate.depth_db, color=colour, lw=2.5)
            panel.hlines(
                floor + candidate.depth_db / 2,
                -candidate.width_hz / 2e3, candidate.width_hz / 2e3,
                color=colour, lw=2.5,
            )

            panel.set_title(
                f"{candidate.frequency_hz / 1e6:.3f} MHz", color=colour,
            )
            note = (
                f"{candidate.depth_db:.1f} dB deep\n"
                f"{candidate.width_hz / 1e3:.1f} kHz wide"
            )
            if rejected:
                # The reason is a sentence, so wrap it rather than let it run
                # off the panel.
                note += "\n" + "\n".join(
                    textwrap.wrap(candidate.rejected_because, 34)
                )
            panel.text(
                0.04, 0.06, note, transform=panel.transAxes, fontsize=12,
                va="bottom",
            )

        # Axis labels on the outer edge only: repeating them in every panel of
        # a 50-candidate grid costs more room than the panels themselves. The x
        # label goes on the lowest *visible* panel of each column, which is not
        # the bottom row when the candidate count does not fill the grid.
        for column in range(axes.shape[1]):
            visible = [panel for panel in axes[:, column] if panel.get_visible()]
            if visible:
                visible[-1].set_xlabel("$f - f_\\mathrm{candidate}$ [kHz]")
        for panel in axes[:, 0]:
            if panel.get_visible():
                panel.set_ylabel("|S21| [dB, norm.]")

        if any(c.rejected_because is not None for c in batch):
            fig.legend(
                [Line2D([], [], color=FOUND_COLOUR, lw=2.5),
                 Line2D([], [], color=REJECTED_COLOUR, lw=2.5)],
                ["accepted", "rejected"],
                loc="outside lower center", ncols=2,
            )

        _titled(fig, title)
        plt.show()
