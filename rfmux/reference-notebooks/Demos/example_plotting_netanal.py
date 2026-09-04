#!/usr/bin/env python3
"""Example plots for network analyses and resonance searches.

Three plots, in the order you would want them::

    import example_plotting_netanal as naplots

    netanal = await crs.take_netanal(amp=0.001, fmin=0.6e9, fmax=1.05e9,
                                     npoints=20_000, module=1)
    naplots.plot_netanal(netanal)                 # what was measured

    found = find_resonances_in_netanal(netanal)
    naplots.plot_resonance_search(found)          # where the finder put dips
    naplots.plot_candidate_details(found)         # what it measured at each

The third is the one to reach for when a sweep gives you a count you did not
expect. It draws the samples as points and the finder's own numbers on top, so
an unresolved dip — two samples across a resonance the netanal was too coarse
to see — looks like what it is rather than like a missing resonator.

Every plot takes any of the shapes its producer returns: one trace, a list of
them, or a dict keyed by module, drawing a figure for each. So the same call
works whether the sweep ran on one module or eight.

Styling follows hidfmux's plotting modules — large type, a grid on every axes,
compact bracketed axis labels, and generous panels. It lives in ``PLOT_STYLE``
and is applied per figure, not to your session. The per-candidate grid is drawn
in batches of ``BATCH_SIZE``, so pointing it at a thousand-candidate search
gives readable figures rather than one that is metres across.

These are meant to be read and copied, and each ``example_plotting_*`` module
here stands alone: the small amount of layout and style bookkeeping is repeated
in each rather than shared, so that one file is the whole story and lifting a
function out of it is a copy-paste.
"""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from rfmux.tuning import magnitude_db

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


# Type big enough to read on a projector, and a grid on every axes: these plots
# get shown to other people, and a netanal without a grid is hard to read a
# frequency off. Applied per figure through ``plt.rc_context`` rather than
# written into ``plt.rcParams`` at import, so importing this module does not
# quietly restyle the rest of your notebook. If you *want* it everywhere::
#
#     plt.rcParams.update(example_plotting_netanal.PLOT_STYLE)
#
PLOT_STYLE = {
    "font.size": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 14,
    "axes.grid": True,
    # A frequency axis otherwise reads "+6.014e8" in the corner with 0.1, 0.2,
    # … on the ticks, which is unreadable at a glance.
    "axes.formatter.useoffset": False,
    "axes.formatter.use_mathtext": True,
    # Deliberately *not* setting axes.formatter.limits here. Every axis in this
    # module is already in readable units — MHz, dB, degrees, kHz offsets — and
    # forcing a shared exponent turns a 600–1050 MHz axis into "0.6 … 1.0
    # x10^3", which is worse than what it replaced.
}

# Candidates per figure in the per-candidate grid. A panel is sized to be read
# rather than to fit, so a whole array in one figure would be metres across.
BATCH_SIZE = 50

FOUND_COLOUR = "red"
REJECTED_COLOUR = "darkorange"


def panels_per_row(count, few=5, many=7):
    """How many panels to put in a row, for a grid of *count* of them.

    A search over a whole band turns up a lot of candidates, and the useful
    shape is not the same at four as at four hundred: a handful go in one row,
    a moderate number in rows of five, and a big grid in rows of seven, which
    is about as wide as stays legible.
    """
    if count > 30:
        return many
    if count < 10:
        return count
    return few


def labelled_traces(result, what):
    """``{label: result}``, whatever shape its producer handed back.

    ``take_netanal`` and ``find_resonances_in_netanal`` both return a single
    result for one module, a list for a list of sweeps, and a dict keyed by
    module number for several modules. Rather than make you unpack that before
    plotting, every plotter here runs this first and then draws a figure per
    entry.
    """
    if isinstance(result, dict) and result and all(
        isinstance(key, (int, np.integer)) for key in result
    ):
        return {f"module {module}": entry for module, entry in result.items()}
    if isinstance(result, (list, tuple)):
        return {f"{what} {i}": entry for i, entry in enumerate(result)}
    return {"": result}


def _batches(items, batchlen):
    """*items* cut into chunks of at most *batchlen*. Falsy means one chunk."""
    if not batchlen or batchlen >= len(items):
        return [items]
    return [items[start:start + batchlen] for start in range(0, len(items), batchlen)]


def _titled(fig, text):
    """A figure title that clears the panel titles under it.

    The layout engine sizes the band it leaves for a figure title as a fraction
    of figure height, which is far too thin for a single row of wide panels —
    the title lands on top of the panel titles. Reserve a fixed band instead,
    so the shape of the grid cannot break it.
    """
    # Wrapped to roughly what the figure is wide enough to hold at the title's
    # type size: a one-panel figure is only a few inches across, and an
    # unwrapped title simply runs off both ends of it.
    columns = max(24, int(fig.get_figwidth() / 0.16))
    lines = textwrap.wrap(text, columns) or [text]
    band = (0.3 + 0.32 * len(lines)) / fig.get_figheight()
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - band))
    fig.suptitle("\n".join(lines), y=1 - band / 2, va="center")


def _netanal_arrays(netanal):
    """The three arrays a netanal carries, with a readable error if it is not one."""
    missing = {"frequencies", "iq_complex"} - set(
        netanal if isinstance(netanal, dict) else {}
    )
    if missing:
        keys = list(netanal) if isinstance(netanal, dict) else type(netanal).__name__
        raise TypeError(
            f"Expected a take_netanal result — a dict with 'frequencies' and "
            f"'iq_complex' — got {keys}. For several modules pass the whole "
            f"dict take_netanal returned; this unpacks it."
        )
    return (
        np.asarray(netanal["frequencies"]),
        np.asarray(netanal["iq_complex"]),
        netanal.get("phase_degrees"),
    )


def plot_netanal(netanal, phase=True, reference=None, figsize=(14.0, 8.0), title=None):
    """|S21| and phase against frequency, one figure per trace measured.

    Args:
        netanal: what ``crs.take_netanal()`` returned — one module's result, a
            list of them, or the dict keyed by module number that a list of
            modules produces. A figure is drawn for each.
        phase: draw the phase in a second panel under the magnitude. Turn it
            off for magnitude alone, which is what the resonance finder sees.
        reference: the magnitude that maps to 0 dB. The default is the median
            of the trace, a robust stand-in for the off-resonance baseline —
            the same default :func:`rfmux.tuning.magnitude_db` uses, so the dB
            axis here matches the one the finder searched.
        figsize: ``(width, height)`` of the whole figure, in inches.
        title: overrides the figure title.

    Raises:
        TypeError: if handed something that is not a netanal result.
    """
    for label, trace in labelled_traces(netanal, "sweep").items():
        frequencies, iq, phase_degrees = _netanal_arrays(trace)

        with plt.rc_context(PLOT_STYLE):
            nrows = 2 if (phase and phase_degrees is not None) else 1
            fig, axes = plt.subplots(
                nrows, 1, figsize=figsize, sharex=True,
                constrained_layout=True, squeeze=False,
            )
            panels = axes[:, 0]

            panels[0].plot(frequencies / 1e6, magnitude_db(iq, reference), lw=1.0)
            panels[0].set_ylabel("|S21| [dB, norm.]")

            if nrows == 2:
                panels[1].plot(frequencies / 1e6, np.asarray(phase_degrees), lw=1.0)
                panels[1].set_ylabel("phase [deg]")

            panels[-1].set_xlabel("frequency [MHz]")

            span = f"{frequencies[0] / 1e6:.1f}–{frequencies[-1] / 1e6:.1f} MHz"
            _titled(fig, title if title is not None else (
                f"network analysis{f' — {label}' if label else ''}: "
                f"{len(frequencies)} points over {span}"
            ))
            plt.show()


def plot_resonance_search(
    search, figsize=(16.0, 6.0), title=None, mark_rejected=True
):
    """The trace the finder searched, with what it found circled.

    Accepted candidates get a circle at the sample they were found on; rejected
    ones get a dashed vertical line, because a finder that returns fewer
    resonances than the array has is much easier to argue with when you can see
    what it threw away. :func:`plot_candidate_details` is where the reasons are.

    Args:
        search: a :class:`~rfmux.tuning.ResonanceSearch`, or the list or
            per-module dict that ``find_resonances_in_netanal`` returns for
            several traces. A figure is drawn for each.
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
    """One panel per candidate, with the depth and width the finder measured.

    The samples are drawn as points, so you can see how much of each dip the
    sweep actually caught. The vertical bar is the dip depth — a prominence
    against the local baseline — and the horizontal bar is the width, at half
    that depth. Both are the finder's own numbers, not a refit, which is what
    makes this the plot for tuning ``min_dip_depth_db`` and the ``Q`` window by
    eye. Neither is a measurement of Q; fitting a multisweep is what gives you
    that.

    Args:
        search: a :class:`~rfmux.tuning.ResonanceSearch`, or the list or
            per-module dict ``find_resonances_in_netanal`` returns.
        include_rejected: draw the rejected candidates too, each with the
            reason it was dropped printed in its panel. This is the reason the
            plot exists when a count comes out wrong.
        span_widths: how many measured widths either side of the candidate to
            show. Widened automatically when that would be too few samples to
            look at.
        ncols: panels per row, or ``None`` to let :func:`panels_per_row` pick.
        panel_size: ``(width, height)`` of one panel, in inches.
        title: overrides the figure title. The batch marker is still appended.
        batchlen: candidates per figure. More than this and several figures are
            drawn rather than one unreadably large one, each labelled with
            which batch it is. ``None`` puts everything in one figure, however
            big that turns out to be.
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
