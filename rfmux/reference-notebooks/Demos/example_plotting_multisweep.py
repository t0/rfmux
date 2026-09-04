#!/usr/bin/env python3
"""Example plots for multisweep data, and a skeleton for your own.

Every function here takes what a sweep macro returned — ``crs.multisweep`` or
``crs.multiamp_multisweep``, one module's value out of the dict they are keyed
by — and draws the whole array at once, a panel per resonator::

    import pickle
    import example_plotting_multisweep as msplots

    with open("my_multisweep.pkl", "rb") as f:
        sweeps = pickle.load(f)
    module_sweeps = sweeps[list(sweeps)[0]]

    msplots.plot_magnitude_panels(module_sweeps)
    msplots.plot_iq_panels(module_sweeps)

The two plotters are the same plot in two projections: ``|S21|`` against
frequency offset, and the IQ loop the sweep traced out. Both stack every
amplitude step the resonator was measured at, colour-coded by drive, and
overlay the two frequency directions as solid and dashed. On a simulated array
the directions lie on top of each other; on real detectors driven hard enough
to bifurcate they part company, and that gap is the thing you are looking for.

All four selection arguments — ``names``, ``iterations``, ``directions`` and
``normalize`` — are there so you can narrow a 100-resonator grid down to the
one you are arguing about without writing a second function.

Whatever you do not narrow is drawn in batches of ``BATCH_SIZE`` resonators,
one figure each, sharing a single colour scale so the batches can be compared.
That is the standing convention for every ``example_plotting_*`` module here,
and it exists so that pointing one of these at a thousand-resonator array
gives you twenty readable figures instead of one that is metres across. Pass
``batchlen=None`` to override it and take the one enormous figure.

Styling follows hidfmux's plotting modules — large type, a grid on every axes,
compact bracketed axis labels, and generous panels — so that a figure from
either code base reads the same way and survives being put on a projector. It
lives in ``PLOT_STYLE`` and is applied per figure, not to your session.

These are meant to be read and copied. They deliberately use nothing but
matplotlib, numpy and the readers in ``rfmux.tuning``, so lifting one into your
own analysis script is a copy-paste and not a dependency. If you want a
different layout, a log x-axis, or your own colours, start from the body of
``_plot_panels`` — the selection and colour bookkeeping above it is the part
that is tedious to get right, and it is shared.
"""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D

from rfmux.tuning import collect_amplitude_iterations_for

__all__ = [
    "AMPLITUDE_CMAP",
    "BATCH_SIZE",
    "DIRECTION_LINESTYLES",
    "PLOT_STYLE",
    "amplitude_mappable",
    "offset_khz",
    "sweep_iq",
    "section_names",
    "panels_per_row",
    "square_axes",
    "plot_magnitude_panels",
    "plot_iq_panels",
]


# gnuplot runs black -> purple -> red -> orange -> yellow, so it stays
# saturated from end to end and every trace reads against a white background.
AMPLITUDE_CMAP = plt.cm.gnuplot

# Direction as line style, so colour is left to mean amplitude and nothing
# else. Anything not listed here falls back to dotted.
DIRECTION_LINESTYLES = {"upward": "-", "downward": "--"}
FALLBACK_LINESTYLE = ":"

# Resonators per figure. A panel is sized to be read rather than to fit, so a
# kilopixel array in one figure would be metres across and take minutes to
# render — this splits it into figures that can actually be looked at. Fifty is
# hidfmux's number, and about as many panels as one figure can carry.
BATCH_SIZE = 50

# Type big enough to read on a projector, and a grid on every axes: these
# plots get shown to other people, and a resonator plot without a grid is hard
# to read a number off. Applied per figure through ``plt.rc_context`` rather
# than written into ``plt.rcParams`` at import, so importing this module does
# not quietly restyle the rest of your notebook. If you *want* it everywhere::
#
#     plt.rcParams.update(example_plotting_multisweep.PLOT_STYLE)
#
PLOT_STYLE = {
    "font.size": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 14,
    "axes.grid": True,
    # An axis whose ticks all sit near 601.4 MHz otherwise reads "+6.014e8" in
    # the corner and 0.1, 0.2, … on the ticks, which is unreadable at a glance.
    "axes.formatter.useoffset": False,
    "axes.formatter.use_mathtext": True,
    # Anything past a thousand gets a shared exponent rather than six-digit
    # ticks. Without this the choice is made per axes, so one IQ panel ends up
    # labelled 600000 and the panel beside it 0.6 with a x10^6 in the corner.
    "axes.formatter.limits": (-3, 3),
}


def amplitude_mappable(amplitudes, cmap=AMPLITUDE_CMAP):
    """A colour scale graded over *amplitudes*, and the colourbar's handle.

    Colour a trace with ``mappable.to_rgba(sweep["sweep_amplitude"])`` rather
    than by its position in the ladder. Under a *multiplicative* amplitude
    schedule each resonator is driven at its own amplitude on the same step, so
    step number and drive are not the same thing, and a colourbar that claims
    to show drive has to be built from the drives.

    Log-scaled, because an amplitude schedule is log-spaced by default and a
    linear scale bunches every quiet step into one shade. A schedule that
    reaches zero cannot be log-scaled at all, so that one falls back to linear.

    Args:
        amplitudes: every drive amplitude that will be drawn, in any order.
        cmap: any matplotlib colormap.

    Returns:
        matplotlib.cm.ScalarMappable: pass it to ``fig.colorbar`` and call
        ``.to_rgba`` on it per trace.
    """
    low, high = min(amplitudes), max(amplitudes)
    if high <= low:
        # One amplitude, or several identical ones: nothing to grade, so widen
        # the range slightly and let every trace land mid-scale.
        low, high = low * 0.9, high * 1.1
    norm = LogNorm(vmin=low, vmax=high) if low > 0 else Normalize(vmin=low, vmax=high)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def offset_khz(sweep):
    """A sweep's frequencies as kHz either side of where it was centred."""
    return (sweep["frequencies"] - sweep["original_center_frequency"]) / 1e3


def sweep_iq(sweep, normalize=True):
    """A sweep's IQ, optionally divided by the drive that produced it.

    Normalized is usually what you want when several amplitude steps share an
    axes: it compares the traces by shape, instead of showing you the loudest
    one sitting on top of the others. Turn it off to see the raw counts.
    """
    if normalize:
        return sweep["iq_counts"] / sweep["sweep_amplitude"]
    return sweep["iq_counts"]


def section_names(results):
    """Every sweep section in one module's results, in the order measured.

    Raises:
        TypeError: if handed the dict a sweep macro returns rather than one
            module's value out of it. That is the easy mistake to make, and it
            is worth a sentence saying so.
    """
    try:
        iterations = results["results"]
    except (TypeError, KeyError):
        keys = list(results) if isinstance(results, dict) else type(results).__name__
        raise TypeError(
            "Expected one module's sweep results — the value of "
            "sweeps[module_id] — rather than the dict a sweep macro returns "
            f"keyed by module identifier. Got {keys}. If there is only one "
            "module in play, sweeps[list(sweeps)[0]] is the thing to pass."
        ) from None

    for by_direction in iterations.values():
        for sections in by_direction.values():
            return list(sections)
    return []


def _as_list(value):
    """One item or many, always a list. ``None`` means "everything"."""
    if value is None:
        return None
    if isinstance(value, (str, int, np.integer)):
        return [value]
    return list(value)


def _collect_traces(results, names, iterations, directions):
    """What was asked for: ``{name: [(iteration, direction, sweep), ...]}``.

    Selection lives here, in one place, so the two plotters cannot drift apart
    in what they accept. Raises rather than drawing an empty grid when the
    selection matches nothing — a blank figure is a much worse way to find out
    you asked for a direction that was never swept.
    """
    # Always called, even when names were given, so that being handed the
    # whole per-module container is caught here with a sentence about it.
    measured_names = section_names(results)
    if not measured_names:
        raise ValueError("This measurement holds no sweeps, so there is nothing to draw.")
    wanted_names = _as_list(names) or measured_names
    wanted_iterations = _as_list(iterations)
    wanted_directions = _as_list(directions)

    collected = {}
    for name in wanted_names:
        traces = []
        # collect_amplitude_iterations_for raises a helpful KeyError naming the
        # sections in play, so a mistyped resonator name is already covered.
        measured = collect_amplitude_iterations_for(results, name)
        for iteration, by_direction in measured.items():
            if wanted_iterations is not None and iteration not in wanted_iterations:
                continue
            for direction in wanted_directions:
                if direction in by_direction:
                    traces.append((iteration, direction, by_direction[direction]))
        if traces:
            collected[name] = traces

    if not collected:
        available = collect_amplitude_iterations_for(results, wanted_names[0])
        directions_swept = sorted(
            {d for by_direction in available.values() for d in by_direction}
        )
        raise ValueError(
            f"Nothing to plot: iterations={iterations!r} directions={directions!r} "
            f"selected none of the sweeps taken. This measurement has "
            f"iterations {list(available)} and directions {directions_swept}."
        )
    return collected


def panels_per_row(count, few=5, many=7):
    """How many panels to put in a row, for a grid of *count* of them.

    A whole array is a lot of panels, and the useful shape is not the same at
    four resonators as at four hundred: a handful go in one row, a moderate
    number in rows of five, and a big grid in rows of seven, which is about as
    wide as stays legible.
    """
    if count > 30:
        return many
    if count < 10:
        return count
    return few


def square_axes(panel):
    """Equal scale on both axes, so a circle is drawn as a circle.

    ``adjustable="datalim"`` rather than hidfmux's ``"box"``: fixing the *box*
    to a square is prettier, but matplotlib's layout engines place fixed-aspect
    axes after they have finished, so the room reserved for a figure title is
    taken back and the title lands on top of the panel titles. Fixing the
    *limits* instead gives the same guarantee about the data — one unit of I is
    one unit of Q — and leaves the layout alone. Keep ``panel_size`` square and
    the panel comes out square too.
    """
    panel.set_aspect("equal", adjustable="datalim")
    # A square panel is narrower than the default tick count assumes, and at
    # this type size the labels run into each other.
    panel.locator_params(axis="both", nbins=5)


def _batches(items, batchlen):
    """*items* cut into chunks of at most *batchlen*. Falsy means one chunk."""
    if not batchlen or batchlen >= len(items):
        return [items]
    return [items[start:start + batchlen] for start in range(0, len(items), batchlen)]


def _titled(fig, text):
    """A figure title that clears the panel titles under it.

    The layout engine sizes the band it leaves for a figure title as a fraction
    of figure height, which is far too thin for a single row of very wide
    panels — the title lands on top of the panel titles. Reserve a fixed band
    instead, so the shape of the grid cannot break it.
    """
    # Wrapped to roughly what the figure is wide enough to hold at the title's
    # type size: a one-panel figure is only a few inches across, and an
    # unwrapped title simply runs off both ends of it.
    columns = max(24, int(fig.get_figwidth() / 0.16))
    lines = textwrap.wrap(text, columns) or [text]
    band = (0.3 + 0.32 * len(lines)) / fig.get_figheight()
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - band))
    fig.suptitle("\n".join(lines), y=1 - band / 2, va="center")


def _panel_grid(count, ncols, panel_size):
    """A grid of *ncols* columns big enough for *count* panels, panels flat.

    Every leftover panel is hidden rather than left as empty axes, which is
    what lets a short final batch keep the same width as the full ones.
    """
    nrows = -(-count // ncols)  # ceiling division, no import needed
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    panels = axes.ravel()
    for spare in panels[count:]:
        spare.set_visible(False)
    return fig, axes, panels[:count]


def _plot_panels(
    results,
    draw,
    xlabel,
    ylabel,
    what,
    names=None,
    iterations=None,
    directions=("upward", "downward"),
    normalize=True,
    ncols=None,
    panel_size=(7.0, 5.0),
    title=None,
    batchlen=BATCH_SIZE,
    equal_aspect=False,
):
    """The grid both plotters draw; *draw* is the only difference between them.

    *draw* is called as ``draw(panel, sweep, colour, linestyle, normalize)``
    and is the whole of what makes this a magnitude plot or an IQ plot.
    *equal_aspect* is the other difference: an IQ loop has to be square or it
    is not a loop, while a magnitude trace has no business being square.
    """
    traces_by_name = _collect_traces(results, names, iterations, directions)
    every_trace = [
        sweep for traces in traces_by_name.values() for _, _, sweep in traces
    ]

    # One colour scale for every figure this call produces, built before the
    # batching: a colourbar that meant something different from one batch to
    # the next would make the batches impossible to compare, which is the
    # whole point of splitting them rather than plotting a subset.
    mappable = amplitude_mappable([sweep["sweep_amplitude"] for sweep in every_trace])
    steps = sorted(
        {iteration for traces in traces_by_name.values() for iteration, _, _ in traces}
    )

    batches = _batches(list(traces_by_name.items()), batchlen)
    # One column count for every figure, taken from a full batch, so that a
    # short final batch is drawn at the same width as the ones before it
    # instead of stretching to fill the row.
    columns = ncols if ncols is not None else panels_per_row(len(batches[0]))
    columns = min(columns, len(batches[0]))

    for batch_number, batch in enumerate(batches, start=1):
        _draw_figure(
            batch, draw, mappable, xlabel, ylabel, normalize, columns, panel_size,
            equal_aspect,
            title=_figure_title(
                title, what, len(traces_by_name), steps, batch_number, len(batches)
            ),
        )


def _figure_title(title, what, section_count, steps, batch_number, batch_count):
    """What goes above the figure, plus which batch of how many it is."""
    if title is None:
        title = (
            f"{what}: {section_count} sweep "
            f"section{'s' if section_count != 1 else ''}, "
            f"{len(steps)} amplitude step{'s' if len(steps) != 1 else ''}"
        )
    if batch_count > 1:
        return f"{title}  [batch {batch_number} of {batch_count}]"
    return title


def _draw_figure(
    batch, draw, mappable, xlabel, ylabel, normalize, columns, panel_size,
    equal_aspect, title,
):
    """One figure, holding one batch of sweep sections."""
    # Every artist below takes its size from the rcParams in force when it is
    # created, so the whole figure is built inside the style.
    with plt.rc_context(PLOT_STYLE):
        fig, axes, panels = _panel_grid(len(batch), columns, panel_size)

        directions_drawn = []
        for panel, (name, traces) in zip(panels, batch):
            for iteration, direction, sweep in traces:
                draw(
                    panel,
                    sweep,
                    mappable.to_rgba(sweep["sweep_amplitude"]),
                    DIRECTION_LINESTYLES.get(direction, FALLBACK_LINESTYLE),
                    normalize,
                )
                if direction not in directions_drawn:
                    directions_drawn.append(direction)

            centre_mhz = traces[0][2]["original_center_frequency"] / 1e6
            panel.set_title(f"{name}  {centre_mhz:.3f} MHz")
            if equal_aspect:
                square_axes(panel)

        # Axis labels on the outer edge only: repeating them in every panel of
        # a 40-resonator grid costs more room than the panels themselves. The
        # x label goes on the lowest *visible* panel of each column, which is
        # not the bottom row when the section count does not fill the grid.
        for column in range(axes.shape[1]):
            visible = [panel for panel in axes[:, column] if panel.get_visible()]
            if visible:
                visible[-1].set_xlabel(xlabel)
        for panel in axes[:, 0]:
            if panel.get_visible():
                panel.set_ylabel(ylabel)

        # One legend entry per direction rather than one per trace, and only
        # when there is a distinction left to make. Outside the panels,
        # because an IQ loop fills its axes and a legend inside would sit on
        # top of the data.
        if len(directions_drawn) > 1:
            handles = [
                Line2D([], [], color="0.3",
                       ls=DIRECTION_LINESTYLES.get(direction, FALLBACK_LINESTYLE))
                for direction in directions_drawn
            ]
            fig.legend(
                handles, directions_drawn,
                loc="outside lower center", ncols=len(handles),
            )

        fig.colorbar(mappable, ax=axes, label="drive amp. [norm.]")

        _titled(fig, title)
        plt.show()


def plot_magnitude_panels(
    results,
    names=None,
    iterations=None,
    directions=("upward", "downward"),
    normalize=True,
    ncols=None,
    panel_size=(7.0, 5.0),
    title=None,
    batchlen=BATCH_SIZE,
):
    """|S21| against frequency offset, a panel per resonator.

    Args:
        results: one module's sweep results — the value of ``sweeps[module_id]``
            for whatever ``multisweep`` or ``multiamp_multisweep`` returned.
        names: which sweep sections to draw. A name, a list of names, or
            ``None`` for the whole array.
        iterations: which amplitude steps to draw. A step number, a list of
            them, or ``None`` for all of them. A plain ``multisweep`` has only
            step 0.
        directions: which frequency directions to draw, as a string or a list.
            Pass ``"upward"`` for one direction only. Directions that were not
            swept are skipped, so the default is safe on a single-direction
            sweep.
        normalize: divide each trace by its own drive amplitude, so the steps
            are compared by shape rather than by which was loudest.
        ncols: panels per row, or ``None`` to let :func:`panels_per_row` pick
            from how many there are.
        panel_size: ``(width, height)`` of one panel, in inches. Generous by
            default, because the type is sized to be read rather than to fit.
        title: overrides the figure title. The batch marker is still appended.
        batchlen: resonators per figure. More than this and the call draws
            several figures rather than one unreadably large one, each labelled
            with which batch it is; every batch shares one colour scale, so
            they can still be compared. ``None`` puts everything in one figure,
            however big that turns out to be.

    Raises:
        KeyError: if a requested name was never swept.
        ValueError: if the selection matches no sweeps at all.
    """

    def draw(panel, sweep, colour, linestyle, normalize):
        iq = sweep_iq(sweep, normalize)
        panel.plot(
            offset_khz(sweep),
            20 * np.log10(np.abs(iq)),
            lw=1.5,
            color=colour,
            ls=linestyle,
        )

    _plot_panels(
        results,
        draw,
        xlabel="$f - f_\\mathrm{centre}$ [kHz]",
        ylabel="|S21| [dB, norm.]" if normalize else "|S21| [dB]",
        what="magnitude",
        names=names,
        iterations=iterations,
        directions=directions,
        normalize=normalize,
        ncols=ncols,
        panel_size=panel_size,
        title=title,
        batchlen=batchlen,
    )


def plot_iq_panels(
    results,
    names=None,
    iterations=None,
    directions=("upward", "downward"),
    normalize=True,
    ncols=None,
    panel_size=(6.0, 6.0),
    title=None,
    batchlen=BATCH_SIZE,
):
    """The IQ loop each sweep traced out, a panel per resonator.

    Same arguments and same selection as :func:`plot_magnitude_panels` — see
    there for what each one does. The only difference is the default panel,
    which is square here: the axes are held to equal scale by
    :func:`square_axes`, so a circle reads as a circle, which is the whole
    point of looking at a resonator this way.
    """

    def draw(panel, sweep, colour, linestyle, normalize):
        iq = sweep_iq(sweep, normalize)
        panel.plot(iq.real, iq.imag, lw=1.5, color=colour, ls=linestyle)

    _plot_panels(
        results,
        draw,
        xlabel="I [norm.]" if normalize else "I [counts]",
        ylabel="Q [norm.]" if normalize else "Q [counts]",
        what="IQ",
        names=names,
        iterations=iterations,
        directions=directions,
        normalize=normalize,
        ncols=ncols,
        panel_size=panel_size,
        title=title,
        batchlen=batchlen,
        equal_aspect=True,
    )
