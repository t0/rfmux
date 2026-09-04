#!/usr/bin/env python3
"""Example plots for bias finding: where the tone went, and why.

``find_bias_points`` reads a multi-amplitude sweep and returns a
:class:`~rfmux.tuning.BiasReport` — a catalog carrying the operating points,
plus one finding per resonator recording how each was arrived at. These plots
draw the report against the sweeps it came from::

    import example_plotting_bias as biasplots

    module_sweeps = sweeps[crs.module[1].index()]
    report = find_bias_points(module_sweeps)

    biasplots.plot_bias_points(report, module_sweeps)        # where the tone sits
    biasplots.plot_bifurcation_checks(report, module_sweeps) # why that amplitude
    biasplots.plot_arc_speed_panels(module_sweeps)           # what the tests saw

The report carries the conclusions, not the traces, so the first two want the
sweeps as well as the report.

The middle one is the plot to reach for before trusting either bifurcation
detector. Both defaults ship uncalibrated against a real array — a
``BifurcationCheck`` carries its ``metric`` and its ``threshold`` precisely so
that the margin between them can be read off across the amplitude steps of a
resonator known to bifurcate, and this draws exactly that.

A bias point that is a *default* rather than a measurement — nothing
bifurcated, so the loudest step won by being loudest; or the quietest step was
already bifurcated, so there was nothing to fall back to — is flagged on the
report, and drawn here in a warning colour with the reason on it. Those points
are usable and are the best available, but they are not what the analysis set
out to find, and a plot that drew them like the others would be hiding the one
thing worth knowing.

Styling follows hidfmux's plotting modules — large type, a grid on every axes,
compact bracketed axis labels, and generous panels. It lives in ``PLOT_STYLE``
and is applied per figure, not to your session. Panels are drawn in batches of
``BATCH_SIZE`` resonators, so pointing these at a whole array gives readable
figures rather than one that is metres across.

These are meant to be read and copied, and each ``example_plotting_*`` module
here stands alone: the small amount of layout and style bookkeeping is repeated
in each rather than shared, so that one file is the whole story and lifting a
function out of it is a copy-paste.
"""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D

from rfmux.tuning import (
    bifurcated_by_derivative,
    collect_amplitude_iterations_for,
    iq_arc_speed,
    normalized_arc_speed,
)

__all__ = [
    "AMPLITUDE_CMAP",
    "ARC_QUANTITIES",
    "BATCH_SIZE",
    "BIAS_COLOUR",
    "FLAGGED_COLOUR",
    "PLOT_STYLE",
    "PREFERRED_DIRECTION",
    "amplitude_mappable",
    "offset_khz",
    "panels_per_row",
    "plot_arc_speed_panels",
    "plot_bias_points",
    "plot_bifurcation_checks",
    "square_axes",
]


# Type big enough to read on a projector, and a grid on every axes: these plots
# get shown to other people, and the bifurcation plot exists to have a number
# read off it. Applied per figure through ``plt.rc_context`` rather than
# written into ``plt.rcParams`` at import, so importing this module does not
# quietly restyle the rest of your notebook. If you *want* it everywhere::
#
#     plt.rcParams.update(example_plotting_bias.PLOT_STYLE)
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
    # Arc speeds run to 1e-7 per hertz and IQ counts to 1e6, so a shared
    # exponent per axis beats either six digits or six decimal places. Without
    # it the choice is made per axes and neighbouring panels disagree.
    "axes.formatter.limits": (-3, 3),
}

# Resonators per figure. A panel is sized to be read rather than to fit.
BATCH_SIZE = 50

# gnuplot runs black -> purple -> red -> orange -> yellow, so it stays
# saturated from end to end and every trace reads against a white background.
AMPLITUDE_CMAP = plt.cm.gnuplot

BIAS_COLOUR = "royalblue"  # the chosen operating point
FLAGGED_COLOUR = "darkorange"  # a bias point that is a default, not a finding

# Mirrors ``rfmux.tuning.bias.PREFERRED_DIRECTION``: which direction a bias
# frequency is measured on when a step has both and the caller did not say.
PREFERRED_DIRECTION = "upward"

# What can be drawn against frequency, and what each one is *for*. The point of
# offering three is that they are what the two methods actually looked at:
# picking a threshold off a re-derivation of the quantity rather than the
# quantity itself is how a plot comes to disagree with the code.
ARC_QUANTITIES = {
    "arc_speed": {
        "reader": iq_arc_speed,
        "label": "$|dI + jdQ|/df$ [counts/Hz]",
        "what": "what the iq_derivative frequency method maximizes",
        # Its maximum is the answer, so mark it.
        "annotation": "maximum",
    },
    "normalized_speed": {
        "reader": normalized_arc_speed,
        "label": "normalized speed [1/Hz]",
        "what": "what the derivative bifurcation test differentiates",
        # Nothing here is compared against a threshold — the test's bars apply
        # to the *difference* of this, which is the "spikes" quantity below.
        "annotation": None,
    },
    "spikes": {
        "reader": None,  # np.diff of normalized_arc_speed — see _arc_quantity
        "label": "$\\Delta$ normalized speed",
        "what": "what the derivative test looks for spikes in",
        # The bar a spike has to clear, per amplitude step.
        "annotation": "threshold",
    },
}


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


def amplitude_mappable(amplitudes, cmap=AMPLITUDE_CMAP):
    """A colour scale graded over *amplitudes*, and the colourbar's handle.

    Colour a trace with ``mappable.to_rgba(sweep["sweep_amplitude"])`` rather
    than by its position in the ladder. Under a *multiplicative* amplitude
    schedule each resonator is driven at its own amplitude on the same step, so
    step number and drive are not the same thing.

    Log-scaled, because an amplitude schedule is log-spaced by default and a
    linear scale bunches every quiet step into one shade. A schedule that
    reaches zero cannot be log-scaled at all, so that one falls back to linear.
    """
    low, high = min(amplitudes), max(amplitudes)
    if high <= low:
        low, high = low * 0.9, high * 1.1
    norm = LogNorm(vmin=low, vmax=high) if low > 0 else Normalize(vmin=low, vmax=high)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def offset_khz(entry, frequencies=None):
    """Frequencies as kHz either side of where the sweep was centred.

    Pass *frequencies* to convert a grid other than the entry's own — the
    midpoint grid a point-to-point difference lands on, for instance.
    """
    if frequencies is None:
        frequencies = entry["frequencies"]
    return (np.asarray(frequencies) - entry["original_center_frequency"]) / 1e3


def square_axes(panel):
    """Equal scale on both axes, so a circle is drawn as a circle.

    ``adjustable="datalim"`` rather than a fixed box: matplotlib's layout
    engines place fixed-aspect axes after they have finished, so the room
    reserved for a figure title is taken back and the title lands on top of the
    panel titles. Fixing the limits instead gives the same guarantee about the
    data — one unit of I is one unit of Q.
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
    of figure height, which is far too thin for a single row of wide panels —
    the title lands on top of the panel titles. Reserve a fixed band instead.
    """
    # Wrapped to roughly what the figure is wide enough to hold at the title's
    # type size: a one-panel figure is only a few inches across, and an
    # unwrapped title simply runs off both ends of it.
    columns = max(24, int(fig.get_figwidth() / 0.16))
    lines = textwrap.wrap(text, columns) or [text]
    band = (0.3 + 0.32 * len(lines)) / fig.get_figheight()
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - band))
    fig.suptitle("\n".join(lines), y=1 - band / 2, va="center")


def _panel_grid(count, columns, panel_size):
    """A grid of *columns* columns big enough for *count* panels, panels flat."""
    nrows = -(-count // columns)  # ceiling division, no import needed
    fig, axes = plt.subplots(
        nrows, columns,
        figsize=(panel_size[0] * columns, panel_size[1] * nrows),
        constrained_layout=True, squeeze=False,
    )
    panels = axes.ravel()
    for spare in panels[count:]:
        spare.set_visible(False)
    return fig, axes, panels[:count]


def _outer_labels(axes, xlabel, ylabel):
    """Axis labels on the outer edge only.

    Repeating them in every panel of a 40-resonator grid costs more room than
    the panels themselves. The x label goes on the lowest *visible* panel of
    each column, which is not the bottom row when the count does not fill the
    grid.
    """
    for column in range(axes.shape[1]):
        visible = [panel for panel in axes[:, column] if panel.get_visible()]
        if visible:
            visible[-1].set_xlabel(xlabel)
    for panel in axes[:, 0]:
        if panel.get_visible():
            panel.set_ylabel(ylabel)


def _columns_for(batches, ncols):
    """One column count for every figure, taken from a full batch.

    So that a short final batch is drawn at the same width as the ones before
    it, rather than stretching a handful of panels across the row.
    """
    return min(
        ncols if ncols is not None else panels_per_row(len(batches[0])),
        len(batches[0]),
    )


def _batch_title(title, what, count, batch_number, batch_count):
    """What goes above the figure, plus which batch of how many it is."""
    if title is None:
        title = f"{what}: {count} resonator{'s' if count != 1 else ''}"
    if batch_count > 1:
        return f"{title}  [batch {batch_number} of {batch_count}]"
    return title


def _section_names(results):
    """Every sweep section in one module's results, in the order measured."""
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


def _findings(report, names):
    """The findings to draw, in the report's own order."""
    wanted = _as_list(names)
    if wanted is None:
        findings = list(report.findings)
    else:
        # report[name] raises a KeyError naming the miss, which is what we want
        findings = [report[name] for name in wanted]
    if not findings:
        raise ValueError("This report has no findings, so there is nothing to draw.")
    return findings


def _direction_for(report, results, direction):
    """Which sweep direction to draw against.

    The report records the ``direction`` it was given, which is ``None`` when
    the caller let :func:`find_bias_points` pick — so fall back the same way it
    does, to the preferred direction if it was swept and to whatever was swept
    otherwise.
    """
    if direction is not None:
        return direction
    recorded = (getattr(report, "settings", None) or {}).get("direction")
    if recorded is not None:
        return recorded

    swept = [
        d
        for by_direction in results["results"].values()
        for d in by_direction
    ]
    return PREFERRED_DIRECTION if PREFERRED_DIRECTION in swept else swept[0]


def _entry_for(results, name, iteration, direction):
    """The one sweep a finding came off, with a readable error if it is absent."""
    try:
        return results["results"][iteration][direction][name]
    except (KeyError, TypeError):
        raise ValueError(
            f"The sweeps passed in do not hold {name!r} at amplitude step "
            f"{iteration} swept {direction!r}, which is where this bias point "
            f"came from. These are the sweeps of a different measurement than "
            f"the report was made from."
        ) from None


def plot_bias_points(
    report,
    results,
    projection="magnitude",
    names=None,
    direction=None,
    ncols=None,
    panel_size=None,
    title=None,
    batchlen=BATCH_SIZE,
):
    """The sweep each bias point was chosen from, with the tone marked on it.

    One panel per resonator, showing the amplitude step the search settled on
    and where in it the tone was placed. A resonator whose bias point is a
    default rather than a measurement is drawn in :data:`FLAGGED_COLOUR` with
    the reason on the panel.

    Args:
        report: what :func:`~rfmux.tuning.find_bias_points` returned.
        results: the same module's sweep results the report was made from —
            the value of ``sweeps[module_id]``. The report carries the
            conclusions, not the traces, so both are needed.
        projection: ``"magnitude"`` draws ``|S21|`` against frequency with the
            bias frequency as a vertical line; ``"iq"`` draws the loop with the
            bias point marked on it. The magnitude view says whether the tone
            is on the dip; the IQ view says how much loop there is to move
            along, which is the other half of the question.
        names: which resonators to draw. A name, a list of names, or ``None``
            for every finding in the report.
        direction: which sweep direction to draw. ``None`` follows what the
            report was run with.
        ncols: panels per row, or ``None`` to let :func:`panels_per_row` pick.
        panel_size: ``(width, height)`` of one panel, in inches. The default is
            square for the IQ projection.
        title: overrides the figure title. The batch marker is still appended.
        batchlen: resonators per figure. ``None`` for one figure however big.

    Raises:
        KeyError: if a requested name has no finding.
        TypeError: if handed the whole per-module container as *results*.
        ValueError: for an unknown projection, or if the sweeps passed in are
            not the ones the report was made from.
    """
    if projection not in ("magnitude", "iq"):
        raise ValueError(
            f"Unknown projection {projection!r}. This draws 'magnitude' or 'iq'."
        )
    if panel_size is None:
        panel_size = (6.0, 6.0) if projection == "iq" else (7.0, 5.0)

    findings = _findings(report, names)
    _section_names(results)  # raises the useful error for the wrong container
    swept_direction = _direction_for(report, results, direction)

    batches = _batches(findings, batchlen)
    columns = _columns_for(batches, ncols)

    for batch_number, batch in enumerate(batches, start=1):
        with plt.rc_context(PLOT_STYLE):
            fig, axes, panels = _panel_grid(len(batch), columns, panel_size)

            for panel, finding in zip(panels, batch):
                entry = _entry_for(
                    results, finding.name, finding.iteration, swept_direction
                )
                colour = FLAGGED_COLOUR if not finding.good else BIAS_COLOUR
                iq = np.asarray(entry["iq_counts"])

                if projection == "magnitude":
                    panel.plot(offset_khz(entry), 20 * np.log10(np.abs(iq)),
                               lw=1.5, color="0.35")
                    panel.axvline(
                        (finding.frequency_hz - entry["original_center_frequency"])
                        / 1e3,
                        color=colour, lw=2.5,
                    )
                else:
                    panel.plot(iq.real, iq.imag, lw=1.5, color="0.35")
                    # Where the tone sits on the loop: the measured sample
                    # nearest the chosen frequency, since the loop is only
                    # known at the points that were measured.
                    nearest = int(
                        np.argmin(np.abs(
                            np.asarray(entry["frequencies"]) - finding.frequency_hz
                        ))
                    )
                    panel.plot(iq.real[nearest], iq.imag[nearest],
                               marker="o", ms=14, mfc="none", mew=3, color=colour)
                    square_axes(panel)

                panel.set_title(
                    f"{finding.name}  {finding.amplitude:.4g}", color=colour,
                )
                note = (
                    f"step {finding.iteration}\n"
                    f"{finding.frequency_hz / 1e6:.6f} MHz"
                )
                if finding.bifurcated_at is not None:
                    note += f"\nbifurcated at {finding.bifurcated_at:.4g}"
                if not finding.good:
                    note += "\n" + "\n".join(
                        textwrap.wrap(finding.flagged_because, 32)
                    )
                panel.text(0.03, 0.03, note, transform=panel.transAxes,
                           fontsize=11, va="bottom", color=colour)

            if projection == "magnitude":
                _outer_labels(axes, "$f - f_\\mathrm{centre}$ [kHz]", "|S21| [dB]")
            else:
                _outer_labels(axes, "I [counts]", "Q [counts]")

            flagged = sum(1 for f in batch if not f.good)
            handles = [Line2D([], [], color=BIAS_COLOUR, lw=2.5)]
            labels = ["bias point"]
            if flagged:
                handles.append(Line2D([], [], color=FLAGGED_COLOUR, lw=2.5))
                labels.append("flagged — a default, not a measurement")
            fig.legend(handles, labels, loc="outside lower center",
                       ncols=len(handles))

            _titled(fig, _batch_title(
                title,
                f"bias points, {swept_direction} (panel titles are the "
                f"chosen amplitude)",
                len(findings), batch_number, len(batches),
            ))
            plt.show()


def plot_bifurcation_checks(
    report,
    results,
    names=None,
    direction=None,
    ncols=None,
    panel_size=(7.0, 5.0),
    title=None,
    batchlen=BATCH_SIZE,
):
    """Each bifurcation test's metric against its threshold, step by step.

    One panel per resonator: the metric the detector computed at every
    amplitude step it examined, the threshold it was compared against, and a
    line at the amplitude that was chosen. Where the two curves cross is where
    the detector fired, and how far apart they are everywhere else is the
    margin — which is the number to read off before quoting either detector's
    defaults as a recommendation.

    Both series are in the detector's own units, so they belong on one axes.
    ``derivative`` reports the larger of its two bars as the threshold, since a
    spike has to clear both.

    Note that a detector stops examining steps once it fires, so a resonator
    that bifurcated part-way up the ladder has fewer points here than it has
    amplitude steps. That is the search being efficient, not data missing. A
    single-amplitude sweep gives one point per panel, which still answers the
    only question there is to ask of it — whether that one step was already
    bifurcated.

    Args:
        report: what :func:`~rfmux.tuning.find_bias_points` returned.
        results: the same module's sweep results the report was made from. A
            finding records only the amplitude it *chose*, so the drive each
            examined step sat at is read back off the sweeps — which is the
            axis this plot is worth having.
        names: which resonators to draw. ``None`` for every finding.
        direction: which sweep direction to read amplitudes off. ``None``
            follows what the report was run with.
        ncols: panels per row, or ``None`` to let :func:`panels_per_row` pick.
        panel_size: ``(width, height)`` of one panel, in inches.
        title: overrides the figure title. The batch marker is still appended.
        batchlen: resonators per figure. ``None`` for one figure however big.

    Raises:
        KeyError: if a requested name has no finding.
        TypeError: if handed the whole per-module container as *results*.
        ValueError: if no finding has any checks to draw.
    """
    findings = _findings(report, names)
    if not any(f.checks for f in findings):
        raise ValueError(
            "None of these findings recorded a bifurcation check, so there is "
            "nothing to draw. Every finding from find_bias_points carries at "
            "least one, so a report whose findings carry none did not come "
            "from an amplitude search."
        )

    _section_names(results)  # raises the useful error for the wrong container
    swept_direction = _direction_for(report, results, direction)
    method = (getattr(report, "settings", None) or {}).get(
        "amplitude_method", "bifurcation"
    )
    batches = _batches(findings, batchlen)
    columns = _columns_for(batches, ncols)

    for batch_number, batch in enumerate(batches, start=1):
        with plt.rc_context(PLOT_STYLE):
            fig, axes, panels = _panel_grid(len(batch), columns, panel_size)

            for panel, finding in zip(panels, batch):
                checks = finding.checks
                if not checks:
                    panel.text(0.5, 0.5, "no steps examined", ha="center",
                               va="center", transform=panel.transAxes,
                               color=FLAGGED_COLOUR)
                    panel.set_title(finding.name)
                    continue

                # The checks are keyed by amplitude step, but the x axis that
                # means anything is the drive each step actually used.
                steps = sorted(checks)
                amplitudes = [
                    _entry_for(results, finding.name, step, swept_direction)[
                        "sweep_amplitude"
                    ]
                    for step in steps
                ]
                metrics = [checks[step].metric for step in steps]
                thresholds = [checks[step].threshold for step in steps]

                panel.plot(amplitudes, metrics, marker="o", ms=8, lw=2,
                           color="crimson")
                panel.plot(amplitudes, thresholds, marker="s", ms=8, lw=2,
                           ls="--", color="0.35")

                # Where it fired, and where the search settled — one step
                # below, which is the whole point of the search.
                fired = [step for step in steps if checks[step].bifurcated]
                if fired:
                    panel.axvline(amplitudes[steps.index(fired[0])],
                                  color=FLAGGED_COLOUR, lw=2.5, alpha=0.8)
                panel.axvline(finding.amplitude, color=BIAS_COLOUR, lw=2.5,
                              alpha=0.8)

                panel.set_xscale("log")
                panel.set_yscale("log")
                colour = FLAGGED_COLOUR if not finding.good else "black"
                panel.set_title(finding.name, color=colour)
                if not finding.good:
                    panel.text(
                        0.03, 0.03,
                        "\n".join(textwrap.wrap(finding.flagged_because, 32)),
                        transform=panel.transAxes, fontsize=11, va="bottom",
                        color=FLAGGED_COLOUR,
                    )

            _outer_labels(axes, "drive amp. [norm.]", f"{method} metric")

            fig.legend(
                [Line2D([], [], color="crimson", lw=2, marker="o"),
                 Line2D([], [], color="0.35", lw=2, ls="--", marker="s"),
                 Line2D([], [], color=BIAS_COLOUR, lw=2.5),
                 Line2D([], [], color=FLAGGED_COLOUR, lw=2.5)],
                ["metric", "threshold", "chosen", "bifurcated"],
                loc="outside lower center", ncols=4,
            )
            _titled(fig, _batch_title(
                title, f"{method} bifurcation checks, {swept_direction}",
                len(findings), batch_number, len(batches),
            ))
            plt.show()


def _arc_quantity(quantity, entry):
    """``(frequencies, values)`` for one of :data:`ARC_QUANTITIES`."""
    if quantity == "spikes":
        # What the detector looks for spikes in is the *difference* of the
        # normalized speed, on the midpoints of the pairs it differenced.
        frequencies, speed = normalized_arc_speed(entry)
        return 0.5 * (frequencies[:-1] + frequencies[1:]), np.diff(speed)
    return ARC_QUANTITIES[quantity]["reader"](entry)


def plot_arc_speed_panels(
    results,
    quantity="arc_speed",
    names=None,
    iterations=None,
    direction=PREFERRED_DIRECTION,
    annotate=True,
    spike_prominence_factor=0.5,
    ncols=None,
    panel_size=(7.0, 5.0),
    title=None,
    batchlen=BATCH_SIZE,
):
    """What the bias-finding tests looked at, one panel per resonator.

    Every amplitude step overlaid and coloured by drive, so that the step where
    a quantity stops looking like the others is visible as such. This is the
    plot for choosing ``spike_prominence_factor`` by eye, and for seeing what
    the frequency method had to pick a maximum from.

    What ``annotate`` draws depends on the quantity, because what each one is
    compared against differs:

    ``arc_speed``
        A marker at each step's maximum — which is the frequency the
        ``iq_derivative`` method returns, so this is its answer.
    ``normalized_speed``
        Nothing. No threshold applies to this directly; the bars apply to its
        *difference*, which is ``spikes``.
    ``spikes``
        A dashed line at ``±threshold`` per amplitude step, in that step's
        colour: the bar a spike has to clear for the step to count as
        bifurcated, which is ``spike_prominence_factor`` times the span of
        that step's arc speed. It is read straight off
        :func:`~rfmux.tuning.bifurcated_by_derivative` rather than recomputed
        here, so the line cannot disagree with the detector. A step whose
        verdict is positive gets a solid line instead of a dashed one. The bar
        scales off the sweep, so it moves when the data does, which is why
        there is one line per step rather than one per panel.

    Two reasons an excursion may poke past a dashed line without the step being
    called bifurcated, and neither is a disagreement. The detector compares
    *prominence* — how far a spike stands out of its own neighbourhood — while
    this draws the curve itself, so a spike sitting on a raised shoulder is
    shorter than it looks against the line. And the up-spike still has to be
    followed immediately by a down-spike, on and back off the jump, so a step
    can clear the bar and be rejected on the pattern.

    Every step drawn is evaluated here, which is not the same set the amplitude
    search examined — it stops at the first step that fires, so the loud end of
    the ladder usually has no recorded check. That is the point of doing it
    again: a threshold is calibrated on the steps past the one that tripped it,
    and :func:`plot_bifurcation_checks` can only show what was recorded.

    Args:
        results: one module's sweep results — the value of
            ``sweeps[module_id]``.
        quantity: which of :data:`ARC_QUANTITIES` to draw. ``"arc_speed"`` is
            what the ``iq_derivative`` frequency method maximizes;
            ``"normalized_speed"`` is what the ``derivative`` bifurcation test
            differentiates; ``"spikes"`` is that difference, which is what it
            actually looks for spikes in.
        names: which resonators to draw. ``None`` for the whole array.
        iterations: which amplitude steps to draw. ``None`` for all of them.
        direction: which sweep direction to draw. One direction, not a list:
            these curves are busy enough with one.
        annotate: draw the per-quantity marks described above.
        spike_prominence_factor: as :func:`~rfmux.tuning.find_bias_points`
            takes it. Only used to place the ``spikes`` thresholds — pass the
            factor you are considering and watch the lines move.
        ncols: panels per row, or ``None`` to let :func:`panels_per_row` pick.
        panel_size: ``(width, height)`` of one panel, in inches.
        title: overrides the figure title. The batch marker is still appended.
        batchlen: resonators per figure. ``None`` for one figure however big.

    Raises:
        KeyError: if a requested name was never swept.
        TypeError: if handed the whole per-module container.
        ValueError: for an unknown quantity, or if the selection matches
            nothing, or for a sweep too short to differentiate.
    """
    if quantity not in ARC_QUANTITIES:
        raise ValueError(
            f"Unknown quantity {quantity!r}. This draws "
            f"{', '.join(sorted(ARC_QUANTITIES))}."
        )

    measured_names = _section_names(results)
    if not measured_names:
        raise ValueError("This measurement holds no sweeps, so there is nothing to draw.")
    wanted_names = _as_list(names) or measured_names
    wanted_iterations = _as_list(iterations)

    entries_by_name = {}
    for name in wanted_names:
        entries = []
        for iteration, by_direction in collect_amplitude_iterations_for(
            results, name
        ).items():
            if wanted_iterations is not None and iteration not in wanted_iterations:
                continue
            if direction in by_direction:
                entries.append((iteration, by_direction[direction]))
        if entries:
            entries_by_name[name] = entries

    if not entries_by_name:
        available = collect_amplitude_iterations_for(results, wanted_names[0])
        directions_swept = sorted(
            {d for by_direction in available.values() for d in by_direction}
        )
        raise ValueError(
            f"Nothing to plot: iterations={iterations!r} direction={direction!r} "
            f"selected none of the sweeps taken. This measurement has iterations "
            f"{list(available)} and directions {directions_swept}."
        )

    mappable = amplitude_mappable([
        entry["sweep_amplitude"]
        for entries in entries_by_name.values()
        for _, entry in entries
    ])
    batches = _batches(list(entries_by_name.items()), batchlen)
    columns = _columns_for(batches, ncols)
    spec = ARC_QUANTITIES[quantity]

    for batch_number, batch in enumerate(batches, start=1):
        with plt.rc_context(PLOT_STYLE):
            fig, axes, panels = _panel_grid(len(batch), columns, panel_size)

            bifurcated_any = False
            for panel, (name, entries) in zip(panels, batch):
                for iteration, entry in entries:
                    colour = mappable.to_rgba(entry["sweep_amplitude"])
                    frequencies, values = _arc_quantity(quantity, entry)
                    panel.plot(
                        offset_khz(entry, frequencies), values, lw=1.5,
                        color=colour,
                    )

                    if not annotate:
                        continue
                    if spec["annotation"] == "maximum":
                        peak = int(np.argmax(values))
                        panel.plot(
                            offset_khz(entry, frequencies)[peak], values[peak],
                            marker="o", ms=13, mfc="none", mew=2.5, color=colour,
                        )
                    elif spec["annotation"] == "threshold":
                        # Straight off the detector, on this one step, so the
                        # line is the bar the code actually applied.
                        check = bifurcated_by_derivative(
                            {direction: entry},
                            spike_prominence_factor=spike_prominence_factor,
                        )
                        bifurcated_any |= check.bifurcated
                        for sign in (1, -1):
                            panel.axhline(
                                sign * check.threshold, color=colour, lw=1.5,
                                ls="-" if check.bifurcated else "--", alpha=0.9,
                            )

                centre_mhz = entries[0][1]["original_center_frequency"] / 1e6
                panel.set_title(f"{name}  {centre_mhz:.3f} MHz")

            _outer_labels(axes, "$f - f_\\mathrm{centre}$ [kHz]", spec["label"])
            fig.colorbar(mappable, ax=axes, label="drive amp. [norm.]")

            if annotate and spec["annotation"] == "threshold":
                handles = [Line2D([], [], color="0.35", lw=1.5, ls="--")]
                labels = ["threshold, per step"]
                if bifurcated_any:
                    handles.append(Line2D([], [], color="0.35", lw=1.5))
                    labels.append("threshold, step called bifurcated")
                fig.legend(handles, labels, loc="outside lower center",
                           ncols=len(handles))
            elif annotate and spec["annotation"] == "maximum":
                fig.legend(
                    [Line2D([], [], ls="none", marker="o", ms=13, mfc="none",
                            mew=2.5, color="0.35")],
                    ["maximum — the frequency iq_derivative returns"],
                    loc="outside lower center",
                )

            _titled(fig, _batch_title(
                title, f"{quantity}, {direction} — {spec['what']}",
                len(entries_by_name), batch_number, len(batches),
            ))
            plt.show()
