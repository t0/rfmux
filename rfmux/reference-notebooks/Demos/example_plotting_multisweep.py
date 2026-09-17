#!/usr/bin/env python3
"""Plot magnitude and IQ traces from one module's multisweep.

    import example_plotting_multisweep as msplots

    module_sweeps = sweeps[crs.module[1].index()]
    msplots.plot_magnitude_panels(module_sweeps)
    msplots.plot_iq_panels(module_sweeps)

Each resonator gets a panel, with amplitude steps coloured by drive and sweep
directions distinguished by line style. Select traces with ``names``,
``iterations`` and ``directions``. Magnitude panels mark the starting catalog's
bias frequency and thicken a trace swept at its bias amplitude. Magnitude
defaults to drive-referenced dB; IQ defaults to readout counts divided by the
drive's DAC fraction.

Style is applied per figure. Batches share a colour scale; ``batchlen=None``
puts all resonators in one figure.
"""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.colorbar import Colorbar
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogFormatter

from rfmux.core.resonators import ResonatorCatalog
from rfmux.core.transferfunctions import convert_dacunits_to_dbm, convert_roc_to_dbm
from rfmux.tuning import collect_amplitude_iterations_for

__all__ = [
    "AMPLITUDE_CMAP",
    "BATCH_SIZE",
    "DIRECTION_LINESTYLES",
    "PLOT_STYLE",
    "amplitude_mappable",
    "amplitude_colorbar",
    "offset_khz",
    "sweep_iq",
    "section_names",
    "panels_per_row",
    "square_axes",
    "plot_magnitude_panels",
    "plot_iq_panels",
]


# Truncate pale yellows to keep traces visible on white.
AMPLITUDE_CMAP = LinearSegmentedColormap.from_list(
    "gnuplot_truncated", plt.cm.gnuplot(np.linspace(0.0, 0.9, 256))
)

# Direction as line style, so colour is left to mean amplitude and nothing
# else. Anything not listed here falls back to dotted.
DIRECTION_LINESTYLES = {"upward": "-", "downward": "--"}
FALLBACK_LINESTYLE = ":"

# Resonators per figure.
BATCH_SIZE = 50

# Schedule arithmetic can perturb a DAC fraction by a few last-place bits. A
# relative comparison does not turn that into a large absolute window for the
# small amplitudes used here.
BIAS_AMPLITUDE_RTOL = 1e-6
TRACE_LINEWIDTH = 1.5
BIAS_TRACE_LINEWIDTH = 3.5
BIAS_COLOUR = "0.15"

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
    # Use a shared exponent for large or small values.
    "axes.formatter.limits": (-3, 3),
}


def _decimal_amplitude(value: float, position: float | None = None) -> str:
    """Format a DAC fraction with four significant digits and no exponent."""
    return np.format_float_positional(
        value, precision=4, unique=False, fractional=False, trim="-",
    )


class _DecimalLogFormatter(LogFormatter):
    def __call__(self, value: float, position: float | None = None) -> str:
        # Preserve Matplotlib's minor-label selection across wide log ranges.
        return _decimal_amplitude(value) if super().__call__(value, position) else ""


def amplitude_colorbar(
    fig: plt.Figure, mappable: plt.cm.ScalarMappable, **kwargs,
) -> Colorbar:
    """Draw a normalized-amplitude colourbar with decimal tick labels."""
    bar = fig.colorbar(mappable, format=FuncFormatter(_decimal_amplitude), **kwargs)
    if isinstance(mappable.norm, LogNorm):
        bar.minorformatter = _DecimalLogFormatter()
    return bar


def amplitude_mappable(amplitudes, cmap=AMPLITUDE_CMAP):
    """Return a ScalarMappable for drive amplitudes, usable with ``fig.colorbar``.

    Use ``.to_rgba(amplitude)`` for each trace. The scale is logarithmic for
    positive amplitudes and linear otherwise.
    """
    low, high = min(amplitudes), max(amplitudes)
    if high <= low:
        # One amplitude, or several identical ones: nothing to grade, so widen
        # the range slightly and let every trace land mid-scale.
        low, high = low * 0.9, high * 1.1
    norm = LogNorm(vmin=low, vmax=high) if low > 0 else Normalize(vmin=low, vmax=high)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def offset_khz(sweep):
    """Return frequency offsets from the sweep centre in kHz."""
    return (sweep["frequencies"] - sweep["original_center_frequency"]) / 1e3


def sweep_iq(sweep, normalize=True):
    """Return readout counts, optionally divided by the drive's DAC fraction."""
    if normalize:
        return sweep["iq_counts"] / sweep["sweep_amplitude"]
    return sweep["iq_counts"]


def section_names(results):
    """Read section names from the first sweep step; require a single module block."""
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
    """Select sweeps as ``{name: [(iteration, direction, sweep), ...]}``."""
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
    """Choose a column count from the number of panels."""
    if count > 30:
        return many
    if count < 10:
        return count
    return few


def square_axes(panel):
    """Use equal data scales for I and Q without fixing the axes box."""
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
    """Wrap the figure title and reserve space above the panel titles."""
    # Wrapped to roughly what the figure is wide enough to hold at the title's
    # type size: a one-panel figure is only a few inches across, and an
    # unwrapped title simply runs off both ends of it.
    columns = max(24, int(fig.get_figwidth() / 0.16))
    lines = textwrap.wrap(text, columns) or [text]
    band = (0.3 + 0.32 * len(lines)) / fig.get_figheight()
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - band))
    fig.suptitle("\n".join(lines), y=1 - band / 2, va="center")


def _panel_grid(count, ncols, panel_size):
    """Create a panel grid, hide spare axes, and return figure, axes and panels."""
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
    overlay_bias=False,
):
    """Draw panels with the supplied trace callback.

    The callback receives ``panel, sweep, colour, style, normalize, highlight``.
    Set ``equal_aspect`` for IQ plots.
    """
    traces_by_name = _collect_traces(results, names, iterations, directions)
    bias_points = _catalog_bias_points(results) if overlay_bias else {}
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
            equal_aspect, bias_points,
            title=_figure_title(
                title, what, len(traces_by_name), steps, batch_number, len(batches)
            ),
        )


def _catalog_bias_points(results):
    """Bias points in the catalog snapshot this multisweep was called with."""
    catalog = results.get("call_params", {}).get("catalog")
    if catalog is None:
        return {}
    return {
        resonator.name: resonator.bias
        for resonator in ResonatorCatalog.from_dict(catalog)
    }


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
    equal_aspect, bias_points, title,
):
    """One figure, holding one batch of sweep sections."""
    # Every artist below takes its size from the rcParams in force when it is
    # created, so the whole figure is built inside the style.
    with plt.rc_context(PLOT_STYLE):
        fig, axes, panels = _panel_grid(len(batch), columns, panel_size)

        directions_drawn = []
        bias_trace_drawn = False
        bias_frequency_drawn = False
        for panel, (name, traces) in zip(panels, batch):
            bias = bias_points.get(name)
            for iteration, direction, sweep in traces:
                at_bias_amplitude = bias is not None and np.isclose(
                    sweep["sweep_amplitude"], bias.amplitude,
                    rtol=BIAS_AMPLITUDE_RTOL, atol=0.0,
                )
                draw(
                    panel,
                    sweep,
                    mappable.to_rgba(sweep["sweep_amplitude"]),
                    DIRECTION_LINESTYLES.get(direction, FALLBACK_LINESTYLE),
                    normalize,
                    at_bias_amplitude,
                )
                bias_trace_drawn |= at_bias_amplitude
                if direction not in directions_drawn:
                    directions_drawn.append(direction)

            centre_mhz = traces[0][2]["original_center_frequency"] / 1e6
            if bias is not None:
                panel.axvline(
                    (bias.frequency_hz - traces[0][2]["original_center_frequency"])
                    / 1e3,
                    color=BIAS_COLOUR,
                    lw=2.0,
                    ls="--",
                    zorder=3,
                )
                bias_frequency_drawn = True
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
        handles = []
        labels = []
        if len(directions_drawn) > 1:
            handles.extend([
                Line2D([], [], color="0.3",
                       ls=DIRECTION_LINESTYLES.get(direction, FALLBACK_LINESTYLE))
                for direction in directions_drawn
            ])
            labels.extend(directions_drawn)
        if bias_frequency_drawn:
            handles.append(Line2D([], [], color=BIAS_COLOUR, lw=2.0, ls="--"))
            labels.append("catalog bias frequency")
        if bias_trace_drawn:
            handles.append(Line2D([], [], color="0.3", lw=BIAS_TRACE_LINEWIDTH))
            labels.append("trace at catalog bias amplitude")
        if handles:
            fig.legend(
                handles, labels,
                loc="outside lower center", ncols=len(handles),
            )

        amplitude_colorbar(fig, mappable, ax=axes, label="drive amp. [norm.]")

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
    overlay_bias=True,
):
    """|S21| against frequency offset, a panel per resonator.

    Args:
        results: one module's sweep results — the value of ``sweeps[module_id]``
            for whatever ``multisweep`` returned, at whatever width.
        names: which sweep sections to draw. A name, a list of names, or
            ``None`` for the whole array.
        iterations: which amplitude steps to draw. A step number, a list of
            them, or ``None`` for all of them. A sweep taken without an
            amplitude schedule has only step 0.
        directions: which frequency directions to draw, as a string or a list.
            Pass ``"upward"`` for one direction only. Directions that were not
            swept are skipped, so the default is safe on a single-direction
            sweep.
        normalize: subtract drive power from received power in dBm, using
            the module's saved ``dac_scale_dbm``. This gives transmission in
            dB relative to the board output, including the intervening chain.
            Like the netanal plot's default, it preserves gain and loss.
            False shows received power in dBm and needs no DAC scale.
        ncols: panels per row, or ``None`` to let :func:`panels_per_row` pick
            from how many there are.
        panel_size: ``(width, height)`` of one panel, in inches.
        title: overrides the figure title. The batch marker is still appended.
        batchlen: resonators per figure; None uses one figure.
        overlay_bias: mark each resonator's bias frequency from the catalog
            snapshot in ``call_params``. A sweep whose amplitude matches the
            catalog bias amplitude within a relative tolerance of 1e-6 is
            drawn thicker. Results without a catalog are drawn unchanged.

    Raises:
        KeyError: if a requested name was never swept.
        ValueError: if the selection matches no sweeps, or normalization
            lacks a finite DAC scale or a finite, positive drive amplitude.
    """
    section_names(results)  # Keep the module/container error explicit.
    dac_scale = results.get("dac_scale_dbm")
    if normalize and (dac_scale is None or not np.isfinite(dac_scale)):
        raise ValueError(
            "Drive-referenced dB requires a finite dac_scale_dbm; "
            "use normalize=False to plot received power in dBm."
        )

    def draw(panel, sweep, colour, linestyle, normalize, highlight):
        magnitude = convert_roc_to_dbm(np.abs(sweep["iq_counts"]))
        if normalize:
            drive = sweep["sweep_amplitude"]
            if not np.isfinite(drive) or drive <= 0:
                raise ValueError("Drive normalization requires a finite, positive amplitude.")
            magnitude -= convert_dacunits_to_dbm(drive, dac_scale)
        panel.plot(
            offset_khz(sweep),
            magnitude,
            lw=BIAS_TRACE_LINEWIDTH if highlight else TRACE_LINEWIDTH,
            color=colour,
            ls=linestyle,
            zorder=2 if highlight else 1,
        )

    _plot_panels(
        results,
        draw,
        xlabel="$f - f_\\mathrm{centre}$ [kHz]",
        ylabel="|S21| [dB, drive-referenced]" if normalize else "received power [dBm]",
        what="magnitude",
        names=names,
        iterations=iterations,
        directions=directions,
        normalize=normalize,
        ncols=ncols,
        panel_size=panel_size,
        title=title,
        batchlen=batchlen,
        overlay_bias=overlay_bias,
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
    """Plot IQ loops with equal axis scales, one panel per resonator.

    Selection and layout arguments match ``plot_magnitude_panels``. Normalization
    divides readout counts by the drive's DAC fraction and needs no DAC power
    scale. The default panels are square.
    """

    def draw(panel, sweep, colour, linestyle, normalize, highlight):
        iq = sweep_iq(sweep, normalize)
        panel.plot(iq.real, iq.imag, lw=TRACE_LINEWIDTH,
                   color=colour, ls=linestyle)

    _plot_panels(
        results,
        draw,
        xlabel="I [counts / DAC amplitude]" if normalize else "I [counts]",
        ylabel="Q [counts / DAC amplitude]" if normalize else "Q [counts]",
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
