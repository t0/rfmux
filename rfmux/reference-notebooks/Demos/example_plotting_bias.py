#!/usr/bin/env python3
"""Plot bias points and bifurcation diagnostics from one module's multisweep.

Run ``find_bias_points`` to store ``bias_report`` in the module dict::

    from rfmux.tuning import find_bias_points
    import example_plotting_bias as biasplots

    ms_module_output = multisweep_output[crs.module[1].index()]
    find_bias_points(ms_module_output)
    biasplots.plot_bias_points(ms_module_output)
    biasplots.plot_bifurcation_checks(ms_module_output)
    biasplots.plot_hysteresis_checks(ms_module_output)

``plot_arc_speed_panels`` evaluates the sweeps directly and does not require
a saved report. Flagged bias points are
orange, with the reason shown in the bias-point panel.
``plot_bifurcation_checks`` shows pair strength and both thresholds versus drive,
matching Periscope; ``plot_arc_speed_panels(..., quantity="spikes")`` shows
the raw speed changes and the strongest eligible pair.

Style is applied per figure. Use ``batchlen=None`` for a single figure.
"""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.colorbar import Colorbar
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogFormatter

from rfmux.core.transferfunctions import convert_roc_to_dbm

from rfmux.tuning import (
    BiasReport,
    bifurcated_by_derivative,
    collect_amplitude_iterations_for,
    iq_arc_speed,
    normalized_arc_speed,
    hysteresis_separation,
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
    "amplitude_colorbar",
    "offset_khz",
    "panels_per_row",
    "plot_arc_speed_panels",
    "plot_bias_points",
    "plot_bifurcation_checks",
    "plot_hysteresis_checks",
    "square_axes",
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
    "axes.formatter.limits": (-3, 3),
}

# Resonators per figure.
BATCH_SIZE = 50

# Truncate pale yellows to keep traces visible on white.
AMPLITUDE_CMAP = LinearSegmentedColormap.from_list(
    "gnuplot_truncated", plt.cm.gnuplot(np.linspace(0.0, 0.9, 256))
)

BIAS_COLOUR = "royalblue"  # the chosen operating point
FLAGGED_COLOUR = "darkorange"  # a bias point that is a default, not a finding

# Mirrors ``rfmux.tuning.bias.PREFERRED_DIRECTION``: which direction a bias
# frequency is measured on when a step has both and the caller did not say.
PREFERRED_DIRECTION = "upward"

# Frequency-domain diagnostics used by the bias methods.
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
        # Prominence is measured on differences of this speed.
        "annotation": None,
    },
    "spikes": {
        "reader": None,  # np.diff of normalized_arc_speed — see _arc_quantity
        "label": "$\\Delta$ normalized speed [1/Hz]",
        "what": "what the derivative test looks for spikes in",
        # Mark the strongest eligible pair, even below threshold.
        "annotation": "pair",
    },
}


def panels_per_row(count, few=5, many=7):
    """Choose a column count from the number of panels."""
    if count > 30:
        return many
    if count < 10:
        return count
    return few


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
        low, high = low * 0.9, high * 1.1
    norm = LogNorm(vmin=low, vmax=high) if low > 0 else Normalize(vmin=low, vmax=high)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def offset_khz(entry, frequencies=None):
    """Return frequency offsets from the sweep centre in kHz."""
    if frequencies is None:
        frequencies = entry["frequencies"]
    return (np.asarray(frequencies) - entry["original_center_frequency"]) / 1e3


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


def _panel_grid(count, columns, panel_size):
    """Create a panel grid, hide spare axes, and return figure, axes and panels."""
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
    """Label the left edge and lowest visible panel in each column."""
    for column in range(axes.shape[1]):
        visible = [panel for panel in axes[:, column] if panel.get_visible()]
        if visible:
            visible[-1].set_xlabel(xlabel)
    for panel in axes[:, 0]:
        if panel.get_visible():
            panel.set_ylabel(ylabel)


def _columns_for(batches, ncols):
    """Use the first batch to choose a consistent figure width."""
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


def _section_names(ms_module_output):
    """Read section names from one module's multisweep output."""
    try:
        iterations = ms_module_output["results"]
    except (TypeError, KeyError):
        keys = (list(ms_module_output) if isinstance(ms_module_output, dict)
                else type(ms_module_output).__name__)
        raise TypeError(
            "Expected one module's multisweep output — the value of "
            "multisweep_output[module_id] — rather than the whole output "
            f"keyed by module identifier. Got {keys}. If there is only one "
            "module in play, multisweep_output[list(multisweep_output)[0]] "
            "is the thing to pass."
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


def _bias_report(ms_module_output: dict) -> BiasReport:
    """Read the bias report embedded in one module's multisweep."""
    _section_names(ms_module_output)
    if "bias_report" not in ms_module_output:
        raise ValueError(
            "Multisweep has no bias_report; run "
            "find_bias_points(ms_module_output) first."
        )
    return BiasReport.from_dict(ms_module_output["bias_report"])


def _findings(report, names):
    """Select findings in requested-name order, or report order if names is None."""
    wanted = _as_list(names)
    if wanted is None:
        findings = list(report.findings)
    else:
        # report[name] raises a KeyError naming the miss, which is what we want
        findings = [report[name] for name in wanted]
    if not findings:
        raise ValueError("This report has no findings, so there is nothing to draw.")
    return findings


def _direction_for(report, ms_module_output, direction):
    """Use the requested or recorded direction, then prefer upward if available."""
    if direction is not None:
        return direction
    recorded = (getattr(report, "settings", None) or {}).get("direction")
    if recorded is not None:
        return recorded

    swept = [
        d
        for by_direction in ms_module_output["results"].values()
        for d in by_direction
    ]
    return PREFERRED_DIRECTION if PREFERRED_DIRECTION in swept else swept[0]


def _entry_for(ms_module_output, name, iteration, direction):
    """The one sweep a finding came off, with a readable error if it is absent."""
    try:
        return ms_module_output["results"][iteration][direction][name]
    except (KeyError, TypeError):
        raise ValueError(
            f"The sweeps passed in do not hold {name!r} at amplitude step "
            f"{iteration} swept {direction!r}, which is where this bias point "
            f"came from. These are the sweeps of a different measurement than "
            f"the report was made from."
        ) from None


def plot_bias_points(
    ms_module_output: dict,
    *,
    projection="magnitude",
    names=None,
    direction=None,
    ncols=None,
    panel_size=None,
    title=None,
    batchlen=BATCH_SIZE,
    xlim_khz=None,
):
    """Plot the selected sweep and bias point, one panel per resonator.

    Flagged findings use FLAGGED_COLOUR and include the reason. A nonlinear
    fit's ``a`` is shown when its selected sweep has that parameter.

    Args:
        ms_module_output: one module's output from ``multisweep``, including
            the ``bias_report`` stored by ``find_bias_points``.
        projection: "magnitude" for received power in dBm versus frequency,
            or "iq" for the loop in readout counts. Both mark the bias point.
        names: which resonators to draw. A name, a list of names, or ``None``
            for every finding in the report.
        direction: which sweep direction to draw. ``None`` follows what the
            report was run with.
        ncols: panels per row, or ``None`` to let :func:`panels_per_row` pick.
        panel_size: ``(width, height)`` of one panel, in inches. The default is
            square for the IQ projection.
        title: overrides the figure title. The batch marker is still appended.
        batchlen: resonators per figure; None uses one figure.
        xlim_khz: ``(low, high)`` frequency-offset range to show in every
            magnitude panel, in kHz from the sweep centre; ``None`` shows
            each whole sweep.

    Raises:
        KeyError: if a requested name has no finding.
        TypeError: if handed the whole per-module output container.
        ValueError: for a missing report, unknown projection or missing bias
            sweep, or ``xlim_khz`` with the IQ projection, which has no
            frequency axis.
    """
    if projection not in ("magnitude", "iq"):
        raise ValueError(
            f"Unknown projection {projection!r}. This draws 'magnitude' or 'iq'."
        )
    if xlim_khz is not None and projection == "iq":
        raise ValueError(
            "xlim_khz sets a frequency axis, and the IQ projection has none. "
            "Use projection='magnitude', or leave xlim_khz out."
        )
    if panel_size is None:
        panel_size = (6.0, 6.0) if projection == "iq" else (7.0, 5.0)

    report = _bias_report(ms_module_output)
    findings = _findings(report, names)
    swept_direction = _direction_for(report, ms_module_output, direction)

    batches = _batches(findings, batchlen)
    columns = _columns_for(batches, ncols)

    for batch_number, batch in enumerate(batches, start=1):
        with plt.rc_context(PLOT_STYLE):
            fig, axes, panels = _panel_grid(len(batch), columns, panel_size)

            for panel, finding in zip(panels, batch):
                entry = _entry_for(
                    ms_module_output, finding.name, finding.iteration,
                    swept_direction,
                )
                colour = FLAGGED_COLOUR if not finding.good else BIAS_COLOUR
                iq = np.asarray(entry["iq_counts"])

                if projection == "magnitude":
                    panel.plot(offset_khz(entry), convert_roc_to_dbm(np.abs(iq)),
                               lw=1.5, color="0.35")
                    panel.axvline(
                        (finding.frequency_hz - entry["original_center_frequency"])
                        / 1e3,
                        color=colour, lw=2.5,
                    )
                    if xlim_khz is not None:
                        panel.set_xlim(xlim_khz)
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
                fit = (entry.get("fits") or {}).get("nonlinear") or {}
                a = (fit.get("params") or {}).get("a")
                if a is not None:
                    note += f"\na = {a:.3f}"
                    if fit.get("failed_because") is not None:
                        note += " (fit rejected)"
                if finding.bifurcated_at is not None:
                    note += f"\nbifurcated at {finding.bifurcated_at:.4g}"
                if not finding.good:
                    note += "\n" + "\n".join(
                        textwrap.wrap(finding.flagged_because, 32)
                    )
                panel.text(0.03, 0.03, note, transform=panel.transAxes,
                           fontsize=16, va="bottom", color=colour)

            if projection == "magnitude":
                _outer_labels(axes, "$f - f_\\mathrm{centre}$ [kHz]", "received power [dBm]")
            else:
                _outer_labels(axes, "I [counts]", "Q [counts]")

            flagged = sum(1 for f in batch if not f.good)
            handles = [Line2D([], [], color=BIAS_COLOUR, lw=2.5)]
            labels = ["bias point"]
            if flagged:
                handles.append(Line2D([], [], color=FLAGGED_COLOUR, lw=2.5))
                labels.append("flagged")
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
    ms_module_output: dict,
    *,
    names: str | list[str] | None = None,
    direction: str | None = None,
    ncols: int | None = None,
    panel_size: tuple[float, float] = (7.0, 5.0),
    title: str | None = None,
    batchlen: int | None = BATCH_SIZE,
) -> None:
    """Plot strongest pair strength and both thresholds against drive amplitude.

    Uses the embedded bias report's settings. A pair must reach both curves
    to trigger the derivative detector. Zero strength represents no eligible
    pair when none exists; missing pairs cannot trigger even at zero thresholds.
    Upward sweeps are solid, downward dashed. The vertical line marks the
    selected amplitude. All measured steps are evaluated, including those
    beyond the amplitude search's stopping point. This does not show the
    hysteresis test, which can independently trigger a combined verdict.

    ``names``, ``ncols``, ``panel_size``, ``title`` and ``batchlen`` select and
    arrange resonator panels; ``direction=None`` shows every sweep direction.
    """
    report = _bias_report(ms_module_output)
    findings = _findings(report, names)
    settings = report.settings
    batches = _batches(findings, batchlen)
    columns = _columns_for(batches, ncols)
    for batch_number, batch in enumerate(batches, start=1):
        with plt.rc_context(PLOT_STYLE):
            fig, axes, panels = _panel_grid(len(batch), columns, panel_size)
            for panel, finding in zip(panels, batch):
                by_direction = {}
                for entries in collect_amplitude_iterations_for(
                    ms_module_output, finding.name
                ).values():
                    for swept_direction, entry in entries.items():
                        if direction is not None and swept_direction != direction:
                            continue
                        try:
                            check = bifurcated_by_derivative(
                                {swept_direction: entry},
                                spike_prominence_factor=settings.get("spike_prominence_factor", 0.5),
                                noise_gate_factor=settings.get("noise_gate_factor", 50.0),
                            )
                        except (ValueError, KeyError):
                            continue
                        by_direction.setdefault(swept_direction, []).append((
                            entry["sweep_amplitude"], check.metric["pair_strength"],
                            check.diagnostics["shape_threshold"],
                            check.diagnostics["noise_threshold"],
                        ))
                for swept_direction, rows in by_direction.items():
                    values = np.asarray(sorted(rows)).T
                    for y, label, colour, marker in zip(
                        values[1:], ("Pair strength", "Shape threshold", "Noise threshold"),
                        ("#3366CC", "#CC6633", "#339966"), ("o", "^", "s"),
                    ):
                        panel.plot(values[0], y, color=colour, marker=marker,
                                   ls="--" if swept_direction == "downward" else "-",
                                   label=f"{label} ({swept_direction})")
                if by_direction:
                    panel.axvline(finding.amplitude, color="0.4", ls=":",
                                  label="Selected amplitude")
                    panel.legend(fontsize=10)
                else:
                    panel.text(0.5, 0.5, "no usable derivative traces",
                               ha="center", transform=panel.transAxes)
                panel.set_ylim(bottom=0)
                panel.set_title(finding.name)
            _outer_labels(axes, "Drive amplitude [DAC fraction]", "Prominence [1/Hz]")
            _titled(fig, _batch_title(
                title, "Derivative test — pair strength must reach both thresholds",
                len(findings), batch_number, len(batches),
            ))
            plt.show()


def plot_hysteresis_checks(
    ms_module_output: dict, *, names: str | list[str] | None = None,
    ncols: int | None = None, panel_size: tuple[float, float] = (7.0, 5.0),
    title: str | None = None, batchlen: int | None = BATCH_SIZE,
    xlim_khz: tuple[float, float] | None = None,
) -> None:
    """Plot up/down separation versus frequency, one curve per amplitude.

    Reads compare and max_discrepancy from the embedded bias_report. Values
    above 1 exceed the allowed separation; equality does not trigger. Magnitude
    compares dip-depth fractions; IQ compares loop-radius fractions. For a
    zero limit, show those fractions directly with the threshold at zero on a
    linear axis. Positive limits use a log axis to show small differences.
    Missing or unusable sweep pairs are skipped. The selected amplitude is bold.
    ``xlim_khz`` is a ``(low, high)`` frequency-offset range in kHz from the
    sweep centre, applied to every panel; ``None`` shows each whole sweep.
    """
    report = _bias_report(ms_module_output)
    findings = _findings(report, names)
    compare = report.settings.get("compare", "magnitude")
    limit = report.settings.get("max_discrepancy", 0.1)
    divisor = limit if limit > 0 else 1.0
    units = "dip depth" if compare == "magnitude" else "loop radius"
    ylabel = "Up/down difference / limit" if limit > 0 else f"Up/down difference / {units}"
    entries = {
        f.name: collect_amplitude_iterations_for(ms_module_output, f.name)
        for f in findings
    }
    amplitudes = [sweep["sweep_amplitude"] for steps in entries.values()
                  for pair in steps.values() for sweep in pair.values()]
    if not amplitudes:
        raise ValueError("No sweeps match the selected resonators.")
    mappable = amplitude_mappable(amplitudes)
    batches = _batches(findings, batchlen)
    columns = _columns_for(batches, ncols)
    for batch_number, batch in enumerate(batches, start=1):
        with plt.rc_context(PLOT_STYLE):
            fig, axes, panels = _panel_grid(len(batch), columns, panel_size)
            for panel, finding in zip(panels, batch):
                drawn = False
                for step, pair in entries[finding.name].items():
                    try:
                        frequencies, separation = hysteresis_separation(pair, compare=compare)
                    except (ValueError, KeyError):
                        continue
                    sweep = pair["upward"]
                    panel.plot(
                        offset_khz(sweep, frequencies), separation / divisor,
                        color=mappable.to_rgba(sweep["sweep_amplitude"]),
                        lw=2.5 if step == finding.iteration else 1.0,
                    )
                    drawn = True
                panel.axhline(limit / divisor, color="0.35", ls="--",
                              label=f"Limit: {limit:g} × {units}")
                if not drawn:
                    panel.text(0.5, 0.5, "no usable up/down pairs", ha="center",
                               transform=panel.transAxes)
                if limit > 0:
                    panel.set_yscale("log")
                else:
                    panel.set_ylim(bottom=0)
                if xlim_khz is not None:
                    panel.set_xlim(xlim_khz)
                panel.set_title(finding.name)
                panel.legend(fontsize=10)
            _outer_labels(axes, "$f - f_\\mathrm{centre}$ [kHz]", ylabel)
            amplitude_colorbar(fig, mappable, ax=axes, label="drive amp. [norm.]")
            _titled(fig, _batch_title(
                title, f"Hysteresis ({compare}) — selected amplitude bold",
                len(findings), batch_number, len(batches),
            ))
            plt.show()


def _amplitude(entries):
    """Read the step amplitude from the first direction that contains one."""
    for entry in entries.values():
        amplitude = entry.get("sweep_amplitude")
        if amplitude is not None:
            return amplitude
    return float("nan")


def _arc_quantity(quantity, entry):
    """``(frequencies, values)`` for one of :data:`ARC_QUANTITIES`."""
    if quantity == "spikes":
        # What the detector looks for spikes in is the *difference* of the
        # normalized speed, on the midpoints of the pairs it differenced.
        frequencies, speed = normalized_arc_speed(entry)
        return 0.5 * (frequencies[:-1] + frequencies[1:]), np.diff(speed)
    return ARC_QUANTITIES[quantity]["reader"](entry)


def plot_arc_speed_panels(
    ms_module_output,
    quantity="arc_speed",
    names=None,
    iterations=None,
    direction=PREFERRED_DIRECTION,
    annotate=True,
    spike_prominence_factor=0.5,
    noise_gate_factor=50.0,
    ncols=None,
    panel_size=(7.0, 5.0),
    title=None,
    batchlen=BATCH_SIZE,
    xlim_khz=None,
):
    """Plot arc speed, normalized speed or its difference per resonator.

    With ``annotate=True``, arc speed gets a maximum marker and spikes get
    markers on the strongest eligible peak–trough pair. Filled markers indicate
    detection; open markers indicate a pair below threshold. Normalized speed
    has no annotation.

    All selected steps are evaluated, including those beyond the first
    bifurcation recorded by the amplitude search.

    Args:
        ms_module_output: one module's output from ``multisweep``.
        quantity: which of :data:`ARC_QUANTITIES` to draw. ``"arc_speed"`` is
            what the ``iq_derivative`` frequency method maximizes;
            ``"normalized_speed"`` is what the ``derivative`` bifurcation test
            differentiates; ``"spikes"`` is that difference, which is what it
            actually looks for spikes in.
        names: which resonators to draw. ``None`` for the whole array.
        iterations: which amplitude steps to draw. ``None`` for all of them.
        direction: the sweep direction to draw.
        annotate: draw the per-quantity marks described above.
        spike_prominence_factor: as :func:`~rfmux.tuning.find_bias_points`
            takes it. Used to classify the marked ``spikes`` pair.
        noise_gate_factor: noise requirement for the marked ``spikes`` pair.
        ncols: panels per row, or ``None`` to let :func:`panels_per_row` pick.
        panel_size: ``(width, height)`` of one panel, in inches.
        title: overrides the figure title. The batch marker is still appended.
        batchlen: resonators per figure; None uses one figure.
        xlim_khz: ``(low, high)`` frequency-offset range to show in every
            panel, in kHz from the sweep centre; ``None`` shows each whole
            sweep.

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

    measured_names = _section_names(ms_module_output)
    if not measured_names:
        raise ValueError("This measurement holds no sweeps, so there is nothing to draw.")
    wanted_names = _as_list(names) or measured_names
    wanted_iterations = _as_list(iterations)

    entries_by_name = {}
    for name in wanted_names:
        entries = []
        for iteration, by_direction in collect_amplitude_iterations_for(
            ms_module_output, name
        ).items():
            if wanted_iterations is not None and iteration not in wanted_iterations:
                continue
            if direction in by_direction:
                entries.append((iteration, by_direction[direction]))
        if entries:
            entries_by_name[name] = entries

    if not entries_by_name:
        available = collect_amplitude_iterations_for(
            ms_module_output, wanted_names[0]
        )
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
                    elif spec["annotation"] == "pair":
                        check = bifurcated_by_derivative(
                            {direction: entry},
                            spike_prominence_factor=spike_prominence_factor,
                            noise_gate_factor=noise_gate_factor,
                        )
                        pair = check.diagnostics["pair"]
                        if pair is not None:
                            indices = [pair["positive_index"], pair["negative_index"]]
                            panel.plot(offset_khz(entry, frequencies)[indices], values[indices],
                                       ls="none", marker="o", color=colour,
                                       mfc=colour if check.bifurcated else "none", ms=9)

                if xlim_khz is not None:
                    panel.set_xlim(xlim_khz)
                centre_mhz = entries[0][1]["original_center_frequency"] / 1e6
                panel.set_title(f"{name}  {centre_mhz:.3f} MHz")

            _outer_labels(axes, "$f - f_\\mathrm{centre}$ [kHz]", spec["label"])
            amplitude_colorbar(fig, mappable, ax=axes, label="drive amp. [norm.]")

            if annotate and spec["annotation"] == "pair":
                fig.legend(
                    [Line2D([], [], ls="none", marker="o", color="0.35", mfc=fill)
                     for fill in ("0.35", "none")],
                    ["strongest pair: detected", "strongest pair: below threshold"],
                    loc="outside lower center", ncols=2,
                )
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
