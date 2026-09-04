#!/usr/bin/env python3
"""Example plots for fitted sweeps: models over data, and parameters over drive.

``fit_sweeps`` writes its results into the sweep entries it fitted, under
``fits``, keyed by model. These plots read them back out::

    import example_plotting_fits as fitplots

    module_sweeps = sweeps[crs.module[1].index()]
    fit_sweeps(module_sweeps)

    fitplots.plot_fit_panels(module_sweeps, model="skewed")
    fitplots.plot_fitted_parameters(module_sweeps, model="skewed")

The first draws each resonator's measured trace as points with the fitted model
over it as a smooth line — smooth because the model is evaluated on a grid
``oversample`` times denser than the sweep, so what you see is the model and
not a polyline joining the same few samples the fit had. Judging a fit by eye
is what it is for. The second plots fitted parameters against drive amplitude,
which is where a resonator being driven too hard shows up.

Both resonance models — ``skewed`` and ``nonlinear`` — are drawn against
frequency, because the dip is where a bad fit shows itself, and because it puts
the two models on the same axes when you want to compare them. ``circle`` is
drawn in the IQ plane instead: it has nothing to say about frequency, and the
IQ plane is the only place a circle looks like a circle.

A fit that did not converge is drawn as its data with no model over it and the
reason in the panel, rather than being silently skipped. A missing curve you
can see beats a missing curve you cannot.

Styling follows hidfmux's plotting modules — large type, a grid on every axes,
compact bracketed axis labels, and generous panels. It lives in ``PLOT_STYLE``
and is applied per figure, not to your session. Panels are drawn in batches of
``BATCH_SIZE`` resonators, so pointing these at a whole array gives readable
figures rather than one that is metres across.

These are meant to be read and copied, and each ``example_plotting_*`` module
here stands alone: the small amount of layout and style bookkeeping is repeated
in each rather than shared, so that one file is the whole story and lifting a
function out of it is a copy-paste.

One deliberate difference from ``example_plotting_multisweep.py``: these take a
single ``direction``, not a list. A magnitude trace with its fit over it is
already two lines per amplitude step, and drawing both sweep directions on top
of that makes a panel nobody can read. Call twice if you want to compare them.
"""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D

from rfmux.tuning import (
    collect_amplitude_iterations_for,
    nonlinear_model_iq,
    skewed_model_magnitude,
)

__all__ = [
    "AMPLITUDE_CMAP",
    "BATCH_SIZE",
    "MEASURED_COLOUR",
    "PLOT_STYLE",
    "amplitude_mappable",
    "fitted_value",
    "model_on_a_finer_grid",
    "offset_khz",
    "panels_per_row",
    "parameters_for",
    "plot_fit_panels",
    "plot_fitted_parameters",
    "square_axes",
]


# Type big enough to read on a projector, and a grid on every axes: these plots
# get shown to other people, and a fit you cannot read a number off is a fit
# nobody will argue with. Applied per figure through ``plt.rc_context`` rather
# than written into ``plt.rcParams`` at import, so importing this module does
# not quietly restyle the rest of your notebook. If you *want* it everywhere::
#
#     plt.rcParams.update(example_plotting_fits.PLOT_STYLE)
#
PLOT_STYLE = {
    "font.size": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 14,
    "axes.grid": True,
    # A Q of 3e5 or an axis of raw IQ counts otherwise reads as six digits per
    # tick, or picks a different exponent in each panel of the same figure.
    "axes.formatter.useoffset": False,
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-3, 3),
}

# Resonators per figure. A panel is sized to be read rather than to fit.
BATCH_SIZE = 50

# gnuplot runs black -> purple -> red -> orange -> yellow, so it stays
# saturated from end to end and every trace reads against a white background.
AMPLITUDE_CMAP = plt.cm.gnuplot

# When only one amplitude step is drawn there is no drive for colour to encode,
# so it is spent on telling the data from the model instead. With several steps
# both take the step's colour and are told apart by points versus line, because
# colour is back to meaning drive.
MEASURED_COLOUR = "0.45"
FIT_COLOUR = "crimson"

# How each model is drawn. Both resonance models go against frequency, because
# the dip is where a bad fit shows itself — the nonlinear model is *fitted* to
# the complex trace, but its magnitude is what you read it off. The circle fit
# has nothing to say about frequency at all, so it stays in the IQ plane, which
# is the only place a circle is a circle. Adding a model to fits.py means
# adding a line here.
MODEL_PROJECTION = {"skewed": "magnitude", "nonlinear": "magnitude", "circle": "iq"}

# The parameters worth plotting against drive, per model, and how to draw them.
# ``shift`` subtracts the value at the lowest drive: absolute fr differs by
# hundreds of MHz between resonators, so the shift is what lands on one axis.
# ``log`` puts that panel's y-axis on a log scale, which is what a Q wants.
PARAMETER_PANELS = {
    "skewed": (
        {"name": "fr", "label": "$f_r$ shift [kHz]", "scale": 1e-3, "shift": True},
        {"name": "Qr", "label": "$Q_r$", "log": True},
        {"name": "Qc", "label": "$Q_c$", "log": True},
        {"name": "Qi", "label": "$Q_i$", "log": True},
    ),
    "nonlinear": (
        {"name": "fr", "label": "$f_r$ shift [kHz]", "scale": 1e-3, "shift": True},
        {"name": "Qr", "label": "$Q_r$", "log": True},
        {"name": "a", "label": "$a$ (bifurcates near 0.77)"},
        {"name": "residual", "label": "fit residual"},
    ),
    "circle": (
        {"name": "radius", "label": "loop radius [counts]"},
    ),
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
    denser one a model is drawn on, for instance.
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


def model_on_a_finer_grid(reader, entry, oversample=25):
    """``(frequencies, model)`` from a reader, on a denser axis than was measured.

    The readers in ``rfmux.tuning`` evaluate a model on whatever frequencies
    the entry carries, which for a 101-point sweep means a 101-point polyline —
    the same corners the fit saw, which is not what the model looks like. Hand
    the reader a copy of the entry with a denser grid and it happily obliges.

    Returns the frequencies too, since they are no longer the entry's own and
    the model has to be plotted against them.
    """
    frequencies = np.linspace(
        entry["frequencies"][0],
        entry["frequencies"][-1],
        oversample * len(entry["frequencies"]),
    )
    return frequencies, reader({**entry, "frequencies": frequencies})


def fitted_value(entry, model, parameter):
    """One number off one fit, or ``None`` if there is not one.

    Looks in the fit's ``params`` first, then at the fit dict itself, because
    not everything a fit learned is a fitted parameter: the nonlinear fit's
    ``residual`` and ``gain``, and the circle fit's ``center`` and ``radius``,
    sit beside ``params`` rather than in it.

    ``None`` rather than an exception, so that a resonator whose fit failed at
    one amplitude leaves a gap in a curve instead of taking the whole plot
    down — or worse, being quietly dropped and joining the points either side.
    """
    fit = (entry.get("fits") or {}).get(model)
    if fit is None or fit.get("failed_because") is not None:
        return None
    params = fit.get("params") or {}
    if parameter in params:
        return params[parameter]
    return fit.get(parameter)


def parameters_for(model):
    """The parameter panels this module draws for *model*, as a tuple."""
    try:
        return PARAMETER_PANELS[model]
    except KeyError:
        raise ValueError(
            f"No parameter panels are defined for model {model!r}. Known "
            f"models: {', '.join(sorted(PARAMETER_PANELS))}. Pass "
            f"parameters=('name', ...) to plot something else."
        ) from None


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


def _collect(results, names, iterations, direction):
    """``{name: [(iteration, entry), ...]}`` for what was asked for."""
    measured_names = _section_names(results)
    if not measured_names:
        raise ValueError("This measurement holds no sweeps, so there is nothing to draw.")
    wanted_names = _as_list(names) or measured_names
    wanted_iterations = _as_list(iterations)

    collected = {}
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
            collected[name] = entries

    if not collected:
        available = collect_amplitude_iterations_for(results, wanted_names[0])
        directions_swept = sorted(
            {d for by_direction in available.values() for d in by_direction}
        )
        raise ValueError(
            f"Nothing to plot: iterations={iterations!r} direction={direction!r} "
            f"selected none of the sweeps taken. This measurement has iterations "
            f"{list(available)} and directions {directions_swept}."
        )
    return collected


def _batch_title(title, what, count, steps, batch_number, batch_count):
    """What goes above the figure, plus which batch of how many it is.

    *steps* is the amplitude steps drawn, or ``None`` where the count would say
    nothing — a parameter-against-drive plot has every step in it by
    construction.
    """
    if title is None:
        title = f"{what}: {count} sweep section{'s' if count != 1 else ''}"
        if steps is not None:
            title += (
                f", {len(steps)} amplitude step{'s' if len(steps) != 1 else ''}"
            )
    if batch_count > 1:
        return f"{title}  [batch {batch_number} of {batch_count}]"
    return title


def plot_fit_panels(
    results,
    model="skewed",
    names=None,
    iterations=None,
    direction="upward",
    oversample=25,
    ncols=None,
    panel_size=None,
    title=None,
    batchlen=BATCH_SIZE,
):
    """Measured points with the fitted model over them, a panel per resonator.

    Args:
        results: one module's sweep results — the value of ``sweeps[module_id]``
            — after ``fit_sweeps`` has run on it.
        model: ``"skewed"``, ``"nonlinear"`` or ``"circle"``. Decides both which
            fit is read and how it is drawn: ``skewed`` against frequency, the
            other two in the IQ plane.
        names: which sweep sections to draw. A name, a list of names, or
            ``None`` for the whole array.
        iterations: which amplitude steps to draw. A step number, a list of
            them, or ``None`` for all of them.
        direction: which frequency direction to draw. One direction, not a
            list — see the module docstring.
        oversample: how many times denser than the sweep to evaluate the model
            on, so the curve is a curve. 1 draws it on the sweep's own points.
        ncols: panels per row, or ``None`` to let :func:`panels_per_row` pick.
        panel_size: ``(width, height)`` of one panel, in inches. The default
            depends on the model: square for the IQ projections.
        title: overrides the figure title. The batch marker is still appended.
        batchlen: resonators per figure. ``None`` for one figure however big.

    Raises:
        KeyError: if a requested name was never swept.
        TypeError: if handed the whole per-module container.
        ValueError: for an unknown model, or if the selection matches nothing.
    """
    try:
        projection = MODEL_PROJECTION[model]
    except KeyError:
        raise ValueError(
            f"Unknown model {model!r}. This module draws "
            f"{', '.join(sorted(MODEL_PROJECTION))}."
        ) from None

    if panel_size is None:
        panel_size = (6.0, 6.0) if projection == "iq" else (7.0, 5.0)

    entries_by_name = _collect(results, names, iterations, direction)
    mappable = amplitude_mappable([
        entry["sweep_amplitude"]
        for entries in entries_by_name.values()
        for _, entry in entries
    ])
    steps = sorted(
        {iteration for entries in entries_by_name.values() for iteration, _ in entries}
    )

    batches = _batches(list(entries_by_name.items()), batchlen)
    # One column count for every figure, taken from a full batch, so a short
    # final batch is drawn at the same width as the ones before it.
    columns = min(
        ncols if ncols is not None else panels_per_row(len(batches[0])),
        len(batches[0]),
    )

    for batch_number, batch in enumerate(batches, start=1):
        _draw_fit_batch(
            batch, model, projection, mappable, oversample, columns, panel_size,
            _batch_title(
                title, f"{model} fits, {direction}", len(entries_by_name), steps,
                batch_number, len(batches),
            ),
        )


def _draw_measured_and_model(
    panel, entry, model, projection, colour, fit_colour, oversample
):
    """One amplitude step in one panel: the data, and the model if it converged.

    Returns the reason the fit failed, or ``None``. The caller collects those
    for the panel note — a fit that did not converge still has data worth
    looking at, and the reason belongs next to it.
    """
    fit = (entry.get("fits") or {}).get(model)
    if fit is None:
        raise ValueError(
            f"This sweep has no {model} fit. Run "
            f"fit_sweeps(..., models=({model!r},)) on it first."
        )
    converged = fit.get("failed_because") is None

    if projection == "magnitude":
        # Both models are drawn against the trace divided by its last point.
        # The skewed fit already works in exactly those units; the nonlinear
        # model comes back in readout counts, so it takes the same divisor —
        # one constant applied to both, which leaves the shapes untouched and
        # the two of them on the same axis.
        reference = np.abs(entry["iq_counts"][-1])
        measured = np.abs(entry["iq_counts"]) / reference
        panel.plot(offset_khz(entry), 20 * np.log10(measured),
                   ls="none", marker=".", ms=9, color=colour)
        if converged:
            if model == "skewed":
                frequencies, curve = model_on_a_finer_grid(
                    skewed_model_magnitude, entry, oversample
                )
                curve = np.abs(curve)
            else:
                frequencies, curve = model_on_a_finer_grid(
                    nonlinear_model_iq, entry, oversample
                )
                curve = np.abs(curve) / reference
            panel.plot(offset_khz(entry, frequencies), 20 * np.log10(curve),
                       lw=2.5, color=fit_colour, alpha=0.85)

    else:  # circle
        iq = np.asarray(entry["iq_counts"])
        panel.plot(iq.real, iq.imag, ls="none", marker=".", ms=9, color=colour)
        if converged:
            # The fitted circle is two numbers, so it is drawn rather than
            # evaluated: a full turn at the fitted radius about the fitted
            # centre, which is the whole of what this fit claims.
            centre, radius = fit.get("center"), fit.get("radius")
            turn = np.linspace(0, 2 * np.pi, 400)
            panel.plot(centre.real + radius * np.cos(turn),
                       centre.imag + radius * np.sin(turn),
                       lw=2.5, color=fit_colour, alpha=0.85)
            panel.plot(centre.real, centre.imag, marker="+", ms=16, mew=2.5,
                       color=fit_colour)

    return None if converged else fit.get("failed_because")


def _draw_fit_batch(
    batch, model, projection, mappable, oversample, columns, panel_size, title
):
    """One figure, holding one batch of sweep sections."""
    with plt.rc_context(PLOT_STYLE):
        fig, axes, panels = _panel_grid(len(batch), columns, panel_size)
        one_step = all(len(entries) == 1 for _, entries in batch)

        for panel, (name, entries) in zip(panels, batch):
            failures = []
            for iteration, entry in entries:
                # With a single amplitude step there is nothing for colour to
                # encode, so a plain grey reads better than one arbitrary
                # colour off the middle of the map.
                colour = (
                    MEASURED_COLOUR if one_step
                    else mappable.to_rgba(entry["sweep_amplitude"])
                )
                fit_colour = FIT_COLOUR if one_step else colour
                why = _draw_measured_and_model(
                    panel, entry, model, projection, colour, fit_colour, oversample
                )
                if why is not None:
                    failures.append(f"step {iteration}: {why}")

            centre_mhz = entries[0][1]["original_center_frequency"] / 1e6
            panel.set_title(f"{name}  {centre_mhz:.3f} MHz")
            if projection == "iq":
                square_axes(panel)
            if failures:
                shown = failures[:3]
                extra = len(failures) - len(shown)
                # The reasons are sentences, so wrap them rather than let them
                # run off the panel.
                lines = ["did not converge —"]
                for reason in shown:
                    lines += textwrap.wrap(reason, 30, subsequent_indent="  ")
                if extra:
                    lines.append(f"... {extra} more")
                panel.text(
                    0.03, 0.03, "\n".join(lines),
                    transform=panel.transAxes, fontsize=11, va="bottom",
                    color="firebrick",
                )

        if projection == "magnitude":
            xlabel, ylabel = "$f - f_\\mathrm{centre}$ [kHz]", "|S21| [dB, norm.]"
        else:
            xlabel, ylabel = "I [counts]", "Q [counts]"

        # Axis labels on the outer edge only: repeating them in every panel of a
        # 40-resonator grid costs more room than the panels themselves. The x
        # label goes on the lowest *visible* panel of each column, which is not
        # the bottom row when the section count does not fill the grid.
        for column in range(axes.shape[1]):
            visible = [panel for panel in axes[:, column] if panel.get_visible()]
            if visible:
                visible[-1].set_xlabel(xlabel)
        for panel in axes[:, 0]:
            if panel.get_visible():
                panel.set_ylabel(ylabel)

        fig.legend(
            [Line2D([], [], ls="none", marker=".", ms=12, color=MEASURED_COLOUR),
             Line2D([], [], lw=2.5,
                    color=FIT_COLOUR if one_step else MEASURED_COLOUR)],
            ["measured", f"{model} fit"],
            loc="outside lower center", ncols=2,
        )
        if not one_step:
            fig.colorbar(mappable, ax=axes, label="drive amp. [norm.]")

        _titled(fig, title)
        plt.show()


def plot_fitted_parameters(
    results,
    model="skewed",
    parameters=None,
    names=None,
    direction="upward",
    panel_size=(6.5, 5.0),
    title=None,
    batchlen=BATCH_SIZE,
):
    """Fitted parameters against drive amplitude, one line per resonator.

    This is the plot that says whether a resonator is being driven too hard: a
    Q that falls away and an ``fr`` that walks downwards as the drive comes up
    are what a resonator does on its way to bifurcating.

    A step whose fit did not converge is a gap in the line, not a joined-up
    point — :func:`fitted_value` returns ``None`` and matplotlib leaves the
    break in. A missing point you can see beats an interpolated one you cannot.

    Args:
        results: one module's sweep results, after ``fit_sweeps``.
        model: which model's parameters to read.
        parameters: which parameters to draw, as names. ``None`` uses this
            module's defaults for the model — see ``PARAMETER_PANELS``. Names
            may be anything the fit stored, including what sits beside
            ``params`` (the nonlinear fit's ``residual``, the circle's
            ``radius``).
        names: which resonators to draw. ``None`` for the whole array.
        direction: which frequency direction's fits to read.
        panel_size: ``(width, height)`` of one panel, in inches.
        title: overrides the figure title. The batch marker is still appended.
        batchlen: resonators per figure. Batched on resonators here rather than
            panels, since every resonator is a line on shared axes and a
            hundred of them is not a plot. ``None`` for one figure however big.

    Raises:
        ValueError: for an unknown model with no defaults, or if the selection
            matches nothing.
    """
    if parameters is None:
        panel_specs = parameters_for(model)
    else:
        panel_specs = tuple({"name": p, "label": p} for p in _as_list(parameters))

    entries_by_name = _collect(results, names, None, direction)
    batches = _batches(list(entries_by_name.items()), batchlen)

    for batch_number, batch in enumerate(batches, start=1):
        with plt.rc_context(PLOT_STYLE):
            fig, axes = plt.subplots(
                1, len(panel_specs),
                figsize=(panel_size[0] * len(panel_specs), panel_size[1]),
                constrained_layout=True, squeeze=False,
            )
            panels = axes[0]

            for panel, spec in zip(panels, panel_specs):
                for name, entries in batch:
                    amplitudes = [entry["sweep_amplitude"] for _, entry in entries]
                    values = np.array(
                        [fitted_value(entry, model, spec["name"])
                         for _, entry in entries],
                        dtype=float,  # None becomes nan, which is the gap
                    )
                    if spec.get("shift") and np.isfinite(values).any():
                        # Absolute fr differs by hundreds of MHz between
                        # resonators; the shift from the quietest step is what
                        # puts every curve on one axis.
                        first = values[np.isfinite(values)][0]
                        values = values - first
                    panel.plot(amplitudes, values * spec.get("scale", 1.0),
                               marker="o", ms=7, lw=1.8, label=name)

                panel.set_xscale("log")
                panel.set_xlabel("drive amp. [norm.]")
                panel.set_ylabel(spec["label"])
                if spec.get("log"):
                    panel.set_yscale("log")

            panels[0].legend(ncols=2 if len(batch) > 6 else 1)

            _titled(fig, _batch_title(
                title, f"{model} fit parameters vs drive, {direction}",
                len(entries_by_name), None, batch_number, len(batches),
            ))
            plt.show()
