"""Marking the Periscope-era tuning path as deprecated, in one place.

``bias_kids``, ``fitting``, ``fitting_nonlinear`` and ``df_calibration`` are
the tuning flow as Periscope still drives it: dict-walking analysis over the
pre-schema-2 multisweep shape, with verdicts written onto sweep entries. The
current flow is :mod:`rfmux.tuning` (analysis over saved sweeps, results on a
:class:`~rfmux.core.resonators.ResonatorCatalog`) and
:func:`rfmux.algorithms.operation.apply_bias.apply_bias` (the one board
operation). The old modules stay until Periscope is ported, and every public
entry point in them is wrapped with :func:`deprecated` so that reaching for
one — from a port in progress, say — announces itself and names the
replacement.

The warning is a ``DeprecationWarning``, which Python shows from ``__main__``
and under pytest and hides from library code, so Periscope's console is not
flooded while it is still the caller.
"""

from __future__ import annotations

import functools
import inspect
import warnings

__all__ = ["deprecated", "LEGACY_BANNER"]

#: Prepended to each legacy module's docstring, so the first line anyone reads
#: says what the module is.
LEGACY_BANNER = (
    "**DEPRECATED — legacy Periscope tuning path.** Kept only until Periscope "
    "is ported to :mod:`rfmux.tuning`; do not use in new code. Every public "
    "function here warns and names its replacement."
)


def deprecated(replacement: str, *, note: str | None = None):
    """Wrap a legacy function so calling it warns and names *replacement*.

    Works on coroutine functions and plain ones. The wrapper keeps the
    original's name, signature and docstring (with a ``Deprecated.`` line put
    in front), so ``@macro`` registration, ``help()`` and the tests see the
    same function they always did.

    Args:
        replacement: what to use instead, as the reader should type it —
            ``"rfmux.tuning.find_bias_points + crs.apply_bias"``. Say
            ``"nothing yet"`` and explain in *note* when there is no
            replacement by decision.
        note: one more sentence for the warning and the docstring.
    """

    def decorate(func):
        where = f"{func.__module__.rsplit('.', 1)[-1]}.{func.__qualname__}"
        message = f"{where} is deprecated (legacy Periscope tuning path); use {replacement}."
        if note:
            message += f" {note}"

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                return await func(*args, **kwargs)
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                return func(*args, **kwargs)

        doc = inspect.getdoc(func) or ""
        head = f"Deprecated. Use {replacement}." + (f" {note}" if note else "")
        wrapper.__doc__ = head + ("\n\n" + doc if doc else "")
        wrapper.__deprecated_replacement__ = replacement
        return wrapper

    return decorate
