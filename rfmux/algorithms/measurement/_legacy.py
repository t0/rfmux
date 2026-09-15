"""Deprecation warnings for legacy tuning functions.

New tuning code uses :mod:`rfmux.tuning` for analysis and ``crs.apply_bias``
for programming tones.
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
    """Warn on calls to a legacy sync or async function.

    Preserves function metadata and prepends a deprecation notice to its
    docstring. ``replacement`` names the recommended API; ``note`` adds an
    optional explanation to both the warning and docstring.
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
