"""Board operations, including applying a catalog's bias points.

Importing these modules registers their macros on ``CRS``.
"""

from . import apply_bias

__all__ = [
    "apply_bias",
]
