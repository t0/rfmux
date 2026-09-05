"""Operation algorithms: putting the array into a state.

The sibling of ``algorithms.measurement``, and the distinction is what the
call is *for*. A measurement asks the array a question and hands back the
answer — a netanal, a multisweep, a capture. An operation tells the array how
to be and returns nothing: it either put the board into the state you asked
for or it raised saying why it could not.

That is why nothing in here has a return value to inspect. A caller who wants
to know what the board is doing reads it back with the getters; a caller who
wants to know whether the operation worked has already been told, by the
exception that did not happen.

These imports exist for their SIDE EFFECT, not to re-export names: each
module registers its ``@macro(CRS)`` functions onto the CRS class at import
time, which is what makes ``await crs.apply_bias(...)`` resolve.  Drop one and
its macro quietly stops existing.
"""

from . import apply_bias

__all__ = [
    "apply_bias",
]
