import os

__all__ = []

if "CRS_EMBEDDED" in os.environ:
    # Bypass periscope imports. The embedded environment on the CRS doesn't
    # have e.g. pyqt6, and it wouldn't make sense there anyhow.
    pass
else:
    from . import periscope

    __all__.append("periscope")
