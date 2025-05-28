"""
RFMux Core Package
"""

# Import schema elements that are part of the public API
from .schema import (
    Crate,
    ReadoutModule,
    ReadoutChannel,
    Wafer,
    Resonator,
    ChannelMapping
    # BaseCRS is also in schema, but we usually export the concrete CRS
)

# Import the main mock CRS class (aliased as CRS) and the yaml_hook
# from their new module locations.
# This assumes that when 'rfmux.core.mock' was the flavour,
# 'from rfmux.core import CRS' would give MockCRS.
# If the flavour system points to this __init__.py for 'rfmux.core',
# then this setup should work.

# Check if we are in a "mock" environment.
# This is a common pattern, but the exact mechanism might vary.
# For now, assume that if rfmux.core.mock_server can be imported,
# we intend to use the mock versions.
# A more robust system might use an environment variable or a config setting.
try:
    from .mock_server import yaml_hook
    from .mock_crs_core import CRS  # This is MockCRS aliased as CRS
    MOCK_ENABLED = True
except ImportError:
    # If mock modules can't be imported, fall back to real CRS
    # or raise an error if mock is explicitly required.
    # This part depends on how non-mock CRS is structured.
    # For now, let's assume if mock isn't there, we might be using real.
    # If 'rfmux.core.crs.CRS' is the real one:
    # from .crs import CRS 
    # yaml_hook would not be available in non-mock scenarios or would be a no-op.
    # For this refactoring, we are focused on the mock implementation.
    # If only mock is intended to be exposed this way, then the ImportError
    # should likely be re-raised or handled as a configuration error.
    
    # For the purpose of this task, we assume mock is primary.
    # If mock_server or mock_crs_core is missing, something is wrong with the setup.
    print("Warning: MockCRS components not found. Ensure mock_server.py and mock_crs_core.py are present.")
    # Define CRS and yaml_hook as None or raise error if they are essential
    CRS = None 
    yaml_hook = None
    MOCK_ENABLED = False


__all__ = [
    "CRS",  # Should be MockCRS if MOCK_ENABLED, otherwise the real CRS
    "Crate",
    "ReadoutModule",
    "ReadoutChannel",
    "Wafer",
    "Resonator",
    "ChannelMapping",
]

if MOCK_ENABLED and yaml_hook is not None:
    __all__.append("yaml_hook")
else:
    # If not mock_enabled, yaml_hook might not be relevant or available.
    # Or, if a non-mock yaml_hook exists, it should be handled here.
    pass

# Clean up to avoid polluting namespace if not MOCK_ENABLED and imports failed
if not MOCK_ENABLED:
    if "CRS" in locals() and CRS is None: del CRS
    if "yaml_hook" in locals() and yaml_hook is None: del yaml_hook
    # This cleanup might be too aggressive depending on desired fallback behavior.
