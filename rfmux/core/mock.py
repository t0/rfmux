"""
Shim module for the rfmux.core.mock flavour.

This module re-exports components from the new modularized mock system
to maintain compatibility with the `!flavour "rfmux.core.mock"` directive
in YAML hardware maps.
"""

# Import the yaml_hook from the new mock_server module
from .mock_server import yaml_hook

# Import base CRS from schema - client should use base class, not MockCRS
from .schema import CRS

# Import schema elements that were part of the original mock.py's __all__
from .schema import (
    Crate,
    ReadoutModule,
    ReadoutChannel,
    Wafer,
    Resonator,
    ChannelMapping
)

# Don't import enums on client side - they come through Tuber as properties
# The server-side MockCRS exposes these as properties that return TuberResult-compatible objects

# Define __all__ to specify the public API of this mock flavour module
__all__ = [
    "CRS",
    "Crate",
    "ReadoutModule",
    "ReadoutChannel",
    "Wafer",
    "Resonator",
    "ChannelMapping",
    "yaml_hook"
]
