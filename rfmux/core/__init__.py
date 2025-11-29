"""
RFMux Core Package

This package provides the core schema and CRS interface for rfmux.
Mock functionality has been moved to the separate rfmux.mock package.

To use mock hardware, add to your hardware map YAML:
    !flavour "rfmux.mock"
"""

# Import schema elements that are part of the public API
from .schema import (
    CRS,  # Base CRS class from schema
    Crate,
    ReadoutModule,
    ReadoutChannel,
    Wafer,
    Resonator,
    ChannelMapping
)


__all__ = [
    "CRS",
    "Crate",
    "ReadoutModule",
    "ReadoutChannel",
    "Wafer",
    "Resonator",
    "ChannelMapping",
]
