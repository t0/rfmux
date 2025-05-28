"""
MockCRS Hardware Emulation System - Main Interface

This module provides the main interface for the MockCRS hardware emulation system.
It implements a complete rfmux-compatible mock hardware that can be used for testing
and development without requiring physical CRS hardware.

Features:
- Full CRS API compatibility via Tuber protocol
- Realistic Kinetic Inductance Detector (KID) physics
- Power-dependent frequency shifts and nonlinearity
- UDP packet streaming for real-time data
- Integration with Periscope GUI
- Support for all measurement algorithms (network analysis, multisweep, etc.)

Usage:
    Add to hardware map YAML:
    ```yaml
    !HardwareMap
    - !flavour "rfmux.core.mock"
    - !CRS { serial: "0001", hostname: "127.0.0.1" }
    ```

    Then use normally:
    ```python
    import rfmux
    s = rfmux.load_session(hardware_map_yaml)
    crs = s.query(rfmux.CRS).one()
    await crs.resolve()
    # Use crs as normal - all methods work
    ```

Configuration:
    Customize behavior by modifying rfmux/core/mock_constants.py:
    - Resonator properties (frequency range, Q factors, coupling)
    - Power dependence parameters (kinetic inductance, saturation)
    - Physics model settings (bifurcation, debug flags)

See Also:
    - README_MockCRS.md: Comprehensive documentation
    - mock_constants.py: Configuration parameters
    - mock_resonator_model.py: Physics implementation
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
