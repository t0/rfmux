"""
Periscope Subpackage
====================

This subpackage initializes the Periscope tool, part of the rfmux suite.
It serves as the primary entry point for accessing Periscope's functionalities,
including its main application class, programmatic invocation methods, and key UI components.

The Periscope tool provides a real-time multi-pane viewer and network analysis
capabilities for RF (Radio Frequency) systems.
"""

# --- Core Application and Entry Points ---
# Expose the main application class and the programmatic entry point for Periscope.
# 'Periscope' is the main class for the application.
# 'raise_periscope' is a function to launch the GUI programmatically.
# 'cli_main' is the entry point for command-line interface usage.
from .app import Periscope
from .__main__ import raise_periscope, main as cli_main

# --- UI Components ---
# Expose key UI components that might be used externally or by other tools
# within the rfmux ecosystem or for custom integrations.
from .ui import (
    NetworkAnalysisWindow,
    MultisweepWindow,
    NetworkAnalysisDialog,
    InitializeCRSDialog,
    FindResonancesDialog,
    MultisweepDialog
)

# --- Public API Definition ---
# Defines the public interface of the Periscope subpackage.
# Only these names will be imported when `from rfmux.tools.periscope import *` is used.
__all__ = [
    "Periscope",                # Main application class
    "raise_periscope",          # Function to launch Periscope GUI
    "cli_main",                 # Main function for CLI access
    "NetworkAnalysisWindow",    # UI window for network analysis
    "MultisweepWindow",         # UI window for multisweep operations
    "NetworkAnalysisDialog",    # Dialog for network analysis settings
    "InitializeCRSDialog",      # Dialog for CRS initialization
    "FindResonancesDialog",     # Dialog for finding resonances
    "MultisweepDialog",         # Dialog for multisweep settings
]
