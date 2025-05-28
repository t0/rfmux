"""Combined UI components for Periscope."""

# Imports from within the 'periscope' subpackage
from .utils import *  # For any base Qt classes or utilities needed by UI components
from .tasks import *  # For any task-related signals or enums UI components might need

from .dialogs import ( # Assuming periscope_dialogs.py becomes dialogs.py
    NetworkAnalysisDialogBase,
    NetworkAnalysisDialog,
    NetworkAnalysisParamsDialog,
    InitializeCRSDialog,
    FindResonancesDialog,
    MultisweepDialog,
)
# network_analysis_window.py and multisweep_window.py are sibling modules now
from .network_analysis_window import NetworkAnalysisWindow
from .multisweep_window import MultisweepWindow

__all__ = [
    # From dialogs.py
    "NetworkAnalysisDialogBase",
    "NetworkAnalysisDialog",
    "NetworkAnalysisParamsDialog",
    "InitializeCRSDialog",
    "FindResonancesDialog",
    "MultisweepDialog",
    
    # From window files
    "NetworkAnalysisWindow",
    "MultisweepWindow",

    # Potentially re-exporting things from .utils or .tasks if they are considered part of the UI's public API
    # For example, if UI elements directly use constants from .utils:
    "QtWidgets", "QtCore", "QFont", "QIcon", "pg", # Common Qt/PyQtGraph items from utils
    "ClickableViewBox", # Custom UI element from utils
    # Signal objects from .tasks if UI elements connect to them directly (though usually app class handles this)
    "IQSignals", "PSDSignals", "NetworkAnalysisSignals", "CRSInitializeSignals", "MultisweepSignals",
]
