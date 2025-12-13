"""Main plot panel for Periscope - contains TOD/IQ/FFT/PSD plots with controls."""

from .utils import *

class MainPlotPanel(QtWidgets.QWidget, ScreenshotMixin):
    """
    Dockable panel containing the main TOD/IQ/FFT/PSD plots for Periscope.
    
    This panel includes the toolbar, configuration options, and plot grid.
    It can be docked, floated, or tabbed alongside other analysis panels.
    """
    
    def __init__(self, parent_window, chan_str="1"):
        """
        Initialize the MainPlotPanel.
        
        Args:
            parent_window: The main Periscope window (for accessing methods and state)
            chan_str: Initial channel string for the channel input field
        """
        super().__init__(parent_window)
        
        # Store reference to parent Periscope window
        self.periscope = parent_window
        
        # Create layout for the panel
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Add toolbar
        self._add_toolbar(layout, chan_str)
        
        # Add configuration panel
        self._add_config_panel(layout)
        
        # Create the plot grid container
        self.container = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.container)
        layout.addWidget(self.container)
        
    def get_grid(self):
        """Get the grid layout for adding plots."""
        return self.grid
    
    def _add_toolbar(self, layout, chan_str):
        """Add the toolbar with channel/buffer controls and plot type checkboxes."""
        toolbar_widget = QtWidgets.QWidget()
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)

        # Use references from parent Periscope window
        toolbar_layout.addWidget(QtWidgets.QLabel("Channels:"))
        toolbar_layout.addWidget(self.periscope.e_ch)
        toolbar_layout.addSpacing(20)
        toolbar_layout.addWidget(QtWidgets.QLabel("Buffer:"))
        toolbar_layout.addWidget(self.periscope.e_buf)
        toolbar_layout.addWidget(self.periscope.b_pause)
        toolbar_layout.addSpacing(30)
        
        # Add unit radio buttons
        toolbar_layout.addWidget(QtWidgets.QLabel("Units:"))
        toolbar_layout.addWidget(self.periscope.rb_counts)
        toolbar_layout.addWidget(self.periscope.rb_real_units)
        toolbar_layout.addWidget(self.periscope.rb_df_units)
        toolbar_layout.addSpacing(30)
        
        # Add plot type checkboxes
        for cb in (self.periscope.cb_time, self.periscope.cb_iq, self.periscope.cb_fft, 
                   self.periscope.cb_ssb, self.periscope.cb_dsb):
            toolbar_layout.addWidget(cb)
        toolbar_layout.addStretch(1)
        
        layout.addWidget(toolbar_widget)
    
    def _add_config_panel(self, layout):
        """Add the configuration panel with action buttons."""
        # Action buttons row
        action_buttons_widget = QtWidgets.QWidget()
        action_buttons_layout = QtWidgets.QHBoxLayout(action_buttons_widget)
        action_buttons_layout.setContentsMargins(0, 0, 0, 0)

        # Add all the action buttons from parent
        action_buttons_layout.addWidget(self.periscope.btn_init_crs)
        action_buttons_layout.addWidget(self.periscope.btn_netanal)
        action_buttons_layout.addWidget(self.periscope.btn_load_multi)
        action_buttons_layout.addWidget(self.periscope.btn_load_bias)
        action_buttons_layout.addWidget(self.periscope.btn_toggle_cfg)
        
        # Add mock-specific buttons if in mock mode
        if self.periscope.is_mock_mode:
            action_buttons_layout.addWidget(self.periscope.btn_reconfigure_mock)
            action_buttons_layout.addWidget(self.periscope.btn_qp_pulses)
        
        # Add screenshot button
        self.screenshot_btn = QtWidgets.QPushButton("📷")
        self.screenshot_btn.setToolTip("Export a screenshot of this panel to the session folder (or choose location)")
        self.screenshot_btn.clicked.connect(self._export_screenshot)
        action_buttons_layout.addWidget(self.screenshot_btn)
        
        layout.addWidget(action_buttons_widget)

        # Add the collapsible configuration panel
        layout.addWidget(self.periscope.ctrl_panel)
