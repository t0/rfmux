# Periscope Module Structure

`periscope.py` used to contain all of the implementation for the real-time viewer.
To make the code easier to navigate, the logic has been split into a few helper
modules:

- `periscope_utils.py` – constants and general utility functions used across the
  application.
- `periscope_tasks.py` – worker threads and QRunnable classes that handle
  background processing.
- `periscope_ui.py` – network analysis dialogs and related user interface
  classes.

The main entry point remains `periscope.py` which imports these helpers and
exposes the same API as before.
