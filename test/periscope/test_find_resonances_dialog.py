"""The isolation switch reaches the algorithm."""

import pytest

pytest.importorskip("PyQt6")

from test.qt_helpers import spin  # noqa: E402

from rfmux.tools.periscope.extract_params import ParamKeyExtractor  # noqa: E402
from rfmux.tools.periscope.find_resonances_dialog import (  # noqa: E402
    FindResonancesDialog,
)


def test_isolation_round_trips(qt_app):
    dlg = FindResonancesDialog()

    # Off by default: the separation keeps the meaning it always had.
    assert dlg.require_isolation_check.isChecked() is False
    assert dlg.get_parameters()["require_isolation"] is False

    dlg.require_isolation_check.setChecked(True)
    assert dlg.get_parameters()["require_isolation"] is True

    dlg.close()
    spin(qt_app)


def test_every_dialog_key_is_a_find_resonances_argument():
    """The dialog's output is splatted into find_resonances, so a key it
    does not accept is a TypeError at the end of a network analysis."""
    import inspect

    from rfmux.algorithms.measurement.fitting import find_resonances

    accepted = set(inspect.signature(find_resonances).parameters)
    keys = ParamKeyExtractor(
        "rfmux.tools.periscope.find_resonances_dialog",
        "FindResonancesDialog").extract()
    assert keys <= accepted, sorted(keys - accepted)
