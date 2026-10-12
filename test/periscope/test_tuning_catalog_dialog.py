"""The Tuning Catalog window: a table of the catalog that edits a copy."""

import pytest

pytest.importorskip("PyQt6")

from rfmux.core.resonators import ResonatorCatalog  # noqa: E402
from rfmux.core.transferfunctions import convert_dbm_to_amplitude  # noqa: E402
from rfmux.tools.periscope.tuning_catalog_dialog import (  # noqa: E402
    AMPLITUDE, CHANNEL, FIT_COLUMNS, FREQUENCY, NAME, NONLINEAR_QI, NONLINEARITY,
    NOTES, POWER, SKEWED_QI, TuningCatalogDialog)

SCALE = -0.75


def _catalog():
    catalog = ResonatorCatalog.from_frequencies(
        [1.0e9, 1.1e9, 1.2e9], module=1, amplitude=0.004)
    first = catalog.names()[0]
    catalog[first].update_bias_point(amplitude=0.004, dI_df=1e-6, dQ_df=0.0)
    return catalog


def _dialog(qt_app, catalog=None, read_only=False):
    dialog = TuningCatalogDialog()
    dialog.set_catalog(catalog or _catalog(), SCALE, read_only=read_only)
    return dialog


def _row(dialog, name):
    for row in range(dialog.table.rowCount()):
        if dialog.table.item(row, NAME).text() == name:
            return row
    raise KeyError(name)


def test_each_resonator_is_a_row(qt_app):
    catalog = _catalog()
    dialog = _dialog(qt_app, catalog)

    assert dialog.table.rowCount() == len(catalog)
    for resonator in catalog:
        row = _row(dialog, resonator.name)
        assert dialog.table.item(row, CHANNEL).text() == str(resonator.channel)
        assert float(dialog.table.item(row, FREQUENCY).text()) * 1e6 == \
            pytest.approx(resonator.bias.frequency_hz, abs=1)


def test_name_and_channel_are_read_only(qt_app):
    from PyQt6 import QtCore
    dialog = _dialog(qt_app)
    editable = QtCore.Qt.ItemFlag.ItemIsEditable
    assert not dialog.table.item(0, NAME).flags() & editable
    assert not dialog.table.item(0, CHANNEL).flags() & editable
    assert dialog.table.item(0, AMPLITUDE).flags() & editable


def test_an_edit_changes_the_copy_and_clears_the_calibration(qt_app):
    catalog = _catalog()
    first = catalog.names()[0]
    dialog = _dialog(qt_app, catalog)
    dialog.table.item(_row(dialog, first), AMPLITUDE).setText("0.002")

    assert dialog.catalog()[first].bias.amplitude == 0.002
    assert dialog.catalog()[first].bias.dI_df is None
    assert catalog[first].bias.amplitude == 0.004      # the original stays


def test_a_power_in_dbm_sets_the_amplitude_with_the_dac_scale(qt_app):
    dialog = _dialog(qt_app)
    name = dialog.catalog().names()[1]
    dialog.table.item(_row(dialog, name), POWER).setText("-50")

    assert dialog.catalog()[name].bias.amplitude == pytest.approx(
        convert_dbm_to_amplitude(-50, SCALE))


def test_a_value_the_catalog_refuses_is_reported_and_not_kept(qt_app):
    dialog = _dialog(qt_app)
    name = dialog.catalog().names()[0]
    dialog.table.item(_row(dialog, name), AMPLITUDE).setText("2")

    assert dialog.catalog()[name].bias.amplitude == 0.004
    assert name in dialog.status.text()


def test_apply_sends_the_edited_copy(qt_app):
    dialog = _dialog(qt_app)
    name = dialog.catalog().names()[2]
    dialog.table.item(_row(dialog, name), NOTES).setText("weak")
    applied = []
    dialog.catalog_applied.connect(applied.append)
    dialog.apply_button.click()

    assert applied[0][name].notes["comment"] == "weak"


def test_deleted_rows_leave_the_copy(qt_app):
    dialog = _dialog(qt_app)
    name = dialog.catalog().names()[0]
    dialog.table.selectRow(_row(dialog, name))
    dialog.delete_button.click()

    assert name not in dialog.catalog()
    assert dialog.table.rowCount() == 2


def test_a_read_only_catalog_cannot_be_edited_or_applied(qt_app):
    from PyQt6 import QtCore
    dialog = _dialog(qt_app, read_only=True)
    assert not dialog.apply_button.isEnabled()
    assert not dialog.table.item(0, AMPLITUDE).flags() & \
        QtCore.Qt.ItemFlag.ItemIsEditable


def test_csv_export_and_import_keep_the_bias_points(qt_app, tmp_path):
    catalog = _catalog()
    dialog = _dialog(qt_app, catalog)
    path = tmp_path / "catalog.csv"
    dialog.export_csv(str(path))

    other = _dialog(qt_app, ResonatorCatalog.from_frequencies(
        [2.0e9], module=1, amplitude=0.001))
    other.import_csv(str(path))
    assert other.catalog().names() == catalog.names()
    for resonator in catalog:
        assert other.catalog()[resonator.name].bias.frequency_hz == \
            pytest.approx(resonator.bias.frequency_hz)



def test_other_notes_show_in_the_tooltip_and_stay(qt_app):
    catalog = _catalog()
    name = catalog.names()[0]
    catalog[name].notes.update(comment="weak", flagged="noisy")
    dialog = _dialog(qt_app, catalog)
    item = dialog.table.item(_row(dialog, name), NOTES)

    assert "flagged: noisy" in item.toolTip()
    item.setText("weaker")
    assert dialog.catalog()[name].notes == {"comment": "weaker", "flagged": "noisy"}




def _with_fits(catalog, name, models):
    """Give *name* a bias sweep that carries fits of *models*, as Find Bias
    does."""
    import numpy as np
    params = {
        "skewed": {"fr": 1.0e9, "Qr": 2e4, "Qi": 1e5, "Qc": 2.5e4},
        "nonlinear": {"fr": 1.0e9, "Qr": 2.1e4, "Qi": 9e4, "Qc": 2.7e4,
                      "a": 0.3},
    }
    f = np.linspace(0.999e9, 1.001e9, 5)
    catalog[name].update_bias_point(
        bias_sweep={"frequencies": f, "iq_volts": np.ones(5, complex),
                    "fits": {m: {"params": params[m]} for m in models}})


def test_before_find_bias_only_the_nonlinearity_column_shows(qt_app):
    """Run Fit is not enough: the catalog has fits only from Find Bias."""
    dialog = _dialog(qt_app)
    hidden = dialog.table.isColumnHidden

    assert all(hidden(c) for c, _ in FIT_COLUMNS["skewed"] + FIT_COLUMNS["nonlinear"])
    assert not hidden(NONLINEARITY)


def test_each_model_of_the_bias_sweep_has_its_own_columns(qt_app):
    catalog = _catalog()
    name = catalog.names()[0]
    _with_fits(catalog, name, ("skewed", "nonlinear"))
    dialog = _dialog(qt_app, catalog)
    row = _row(dialog, name)

    assert dialog.table.item(row, SKEWED_QI).text() == "1e+05"
    assert dialog.table.item(row, NONLINEAR_QI).text() == "9e+04"
    assert dialog.table.item(row, NONLINEARITY).text() == "0.3"


def test_only_the_models_that_were_fitted_show(qt_app):
    catalog = _catalog()
    _with_fits(catalog, catalog.names()[0], ("skewed",))
    dialog = _dialog(qt_app, catalog)
    hidden = dialog.table.isColumnHidden

    assert not any(hidden(c) for c, _ in FIT_COLUMNS["skewed"])
    assert all(hidden(c) for c, _ in FIT_COLUMNS["nonlinear"])
    assert not hidden(NONLINEARITY)

def test_save_and_open_round_trip_the_catalog(qt_app, tmp_path, monkeypatch):
    from rfmux.tuning import store
    monkeypatch.setattr(store, "_output_directory", tmp_path)
    catalog = _catalog()
    dialog = _dialog(qt_app, catalog)
    dialog.save()
    path = next(tmp_path.rglob("catalog_*.pkl"))

    other = _dialog(qt_app, ResonatorCatalog.from_frequencies(
        [2.0e9], module=1, amplitude=0.001))
    other.open_catalog(str(path))
    assert other.catalog().to_dict() == catalog.to_dict()


def test_a_catalog_of_another_module_is_not_opened(qt_app, tmp_path, monkeypatch):
    from rfmux.tuning import save_catalog, store
    monkeypatch.setattr(store, "_output_directory", tmp_path)
    path = save_catalog(ResonatorCatalog.from_frequencies(
        [1.0e9], module=3, amplitude=0.004))
    dialog = _dialog(qt_app)
    dialog.open_catalog(path)

    assert dialog.catalog().module == 1
    assert "module 3" in dialog.status.text()
