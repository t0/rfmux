"""The native catalog file: save_catalog and load_catalog."""

from pathlib import Path

import pytest

from rfmux.core import session_folder
from rfmux.core.resonators import ResonatorCatalog
from rfmux.tuning import load_catalog, save_catalog, store

pytestmark = pytest.mark.portable


@pytest.fixture(autouse=True)
def output_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "_output_directory", tmp_path)


def _catalog() -> ResonatorCatalog:
    catalog = ResonatorCatalog.from_frequencies([1.0e9, 1.1e9], module=2,
                                                amplitude=0.004)
    name = catalog.names()[0]
    catalog[name].update_bias_point(amplitude=0.004, dI_df=1e-6, dQ_df=2e-6,
                                    bias_power_dbm=-48.7)
    catalog[name].notes.update(comment="weak", flagged="noisy")
    return catalog


def test_a_saved_catalog_loads_with_its_calibration_and_notes():
    catalog = _catalog()
    path = save_catalog(catalog)

    assert load_catalog(path).to_dict() == catalog.to_dict()


def test_a_saved_catalog_is_listed_in_the_session_folder(tmp_path):
    path = Path(save_catalog(_catalog()))

    listed = session_folder.exports(path.parent, "catalog")
    assert [entry["filename"] for entry in listed] == [path.name]


def test_a_bias_catalog_file_loads_as_a_catalog(tmp_path):
    catalog = _catalog()
    path = store.maybe_save({"measurement": "bias_catalog", "module": 2,
                             "catalog": catalog.to_dict()},
                            "bias_catalog", save=True)

    assert load_catalog(path).to_dict() == catalog.to_dict()


def test_a_file_with_no_catalog_is_refused(tmp_path):
    path = store.maybe_save({"measurement": "netanal"}, "netanal", save=True)

    with pytest.raises(ValueError, match="not a catalog file"):
        load_catalog(path)
