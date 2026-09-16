"""The provisioned reference notebooks: the shipped folders, read-only,
with the repository's docs beside them when the install has them."""

import os

from rfmux import paths


def test_provisioning_puts_the_docs_beside_the_notebooks(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    (docs / "guides").mkdir(parents=True)
    (docs / "guides" / "pulse-capture.md").write_text("# guide\n")
    (docs / "make_figures.py").write_text("print()\n")
    monkeypatch.setattr(paths, "_DOCS", docs)
    monkeypatch.setattr(paths, "get_rfmux_data_dir", lambda: tmp_path / "data")

    dest = paths.get_reference_notebook_dir()
    assert (dest / "Demos" / "pulse_capture.md").is_file()
    guide = dest / paths.DOCS_FOLDER / "guides" / "pulse-capture.md"
    assert guide.is_file()
    assert not (dest / paths.DOCS_FOLDER / "make_figures.py").exists()
    assert not os.access(guide, os.W_OK)
    # Provisioned once per version: a second call is the same folder.
    assert paths.get_reference_notebook_dir() == dest


def test_without_docs_the_notebooks_alone_are_provisioned(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_DOCS", tmp_path / "absent")
    monkeypatch.setattr(paths, "get_rfmux_data_dir", lambda: tmp_path / "data")
    dest = paths.get_reference_notebook_dir()
    assert (dest / "Demos").is_dir()
    assert not (dest / paths.DOCS_FOLDER).exists()
