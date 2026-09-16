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


def test_a_version_provisioned_earlier_gains_the_docs(tmp_path, monkeypatch):
    """An install whose version string was provisioned before the docs
    came along still gets the Guides folder on the next call."""
    import rfmux
    docs = tmp_path / "docs"
    (docs / "guides").mkdir(parents=True)
    (docs / "guides" / "g.md").write_text("# g\n")
    monkeypatch.setattr(paths, "_DOCS", docs)
    monkeypatch.setattr(paths, "get_rfmux_data_dir", lambda: tmp_path / "data")
    dest = tmp_path / "data" / "reference-notebooks" / rfmux.__version__
    (dest / "Demos").mkdir(parents=True)          # as an older rfmux left it
    assert paths.get_reference_notebook_dir() == dest
    guide = dest / paths.DOCS_FOLDER / "guides" / "g.md"
    assert guide.is_file() and not os.access(guide, os.W_OK)


def test_without_docs_the_notebooks_alone_are_provisioned(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_DOCS", tmp_path / "absent")
    monkeypatch.setattr(paths, "get_rfmux_data_dir", lambda: tmp_path / "data")
    dest = paths.get_reference_notebook_dir()
    assert (dest / "Demos").is_dir()
    assert not (dest / paths.DOCS_FOLDER).exists()
