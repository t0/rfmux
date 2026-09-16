"""The provisioned reference notebooks: the shipped folders, read-only,
with the repository's guides and release notes beside them."""

import os

import rfmux
from rfmux import paths


def _docs(tmp_path):
    """A docs/ folder as a working checkout may hold it: the tracked
    pieces and the leftovers that must not be provisioned."""
    docs = tmp_path / "docs"
    (docs / "guides" / "images").mkdir(parents=True)
    (docs / "guides" / "pulse-capture.md").write_text("# guide\n")
    (docs / "guides" / "images" / "a.png").write_bytes(b"\x89PNG")
    (docs / "guides" / ".ipynb_checkpoints").mkdir()
    (docs / "installation.md").write_text("# install\n")
    (docs / "release-notes").mkdir()
    (docs / "release-notes" / "2026-09.md").write_text("# notes\n")
    (docs / "make_figures.py").write_text("print()\n")
    (docs / "Old Demos").mkdir()
    (docs / "stray.ipynb").write_text("{}")
    return docs


def _provision(tmp_path, monkeypatch, docs):
    monkeypatch.setattr(paths, "_DOCS", docs)
    monkeypatch.setattr(paths, "get_rfmux_data_dir", lambda: tmp_path / "data")
    return paths.get_reference_notebook_dir()


def test_the_tracked_docs_are_provisioned_flat_beside_the_notebooks(
        tmp_path, monkeypatch):
    dest = _provision(tmp_path, monkeypatch, _docs(tmp_path))
    assert (dest / "Demos" / "pulse_capture.md").is_file()
    guides = dest / paths.DOCS_FOLDER
    assert (guides / "pulse-capture.md").is_file()
    assert (guides / "images" / "a.png").is_file()
    assert (guides / "installation.md").is_file()
    assert not os.access(guides / "pulse-capture.md", os.W_OK)
    # The release notes join the shipped walkthroughs.
    notes = dest / "Release Notes"
    assert (notes / "2026-09.md").is_file()
    assert any(p.suffix == ".ipynb" for p in notes.iterdir())
    # Nothing else from docs/, and no nested copy of it.
    assert sorted(p.name for p in guides.iterdir()) == \
        ["images", "installation.md", "pulse-capture.md"]
    assert not (dest / "stray.ipynb").exists()
    assert not (dest / "Old Demos").exists()
    # Provisioned once per version: a second call is the same folder.
    assert paths.get_reference_notebook_dir() == dest


def test_a_version_provisioned_earlier_gains_the_docs(tmp_path, monkeypatch):
    """An install whose version string was provisioned before the docs
    came along still gets them on the next call."""
    docs = _docs(tmp_path)
    dest = tmp_path / "data" / "reference-notebooks" / rfmux.__version__
    (dest / "Demos").mkdir(parents=True)          # as an older rfmux left it
    assert _provision(tmp_path, monkeypatch, docs) == dest
    assert (dest / paths.DOCS_FOLDER / "pulse-capture.md").is_file()
    assert (dest / "Release Notes" / "2026-09.md").is_file()


def test_a_copy_of_the_whole_docs_tree_is_redone(tmp_path, monkeypatch):
    """A Guides folder that mirrors docs/ (a nested lower-case guides/
    inside it) is replaced by the flat layout."""
    docs = _docs(tmp_path)
    dest = tmp_path / "data" / "reference-notebooks" / rfmux.__version__
    old = dest / paths.DOCS_FOLDER / "guides"
    old.mkdir(parents=True)
    (old / "pulse-capture.md").write_text("# old\n")
    (dest / paths.DOCS_FOLDER / "stray.ipynb").write_text("{}")
    os.chmod(old, 0o500)                          # read-only, as provisioned
    _provision(tmp_path, monkeypatch, docs)
    guides = dest / paths.DOCS_FOLDER
    assert not (guides / "guides").exists() and not (guides / "stray.ipynb").exists()
    assert (guides / "pulse-capture.md").read_text() == "# guide\n"


def test_without_docs_the_notebooks_alone_are_provisioned(tmp_path, monkeypatch):
    dest = _provision(tmp_path, monkeypatch, tmp_path / "absent")
    assert (dest / "Demos").is_dir()
    assert not (dest / paths.DOCS_FOLDER).exists()
