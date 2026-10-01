"""The shipped reference notebooks execute end to end.

So do the test*.md notebooks beside this file.  The notebooks in
rfmux/reference-notebooks/Demos/ are user-facing
documentation; running them is what stops them drifting away from the
API they describe.  They are jupytext markdown: edit them in JupyterLab
after ``jupytext -o demo.ipynb demo.md`` and convert back before
committing.  The executed .ipynb of a failed run is written to the
test's tmp_path, and the tail of what its cells printed is in the
assertion message.
"""
import pathlib
import shutil

import pytest
import jupytext

import rfmux

# Guarded so a checkout without the test group skips the notebook tests
# instead of aborting collection for the whole suite.
nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

HERE = pathlib.Path(__file__).parent
# Relative to this file: from the repo root a bare "test*.md" matches
# nothing and the parameter set collapses to empty.
NOTEBOOKS = sorted(p.name for p in HERE.glob("test*.md"))

# Located through the package rather than the repo layout so this works
# from an installed rfmux as well as a checkout.
DEMOS = pathlib.Path(rfmux.__file__).parent / "reference-notebooks" / "Demos"
DEMO_NOTEBOOKS = sorted(p.name for p in DEMOS.glob("*.md"))


def _printed(notebook, lines: int = 120) -> str:
    """The tail of what the executed cells printed.  The saved .ipynb is
    out of reach on a CI runner, and the cell that fails is often not the
    one that went wrong."""
    text = "".join(
        f"--- cell {k}\n{out.get('text', '')}"
        for k, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "stream")
    return "\n".join(text.splitlines()[-lines:])


@pytest.mark.parametrize("notebook_file", NOTEBOOKS)
def test_jupytext_notebook(request, notebook_file):
    with open(HERE / notebook_file, "r", encoding="utf-8") as f:
        notebook = jupytext.read(f)

    # Notebooks reference the repo by relative path, so execute them with this
    # directory as the kernel's working directory regardless of pytest's cwd.
    client = nbclient.NotebookClient(
        notebook, timeout=600, kernel_name="python3", resources={
            "metadata": {"path": str(HERE)}})
    try:
        client.execute()
    except Exception as e:
        raise AssertionError(
            f"{notebook_file} failed; its cells printed:\n{_printed(notebook)}"
        ) from e
    finally:
        with open(HERE / f"{request.node.name}.ipynb", "w",
                  encoding="utf-8") as f:
            nbformat.write(notebook, f)


@pytest.mark.parametrize("notebook_file", [
    # The fastrx demo writes and reads its own synthetic recording: no
    # server, no stream, so it runs in the quick tier.
    (name if name == "fastrx_recording.md"
     else pytest.param(name, marks=pytest.mark.slow_acquisition))
    for name in DEMO_NOTEBOOKS])
def test_reference_demo_notebook(request, tmp_path, monkeypatch,
                                 notebook_file):
    """Acquisition-tier demos spawn a MockCRS server and stream real UDP
    over loopback, so they bind the streamer ports: never run them
    alongside another acquisition test.  The kernel runs in tmp_path with
    the shipped Python companions, so imports match a copied demo folder
    and the capture files land there instead of in the package tree."""
    with open(DEMOS / notebook_file, "r", encoding="utf-8") as f:
        notebook = jupytext.read(f)
    for helper in DEMOS.glob("*.py"):
        shutil.copy2(helper, tmp_path / helper.name)
    client = nbclient.NotebookClient(
        notebook, timeout=1800, kernel_name="python3", resources={
            "metadata": {"path": str(tmp_path)}})
    # Exercise the workbook autosave in the isolated test store.
    monkeypatch.setenv("RFMUX_AUTOSAVE", "1")
    try:
        client.execute()
    except Exception as e:
        raise AssertionError(
            f"{notebook_file} failed; its cells printed:\n{_printed(notebook)}"
        ) from e
    finally:
        with open(tmp_path / f"{request.node.name}.ipynb", "w",
                  encoding="utf-8") as f:
            nbformat.write(notebook, f)
