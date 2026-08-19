#!/usr/bin/env -S JUPYTER_PLATFORM_DIRS=1 pytest-3 -s -v
"""
Notebook-Based Tests
====================

These tests execute in Jupyter Notebooks, which is a better environment than
pytest for quantitative tests where visualizations are helpful.

How this works
--------------

Tests are invoked using pytest as you'd expect, from the repo root or from
this directory:

$ pytest test/notebooks/
$ ./test_notebooks.py

The 'test_jupytext_notebook' test case is parameterized, and causes any
JupyterLab notebooks in this directory to be executed. One pytest test is
executed per notebook. If all cells in a given notebook run successfully, the
test is considered as a "pass". Any exceptions raised in any notebook cells
cause the associated test to fail.

'test_reference_demo_notebook' does the same for the notebooks shipped in
rfmux/reference-notebooks/Demos/. Those are user-facing documentation, and
executing them is what stops them drifting away from the API they describe.
They acquire data through a MockCRS server, so they are marked
slow_acquisition and excluded from the default run:

$ pytest -m slow_acquisition test/notebooks/

Viewing Test Results
--------------------

Files matching the pattern test_jupytext_notebook*.ipynb are test results, and
can be viewed like any other .ipynb file. (Modifications to these files are not
saved - you can edit and run them as normal, but should expect them to be
overwritten.)

Modifying Tests
---------------

Test notebooks are stored as .md files and converted into .ipynb files
automatically using jupytext. They can be converted to .ipynb files as follows:

$ jupytext -o filename.ipynb filename.md

You can then start a JupyterLab instance and mofiy the .ipynb file. This file
must be re-converted to Markdown in order to be used as a test case. Only .md
versions of jupyterlab notebooks should be stored in version control.

You can convert a notebook stored in .md format to .ipynb as follows:

$ jupytext -o filename.ipynb filename.md

...and you can convert an .ipynb file back to its .md representation as
follows:

$ jupytext -o filename.md filename.ipynb
"""

import pathlib

import pytest
import jupytext

# Guarded so a checkout without the test group skips the notebook tests
# instead of aborting collection for the whole suite.
nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

import rfmux

HERE = pathlib.Path(__file__).parent

# Glob relative to this file, not the working directory: pytest is normally
# invoked from the repo root, where a bare "test*.md" matches nothing and the
# whole parameter set silently collapses to empty ("got empty parameter set").
NOTEBOOKS = sorted(p.name for p in HERE.glob("test*.md"))

# The shipped reference notebooks are documentation, but executing them is the
# only thing that keeps them honest: they drift silently otherwise, because
# nothing else imports them. Located through the package rather than the repo
# layout so this works from an installed rfmux as well as a checkout.
DEMOS = pathlib.Path(rfmux.__file__).parent / "reference-notebooks" / "Demos"
DEMO_NOTEBOOKS = sorted(p.name for p in DEMOS.glob("*.md"))


@pytest.mark.parametrize("notebook_file", NOTEBOOKS)
def test_jupytext_notebook(request, notebook_file):
    with open(HERE / notebook_file, "r", encoding="utf-8") as f:
        notebook = jupytext.read(f)

    # Notebooks reference the repo by relative path, so execute them with this
    # directory as the kernel's working directory regardless of pytest's cwd.
    client = nbclient.NotebookClient(
        notebook, timeout=600, kernel_name="python3", resources={
            "metadata": {"path": str(HERE)}})

    # Run the notebook and store the results in an .ipynb file, regardless of
    # success/failure.
    try:
        client.execute()
    except Exception as e:
        raise AssertionError(
            f"Notebook execution failed! Check {request.node.name}.ipynb for details."
        ) from e
    finally:
        with open(HERE / f"{request.node.name}.ipynb", "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)


@pytest.mark.slow_acquisition
@pytest.mark.parametrize("notebook_file", DEMO_NOTEBOOKS)
def test_reference_demo_notebook(request, tmp_path, notebook_file):
    """Execute a shipped reference notebook end to end.

    Acquisition-tier: these spawn a MockCRS server and stream real UDP over
    loopback, so they take minutes and bind the streamer ports. Never run
    them alongside another acquisition test — two MockCRS servers on 9876/9877
    starve each other and the failure looks like a detector bug.

    The kernel runs in tmp_path so the capture files land there instead of in
    the package tree.
    """
    with open(DEMOS / notebook_file, "r", encoding="utf-8") as f:
        notebook = jupytext.read(f)

    client = nbclient.NotebookClient(
        notebook, timeout=1800, kernel_name="python3", resources={
            "metadata": {"path": str(tmp_path)}})

    result = tmp_path / f"{request.node.name}.ipynb"
    try:
        client.execute()
    except Exception as e:
        raise AssertionError(
            f"Reference notebook {notebook_file} failed! See {result}"
        ) from e
    finally:
        with open(result, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)
