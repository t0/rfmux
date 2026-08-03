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
import nbformat
import nbclient

HERE = pathlib.Path(__file__).parent

# Glob relative to this file, not the working directory: pytest is normally
# invoked from the repo root, where a bare "test*.md" matches nothing and the
# whole parameter set silently collapses to empty ("got empty parameter set").
NOTEBOOKS = sorted(p.name for p in HERE.glob("test*.md"))


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
