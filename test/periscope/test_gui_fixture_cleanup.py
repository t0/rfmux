"""What a GUI test's fixtures held is gone before the next test runs.

A widget is a reference cycle, so only the cyclic collector frees it;
the conftest runs that collector once pytest has let go of the test's
fixture values.  Without that, a fixture-held widget outlives its test
in generation 2 and is freed, with every other one that piled up
there, whenever a full collection happens to run.  These two tests run
in this order."""
import gc
import weakref

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets  # noqa: E402

_held = []


@pytest.fixture(scope="module", autouse=True)
def _no_full_collections():
    """Only the conftest's own collection may free the widget: a small
    session would otherwise reach a full collection by itself."""
    old = gc.get_threshold()
    gc.set_threshold(old[0], old[1], 10 ** 9)
    yield
    gc.set_threshold(*old)


@pytest.fixture
def widget(qt_app):
    w = QtWidgets.QWidget()
    w.cycle = w
    _held.append(weakref.ref(w))
    yield w


def test_a_fixture_holds_a_widget(widget):
    assert widget.cycle is widget


def test_the_widget_is_freed_before_the_next_test(qt_app):
    assert _held and _held[0]() is None


class _Raises(QtWidgets.QWidget):
    def closeEvent(self, event):
        raise RuntimeError("from a Qt callback")


@pytest.mark.xfail(strict=True, raises=pytest.fail.Exception,
                   reason="an exception in a Qt callback fails the test")
def test_an_exception_in_a_qt_callback_fails_the_test(qt_app):
    w = _Raises()
    w.close()
