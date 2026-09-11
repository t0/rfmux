"""Resources and YAML finalization hooks belong to their session/load."""
import pytest

from rfmux.core.hardware_map import HardwareMap
from rfmux.core.session import YAMLLoader

pytestmark = pytest.mark.portable


def test_finalization_hooks_do_not_leak_between_loads():
    calls = []
    first = YAMLLoader("[]")
    second = YAMLLoader("[]")
    try:
        first.register_finalization_hook(lambda hwm: calls.append(hwm))
        first.get_single_data()
        second.get_single_data()
        assert calls == [None]
    finally:
        first.dispose()
        second.dispose()


def test_close_runs_all_cleanup_even_if_one_fails():
    session = HardwareMap()
    calls = []

    def fail():
        raise RuntimeError("cleanup failed")

    session().on_close(lambda: calls.append("closed"))
    session().on_close(fail)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        session.close()
    session.close()
    assert calls == ["closed"]
