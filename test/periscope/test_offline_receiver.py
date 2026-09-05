"""Offline mode installs DummyReceiver in place of UDPReceiver; the status
bar polls it once a second, so it must answer every counter the real one
does."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope.app import DummyReceiver, Periscope  # noqa: E402
from rfmux.tools.periscope.tasks import UDPReceiver  # noqa: E402


def test_dummy_receiver_answers_every_counter_udp_receiver_has():
    counters = [name for name, attr in vars(UDPReceiver).items()
                if name.startswith("get_") and callable(attr)]
    assert counters, "UDPReceiver exposes no counters?"
    missing = [n for n in counters if not callable(getattr(DummyReceiver, n, None))]
    assert not missing, f"DummyReceiver lacks {missing}"


def test_offline_status_bar_tick_reads_a_healthy_empty_stream(qt_app):
    """One _update_performance_stats pass, as the GUI timer runs it."""
    p = Periscope.__new__(Periscope)
    p.receiver = DummyReceiver()
    p.t_last = 0.0
    p.prev_missing = p.prev_qdrops = p.prev_receive = 0
    p.frame_cnt = p.pkt_cnt = 0
    p.is_mock_mode = False
    p.default_packet_loss_color = "black"
    for name in ("fps_label", "pps_label", "packet_loss_label",
                 "info_text", "dropped_label"):
        setattr(p, name, QtWidgets.QLabel())

    p._update_performance_stats(2.0)

    assert p.dropped_label.text() == "| Lost: 0 missed / 0 dropped"
    assert p.info_text.text() == "", "an empty offline stream is not a fault"
