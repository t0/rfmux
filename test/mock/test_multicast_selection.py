"""The mock streams the way real hardware does, or explains why it cannot.

Real boards always multicast. The mock did too by design -- its socket
is configured for it and MockCRSStreamer defaults to the group -- but
two callers overrode the destination to 127.0.0.1, which made mock mode
exercise a different kernel path from production. That difference is not
academic: unicast to loopback hands each datagram to exactly one
SO_REUSEPORT socket, so a second reader starves, which cannot happen on
hardware.

Multicast is also the thing people most often have trouble with, so mock
mode is where it should be diagnosable: when it does not work here, the
failing step and its remedy are printed, and the mock falls back rather
than refusing to run.
"""

import socket

import pytest

from rfmux.mock.udp_streamer import (LOOPBACK_UNICAST, MockCRSStreamer,
                                     select_stream_destination)
from rfmux.streamer import (MULTICAST_GROUP, PFB_STREAMER_PORT,
                            STREAMER_PORT, MulticastCheck,
                            check_multicast_loopback)


@pytest.mark.portable
def test_the_check_reports_every_step_it_took():
    """Structure, not outcome -- CI machines differ on multicast."""
    check = check_multicast_loopback()
    assert isinstance(check.ok, bool)
    assert check.steps, "the check reported no steps at all"
    names = [name for name, _, _ in check.steps]
    assert names[0] == "interface", \
        "the interface must be checked first: if it silently is not " \
        "loopback, mock packets leave the machine"
    if check.ok:
        assert all(ok for _, ok, _ in check.steps)
        assert names[-1] == "receive"
    else:
        assert check.hint, "a failure with no hint helps nobody"
        assert not check.steps[-1][1], "failed overall but no step failed"


@pytest.mark.portable
def test_a_failure_explains_itself():
    """The report is the whole point of falling back rather than dying."""
    check = MulticastCheck(
        False,
        [("interface", True, "loopback"), ("receive", False, "nothing came back")],
        "  ip link show lo",
    )
    report = check.report()
    assert "FAIL" in report and "receive" in report
    assert "ip link show lo" in report


@pytest.mark.portable
def test_multicast_is_used_when_it_works(monkeypatch):
    monkeypatch.setattr("rfmux.streamer.check_multicast_loopback",
                        lambda **kw: MulticastCheck(True, [("receive", True, "ok")], ""))
    assert select_stream_destination() == MULTICAST_GROUP


@pytest.mark.portable
def test_falls_back_to_unicast_and_says_why(monkeypatch, capsys):
    monkeypatch.setattr(
        "rfmux.streamer.check_multicast_loopback",
        lambda **kw: MulticastCheck(
            False, [("receive", False, "nothing came back")],
            "  sudo ip link set lo multicast on"),
    )
    assert select_stream_destination() == LOOPBACK_UNICAST

    out = capsys.readouterr().out
    assert "falling back" in out.lower()
    assert "sudo ip link set lo multicast on" in out, \
        "fell back without telling the user how to fix multicast"
    assert "no fallback" in out.lower(), \
        "must say the fallback is mock-only -- a real CRS has none"


@pytest.mark.portable
def test_multicast_can_be_switched_off_without_probing(monkeypatch):
    def explode(**kw):
        raise AssertionError("probed despite use_multicast=False")

    monkeypatch.setattr("rfmux.streamer.check_multicast_loopback", explode)
    assert select_stream_destination(use_multicast=False) == LOOPBACK_UNICAST


@pytest.mark.portable
def test_mock_traffic_cannot_leave_this_host():
    """TTL 0 is the containment: the mock shares a group with real boards.

    _make_multicast_socket touches no instance state, so it is called
    unbound rather than standing up a whole streamer.
    """
    sock = MockCRSStreamer._make_multicast_socket(None)
    try:
        ttl = sock.getsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL)
        assert ttl == 0, (
            f"multicast TTL is {ttl}, so simulated detectors can reach the "
            f"network real CRS boards stream on"
        )
    finally:
        sock.close()


@pytest.mark.portable
def test_the_probe_does_not_eat_the_real_stream(monkeypatch):
    """It runs on an ephemeral port, so a live capture is untouched.

    Checked by recording the ports the probe binds rather than by
    holding the streamer port: with SO_REUSEPORT a second listener on
    9876 would take a share of any stream running on this machine.
    """
    ports = []
    real_bind = socket.socket.bind

    def recording_bind(self, address):
        ports.append(address[1])
        return real_bind(self, address)
    monkeypatch.setattr(socket.socket, "bind", recording_bind)

    check_multicast_loopback()
    assert ports, "the probe bound nothing, so nothing was checked"
    assert STREAMER_PORT not in ports and PFB_STREAMER_PORT not in ports
