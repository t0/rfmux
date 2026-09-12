"""record_streams: the coordination (the parser is up before the capture
starts; the recording window opens when noise training ends and lasts
the duration), the session folder, and the bias export lookup.  The
last test drives the real thing against a MockCRS, with the parser as a
subprocess.
"""

import asyncio
import signal
import sys
import time
from types import SimpleNamespace

import click
import pytest

from rfmux.algorithms.measurement import record_streams as rs
from rfmux.core import session_folder as core_session
from rfmux.core.session_folder import load_metadata
from rfmux.pulse_capture.capture_session import PulseCaptureConfig
from test.record_helpers import bias_export as _bias_export, fake_fastrx

TRAIN_S = 0.3
DURATION_S = 0.4
PARSER_UP_S = 0.1


class _Board:
    """A board whose trigger_capture trains for TRAIN_S and then runs for
    time_run, calling on_noise between the two."""
    tuber_hostname = "rfmux0000.local"

    def __init__(self, dies=False):
        self.dies = dies
        self.calls = []

    async def get_decimation(self):
        return 6

    async def trigger_capture(self, **kw):
        self.calls.append(kw)
        self.t_started = time.time()
        await asyncio.sleep(TRAIN_S)
        if self.dies == "before training":
            raise RuntimeError("stream ended")
        kw["on_noise"]({})
        self.t_trained = time.time()
        if self.dies == "after training":
            await asyncio.sleep(kw["time_run"] / 4)
            raise RuntimeError("disk full")
        await asyncio.sleep(kw["time_run"])
        return "capture-result"


@pytest.fixture
def fake_recorders(monkeypatch):
    """The parser subprocess replaced by a log of start and stop times;
    it takes PARSER_UP_S to come up, as the real one takes seconds."""
    log = {}

    async def start(host, interface, wanted, dirfile, logfile):
        log["start"] = time.time()
        log["cmd"] = (host, interface, wanted)
        dirfile.mkdir()
        (dirfile / "serial_0042").mkdir()
        logfile.write_text("")
        handle = rs._Parser(SimpleNamespace(returncode=None), asyncio.Event(), None)

        def up():
            handle.up = True
            handle.ready.set()
        asyncio.get_running_loop().call_later(PARSER_UP_S, up)
        return handle

    async def stop(handle, result, dirfile, logfile):
        log["stop"] = time.time()
        result.dirfile_path = dirfile / "serial_0042"

    monkeypatch.setattr(rs, "_start_parser", start)
    monkeypatch.setattr(rs, "_stop_parser", stop)
    # The parser is faked, so its pygetdata requirement is too.
    monkeypatch.setattr(rs.importlib.util, "find_spec", lambda name: object())
    return log


def test_the_window_opens_when_training_ends_and_lasts_the_duration(
        tmp_path, fake_recorders):
    board = _Board()
    session = core_session.open_session(base=tmp_path)
    t0 = time.time()
    result = asyncio.run(rs.record_streams(
        board, module=2, channels=[3, 1, 2], duration_s=DURATION_S,
        session=session, fastrx=False, verbose=False))

    log = fake_recorders
    # The parser is launched first, the capture once it is up, the
    # window once the capture has trained, for the duration.
    assert t0 <= log["start"] <= board.t_started
    assert board.t_started - log["start"] >= PARSER_UP_S - 0.05
    assert board.t_trained <= result.started_at <= log["stop"]
    assert DURATION_S - 0.05 <= log["stop"] - result.started_at < DURATION_S + 1.0
    assert log["cmd"] == ("127.0.0.1", None, {2: [1, 2, 3]})
    call = board.calls[0]
    assert call["time_run"] == DURATION_S and call["streamer_mode"] == "slow"
    assert call["hdf5_path"].name.startswith("pulse_module2_")
    assert result.capture == "capture-result"
    assert result.dirfile_path.name == "serial_0042"
    assert result.warnings == []


def test_without_a_capture_the_window_opens_once_the_parser_listens(
        tmp_path, fake_recorders):
    board = _Board()
    t0 = time.time()
    result = asyncio.run(rs.record_streams(
        board, module=1, channels=[1], duration_s=DURATION_S,
        session=core_session.open_session(base=tmp_path), capture=False, fastrx=False,
        verbose=False))
    assert board.calls == []
    assert result.started_at - t0 >= PARSER_UP_S - 0.05
    assert DURATION_S - 0.05 <= fake_recorders["stop"] - result.started_at < (
        DURATION_S + 1.0)
    assert result.pulse_path is None and result.training_s == 0.0


def test_a_capture_that_never_trains_opens_no_window(
        tmp_path, fake_recorders, monkeypatch):
    async def hold(*_):
        pytest.fail("the recording window opened")
    monkeypatch.setattr(rs, "_hold", hold)
    with pytest.raises(RuntimeError, match="stream ended"):
        asyncio.run(rs.record_streams(
            _Board(dies="before training"), module=1, channels=[1],
            duration_s=DURATION_S, session=core_session.open_session(base=tmp_path),
            fastrx=False, verbose=False))
    # The parser was launched and is stopped again.
    assert fake_recorders["start"] <= fake_recorders["stop"]


# A parser child without rfmux: up at once, a dirfile, and the real
# parser's exit on SIGINT (its statistics, then exit).
FAKE_PARSER = """
import os, signal, sys, time
print('parser up', file=sys.stderr, flush=True)
a = sys.argv[1:]
os.makedirs(os.path.join(a[a.index('-d') + 1], 'serial_0042'))
def bye(*_):
    print('=== Drop Statistics ===', file=sys.stderr, flush=True)
    sys.exit(0)
signal.signal(signal.SIGINT, bye)
while True:
    time.sleep(0.02)
"""


@pytest.fixture
def fake_parser_child(monkeypatch):
    if sys.platform == "win32":
        pytest.skip("the parser is a Linux tool: an asyncio subprocess "
                    "stopped with SIGINT, neither of which Windows has here")
    monkeypatch.setattr(rs, "PARSER_CHILD", FAKE_PARSER)
    monkeypatch.setattr(rs.importlib.util, "find_spec", lambda name: object())


def test_a_parser_that_dies_on_start_stops_the_run_before_the_capture(
        tmp_path, fake_parser_child, monkeypatch):
    monkeypatch.setattr(rs, "PARSER_CHILD",
                        "import sys; sys.exit('no such interface')")
    board = _Board()
    with pytest.raises(RuntimeError, match="before it was up: no such interface"):
        asyncio.run(rs.record_streams(
            board, module=1, channels=[1], duration_s=DURATION_S,
            session=core_session.open_session(base=tmp_path), fastrx=False,
            verbose=False))
    assert board.calls == []


def test_the_parser_subprocess_is_started_stopped_and_logged(
        tmp_path, fake_parser_child):
    board = _Board()
    result = asyncio.run(rs.record_streams(
        board, module=2, channels=[1, 2], duration_s=DURATION_S,
        session=core_session.open_session(base=tmp_path), fastrx=False, verbose=False))
    assert result.dirfile_path.name == "serial_0042"
    log = result.parser_log.read_text()
    assert log.startswith("parser up") and "Drop Statistics" in log
    assert board.t_trained <= result.started_at
    assert result.warnings == []


def test_a_capture_failing_mid_window_stops_the_parser_cleanly(
        tmp_path, fake_parser_child):
    """The recording ends through its stop event, not by cancelling
    the cleanup: the parser still gets its SIGINT and is reaped."""
    session = core_session.open_session(base=tmp_path)
    t0 = time.time()
    # The board fails a quarter of the way into a long window: the
    # run ends then, not when the window would have.
    with pytest.raises(RuntimeError, match="disk full"):
        asyncio.run(rs.record_streams(
            _Board(dies="after training"), module=1, channels=[1],
            duration_s=4.0, session=session, fastrx=False,
            verbose=False))
    assert time.time() - t0 < 4.0
    run = load_metadata(session)["recordings"][0]
    assert run["dirfile"].endswith("serial_0042")
    log = (session / run["dirfile"]).parent.with_suffix(".log").read_text()
    assert "Drop Statistics" in log


def test_a_recorder_failing_after_training_raises_its_own_error(
        tmp_path, fake_recorders, monkeypatch):
    """The capture is cancelled to end the run; the error raised is the
    recorder's, not that cancellation."""
    async def hold(*_):
        raise RuntimeError("writer gone")
    monkeypatch.setattr(rs, "_hold", hold)
    session = core_session.open_session(base=tmp_path)
    with pytest.raises(RuntimeError, match="writer gone"):
        asyncio.run(rs.record_streams(
            _Board(), module=1, channels=[1], duration_s=DURATION_S,
            session=session, fastrx=False, verbose=False))


def test_a_missing_fastrxd_socket_is_refused_before_the_capture(
        tmp_path, fake_recorders):
    pytest.importorskip("rfmux.fastrx")
    board = _Board()
    session = core_session.open_session(base=tmp_path)
    with pytest.raises(RuntimeError, match="is fastrxd running"):
        asyncio.run(rs.record_streams(
            board, module=1, channels=[1], duration_s=DURATION_S,
            session=session, fastrx_socket=str(tmp_path / "no-daemon"),
            verbose=False))
    assert board.calls == []


def test_the_recording_covers_the_window_and_reports_its_stats(
        tmp_path, fake_recorders, monkeypatch):
    fx = fake_fastrx(monkeypatch, tmp_path)
    session = core_session.open_session(base=tmp_path)
    board = _Board()
    result = asyncio.run(rs.record_streams(
        board, module=2, channels=[3, 9], duration_s=DURATION_S,
        session=session, merge_fastrx=False, verbose=False))
    [w] = fx.writers
    # Channels 1 to the highest, from the end of training, through
    # the daemon's socket.
    assert (w.channels, w.socket) == (9, fx.socket)
    assert w.path == result.fastrx_path
    assert result.fastrx_path.name.startswith("fastrx_module2_")
    assert board.t_trained <= result.started_at
    assert result.fastrx_stats == {"packets": 0, "overruns": 0, "dropouts": 0}
    assert result.warnings == [
        "no channel-stream packets: is the channel streamer on for "
        "module(s) [2]?"]
    meta = load_metadata(session)
    assert meta["recordings"][0]["fastrx_stats"] == result.fastrx_stats
    assert ("fastrx", "module2") in [
        (e["data_type"], e["identifier"]) for e in meta["exports"]]


def test_a_disk_too_small_for_the_recording_warns(
        tmp_path, fake_recorders, monkeypatch):
    fake_fastrx(monkeypatch, tmp_path, packets=5000)
    monkeypatch.setattr(rs.shutil, "disk_usage",
                        lambda p: SimpleNamespace(free=1))
    session = core_session.open_session(base=tmp_path)
    result = asyncio.run(rs.record_streams(
        _Board(), module=1, channels=[1], duration_s=DURATION_S,
        session=session, capture=False, merge_fastrx=False, verbose=False))
    assert result.fastrx_stats["packets"] == 5000
    assert result.warnings == [
        f"0 GB free in {session} for a recording of about 0 GB"]


def test_a_parser_that_ignores_sigint_is_terminated_and_no_dirfile_warns(
        tmp_path, monkeypatch):
    class Proc:
        returncode = None

        def __init__(self):
            self.signals = []

        def send_signal(self, s):
            self.signals.append(s)

        def terminate(self):
            self.signals.append("terminate")
            self.returncode = -15

        async def wait(self):
            while self.returncode is None:
                await asyncio.sleep(0.01)

    monkeypatch.setattr(rs, "PARSER_EXIT_S", 0.05)
    log = tmp_path / "p.log"
    log.write_text("parser up\nbind: no such interface\n")
    result = _result(tmp_path)
    proc = Proc()

    async def stop():
        await rs._stop_parser(rs._Parser(proc, asyncio.Event(), None),
                              result, tmp_path / "p.dirfile", log)
    asyncio.run(stop())
    assert proc.signals == [signal.SIGINT, "terminate"]
    assert result.dirfile_path is None
    assert result.warnings == [
        "the parser wrote no dirfile: parser up | bind: no such interface"]


def _result(tmp_path, **kw):
    return rs.RecordResult(session=tmp_path, module=1, channels=[1],
                           duration_s=1.0, training_s=0.0, **kw)


def test_the_recording_is_merged_into_the_pulse_file(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rs, "_merge", lambda p, f: calls.append((p, f)))
    pulse, fx = tmp_path / "pulse.h5", tmp_path / "run.fastrx"
    pulse.touch(); fx.touch()
    result = _result(tmp_path, pulse_path=pulse, fastrx_path=fx)
    rs._merge_recording(result)
    assert calls == [(pulse, fx)] and result.merged_fastrx
    assert result.warnings == []


def test_a_merge_that_fails_is_a_warning(tmp_path, monkeypatch):
    def boom(p, f):
        raise ValueError("no disciplined timestamp")
    monkeypatch.setattr(rs, "_merge", boom)
    pulse, fx = tmp_path / "pulse.h5", tmp_path / "run.fastrx"
    pulse.touch(); fx.touch()
    result = _result(tmp_path, pulse_path=pulse, fastrx_path=fx)
    rs._merge_recording(result)
    assert not result.merged_fastrx
    assert result.warnings == [
        "fastrx not merged into pulse.h5: no disciplined timestamp"]


def test_nothing_is_merged_without_both_products(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rs, "_merge", lambda p, f: calls.append((p, f)))
    pulse = tmp_path / "pulse.h5"
    pulse.touch()
    rs._merge_recording(_result(tmp_path, pulse_path=pulse))
    assert calls == []


def test_pulse_summary_lines_name_the_busiest_channel_first():
    capture = SimpleNamespace(primary=SimpleNamespace(summaries={
        1: {1: {"snr": 6.0}},
        2: {1: {"snr": 8.0}, 2: {"snr": 5.5}},
        3: {}}))
    assert rs.pulse_summary_lines(capture) == [
        "channel 2: 2 pulses, best 8.0\u03c3",
        "channel 1: 1 pulse, best 6.0\u03c3",
        "3 pulses on 2 of 3 channels"]
    capture = SimpleNamespace(primary=SimpleNamespace(summaries={
        (2, 5): {1: {"snr": 6.0}}, (3, 1): {}}))
    assert rs.pulse_summary_lines(capture)[0] == \
        "module 2 channel 5: 1 pulse, best 6.0\u03c3"


def test_a_run_across_modules_is_keyed_by_pairs_and_named_for_them(
        tmp_path, fake_recorders):
    board = _Board()
    session = core_session.open_session(base=tmp_path)
    result = asyncio.run(rs.record_streams(
        board, module=None, channels={3: [2, 1], 2: [5]},
        duration_s=DURATION_S, session=session, fastrx=False, verbose=False))
    assert fake_recorders["cmd"] == ("127.0.0.1", None, {2: [5], 3: [1, 2]})
    call = board.calls[0]
    assert call["channel"] == {2: [5], 3: [1, 2]} and call["module"] is None
    assert call["hdf5_path"].name.startswith("pulse_modules2+3_")
    assert result.module is None and result.modules == [2, 3]
    assert result.channels == [(2, 5), (3, 1), (3, 2)]
    meta = load_metadata(session)
    assert meta["recordings"][-1]["modules"] == [2, 3]
    assert meta["recordings"][-1]["channels"] == [[2, 5], [3, 1], [3, 2]]
    assert meta["exports"][-1]["identifier"] == "modules2+3"


def test_a_mapping_of_one_module_is_a_plain_one_module_run(
        tmp_path, fake_recorders):
    board = _Board()
    result = asyncio.run(rs.record_streams(
        board, module=None, channels={2: [3, 1]}, duration_s=DURATION_S,
        session=core_session.open_session(base=tmp_path), fastrx=False, verbose=False))
    assert board.calls[0]["channel"] == [1, 3] and board.calls[0]["module"] == 2
    assert result.module == 2 and result.channels == [1, 3]


def test_the_parser_gets_one_range_per_module(tmp_path, monkeypatch):
    """The real command line: one -c MODULE:RANGE per module."""
    argv = []

    class Stderr:
        lines = [b"parser up\n", b""]

        async def readline(self):
            return self.lines.pop(0)

    async def fake_exec(*cmd, **kw):
        argv.extend(cmd)
        return SimpleNamespace(stderr=Stderr(), returncode=None)

    monkeypatch.setattr(rs.asyncio, "create_subprocess_exec", fake_exec)

    async def run():
        handle = await rs._start_parser("127.0.0.1", None, {2: [5], 3: [1, 2]},
                                        tmp_path / "p.dirfile", tmp_path / "p.log")
        await handle.ready.wait()
        await handle.pump
        return handle.up

    assert asyncio.run(run())
    assert argv[argv.index("-d") + 1:] == [
        str(tmp_path / "p.dirfile"), "-c", "2:5", "-c", "3:1-2", "--drop-stats"]


def test_the_channel_streamer_is_turned_on_first_when_asked(
        tmp_path, fake_recorders, monkeypatch):
    fx = fake_fastrx(monkeypatch, tmp_path, modules_seen=0b0001)
    monkeypatch.setattr(rs, "CHANNEL_STREAMER_SETTLE_S", 0.0)
    board = _Board()
    board.streamer = []

    async def set_channel_streamer(**kw):
        board.streamer.append(kw)
    board.set_channel_streamer = set_channel_streamer
    # Every recorded module, channels 1 to the highest of any of them
    # rounded up to the board's multiple of 16, before the probe: module
    # 2's stream still missing is still refused.
    with pytest.raises(RuntimeError, match=r"module\(s\) \[2\]"):
        asyncio.run(rs.record_streams(
            board, module=None, channels={1: [1, 9], 2: [3]}, duration_s=0.1,
            session=core_session.open_session(base=tmp_path), sample_trunc="MID",
            channel_streamer=True, fastrx_socket=fx.socket, verbose=False))
    assert board.streamer == [
        {"channels": 16, "module": 1, "sample_trunc": "MID"},
        {"channels": 16, "module": 2, "sample_trunc": "MID"}]
    assert board.calls == []
    # A board without the call says so rather than failing inside it.
    with pytest.raises(RuntimeError, match="no channel streamer"):
        asyncio.run(rs.record_streams(
            _Board(), module=1, channels=[1], duration_s=0.1,
            session=core_session.open_session(base=tmp_path), channel_streamer=True,
            fastrx_socket=fx.socket, verbose=False))


def test_a_module_the_channel_stream_lacks_is_refused_before_the_run(
        tmp_path, fake_recorders, monkeypatch):
    fake_fastrx(monkeypatch, tmp_path, modules_seen=0b0001)
    board = _Board()
    with pytest.raises(RuntimeError, match=r"module\(s\) \[2\]"):
        asyncio.run(rs.record_streams(
            board, module=None, channels={1: [1], 2: [1]}, duration_s=0.1,
            session=core_session.open_session(base=tmp_path), verbose=False))
    assert board.calls == []


def test_the_command_takes_per_module_ranges_and_bias_exports(
        tmp_path, monkeypatch):
    from rfmux.tools import record
    folder = tmp_path / "session_x"
    folder.mkdir()
    _bias_export(folder / "bias_module2_1.pkl", 2, [1, 2])
    _bias_export(folder / "bias_module3_1.pkl", 3, [7])
    seen = {}

    async def fake_main(serial, hostname, **kw):
        seen.update(kw)
        return rs.RecordResult(session=folder, module=kw["module"],
                               channels=[], duration_s=1.0, training_s=0.0)

    monkeypatch.setattr(record, "_main", fake_main)
    common = dict(serial="0156", hostname=None, duration=1.0,
                  session=str(folder), session_dir=".", capture=True,
                  parser=False, fastrx=False, parser_interface=None,
                  fastrx_interface=None, fastrx_socket=None,
                  merge_fastrx=False, show="none", bias=None,
                  config=PulseCaptureConfig(), quiet=True)
    # Per-module ranges name the modules; the exports give calibrations.
    record._run(modules=[1], channels="3:5,2:1-2", **common)
    assert seen["module"] is None
    assert seen["channels"] == {2: [1, 2], 3: [5]}
    assert set(seen["tuning"]) == {(2, 1), (2, 2), (3, 7)}
    assert seen["tuning"][(3, 7)]["nco_frequency_hz"] == 1.0e9
    # Several modules with no ranges: each module's newest export.
    record._run(modules=[2, 3], channels=None, **common)
    assert seen["channels"] == {2: [1, 2], 3: [7]}
    # One module keeps plain channel keys.
    record._run(modules=[2], channels=None, **common)
    assert seen["module"] == 2 and seen["channels"] == [1, 2]
    assert set(seen["tuning"]) == {1, 2}
    assert (seen["channel_streamer"], seen["sample_trunc"]) == (False, "LOW")
    record._run(modules=[2], channels=None, channel_streamer=True,
                sample_trunc="HIGH", **common)
    assert (seen["channel_streamer"], seen["sample_trunc"]) == (True, "HIGH")
    with pytest.raises(click.UsageError, match="no bias export for module 4"):
        record._run(modules=[2, 4], channels=None, **common)
    # Modules run 1-4, the same bound the parser and the wire have.
    with pytest.raises(click.UsageError, match="modules run 1-4"):
        record._run(modules=[5], channels="1-4", **common)
    with pytest.raises(click.UsageError, match="Modules run 1-4"):
        record._run(modules=[1], channels="5:1-4", **common)


def test_the_command_exits_on_a_failed_run_and_after_a_warning(
        tmp_path, monkeypatch, capsys):
    from rfmux.tools import record
    folder = tmp_path / "session_x"
    folder.mkdir()
    _bias_export(folder / "bias_module2_1.pkl", 2, [1])
    common = dict(serial="0156", hostname=None, duration=1.0,
                  session=str(folder), session_dir=".", capture=True,
                  parser=False, fastrx=False, parser_interface=None,
                  fastrx_interface=None, fastrx_socket=None,
                  merge_fastrx=False, show="none",
                  config=PulseCaptureConfig(), quiet=True)

    async def dies(serial, hostname, **kw):
        raise RuntimeError("the parser exited before it was up")
    monkeypatch.setattr(record, "_main", dies)
    with pytest.raises(click.ClickException, match="parser exited"):
        record._run(modules=[2], channels=None, bias=None, **common)

    async def warns(serial, hostname, **kw):
        return rs.RecordResult(session=folder, module=2, channels=[1],
                               duration_s=1.0, training_s=0.0,
                               warnings=["no channel-stream packets"])
    monkeypatch.setattr(record, "_main", warns)
    with pytest.raises(SystemExit) as info:
        record._run(modules=[2], channels=None, bias=None, **common)
    assert info.value.code == 1
    assert "[record] warning: no channel-stream packets" in \
        capsys.readouterr().err
    # --bias names one export, so one module.
    with pytest.raises(click.UsageError, match="one module's export"):
        record._run(modules=[2, 3], channels=None,
                    bias=str(folder / "bias_module2_1.pkl"), **common)


def test_show_names_the_viewer_without_a_display_and_launches_it_with_one(
        tmp_path, monkeypatch, capsys):
    from rfmux.tools import record
    monkeypatch.setattr(record.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    capture = SimpleNamespace(summaries={1: {}, 2: {1: {"snr": 6.0}}})
    result = rs.RecordResult(
        session=tmp_path, module=1, channels=[1, 2], duration_s=1.0,
        training_s=0.0, capture=capture, pulse_path=tmp_path / "pulse.h5",
        fastrx_path=tmp_path / "run.fastrx",
        dirfile_path=tmp_path / "run.dirfile" / "serial_0042")
    record._show(result, "none")
    assert capsys.readouterr().out == ""
    record._show(result, "periscope")
    assert capsys.readouterr().out == (
        "[record] no display; to review: -m rfmux.tools.periscope "
        f"--review {tmp_path / 'pulse.h5'}\n")
    record._show(result, "overlay")
    assert capsys.readouterr().out == (
        f"[record] no display; to view: rfmux fastrx overlay "
        f"{tmp_path / 'pulse.h5'} {tmp_path / 'run.fastrx'} --channel 2 "
        f"--pad 5 --dirfile {result.dirfile_path}\n")
    launched = []
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(record.subprocess, "Popen",
                        lambda cmd, **kw: launched.append((cmd, kw)))
    record._show(result, "periscope")
    assert launched == [(record.periscope_review_command(result.pulse_path),
                         {"start_new_session": True})]


def test_periscope_is_launched_on_the_pulse_file_in_review_mode(tmp_path):
    import sys
    from rfmux.tools.record import periscope_review_command
    assert periscope_review_command(tmp_path / "pulse.h5") == [
        sys.executable, "-m", "rfmux.tools.periscope", "--review",
        str(tmp_path / "pulse.h5")]


def test_a_bare_record_command_asks_the_dialog(monkeypatch):
    from click.testing import CliRunner
    from rfmux.tools import record
    from rfmux.tools.record_dialog import RecordDialog
    runs = []
    monkeypatch.setattr(record, "_run", lambda **kw: runs.append(kw))
    monkeypatch.setattr(RecordDialog, "ask", classmethod(lambda cls: None))
    assert CliRunner().invoke(record.cli, []).exit_code == 0
    assert runs == []
    monkeypatch.setattr(RecordDialog, "ask",
                        classmethod(lambda cls: {"serial": "0156"}))
    assert CliRunner().invoke(record.cli, []).exit_code == 0
    assert runs == [{"serial": "0156", "quiet": False}]


def test_options_without_a_serial_are_refused_not_dropped(monkeypatch):
    from click.testing import CliRunner
    from rfmux.tools import record
    from rfmux.tools.record_dialog import RecordDialog
    monkeypatch.setattr(RecordDialog, "ask",
                        classmethod(lambda cls: {"serial": "0156"}))
    result = CliRunner().invoke(record.cli, ["--module", "2"])
    assert result.exit_code == 2 and "--serial is required" in result.output


def test_products_are_listed_in_the_session_metadata(tmp_path, fake_recorders):
    session = core_session.open_session(base=tmp_path)
    assert session.name.startswith("session_")
    result = asyncio.run(rs.record_streams(
        _Board(), module=2, channels=[1], duration_s=DURATION_S,
        session=session, fastrx=False, verbose=False))
    meta = load_metadata(session)
    assert [(e["data_type"], e["identifier"]) for e in meta["exports"]] == [
        ("parser", "module2")]          # the fake capture wrote no file
    run = meta["recordings"][0]
    assert run["dirfile"] == str(result.dirfile_path.relative_to(session))
    assert run["duration_s"] == DURATION_S
    assert run["training_s"] == pytest.approx(
        PulseCaptureConfig().noise_train_ms / 1e3, rel=0.05)
    assert run["capture_config"]["threshold_sigma"] == 5.0


def test_an_existing_session_keeps_its_metadata(tmp_path):
    folder = tmp_path / "session_20260909_153654"
    folder.mkdir()
    (folder / core_session.METADATA_FILE).write_text('{"created": "then", "exports": [{"filename": "x"}]}')
    assert core_session.open_session(folder) == folder
    core_session.register_export(folder, "pulse_module2_1.h5", "pulse", "module2")
    meta = load_metadata(folder)
    assert meta["created"] == "then"
    assert [e["filename"] for e in meta["exports"]] == ["x", "pulse_module2_1.h5"]


def test_newest_bias_export_for_the_module_gives_channels_and_tuning(tmp_path):
    # Written newest first: a copied folder keeps no file times, so the
    # listing's timestamp decides; a file the listing lacks is not an
    # export.
    _bias_export(tmp_path / "bias_module2_120000.pkl", 2, [4, 5], calibrated=False,
                 timestamp="2026-09-09T12:00:00")
    _bias_export(tmp_path / "bias_module1_110000.pkl", 1, [7],
                 timestamp="2026-09-09T11:00:00")
    _bias_export(tmp_path / "bias_module2_100000.pkl", 2, [1, 2, 3],
                 timestamp="2026-09-09T10:00:00")

    newest = rs.latest_bias_export(tmp_path, 2)
    assert newest.name == "bias_module2_120000.pkl"
    chans, rows = rs.biased_channels(newest)
    assert chans == [4, 5] and set(rows) == {4, 5} and rs.calibrated(rows) == 0
    chans, rows = rs.biased_channels(tmp_path / "bias_module2_100000.pkl")
    assert chans == [1, 2, 3] and rs.calibrated(rows) == 3
    # The row is the export's entry, keyed by its channel, with the NCO
    # the tones were placed against.
    assert rows[2] == {"bias_channel": 2, "df_calibration": 2e6 - 1e5j,
                       "nco_frequency_hz": 1.0e9}
    assert rs.latest_bias_export(tmp_path, 3) is None
    (tmp_path / "bias_module3_130000.pkl").write_bytes(b"")
    assert rs.latest_bias_export(tmp_path, 3) is None


def test_the_requirements_are_checked_before_anything_runs(tmp_path, monkeypatch):
    session = core_session.open_session(base=tmp_path)
    board = _Board()
    monkeypatch.setattr(rs.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="pygetdata"):
        asyncio.run(rs.record_streams(board, module=1, channels=[1],
                                      duration_s=1.0, session=session,
                                      fastrx=False))
    with pytest.raises(ValueError, match="nothing selected"):
        asyncio.run(rs.record_streams(board, module=1, channels=[1],
                                      duration_s=1.0, session=session,
                                      capture=False, parser=False, fastrx=False))
    with pytest.raises(ValueError, match="not a session folder"):
        asyncio.run(rs.record_streams(board, module=1, channels=[1],
                                      duration_s=1.0, session=tmp_path,
                                      parser=False, fastrx=False))
    assert board.calls == []


# ── Against the simulator ──────────────────────────────────────────

@pytest.mark.slow_acquisition
def test_mock_capture_and_parser_cover_the_same_stretch(tmp_path):
    """The real coordination: trigger_capture on the mock's slow stream
    and the parser as a subprocess, both products in the session, the
    window opening after the capture's training span and the dirfile
    covering training and window."""
    pytest.importorskip("pygetdata")
    from rfmux.streamer import check_multicast_loopback
    if not check_multicast_loopback().ok:
        pytest.skip("the mock falls back to unicast, which feeds only "
                    "one of two listeners")
    from rfmux.mock.helpers import create_mock_crs
    from rfmux.pulse_capture.hdf5 import PulseHDF5Reader
    import pygetdata as gd

    config = PulseCaptureConfig(noise_train_ms=2000.0)
    duration = 2.0

    async def run():
        crs = await create_mock_crs(
            module=1, config={"num_resonances": 2, "resonator_random_seed": 3,
                              "auto_bias_kids": True}, verbose=False)
        await asyncio.sleep(2.0)
        try:
            return await rs.record_streams(
                crs, module=1, channels=[1, 2], duration_s=duration,
                session=core_session.open_session(base=tmp_path), fastrx=False,
                config=config, verbose=False)
        finally:
            await crs.stop_udp_streaming()

    result = asyncio.run(run())
    assert result.warnings == []
    assert result.training_s == pytest.approx(2.0, rel=0.1)

    with PulseHDF5Reader(result.pulse_path) as reader:
        assert reader.channels == [1, 2]
        rate = float(reader.metadata["sample_rate_slow"])
    assert result.dirfile_path.name == "serial_0000"
    df = gd.dirfile(str(result.dirfile_path), gd.RDONLY)
    # The parser drops the batch in flight when it is stopped, up to
    # 256 packets.
    assert (duration * rate <= df.nframes
            <= (result.training_s + duration) * rate + 256)
    assert "Drop Statistics" in result.parser_log.read_text()

    meta = load_metadata(result.session)
    assert sorted(e["data_type"] for e in meta["exports"]) == ["parser", "pulse"]
    run_meta = meta["recordings"][0]
    assert run_meta["started_at"] - result.capture.start_time >= 0.9 * result.training_s
