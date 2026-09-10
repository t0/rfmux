"""record_streams: the coordination (the parser is up before the capture
starts; the recording window opens when noise training ends and lasts
the duration), the session folder, and the bias export lookup.  The
last test drives the real thing against a MockCRS, with the parser as a
subprocess.
"""

import asyncio
import pickle
import time
from types import SimpleNamespace

import pytest

from rfmux.algorithms.measurement import record_streams as rs
from rfmux.pulse_capture.capture_session import PulseCaptureConfig

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

    async def start(host, interface, module, channels, dirfile, logfile):
        log["start"] = time.time()
        log["cmd"] = (host, interface, module, channels)
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
    return log


def test_the_window_opens_when_training_ends_and_lasts_the_duration(
        tmp_path, fake_recorders):
    board = _Board()
    session = rs.open_session(base=tmp_path)
    t0 = time.time()
    result = asyncio.run(rs.record_streams(
        board, module=2, channels=[3, 1, 2], duration_s=DURATION_S,
        session=session, fastrx=False, verbose=False))

    log = fake_recorders
    # The parser is launched first and the capture waits until it is up.
    assert log["start"] == pytest.approx(t0, abs=0.05)
    assert board.t_started == pytest.approx(t0 + PARSER_UP_S, abs=0.05)
    assert result.started_at == pytest.approx(board.t_trained, abs=0.05)
    assert log["stop"] - result.started_at == pytest.approx(DURATION_S, abs=0.1)
    assert log["cmd"] == ("127.0.0.1", None, 2, [1, 2, 3])
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
        session=rs.open_session(base=tmp_path), capture=False, fastrx=False,
        verbose=False))
    assert board.calls == []
    assert result.started_at == pytest.approx(t0 + PARSER_UP_S, abs=0.05)
    assert fake_recorders["stop"] - result.started_at == pytest.approx(
        DURATION_S, abs=0.1)
    assert result.pulse_path is None and result.training_s == 0.0


def test_a_capture_that_never_trains_opens_no_window(tmp_path, fake_recorders):
    with pytest.raises(RuntimeError, match="stream ended"):
        asyncio.run(rs.record_streams(
            _Board(dies="before training"), module=1, channels=[1],
            duration_s=DURATION_S, session=rs.open_session(base=tmp_path),
            fastrx=False, verbose=False))
    # The parser was launched and is stopped again at once.
    assert fake_recorders["stop"] - fake_recorders["start"] < (
        PARSER_UP_S + TRAIN_S + 0.1)


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
            session=rs.open_session(base=tmp_path), fastrx=False,
            verbose=False))
    assert board.calls == []


def test_the_parser_subprocess_is_started_stopped_and_logged(
        tmp_path, fake_parser_child):
    board = _Board()
    result = asyncio.run(rs.record_streams(
        board, module=2, channels=[1, 2], duration_s=DURATION_S,
        session=rs.open_session(base=tmp_path), fastrx=False, verbose=False))
    assert result.dirfile_path.name == "serial_0042"
    log = result.parser_log.read_text()
    assert log.startswith("parser up") and "Drop Statistics" in log
    assert result.started_at == pytest.approx(board.t_trained, abs=0.05)
    assert result.warnings == []


def test_a_capture_failing_mid_window_stops_the_parser_cleanly(
        tmp_path, fake_parser_child):
    """The recording ends through its stop event, not by cancelling
    the cleanup: the parser still gets its SIGINT and is reaped."""
    session = rs.open_session(base=tmp_path)
    t0 = time.time()
    with pytest.raises(RuntimeError, match="disk full"):
        asyncio.run(rs.record_streams(
            _Board(dies="after training"), module=1, channels=[1],
            duration_s=DURATION_S, session=session, fastrx=False,
            verbose=False))
    assert time.time() - t0 < PARSER_UP_S + TRAIN_S + DURATION_S
    run = rs._load_metadata(session)["recordings"][0]
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
    session = rs.open_session(base=tmp_path)
    with pytest.raises(RuntimeError, match="writer gone"):
        asyncio.run(rs.record_streams(
            _Board(), module=1, channels=[1], duration_s=DURATION_S,
            session=session, fastrx=False, verbose=False))


def test_a_missing_fastrxd_socket_is_refused_before_the_capture(
        tmp_path, fake_recorders):
    pytest.importorskip("rfmux.fastrx")
    board = _Board()
    session = rs.open_session(base=tmp_path)
    with pytest.raises(RuntimeError, match="is fastrxd running"):
        asyncio.run(rs.record_streams(
            board, module=1, channels=[1], duration_s=DURATION_S,
            session=session, fastrx_socket=str(tmp_path / "no-daemon"),
            verbose=False))
    assert board.calls == []


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
    session = rs.open_session(base=tmp_path)
    assert session.name.startswith("session_")
    result = asyncio.run(rs.record_streams(
        _Board(), module=2, channels=[1], duration_s=DURATION_S,
        session=session, fastrx=False, verbose=False))
    meta = rs._load_metadata(session)
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
    (folder / rs.METADATA_FILE).write_text('{"created": "then", "exports": [{"filename": "x"}]}')
    assert rs.open_session(folder) == folder
    rs.register_export(folder, "pulse_module2_1.h5", "pulse", "module2")
    meta = rs._load_metadata(folder)
    assert meta["created"] == "then"
    assert [e["filename"] for e in meta["exports"]] == ["x", "pulse_module2_1.h5"]


def _bias_export(path, module, channels, calibrated=True, timestamp=""):
    out = {c: {"bias_channel": c,
               "df_calibration": (complex(1e6 * c, -1e5) if calibrated else None)}
           for c in channels}
    with open(path, "wb") as f:
        pickle.dump({"target_module": module, "timestamp": timestamp,
                     "bias_kids_output": out}, f)


def test_newest_bias_export_for_the_module_gives_channels_and_calibrations(tmp_path):
    # Written newest first: a copied folder keeps no file times, so the
    # export's own timestamp decides.
    _bias_export(tmp_path / "bias_module2_120000.pkl", 2, [4, 5], calibrated=False,
                 timestamp="2026-09-09T12:00:00")
    _bias_export(tmp_path / "bias_module1_110000.pkl", 1, [7],
                 timestamp="2026-09-09T11:00:00")
    _bias_export(tmp_path / "bias_module2_100000.pkl", 2, [1, 2, 3],
                 timestamp="2026-09-09T10:00:00")

    newest = rs.latest_bias_export(tmp_path, 2)
    assert newest.name == "bias_module2_120000.pkl"
    assert rs.biased_channels(newest) == ([4, 5], {})
    assert rs.biased_channels(tmp_path / "bias_module2_100000.pkl") == (
        [1, 2, 3], {1: 1e6 - 1e5j, 2: 2e6 - 1e5j, 3: 3e6 - 1e5j})
    assert rs.latest_bias_export(tmp_path, 3) is None


def test_channel_spec_is_the_parsers_grammar():
    from rfmux.tools.parser import parse_ranges
    channels = [1, 2, 3, 5, 7, 8, 9, 20]
    spec = rs.channel_spec(channels)
    assert spec == "1-3,5,7-9,20"
    assert [c + 1 for r in parse_ranges(spec, 1, 1024, "channel") for c in r] == channels


def test_the_requirements_are_checked_before_anything_runs(tmp_path, monkeypatch):
    session = rs.open_session(base=tmp_path)
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
                session=rs.open_session(base=tmp_path), fastrx=False,
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

    meta = rs._load_metadata(result.session)
    assert sorted(e["data_type"] for e in meta["exports"]) == ["parser", "pulse"]
    run_meta = meta["recordings"][0]
    assert run_meta["started_at"] - result.capture.start_time >= 0.9 * result.training_s
