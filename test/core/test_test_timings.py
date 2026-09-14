"""Contracts for the automatic pytest timing log."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.portable
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def timing_suite(tmp_path: Path) -> Path:
    shutil.copyfile(ROOT / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test" / "example").mkdir(parents=True)
    return tmp_path


def _command(suite: Path, *args: str) -> list[str]:
    return [sys.executable, "-m", "pytest", "-q", "-c",
            str(suite / "pytest.ini"), *args]


def _env() -> dict[str, str]:
    return {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_ADDOPTS": ""}


def _events(suite: Path) -> list[dict]:
    paths = list((suite / "test-timings").glob("*.jsonl"))
    if not paths:
        return []
    return [json.loads(line) for line in paths[0].read_text().split("\n")[:-1]]


def test_completed_run_records_phases_and_summary(timing_suite: Path) -> None:
    (timing_suite / "test/example/test_example.py").write_text(
        "def test_ok(): pass\n", encoding="utf-8")
    result = subprocess.run(_command(timing_suite), cwd=timing_suite,
                            env=_env(), capture_output=True, text=True,
                            timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    events = _events(timing_suite)
    assert events[0]["event"] == "run_start"
    assert {"args", "tier", "commit", "machine", "python"} <= events[0].keys()
    phases = [e for e in events if e["event"] == "phase_finish"]
    assert [e["phase"] for e in phases] == ["setup", "call", "teardown"]
    assert all(e["duration_seconds"] >= 0 for e in phases)
    finish = events[-1]
    assert finish["event"] == "run_finish"
    assert finish["exitstatus"] == 0
    assert finish["sections"]["test/example"] == {
        e["phase"]: e["duration_seconds"] for e in phases}
    assert finish["elapsed_seconds"] >= finish["collection_seconds"] >= 0
    assert "test/example" in result.stdout
    assert "Timing log:" in result.stdout


@pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
def test_failed_phase_is_recorded(timing_suite: Path, phase: str) -> None:
    source = {
        "setup": "import pytest\n@pytest.fixture(autouse=True)\n"
                 "def fixture(): assert False\ndef test_example(): pass\n",
        "call": "def test_example(): assert False\n",
        "teardown": "import pytest\n@pytest.fixture(autouse=True)\n"
                    "def fixture():\n    yield\n    assert False\n"
                    "def test_example(): pass\n",
    }[phase]
    (timing_suite / "test/example/test_example.py").write_text(source)
    result = subprocess.run(_command(timing_suite), cwd=timing_suite,
                            env=_env(), capture_output=True, text=True,
                            timeout=30)
    assert result.returncode == 1, result.stdout + result.stderr
    events = _events(timing_suite)
    assert any(e["event"] == "phase_finish" and e["phase"] == phase
               and e["outcome"] == "failed" for e in events)
    assert events[-1]["exitstatus"] == 1


@pytest.mark.parametrize("phase", ["collection", "call", "teardown"])
def test_killed_run_preserves_active_work(timing_suite: Path, phase: str) -> None:
    source = {
        "collection": "import time\ntime.sleep(60)\n",
        "call": "import time\ndef test_example(): time.sleep(60)\n",
        "teardown": "import time, pytest\n@pytest.fixture(autouse=True)\n"
                    "def fixture():\n    yield\n    time.sleep(60)\n"
                    "def test_example(): pass\n",
    }[phase]
    (timing_suite / "test/example/test_example.py").write_text(source)
    process = subprocess.Popen(_command(timing_suite), cwd=timing_suite,
                               env=_env(), stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            events = _events(timing_suite)
            if phase == "collection":
                ready = any(e["event"] == "collector_start" and
                            e["nodeid"].endswith("test_example.py") for e in events)
            else:
                ready = any(e["event"] == "phase_start" and
                            e["phase"] == phase for e in events)
            if ready:
                break
            assert process.poll() is None, "pytest exited before the phase started"
            time.sleep(0.02)
        else:
            pytest.fail(f"No start event for {phase}")
    finally:
        process.kill()
        process.wait(timeout=10)
    events = _events(timing_suite)
    assert not any(e["event"] == "run_finish" for e in events)
    if phase != "collection":
        assert events[-1]["event"] == "phase_start"
        assert events[-1]["phase"] == phase


def test_timing_opt_out(timing_suite: Path) -> None:
    (timing_suite / "test/example/test_example.py").write_text(
        "def test_ok(): pass\n")
    result = subprocess.run(_command(timing_suite, "--no-test-timings"),
                            cwd=timing_suite, env=_env(), capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (timing_suite / "test-timings").exists()
    assert "Timing log:" not in result.stdout
