"""The QC report: one self-contained HTML file that is also the record of its runs."""

import re

import matplotlib.figure
import numpy as np
import pytest
from click.testing import CliRunner

from rfmux.tools.qc import cli, report_generator as rg
from rfmux.tools.qc.crs_qc import CheckTable, render_markdown, render_plot
from rfmux.tools.qc.record import Record

pytestmark = pytest.mark.portable

PASSBAND = "dac_passband[2]"
SCALE = "dac_scale_transfer[2]"
VOLTAGES = "housekeeping_voltages"
TIME = "2026-10-27T09:30:00"


def database(*results, serial="0246", run="run-1", started=TIME, db=None):
    """A record with one run of these (test, outcome, blocks[, failures]) results"""
    db = db or Record()
    run = db.run(run, serial, ["rfmux", "qc"], started=started)
    for position, (test, outcome, blocks, *failures) in enumerate(results):
        result = db.result(run, test, position, outcome, started, started,
                           failures[0] if failures else [])
        db.add_blocks(result, [b.to_block() if hasattr(b, "to_block") else b
                               for b in blocks])
    return db


def sql(record, query):
    record.to_bytes()  # committed
    return record.connection.execute(query).fetchall()


def section(test):
    return [render_markdown(f"## Section {test}")]


@pytest.fixture
def report(tmp_path):
    """Write a database as its report, and return the report's HTML"""

    def build(db):
        rg.write(db, tmp_path / "qc.html")
        return (tmp_path / "qc.html").read_text()

    return build


def section_of(text, test):
    """The opening tag and body of the (first) section of a test's result"""
    for m in re.finditer(r'(<section id="result-\d+"[^>]*>)(.*?)</section>', text, re.S):
        if test in m[2]:
            return m[1], m[2]


def index(text):
    """The results' index the page's overview is drawn from, by test"""
    import json

    data = json.loads(re.search(r'<script type="application/json" id="qc-index">(.*?)</script>',
                                text, re.S)[1])
    return data, {(e["test"], e["serial"], e["run"]): e for e in data["results"]}


def plot_levels(data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(data["frequency_mhz"], data["level_dbm"], label="Module 2")
    return fig


LEVELS = dict(frequency_mhz=np.arange(79) * 1e2, level_dbm=np.linspace(-2, -9, 79))


def test_the_report_starts_as_html(report):
    text = report(database((PASSBAND, "passed", section(PASSBAND))))
    assert text.lower().startswith("<!doctype html><html")


def test_the_report_markup_refers_to_nothing_outside_it(report):
    text = report(database((PASSBAND, "passed", [render_plot(plot_levels, LEVELS)])))
    assert "<style>" in text and 'class="qc-plot"' in text
    markup = re.sub(r"(<script[^>]*>).*?</script>", r"\1", text, flags=re.S)
    assert not re.search(r'<link|(src|href)="(?!data:|#)', markup)


def test_the_report_holds_its_database(tmp_path):
    checks = CheckTable()
    checks.check("Weakest tone", -32.4, "dBm", low=-40, warn=10, module=2)
    db = database((PASSBAND, "failed", [render_markdown("## Passband"), checks,
                                        render_plot(plot_levels, LEVELS),
                                        {"data": {"module": 2, "errors": [0, 1]}}],
                   ["off by 7 dB"]))
    rg.write(db, tmp_path / "qc.html")
    again = rg.read(tmp_path / "qc.html")
    for table in ("run", "result", "block", "checks"):
        assert sql(again, f"SELECT * FROM {table}") == sql(db, f"SELECT * FROM {table}")


def test_checks_can_be_queried(tmp_path):
    checks = CheckTable()
    checks.check("Weakest tone", -32.4, "dBm", low=-40, warn=10, module=2)
    checks.check("Weakest tone", -9.3, "dBm", low=-40, warn=10, module=3)
    rg.write(database((PASSBAND, "passed", [checks])), tmp_path / "qc.html")
    assert sql(rg.read(tmp_path / "qc.html"),
               "SELECT run.serial, result.test, json_extract(checks.context, '$.module') "
               "FROM checks JOIN result ON checks.result = result.id "
               "JOIN run ON result.run = run.id WHERE status = 'NEAR'"
               ) == [("0246", PASSBAND, 2)]


def test_merging_a_report_with_itself_adds_nothing():
    db = database((PASSBAND, "passed", section(PASSBAND)))
    merged = Record()
    merged.merge(db)
    merged.merge(db)
    assert sql(merged, "SELECT count(*) FROM run") == [(1,)]
    assert sql(merged, "SELECT count(*) FROM block") == [(1,)]


def test_input_rebuilds_a_report_in_place(tmp_path):
    rg.write(database((PASSBAND, "failed", section(PASSBAND))), tmp_path / "qc.html")
    stale = (tmp_path / "qc.html").read_text().replace("Section dac", "Stale dac")
    (tmp_path / "qc.html").write_text(stale)
    result = CliRunner().invoke(cli, ["--input", str(tmp_path / "qc.html")])
    assert result.exit_code == 0, result.output
    assert "Section dac" in (tmp_path / "qc.html").read_text()


def test_input_writes_to_output_and_leaves_the_original(tmp_path):
    rg.write(database((PASSBAND, "passed", section(PASSBAND))), tmp_path / "qc.html")
    original = (tmp_path / "qc.html").read_text()
    result = CliRunner().invoke(cli, ["--input", str(tmp_path / "qc.html"),
                                      "--output", str(tmp_path / "new.html")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "qc.html").read_text() == original
    assert sql(rg.read(tmp_path / "new.html"), "SELECT * FROM block") == \
        sql(rg.read(tmp_path / "qc.html"), "SELECT * FROM block")


def test_input_several_reports_merges_them(tmp_path):
    rg.write(database((PASSBAND, "passed", section(PASSBAND)), serial="0246", run="a"),
             tmp_path / "a.html")
    rg.write(database((PASSBAND, "passed", section(PASSBAND)), serial="0061", run="b"),
             tmp_path / "b.html")
    result = CliRunner().invoke(cli, [f"--input={tmp_path / 'a.html'}",
                                      f"--input={tmp_path / 'b.html'}",
                                      f"--output={tmp_path / 'both.html'}"])
    assert result.exit_code == 0, result.output
    assert sql(rg.read(tmp_path / "both.html"),
               "SELECT serial FROM run ORDER BY serial") == [("0061",), ("0246",)]


def test_input_a_missing_report_fails_and_writes_nothing(tmp_path):
    result = CliRunner().invoke(cli, [f"--input={tmp_path / 'qc.html'}"])
    assert result.exit_code != 0
    assert "no QC results" in result.output
    assert list(tmp_path.iterdir()) == []


def test_a_plot_function_must_stand_alone():
    def plot_with_a_global(data):
        return matplotlib.figure.Figure()

    with pytest.raises(NameError):
        render_plot(plot_with_a_global, LEVELS)


def test_non_finite_values_reach_the_figure_function_unchanged(tmp_path):
    # a log of a zero power is -inf
    levels = dict(LEVELS, level_dbm=np.r_[-np.inf, np.nan, LEVELS["level_dbm"][2:]])
    rg.write(database((PASSBAND, "passed", [render_plot(plot_levels, levels)])),
             tmp_path / "qc.html")
    (figure,) = rg.load(tmp_path / "qc.html")["figures"]
    (line,) = figure["plot"](figure["data"]).axes[0].lines
    np.testing.assert_array_equal(line.get_ydata(), levels["level_dbm"])
    assert sql(rg.read(tmp_path / "qc.html"),
               "SELECT json_extract(content, '$.data.level_dbm[0]') FROM block") == [(-np.inf,)]


def test_a_check_outside_its_limits_fails():
    checks = CheckTable()
    assert not checks.check("Weakest tone", -45, "dBm", low=-40, module=2)
    assert checks.records[0]["status"] == "FAIL"
    assert checks.failures[0].startswith("Weakest tone (module 2): -45 dBm")


def test_a_check_within_its_warning_margin_is_a_near_miss():
    checks = CheckTable()
    assert checks.check("Weakest tone", -32.4, "dBm", low=-40, warn=10)
    assert checks.check("Weakest tone", -9.3, "dBm", low=-40, warn=10)
    assert [r["status"] for r in checks.records] == ["NEAR", "PASS"]
    assert not checks.failures


def test_load_returns_the_checks_and_what_redraws_each_figure(tmp_path):
    checks = CheckTable()
    checks.check("Weakest tone", -32.4, "dBm", low=-40, warn=10, module=2)
    rg.write(database((PASSBAND, "passed", [checks, render_plot(plot_levels, LEVELS)])),
             tmp_path / "qc.html")

    data = rg.load(tmp_path / "qc.html")
    assert data["checks"] == [dict(checks.records[0], serial="0246", run="run-1",
                                   test=PASSBAND)]
    (check,) = data["record"].runs[0].results[0].checks
    assert check.record() == checks.records[0]
    (figure,) = data["figures"]
    (line,) = figure["plot"](figure["data"]).axes[0].lines
    np.testing.assert_array_equal(line.get_ydata(), LEVELS["level_dbm"])


def test_input_a_report_to_a_db_file_gives_a_plain_sqlite_database(tmp_path):
    import sqlite3

    rg.write(database((PASSBAND, "failed", section(PASSBAND))), tmp_path / "qc.html")
    result = CliRunner().invoke(cli, [f"--input={tmp_path / 'qc.html'}",
                                      f"--output={tmp_path / 'qc.db'}"])
    assert result.exit_code == 0, result.output
    db = sqlite3.connect(tmp_path / "qc.db")
    assert db.execute("SELECT serial, test, outcome FROM run JOIN result "
                      "ON result.run = run.id").fetchall() == [("0246", PASSBAND, "failed")]


def test_input_a_db_file_back_to_a_report(tmp_path):
    rg.write(database((PASSBAND, "failed", section(PASSBAND))), tmp_path / "qc.db")
    result = CliRunner().invoke(cli, [f"--input={tmp_path / 'qc.db'}",
                                      f"--output={tmp_path / 'qc.html'}"])
    assert result.exit_code == 0, result.output
    assert f"Section {PASSBAND}" in section_of((tmp_path / "qc.html").read_text(), PASSBAND)[1]


def test_input_a_file_that_is_not_a_qc_database_fails(tmp_path):
    (tmp_path / "other.db").write_bytes(b"not a database")
    result = CliRunner().invoke(cli, [f"--input={tmp_path / 'other.db'}",
                                      f"--output={tmp_path / 'qc.html'}"])
    assert result.exit_code != 0
    assert "no QC results" in result.output


def test_a_record_of_another_schema_version_is_refused(tmp_path):
    import sqlite3

    rg.write(database((PASSBAND, "passed", section(PASSBAND))), tmp_path / "old.db")
    with sqlite3.connect(tmp_path / "old.db") as old:
        old.execute("PRAGMA user_version = 0")
    result = CliRunner().invoke(cli, [f"--input={tmp_path / 'old.db'}",
                                      f"--output={tmp_path / 'qc.html'}"])
    assert result.exit_code != 0
    assert "schema version 0" in result.output


def test_each_checks_table_has_its_own_columns(report):
    with_module, plain = CheckTable(), CheckTable()
    with_module.check("Weakest tone", -9.3, "dBm", low=-40, module=2)
    plain.check("RFSOC_VCCAUX", 1.8, "V", nominal=1.8, low=1.71, high=1.89)
    text = report(database((PASSBAND, "passed", [with_module]),
                           (VOLTAGES, "passed", [plain])))
    cells = {}
    for t in re.findall(r'<table class="checks">(.*?)</table>', text, re.S):
        header = re.findall(r"<th>(.*?)</th>", t)
        (row,) = re.findall(r"<tbody><tr>(.*?)</tr>", t)
        values = [re.sub(r"<[^>]+>", "", c) for c in re.findall(r"<td[^>]*>(.*?)</td>", row)]
        cells[values[header.index("Check")]] = dict(zip(header, values))
    assert cells["Weakest tone"]["Module"] == "2"
    assert "Module" not in cells["RFSOC_VCCAUX"]
    assert cells["RFSOC_VCCAUX"]["Nominal"] == "1.8"


def test_a_failure_shows_what_failed_without_its_traceback(report):
    text = report(database((PASSBAND, "failed", section(PASSBAND),
                            ["ValueError: off by 7 dB\nTraceback (most recent call last):"])))
    failures = re.search(r'<ul class="failures">(.*?)</ul>', section_of(text, PASSBAND)[1])[1]
    assert failures == "<li>ValueError: off by 7 dB</li>"


def test_each_result_links_to_its_section(report):
    text = report(database((PASSBAND, "failed", section(PASSBAND), ["off by 7 dB"]),
                           (VOLTAGES, "passed", section(VOLTAGES))))
    _, results = index(text)
    for (test, *_), entry in results.items():
        assert f'<section id="{entry["id"]}"' in text
        assert entry["status"] == (["FAIL", "fail"] if "passband" in test else ["PASS", "pass"])


def test_a_merged_report_offers_each_board_and_run(report):
    db = Record()
    db.merge(database((PASSBAND, "failed", section(PASSBAND)), serial="0246", run="a"))
    db.merge(database((PASSBAND, "passed", section(PASSBAND)), serial="0061", run="b"))
    text = report(db)
    data, results = index(text)
    assert set(data["runs"]) == {"a", "b"}
    boards = re.search(r'<select id="filter-board">(.*?)</select>', text)[1]
    assert re.findall(r'<option value="([^"]+)">', boards) == ["all", "0246", "0061"]


def test_each_boards_latest_result_of_each_test_is_marked(report):
    db = database((PASSBAND, "failed", section(PASSBAND)), run="older",
                  started="2026-10-27T09:00:00")
    db = database((PASSBAND, "passed", section(PASSBAND)), run="newer",
                  started="2026-10-27T10:00:00", db=db)
    _, results = index(report(db))
    assert results[("dac_passband", "0246", "older")]["latest"] is False
    assert results[("dac_passband", "0246", "newer")]["latest"] is True


def test_a_result_is_titled_by_its_first_heading_with_the_rest_folded_away(report):
    text = report(database((PASSBAND, "passed", [render_markdown("""
        ## DAC Passband Checks (Module 2)

        How the passband is measured.
        """)])))
    _, body = section_of(text, PASSBAND)
    assert re.search(r"<h2>.*DAC Passband Checks", body)
    assert re.search(r'<details class="about"><summary>About this test</summary>'
                     r"<p>How the passband is measured.</p>", body)
    _, results = index(text)
    assert results[("dac_passband", "0246", "run-1")]["title"] == "DAC Passband Checks"


def test_problems_are_failing_checks_near_misses_and_other_failures(report):
    checks = CheckTable()
    checks.check("Weakest tone", -32.4, "dBm", low=-40, warn=10, module=2)
    checks.check("RFSOC_VCCAMS", 0.0, "V", low=1.71, high=1.89)
    _, results = index(report(database(
        (VOLTAGES, "failed", [checks], checks.failures),
        (PASSBAND, "failed", section(PASSBAND), ["off by 7 dB"]))))
    voltages = results[("housekeeping_voltages", "0246", "run-1")]["problems"]
    # the check's own failure message is not repeated
    assert [p["status"][0] for p in voltages] == ["NEAR", "FAIL"]
    assert "RFSOC_VCCAMS, 0 V (limits 1.71 to 1.89)" in voltages[1]["text"]
    passband = results[("dac_passband", "0246", "run-1")]["problems"]
    assert [p["text"] for p in passband] == ["off by 7 dB"]


def test_a_result_is_of_the_module_its_name_gives_or_of_the_board(report):
    _, results = index(report(database((PASSBAND, "passed", section(PASSBAND)),
                                       (VOLTAGES, "passed", section(VOLTAGES)))))
    assert results[("dac_passband", "0246", "run-1")]["module"] == 2
    assert results[("housekeeping_voltages", "0246", "run-1")]["module"] is None
