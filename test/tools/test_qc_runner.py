"""The QC runner: the plan, in its order, on one CRS, into the report."""

import asyncio
import json
import re
from functools import partial

import click
import pytest
import rfmux
from click.testing import CliRunner

from rfmux.tools.qc import _modules, cli, report_generator as rg, runner
from rfmux.tools.qc.crs_qc import CheckTable, render_markdown

pytestmark = pytest.mark.portable


async def zeta(d, report):
    report.append(render_markdown("## Zeta"))


async def alpha(d, report):
    report.append(render_markdown("## Alpha"))


async def raises(d, report):
    raise ValueError("no tone")


async def board(serial):
    """A CRS that is never resolved (the steps here don't talk to it), without
    modules, which take a while to make and no step here uses"""
    hwm = rfmux.HardwareMap()
    hwm.add(rfmux.CRS(serial=serial, modules=[]))
    hwm.commit()
    return hwm


def run(plan, tmp_path, **kwargs):
    kwargs.setdefault("connect", board)
    return asyncio.run(runner.run("0246", tmp_path / "qc.html", plan=plan, **kwargs))


def outcomes(tmp_path):
    """(test, outcome, failures' first lines) of every result, in the plan's order"""
    return [(r.test, r.outcome, [f.partition("\n")[0] for f in r.failures])
            for r in sorted(rg.read(tmp_path / "qc.html").runs[0].results,
                            key=lambda r: r.position)]


def names(tmp_path):
    return [test for test, _, _ in outcomes(tmp_path)]


def test_the_plan_runs_the_board_steps_then_the_datapath_on_each_bank(
        tmp_path, monkeypatch):
    banks = []

    async def set_analog_bank(d, high):
        banks.append(high)
    monkeypatch.setattr(rfmux.CRS, "set_analog_bank", set_analog_bank, raising=False)

    async def step(d, report, module=None):
        pass

    def stub(entry):
        """A step that does nothing, by the entry's name and with its module"""
        if not runner.reported(entry):
            return entry
        if isinstance(entry, partial):
            return partial(stub(entry.func), **entry.keywords)
        return type(step)(step.__code__.replace(co_name=entry.__name__), {})

    run(tuple(stub(e) for e in runner.PLAN), tmp_path)
    datapath = ["dac_passband", "dac_amplitude_transfer", "dac_scale_transfer",
                "dac_mixmodes", "adc_attenuation", "wideband_noise"]
    assert names(tmp_path) == [
        "communications", "metadata", "housekeeping_temperature",
        "housekeeping_voltages", "housekeeping_currents",
        *(f"{n}[{m}]" for n in datapath for m in range(1, 5)),
        *(f"{n}[{m}]" for n in datapath for m in range(5, 9))]
    assert banks == [False, True]


def test_steps_run_and_are_reported_in_the_plans_order(tmp_path):
    ran = []

    async def second(d, report):
        ran.append("second")

    async def first(d, report):
        ran.append("first")

    run((zeta, alpha, second, first), tmp_path)
    assert ran == ["second", "first"]
    shown = re.search(r'<script type="application/json" id="qc-index">(.*?)</script>',
                      (tmp_path / "qc.html").read_text(), re.S)[1]
    assert [e["test"] for e in json.loads(shown)["results"]] == [
        "zeta", "alpha", "second", "first"]


def test_every_entry_has_the_one_crs(tmp_path):
    seen = []

    async def crs(d, report, module=None):
        seen.append(d)

    async def quiet(d):
        seen.append(d)

    run((crs, quiet, partial(crs, module=1)), tmp_path)
    assert len(seen) == 3 and all(d is seen[0] for d in seen)


def test_a_step_given_a_module_runs_on_it_and_is_named_by_it(tmp_path):
    seen = []

    async def on(d, report, module):
        seen.append(module)

    run((partial(on, module=3), partial(on, module=7)), tmp_path)
    assert seen == [3, 7]
    assert names(tmp_path) == ["on[3]", "on[7]"]


def test_an_entry_without_a_report_is_reported_only_if_it_fails(tmp_path):
    async def quiet(d):
        pass

    async def breaks(d):
        raise OSError("no bank")

    assert not run((quiet, zeta, breaks, alpha), tmp_path)
    assert outcomes(tmp_path) == [
        ("zeta", "passed", []), ("breaks", "failed", ["OSError: no bank"]),
        ("alpha", "passed", [])]


def test_a_step_that_raises_fails_and_the_next_still_runs(tmp_path):
    assert not run((raises, zeta), tmp_path)
    assert outcomes(tmp_path) == [
        ("raises", "failed", ["ValueError: no tone"]), ("zeta", "passed", [])]


def test_a_failure_the_step_reports_fails_it(tmp_path):
    async def reports_a_failure(d, report):
        report.fail("no multicast")

    run((reports_a_failure,), tmp_path)
    assert outcomes(tmp_path) == [("reports_a_failure", "failed", ["no multicast"])]


def test_a_failure_without_a_message_is_still_reported(tmp_path):
    async def times_out(d, report):
        report.fail(str(TimeoutError()))

    run((times_out,), tmp_path)
    assert outcomes(tmp_path) == [("times_out", "failed", [""])]


def test_a_check_outside_its_limits_fails_its_step(tmp_path):
    async def checks_out_of_limits(d, report):
        report.append(checks := CheckTable())
        checks.check("Reading", 0.0, "V", low=1.71, high=1.89)

    run((checks_out_of_limits,), tmp_path)
    ((_, outcome, failures),) = outcomes(tmp_path)
    assert outcome == "failed"
    assert failures[0].startswith("Reading: 0 V outside")


def test_exitfirst_stops_at_the_first_failure(tmp_path):
    run((zeta, raises, alpha), tmp_path, exitfirst=True)
    assert names(tmp_path) == ["zeta", "raises"]


def test_a_failed_connection_is_the_one_result(tmp_path):
    async def connect(serial):
        raise OSError("no route to host")

    assert not run((zeta, alpha), tmp_path, connect=connect)
    assert outcomes(tmp_path) == [("connect", "error", ["OSError: no route to host"])]


def test_steps_runs_the_named_steps_on_every_module_and_every_other_entry(tmp_path):
    ran = []

    async def on(d, report, module):
        pass

    async def quiet(d):
        ran.append("quiet")

    run((zeta, quiet, partial(on, module=1), alpha, quiet, partial(on, module=5)),
        tmp_path, steps=["alpha", "on"])
    assert names(tmp_path) == ["on[1]", "alpha", "on[5]"]
    assert ran == ["quiet", "quiet"]


def test_modules_runs_the_steps_on_them_and_not_the_boards(tmp_path):
    async def on(d, report, module):
        pass

    run((zeta, *(partial(on, module=m) for m in range(1, 9))), tmp_path,
        modules={5, 6, 7, 8})
    assert names(tmp_path) == ["on[5]", "on[6]", "on[7]", "on[8]"]


@pytest.mark.parametrize("ranges, modules", [
    ("5-8", {5, 6, 7, 8}), ("1,3,5-6", {1, 3, 5, 6}), ("4", {4})])
def test_modules_are_given_as_ranges(ranges, modules):
    assert _modules(None, None, ranges) == modules


@pytest.mark.parametrize("ranges", ["five", "0-2", "8-9", "1,"])
def test_modules_outside_1_to_8_are_refused(ranges):
    with pytest.raises(click.BadParameter):
        _modules(None, None, ranges)


def test_an_unknown_step_is_refused_before_connecting():
    result = CliRunner().invoke(cli, ["--serial=0246", "--step=dac_passbnd"])
    assert result.exit_code == 2
    assert "no step dac_passbnd" in result.output


def test_a_run_into_an_existing_report_adds_to_it(tmp_path):
    run((zeta,), tmp_path)
    run((alpha,), tmp_path)
    assert rg.load(tmp_path / "qc.html")["db"].execute(
        "SELECT test FROM result JOIN run ON result.run = run.id ORDER BY run.started"
    ).fetchall() == [("zeta",), ("alpha",)]


def test_a_step_tracking_a_loop_gets_every_item_in_order(tmp_path):
    seen = []

    async def tracks(d, report):
        seen.extend(report.track(m for m in range(5, 9)))

    run((tracks,), tmp_path)
    assert seen == [5, 6, 7, 8]
