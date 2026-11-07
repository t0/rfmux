"""
Runs the QC plan on one CRS: each entry in the plan's order, with each
result and report section going into the record, which is written out after
every step so a run cut short keeps what it did.

An entry is called with the CRS, d, and returns something to await. A step,
"async def step(d, report)", is reported: report is its report section, a
list of crs_qc blocks. It fails if it calls report.fail(message), if a
CheckTable it reports has a value outside its limits, or if it raises. An
entry that takes d alone only does something (lambda d:
d.set_analog_bank(high=True), say) and is reported only if it raises. A step
for one module is a functools.partial giving it "module", which also names
its result.
"""

import contextlib
import datetime
import functools
import inspect
import os
import sys
import traceback
import uuid
import warnings

import matplotlib
import tqdm
from tqdm.contrib import DummyTqdmFile

import rfmux

from . import basics, datapath, housekeeping, report_generator
from .crs_qc import CheckTable
from .record import Record

matplotlib.use("Agg")  # figures are drawn to files, never shown

DATAPATH = (
    # Tests require analog loopback cables (DAC1 -> ADC1, ...)
    datapath.dac_passband,
    datapath.dac_amplitude_transfer,
    datapath.dac_scale_transfer,
    datapath.dac_mixmodes,
    datapath.adc_attenuation,
    datapath.wideband_noise,
)

PLAN = (
    # No setup required
    basics.communications,
    basics.metadata,
    housekeeping.housekeeping_temperature,
    housekeeping.housekeeping_voltages,
    housekeeping.housekeeping_currents,

    # Datapath tests are repeated twice, once on "low" banking, and once on
    # "high" banking.
    lambda d: d.set_analog_bank(high=False),
    *(functools.partial(step, module=m)
      for step in DATAPATH for m in range(1, 5)),

    lambda d: d.set_analog_bank(high=True),
    *(functools.partial(step, module=m)
      for step in DATAPATH for m in range(5, 9)),
)


class Section(list):
    """
    A step's report section, the failures it reports itself, and its progress
    bar, which shows its name and time until the step tracks a loop with it
    """

    def __init__(self, bar=None):
        super().__init__()
        self.failures = []
        self.bar = bar

    def fail(self, message: str):
        self.failures.append(message)

    def track(self, iterable):
        """The items of iterable, counting through them on the step's bar"""
        items = list(iterable)
        if self.bar is not None:
            self.bar.bar_format = "{desc}: {n_fmt}/{total_fmt} |{bar}| {elapsed}"
            self.bar.reset(total=len(items))
        for item in items:
            yield item
            if self.bar is not None:
                self.bar.update()


def step_name(entry) -> str:
    return (entry.func if isinstance(entry, functools.partial) else entry).__name__


def module_of(entry) -> int | None:
    """The module the entry is given, or None for the board"""
    return entry.keywords["module"] if isinstance(entry, functools.partial) else None


def result_name(entry) -> str:
    """The entry's name, and the module it is given: dac_passband[3]"""
    module = module_of(entry)
    return step_name(entry) + ("" if module is None else f"[{module}]")


def reported(entry) -> bool:
    return "report" in inspect.signature(entry).parameters


async def connect(serial: str) -> rfmux.HardwareMap:
    """
    The hardware map of the one CRS, resolved: its session, which holds the
    CRS only while the map lives
    """
    hwm = rfmux.HardwareMap()
    hwm.add(rfmux.CRS(serial=serial))
    hwm.commit()
    d = hwm.query(rfmux.CRS).one()
    await d.tuber_resolve()
    await d.set_timestamp_port(d.TIMESTAMP_PORT.TEST)
    return hwm


def _failure(e: BaseException) -> str:
    """What failed, on its first line, then the traceback"""
    return f"{type(e).__name__}: {e}\n" + "".join(traceback.format_exception(e))


def chosen(entry, steps=(), modules=None) -> bool:
    """
    Whether to run the entry: every entry that isn't reported; otherwise
    only the steps named in "steps", if any, and only those on "modules", if
    given (which leaves out the board's steps)
    """
    if not reported(entry):
        return True
    return ((not steps or step_name(entry) in steps)
            and (modules is None or module_of(entry) in modules))


async def run(serial: str, filename, *, steps=(), modules=None, exitfirst=False,
              plan=PLAN, connect=connect) -> bool:
    """
    Run the plan's chosen entries into the report "filename" (an existing one
    gains this run). True if nothing failed.
    """
    record = report_generator.read(filename) if os.path.exists(filename) else Record()
    this_run = record.run(str(uuid.uuid4()), serial, sys.argv)

    def report(name, position, started, failures, blocks=(), caught=(), outcome=None):
        """A result into the record and report, and to the terminal"""
        stopped = datetime.datetime.now()
        outcome = outcome or ("failed" if failures else "passed")
        result = record.result(this_run, name, position, outcome, started.isoformat(),
                               stopped.isoformat(), failures,
                               [str(w.message) for w in caught])
        record.add_blocks(result, [b.to_block() if hasattr(b, "to_block") else b
                                   for b in blocks])
        report_generator.write(record, filename)

        word, _ = report_generator.status_of(outcome)
        print(f"{word:5} {name} ({(stopped - started).total_seconds():.1f} s)")
        for f in failures:
            print("      " + f.rstrip().replace("\n", "\n      "))

    entries = [(position, entry) for position, entry in enumerate(plan)
               if chosen(entry, steps, modules)]

    # Each step's bar writes to the terminal, and everything else is written
    # above it
    terminal = sys.stdout
    with (contextlib.redirect_stdout(DummyTqdmFile(sys.stdout)),
          contextlib.redirect_stderr(DummyTqdmFile(sys.stderr))):
        started = datetime.datetime.now()
        try:
            hwm = await connect(serial)
        except Exception as e:
            report("connect", -1, started, [_failure(e)], outcome="error")
            return False

        d = hwm.query(rfmux.CRS).one()
        passed = True
        for position, entry in entries:
            name = result_name(entry)
            started = datetime.datetime.now()
            with (tqdm.tqdm(desc=name, bar_format="{desc}: {elapsed}", file=terminal,
                            leave=False, disable=None if reported(entry) else True)
                  as bar, warnings.catch_warnings(record=True) as caught):
                warnings.simplefilter("always")
                section = Section(bar)
                try:
                    await (entry(d, section) if reported(entry) else entry(d))
                    raised = []
                except Exception as e:
                    raised = [_failure(e)]
            failures = section.failures + [
                f for b in section if isinstance(b, CheckTable) for f in b.failures] + raised

            if reported(entry) or failures:
                report(name, position, started, failures, section, caught)
            if failures:
                passed = False
                if exitfirst:
                    break
        return passed
