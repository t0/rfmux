"""
The record of QC runs: an SQLite database, defined by these mapped classes.

A run is one "rfmux qc" invocation on one board; a result is one step of the
QC plan in a run; a result's report section is its blocks in order (crs_qc's
markdown, tables, figures, images and data, each a JSON document), with its
checks as rows of their own. Other tools read the tables directly (sqlite3, pandas,
the report's Python in the browser), so the tables and their columns are the
interface; user_version records the schema they follow.
"""

import datetime
import getpass
import importlib.metadata
import json
import os
import pathlib
import socket
import sqlite3
import sys
import tempfile

from sqlalchemy import JSON, ForeignKey, UniqueConstraint, create_engine, select
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from sqlalchemy.pool import StaticPool


# Increases with each change to the tables; a record of another version is
# refused rather than misread
SCHEMA_VERSION = 2

# The fields of a crs_qc check record that are columns; the rest (module=...)
# are its context
CHECK_FIELDS = ("check", "value", "unit", "nominal", "low", "high", "margin", "warn", "status")


class Base(DeclarativeBase):
    type_annotation_map = {dict: JSON, list: JSON, object: JSON}


class Run(Base):
    """One "rfmux qc" invocation on one board"""

    __tablename__ = "run"
    id: Mapped[int] = mapped_column(primary_key=True)
    uuid: Mapped[str] = mapped_column(unique=True)  # the same in every report it is in
    serial: Mapped[str]
    started: Mapped[str]  # ISO 8601
    host: Mapped[str]
    user: Mapped[str]
    rfmux: Mapped[str]  # version
    python: Mapped[str]  # version
    args: Mapped[list]  # the command line
    results: Mapped[list["Result"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="Result.started")


class Result(Base):
    """One step of the plan in a run, and its report section"""

    __tablename__ = "result"
    __table_args__ = (UniqueConstraint("run", "test"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column("run", ForeignKey("run.id"))
    run: Mapped[Run] = relationship(back_populates="results")
    test: Mapped[str]  # the step, and its modules: dac_passband[1-4]
    position: Mapped[int]  # the step's place in the plan
    outcome: Mapped[str]  # passed, failed or error (it could not run)
    started: Mapped[str]
    stopped: Mapped[str]
    failures: Mapped[list]
    warnings: Mapped[list]
    blocks: Mapped[list["Block"]] = relationship(
        cascade="all, delete-orphan", order_by="Block.position",
        collection_class=ordering_list("position"))
    checks: Mapped[list["Check"]] = relationship(
        cascade="all, delete-orphan", order_by="Check.id")


class Block(Base):
    """One part of a result's report section"""

    __tablename__ = "block"
    result_id: Mapped[int] = mapped_column("result", ForeignKey("result.id"), primary_key=True)
    position: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str]  # markdown, table, checks, figure, image or data
    content: Mapped[object]  # for "checks", the checks with this block's position


class Check(Base):
    """A measurement against its limits (crs_qc.CheckTable)"""

    __tablename__ = "checks"
    id: Mapped[int] = mapped_column(primary_key=True)
    result_id: Mapped[int] = mapped_column("result", ForeignKey("result.id"))
    block: Mapped[int]  # the position of the "checks" block that shows it
    check: Mapped[str]
    value: Mapped[float | None]
    unit: Mapped[str]
    nominal: Mapped[float | None]
    low: Mapped[float | None]
    high: Mapped[float | None]
    margin: Mapped[float | None]
    warn: Mapped[float | None]
    status: Mapped[str]  # PASS, NEAR or FAIL
    context: Mapped[dict]  # what else identifies it, e.g. {"module": 2}

    def record(self):
        """As crs_qc.CheckTable recorded it"""
        return dict(self.context, **{f: getattr(self, f) for f in CHECK_FIELDS})


class Record:
    """A record of QC runs, held in memory as one SQLite database"""

    def __init__(self, data: bytes | None = None):
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        if data is not None:
            _deserialize(self.connection, data)
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                raise ValueError(f"a QC record of schema version {version}; this rfmux "
                                 f"reads version {SCHEMA_VERSION}")
        self.engine = create_engine(
            "sqlite://", creator=lambda: self.connection, poolclass=StaticPool,
            # numpy values as lists and numbers; NaN and infinities stay, as
            # NaN and Infinity, which Python and SQLite's JSON functions read
            json_serializer=lambda o: json.dumps(o, default=lambda v: v.tolist()))
        if data is None:
            Base.metadata.create_all(self.engine)
            self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.session = Session(self.engine)

    def to_bytes(self):
        self.session.commit()  # what is still in a transaction would be left out
        if hasattr(self.connection, "serialize"):  # Python 3.11
            return self.connection.serialize()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "qc.db")
            disk = sqlite3.connect(path)
            self.connection.backup(disk)
            disk.close()
            return pathlib.Path(path).read_bytes()

    @property
    def runs(self):
        """Most recent first"""
        return self.session.scalars(select(Run).order_by(Run.started.desc())).all()

    def run(self, uuid, serial, args, started=None):
        """The run with this UUID, added if the record doesn't have it"""
        if (run := self.session.scalars(select(Run).filter_by(uuid=uuid)).first()):
            return run
        run = Run(uuid=uuid, serial=serial, started=started or datetime.datetime.now().isoformat(),
                  host=socket.gethostname(), user=getpass.getuser(),
                  rfmux=importlib.metadata.version("rfmux"), python=sys.version.split()[0],
                  args=list(args))
        self.session.add(run)
        return run

    def result(self, run, test, position, outcome, started, stopped, failures=(),
               warnings=()):
        """A step's result in a run, replacing any it had"""
        for old in [r for r in run.results if r.test == test]:
            run.results.remove(old)
        self.session.flush()
        run.results.append(result := Result(
            test=test, position=position, outcome=outcome, started=started,
            stopped=stopped, failures=list(failures), warnings=list(warnings)))
        return result

    def add_blocks(self, result, blocks):
        """A result's report section: crs_qc blocks, each a single-key dict"""
        for block in blocks:
            ((kind, content),) = block.items()
            position = len(result.blocks)
            if kind == "checks":
                for record in content:
                    result.checks.append(Check(
                        block=position, context={k: v for k, v in record.items()
                                                 if k not in CHECK_FIELDS},
                        **{f: record[f] for f in CHECK_FIELDS}))
                content = None
            result.blocks.append(Block(kind=kind, content=content))

    def merge(self, other):
        """Add the other record's runs, except those this one has"""
        have = set(self.session.scalars(select(Run.uuid)))
        copy = lambda obj, **extra: type(obj)(**{
            c.key: getattr(obj, c.key) for c in obj.__mapper__.column_attrs
            if not c.columns[0].primary_key and not c.columns[0].foreign_keys
        }, **extra)
        for run in other.runs:
            if run.uuid not in have:
                self.session.add(copy(run, results=[
                    copy(r, blocks=[copy(b, position=b.position) for b in r.blocks],
                         checks=[copy(c) for c in r.checks])
                    for r in run.results]))


def _deserialize(connection, data):
    if hasattr(connection, "deserialize"):  # Python 3.11
        connection.deserialize(data)
        return
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "qc.db"
        path.write_bytes(data)
        disk = sqlite3.connect(path)
        disk.backup(connection)
        disk.close()
