"""
The QC report: one HTML file that is also the record of its runs (record.py),
an SQLite database embedded gzipped and base64-encoded. Reports merge: a
report can hold runs of several boards, and offers a view of each run and of
each board's latest results.
"""

import base64
import datetime
import gzip
import json
import os
import pathlib
import re
import socket
import sqlite3

import markdown
import markupsafe

from .record import Record
from htpy import (
    body,
    button,
    details,
    div,
    figcaption,
    figure,
    h1,
    h2,
    head,
    header,
    html,
    img,
    input as input_,
    label,
    li,
    main,
    meta,
    option,
    p,
    pre,
    script,
    section,
    select,
    span,
    style,
    summary,
    table,
    tbody,
    td,
    textarea,
    th,
    thead,
    title,
    tr,
    ul,
)


# outcome -> (status word, CSS class); colour only repeats the word
STATUS = {
    "passed": ("PASS", "pass"),
    "failed": ("FAIL", "fail"),
    "error": ("ERROR", "fail"),
    "skipped": ("SKIP", "skip"),
}

# Views of a report's database for analysis, run by load() here and by the
# report's Python console in the browser: given "db", defines "checks" (every
# check record) and "figures" (every figure, with "plot" to redraw it:
# figure["plot"](figure["data"])), each with its serial, run and test
ANALYSIS = """
import json

def _plot_function(figure):
    namespace = {}
    exec(figure["source"], namespace)
    return namespace[figure["name"]]

checks = [
    dict(json.loads(context), check=check, value=value, unit=unit, nominal=nominal,
         low=low, high=high, margin=margin, warn=warn, status=status,
         serial=serial, run=run, test=test)
    for serial, run, test, check, value, unit, nominal, low, high, margin, warn, status,
        context in db.execute(
        'SELECT run.serial, run.uuid, result.test, checks."check", value, unit, nominal, '
        'low, high, margin, warn, status, context FROM checks '
        'JOIN result ON checks.result = result.id JOIN run ON result.run = run.id '
        'ORDER BY checks.id')
]
figures = []
for serial, run, test, content in db.execute(
        "SELECT run.serial, run.uuid, result.test, block.content FROM block "
        "JOIN result ON block.result = result.id JOIN run ON result.run = run.id "
        "WHERE block.kind = 'figure' ORDER BY result.id, block.position"):
    figure = json.loads(content)
    figures.append(dict(figure, serial=serial, run=run, test=test,
                        plot=_plot_function(figure)))
"""

# Run in the browser once Pyodide is loaded: opens the embedded database,
# and defines qc_show (show a figure interactively, with clicking a legend
# entry hiding or showing what it labels), qc_draw and qc_copy
BROWSER = """
import base64, gzip, json, sqlite3, js

db = sqlite3.connect(":memory:")
db.deserialize(gzip.decompress(base64.b64decode(
    js.document.getElementById("qc-db").textContent)))

def qc_figure(result, position):
    (content,) = db.execute(
        "SELECT content FROM block WHERE result = ? AND position = ?",
        (result, position)).fetchone()
    return json.loads(content)

def qc_show(fig):
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        labelled = {a.get_label(): a for a in ax.get_children()}
        for handle, text in zip(legend.legend_handles, legend.get_texts()):
            if (artist := labelled.get(text.get_text())) is not None:
                for picked in (handle, text):
                    picked.set_picker(True)
                    picked.qc_toggles = (artist, handle, text)

    def toggle(event):
        if (toggles := getattr(event.artist, "qc_toggles", None)) is None:
            return
        artist, handle, text = toggles
        artist.set_visible(not artist.get_visible())
        for shown in (handle, text):
            shown.set_alpha(1.0 if artist.get_visible() else 0.2)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", toggle)
    fig.canvas.manager.show()

def qc_draw(result, position):
    figure = qc_figure(result, position)
    namespace = {}
    exec(figure["source"], namespace)
    qc_show(namespace[figure["name"]](figure["data"]))

def qc_copy(result, position):
    figure = qc_figure(result, position)
    return (f"import json\\n\\n{figure['source']}\\n"
            f"data = json.loads({json.dumps(json.dumps(figure['data']))})\\n"
            f"fig = {figure['name']}(data)\\n")
"""

# Each figure is drawn by its own Python, run with Pyodide: all of them, in
# the background, one at a time, those in view first, yielding between
# figures so the page stays usable; drawing and the console take turns,
# since matplotlib draws into one target at a time. The view selector shows
# one run, or one board's latest results; one result's detail shows at a
# time (named by the URL's #), chosen from the overview or by stepping
# through the problems; "show all" (and printing) lists every result
SCRIPT = """
// matplotlib's figure script focuses each new figure 100 ms after drawing it,
// which scrolls the page to it and takes focus from wherever the reader is;
// let a figure take focus only from the reader's own click
const focusElement = HTMLElement.prototype.focus;
HTMLElement.prototype.focus = function (options) {
    if (this.closest(".qc-plot") && !navigator.userActivation?.isActive) return;
    focusElement.call(this, options);
};
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";
let pyodide = null;
function python() {
    pyodide ??= (async () => {
        await new Promise((resolve, reject) => {
            const s = document.createElement("script");
            s.src = PYODIDE + "pyodide.js";
            s.onload = resolve;
            s.onerror = () => reject(new Error("cannot reach " + PYODIDE));
            document.head.append(s);
        });
        const py = await loadPyodide({indexURL: PYODIDE});
        await py.loadPackage(["numpy", "matplotlib", "sqlite3"]);
        // load the interactive backend's script and style, which only
        // plt.show() otherwise does, so figures can be shown one at a time
        await py.runPythonAsync(
            "from matplotlib.backends.backend_webagg import WebAggApplication\\n"
            + "WebAggApplication.initialize()\\n"
            + document.getElementById("qc-browser").textContent);
        return py;
    })().catch(e => { pyodide = null; throw e; });  // let the next try start over
    return pyodide;
}
let turn = Promise.resolve();
function inTurn(task) {
    const next = turn.then(task);
    turn = next.catch(() => {});
    return next;
}
async function draw(plot) {
    plot.dataset.state = "drawing";
    const status = plot.querySelector(".qc-plot-status");
    try {
        const py = await python();
        const target = document.createElement("div");
        plot.append(target);
        document.pyodideMplTarget = target;
        await py.runPythonAsync(`qc_draw(${plot.dataset.result}, ${plot.dataset.position})`);
        status.remove();
        plot.dataset.state = "drawn";
    } catch (e) {
        status.textContent = "Could not draw this figure: " + e.message;
        plot.dataset.state = "failed";
    }
}
const inView = new Set();
let drawing = false;
async function drawAll() {
    if (drawing) return;
    drawing = true;
    const waiting = () => [...document.querySelectorAll(".qc-plot:not([data-state])")];
    for (let next; (next = [...inView].find(p => !p.dataset.state) ?? waiting()[0]); ) {
        await inTurn(() => draw(next));
        await new Promise(resolve => setTimeout(resolve, 0));  // let input through
    }
    drawing = false;
}
async function copyPython(button) {
    const plot = button.closest("figure").querySelector(".qc-plot");
    try {
        const py = await python();
        const text = py.runPython(`qc_copy(${plot.dataset.result}, ${plot.dataset.position})`);
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied";
    } catch (e) {
        button.textContent = "Could not copy: " + e.message;
    }
}
let consoleReady = null;
function runConsole() {
    const output = document.getElementById("console-output");
    output.textContent = "Loading Python...";
    return inTurn(async () => {
        try {
            const py = await python();
            consoleReady ??= py.runPythonAsync(document.getElementById("qc-analysis").textContent);
            await consoleReady;
            output.textContent = "";
            document.pyodideMplTarget = output;
            py.setStdout({batched: s => output.append(s + "\\n")});
            py.setStderr({batched: s => output.append(s + "\\n")});
            // show only the figures this code made, not the report's own
            await py.runPythonAsync("import matplotlib.pyplot as plt\\nqc_before = set(plt.get_fignums())");
            const result = await py.runPythonAsync(document.getElementById("console-input").value);
            if (result !== undefined) output.append(String(result) + "\\n");
            await py.runPythonAsync("for n in sorted(set(plt.get_fignums()) - qc_before):\\n"
                                    + "    qc_show(plt.figure(n))");
        } catch (e) {
            output.append(e.message);
        }
    });
}
// The overview, drawn from the results' index for the chosen board, run,
// module and status: the verdict, the problems, and a matrix of tests by
// module, or by board when all boards are chosen
let index = null;
const RANK = {pass: 0, skip: 0, near: 1, fail: 2};
function el(tag, attrs, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v != null) e.setAttribute(k, v);
    e.append(...children.flat().filter(c => c != null && c !== false));
    return e;
}
const word = ([w, cls]) => el("span", {class: "status " + cls}, w);
const link = (r, ...text) => el("a", {href: "#" + r.id}, ...text);
function filters() {
    return Object.fromEntries(["board", "run", "module", "show"].map(
        k => [k, document.getElementById("filter-" + k).value]));
}
function chosen(f) {
    return index.results.filter(r =>
        (f.board === "all" || r.serial === f.board)
        && (f.run === "latest" ? r.latest : r.run === f.run)
        && (f.module === "all" || r.module === Number(f.module))
        && (f.show === "all" || RANK[r.status[1]] > 0));
}
function fillRuns() {
    const board = document.getElementById("filter-board").value;
    const select = document.getElementById("filter-run");
    select.replaceChildren(el("option", {value: "latest"}, "latest results"),
        ...Object.entries(index.runs).filter(([, run]) => board === "all" || run.serial === board)
            .map(([uuid, run]) => el("option", {value: uuid}, `${run.serial}: run of ${run.started}`)));
}
// The runs shown, in one line: what they share once, what differs listed
function facts(runs, f) {
    const each = key => [...new Set(runs.map(run => run[key]))].join(", ");
    const what = f.run === "latest" && runs.length > 1
        ? `latest results from ${runs.length} runs (${each("started")})`
        : `run of ${each("started")}`;
    const by = [...new Set(runs.map(run => `${run.user}@${run.host}`))].join(", ");
    const boards = new Set(runs.map(run => run.serial)).size > 1 ? "Boards" : "Board";
    return runs.length ? `${boards} ${each("serial")}, ${what}; ${by}; rfmux ${each("rfmux")}; `
                         + `Python ${each("python")}` : "No results";
}
function render() {
    const f = filters();
    const members = chosen(f);
    const failed = members.filter(r => r.status[1] === "fail").length;
    const near = members.filter(r => r.status[1] === "near").length;
    const runs = [...new Set(members.map(r => r.run))].map(uuid => index.runs[uuid]);

    const problems = members.flatMap(r => r.problems
        .map(p => [RANK[p.status[1]], el("li", {}, word(p.status), " ",
                   link(r, r.title + (r.module ? `, module ${r.module}` : "")), ": ", p.text)]))
        .sort((a, b) => b[0] - a[0]).map(p => p[1]);

    // Columns: boards, or the board and its modules
    const boards = [...new Set(index.results.map(r => r.serial))];
    const byBoard = f.board === "all" && boards.length > 1;
    const modules = f.module !== "all" ? [Number(f.module)]
        : [...new Set(members.map(r => r.module).filter(m => m !== null))]
            .sort((a, b) => a - b);
    // a chosen module leaves no board-level results
    const columns = byBoard ? boards : f.module !== "all" ? modules : ["board", ...modules];
    const tests = new Map();
    for (const r of members) tests.set(r.test, [...(tests.get(r.test) || []), r]);
    const rows = [...tests.values()].map(rs => {
        const cells = [el("td", {class: "test"}, rs[0].title)];
        for (const column of columns) {
            let r = null, status = null;
            if (byBoard) {
                for (const candidate of rs.filter(x => x.serial === column)) {
                    const s = candidate.status;
                    if (!status || RANK[s[1]] > RANK[status[1]]) [r, status] = [candidate, s];
                }
            } else {
                r = rs.find(x => x.module === (column === "board" ? null : column));
                status = r && r.status;
            }
            cells.push(status ? el("td", {class: status[1]}, link(r, status[0]))
                              : el("td", {class: "none"}, "—"));
        }
        return el("tr", {}, cells);
    });

    document.getElementById("overview").replaceChildren(
        el("div", {class: "facts"}, facts(runs, f)),
        el("div", {class: "verdict " + (failed ? "fail" : "pass")},
           word(failed ? ["FAIL", "fail"] : ["PASS", "pass"]),
           failed ? ` ${failed} of ${members.length} results failed`
                  : ` all ${members.length} results passed`,
           near ? `, ${near} near miss${near > 1 ? "es" : ""}` : ""),
        problems.length ? el("div", {class: "problems"}, el("h2", {}, "Problems"),
                             el("ul", {}, problems)) : null,
        el("table", {class: "matrix"},
           el("thead", {}, el("tr", {}, el("th", {}, "Test"),
              columns.map(c => el("th", {}, byBoard ? `Board ${c}` : c === "board" ? "Board" : `Module ${c}`)))),
           el("tbody", {}, rows)));

    const ids = new Set(members.map(r => r.id));
    document.querySelectorAll(".result").forEach(e => { e.dataset.chosen = ids.has(e.id); });
    // keep the result the URL names; otherwise show the first problem, leaving
    // the URL alone, so that opening the report again starts at the top
    if (ids.has(location.hash.slice(1))) {
        select(location.hash.slice(1), false);
    } else {
        const first = document.querySelector("#overview .problems a, #overview .matrix a");
        select(first && first.hash.slice(1), false, false);
    }
}
function select(id, scroll = true, remember = true) {
    const chosen = id && document.getElementById(id);
    document.querySelectorAll(".result").forEach(e => e.classList.toggle("selected", e === chosen));
    document.querySelectorAll("#overview a").forEach(a => {
        a.classList.toggle("selected", !!chosen && a.hash === "#" + id);
    });
    document.getElementById("nothing-selected").hidden = !!chosen;
    if (chosen && remember) history.replaceState(null, "", "#" + id);
    if (chosen && scroll) document.getElementById("detail").scrollIntoView({behavior: "smooth"});
}
function step(by) {
    const ids = [...new Set([...document.querySelectorAll("#overview .problems a")]
                            .map(a => a.hash.slice(1)))];
    if (!ids.length) return;
    const at = ids.indexOf(location.hash.slice(1));
    select(ids[(at + by + ids.length) % ids.length]);
}
document.addEventListener("click", e => {
    const a = e.target.closest("#overview a");
    if (!a) return;
    e.preventDefault();
    select(a.hash.slice(1));
});
document.addEventListener("change", e => {
    if (!e.target.id.startsWith("filter-")) return;
    if (e.target.id === "filter-board") fillRuns();
    render();
});
window.addEventListener("hashchange", () => select(location.hash.slice(1)));
window.addEventListener("beforeprint", () => document.querySelectorAll("details").forEach(d => d.open = true));
window.addEventListener("DOMContentLoaded", () => {
    index = JSON.parse(document.getElementById("qc-index").textContent);
    fillRuns();
    render();
    const plots = document.querySelectorAll(".qc-plot");
    if (!plots.length) return;
    const watch = new IntersectionObserver(entries => entries.forEach(e => {
        e.isIntersecting ? inView.add(e.target) : inView.delete(e.target);
    }), {rootMargin: "200px"});
    plots.forEach(p => watch.observe(p));
    drawAll();
});
"""

CONSOLE_HINT = """\
# Python, with this report's data: db (sqlite3), checks, figures. For example:
#   [(c["serial"], c["module"], c["check"]) for c in checks if c["status"] != "PASS"]
#   f = figures[0]; f["plot"](f["data"])
#   db.execute("SELECT serial, test, outcome FROM run JOIN result ON result.run = run.id").fetchall()
"""


def status_of(outcome):
    return STATUS.get(outcome, (str(outcome).upper(), "skip"))


# Filenames that mean a plain SQLite database rather than a report
SQLITE = (".db", ".sqlite", ".sqlite3")


def read(filename):
    """The record in a report, or in an SQLite file (by its extension)"""
    path = pathlib.Path(filename)
    if path.suffix in SQLITE:
        data = path.read_bytes()
    else:
        m = re.search(r'<script type="application/gzip" id="qc-db">(.*?)</script>',
                      path.read_text(), re.S)
        if m is None:
            raise ValueError(f"{filename} holds no QC record")
        data = gzip.decompress(base64.b64decode(m[1]))
    try:
        return Record(data)
    except sqlite3.DatabaseError:
        raise ValueError(f"{filename} holds no QC record")


def load(filename):
    """
    A report's data for analysis: {"record", "db", "checks", "figures"}: the
    record (its runs, results, blocks and checks as objects), its sqlite3
    database, every check record, and every figure, where
    figure["plot"](figure["data"]) redraws it with matplotlib; check records
    and figures carry their serial, run and test. A list of records makes a
    pandas DataFrame.
    """
    record = read(filename)
    namespace = {"db": record.connection}
    exec(ANALYSIS, namespace)
    return {"record": record, **{k: namespace[k] for k in ("db", "checks", "figures")}}


def anchor(result):
    return f"result-{result}"


def write(record, filename):
    """
    A record as a report, or as an SQLite file (by its extension), replacing
    the file only once complete
    """
    path = pathlib.Path(filename)
    temporary = path.with_name(path.name + ".tmp")
    if path.suffix in SQLITE:
        temporary.write_bytes(record.to_bytes())
        os.replace(temporary, path)
        return

    record.session.flush()  # IDs, for anchors
    runs = record.runs  # most recent first
    results = sorted((r for run in runs for r in run.results),
                     key=lambda r: (r.run.serial, r.test, r.started))

    # A passing result with a near miss is NEAR
    status = {}
    for result in results:
        word, cls = status_of(result.outcome)
        near = any(c.status == "NEAR" for c in result.checks)
        status[result] = ("NEAR", "near") if cls == "pass" and near else (word, cls)

    latest = {(r.run.serial, r.test): r for r in results}  # by start: later ones win

    # Each result's title and description: its first heading, and the rest of
    # the markdown block that heading opens
    title_of, about_of = {}, {}
    for result in results:
        title_of[result] = re.sub(r"\[.*\]$", "", result.test)
        first = result.blocks[0] if result.blocks else None
        if first is not None and first.kind == "markdown":
            m = re.match(r"\s*#+\s*(.+?)\s*\n(.*)", first.content + "\n", re.S)
            if m:
                title_of[result] = re.sub(r"\s*\(Modules?[^)]*\)$", "", m[1])
                about_of[result] = m[2].strip()
    module_of = {r: (int(m[1]) if (m := re.search(r"\[(\d+)\]$", r.test)) else None)
                 for r in results}

    # Failures, to the line that says what failed (an exception's traceback
    # follows it)
    def failures(result):
        return [f.partition("\n")[0] for f in result.failures]

    # The index the page's overview is drawn from, filtered there by board,
    # run, module and status. Per result: its status, its module (none for
    # the board's), and its problems: failing checks and near misses, and
    # failures no check covers
    entries = []
    for result in results:
        word, cls = status[result]
        problems = []
        flagged = [c for c in result.checks if c.status != "PASS"]
        for c in flagged:
            context = ", ".join(f"{k} {v}" for k, v in c.context.items())
            limits = " to ".join(f"{v:.4g}" for v in (c.low, c.high) if v is not None)
            problems.append(dict(
                status=(c.status, c.status.lower()),
                text=f"{c.check}" + (f" ({context})" if context else "")
                     + f", {c.value:.4g} {c.unit}".rstrip()
                     + (f" (limits {limits})" if limits else "")))
        names = [c.check for c in flagged if c.status == "FAIL"]
        for failure in failures(result):
            if not any(failure.startswith(n) for n in names):
                problems.append(dict(status=("FAIL", "fail"), text=failure))
        if cls == "fail" and not problems:
            problems.append(dict(status=(word, cls), text=result.outcome))

        entries.append(dict(
            id=anchor(result.id), run=result.run.uuid, serial=result.run.serial,
            test=result.test.split("[")[0], title=title_of[result],
            module=module_of[result], status=(word, cls), problems=problems,
            latest=latest[(result.run.serial, result.test)] is result,
            position=result.position))
    entries.sort(key=lambda e: (e["position"], e["serial"]))  # the plan's order
    report_index = dict(
        results=entries,
        runs={run.uuid: dict(
            serial=run.serial, user=run.user, host=run.host, rfmux=run.rfmux,
            python=run.python,
            started=f"{datetime.datetime.fromisoformat(run.started):%Y-%m-%d %H:%M}")
            for run in runs})
    serials = list(dict.fromkeys(run.serial for run in runs))

    # Runs, and each result
    run_rows = []
    for run in runs:
        outcomes = [r.outcome for r in run.results]
        n_failed = sum(o in ("failed", "error") for o in outcomes)
        cls = "fail" if n_failed else "pass" if outcomes else None
        run_rows.append(tr(class_=cls)[
            td[run.serial], td[f"{datetime.datetime.fromisoformat(run.started):%Y-%m-%d %H:%M:%S}"],
            td[f"{run.user}@{run.host}"], td[run.rfmux], td[run.python], td[" ".join(run.args)],
            td[sum(o == "passed" for o in outcomes)], td[n_failed], td[len(outcomes)],
        ])
    result_rows = []
    for result in results:
        start = datetime.datetime.fromisoformat(result.started)
        stop = datetime.datetime.fromisoformat(result.stopped)
        word, cls = status_of(result.outcome)
        result_rows.append(tr[
            td[result.run.serial],
            td[f"{datetime.datetime.fromisoformat(result.run.started):%Y-%m-%d %H:%M}"],
            td[result.test], td[span(class_=f"status {cls}")[word]],
            td[f"{start:%H:%M:%S}"], td[f"{(stop - start).total_seconds():.1f} s"],
        ])
    run_details = details(class_="section")[
        summary["Runs"],
        table[thead[tr[[th[c] for c in ("Serial", "Started", "By", "rfmux", "Python",
                                         "Command", "Passed", "Failed", "Tests")]]],
              tbody[run_rows]],
        h2["Results"],
        table[thead[tr[[th[c] for c in ("Serial", "Run", "Test", "Status", "Start",
                                         "Duration")]]], tbody[result_rows]],
    ]

    # Each result's detail: measurements and figures first, its description
    # folded away; lists, not generators, throughout: htpy renders when the
    # page is written, after these loops have moved on
    fmt = lambda v: "" if v is None else f"{v:.4g}"
    sections = []
    for result in results:
        word, cls = status[result]
        content = []
        for block in result.blocks:
            if block.kind == "markdown":
                if block.position == 0 and result in about_of:
                    continue  # the title, with the description below
                content.append(markupsafe.Markup(markdown.markdown(
                    block.content, extensions=["tables", "attr_list"])))
            elif block.kind == "table":
                t = block.content
                content.append(table[
                    thead[tr[[th[c] for c in t["columns"]]]],
                    tbody[[tr[[td(class_=s)[v] for v in row]]
                           for row, s in zip(t["rows"], t["status"])]],
                ])
            elif block.kind == "checks":
                checks = [c for c in result.checks if c.block == block.position]
                context = list(dict.fromkeys(k for c in checks for k in c.context))
                has_nominal = any(c.nominal is not None for c in checks)
                content.append(table(class_="checks")[
                    thead[tr[
                        [th[k.capitalize()] for k in context],
                        th["Check"], th["Value"], has_nominal and th["Nominal"],
                        th["Low"], th["High"], th["Margin"], th["Status"],
                    ]],
                    tbody[[tr[
                        [td[str(c.context.get(k, ""))] for k in context],
                        td[c.check],
                        td[f"{fmt(c.value)} {c.unit}"],
                        has_nominal and td[fmt(c.nominal)],
                        td[fmt(c.low)],
                        td[fmt(c.high)],
                        td[fmt(c.margin)],
                        td(class_=c.status.lower())[
                            span(class_=f"status {c.status.lower()}")[c.status]
                        ],
                    ] for c in checks]],
                ])
            elif block.kind == "figure":
                w, h = block.content.get("size", (640, 480))
                content.append(figure[
                    div(class_="qc-plot", data_result=str(result.id),
                        data_position=str(block.position),
                        style=f"width: {w}px; aspect-ratio: {w} / {h}")[
                        div(class_="qc-plot-status")["Drawing with Python..."]
                    ],
                    figcaption[
                        block.content["caption"], " ",
                        button(class_="copy", onclick="copyPython(this)")["Copy as Python"],
                    ],
                ])
            elif block.kind == "image":
                content.append(figure[
                    img(src=f"data:image/png;base64,{block.content['png']}"),
                    figcaption[block.content["caption"]],
                ])

        when = datetime.datetime.fromisoformat(result.run.started)
        sections.append(section(id=anchor(result.id), class_=f"result {cls}")[
            h2[span(class_=f"status {cls}")[word], " ", title_of[result],
               module_of[result] is not None
               and span(class_="module")[f" module {module_of[result]}"]],
            div(class_="meta")[f"{result.run.serial}, run of {when:%Y-%m-%d %H:%M}, {result.test}"],
            ul(class_="failures")[[li[f] for f in failures(result)]],
            about_of.get(result) and details(class_="about")[
                summary["About this test"],
                markupsafe.Markup(markdown.markdown(about_of[result],
                                                    extensions=["tables", "attr_list"])),
            ],
            content,
        ])

    title_ = f"CRS QC: {', '.join(serials)}"
    css = (pathlib.Path(__file__).parent / "style.css").read_text()
    doc = html(lang="en")[
        head[
            meta(charset="utf-8"),
            meta(name="viewport", content="width=device-width, initial-scale=1"),
            meta(name="generator", content=f"rfmux qc, {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
                                           f" on {socket.gethostname()}"),
            title[title_],
            style[markupsafe.Markup(css)],
            script[markupsafe.Markup(SCRIPT)],
        ],
        body[
            header(class_="topbar")[
                h1[f"CRS QC: {', '.join(serials)}"],
                label["Board ", select(id="filter-board")[
                    len(serials) > 1 and option(value="all")["all boards"],
                    [option(value=serial)[serial] for serial in serials],
                ]],
                label["Run ", select(id="filter-run")],
                label["Module ", select(id="filter-module")[
                    option(value="all")["all modules"],
                    [option(value=str(m))[m] for m in sorted(
                        {e["module"] for e in entries if e["module"] is not None})],
                ]],
                label["Show ", select(id="filter-show")[
                    option(value="all")["all results"],
                    option(value="problems")["problems only"],
                ]],
                label(class_="show-all")[
                    input_(type="checkbox", onchange="document.body.classList.toggle("
                                                     "'show-all', this.checked)"),
                    " Every result's detail",
                ],
            ],
            main[
                div(id="overview"),
                div(id="detail")[
                    div(class_="stepper")[
                        button(onclick="step(-1)")["\u2190 Previous problem"],
                        button(onclick="step(1)")["Next problem \u2192"],
                    ],
                    p(id="nothing-selected")["Choose a result above."],
                    sections,
                ],
                run_details,
                details(class_="section", id="console")[
                    summary["Python console"],
                    textarea(id="console-input", rows="8", spellcheck="false")[
                        CONSOLE_HINT
                    ],
                    button(onclick="runConsole()")["Run"],
                    pre(id="console-output"),
                ],
            ],
            script(type="application/json", id="qc-index")[
                markupsafe.Markup(json.dumps(report_index).replace("</", "<\\/"))
            ],
            script(type="text/x-python", id="qc-analysis")[markupsafe.Markup(ANALYSIS)],
            script(type="text/x-python", id="qc-browser")[markupsafe.Markup(BROWSER)],
            script(type="application/gzip", id="qc-db")[
                base64.b64encode(gzip.compress(record.to_bytes(), mtime=0)).decode()
            ],
        ],
    ]

    # A run interrupted mid-write must not lose the record
    temporary.write_text(str(doc))  # htpy writes the doctype
    os.replace(temporary, path)
