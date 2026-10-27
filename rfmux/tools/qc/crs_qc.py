"""
Blocks a QC test appends to its report section. Each is plain data (JSON once
numpy arrays become lists), so the report embeds them as its record and
report_generator renders them.
"""

import base64
import contextlib
import inspect
import io
import json
import matplotlib.figure
import matplotlib.pyplot as plt
import textwrap


def render_markdown(source: str):
    """A Markdown block; indentation common to its lines is removed"""
    return {"markdown": inspect.cleandoc(source)}


def render_fig(fig: matplotlib.figure.Figure, caption: str = ""):
    """A static figure, as PNG"""
    with contextlib.closing(io.BytesIO()) as buf:
        fig.savefig(buf, format="png")
        png = base64.b64encode(buf.getbuffer()).decode()
    return {"image": {"png": png, "caption": caption}}


def render_plot(plot, data, caption: str = ""):
    """
    A figure as the function that draws it and its data. The report runs it
    with Pyodide to draw the figure interactively, and
    report_generator.load() returns both.

    "plot" takes the data as JSON gives it back (numpy arrays as lists) and
    returns a matplotlib figure; it must stand alone, imports included. It
    runs here from its source on that data, so a function leaning on anything
    outside itself, or that can't take the data, fails now rather than in a
    reader's browser.
    """
    source = textwrap.dedent(inspect.getsource(plot))
    data = json.loads(json.dumps(data, default=lambda v: v.tolist()))
    namespace = {}
    exec(source, namespace)
    fig = namespace[plot.__name__](data)
    size = [round(v) for v in fig.get_size_inches() * fig.dpi]  # pixels, as drawn
    plt.close(fig)
    return {"figure": dict(name=plot.__name__, source=source, data=data, size=size,
                           caption=caption)}


class ResultTable:
    """
    A table where each row is a "pass/fail" result.

        >>> st = ResultTable("col1", "col2", "col3")
        >>> st.pass_("val1", "val2", "val3")  # for "green" results
        >>> st.fail("val1", "val2", "val3")  # for "red" results
    """

    @staticmethod
    def formatter(x):
        """Default numeric formatter - 3 significant figures for floats"""
        match x:
            case int():
                return f"{x:n}"
            case float():
                return f"{x:.03f}"
            case _:
                return str(x)

    def __init__(self, *headings):
        self.headings = headings
        self.entries = []
        self.classes = []

    def pass_(self, *row):
        self.entries.append(row)
        self.classes.append("pass")

    def fail(self, *row):
        self.entries.append(row)
        self.classes.append("fail")

    def row(self, *row):
        self.entries.append(row)
        self.classes.append(None)

    def to_block(self):
        return {"table": dict(
            columns=list(self.headings),
            rows=[[self.formatter(v) for v in row] for row in self.entries],
            status=self.classes,
        )}


class CheckTable:
    """
    Checks measurements against limits, and records them for the report.

        >>> ct = CheckTable()
        >>> ct.check("Weakest tone", -32.4, "dBm", low=-40, warn=10, module=2)

    A value outside its limits (exclusive) fails its step, with a message in
    "failures"; one within "warn" of a limit is a near miss. Keyword
    arguments beyond these (module=2) become leading columns.
    """

    def __init__(self):
        self.records = []
        self.failures = []

    def check(self, name, value, unit="", *, low=None, high=None, nominal=None,
              warn=None, **context):
        value = float(value)
        margins = [value - low] if low is not None else []
        margins += [high - value] if high is not None else []
        margin = min(margins) if margins else None

        if margin is not None and margin <= 0:
            status = "FAIL"
            where = ", ".join(f"{k} {v}" for k, v in context.items())
            self.failures.append(f"{name}{f' ({where})' if where else ''}: {value:.4g} {unit} "
                       f"outside ({low}, {high})")
        elif warn is not None and margin < warn:
            status = "NEAR"
        else:
            status = "PASS"

        self.records.append(
            dict(context, check=name, value=value, unit=unit, nominal=nominal, low=low,
                 high=high, margin=margin, warn=warn, status=status)
        )
        return status != "FAIL"

    def to_block(self):
        return {"checks": self.records}
