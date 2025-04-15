import base64
import contextlib
import htpy
import io
import markdown
import markupsafe
import matplotlib.figure
import textwrap


def render_fig(fig: matplotlib.figure.Figure):
    """Convert a Figure into an embedded base64-encoded string"""
    with contextlib.closing(io.BytesIO()) as buf:
        fig.savefig(buf, format="png")
        img_base64 = base64.b64encode(buf.getbuffer()).decode()
        return htpy.img(src=f"data:image/png;base64,{img_base64}")


def render_markdown(source: str):
    """
    Convert Markdown (in a string) to markupsafe AST nodes

    The declarative HTML library we use (htpy) understands these directly.
    """
    docstring = textwrap.dedent(source)
    md = markdown.markdown(docstring, extensions=["tables", "attr_list"])
    doc = markupsafe.Markup(md)
    return doc


class ResultTable:
    """
    Renders a table where each row is a "pass/fail" result.

    You instantiate a table with its columns as arguments as follows:

        >>> st = ResultTable("col1", "col2", "col3")

    then populate rows as follows:

        >>> st.pass_("val1", "val2", "val3")  # for "green" results
        >>> st.fail("val1", "val2", "val3")  # for "red" results

    HTML can then be generated with the __call__ method. Since htpy expects
    callables, you can passed "st" to htpy directly.
    """

    @staticmethod
    def formatter(x):
        """Default numeric formatter - 3 significant figures for floats"""
        match x:
            case int():
                return f"{x:n}"
            case float():
                return f"{x:.03f}"
            case str():
                return x
            case _:
                # assume they know what they're doing - might be a htpy native type
                return x

    def __init__(self, *headings):
        self.headings = headings
        self.entries = []
        self.classes = []

    def set_formatter(self, formatter):
        self.formatter = formatter

    def pass_(self, *row):
        self.entries.append(row)
        self.classes.append("pass")

    def fail(self, *row):
        self.entries.append(row)
        self.classes.append("fail")

    def row(self, *row):
        self.entries.append(row)
        self.classes.append(None)

    def __call__(self):
        return htpy.table[
            htpy.thead[(htpy.th[c] for c in self.headings)],
            htpy.tbody[
                (
                    htpy.tr[
                        htpy.td(class_=c)[s],
                        (htpy.td(class_=c)[v] for v in map(self.formatter, vs)),
                    ]
                    for (c, (s, *vs)) in zip(self.classes, self.entries)
                )
            ],
        ]
