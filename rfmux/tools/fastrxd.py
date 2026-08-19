"""Pass-through launcher for the fastrxd binary, wired into "rfmux fastrxd".

Since fastrxd needs root (which this launcher should never satisfy!), we just
invoke fastrxd unprivileged and rely on it to bail with helpful instructions.
"""

import click
import os

@click.command(name="fastrxd", context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
    help_option_names=[],  # forward --help to fastrxd's own, not click's
))
@click.argument("fastrxd_args", nargs=-1, type=click.UNPROCESSED)
def cli(fastrxd_args):
    """Launch fastrxd"""

    # Imported here rather than at module scope so this command still
    # registers in a build without fastrx.
    try:
        from ..streamer import _fastrx
    except ImportError:
        raise click.ClickException(
            "this rfmux build does not include fastrxd. It is built "
            "automatically on Linux when clang, libxdp and libbpf are "
            "present at install time; install them and reinstall rfmux "
            "(e.g. uv pip install -e . --force-reinstall).")

    # fastrxd is a sibling of _fastrx.so, wherever this install put it.
    exe = os.path.join(os.path.dirname(os.path.abspath(_fastrx.__file__)), "fastrxd")
    os.execv(exe, [exe, *fastrxd_args])
