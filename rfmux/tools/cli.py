#!/usr/bin/env python3
"""
rfmux - Single point-of-entry CLI for rfmux tooling.

This avoids the need to give rfmux tools clever/unique names - they can be
accessed via

    $ rfmux parser
    $ rfmux periscope

rather than, generically

    $ parser

(which seems like a fairly easy command-name collision to engineer). Depending
on your Python environment, another noun might need to get stacked in front:

    $ uv run rfmux parser
    $ uv run rfmux periscope

...which is admittedly a little clunky, but at least organized.

Subcommands are discovered automatically: any module or subpackage directly
under rfmux/tools/ that exposes a module-level "cli" click command becomes
"rfmux <module-name>". There is nothing further to register by hand - drop a
new tool in rfmux/tools/ with a "cli" click.Command/Group and it appears.

Discovery only inspects module names (via pkgutil); it does not import
candidate modules until the corresponding subcommand is actually invoked (or
its --help is requested). This matters because some tools (e.g. periscope)
pull in heavyweight/GUI dependencies that shouldn't be a tax on unrelated
commands or on "rfmux --help".
"""

import importlib
import pkgutil

import click

import rfmux.tools as _tools_pkg


def _discover_command_names():
    """Names of rfmux/tools/ modules and subpackages that might export a cli."""
    names = []
    for module_info in pkgutil.iter_modules(_tools_pkg.__path__):
        name = module_info.name
        if name in ("cli",) or name.startswith("_"):
            continue
        names.append(name)
    return sorted(names)


def _load_command(name):
    """Import rfmux.tools.<name> and return its "cli" attribute, if any."""
    module = importlib.import_module(f"rfmux.tools.{name}")
    return getattr(module, "cli", None)


class LazyToolGroup(click.Group):
    """A click.Group whose subcommands are discovered from rfmux/tools/.

    Candidate modules are only imported when their subcommand is actually
    used, so unrelated commands never pay the cost (or dependency risk) of
    importing every tool in the package.
    """

    def list_commands(self, ctx):
        return _discover_command_names()

    def get_command(self, ctx, name):
        try:
            command = _load_command(name)
        except ImportError:
            return None
        if not isinstance(command, click.BaseCommand):
            return None
        return command


@click.group(cls=LazyToolGroup)
@click.version_option()
def main():
    """RF Multiplexer suite of tools"""
    pass


if __name__ == '__main__':
    main()
