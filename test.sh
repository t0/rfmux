#!/bin/bash

if ! which -s uv
then
	cat <<EOF >&2
Please install uv! Instructions are available at:

	https://docs.astral.sh/uv/getting-started/installation

They will tell you to run something like:

	curl -LsSf https://astral.sh/uv/install.sh | sh

You should generally be suspicious of instructions like these.
EOF
	exit -1
fi

uv tool install -q tox --with tox-uv
uv run tox run-parallel
