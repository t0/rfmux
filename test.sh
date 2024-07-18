#!/bin/bash

# Check pre-requisites. Most of these are needed for a complete python build,
# but it's reasonable to include other ingredients here as well.
PACKAGES="sqlite3 readline tk libffi liblzma"
for package in $PACKAGES;
do
	if ! pkg-config --exists $package; then
		echo "Need $package to correctly build a Python environment for testing. Please install it."
		echo "hint: sudo apt-get install libsqlite3-dev libreadline-dev tk-dev libffi-dev liblzma-dev"
		exit -1
	fi
done

# Get pyenv into our environment, installing if necessary
if ! command -v pyenv &> /dev/null; then

	if [ ! -e ~/.pyenv ]; then
		echo "pyenv not found, installing..."
		# This is alarming - but you're running arbitrary code
		# regardless, whether we use git or this abomination
		curl https://pyenv.run | bash
	fi

	# Add pyenv to PATH (consider adding this to your shell configuration)
	export PATH="$HOME/.pyenv/bin:$PATH"
	eval "$(pyenv init --path)"
fi

# Install necessary Python versions using pyenv
PYTHON_VERSIONS=("3.10.13" "3.11.8" "3.12.2")
for version in "${PYTHON_VERSIONS[@]}"; do
    if ! pyenv versions --bare | grep -q "^$version$"; then
        pyenv install "$version"
    fi
done

# Optionally set a local Python version for the project
pyenv local 3.10.13 3.11.8 3.12.2

# Ensure tox is installed locally (don't use the distro version)
pip3.10 install --upgrade pip setuptools tox

# Now kick off regression tests
python3.10 -m tox
