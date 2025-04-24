## Getting started with development

### Prerequisites

You should have the following installed:

- [uv](https://docs.astral.sh/uv/getting-started/installation/): Python package manager

Note that `uv` should be installed via the stand-alone installer (i.e. not via `pip install uv`) in order to make use
of all the features that we depend on.


### Development tooling

The `ci.py` script at the top level of the repository provides a `click`-based
CLI for performing common development tasks. It's designed to be run with
`uv run`. To see an overview of available commands, execute

    uv run ci.py

Developers may want to set up a shell alias, for example with

  - **bash/zsh**

        alias ci="uv --quiet run ci.py"

The following documentation assumes the existence of such an alias.

#### Installing the development environment

You can install the Python packages necessary for development using:

```
ci install
```

This creates a Python `venv` virtual environment under the name `.venv`
in the top-level directory of the repository. If necessary, that virtual
environment can be activated in the normal way with

   - **macOS/Linux**

     ```
     source .venv/bin/activate
     ```

   - **Windows**
     ```
     .venv\Scripts\activate
     ```

However, it's not necessary to activate the environment to run the `ci`
commands: the expected workflow is that the `ci` commands are run _outside_ the
activated environment.

#### Managing dependencies

To regenerate all project dependencies files, run:

```
ci update-dependencies
```

In general, there's no need to run this command on a daily or even weekly basis.
Dependencies should be updated:

- whenever you change the list of packages in any tracked `pyproject.toml` file
- occasionally (say every few weeks) to ensure that you're using packages with the
  latest bugfixes and security fixes, and to get early warning about potential
  incompatibilities with future versions of packages
- not just before a release!

#### Unit tests

The full unit test suite can be run using:

```
ci test
```

#### Linting and code style

Python code is styled using Ruff. Ruff is configured in the ci commands to
cover typical flake8, black and isort behavior.

Formatting fixes for most code style rules can be applied using:

```
ci format
```

Conformance to the styling rules can be checked with:

```
ci lint
```