## ap-sim: Action Potential Simulator

`ap_sim` is a Python library for simulating neuronal electrophysiology using the
[Hodgkin-Huxley model](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model).
It models how action potentials arise at a neuron's membrane by simulating the dynamics
of voltage-gated ion channels (sodium, potassium, and leak) over time.

Key features:

- **Hodgkin-Huxley neuron model** with configurable conductances, ion concentrations,
  and temperature. Reversal potentials are automatically derived from ion concentration
  gradients via the Nernst equation.
- **Current clamp** simulations: inject a stimulus current and record the resulting
  membrane voltage response.
- **Voltage clamp** simulations: hold the membrane at a commanded voltage and record the
  resulting ionic currents.
- A library of **stimulation protocols** covering step, ramp, pulse train, sinusoidal,
  chirp, noise, I-V curve, and activation waveforms.
- Simulation results are returned as `pandas.DataFrame` objects indexed by time (ms),
  making them straightforward to analyse or plot with standard Python tools.
- Numerically stable handling of the singularity points in the Hodgkin-Huxley rate
  equations.

---

## Getting started with development

### Prerequisites

You should have the following installed:

- [uv](https://docs.astral.sh/uv/getting-started/installation/): Python package manager


### Development tooling

Common development tasks are run directly via `uv`. The commands below assume
you are at the top level of the repository.

#### Installing the development environment

```
uv sync --frozen --extra=dev
```

This creates a Python `venv` virtual environment under the name `.venv`
in the top-level directory of the repository.

#### Managing dependencies

To regenerate all project dependencies files, run:

```
uv lock --upgrade
```

#### Unit tests

The full unit test suite can be run using:

```
uv run --frozen -m pytest --verbose
```

#### Linting and code style

Python code is styled using Ruff. Ruff is configured to cover typical flake8,
black and isort behavior.

Formatting fixes can be applied using:

```
uv tool run ruff format .
uv tool run ruff check . --fix-only
```

Conformance to the styling rules can be checked with:

```
uv tool run ruff check .
uv tool run ruff format --check .
```

#### Type checking

```
uv run --frozen -m mypy .
```
