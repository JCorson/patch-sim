## patch-sim: Patch Clamp Simulator

`patch_sim` is a Python library for simulating neuronal electrophysiology using the
[Hodgkin-Huxley model](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model).
It models how action potentials arise at a neuron's membrane by simulating the dynamics
of voltage-gated ion channels over time. The core model includes the classic sodium,
potassium, and leak channels, and can be extended with 10 additional channel types
(Ih, IKa, INaP, INaR, IM, IKir, IKCa, ICaL, ICaT, ICaN) via a pluggable channel
architecture.

Key features:

- **Hodgkin-Huxley neuron model** with configurable conductances, ion concentrations,
  and temperature. Reversal potentials are derived via the Nernst equation, with
  Goldman-Hodgkin-Katz support for mixed-ion channels (e.g. Ih). Optional intracellular
  calcium dynamics are coupled to calcium-permeable channels.
- **Current clamp** simulations: inject a stimulus current and record the resulting
  membrane voltage response.
- **Voltage clamp** simulations: hold the membrane at a commanded voltage and record the
  resulting ionic currents.
- A library of **stimulation protocols**: current clamp (step, ramp, pulse train,
  sinusoidal, chirp, noise) and voltage clamp (step, ramp, pulse train, I-V curve).

---

## Web UI

`patch_sim` includes an interactive web interface inspired by pClamp electrophysiology software.

- Configure neuron parameters and enable/disable additional ion channels with conductance sliders
- Run current clamp and voltage clamp protocols
- Sweep overlay with click/hover/keyboard selection
- Continuous simulation mode for oscilloscope-like real-time recording
- 6 built-in presets: Action Potential, Subthreshold Response, Repetitive Firing, I-V Curve, Na+ Channel Activation, Frequency Response

### Installing the UI

```
uv sync --frozen --extra=ui
```

### Running the UI

```
uv run reflex run
```

Then open `http://localhost:3000` in your browser.

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

To regenerate and update project dependencies, run:

```
uv lock --upgrade
```

#### Unit tests

The full unit test suite can be run using:

```
uv run --frozen -m pytest --verbose
```

#### Test coverage

To see a line-by-line coverage report for the core library:

```
uv run --frozen -m pytest --cov=patch_sim --cov-report=term-missing
```

To generate an HTML report (written to `htmlcov/`):

```
uv run --frozen -m pytest --cov=patch_sim --cov-report=html
```

#### Linting and code style

Python code is styled using Ruff. Ruff is configured to cover typical flake8,
black and isort behavior.

Formatting fixes can be applied using:

```
uvx ruff format .
uvx ruff check . --fix-only
```

Conformance to the styling rules can be checked with:

```
uvx ruff check .
uvx ruff format --check .
```

#### Type checking

```
uvx ty check
```

---

## License

This project is licensed under the [GNU General Public License v3.0 or later](LICENSE).
