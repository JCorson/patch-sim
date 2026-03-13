# Agent Instructions

## Project overview

Hodgkin-Huxley action potential simulator with a core library (`ap_sim/`) and a Reflex web UI (`ap_sim_ui/`). The core library models ion channels, gating variables, voltage/current clamp protocols, and the HH differential equations. The UI provides interactive controls and live trace plots.

## Architecture

- `ap_sim/` — pure Python library, **no Reflex dependency**. All simulation logic lives here.
- `ap_sim_ui/` — Reflex application. State lives in `ap_sim_ui/state.py`; components are Python functions returning `rx.Component`.

Keep these layers cleanly separated: do not import Reflex types into `ap_sim/`, and do not import `ap_sim_ui` modules from `ap_sim/`.

## UI framework

The UI uses **Reflex** (v0.8+), a Python framework that compiles to Next.js. To run the app locally:

```bash
uv run --frozen reflex run
```

## Package manager

This project uses **uv**. Use `uv run --frozen` to run commands and `uv add` to add dependencies. Never use `pip` directly.

## Docstring requirements

Every function and method — public **and** private, including dunder methods — must have a docstring. Use **Google style** consistently:

```python
def example(x: float, y: int) -> str:
    """One-line summary.

    Optional extended description.

    Args:
        x: Description of x.
        y: Description of y.

    Returns:
        Description of the return value.
    """
```

- Summary goes on the **first line** immediately after the opening `"""`
- Omit type annotations from `Args:` — they are already in the signature
- Use `Args:`, `Returns:`, `Raises:`, `Attributes:` (never `Parameters:`)

## Testing conventions

- Test files live in `tests/`
- Use plain functions, not test classes
- Use the shared `hh_model` fixture from `tests/conftest.py` when a model instance is needed
- Use `pytest.approx` for float comparisons
- Every test function needs a Google-style docstring

## Type checking

mypy only enforces types in `ap_sim/` (core library). UI code (`ap_sim_ui/`) is excluded — see the `[[tool.mypy.overrides]]` section in `pyproject.toml`. Do not spend time fixing mypy errors in UI code.

## Before finishing any task

Run all four checks and ensure they pass before marking a task complete:

```bash
uv run --frozen -m pytest --verbose          # run the test suite
uv tool run ruff check .                     # check style
uv tool run ruff format --check .            # check formatting
uv run --frozen -m mypy .                    # run mypy type checking
```

All four must pass with no errors.
