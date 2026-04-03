# Agent Instructions

## Project overview

Patch clamp experiment simulator with a core library (`patch_sim/`) and a Reflex web UI (`patch_sim_ui/`). The core library models ion channels, gating variables, voltage/current clamp protocols, and the HH differential equations. The UI provides interactive controls and live trace plots.

## Architecture

- `patch_sim/` — pure Python library, **no Reflex dependency**. All simulation logic lives here.
- `patch_sim_ui/` — Reflex application. State lives in `patch_sim_ui/state.py`; components are Python functions returning `rx.Component`.

Keep these layers cleanly separated: do not import Reflex types into `patch_sim/`, and do not import `patch_sim_ui` modules from `patch_sim/`.

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

mypy only enforces types in `patch_sim/` (core library). UI code (`patch_sim_ui/`) is excluded — see the `[[tool.mypy.overrides]]` section in `pyproject.toml`. Do not spend time fixing mypy errors in UI code.

## GitHub CLI conventions

`gh issue view <number>` triggers a GraphQL deprecation error because it fetches `projectCards`. Always request explicit fields instead:

```bash
gh issue view <number> --json number,title,body,state,labels,assignees,comments
```

## Working on GitHub issues

When addressing a GitHub issue, create a dedicated branch off `main` before writing any code:

```bash
git checkout main
git pull
git checkout -b <branch-name>
```

Name the branch descriptively (e.g. `issue-42-voltage-clamp-fix`). Do all work for that issue on that branch. Do not commit issue-related changes directly to `main`.

## Committing during multi-step plans

When executing a plan that has discrete numbered steps, commit after each step
rather than staging everything at the end. Each commit should cover exactly one
logical change (one fix, one refactor, one feature addition) so that the git
log reflects the plan's structure.

If a step touches several files but they all belong to the same logical change,
group them in a single commit. If a file contains changes from different plan
steps, stage only the relevant hunks for each commit.

Run the four checks listed below before every commit — not just at the end.

## Before finishing any task

Run all four checks and ensure they pass before marking a task complete:

```bash
uv run --frozen -m pytest --verbose          # run the test suite
uvx ruff check .                     # check style
uvx ruff format --check .            # check formatting
uvx ty check                                 # run ty type checking
```

All four must pass with no errors.
