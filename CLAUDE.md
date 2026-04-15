# Agent Instructions

When working with Python, invoke the relevant /astral:<skill> for uv, ty, and ruff to ensure best practices are followed.

### Code Intelligence

Prefer LSP over Grep/Glob/Read for code navigation:
- `goToDefinition` / `goToImplementation` to jump to source
- `findReferences` to see all usages across the codebase
- `workspaceSymbol` to find where something is defined
- `documentSymbol` to list all symbols in a file
- `hover` for type info without reading the file
- `incomingCalls` / `outgoingCalls` for call hierarchy

Before renaming or changing a function signature, use
`findReferences` to find all call sites first.

Use Grep/Glob only for text/pattern searches (comments,
strings, config values) where LSP doesn't help.

After writing or editing code, check LSP diagnostics before
moving on. Fix any type errors or missing imports immediately.
Only fall back to Grep/Glob when the LSP cannot answer the question (e.g. searching for a raw string pattern, or locating non-Python files).

## Project overview

Patch clamp experiment simulator with a core library (`patch_sim/`) and a Reflex web UI (`patch_sim_ui/`). The core library models ion channels, gating variables, voltage/current clamp protocols, and the HH differential equations. The UI provides interactive controls and live trace plots.

## Architecture

- `patch_sim/` — pure Python library, **no Reflex dependency**. All simulation logic lives here.
- `patch_sim_ui/` — Reflex application. State is split across `patch_sim_ui/state/` (neuron, protocol, simulation, analysis, visibility, log); components are Python functions returning `rx.Component`.

Keep these layers cleanly separated: do not import Reflex types into `patch_sim/`, and do not import `patch_sim_ui` modules from `patch_sim/`.

## UI framework

The UI uses **Reflex** (v0.8+), a Python framework that compiles to Next.js.

## Package manager

This project uses **uv**. Use `/uv run --frozen` to run commands and `/uv add` to add dependencies. Never use `pip` directly.

## Docstring requirements

Every function and method — public, private, and dunder — must have a Google-style docstring. Ruff enforces public items automatically; private methods (`_foo`) are not checked by ruff but are still required.

- Omit type annotations from `Args:` — they are already in the signature

## Testing conventions

- Test files live in `tests/`
- Use plain functions, not test classes
- Use the shared `hh_model` fixture from `tests/conftest.py` when a model instance is needed
- Use `pytest.approx` for float comparisons
- Every test function needs a Google-style docstring

## Type checking

ty enforces types in `patch_sim/` (core library) and UI code (`patch_sim_ui/`). Reflex, Plotly, and `patch_sim_ui` imports are replaced with `Any` for UI files and selected test files — see the `[[tool.ty.overrides]]` section in `pyproject.toml`. Do not spend time fixing ty errors in UI code.

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

**Test suite:** `/uv run --frozen -m pytest --verbose`
**Formatting:** `/ruff check .` and `/ruff format --check .`
**Type checking:** `/ty check`

All four must pass with no errors.
