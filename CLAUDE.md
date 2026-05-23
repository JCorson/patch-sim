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

## Spelling

Use **US English** in all code, comments, docstrings, and documentation. LLM-assisted writing on neuroscience topics tends to drift toward British forms (depolarise, hyperpolarise, behaviour, modelling, centre, fibre, colour, analyse, initialise, etc.) because of the training-corpus mix. Always emit the US form: `depolarize`, `hyperpolarize`, `behavior`, `modeling`, `center`, `fiber`, `color`, `analyze`, `initialize`, etc.

## Biological accuracy

**Prioritize biological accuracy** when designing or modifying neuron presets. If a preset produces behavior that doesn't match known physiology (wrong resting potential, wrong firing pattern, missing channels), fix the underlying biology rather than working around it. When a full biological fix is out of scope (e.g. a channel type not yet implemented), document the known limitation explicitly in the preset comment and in an issue — do not silently accept a biologically wrong default.

## Comments and docstrings

Comments and docstrings explain the **current** design — what the code does and why it's biologically or numerically correct — in the present tense. They are not a changelog.

Do **not** include:

- Issue or PR numbers cited as "this fixes…" or "this changed in…" (`#347`, `#348`).
- Temporal framing about prior states: "before this PR", "post-#348", "prior to the alignment", "was previously", "originally proposed", "legacy", "restores".
- Descriptions of removed code or the bug that was fixed ("was silently stretched 2.5×", "the earlier rate was a 100 kHz protocol re-interpreted as 40 kHz").
- "Widened from X to Y" or "extended from X to Y" framings — just state the current band/duration and why.

Investigation history belongs in the PR body and the issue thread, not in source.

Issue links **are** appropriate when documenting an open limitation that has not been fixed yet (see `## Biological accuracy`).

## Architecture

- `patch_sim/` — pure Python library, **no Reflex dependency**. All simulation logic lives here. Notable subpackages: `patch_sim/analysis/` (post-hoc metrics: AP metrics, F-I, G-V, I-V, passive properties, burst metrics, etc.) and `patch_sim/protocols/` (current/voltage protocol builders).
- `patch_sim_ui/` — Reflex application. State is split across `patch_sim_ui/state/` (neuron, protocol, simulation, analysis, visibility, log); components are Python functions returning `rx.Component`.

Keep these layers cleanly separated: do not import Reflex types into `patch_sim/`, and do not import `patch_sim_ui` modules from `patch_sim/`.

## UI framework

The UI uses **Reflex** (v0.8+), a Python framework that compiles to Next.js.

Install UI dependencies and run the dev server:

```bash
uv sync --frozen --group=ui
uv run reflex run
```

The app serves on `http://localhost:3000`.

## Package manager

This project uses **uv**. Use `/uv run --frozen` to run commands and `/uv add` to add dependencies. Never use `pip` directly.

## Docstring requirements

Every function and method — public, private, and dunder — must have a Google-style docstring. Ruff enforces public items automatically; private methods (`_foo`) are not checked by ruff but are still required.

- Omit type annotations from `Args:` — they are already in the signature

## Documentation

There are two documentation tracks, plus the docstrings that feed the API reference. Evaluate and update whatever a change touches:

- **Library documentation** — the mkdocs site, for developers using `patch_sim` in Python. Prose pages live in `docs/` (`index.md`, `presets.md`, `protocols-and-analysis.md`) and the API reference is auto-generated under `docs/api/`. Update the relevant prose page when library behavior or supported options change. Build it to catch broken references: `uv run --frozen --group=docs mkdocs build --strict`. Avoid bare `$` and keep to Markdown that renders under both mkdocs and GitHub.
- **Application documentation** — the in-app `/help` route, a user guide for the web app. Pages live in `patch_sim_ui/help_content/*.md` and are rendered by `rx.markdown`; add a new topic to `_TOPICS` in `patch_sim_ui/docs_loader.py` when you add a page. Update these when the UI's controls, panels, or workflow change. Screenshots are served from `assets/screenshots/` and regenerated with `uv run --frozen --group screenshots python tools/capture_screenshots.py` (against a running `uv run reflex run`); refresh them when the UI's appearance changes.
- **Docstrings** — Google-style docstrings on public symbols feed the mkdocstrings API reference. When you add or change a public symbol in `patch_sim/`, update its docstring and add a `:::` entry under `docs/api/` if it is a new top-level export.

## Testing conventions

Tests are split into four buckets under `tests/`:

- `tests/unit/` — fast, pure-function tests; no simulation runs. Run alone during development: `uv run --frozen -m pytest tests/unit`
- `tests/integration/` — end-to-end protocol → simulation pipeline tests. Simulation-calling tests extracted from mixed files are named `*_simulation.py`.
- `tests/ui/` — Reflex and plotting layer tests; skipped when `reflex` is not installed.
- `tests/e2e/` — headless full-pipeline tests that drive Reflex state handlers directly (no dev server, no browser). **Not in default `testpaths`** — run explicitly with `uv run --frozen -m pytest tests/e2e`.

The shared `hh_model` fixture lives in `tests/conftest.py` and is discoverable by all four subdirectories. `tests/e2e/conftest.py` adds e2e-specific fixtures.

Pytest is configured with `-n auto --dist=loadscope` (parallel via xdist, tests in the same module pinned to the same worker).

Additional conventions:
- Use plain functions, not test classes
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

Run all five checks and ensure they pass before marking a task complete:

**Test suite:** `/uv run --frozen -m pytest --verbose`
**Formatting:** `/ruff check .` and `/ruff format --check .`
**Type checking:** `/ty check`
**Docs build (when `docs/` or public docstrings changed):** `uv run --frozen --group=docs mkdocs build --strict`

All five must pass with no errors.
