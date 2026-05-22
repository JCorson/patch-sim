# Developer guide

This guide covers the architecture, conventions, and tooling for working on
`patch_sim` itself, and how to use the library programmatically.

## The layer rule

The codebase has two layers that must stay cleanly separated:

- **`patch_sim/`** is a pure-Python library with no UI dependency. All
  simulation, channel, protocol, and analysis logic lives here.
- **`patch_sim_ui/`** is the Reflex web application.

Never import Reflex (or any `patch_sim_ui` module) into `patch_sim/`, and never
import `patch_sim_ui` from `patch_sim/`. The dependency direction is one-way:
the UI depends on the core, not the reverse.

## Repository layout

Core library (`patch_sim/`):

- `neuron.py` — the immutable `Neuron` model.
- `clamp_simulations.py` — the current/voltage-clamp runners and `SimulationResult`.
- `channels/` — the ion-channel library (HH52, Pospischil, Mainen-Sejnowski,
  Purkinje, thalamic, TRN, SNc, STN, and auxiliary channels) plus the gating
  abstractions.
- `protocols/` — stimulus builders for current and voltage clamp.
- `analysis/` — post-hoc metric extraction (AP metrics, F-I, I-V, G-V,
  inactivation, bursts, calcium transients, sag, impedance, SFA, tau-V, passive
  properties).
- `presets/` — the nine cell-type factories, their protocol adjustments, and the
  protocol-preset dispatcher.
- `electrochemistry.py`, `equilibrium.py`, `calcium.py` — Nernst/Goldman
  potentials, equilibrium solvers, and calcium dynamics.

UI (`patch_sim_ui/`):

- `state/` — application state, split by concern (neuron, protocol, simulation,
  analysis, visibility, log).
- `components/` — Reflex components, each a function returning `rx.Component`.
- `patch_sim_ui.py` — the app entry point and page layout.

## Conventions

- **Spelling** — use US English everywhere (`depolarize`, `analyze`, `color`,
  `fiber`), in code, comments, docstrings, and docs.
- **Docstrings** — every function and method (public, private, and dunder) has a
  Google-style docstring. Omit type annotations from the `Args:` section; they
  are already in the signature.
- **Biological accuracy** — prioritize correct physiology when designing or
  modifying presets. Fix the underlying biology rather than working around it;
  document known limitations explicitly where a full fix is out of scope.
- **Comments describe the current design** — explain what the code does and why
  it is correct, in the present tense. Comments are not a changelog; investigation
  history belongs in the PR body and issue thread.

## Tooling

The project uses [uv](https://docs.astral.sh/uv/) for everything; do not call
`pip` directly. [Ruff](https://docs.astral.sh/ruff/) handles linting and
formatting (Google docstring convention), and [ty](https://github.com/astral-sh/ty)
handles type checking.

Before finishing any change, all of these must pass:

```
uv run --frozen -m pytest --verbose      # tests
uvx ruff check .                          # lint
uvx ruff format --check .                 # format
uvx ty check                              # types
```

## Testing

Tests live under `tests/` in four buckets:

- `tests/unit/` — fast, pure-function tests; no simulation runs.
- `tests/integration/` — end-to-end protocol-to-simulation pipeline tests.
- `tests/ui/` — Reflex and plotting tests (skipped when Reflex is absent).
- `tests/e2e/` — headless full-pipeline tests that drive UI state handlers
  directly (not in the default test paths; run explicitly).

The shared `hh_model` fixture is in `tests/conftest.py`. Use plain test functions
(not classes), `pytest.approx` for float comparisons, and a Google-style
docstring on every test.

## The documentation system

Documentation has a single source of truth in `docs/`:

- The four prose pages (`index.md`, `presets.md`, `protocols-and-analysis.md`,
  `developer-guide.md`) are plain Markdown, rendered both by this
  [mkdocs](https://www.mkdocs.org/) site and by the in-app `/help` page.
- The `docs/api/` pages contain
  [mkdocstrings](https://mkdocstrings.github.io/) directives that pull the API
  reference straight from the Google-style docstrings. They are part of the
  mkdocs site only.

Build and preview the site locally:

```
uv run --frozen --group=docs mkdocs serve          # live preview at :8000
uv run --frozen --group=docs mkdocs build --strict  # strict build (CI gate)
```

The strict build fails on broken cross-references, so run it after changing docs
or public docstrings. Keep prose to Markdown constructs that render the same in
both the UI and mkdocs (fenced code blocks and tables are safe). Avoid bare `$`
in prose: the UI's Markdown renderer interprets dollar signs as math.

## Extending the library

When you add a public capability, update code, docstrings, and docs together:

- **A new preset** — add the factory under `patch_sim/presets/`, register it in
  the presets package, add a subsection to `docs/presets.md`, and add a
  `:::`-directive entry under `docs/api/presets.md` if it introduces a new
  top-level export.
- **A new protocol** — add the builder under `patch_sim/protocols/`, wire it into
  the relevant `CURRENT_PROTOCOLS` / `VOLTAGE_PROTOCOLS` list, document it in
  `docs/protocols-and-analysis.md`, and reference it from `docs/api/protocols.md`.
- **A new analysis** — add the module under `patch_sim/analysis/`, export the
  entry point and result type, document it in `docs/protocols-and-analysis.md`,
  and add it to `docs/api/analysis.md`.

In all cases, finish by running the strict docs build alongside the test, lint,
format, and type checks.
