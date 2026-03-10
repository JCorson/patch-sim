# Agent Instructions

## Before finishing any task

Run all three checks and ensure they pass before marking a task complete:

```bash
uv run ci.py test       # run the test suite
uv run ci.py lint       # check formatting and style
uv run ci.py typecheck  # run mypy type checking
```

All three must pass with no errors.
