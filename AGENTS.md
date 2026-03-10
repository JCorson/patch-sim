# Agent Instructions

## Before finishing any task

Run all three checks and ensure they pass before marking a task complete:

```bash
uv run --frozen -m pytest --verbose          # run the test suite
uv tool run ruff check .                     # check style
uv tool run ruff format --check .            # check formatting
uv run --frozen -m mypy .                    # run mypy type checking
```

All three must pass with no errors.
