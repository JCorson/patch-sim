# Agent Instructions

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

## Before finishing any task

Run all three checks and ensure they pass before marking a task complete:

```bash
uv run --frozen -m pytest --verbose          # run the test suite
uv tool run ruff check .                     # check style
uv tool run ruff format --check .            # check formatting
uv run --frozen -m mypy .                    # run mypy type checking
```

All three must pass with no errors.
