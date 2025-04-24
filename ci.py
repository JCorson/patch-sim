"""
This script provides a click-based CLI with the twin goals of:

- simplifying common development tasks for project developers
- providing commands for step-sized tasks for GitHub Actions workflows

The recommended way to run the script is to use `uv run`. For example, to see
the list of supported commands, run:

```
uv run ci.py
```

For detailed usage instructions, see the top-level README.md in the repository.
"""

# PEP 723 script metadata

# /// script
# requires-python = ">=3.11"
# dependencies = ["click"]
# ///

import os
import subprocess
from pathlib import Path
from typing import cast

import click

# --- Configuration

#: Names of Python import packages that will be installed into the dev environment.
PYTHON_PACKAGES = ["ap_sim"]

# --- Paths

#: Path to repository root
ROOT = Path(__file__).parent.resolve()
#: Path to backend server code
BACKEND_SRC_DIR = ROOT / "src"


# --- Click interface


@click.group("cli")
def cli() -> None:
    """Development helper tool for the project."""


@cli.command("install")
def install() -> None:
    """Set up Python environments for development."""
    # Install Python dependencies and packages.
    uv("sync", "--frozen", "--extra=dev")


@cli.command("update-dependencies")
def update_dependencies() -> None:
    """Regenerate the uv dependency lock file for the project."""
    uv("lock", "--upgrade")


@cli.command("format")
def format() -> None:
    """Apply code formatting."""

    click.echo("\nApplying ruff format formatting...")
    uv("tool", "run", "ruff", "format", ROOT)

    click.echo("\nApplying ruff check formatting...")
    uv("tool", "run", "ruff", "check", ROOT, "--fix-only")


@cli.command("lint")
def lint() -> None:
    """Run linting tools against Python source."""
    failed_checks = []

    if uv("tool", "run", "ruff", "check", ROOT, check=False).returncode:
        failed_checks.append("ruff check")
    if uv("tool", "run", "ruff", "format", "--check", ROOT, check=False).returncode:
        failed_checks.append("ruff format")

    if failed_checks:
        report_failure(f"linting failures: {', '.join(sorted(failed_checks))}")
    else:
        report_success("All lint checks passed")


@cli.command("serve")
def run_server() -> None:
    """Run the application."""
    pass


@cli.command("test")
def test() -> None:
    """Run Python unit tests for all packages."""
    failed_checks = []

    for pkg in PYTHON_PACKAGES:
        click.echo(f"Running {pkg} tests...")
        if pythonm("unittest", "discover", "--verbose", pkg, check=False).returncode:
            failed_checks.append(pkg)

    if failed_checks:
        report_failure(
            f"test failures for packages: {', '.join(sorted(failed_checks))}"
        )
    else:
        report_success("All tests passed")


@cli.command("typecheck")
def typecheck() -> None:
    """Run type checking tools against source code."""
    failed_checks = []

    click.echo("Checking Python with mypy...")
    if pythonm("mypy", ROOT, check=False).returncode:
        failed_checks.append("mypy")

    if failed_checks:
        report_failure(f"type check failures: {', '.join(sorted(failed_checks))}")
    else:
        report_success("All type checks passed")


# ----------------------------------------------------------------------------
# --- Helpers


def pythonm(
    *args,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a 'python -m' command in the development environment."""
    # Include --frozen to avoid side-effect updates to the lock file.
    return uv("run", "--frozen", "-m", *args, check=check, env=env)


def report_failure(msg: str):
    """Report a ci command failure.

    Exits the CLI with an error message and a nonzero exit code.
    """
    click.echo("")
    raise click.ClickException(click.style(msg, fg="red"))


def report_success(msg: str):
    """Report a ci command success."""
    click.echo("")
    click.secho(msg, fg="green")


def run_process(
    proc: subprocess.Popen, check: bool = False
) -> subprocess.CompletedProcess:
    """Run a process in the same way as subprocess.run."""
    try:
        stdout, stderr = proc.communicate(None)
    except Exception:
        proc.kill()
        raise

    # Satisfy mypy since `poll` can return None
    retcode = proc.poll() or 0
    if check and retcode:
        cmd_str: str = str(cast(list, proc.args))
        report_failure(f"Command {cmd_str} failed with return code {retcode}.")

    return subprocess.CompletedProcess(proc.args, retcode, stdout, stderr)


def start_uv_process(
    *args,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Create a process in the development environment"""
    environ = os.environ.copy()
    # Remove VIRTUAL_ENV from the environment, else uv gets confused and
    # issues a warning message.
    environ.pop("VIRTUAL_ENV", None)

    return subprocess.Popen(["uv", *args], encoding="utf-8", env=environ | (env or {}))


def uv(
    *args,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a uv command in the development environment."""
    process = start_uv_process(*args, env=env)
    return run_process(process, check=check)


if __name__ == "__main__":
    cli(prog_name="uv run ci.py")
