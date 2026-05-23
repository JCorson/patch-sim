"""Load the application-documentation pages for the in-app ``/help`` route.

The Markdown files under ``patch_sim_ui/help_content/`` are the user guide for
the web application: how to configure a neuron, set up a protocol, run a
simulation, and read the analysis. They are read once at import and rendered by
``rx.markdown`` on the ``/help`` route.

This is the *application* documentation track. The *library* documentation (the
mkdocs API reference) lives separately under the repository-root ``docs/`` and
is published as a static site; the help page links out to it.

Help content lives inside the package, so it resolves relative to this module
regardless of the working directory. Screenshots referenced by the prose are
served by Reflex from the repository-root ``assets/screenshots/`` directory at
``/screenshots/...``.
"""

import pathlib
import re

_DOCS_DIR = pathlib.Path(__file__).parent / "help_content"

#: URL of the deployed mkdocs API-reference site, linked from the help page.
API_DOCS_URL: str = "https://jcorson.github.io/patch-sim/"

#: Ordered ``(slug, label, filename)`` triples driving the help topic nav.
_TOPICS: list[tuple[str, str, str]] = [
    ("getting-started", "Getting started", "getting-started.md"),
    ("neuron", "Configuring the neuron", "neuron.md"),
    ("protocol", "Setting up a protocol", "protocol.md"),
    ("running", "Running & reading results", "running.md"),
    ("analysis", "Analysis panels", "analysis.md"),
]

#: Ordered ``(slug, label)`` pairs for building the topic nav.
HELP_TOPICS: list[tuple[str, str]] = [(slug, label) for slug, label, _ in _TOPICS]

#: Default topic shown when the help page first loads.
DEFAULT_TOPIC: str = _TOPICS[0][0]

# Markdown links whose target is not an absolute http(s) URL — i.e. the
# cross-topic links between help pages. They have no meaning on the help page
# (which switches topics by state, not by URL), so they are reduced to their
# link text. The ``(?<!\!)`` lookbehind leaves image syntax (``![alt](src)``)
# untouched so screenshots still render.
_INTERNAL_LINK = re.compile(r"(?<!\!)\[([^\]]+)\]\((?!https?://)[^)]*\)")


def _load(filename: str) -> str:
    """Read one help page and flatten its cross-topic links.

    Args:
        filename: Name of the Markdown file under ``help_content/``.

    Returns:
        The file's contents with internal cross-topic links flattened to plain
        text; image references and external (http/https) links are left intact.
    """
    text = (_DOCS_DIR / filename).read_text(encoding="utf-8")
    return _INTERNAL_LINK.sub(r"\1", text)


#: Topic slug -> rendered Markdown body, loaded once at import.
PROSE: dict[str, str] = {slug: _load(filename) for slug, _, filename in _TOPICS}
