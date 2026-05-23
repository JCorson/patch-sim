"""Load the shared ``docs/`` prose pages for the in-app ``/help`` route.

The four prose Markdown files under the repository-root ``docs/`` directory are
the single source of truth shared by the mkdocs site and the UI help page.
They are read once at import and rendered by ``rx.markdown`` on the ``/help``
route.

Like the ``assets/`` loader in :mod:`patch_sim_ui.state._figure_js`, this module
assumes the app runs from the repository root (the documented ``uv run reflex
run`` workflow) so that ``docs/`` resolves relative to the package.  Only the
four named prose files are read; the ``docs/api/`` pages contain mkdocstrings
``:::`` directives that are not valid Markdown for the UI renderer and are never
loaded here.
"""

import pathlib
import re

_DOCS_DIR = pathlib.Path(__file__).parents[1] / "docs"

#: URL of the deployed mkdocs API-reference site, linked from the help page.
API_DOCS_URL: str = "https://jcorson.github.io/patch-sim/"

#: Ordered ``(slug, label, filename)`` triples driving the help topic nav.
_TOPICS: list[tuple[str, str, str]] = [
    ("overview", "Overview", "index.md"),
    ("presets", "Neuron presets", "presets.md"),
    ("protocols", "Protocols & analysis", "protocols-and-analysis.md"),
]

#: Ordered ``(slug, label)`` pairs for building the topic nav.
HELP_TOPICS: list[tuple[str, str]] = [(slug, label) for slug, label, _ in _TOPICS]

#: Default topic shown when the help page first loads.
DEFAULT_TOPIC: str = _TOPICS[0][0]

# Markdown links whose target is not an absolute http(s) URL — i.e. the
# mkdocs-internal cross-page and API links.  They have no meaning on the UI
# help page (which switches topics by state, not by URL), so they are reduced
# to their link text to avoid rendering dead links.
_INTERNAL_LINK = re.compile(r"\[([^\]]+)\]\((?!https?://)[^)]*\)")


def _load(filename: str) -> str:
    """Read one prose file and strip its mkdocs-internal links.

    Args:
        filename: Name of the Markdown file under ``docs/``.

    Returns:
        The file's contents with internal Markdown links flattened to plain
        text and external (http/https) links left intact.
    """
    text = (_DOCS_DIR / filename).read_text(encoding="utf-8")
    return _INTERNAL_LINK.sub(r"\1", text)


#: Topic slug -> rendered Markdown body, loaded once at import.
PROSE: dict[str, str] = {slug: _load(filename) for slug, _, filename in _TOPICS}
