"""Pure Plotly-JS asset loaders + token substituters for SimulationState.

The JS modules under ``assets/`` are loaded once at import and have a small
set of placeholder tokens (``/*API_URL*/``, ``/*TOKEN*/``, etc.) that get
substituted per call.  Loading + substitution is pure string work; pulling
it out of :class:`SimulationState` lets the state class focus on
state-coupled JS assembly (which still lives there because it depends on
``self._current_sweeps``, ``self.show_hover`` and friends).
"""

import json
import pathlib

from patch_sim_ui import constants

_ASSETS_DIR = pathlib.Path(__file__).parents[2] / "assets"

#: Client-side sweep highlight / selection module.  Injected via
#: ``rx.call_script()`` after every figure render in multi-sweep mode.
#: Placeholder tokens (``/*DIM_OPACITY*/``, ``/*HOVER_WIDTH*/``,
#: ``/*DIM_WIDTH*/``, ``/*SELECTED_SWEEP*/``) are substituted at call time
#: in :func:`_render_sweep_highlight_js`.
_SWEEP_HIGHLIGHT_JS: str = (_ASSETS_DIR / "sweep_highlight.js").read_text()

#: Client-side fetch-and-swap for side-channel figure delivery.  Placeholder
#: tokens (``/*API_URL*/``, ``/*TOKEN*/``, ``/*DARK_AXIS_STYLE*/``,
#: ``/*POST_JS*/``) are substituted per call in
#: :func:`_render_fetch_figure_js`.
_FETCH_FIGURE_JS: str = (_ASSETS_DIR / "fetch_figure.js").read_text()


def _render_sweep_highlight_js(selected_sweep: int) -> str:
    """Return the sweep-highlight JS with styling constants substituted.

    Args:
        selected_sweep: Index of the sweep to seed as initially selected
            (``-1`` = none).  Substituted into ``/*SELECTED_SWEEP*/``.

    Returns:
        A self-executing JS function string ready for ``rx.call_script``.
    """
    return (
        _SWEEP_HIGHLIGHT_JS.replace(
            "/*DIM_OPACITY*/", str(constants.HIGHLIGHT_DIM_OPACITY)
        )
        .replace("/*HOVER_WIDTH*/", str(constants.HIGHLIGHT_HOVER_WIDTH))
        .replace("/*DIM_WIDTH*/", str(constants.HIGHLIGHT_DIM_WIDTH))
        .replace("/*SELECTED_SWEEP*/", str(selected_sweep))
    )


def _render_fetch_figure_js(token: str, post_js: str, api_url: str) -> str:
    """Return the fetch-and-swap JS with all placeholder tokens substituted.

    Args:
        token: UUID hex token identifying the figure in the side-channel
            store.
        post_js: Inline JS to run after the swap completes (typically a
            visibility re-apply, optionally followed by a hover-mode tweak
            and the sweep-highlight wiring).
        api_url: Base URL for the figure-fetch endpoint, with any trailing
            slash already stripped.

    Returns:
        A self-contained JS string ready for ``rx.call_script``.
    """
    return (
        _FETCH_FIGURE_JS.replace("/*API_URL*/", api_url)
        .replace("/*TOKEN*/", token)
        .replace("/*DARK_AXIS_STYLE*/", json.dumps(constants.DARK_AXIS_STYLE))
        .replace("/*POST_JS*/", post_js)
    )
