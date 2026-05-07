"""Unit tests for the pure Plotly-JS render helpers in `_figure_js`."""

import os

import pytest

# Reflex is not strictly imported here, but importing patch_sim_ui state
# transitively pulls Reflex; keep the testing-env guard for safety.
os.environ.setdefault("PYTEST_CURRENT_TEST", "test_figure_js.py::setup")
pytest.importorskip("reflex")

from patch_sim_ui.state._figure_js import (  # noqa: E402
    _render_fetch_figure_js,
    _render_sweep_highlight_js,
)


def test_render_sweep_highlight_js_substitutes_selected_sweep() -> None:
    """The selected_sweep value lands in the rendered JS string."""
    js = _render_sweep_highlight_js(2)
    assert "/*SELECTED_SWEEP*/" not in js
    assert "/*DIM_OPACITY*/" not in js
    assert "/*HOVER_WIDTH*/" not in js
    assert "/*DIM_WIDTH*/" not in js
    # The seeded value appears verbatim somewhere in the substituted JS.
    assert "2" in js


def test_render_sweep_highlight_js_default_no_selection() -> None:
    """``-1`` is the documented sentinel for no client-side selection."""
    js = _render_sweep_highlight_js(-1)
    assert "-1" in js
    assert "/*SELECTED_SWEEP*/" not in js


def test_render_fetch_figure_js_substitutes_all_placeholders() -> None:
    """Every placeholder token is replaced; the body carries the inputs."""
    js = _render_fetch_figure_js(
        token="abcdef",
        post_js="POST_JS_BODY",
        api_url="http://example.test",
    )
    for placeholder in (
        "/*API_URL*/",
        "/*TOKEN*/",
        "/*DARK_AXIS_STYLE*/",
        "/*POST_JS*/",
    ):
        assert placeholder not in js
    assert "abcdef" in js
    assert "POST_JS_BODY" in js
    assert "http://example.test" in js
