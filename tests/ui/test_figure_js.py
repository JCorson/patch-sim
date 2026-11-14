"""Behavior tests for the Plotly-JS render helpers in ``_figure_js``."""

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


def test_no_stale_placeholder_tokens_remain_after_render() -> None:
    """Both helpers substitute every placeholder they advertise.

    A stale ``/*FOO*/`` token in the rendered output would be a JavaScript
    syntax error in the browser (an unterminated comment / orphan
    expression), so this is the load-bearing behavior contract: the
    renderers must replace every placeholder, not just the ones the caller
    happens to read back.
    """
    sweep_js = _render_sweep_highlight_js(selected_sweep=2)
    for placeholder in (
        "/*SELECTED_SWEEP*/",
        "/*DIM_OPACITY*/",
        "/*HOVER_WIDTH*/",
        "/*DIM_WIDTH*/",
    ):
        assert placeholder not in sweep_js

    fetch_js = _render_fetch_figure_js(
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
        assert placeholder not in fetch_js
    # Caller-supplied values must reach the rendered body — otherwise the
    # browser would fetch the wrong figure or run the wrong post-swap JS.
    assert "abcdef" in fetch_js
    assert "POST_JS_BODY" in fetch_js
    assert "http://example.test" in fetch_js


def test_download_js_targets_export_route() -> None:
    """The download snippet loads /api/export/{token} in a self-removing iframe."""
    from patch_sim_ui.state._figure_js import _render_download_js

    js = _render_download_js("http://example.test", "tok123")
    assert '"http://example.test/api/export/tok123"' in js
    assert "iframe" in js
    assert "remove()" in js
    assert "click()" not in js


def test_export_figure_js_substitutes_all_placeholders() -> None:
    """The export snippet has no stale placeholders and JSON-quotes its strings."""
    from patch_sim_ui.state._figure_js import _render_export_figure_js

    js = _render_export_figure_js("tok9", "svg", 'a"b', "http://example.test")
    for placeholder in ("/*API_URL*/", "/*TOKEN*/", "/*FORMAT*/", "/*FILENAME*/"):
        assert placeholder not in js
    assert 'var format = "svg";' in js
    assert 'var filename = "a\\"b";' in js
    assert "http://example.test" in js
    assert "tok9" in js
