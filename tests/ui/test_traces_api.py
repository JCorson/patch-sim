"""Tests for the side-channel figure store (patch_sim_ui/api/traces.py)."""

import json

import pytest

pytest.importorskip("plotly")
pytest.importorskip("starlette")

import plotly.graph_objects as go  # noqa: E402


def _make_fig(n_traces: int = 2, n_pts: int = 10) -> go.Figure:
    """Return a small test figure with *n_traces* Scattergl traces.

    Args:
        n_traces: Number of Scattergl traces to add.
        n_pts: Number of data points per trace.

    Returns:
        A ``go.Figure`` suitable for round-trip serialisation tests.
    """
    import numpy as np

    fig = go.Figure()
    for i in range(n_traces):
        x = list(np.linspace(0, 1, n_pts))
        y = list(np.sin(np.array(x) + i))
        fig.add_trace(go.Scattergl(x=x, y=y, name=f"trace-{i}"))
    return fig


class TestTracesStore:
    """Tests for put / get_bytes and the FIFO eviction policy."""

    def setup_method(self) -> None:
        """Clear the store before each test."""
        from patch_sim_ui.api.traces import _STORE

        _STORE.clear()

    def test_put_and_get_round_trip(self) -> None:
        """put then get_bytes returns valid JSON-encoded figure dict."""
        from patch_sim_ui.api.traces import get_bytes, put

        fig = _make_fig()
        put("tok1", fig)
        raw = get_bytes("tok1")
        decoded = json.loads(raw)
        assert "data" in decoded
        assert len(decoded["data"]) == 2

    def test_trace_values_preserved(self) -> None:
        """Float values in the stored figure match the original sweep arrays."""
        from patch_sim_ui.api.traces import get_bytes, put

        fig = _make_fig(n_traces=1, n_pts=5)
        original_x = list(fig.data[0].x)
        put("tok-vals", fig)
        decoded = json.loads(get_bytes("tok-vals"))
        stored_x = decoded["data"][0]["x"]
        assert stored_x == pytest.approx(original_x)

    def test_missing_token_raises(self) -> None:
        """get_bytes raises KeyError for an unknown token."""
        from patch_sim_ui.api.traces import get_bytes

        with pytest.raises(KeyError):
            get_bytes("no-such-token")

    def test_fifo_eviction(self) -> None:
        """Inserting more than _MAX_TOKENS entries evicts the oldest."""
        from patch_sim_ui.api import traces
        from patch_sim_ui.api.traces import put

        max_t = traces._MAX_TOKENS
        fig = _make_fig()
        tokens = [f"t{i}" for i in range(max_t + 1)]
        for tok in tokens:
            put(tok, fig)
        assert len(traces._STORE) == max_t
        assert tokens[0] not in traces._STORE
        for tok in tokens[1:]:
            assert tok in traces._STORE


class TestTracesRoute:
    """Tests for the Starlette HTTP route."""

    def setup_method(self) -> None:
        """Clear the store before each test."""
        from patch_sim_ui.api.traces import _STORE

        _STORE.clear()

    def test_route_200_for_known_token(self) -> None:
        """GET /api/figure/{token} returns 200 with JSON body for a known token."""
        from starlette.testclient import TestClient

        from patch_sim_ui.api.traces import put, starlette_app

        put("tok-http", _make_fig())
        client = TestClient(starlette_app)
        resp = client.get("/api/figure/tok-http")
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body

    def test_route_404_for_unknown_token(self) -> None:
        """GET /api/figure/{token} returns 404 when the token is absent."""
        from starlette.testclient import TestClient

        from patch_sim_ui.api.traces import starlette_app

        client = TestClient(starlette_app)
        resp = client.get("/api/figure/nonexistent")
        assert resp.status_code == 404
