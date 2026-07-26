"""Tests for the side-channel figure store (patch_sim_ui/api/traces.py)."""

import json

import numpy as np
import pytest

pytest.importorskip("plotly")
pytest.importorskip("starlette")

import plotly.graph_objects as go  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from patch_sim_ui.api import traces  # noqa: E402
from patch_sim_ui.api.traces import get_bytes, put, starlette_app  # noqa: E402


def _make_fig(n_traces: int = 2, n_pts: int = 10) -> go.Figure:
    """Return a small test figure with *n_traces* Scattergl traces.

    Args:
        n_traces: Number of Scattergl traces to add.
        n_pts: Number of data points per trace.

    Returns:
        A ``go.Figure`` suitable for round-trip serialisation tests.
    """
    fig = go.Figure()
    for i in range(n_traces):
        x = list(np.linspace(0, 1, n_pts))
        y = list(np.sin(np.array(x) + i))
        fig.add_trace(go.Scattergl(x=x, y=y, name=f"trace-{i}"))
    return fig


@pytest.fixture(autouse=True)
def _clear_store():
    """Reset the side-channel figure store to empty before each test."""
    traces._STORE.clear()


# ---------------------------------------------------------------------------
# put / get_bytes and FIFO eviction
# ---------------------------------------------------------------------------


def test_put_and_get_round_trip() -> None:
    """Put then get_bytes returns valid JSON-encoded figure dict."""
    fig = _make_fig()
    put("tok1", fig)
    raw = get_bytes("tok1")
    decoded = json.loads(raw)
    assert "data" in decoded
    assert len(decoded["data"]) == 2


def test_trace_values_preserved() -> None:
    """Float values in the stored figure match the original sweep arrays."""
    fig = _make_fig(n_traces=1, n_pts=5)
    original_x = list(fig.data[0].x)
    put("tok-vals", fig)
    decoded = json.loads(get_bytes("tok-vals"))
    stored_x = decoded["data"][0]["x"]
    assert stored_x == pytest.approx(original_x)


def test_missing_token_raises() -> None:
    """get_bytes raises KeyError for an unknown token."""
    with pytest.raises(KeyError):
        get_bytes("no-such-token")


def test_fifo_eviction() -> None:
    """Inserting more than _MAX_TOKENS entries evicts the oldest."""
    max_t = traces._MAX_TOKENS
    fig = _make_fig()
    tokens = [f"t{i}" for i in range(max_t + 1)]
    for tok in tokens:
        put(tok, fig)
    assert len(traces._STORE) == max_t
    assert tokens[0] not in traces._STORE
    for tok in tokens[1:]:
        assert tok in traces._STORE


# ---------------------------------------------------------------------------
# Starlette HTTP route
# ---------------------------------------------------------------------------


def test_route_200_for_known_token() -> None:
    """GET /api/figure/{token} returns 200 with JSON body for a known token."""
    put("tok-http", _make_fig())
    client = TestClient(starlette_app)
    resp = client.get("/api/figure/tok-http")
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body


def test_route_404_for_unknown_token() -> None:
    """GET /api/figure/{token} returns 404 when the token is absent."""
    client = TestClient(starlette_app)
    resp = client.get("/api/figure/nonexistent")
    assert resp.status_code == 404


def test_testclient_is_backed_by_httpx2() -> None:
    """The Starlette test client binds httpx2 rather than the httpx fallback.

    The ``filterwarnings`` guard in pyproject.toml keys off the text of
    Starlette's deprecation warning, so it stops matching if that wording ever
    changes.  This check keys off the bound module instead, so the two fail
    independently and a reworded warning cannot leave the fallback unnoticed.
    """
    import starlette.testclient

    assert starlette.testclient.httpx.__name__ == "httpx2"
