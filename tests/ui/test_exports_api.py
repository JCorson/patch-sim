"""Tests for the one-shot export store (patch_sim_ui/api/exports.py)."""

import pytest

pytest.importorskip("starlette")

from starlette.testclient import TestClient  # noqa: E402

from patch_sim_ui.api import exports  # noqa: E402
from patch_sim_ui.api.exports import put, starlette_app  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_store():
    """Reset the export store to empty before each test."""
    exports._STORE.clear()


def test_route_serves_csv_attachment() -> None:
    """GET returns 200 with the payload, CSV media type and attachment header."""
    put("tok", "patch-sim_x.csv", b"a,b\n1,2\n")
    resp = TestClient(starlette_app).get("/api/export/tok")
    assert resp.status_code == 200
    assert resp.content == b"a,b\n1,2\n"
    assert resp.headers["content-type"] == "text/csv; charset=utf-8"
    assert (
        resp.headers["content-disposition"] == 'attachment; filename="patch-sim_x.csv"'
    )


def test_token_is_single_use() -> None:
    """A second GET for the same token returns 404."""
    put("tok", "f.csv", b"x")
    client = TestClient(starlette_app)
    assert client.get("/api/export/tok").status_code == 200
    assert client.get("/api/export/tok").status_code == 404


def test_unknown_token_404() -> None:
    """GET for an unknown token returns 404."""
    assert TestClient(starlette_app).get("/api/export/nope").status_code == 404


def test_fifo_eviction() -> None:
    """Inserting more than _MAX_TOKENS entries evicts the oldest."""
    tokens = [f"t{i}" for i in range(exports._MAX_TOKENS + 1)]
    for tok in tokens:
        put(tok, "f.csv", b"x")
    assert len(exports._STORE) == exports._MAX_TOKENS
    assert tokens[0] not in exports._STORE
    assert tokens[-1] in exports._STORE
