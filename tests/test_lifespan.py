"""Tests for server lifespan."""

import builtins

import pytest

from zotero_mcp._app import server_lifespan
from zotero_mcp.tools.search import _maybe_fire_presearch_sync


@pytest.mark.asyncio
async def test_lifespan_yields_without_semantic_startup_work(monkeypatch):
    """Server startup must not import or initialize semantic search."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "zotero_mcp.semantic_search" or name.startswith("zotero_mcp.semantic_search."):
            raise AssertionError("semantic search must not be imported during server startup")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    async with server_lifespan(None) as ctx:
        assert ctx == {}


def test_presearch_sync_is_disabled():
    """Semantic search must not kick off a background update before querying."""

    class SearchDouble:
        def should_update_database(self):
            raise AssertionError("pre-search sync should not inspect update config")

        def update_database(self, **kwargs):
            raise AssertionError("pre-search sync should not update the database")

    assert _maybe_fire_presearch_sync(SearchDouble()) is None
