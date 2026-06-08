"""Tests for server lifespan."""

import builtins

import pytest

from zotero_mcp._app import DISABLED_MCP_TOOLS, mcp, server_lifespan, should_register_tool
from zotero_mcp.tools import search as search_tools
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


@pytest.mark.parametrize(
    "tool_name",
    [
        "search",
        "fetch",
        "zotero_update_search_database",
        "zotero_get_search_database_status",
    ],
)
def test_semantic_database_tools_are_not_registered(tool_name):
    """Database maintenance tools should not be advertised by the MCP server."""
    assert should_register_tool(tool_name) is False


@pytest.mark.asyncio
async def test_disabled_semantic_database_tools_absent_from_registry():
    """Importing all tool modules should not register disabled database tools."""
    import zotero_mcp.tools  # noqa: F401

    tool_names = {tool.name for tool in await mcp.list_tools()}
    assert tool_names.isdisjoint(DISABLED_MCP_TOOLS)


def test_semantic_search_tool_uses_worker_without_importing_semantic_stack(monkeypatch, tmp_path):
    """The MCP semantic search tool must not import semantic_search in-process."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "zotero_mcp.semantic_search" or name.startswith("zotero_mcp.semantic_search."):
            raise AssertionError("semantic search must stay inside the worker process")
        return original_import(name, *args, **kwargs)

    class WorkerDouble:
        def search(self, **kwargs):
            return {"results": []}

    class ContextDouble:
        def info(self, message):
            pass

        def error(self, message):
            pass

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(search_tools, "get_semantic_worker_client", lambda: WorkerDouble())
    monkeypatch.setattr(search_tools.Path, "home", lambda: tmp_path)

    result = search_tools.semantic_search(query="test query", ctx=ContextDouble())
    assert result == "No semantically similar items found for query: 'test query'"


@pytest.mark.parametrize(
    ("func_name", "expected_command"),
    [
        ("update_search_database", "zotero-mcp update-db"),
        ("get_search_database_status", "zotero-mcp db-status"),
    ],
)
def test_removed_database_wrappers_are_inert(monkeypatch, func_name, expected_command):
    """Direct calls to removed MCP DB wrappers must not import semantic_search."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "zotero_mcp.semantic_search" or name.startswith("zotero_mcp.semantic_search."):
            raise AssertionError("removed database wrapper imported semantic search")
        return original_import(name, *args, **kwargs)

    class ContextDouble:
        def info(self, message):
            pass

        def error(self, message):
            pass

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = getattr(search_tools, func_name)(ctx=ContextDouble())
    assert "intentionally removed" in result
    assert expected_command in result
