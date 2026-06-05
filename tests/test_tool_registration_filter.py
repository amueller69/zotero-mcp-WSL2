"""Tests for hiding tools that cannot work in local-only mode."""

import importlib.util
import sys
import types
from pathlib import Path


class FakeFastMCP:
    def __init__(self, *_args, **_kwargs):
        self.registered_tools = []

    def tool(self, *args, **kwargs):
        def decorator(func):
            self.registered_tools.append(kwargs.get("name") or func.__name__)
            return func

        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return decorator(args[0])
        return decorator


def load_app_module(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "fastmcp",
        types.SimpleNamespace(FastMCP=FakeFastMCP),
    )

    app_path = Path(__file__).parents[1] / "src" / "zotero_mcp" / "_app.py"
    spec = importlib.util.spec_from_file_location("_zotero_mcp_app_test", app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def register_named_tool(app_module, name):
    @app_module.mcp.tool(name=name, description="test")
    def sample_tool():
        return "ok"

    return sample_tool


def test_local_only_hides_incompatible_tools_by_default(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)

    app_module = load_app_module(monkeypatch)
    func = register_named_tool(app_module, "zotero_add_by_doi")

    assert func() == "ok"
    assert app_module.mcp.registered_tools == []


def test_local_only_still_registers_compatible_tools(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)

    app_module = load_app_module(monkeypatch)
    register_named_tool(app_module, "zotero_search_items")

    assert app_module.mcp.registered_tools == ["zotero_search_items"]


def test_hybrid_mode_registers_write_tools(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_API_KEY", "secret")
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "12345")

    app_module = load_app_module(monkeypatch)
    register_named_tool(app_module, "zotero_add_by_doi")

    assert app_module.mcp.registered_tools == ["zotero_add_by_doi"]


def test_filter_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    monkeypatch.setenv("ZOTERO_MCP_HIDE_LOCAL_INCOMPATIBLE_TOOLS", "false")

    app_module = load_app_module(monkeypatch)
    register_named_tool(app_module, "zotero_add_by_doi")

    assert app_module.mcp.registered_tools == ["zotero_add_by_doi"]
