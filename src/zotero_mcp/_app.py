"""FastMCP application instance and server lifecycle."""

import logging
import os
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

# Configure logging from environment variable
# Set ZOTERO_MCP_LOG_LEVEL=DEBUG in Claude Desktop config to enable debug logs
_log_level = os.environ.get("ZOTERO_MCP_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.WARNING),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)


@asynccontextmanager
async def server_lifespan(server: FastMCP):
    """Manage server startup and shutdown lifecycle."""
    sys.stderr.write("Starting Zotero MCP server...\n")
    yield {}

    sys.stderr.write("Shutting down Zotero MCP server...\n")


# Create an MCP server (fastmcp 2.14+ no longer accepts `dependencies`)
mcp = FastMCP("Zotero", lifespan=server_lifespan)


LOCAL_ONLY_INCOMPATIBLE_TOOLS = frozenset(
    {
        "zotero_add_by_bibtex",
        "zotero_add_by_csl_json",
        "zotero_add_by_doi",
        "zotero_add_by_isbn",
        "zotero_add_by_url",
        "zotero_add_from_file",
        "zotero_add_item_relation",
        "zotero_batch_update_tags",
        "zotero_create_annotation",
        "zotero_create_area_annotation",
        "zotero_create_collection",
        "zotero_create_note",
        "zotero_delete_annotation",
        "zotero_delete_collection",
        "zotero_delete_item",
        "zotero_delete_note",
        "zotero_manage_collections",
        "zotero_merge_duplicates",
        "zotero_remove_item_relation",
        "zotero_update_annotation",
        "zotero_update_item",
        "zotero_update_note",
    }
)

_FALSEY_ENV_VALUES = {"0", "false", "no", "off"}
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY_ENV_VALUES


def _hide_local_incompatible_tools_enabled() -> bool:
    value = os.environ.get("ZOTERO_MCP_HIDE_LOCAL_INCOMPATIBLE_TOOLS", "true")
    return value.strip().lower() not in _FALSEY_ENV_VALUES


def _is_local_only_mode() -> bool:
    if not _env_truthy("ZOTERO_LOCAL"):
        return False
    return not (os.environ.get("ZOTERO_API_KEY") and os.environ.get("ZOTERO_LIBRARY_ID"))


def should_register_tool(tool_name: str | None) -> bool:
    """Return False for tools that cannot work in local-only mode.

    The implementation functions remain importable; this only controls whether
    FastMCP advertises the tool to clients during server startup.
    """
    if not tool_name:
        return True
    if not _hide_local_incompatible_tools_enabled():
        return True
    if not _is_local_only_mode():
        return True
    return tool_name not in LOCAL_ONLY_INCOMPATIBLE_TOOLS


_fastmcp_tool = mcp.tool


def _passthrough_tool_decorator(func: Callable[..., Any] | None = None):
    if func is not None:
        return func

    def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
        return inner

    return decorator


def _filtered_tool(*args: Any, **kwargs: Any):
    tool_name = kwargs.get("name")

    # Support bare ``@mcp.tool`` usage even though the codebase currently uses
    # ``@mcp.tool(...)`` everywhere.
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        func = args[0]
        if not should_register_tool(getattr(func, "__name__", None)):
            return func
        return _fastmcp_tool(func)

    if not should_register_tool(tool_name):
        return _passthrough_tool_decorator()
    return _fastmcp_tool(*args, **kwargs)


mcp.tool = _filtered_tool
