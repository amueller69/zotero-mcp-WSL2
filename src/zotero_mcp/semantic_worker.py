"""Child-process entry point for semantic search operations.

This module is intentionally allowed to import the semantic stack. It runs in a
disposable subprocess so Chroma, sentence-transformers, PyTorch, and ROCm/HSA
state do not become resident in the long-lived MCP server process.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

_search_cache: dict[tuple[str | None, str | None], Any] = {}


def _get_search(config_path: str | None, db_path: str | None = None) -> Any:
    cache_key = (config_path, db_path)
    if cache_key not in _search_cache:
        from zotero_mcp.semantic_search import create_semantic_search

        _search_cache[cache_key] = create_semantic_search(config_path, db_path=db_path)
    return _search_cache[cache_key]


def _handle_request(request: dict[str, Any]) -> dict[str, Any]:
    method = request.get("method")
    params = request.get("params") or {}

    if method == "search":
        search = _get_search(
            params.get("config_path"),
            params.get("db_path"),
        )
        return {
            "ok": True,
            "result": search.search(
                query=params["query"],
                limit=params.get("limit", 10),
                filters=params.get("filters"),
            ),
        }

    if method == "shutdown":
        return {"ok": True, "result": {"shutdown": True}}

    return {
        "ok": False,
        "error": f"Unknown semantic worker method: {method}",
        "error_type": "ValueError",
    }


def _response_for(request: dict[str, Any]) -> dict[str, Any]:
    request_id = request.get("id")
    try:
        response = _handle_request(request)
    except Exception as exc:
        response = {
            "ok": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
    response["id"] = request_id
    return response


def main() -> int:
    # Keep stdout reserved for protocol messages. Any accidental prints from
    # imported libraries go to stderr instead of corrupting JSON responses.
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        should_shutdown = False
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {
                "id": None,
                "ok": False,
                "error": f"Invalid JSON request: {exc}",
                "error_type": "JSONDecodeError",
            }
        else:
            response = _response_for(request)
            should_shutdown = request.get("method") == "shutdown"

        protocol_out.write(json.dumps(response, default=str, separators=(",", ":")) + "\n")
        protocol_out.flush()

        if should_shutdown:
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
