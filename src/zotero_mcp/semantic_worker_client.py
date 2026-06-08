"""Main-process client for the disposable semantic worker."""

from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

_DEFAULT_IDLE_SECONDS = 180.0
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0


class SemanticWorkerError(RuntimeError):
    """Raised when the semantic worker cannot complete a request."""


class SemanticWorkerClient:
    """Manage one reusable semantic worker subprocess."""

    def __init__(
        self,
        *,
        idle_seconds: float | None = None,
        request_timeout_seconds: float | None = None,
    ) -> None:
        self.idle_seconds = (
            idle_seconds
            if idle_seconds is not None
            else _float_env("ZOTERO_MCP_SEMANTIC_WORKER_IDLE_SECONDS", _DEFAULT_IDLE_SECONDS)
        )
        self.request_timeout_seconds = (
            request_timeout_seconds
            if request_timeout_seconds is not None
            else _float_env(
                "ZOTERO_MCP_SEMANTIC_WORKER_TIMEOUT_SECONDS",
                _DEFAULT_REQUEST_TIMEOUT_SECONDS,
            )
        )
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._idle_timer: threading.Timer | None = None

    def search(
        self,
        *,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run semantic search inside the worker process."""
        params = {
            "query": query,
            "limit": limit,
            "filters": filters,
            "config_path": str(config_path) if config_path is not None else None,
        }
        return self._request("search", params)

    def stop(self) -> None:
        """Terminate the worker if it is running."""
        with self._lock:
            self._cancel_idle_timer()
            self._stop_locked()

    def _request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            self._cancel_idle_timer()
            process = self._ensure_process_locked()
            response_queue: queue.Queue[str | None] = queue.Queue(maxsize=1)
            reader = threading.Thread(
                target=self._read_response,
                args=(process, response_queue),
                daemon=True,
            )
            reader.start()

            request = {
                "id": uuid.uuid4().hex,
                "method": method,
                "params": params or {},
            }

            try:
                assert process.stdin is not None
                process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                process.stdin.flush()
                line = response_queue.get(timeout=self.request_timeout_seconds)
            except queue.Empty as exc:
                self._stop_locked()
                raise SemanticWorkerError(
                    f"Semantic worker timed out after {self.request_timeout_seconds:g}s"
                ) from exc
            except Exception as exc:
                self._stop_locked()
                raise SemanticWorkerError(f"Semantic worker request failed: {exc}") from exc

            if line is None:
                return_code = process.poll()
                self._stop_locked()
                raise SemanticWorkerError(
                    "Semantic worker exited before responding"
                    + (f" (exit code {return_code})" if return_code is not None else "")
                )

            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                self._stop_locked()
                raise SemanticWorkerError(f"Semantic worker returned invalid JSON: {line[:500]}") from exc

            if response.get("id") != request["id"]:
                self._stop_locked()
                raise SemanticWorkerError("Semantic worker response id mismatch")

            if not response.get("ok"):
                error = response.get("error") or "Unknown semantic worker error"
                error_type = response.get("error_type") or "Error"
                raise SemanticWorkerError(f"{error_type}: {error}")

            self._schedule_idle_stop_locked()
            result = response.get("result")
            return result if isinstance(result, dict) else {"result": result}

    def _ensure_process_locked(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process

        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        self._process = subprocess.Popen(
            [sys.executable, "-m", "zotero_mcp.semantic_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            env=env,
        )
        return self._process

    @staticmethod
    def _read_response(
        process: subprocess.Popen[str],
        response_queue: queue.Queue[str | None],
    ) -> None:
        if process.stdout is None:
            response_queue.put(None)
            return
        line = process.stdout.readline()
        response_queue.put(line if line else None)

    def _schedule_idle_stop_locked(self) -> None:
        if self.idle_seconds <= 0:
            self._stop_locked()
            return
        self._idle_timer = threading.Timer(self.idle_seconds, self.stop)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _stop_locked(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return

        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            process.kill()
            process.wait(timeout=5)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


_client = SemanticWorkerClient()
atexit.register(_client.stop)


def get_semantic_worker_client() -> SemanticWorkerClient:
    return _client
