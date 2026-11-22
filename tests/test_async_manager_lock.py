"""Async code must not take the manager reload lock on the event loop.

A settings reload holds MANAGER_RELOAD_LOCK while it stops the previous
manager and starts the next one (about 12 s on a production server). An
async route or task that waits for that lock blocks the event loop, so every
request stalls until the reload ends. Take it in a worker thread instead
(asyncio.to_thread).
"""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "survng" / "app"
LOCK_NAMES = ("manager_lock", "MANAGER_RELOAD_LOCK")
# Startup and shutdown run before requests are served or after they drain.
ALLOWED = {("main.py", "lifespan")}


def _locks_taken_in_async_bodies() -> list[str]:
    found: list[str] = []
    for path in sorted(APP.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if not isinstance(function, ast.AsyncFunctionDef):
                continue
            if (path.name, function.name) in ALLOWED:
                continue
            pending = list(function.body)
            while pending:
                node = pending.pop()
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue  # nested sync helpers run in threads
                if isinstance(node, (ast.With, ast.AsyncWith)):
                    for item in node.items:
                        expression = ast.unparse(item.context_expr)
                        if any(name in expression for name in LOCK_NAMES):
                            relative = path.relative_to(APP)
                            found.append(f"{relative}:{node.lineno} {function.name}")
                pending.extend(ast.iter_child_nodes(node))
    return found


def test_no_async_function_waits_for_the_manager_reload_lock() -> None:
    assert _locks_taken_in_async_bodies() == []
