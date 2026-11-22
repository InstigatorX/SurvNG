"""Async code must not take the manager reload lock on the event loop.

A settings reload holds MANAGER_RELOAD_LOCK while it stops the previous
manager and starts the next one (about 12 s on a production server). An
async route or task that waits for that lock blocks the event loop, so every
request stalls until the reload ends. Take it in a worker thread instead
(asyncio.to_thread).

Checked statically: an async function body (excluding nested functions,
which run wherever they are sent) must not
- enter ``with <manager lock>`` or call ``<manager lock>.acquire()``;
- directly call a sync function in survng/app whose body does either.
Calls passed to to_thread/run_in_threadpool/run_in_executor are fine.
"""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "survng" / "app"
LOCK_NAMES = ("manager_lock", "MANAGER_RELOAD_LOCK")
OFF_LOOP = ("to_thread", "run_in_threadpool", "run_sync", "run_in_executor")
# Startup and shutdown run before requests are served or after they drain.
ALLOWED = {("main.py", "lifespan")}


def _own_nodes(function: ast.AST):
    """Nodes of a function body, not descending into nested functions."""
    pending = list(ast.iter_child_nodes(function))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield node
        pending.extend(ast.iter_child_nodes(node))


def _takes_lock(node: ast.AST) -> bool:
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return any(
            any(name in ast.unparse(item.context_expr) for name in LOCK_NAMES)
            for item in node.items
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr == "acquire" and any(
            name in ast.unparse(node.func.value) for name in LOCK_NAMES
        )
    return False


def _trees(app: Path) -> dict[Path, ast.Module]:
    return {
        path: ast.parse(path.read_text(encoding="utf-8")) for path in sorted(app.rglob("*.py"))
    }


def _locking_sync_functions(trees: dict[Path, ast.Module]) -> set[str]:
    return {
        function.name
        for tree in trees.values()
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef)
        and any(_takes_lock(node) for node in _own_nodes(function))
    }


def _violations(app: Path = APP) -> list[str]:
    trees = _trees(app)
    lockers = _locking_sync_functions(trees)
    found: list[str] = []
    for path, tree in trees.items():
        for function in ast.walk(tree):
            if not isinstance(function, ast.AsyncFunctionDef):
                continue
            if (path.name, function.name) in ALLOWED:
                continue
            for node in _own_nodes(function):
                where = f"{path.relative_to(app)}:{getattr(node, 'lineno', 0)} {function.name}"
                if _takes_lock(node):
                    found.append(f"{where} takes the lock")
                elif isinstance(node, ast.Call):
                    callee = ast.unparse(node.func)
                    if callee.endswith(OFF_LOOP):
                        continue
                    if callee.split(".")[-1] in lockers:
                        found.append(f"{where} calls {callee}, which takes the lock")
    return found


def test_no_async_function_waits_for_the_manager_reload_lock() -> None:
    assert _violations() == []


def test_the_check_sees_direct_and_indirect_lock_use(tmp_path: Path) -> None:
    module = tmp_path / "probe.py"
    module.write_text(
        "import asyncio\n"
        "def helper(deps):\n"
        "    with deps.manager_lock:\n"
        "        return 1\n"
        "async def direct(deps):\n"
        "    deps.manager_lock.acquire()\n"
        "async def indirect(deps):\n"
        "    return helper(deps)\n"
        "async def fine(deps):\n"
        "    return await asyncio.to_thread(helper, deps)\n",
        encoding="utf-8",
    )
    found = _violations(tmp_path)

    assert found == [
        "probe.py:6 direct takes the lock",
        "probe.py:8 indirect calls helper, which takes the lock",
    ]
