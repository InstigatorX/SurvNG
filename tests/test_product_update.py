from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from survng.app.operations_routes import (
    OperationsRouteDependencies,
    create_operations_router,
)
from survng.app.product_update import (
    ProductUpdateService,
    _format_git_failure,
    _git_command,
    resolve_repo_root,
)

def _endpoint(router, path: str, method: str):
    return next(
        route.endpoint
        for route in router.routes
        if route.path == path and method in route.methods
    )


def _router(*, request_server_restart: Mock | None = None, product_update: Mock | None = None):
    return create_operations_router(OperationsRouteDependencies(
        get_manager=Mock(),
        manager_lock=threading.RLock(),
        log_rows=lambda: (),
        storage_maintenance=Mock(),
        request_server_restart=request_server_restart or Mock(),
        product_update=product_update or Mock(),
    ))


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _init_survng_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "SurvNG"
    repo.mkdir()
    (repo / "survng").mkdir()
    (repo / "survng" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "requirements.txt").write_text("# test\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "survng", "requirements.txt")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "branch", "-M", "main")
    return repo


def test_restart_server_returns_scheduled_restart() -> None:
    request_restart = Mock(return_value={
        "ok": True,
        "status": "restart_scheduled",
        "instance_id": "old-instance",
    })

    result = _endpoint(
        _router(request_server_restart=request_restart),
        "/api/system/restart",
        "POST",
    )()

    assert result["status"] == "restart_scheduled"
    request_restart.assert_called_once_with()


def test_restart_server_reports_restart_safety_conflict() -> None:
    request_restart = Mock(side_effect=RuntimeError("storage repair is active"))

    with pytest.raises(HTTPException) as raised:
        _endpoint(
            _router(request_server_restart=request_restart),
            "/api/system/restart",
            "POST",
        )()

    assert raised.value.status_code == 409
    assert raised.value.detail == "storage repair is active"


def test_product_update_status_route_returns_service_payload() -> None:
    product_update = Mock()
    product_update.status.return_value = {
        "deployment_mode": "native_git",
        "can_update": False,
        "behind_count": 0,
        "message": "SurvNG is up to date with the remote branch.",
    }

    result = _endpoint(
        _router(product_update=product_update),
        "/api/system/update",
        "GET",
    )(refresh_remote=True)

    assert result["deployment_mode"] == "native_git"
    product_update.status.assert_called_once_with(refresh_remote=True, branch=None)


def test_product_update_start_route_reports_conflict() -> None:
    product_update = Mock()
    product_update.start.side_effect = RuntimeError("No product update is available")

    with pytest.raises(HTTPException) as raised:
        _endpoint(
            _router(product_update=product_update),
            "/api/system/update",
            "POST",
        )()

    assert raised.value.status_code == 409
    assert raised.value.detail == "No product update is available"


def test_product_update_start_route_passes_branch() -> None:
    product_update = Mock()
    product_update.start.return_value = {"status": "running", "target_branch": "develop"}
    from survng.app.operations_routes import ProductUpdateRequest

    result = _endpoint(
        _router(product_update=product_update),
        "/api/system/update",
        "POST",
    )(ProductUpdateRequest(branch="develop"))

    assert result["status"] == "running"
    product_update.start.assert_called_once_with(branch="develop")


def test_resolve_repo_root_finds_survng_checkout(tmp_path: Path) -> None:
    repo = _init_survng_repo(tmp_path)
    assert resolve_repo_root(repo / "survng") == repo.resolve()


def test_git_commands_allow_survng_checkout_safe_directory(tmp_path: Path) -> None:
    repo = _init_survng_repo(tmp_path)
    command = _git_command(repo, ["rev-parse", "--abbrev-ref", "HEAD"])
    assert command[1:3] == ["-c", f"safe.directory={repo}"]
    assert command[-3:] == ["rev-parse", "--abbrev-ref", "HEAD"]


def test_format_git_failure_prefers_stderr_and_ownership_guidance() -> None:
    error = subprocess.CalledProcessError(
        128,
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        stderr="fatal: detected dubious ownership in repository at '/root/SurvNG'\n",
    )
    assert "ownership" in _format_git_failure(error).lower()

    missing = FileNotFoundError("git")
    assert "not installed" in _format_git_failure(missing).lower()


def test_product_update_status_reports_behind_commits(tmp_path: Path) -> None:
    repo = _init_survng_repo(tmp_path)
    bare = tmp_path / "remote.git"
    _git(tmp_path, "clone", "--bare", str(repo), str(bare))
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "fetch", "origin")

    work = tmp_path / "publisher"
    _git(tmp_path, "clone", str(bare), str(work))
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "user.name", "Test")
    (work / "survng" / "marker.txt").write_text("next\n", encoding="utf-8")
    _git(work, "add", "survng/marker.txt")
    _git(work, "commit", "-m", "next change")
    _git(work, "push", "origin", "main")

    service = ProductUpdateService(repo_root=repo)
    status = service.status(refresh_remote=True)

    assert status["deployment_mode"] == "native_git"
    assert status["behind_count"] == 1
    assert status["can_update"] is True
    assert status["target_branch"] == "main"
    assert "main" in status["branches"]
    assert status["commits_behind"][0]["subject"] == "next change"


def test_product_update_status_supports_selected_branch(tmp_path: Path) -> None:
    repo = _init_survng_repo(tmp_path)
    bare = tmp_path / "remote.git"
    _git(tmp_path, "clone", "--bare", str(repo), str(bare))
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "fetch", "origin")

    work = tmp_path / "publisher"
    _git(tmp_path, "clone", str(bare), str(work))
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "user.name", "Test")
    _git(work, "checkout", "-b", "develop")
    (work / "survng" / "develop.txt").write_text("develop\n", encoding="utf-8")
    _git(work, "add", "survng/develop.txt")
    _git(work, "commit", "-m", "develop change")
    _git(work, "push", "origin", "develop")

    service = ProductUpdateService(repo_root=repo)
    status = service.status(refresh_remote=True, branch="develop")

    assert status["branch"] == "main"
    assert status["target_branch"] == "develop"
    assert status["needs_checkout"] is True
    assert status["can_update"] is True
    assert "develop" in status["branches"]
    assert status["behind_count"] == 1
    assert status["commits_behind"][0]["subject"] == "develop change"
    assert "develop" in status["message"]


def test_product_update_start_checks_out_selected_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_survng_repo(tmp_path)
    bare = tmp_path / "remote.git"
    _git(tmp_path, "clone", "--bare", str(repo), str(bare))
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "fetch", "origin")

    work = tmp_path / "publisher"
    _git(tmp_path, "clone", str(bare), str(work))
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "user.name", "Test")
    _git(work, "checkout", "-b", "develop")
    (work / "survng" / "develop.txt").write_text("develop\n", encoding="utf-8")
    _git(work, "add", "survng/develop.txt")
    _git(work, "commit", "-m", "develop change")
    _git(work, "push", "origin", "develop")

    restart = Mock(return_value={"ok": True, "status": "restart_scheduled", "instance_id": "abc"})
    service = ProductUpdateService(repo_root=repo, request_server_restart=restart)
    real_run = subprocess.run
    real_which = shutil.which

    def fake_run(command, **kwargs):
        cmd = [str(part) for part in command]
        if len(cmd) >= 4 and cmd[1:4] == ["-m", "pip", "install"]:
            return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")
        if Path(cmd[0]).name == "npm":
            return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")
        return real_run(command, **kwargs)

    monkeypatch.setattr("survng.app.product_update.subprocess.run", fake_run)
    monkeypatch.setattr(
        "survng.app.product_update.shutil.which",
        lambda name: None if name == "npm" else real_which(name),
    )
    monkeypatch.setattr("survng.app.product_update.time.sleep", lambda _seconds: None)

    started = service.start(branch="develop")
    assert started["status"] == "running"

    for _ in range(200):
        status = service.status(refresh_remote=False, branch="develop")
        if status["status"] in {"restarting", "complete", "failed"}:
            break
        threading.Event().wait(0.05)
    status = service.status(refresh_remote=False, branch="develop")
    assert status["status"] == "restarting", status
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "develop"
    assert (repo / "survng" / "develop.txt").read_text(encoding="utf-8") == "develop\n"
    restart.assert_called_once_with()


def test_product_update_start_fast_forwards_and_restarts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _init_survng_repo(tmp_path)
    bare = tmp_path / "remote.git"
    _git(tmp_path, "clone", "--bare", str(repo), str(bare))
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "fetch", "origin")

    work = tmp_path / "publisher"
    _git(tmp_path, "clone", str(bare), str(work))
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "user.name", "Test")
    (work / "survng" / "marker.txt").write_text("next\n", encoding="utf-8")
    _git(work, "add", "survng/marker.txt")
    _git(work, "commit", "-m", "next change")
    _git(work, "push", "origin", "main")

    restart = Mock(return_value={"ok": True, "status": "restart_scheduled", "instance_id": "abc"})
    service = ProductUpdateService(repo_root=repo, request_server_restart=restart)
    real_run = subprocess.run
    real_which = shutil.which

    def fake_run(command, **kwargs):
        cmd = [str(part) for part in command]
        if len(cmd) >= 4 and cmd[1:4] == ["-m", "pip", "install"]:
            return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")
        if Path(cmd[0]).name == "npm":
            return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")
        return real_run(command, **kwargs)

    monkeypatch.setattr("survng.app.product_update.subprocess.run", fake_run)
    monkeypatch.setattr(
        "survng.app.product_update.shutil.which",
        lambda name: None if name == "npm" else real_which(name),
    )
    monkeypatch.setattr("survng.app.product_update.time.sleep", lambda _seconds: None)

    started = service.start()
    assert started["status"] == "running"

    for _ in range(200):
        status = service.status(refresh_remote=False)
        if status["status"] in {"restarting", "complete", "failed"}:
            break
        threading.Event().wait(0.05)
    status = service.status(refresh_remote=False)
    assert status["status"] == "restarting", status
    assert (repo / "survng" / "marker.txt").read_text(encoding="utf-8") == "next\n"
    restart.assert_called_once_with()
