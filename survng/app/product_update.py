"""Git-backed product update status and apply jobs for native checkouts."""

from __future__ import annotations

import copy
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
DEFAULT_REMOTE = "origin"
DEFAULT_BRANCH = "main"
UPDATE_TIMEOUT_SECONDS = 120.0
FETCH_TIMEOUT_SECONDS = 120.0
PULL_TIMEOUT_SECONDS = 180.0
PIP_TIMEOUT_SECONDS = 900.0
FRONTEND_TIMEOUT_SECONDS = 900.0
HELPER_TIMEOUT_SECONDS = 3600.0
MAX_COMMIT_PREVIEW = 12


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_root(start: Path | None = None) -> Path | None:
    """Return the SurvNG git checkout root, or None when unavailable."""
    candidates: list[Path] = []
    env_root = os.environ.get("SURVNG_REPO_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())
    if start is not None:
        candidates.append(start)
    candidates.append(Path.cwd())
    # survng/app/product_update.py -> repository root
    candidates.append(Path(__file__).resolve().parents[2])

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        for path in (resolved, *resolved.parents):
            if path in seen:
                continue
            seen.add(path)
            if _looks_like_survng_checkout(path) and _git_is_work_tree(path):
                return path
    return None


def _looks_like_survng_checkout(path: Path) -> bool:
    return (path / ".git").exists() and (path / "survng").is_dir()


def _git_executable() -> str | None:
    return shutil.which("git")


def _git_command(repo_root: Path, args: Sequence[str]) -> list[str]:
    git = _git_executable()
    if git is None:
        raise FileNotFoundError("git")
    # SurvNG owns this checkout path; allow the service user to inspect it even
    # when directory ownership does not match the process user.
    return [
        git,
        "-c",
        f"safe.directory={repo_root}",
        *args,
    ]


def _git_is_work_tree(repo_root: Path) -> bool:
    try:
        completed = subprocess.run(
            _git_command(repo_root, ["rev-parse", "--is-inside-work-tree"]),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def _format_git_failure(error: BaseException) -> str:
    if isinstance(error, FileNotFoundError):
        return "Git is not installed or not on PATH for the SurvNG service."
    if isinstance(error, subprocess.TimeoutExpired):
        return "Git timed out while inspecting the SurvNG checkout."
    if isinstance(error, subprocess.CalledProcessError):
        detail = (error.stderr or error.stdout or "").strip() or str(error)
        lines = [line.strip() for line in detail.splitlines() if line.strip()]
        detail = lines[-1] if lines else str(error)
        if detail.lower().startswith("fatal: "):
            detail = detail[7:]
        lowered = detail.lower()
        if "dubious ownership" in lowered:
            return (
                "Git blocked the SurvNG checkout due to directory ownership. "
                "Fix checkout ownership for the service user, or set SURVNG_REPO_ROOT."
            )
        if "not a git repository" in lowered:
            return (
                "SurvNG could not open a valid Git checkout. Set SURVNG_REPO_ROOT "
                "to the repository root that contains .git and survng/."
            )
        return detail
    return str(error)


def _run_git(
    repo_root: Path,
    args: Sequence[str],
    *,
    timeout: float,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _git_command(repo_root, args),
        cwd=repo_root,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _git_output(repo_root: Path, args: Sequence[str], *, timeout: float = 30.0) -> str:
    return _run_git(repo_root, args, timeout=timeout).stdout.strip()


def _baked_version() -> dict[str, str]:
    sha = os.environ.get("SURVNG_GIT_SHA", "").strip()
    if not sha:
        marker = Path("/app/SURVNG_GIT_SHA")
        if marker.is_file():
            try:
                sha = marker.read_text(encoding="utf-8").strip()
            except OSError:
                sha = ""
    short = sha[:12] if sha else ""
    return {"sha": sha, "short_sha": short}


def _deployment_mode(repo_root: Path | None) -> str:
    if os.environ.get("SURVNG_CONFIG_PATH", "").startswith("/config"):
        return "docker"
    if Path("/.dockerenv").exists():
        return "docker"
    if repo_root is not None:
        return "native_git"
    return "unavailable"


def _parse_commits(log_text: str) -> list[dict[str, str]]:
    commits: list[dict[str, str]] = []
    for line in log_text.splitlines():
        sha, separator, subject = line.partition(" ")
        if not separator:
            continue
        commits.append({"sha": sha, "subject": subject.strip()})
    return commits


class ProductUpdateService:
    """Inspect and apply fast-forward product updates from the configured remote."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        request_server_restart: Callable[[], dict[str, Any]] | None = None,
        active_storage_tasks: Callable[[], Sequence[str]] | None = None,
    ) -> None:
        self._configured_root = repo_root
        self._request_server_restart = request_server_restart
        self._active_storage_tasks = active_storage_tasks or (lambda: ())
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {"status": "idle"}
        self._thread: threading.Thread | None = None

    def status(self, *, refresh_remote: bool = False) -> dict[str, Any]:
        """Return local version identity and optional remote-ahead summary."""
        with self._lock:
            job = copy.deepcopy(self._state)
        repo_root = self._configured_root or resolve_repo_root()
        mode = _deployment_mode(repo_root)
        baked = _baked_version()
        helper = self._update_helper_path()

        payload: dict[str, Any] = {
            "deployment_mode": mode,
            "status": job.get("status") or "idle",
            "job": job if job.get("status") not in {None, "idle"} else None,
            "repo_root": str(repo_root) if repo_root else None,
            "remote": DEFAULT_REMOTE,
            "branch": None,
            "current_sha": baked.get("sha") or None,
            "current_short_sha": baked.get("short_sha") or None,
            "upstream_sha": None,
            "upstream_short_sha": None,
            "behind_count": 0,
            "ahead_count": 0,
            "commits_behind": [],
            "dirty": False,
            "can_check": False,
            "can_update": False,
            "update_method": None,
            "message": "",
        }

        if helper is not None:
            payload["update_method"] = "host_helper"
            payload["can_check"] = repo_root is not None
            payload["can_update"] = True
            payload["message"] = (
                "Updates run through the configured host helper script."
            )

        if repo_root is None:
            if _git_executable() is None:
                payload["message"] = (
                    "Git is not installed or not on PATH for the SurvNG service."
                )
            elif mode == "docker":
                payload["message"] = (
                    "This Docker image is immutable. Run "
                    "scripts/update-from-git.sh on the host checkout, or set "
                    "SURVNG_UPDATE_HELPER to a host-mounted updater."
                )
            else:
                payload["message"] = (
                    "SurvNG is not running from a readable Git checkout. "
                    "Set SURVNG_REPO_ROOT to the repository root, or use "
                    "scripts/update-from-git.sh on the host."
                )
            return payload

        try:
            branch = _git_output(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
            current_sha = _git_output(repo_root, ["rev-parse", "HEAD"])
            dirty = bool(
                _run_git(
                    repo_root,
                    ["status", "--porcelain", "--untracked-files=no"],
                    timeout=30.0,
                    check=False,
                ).stdout.strip()
            )
            payload.update(
                {
                    "branch": branch,
                    "current_sha": current_sha,
                    "current_short_sha": current_sha[:12],
                    "dirty": dirty,
                    "can_check": True,
                }
            )
            if refresh_remote:
                self._fetch(repo_root)
            upstream = f"{DEFAULT_REMOTE}/{branch if branch != 'HEAD' else DEFAULT_BRANCH}"
            if branch == "HEAD":
                upstream = f"{DEFAULT_REMOTE}/{DEFAULT_BRANCH}"
            upstream_exists = (
                _run_git(
                    repo_root,
                    ["rev-parse", "--verify", upstream],
                    timeout=30.0,
                    check=False,
                ).returncode
                == 0
            )
            if not upstream_exists:
                payload["message"] = (
                    f"Remote branch {upstream} was not found. Fetch updates first."
                )
                payload["can_update"] = False
                return payload

            upstream_sha = _git_output(repo_root, ["rev-parse", upstream])
            counts = _git_output(
                repo_root,
                ["rev-list", "--left-right", "--count", f"HEAD...{upstream}"],
            )
            ahead_text, _, behind_text = counts.partition("\t")
            if not behind_text:
                ahead_text, _, behind_text = counts.partition(" ")
            ahead_count = int(ahead_text or "0")
            behind_count = int(behind_text or "0")
            commit_log = _git_output(
                repo_root,
                [
                    "log",
                    "--pretty=format:%h %s",
                    f"--max-count={MAX_COMMIT_PREVIEW}",
                    f"HEAD..{upstream}",
                ],
            )
            payload.update(
                {
                    "upstream_sha": upstream_sha,
                    "upstream_short_sha": upstream_sha[:12],
                    "ahead_count": ahead_count,
                    "behind_count": behind_count,
                    "commits_behind": _parse_commits(commit_log),
                }
            )
            if payload["update_method"] is None and mode == "native_git":
                payload["update_method"] = "native_git"
            can_update = (
                behind_count > 0
                and ahead_count == 0
                and not dirty
                and payload["status"] not in {"running", "restarting"}
                and (
                    payload["update_method"] in {"native_git", "host_helper"}
                )
            )
            payload["can_update"] = can_update
            if dirty:
                payload["message"] = (
                    "Tracked local changes are present. Commit or stash them "
                    "before updating."
                )
            elif ahead_count > 0:
                payload["message"] = (
                    "This checkout has local commits that are not on the "
                    "remote. Fast-forward updates are blocked."
                )
            elif behind_count == 0:
                payload["message"] = "SurvNG is up to date with the remote branch."
            elif payload["update_method"] in {"native_git", "host_helper"}:
                payload["message"] = (
                    f"{behind_count} commit"
                    f"{'' if behind_count == 1 else 's'} available from {upstream}."
                )
            else:
                payload["message"] = (
                    f"{behind_count} commit"
                    f"{'' if behind_count == 1 else 's'} available from {upstream}. "
                    "Run scripts/update-from-git.sh on the host checkout, or set "
                    "SURVNG_UPDATE_HELPER to a host-mounted updater."
                )
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            LOGGER.warning("product update status failed: %s", error)
            payload["message"] = (
                f"Unable to inspect Git state: {_format_git_failure(error)}"
            )
            payload["can_update"] = False
        return payload

    def start(self) -> dict[str, Any]:
        """Start one update job. Returns the live job status payload."""
        with self._lock:
            if self._state.get("status") in {"running", "restarting"}:
                raise RuntimeError("a product update is already running")
            active = list(self._active_storage_tasks())
            if active:
                raise RuntimeError(
                    "SurvNG cannot update while storage work is active: "
                    f"{', '.join(active)}"
                )
            snapshot = self.status(refresh_remote=True)
            if not snapshot.get("can_update"):
                raise RuntimeError(
                    str(snapshot.get("message") or "No product update is available")
                )
            self._state = {
                "status": "running",
                "started_at": _utc_now(),
                "phase": "Starting",
                "current_sha": snapshot.get("current_sha"),
                "target_sha": snapshot.get("upstream_sha"),
                "behind_count": snapshot.get("behind_count"),
                "update_method": snapshot.get("update_method"),
                "log": [],
            }
            thread = threading.Thread(
                target=self._run,
                name="product-update",
                daemon=True,
            )
            self._thread = thread
            try:
                thread.start()
            except BaseException:
                self._thread = None
                self._state = {"status": "idle"}
                raise
        return self.status()

    def _update_helper_path(self) -> Path | None:
        raw = os.environ.get("SURVNG_UPDATE_HELPER", "").strip()
        if not raw:
            return None
        path = Path(raw).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path
        return None

    def _fetch(self, repo_root: Path) -> None:
        _run_git(
            repo_root,
            ["fetch", "--prune", DEFAULT_REMOTE],
            timeout=FETCH_TIMEOUT_SECONDS,
        )

    def _append_log(self, line: str) -> None:
        with self._lock:
            if self._state.get("status") not in {"running", "restarting"}:
                return
            log = list(self._state.get("log") or [])
            log.append(line)
            self._state = {
                **self._state,
                "phase": line,
                "log": log[-40:],
            }

    def _run(self) -> None:
        try:
            method = None
            with self._lock:
                method = self._state.get("update_method")
            if method == "host_helper":
                self._run_host_helper()
            else:
                self._run_native_git()
        except BaseException as error:
            LOGGER.exception("product update failed")
            with self._lock:
                self._state = {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "error": str(error),
                    "phase": "Failed",
                    "log": list(self._state.get("log") or []),
                }
        finally:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None

    def _run_host_helper(self) -> None:
        helper = self._update_helper_path()
        if helper is None:
            raise RuntimeError("SURVNG_UPDATE_HELPER is not an executable file")
        self._append_log(f"Running host helper {helper}")
        completed = subprocess.run(
            [str(helper)],
            check=False,
            capture_output=True,
            text=True,
            timeout=HELPER_TIMEOUT_SECONDS,
        )
        if completed.stdout.strip():
            self._append_log(completed.stdout.strip().splitlines()[-1][:240])
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "helper failed").strip()
            raise RuntimeError(detail[-500:])
        with self._lock:
            self._state = {
                **self._state,
                "status": "complete",
                "finished_at": _utc_now(),
                "phase": "Host helper finished",
            }

    def _run_native_git(self) -> None:
        repo_root = self._configured_root or resolve_repo_root()
        if repo_root is None:
            raise RuntimeError("SurvNG git checkout was not found")
        branch = _git_output(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
        if branch == "HEAD":
            branch = DEFAULT_BRANCH
        upstream = f"{DEFAULT_REMOTE}/{branch}"

        self._append_log("Fetching remote commits")
        self._fetch(repo_root)

        dirty = _run_git(
            repo_root,
            ["status", "--porcelain", "--untracked-files=no"],
            timeout=30.0,
            check=False,
        ).stdout.strip()
        if dirty:
            raise RuntimeError("Tracked local changes appeared before pull")

        counts = _git_output(
            repo_root,
            ["rev-list", "--left-right", "--count", f"HEAD...{upstream}"],
        )
        ahead_text, _, behind_text = counts.partition("\t")
        if not behind_text:
            ahead_text, _, behind_text = counts.partition(" ")
        if int(ahead_text or "0") > 0:
            raise RuntimeError("Local commits prevent a fast-forward update")
        if int(behind_text or "0") <= 0:
            raise RuntimeError("No remote commits are available to apply")

        self._append_log(f"Fast-forwarding to {upstream}")
        _run_git(
            repo_root,
            ["pull", "--ff-only", DEFAULT_REMOTE, branch],
            timeout=PULL_TIMEOUT_SECONDS,
        )
        new_sha = _git_output(repo_root, ["rev-parse", "HEAD"])
        self._append_log(f"Now at {new_sha[:12]}")

        self._append_log("Installing Python dependencies")
        pip = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=PIP_TIMEOUT_SECONDS,
        )
        if pip.returncode != 0:
            detail = (pip.stderr or pip.stdout or "pip install failed").strip()
            raise RuntimeError(detail[-500:])

        frontend_dir = repo_root / "frontend"
        if (frontend_dir / "package.json").is_file():
            npm = shutil.which("npm")
            if npm is None:
                self._append_log("npm not found; skipping frontend rebuild")
            else:
                self._append_log("Building frontend")
                install = subprocess.run(
                    [npm, "ci", "--no-audit", "--no-fund"],
                    cwd=frontend_dir,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=FRONTEND_TIMEOUT_SECONDS,
                )
                if install.returncode != 0:
                    detail = (install.stderr or install.stdout or "npm ci failed").strip()
                    raise RuntimeError(detail[-500:])
                build = subprocess.run(
                    [npm, "run", "build"],
                    cwd=frontend_dir,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=FRONTEND_TIMEOUT_SECONDS,
                )
                if build.returncode != 0:
                    detail = (build.stderr or build.stdout or "npm build failed").strip()
                    raise RuntimeError(detail[-500:])

        if self._request_server_restart is None:
            with self._lock:
                self._state = {
                    **self._state,
                    "status": "complete",
                    "finished_at": _utc_now(),
                    "phase": "Update complete",
                    "current_sha": new_sha,
                }
            return

        self._append_log("Scheduling SurvNG restart")
        with self._lock:
            self._state = {
                **self._state,
                "status": "restarting",
                "phase": "Restarting SurvNG",
                "current_sha": new_sha,
            }
        # Brief pause so the final status poll can observe restarting.
        time.sleep(0.2)
        restart = self._request_server_restart()
        with self._lock:
            self._state = {
                **self._state,
                "status": "restarting",
                "finished_at": _utc_now(),
                "phase": "Restart scheduled",
                "restart": restart,
                "current_sha": new_sha,
            }
