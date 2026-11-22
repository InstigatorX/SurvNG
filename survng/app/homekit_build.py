"""Build and verify the HomeKit worker bundle served to local workers and encoders.

A build records its input fingerprint and prepares the encoder archive before
publishing a complete release. Readers pin an immutable directory so a
concurrent build cannot change which program or dependencies they see.

This module uses only the standard library: updaters and the worker bootstrap
run it as ``python -m survng.app.homekit_build`` without the service's
dependencies.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

WORKER_ROOT = Path(__file__).resolve().parents[2] / "homekit-worker"
MANIFEST = Path("dist") / "build-manifest.json"
MANIFEST_SCHEMA = 2
RELEASES = ".releases"
CURRENT = "current"
ARCHIVE = "agent.tar.gz"
# Every file that can change the compiled program or its installed packages.
INPUT_FILES = ("package.json", "package-lock.json", "tsconfig.json")
INPUT_DIRECTORIES = ("scripts", "src", "test")
# Modules the server starts or ships; a build without them is incomplete.
AGENT_OUTPUTS = tuple(Path("dist/src") / name for name in ("agent.js", "agent-launcher.js", "agent-generations.js"))
REQUIRED_OUTPUTS = (*AGENT_OUTPUTS, Path("dist/src/camera.js"))
HAP_PACKAGE = Path(".cache") / "hap-nodejs.tgz"
STAGING_PREFIX = ".build-"
# Per step (HAP prepare, npm ci, compile). Callers allow at least three steps.
BUILD_TIMEOUT_SECONDS = 900.0

NOT_BUILT = "worker_not_built"
STALE = "worker_build_stale"
DEPENDENCIES_STALE = "worker_dependencies_stale"


class BuildError(RuntimeError):
    pass


def _input_paths(root: Path) -> list[Path]:
    paths = [root / name for name in INPUT_FILES]
    for directory in INPUT_DIRECTORIES:
        base = root / directory
        if base.is_dir() and not base.is_symlink():
            paths.extend(path for path in base.rglob("*") if path.is_file() and not path.is_symlink())
    return sorted(path for path in paths if path.is_file() and not path.is_symlink())


def input_fingerprint(root: Path = WORKER_ROOT) -> str:
    """SHA-256 over every build input's relative path and content."""
    digest = hashlib.sha256()
    for path in _input_paths(root):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _manifest(path: Path) -> dict | None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _dependency_drift(root: Path) -> str | None:
    """First locked package whose installed version differs, or None."""
    try:
        packages = json.loads((root / "package-lock.json").read_text(encoding="utf-8")).get("packages") or {}
    except (OSError, ValueError, AttributeError):
        return "package-lock.json"
    if not isinstance(packages, dict):
        return "package-lock.json"
    for key, entry in packages.items():
        if not key.startswith("node_modules/") or not isinstance(entry, dict) or entry.get("link"):
            continue
        try:
            installed = json.loads((root / key / "package.json").read_text(encoding="utf-8")).get("version")
        except (OSError, ValueError, AttributeError):
            installed = None
        if installed is None and entry.get("optional"):
            continue
        if installed != entry.get("version"):
            return key.removeprefix("node_modules/")
    return None


def _current(root: Path) -> Path | None:
    releases = root / RELEASES
    try:
        if releases.is_symlink():
            return None
        target = (releases / CURRENT).readlink()
        if len(target.parts) == 1 and target.name not in {".", "..", CURRENT} and not target.is_absolute():
            release = releases / target
            if not release.is_symlink() and release.is_dir():
                return release
    except OSError:
        pass
    return None


def read_manifest(root: Path = WORKER_ROOT) -> dict | None:
    release = _current(root)
    return _manifest((release or root) / MANIFEST)


def _signature(root: Path, release: Path) -> tuple:
    paths = [*_input_paths(root), release / MANIFEST, release / ARCHIVE,
             *(release / path for path in REQUIRED_OUTPUTS), release / "package-lock.json"]
    modules = release / "node_modules"
    if modules.is_dir() and not modules.is_symlink():
        paths.extend(modules.glob("*/package.json"))
        paths.extend(modules.glob("@*/*/package.json"))
    rows: list[tuple] = []
    for path in sorted(paths):
        try:
            info = path.stat()
        except OSError:
            rows.append((str(path), None))
            continue
        rows.append((str(path), info.st_mtime_ns, info.st_ctime_ns, info.st_size, info.st_ino))
    return tuple(rows)


_verify_lock = threading.Lock()
_verify_cache: dict[Path, tuple[tuple, str | None]] = {}


def verified_release(root: Path = WORKER_ROOT) -> tuple[Path | None, str | None]:
    """Pin one complete release while it matches the checkout, or report why it cannot run."""
    root = root.resolve()
    release = _current(root)
    if release is None:
        legacy = (root / "dist").exists() or (root / "node_modules").exists()
        return None, STALE if legacy else NOT_BUILT
    signature = _signature(root, release)
    with _verify_lock:
        cached = _verify_cache.get(root)
        if cached is not None and cached[0] == signature:
            return (release, None) if cached[1] is None else (None, cached[1])
    manifest = _manifest(release / MANIFEST)
    try:
        if (any(not (release / path).is_file() for path in (*REQUIRED_OUTPUTS, Path(ARCHIVE)))
                or not (release / "node_modules").is_dir()):
            problem = NOT_BUILT
        elif (manifest is None or manifest.get("schema") != MANIFEST_SCHEMA
              or manifest.get("inputs") != input_fingerprint(root)
              or not _valid_digest(manifest.get("archive"))):
            problem = STALE
        elif _dependency_drift(release) is not None:
            problem = DEPENDENCIES_STALE
        else:
            with (release / ARCHIVE).open("rb") as archive:
                problem = None if hashlib.file_digest(archive, "sha256").hexdigest() == manifest["archive"] else STALE
    except OSError:
        return None, STALE  # A concurrent checkout may temporarily remove an input.
    if signature != _signature(root, release):
        return None, STALE
    with _verify_lock:
        _verify_cache[root] = (signature, problem)
    return (release, None) if problem is None else (None, problem)


def verify(root: Path = WORKER_ROOT) -> str | None:
    return verified_release(root)[1]


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def release_archive_version(release: Path) -> str:
    manifest = _manifest(release / MANIFEST)
    digest = (manifest or {}).get("archive")
    if not _valid_digest(digest):
        raise BuildError("HomeKit encoder archive manifest is invalid")
    return digest


@lru_cache(maxsize=2)
def _read_archive(path: Path, digest: str, identity: tuple) -> bytes:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise BuildError("HomeKit encoder archive checksum mismatch")
    return payload


def read_release_archive(release: Path) -> tuple[str, bytes]:
    digest = release_archive_version(release)
    path = release / ARCHIVE
    stat = path.stat()
    identity = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino)
    return digest, _read_archive(path, digest, identity)


def agent_archive(root: Path) -> bytes:
    if any(not (root / path).exists() for path in ("package.json", *AGENT_OUTPUTS, "node_modules")):
        raise FileNotFoundError("HomeKit encoder agent is not built")
    buffer = io.BytesIO()

    def accept(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        name = Path(member.name)
        if name == MANIFEST or any(part in {".cache", ".git"} for part in name.parts):
            return None
        if not (member.isfile() or member.isdir()):
            return None
        # Package metadata and file contents are inputs. Deployment owner and
        # build timestamps must not change the advertised encoder version.
        member.uid = member.gid = member.mtime = 0
        member.uname = member.gname = ""
        return member

    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed, tarfile.open(fileobj=compressed, mode="w") as archive:
        for relative in ("package.json", "dist", "node_modules"):
            archive.add(root / relative, arcname=relative, filter=accept)
    return buffer.getvalue()


def _hap_package_current(root: Path) -> bool:
    """The packed HAP archive matches the lockfile's integrity pin."""
    try:
        lock = json.loads((root / "package-lock.json").read_text(encoding="utf-8"))
        integrity = lock["packages"]["node_modules/@homebridge/hap-nodejs"]["integrity"]
        algorithm, _, expected = integrity.partition("-")
        actual = hashlib.new(algorithm, (root / HAP_PACKAGE).read_bytes()).digest()
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
    return base64.b64encode(actual).decode() == expected


def _node_major(node: str) -> str | None:
    try:
        completed = subprocess.run([node, "-p", "process.versions.node.split('.')[0]"],
            capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def resolve_node(root: Path, requested: str | None = None) -> str:
    """Pick a Node.js 24 executable: explicit, environment, previous build, then PATH."""
    previous = (read_manifest(root) or {}).get("node")
    candidates = [requested, os.environ.get("SURVNG_HOMEKIT_NODE"),
        previous if isinstance(previous, str) else None, shutil.which("node")]
    explicit = requested or os.environ.get("SURVNG_HOMEKIT_NODE")
    for candidate in candidates:
        if not candidate:
            continue
        path = shutil.which(candidate) or candidate
        if _node_major(path) == "24":
            return str(Path(path).resolve())
        if candidate == explicit:
            raise BuildError(f"{candidate} is not Node.js 24")
    raise BuildError("Node.js 24 was not found; pass --node or set SURVNG_HOMEKIT_NODE")


Runner = Callable[..., subprocess.CompletedProcess]


def _run(runner: Runner, command: Sequence[str], cwd: Path, env: dict) -> None:
    completed = runner(list(command), cwd=cwd, env=env, check=False, capture_output=True, text=True,
        timeout=BUILD_TIMEOUT_SECONDS)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-800:]
        name = " ".join([Path(command[0]).name, *command[1:2]])
        raise BuildError(f"{name} failed: {detail}" if detail else f"{name} failed")


def _copy_inputs(root: Path, staging: Path) -> None:
    for path in _input_paths(root):
        target = staging / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (staging / HAP_PACKAGE).parent.mkdir(parents=True, exist_ok=True)
    if (root / HAP_PACKAGE).is_file():
        shutil.copyfile(root / HAP_PACKAGE, staging / HAP_PACKAGE)


def _cache_hap_package(root: Path, staging: Path) -> None:
    cache = (root / HAP_PACKAGE).parent
    cache.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".hap-", dir=cache, delete=False) as temporary:
        candidate = Path(temporary.name)
    try:
        shutil.copyfile(staging / HAP_PACKAGE, candidate)
        candidate.replace(root / HAP_PACKAGE)
    finally:
        candidate.unlink(missing_ok=True)


def _publish(root: Path, staging: Path, fingerprint: str) -> None:
    """Publish a complete release without modifying any directory pinned by a reader."""
    releases = root / RELEASES
    if releases.is_symlink():
        raise BuildError("HomeKit release directory cannot be a symlink")
    releases.mkdir(exist_ok=True)
    bundle = staging / "release"
    destination = releases / f"{fingerprint[:12]}-{staging.name.removeprefix(STAGING_PREFIX)}"
    if destination.exists() or destination.is_symlink():
        raise BuildError("HomeKit release already exists")
    candidate = staging / "publish"
    candidate.symlink_to(destination.name)
    bundle.rename(destination)
    try:
        if input_fingerprint(root) != fingerprint:
            raise BuildError("HomeKit worker inputs changed during the build")
        candidate.replace(releases / CURRENT)
    finally:
        if _current(root) != destination:
            shutil.rmtree(destination)


LOCK_WAIT_SECONDS = 3 * BUILD_TIMEOUT_SECONDS
# Leave time for activation and cleanup after the lock wait and build steps.
BUILDER_TIMEOUT_SECONDS = LOCK_WAIT_SECONDS + 3 * BUILD_TIMEOUT_SECONDS + 60


def _acquire(lock, log: Callable[[str], None], wait_seconds: float | None = None) -> None:
    """Take the build lock, waiting a bounded time for another builder to finish.

    The holder (server auto-build or an updater) usually leaves the worker
    current, which the caller then re-checks.
    """
    import time

    deadline = time.monotonic() + (LOCK_WAIT_SECONDS if wait_seconds is None else wait_seconds)
    announced = False
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise BuildError("Timed out waiting for another HomeKit worker build to finish") from None
            if not announced:
                log("Waiting for another HomeKit worker build to finish")
                announced = True
            time.sleep(1)


def build(root: Path = WORKER_ROOT, *, node: str | None = None, force: bool = False,
          runner: Runner = subprocess.run, log: Callable[[str], None] = print) -> bool:
    """Build the worker unless it is already current. Returns True when it rebuilt.

    The build runs in a private staging directory; the existing bundle stays in
    place until the new one has compiled and verified.
    """
    root = root.resolve()
    lock_path = root / ".build.lock"
    with open(lock_path, "a+") as lock:
        _acquire(lock, log)
        for stale in root.glob(f"{STAGING_PREFIX}*"):
            if stale.is_dir() and not stale.is_symlink():
                shutil.rmtree(stale, ignore_errors=True)
        if not force and verify(root) is None:
            log("HomeKit worker is current")
            return False
        executable = resolve_node(root, node)
        npm = Path(executable).parent / "npm"
        npm_command = str(npm) if npm.is_file() else shutil.which("npm")
        if npm_command is None:
            raise BuildError("npm was not found next to Node.js 24 or on PATH")
        env = dict(os.environ, PATH=f"{Path(executable).parent}{os.pathsep}{os.environ.get('PATH', '')}")
        staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=root))
        try:
            fingerprint = input_fingerprint(root)
            _copy_inputs(root, staging)
            if input_fingerprint(staging) != fingerprint:
                raise BuildError("HomeKit worker inputs changed while preparing the build")
            if not _hap_package_current(staging):
                # Run the copied bootstrap against the copied lockfile.
                log("Preparing the pinned HAP-NodeJS package")
                _run(runner, [sys.executable, str(staging / "scripts" / "bootstrap.py"), "--hap-only"], staging, env)
                if not _hap_package_current(staging):
                    raise BuildError("The HAP-NodeJS package does not match the lockfile integrity")
                _cache_hap_package(root, staging)
            log("Installing locked HomeKit worker packages")
            _run(runner, [npm_command, "ci", "--ignore-scripts", "--no-audit", "--no-fund"], staging, env)
            log("Compiling the HomeKit worker")
            _run(runner, [executable, "scripts/build.mjs"], staging, env)
            if input_fingerprint(staging) != fingerprint:
                raise BuildError("The build modified its HomeKit worker inputs")
            if any(not (staging / output).is_file() for output in REQUIRED_OUTPUTS):
                raise BuildError(f"The staged HomeKit worker failed verification ({NOT_BUILT})")
            if _dependency_drift(staging) is not None:
                raise BuildError(f"The staged HomeKit worker failed verification ({DEPENDENCIES_STALE})")
            bundle = staging / "release"
            bundle.mkdir()
            for name in ("package.json", "package-lock.json", "dist", "node_modules"):
                (staging / name).rename(bundle / name)
            archive = agent_archive(bundle)
            (bundle / ARCHIVE).write_bytes(archive)
            (bundle / MANIFEST).write_text(json.dumps({"schema": MANIFEST_SCHEMA, "inputs": fingerprint,
                "archive": hashlib.sha256(archive).hexdigest(), "node": executable}, indent=2) + "\n", encoding="utf-8")
            _publish(root, staging, fingerprint)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        log(f"HomeKit worker built ({fingerprint[:12]})")
        return True


def previously_built(root: Path = WORKER_ROOT) -> bool:
    return _current(root) is not None or any((root / name).exists() for name in ("dist", "node_modules"))


def rebuild_needed(root: Path = WORKER_ROOT) -> str | None:
    """The verify error for a checkout that has built the worker before, else None."""
    return verify(root) if previously_built(root) else None


def _build_command(root: Path, node: str | None) -> list[str]:
    return [sys.executable, "-m", "survng.app.homekit_build", "--root", str(root),
        *(["--node", node] if node else [])]


class AutoBuilder:
    """Rebuilds a stale or missing worker in the background for the running server.

    Builds run as a separate process (the CLI above) so the server never imports
    Node tooling or blocks on it. One build runs at a time; after a failure the
    next automatic attempt waits ``retry_seconds`` unless a caller forces it.
    """

    def __init__(self, root: Path = WORKER_ROOT, *, runner: Runner = subprocess.run,
                 retry_seconds: float = 15 * 60, clock: Callable[[], float] | None = None,
                 log: Callable[[str], None] | None = None, enabled: bool | None = None) -> None:
        import logging
        import time

        self.root = root
        # SURVNG_HOMEKIT_AUTO_BUILD=0 leaves builds to the updaters and operators.
        self.enabled = os.environ.get("SURVNG_HOMEKIT_AUTO_BUILD", "1") != "0" if enabled is None else enabled
        self.runner = runner
        self.retry_seconds = retry_seconds
        self.clock = clock or time.monotonic
        self.wall = time.time
        self.log = log or logging.getLogger(__name__).warning
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.state = "idle"
        self.error: str | None = None
        self.failed_at: float | None = None
        self.finished_at: float | None = None

    def available(self) -> bool:
        # Docker images and installs without the worker sources have nothing to build.
        return self.enabled and (self.root / "package.json").is_file() and (self.root / "package-lock.json").is_file()

    def ensure(self, node: str | None, *, force: bool = False) -> bool:
        """Start a build when the worker is not current. Returns True when one is running."""
        if not self.available():
            return False
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return True
            # Inside the backoff nothing can start, so skip the freshness walk.
            if (not force and self.state == "failed" and self.failed_at is not None
                    and self.clock() - self.failed_at < self.retry_seconds):
                return False
        if verify(self.root) is None:
            with self.lock:
                if self.state == "failed":
                    self.state, self.error = "idle", None
            return False
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return True
            if (not force and self.state == "failed" and self.failed_at is not None
                    and self.clock() - self.failed_at < self.retry_seconds):
                return False
            self.state, self.error = "building", None
            self.thread = threading.Thread(target=self._run, args=(node,), name="homekit-worker-build", daemon=True)
            self.thread.start()
            return True

    def _run(self, node: str | None) -> None:
        command = _build_command(self.root, node)
        try:
            completed = self.runner(command, cwd=self.root.parent, check=False, capture_output=True, text=True,
                timeout=BUILDER_TIMEOUT_SECONDS)
            error = None if completed.returncode == 0 else (
                (completed.stderr or completed.stdout or "").strip()[-400:] or "build failed")
        except (OSError, subprocess.SubprocessError) as failure:
            error = str(failure)[-400:] or type(failure).__name__
        problem = verify(self.root) if error is None else None
        if problem is not None:
            error = f"build finished but the worker is still {problem}"
        with self.lock:
            self.finished_at = self.wall()
            if error is None:
                self.state, self.error, self.failed_at = "idle", None, None
            else:
                self.state, self.error, self.failed_at = "failed", error, self.clock()
        if error is not None:
            self.log(f"HomeKit worker build failed: {error}")

    def status(self) -> dict:
        with self.lock:
            failed = self.state == "failed"
        # A build by an updater, the CLI, or a later retry clears an old failure.
        if failed and verify(self.root) is None:
            with self.lock:
                if self.state == "failed":
                    self.state, self.error, self.failed_at = "idle", None, None
        with self.lock:
            return {"state": self.state, "error": self.error, "available": self.available(),
                "finishedAt": int(self.finished_at) if self.finished_at else None}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=WORKER_ROOT)
    parser.add_argument("--node", help="Node.js 24 executable")
    parser.add_argument("--force", action="store_true", help="rebuild even when current")
    parser.add_argument("--if-present", action="store_true",
        help="skip when the worker has never been built in this checkout")
    parser.add_argument("--check", action="store_true", help="report freshness without building")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.check:
        problem = verify(root)
        print(problem or "current")
        return 0 if problem is None else 1
    if args.if_present and not previously_built(root):
        print("HomeKit worker was never built here; skipping")
        return 0
    try:
        build(root, node=args.node, force=args.force)
    except (BuildError, OSError, subprocess.SubprocessError) as error:
        print(f"HomeKit worker build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
