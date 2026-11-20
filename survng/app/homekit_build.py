"""Build and verify the HomeKit worker bundle served to local workers and encoders.

A build records a fingerprint of its inputs in ``dist/build-manifest.json``.
The server offers or starts the worker only while that fingerprint matches the
checkout and the installed packages match the lockfile, so a checkout switch
without a rebuild blocks HomeKit instead of serving an older program.

This module uses only the standard library: updaters and the worker bootstrap
run it as ``python -m survng.app.homekit_build`` without the service's
dependencies.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

WORKER_ROOT = Path(__file__).resolve().parents[2] / "homekit-worker"
MANIFEST = Path("dist") / "build-manifest.json"
MANIFEST_SCHEMA = 1
# Every file that can change the compiled program or its installed packages.
INPUT_FILES = ("package.json", "package-lock.json", "tsconfig.json")
INPUT_DIRECTORIES = ("scripts", "src", "test")
# Modules the server starts or ships; a build without them is incomplete.
REQUIRED_OUTPUTS = (Path("dist") / "src" / "agent.js", Path("dist") / "src" / "camera.js")
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


def read_manifest(root: Path = WORKER_ROOT) -> dict | None:
    try:
        manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _dependency_drift(root: Path) -> str | None:
    """First locked package whose installed version differs, or None."""
    try:
        packages = json.loads((root / "package-lock.json").read_text(encoding="utf-8")).get("packages") or {}
    except (OSError, ValueError, AttributeError):
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


def _check(root: Path) -> str | None:
    if any(not (root / path).is_file() for path in REQUIRED_OUTPUTS) or not (root / "node_modules").is_dir():
        return NOT_BUILT
    manifest = read_manifest(root)
    if (manifest is None or manifest.get("schema") != MANIFEST_SCHEMA
            or manifest.get("inputs") != input_fingerprint(root)):
        return STALE
    if _dependency_drift(root) is not None:
        return DEPENDENCIES_STALE
    return None


def _signature(root: Path) -> tuple:
    paths = [*_input_paths(root), root / MANIFEST, *(root / path for path in REQUIRED_OUTPUTS)]
    modules = root / "node_modules"
    if modules.is_dir() and not modules.is_symlink():
        paths.extend(modules.glob("*/package.json"))
        paths.extend(modules.glob("@*/*/package.json"))
    rows: list[tuple] = []
    for path in paths:
        try:
            info = path.stat()
        except OSError:
            rows.append((str(path), None))
            continue
        rows.append((str(path), info.st_mtime_ns, info.st_size, info.st_ino))
    return tuple(rows)


_verify_lock = threading.Lock()
_verify_cache: dict[Path, tuple[tuple, str | None]] = {}


def verify(root: Path = WORKER_ROOT) -> str | None:
    """None when the built worker matches the checkout, else an error code.

    Results are cached against file metadata so callers on hot paths (encoder
    heartbeats, status polls) only rehash after something changed.
    """
    root = root.resolve()
    signature = _signature(root)
    with _verify_lock:
        cached = _verify_cache.get(root)
        if cached is not None and cached[0] == signature:
            return cached[1]
    result = _check(root)
    with _verify_lock:
        _verify_cache[root] = (signature, result)
    return result


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
    shutil.copy2(root / HAP_PACKAGE, staging / HAP_PACKAGE)


def _activate(root: Path, staging: Path) -> None:
    """Move the staged packages, then the staged program, into place.

    The manifest lives in ``dist`` and moves last. If activation stops between
    the two moves, the old manifest no longer matches and HomeKit stays blocked
    until the next build.
    """
    retired = staging / "retired"
    retired.mkdir()
    for name in ("node_modules", "dist"):
        current = root / name
        if current.exists() or current.is_symlink():
            current.rename(retired / name)
        (staging / name).rename(current)


def build(root: Path = WORKER_ROOT, *, node: str | None = None, force: bool = False,
          runner: Runner = subprocess.run, log: Callable[[str], None] = print) -> bool:
    """Build the worker unless it is already current. Returns True when it rebuilt.

    The build runs in a private staging directory; the existing bundle stays in
    place until the new one has compiled and verified.
    """
    root = root.resolve()
    lock_path = root / ".build.lock"
    with open(lock_path, "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BuildError("Another HomeKit worker build is running") from None
        for stale in root.glob(f"{STAGING_PREFIX}*"):
            if stale.is_dir() and not stale.is_symlink():
                shutil.rmtree(stale, ignore_errors=True)
        if not force and verify(root) is None:
            log("HomeKit worker is current")
            return False
        executable = resolve_node(root, node)
        if not _hap_package_current(root):
            # The pinned HAP package is built from audited upstream source.
            log("Preparing the pinned HAP-NodeJS package")
            env = dict(os.environ, PATH=f"{Path(executable).parent}{os.pathsep}{os.environ.get('PATH', '')}")
            _run(runner, [sys.executable, str(root / "scripts" / "bootstrap.py"), "--hap-only"], root, env)
            if not _hap_package_current(root):
                raise BuildError("The HAP-NodeJS package does not match the lockfile integrity")
        npm = Path(executable).parent / "npm"
        npm_command = str(npm) if npm.is_file() else shutil.which("npm")
        if npm_command is None:
            raise BuildError("npm was not found next to Node.js 24 or on PATH")
        env = dict(os.environ, PATH=f"{Path(executable).parent}{os.pathsep}{os.environ.get('PATH', '')}")
        staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=root))
        try:
            _copy_inputs(root, staging)
            fingerprint = input_fingerprint(staging)
            log("Installing locked HomeKit worker packages")
            _run(runner, [npm_command, "ci", "--ignore-scripts", "--no-audit", "--no-fund"], staging, env)
            log("Compiling the HomeKit worker")
            _run(runner, [executable, "scripts/build.mjs"], staging, env)
            (staging / MANIFEST).write_text(json.dumps({"schema": MANIFEST_SCHEMA, "inputs": fingerprint,
                "node": executable}, indent=2) + "\n", encoding="utf-8")
            problem = _check(staging)
            if problem is not None:
                raise BuildError(f"The staged HomeKit worker failed verification ({problem})")
            _activate(root, staging)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        result = verify(root)
        if result is not None:
            raise BuildError(f"The activated HomeKit worker failed verification ({result})")
        log(f"HomeKit worker built ({fingerprint[:12]})")
        return True


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
    if args.if_present and not any((root / name).exists() for name in ("dist", "node_modules")):
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
