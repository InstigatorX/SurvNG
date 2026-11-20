"""HomeKit worker builder and the freshness gate that keeps stale bundles from being offered."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from threading import Event

import pytest

from survng.app import homekit_build

HAP = b"pinned hap package"


def _lock(werift: str = "0.28.0", hap: bytes = HAP) -> dict:
    return {"lockfileVersion": 3, "packages": {
        "": {"name": "survng-homekit-worker"},
        "node_modules/werift": {"version": werift},
        "node_modules/@homebridge/hap-nodejs": {"version": "2.3.0", "resolved": "file:.cache/hap-nodejs.tgz",
            "integrity": "sha512-" + base64.b64encode(hashlib.sha512(hap).digest()).decode()},
        "node_modules/typescript": {"version": "5.9.3", "dev": True},
        "node_modules/fsevents": {"version": "2.3.3", "optional": True},
    }}


def _worker(tmp_path: Path) -> Path:
    root = tmp_path / "homekit-worker"
    for directory in ("scripts", "src", "test", ".cache"):
        (root / directory).mkdir(parents=True)
    (root / "package.json").write_text('{"name":"survng-homekit-worker"}\n')
    (root / "package-lock.json").write_text(json.dumps(_lock()))
    (root / "tsconfig.json").write_text("{}\n")
    (root / "scripts" / "build.mjs").write_text("// build\n")
    (root / "src" / "agent.ts").write_text("export const release = 1;\n")
    (root / "test" / "agent.test.ts").write_text("// test\n")
    (root / ".cache" / "hap-nodejs.tgz").write_bytes(HAP)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "node").write_text("")
    (tmp_path / "bin" / "npm").write_text("")
    return root


class FakeToolchain:
    """Simulates npm ci and the TypeScript build inside the staging directory."""

    def __init__(self, *, fail: str | None = None, installed: dict[str, str] | None = None) -> None:
        self.fail = fail
        self.installed = installed or {}
        self.commands: list[list[str]] = []

    def __call__(self, command, *, cwd, **_kwargs):
        self.commands.append(command)
        cwd = Path(cwd)
        step = "bootstrap" if command[-1] == "--hap-only" else "ci" if command[1:2] == ["ci"] else "build"
        if step == self.fail:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr=f"{step} exploded")
        if step == "bootstrap":
            (cwd / ".cache" / "hap-nodejs.tgz").write_bytes(HAP)
        elif step == "ci":
            lock = json.loads((cwd / "package-lock.json").read_text())
            for key, entry in lock["packages"].items():
                if key.startswith("node_modules/") and not entry.get("optional"):
                    name = key.removeprefix("node_modules/")
                    (cwd / key).mkdir(parents=True)
                    (cwd / key / "package.json").write_text(json.dumps({"version": self.installed.get(name, entry["version"])}))
        else:
            (cwd / "dist" / "src").mkdir(parents=True)
            source = (cwd / "src" / "agent.ts").read_text()
            (cwd / "dist" / "src" / "agent.js").write_text(source)
            (cwd / "dist" / "src" / "camera.js").write_text(source)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")


@pytest.fixture(autouse=True)
def node_24(monkeypatch):
    monkeypatch.setattr(homekit_build, "_node_major", lambda node: "24")
    monkeypatch.delenv("SURVNG_HOMEKIT_NODE", raising=False)


def _build(root: Path, toolchain: FakeToolchain, **kwargs) -> bool:
    return homekit_build.build(root, node=str(root.parent / "bin" / "node"), runner=toolchain, log=lambda _line: None, **kwargs)


def _active(root: Path) -> Path:
    release, problem = homekit_build.verified_release(root)
    assert problem is None and release is not None
    return release


def test_build_activates_a_verified_bundle_and_skips_when_current(tmp_path):
    root = _worker(tmp_path)
    assert homekit_build.verify(root) == homekit_build.NOT_BUILT
    toolchain = FakeToolchain()

    assert _build(root, toolchain) is True
    assert homekit_build.verify(root) is None
    manifest = homekit_build.read_manifest(root)
    assert manifest["inputs"] == homekit_build.input_fingerprint(root)
    release = _active(root)
    assert (root / ".releases" / "current").readlink() == Path(release.name)
    digest, payload = homekit_build.read_release_archive(release)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert digest == homekit_build.release_archive_version(release)
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        assert {"dist/src/agent.js", "dist/src/camera.js", "node_modules/werift/package.json"} <= set(archive.getnames())
        assert str(homekit_build.MANIFEST) not in archive.getnames()
    assert not list(root.glob(".build-*"))
    # Packages install from the lockfile in staging, never in the live tree.
    assert all(Path(command[0]).name != "npm" or command[1] == "ci" for command in toolchain.commands)

    again = FakeToolchain()
    assert _build(root, again) is False
    assert again.commands == []


def test_bundle_without_manifest_requires_migration(tmp_path):
    """A build left over from another checkout (the reported incident) is stale."""
    root = _worker(tmp_path)
    (root / "dist" / "src").mkdir(parents=True)
    (root / "dist" / "src" / "agent.js").write_text("old")
    (root / "dist" / "src" / "camera.js").write_text("old")
    (root / "node_modules" / "werift").mkdir(parents=True)
    (root / "node_modules" / "werift" / "package.json").write_text('{"version":"0.24.4"}')

    assert homekit_build.verified_release(root) == (None, homekit_build.STALE)
    assert homekit_build.previously_built(root)
    _build(root, FakeToolchain())
    assert (root / "dist" / "src" / "agent.js").read_text() == "old"
    assert "release = 1" in (_active(root) / "dist" / "src" / "agent.js").read_text()


def test_source_change_blocks_the_release_until_rebuilt(tmp_path):
    root = _worker(tmp_path)
    _build(root, FakeToolchain())
    previous = _active(root)
    before = homekit_build.release_archive_version(previous)

    (root / "src" / "agent.ts").write_text("export const release = 2;\n")
    assert homekit_build.verified_release(root) == (None, homekit_build.STALE)

    assert _build(root, FakeToolchain()) is True
    current = _active(root)
    after = homekit_build.release_archive_version(current)
    assert after != before
    assert homekit_build.read_release_archive(previous)[0] == before
    assert "release = 1" in (previous / "dist" / "src" / "agent.js").read_text()
    assert "release = 2" in (current / "dist" / "src" / "agent.js").read_text()


@pytest.mark.parametrize("step", ["ci", "build"])
def test_failed_build_keeps_the_previous_bundle_blocked_and_retry_recovers(tmp_path, step):
    root = _worker(tmp_path)
    _build(root, FakeToolchain())
    previous = _active(root)
    (root / "src" / "agent.ts").write_text("export const release = 2;\n")

    with pytest.raises(homekit_build.BuildError, match=f"{step} exploded" if step == "ci" else "build.mjs failed"):
        _build(root, FakeToolchain(fail=step))
    # The live bundle is untouched, and the freshness gate still blocks it.
    assert (root / ".releases" / "current").readlink() == Path(previous.name)
    assert "release = 1" in (previous / "dist" / "src" / "agent.js").read_text()
    assert homekit_build.verify(root) == homekit_build.STALE
    assert not list(root.glob(".build-*"))

    assert _build(root, FakeToolchain()) is True
    assert homekit_build.verify(root) is None


def test_installed_package_drift_blocks_the_release(tmp_path):
    root = _worker(tmp_path)
    _build(root, FakeToolchain())
    (_active(root) / "node_modules" / "werift" / "package.json").write_text('{"version":"0.24.4"}')
    assert homekit_build.verify(root) == homekit_build.DEPENDENCIES_STALE


def test_staged_dependency_mismatch_never_activates(tmp_path):
    root = _worker(tmp_path)
    with pytest.raises(homekit_build.BuildError, match="worker_dependencies_stale"):
        _build(root, FakeToolchain(installed={"werift": "0.24.4"}))
    assert not (root / "dist").exists()
    assert homekit_build.verify(root) == homekit_build.NOT_BUILT


def test_hap_package_is_rebuilt_when_the_lock_pins_another(tmp_path):
    root = _worker(tmp_path)
    (root / ".cache" / "hap-nodejs.tgz").write_bytes(b"previous hap pin")
    toolchain = FakeToolchain()
    _build(root, toolchain)
    script, switch = toolchain.commands[0][-2:]
    assert Path(script).relative_to(root).parts[0].startswith(".build-")
    assert switch == "--hap-only"
    assert (root / ".cache" / "hap-nodejs.tgz").read_bytes() == HAP
    assert homekit_build.verify(root) is None


def test_explicit_node_must_be_version_24(tmp_path, monkeypatch):
    root = _worker(tmp_path)
    monkeypatch.setattr(homekit_build, "_node_major", lambda node: "22")
    with pytest.raises(homekit_build.BuildError, match="is not Node.js 24"):
        _build(root, FakeToolchain())


def test_cli_if_present_skips_checkouts_that_never_built(tmp_path, capsys):
    root = _worker(tmp_path)
    assert homekit_build.main(["--root", str(root), "--if-present"]) == 0
    assert "never built" in capsys.readouterr().out
    assert homekit_build.main(["--root", str(root), "--check"]) == 1


def test_lockfile_change_blocks_the_release(tmp_path):
    root = _worker(tmp_path)
    assert homekit_build.verify(root) == homekit_build.NOT_BUILT
    _build(root, FakeToolchain())
    assert homekit_build.verify(root) is None
    (root / "package-lock.json").write_text(json.dumps(_lock(werift="0.29.0")))
    assert homekit_build.verify(root) == homekit_build.STALE


def test_rebuild_keeps_readers_on_one_complete_release(tmp_path):
    root = _worker(tmp_path)
    _build(root, FakeToolchain())
    previous = _active(root)
    previous_archive = homekit_build.read_release_archive(previous)
    compiled = Event()
    publish = Event()

    class BlockingToolchain(FakeToolchain):
        def __call__(self, command, **kwargs):
            result = super().__call__(command, **kwargs)
            if command[-1] == "scripts/build.mjs":
                compiled.set()
                assert publish.wait(5)
            return result

    with ThreadPoolExecutor(max_workers=1) as pool:
        build = pool.submit(_build, root, BlockingToolchain(), force=True)
        try:
            assert compiled.wait(5)
            assert _active(root) == previous
            assert homekit_build.read_release_archive(previous) == previous_archive
        finally:
            publish.set()
        assert build.result(timeout=5)

    current = _active(root)
    assert current != previous
    assert homekit_build.read_release_archive(current) == previous_archive
    assert homekit_build.read_release_archive(previous) == previous_archive


def test_source_change_during_build_does_not_publish(tmp_path):
    root = _worker(tmp_path)
    _build(root, FakeToolchain())
    previous = _active(root)

    class ChangingToolchain(FakeToolchain):
        def __call__(self, command, **kwargs):
            result = super().__call__(command, **kwargs)
            if command[-1] == "scripts/build.mjs":
                (root / "src" / "agent.ts").write_text("export const release = 2;\n")
            return result

    with pytest.raises(homekit_build.BuildError, match="inputs changed during the build"):
        _build(root, ChangingToolchain(), force=True)
    assert (root / ".releases" / "current").readlink() == Path(previous.name)
    assert set((root / ".releases").iterdir()) == {previous, root / ".releases" / "current"}
    assert homekit_build.verify(root) == homekit_build.STALE
    assert not list(root.glob(".build-*"))


def test_interrupted_publication_keeps_previous_complete_release(tmp_path, monkeypatch):
    root = _worker(tmp_path)
    _build(root, FakeToolchain())
    previous = _active(root)
    previous_archive = homekit_build.read_release_archive(previous)
    replace = Path.replace

    def interrupted(candidate, target):
        if target == root / ".releases" / "current":
            raise KeyboardInterrupt
        return replace(candidate, target)

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _build(root, FakeToolchain(), force=True)
    assert _active(root) == previous
    assert homekit_build.read_release_archive(previous) == previous_archive
    assert set((root / ".releases").iterdir()) == {previous, root / ".releases" / "current"}
    assert not list(root.glob(".build-*"))


def test_agent_archive_checksum_mismatch_is_detected(tmp_path):
    root = _worker(tmp_path)
    _build(root, FakeToolchain())
    release = _active(root)
    (release / homekit_build.ARCHIVE).write_bytes(b"corrupt")
    assert homekit_build.verify(root) == homekit_build.STALE
    with pytest.raises(homekit_build.BuildError, match="checksum mismatch"):
        homekit_build.read_release_archive(release)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _auto_builder(root: Path, outcomes: list, clock: FakeClock) -> tuple[homekit_build.AutoBuilder, list]:
    """AutoBuilder whose build process either builds with the fake toolchain or fails."""
    commands: list = []

    def runner(command, **_kwargs):
        commands.append(command)
        if outcomes.pop(0):
            _build(root, FakeToolchain())
            return subprocess.CompletedProcess(command, 0, stdout="built", stderr="")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="Node.js 24 was not found")

    builder = homekit_build.AutoBuilder(root, runner=runner, clock=clock, log=lambda _line: None,
        enabled=True, retry_seconds=60)
    return builder, commands


def _settle(builder: homekit_build.AutoBuilder) -> dict:
    if builder.thread is not None:
        builder.thread.join(5)
    return builder.status()


def test_auto_builder_rebuilds_a_stale_worker_in_the_background(tmp_path):
    root = _worker(tmp_path)
    builder, commands = _auto_builder(root, [True], FakeClock())
    assert builder.ensure("/opt/node24/bin/node") is True
    assert _settle(builder)["state"] == "idle"
    assert homekit_build.verify(root) is None
    assert commands[0][1:] == ["-m", "survng.app.homekit_build", "--root", str(root), "--node", "/opt/node24/bin/node"]
    # Current builds start nothing.
    assert builder.ensure("/opt/node24/bin/node") is False
    assert len(commands) == 1


def test_auto_builder_backs_off_after_a_failure_and_force_retries(tmp_path):
    root = _worker(tmp_path)
    clock = FakeClock()
    builder, commands = _auto_builder(root, [False, False, True], clock)
    builder.ensure(None)
    job = _settle(builder)
    assert job["state"] == "failed" and "Node.js 24 was not found" in job["error"]

    assert builder.ensure(None) is False, "automatic retries wait for the backoff"
    assert builder.ensure(None, force=True) is True, "an operator retry starts immediately"
    assert _settle(builder)["state"] == "failed"
    clock.now += 61
    assert builder.ensure(None) is True
    job = _settle(builder)
    assert (job["state"], job["error"], job["available"]) == ("idle", None, True)
    assert len(commands) == 3


def test_auto_builder_clears_a_failure_once_another_build_succeeds(tmp_path):
    root = _worker(tmp_path)
    builder, _commands = _auto_builder(root, [False], FakeClock())
    builder.ensure(None)
    assert _settle(builder)["state"] == "failed"
    _build(root, FakeToolchain())  # e.g. the Admin updater's rebuild job
    assert builder.status()["state"] == "idle"
    assert builder.status()["error"] is None


def test_waiting_for_another_build_is_bounded(tmp_path):
    import fcntl

    root = _worker(tmp_path)
    with open(root / ".build.lock", "a+") as holder, open(root / ".build.lock", "a+") as waiter:
        fcntl.flock(holder, fcntl.LOCK_EX)
        with pytest.raises(homekit_build.BuildError, match="Timed out waiting"):
            homekit_build._acquire(waiter, lambda _line: None, wait_seconds=0)


def test_auto_builder_is_inert_without_sources_or_when_disabled(tmp_path):
    root = _worker(tmp_path)
    disabled = homekit_build.AutoBuilder(root, runner=lambda *a, **k: pytest.fail("no build"), enabled=False)
    assert disabled.ensure(None) is False
    assert disabled.status()["available"] is False, "the UI must offer the manual command instead"
    missing = homekit_build.AutoBuilder(tmp_path / "absent", runner=lambda *a, **k: pytest.fail("no build"), enabled=True)
    assert missing.ensure(None) is False
