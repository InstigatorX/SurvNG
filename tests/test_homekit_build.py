"""HomeKit worker builder and the freshness gate that keeps stale bundles from being offered."""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from survng.app import homekit, homekit_build, homekit_nodes

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


def test_build_activates_a_verified_bundle_and_skips_when_current(tmp_path):
    root = _worker(tmp_path)
    assert homekit_build.verify(root) == homekit_build.NOT_BUILT
    toolchain = FakeToolchain()

    assert _build(root, toolchain) is True
    assert homekit_build.verify(root) is None
    manifest = homekit_build.read_manifest(root)
    assert manifest["inputs"] == homekit_build.input_fingerprint(root)
    assert not list(root.glob(".build-*"))
    # Packages install from the lockfile in staging, never in the live tree.
    assert all(Path(command[0]).name != "npm" or command[1] == "ci" for command in toolchain.commands)

    again = FakeToolchain()
    assert _build(root, again) is False
    assert again.commands == []


def test_bundle_without_manifest_is_never_offered(tmp_path):
    """A build left over from another checkout (the reported incident) is stale."""
    root = _worker(tmp_path)
    (root / "dist" / "src").mkdir(parents=True)
    (root / "dist" / "src" / "agent.js").write_text("old")
    (root / "dist" / "src" / "camera.js").write_text("old")
    (root / "node_modules" / "werift").mkdir(parents=True)
    (root / "node_modules" / "werift" / "package.json").write_text('{"version":"0.24.4"}')

    assert homekit_build.verify(root) == homekit_build.STALE
    with pytest.raises(RuntimeError, match=homekit_build.STALE):
        homekit_nodes.agent_release(root)
    assert "agent" not in homekit_nodes.node_releases("/no/such/ffmpeg", root, lib_dir=tmp_path / "none")


def test_source_change_blocks_the_release_until_rebuilt(tmp_path, monkeypatch):
    root = _worker(tmp_path)
    _build(root, FakeToolchain())
    monkeypatch.setattr(homekit_nodes, "_agent_cache", None)
    before = homekit_nodes.agent_release(root)[0]

    (root / "src" / "agent.ts").write_text("export const release = 2;\n")
    assert homekit_build.verify(root) == homekit_build.STALE
    with pytest.raises(RuntimeError, match=homekit_build.STALE):
        homekit_nodes.agent_release(root)
    assert homekit_nodes.available_agent_version(root) is None

    assert _build(root, FakeToolchain()) is True
    after = homekit_nodes.agent_release(root)[0]
    assert after != before
    assert "release = 2" in (root / "dist" / "src" / "agent.js").read_text()


@pytest.mark.parametrize("step", ["ci", "build"])
def test_failed_build_keeps_the_previous_bundle_blocked_and_retry_recovers(tmp_path, step):
    root = _worker(tmp_path)
    _build(root, FakeToolchain())
    (root / "src" / "agent.ts").write_text("export const release = 2;\n")

    with pytest.raises(homekit_build.BuildError, match=f"{step} exploded" if step == "ci" else "build.mjs failed"):
        _build(root, FakeToolchain(fail=step))
    # The live bundle is untouched, and the freshness gate still blocks it.
    assert "release = 1" in (root / "dist" / "src" / "agent.js").read_text()
    assert homekit_build.verify(root) == homekit_build.STALE
    assert not list(root.glob(".build-*"))

    assert _build(root, FakeToolchain()) is True
    assert homekit_build.verify(root) is None


def test_installed_package_drift_blocks_the_release(tmp_path):
    root = _worker(tmp_path)
    _build(root, FakeToolchain())
    (root / "node_modules" / "werift" / "package.json").write_text('{"version":"0.24.4"}')
    assert homekit_build.verify(root) == homekit_build.DEPENDENCIES_STALE
    with pytest.raises(RuntimeError, match=homekit_build.DEPENDENCIES_STALE):
        homekit_nodes.agent_release(root)


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
    assert toolchain.commands[0][-2:] == [str(root / "scripts" / "bootstrap.py"), "--hap-only"]
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


def test_local_camera_worker_is_blocked_by_a_stale_build(tmp_path, monkeypatch):
    root = _worker(tmp_path)
    monkeypatch.setattr(homekit, "WORKER_ROOT", root)
    monkeypatch.setattr(homekit, "WORKER", root / "dist" / "src" / "camera.js")
    assert homekit.worker_build_problem() == homekit_build.NOT_BUILT
    _build(root, FakeToolchain())
    assert homekit.worker_build_problem() is None
    (root / "package-lock.json").write_text(json.dumps(_lock(werift="0.29.0")))
    assert homekit.worker_build_problem() == homekit_build.STALE


def test_blocked_camera_worker_resumes_after_a_rebuild(tmp_path, monkeypatch):
    from types import SimpleNamespace

    states: list[dict] = []
    checks = iter([homekit_build.STALE, homekit_build.STALE, None])
    worker = SimpleNamespace(
        service=SimpleNamespace(directory=tmp_path / "state"), path=tmp_path / "state" / "camera",
        stopping=SimpleNamespace(wait=lambda _delay: False),
        config=SimpleNamespace(homekit=SimpleNamespace(bind="bad address")),
        _state=lambda **values: states.append(values), _terminate=lambda: None)
    monkeypatch.setattr(homekit, "worker_build_problem", lambda: next(checks))
    homekit.CameraWorker._run(worker)
    # Blocked twice on the stale build, then proceeded to the next preflight step.
    assert states[:2] == [{"state": "blocked", "error": homekit_build.STALE}] * 2
    assert states[-1] == {"state": "blocked", "error": "bind_address_unavailable"}
