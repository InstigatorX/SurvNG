"""Updater integration contract using harmless Git and Docker executables."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("selected", "target"),
    [
        (["compose.yaml"], None),
        (["compose.yaml", "compose.intel-gpu.yaml"], None),
        (["compose.yaml", "compose.lxc.yaml"], "runtime"),
        (["compose.yaml", "compose.intel-gpu.yaml", "compose.lxc.yaml"], "runtime-intel"),
        (["compose.yaml", "custom storage.yaml"], None),
    ],
)
def test_update_preserves_running_compose_files(tmp_path, selected, target):
    result, calls, root = _run_update(tmp_path, selected)
    assert result.returncode == 0, result.stderr
    up = next(line for line in calls if " up -d " in line)
    expected_files = " ".join(f"-f {root / name}" for name in selected)
    assert up.startswith(f"compose {expected_files} up -d ")
    if target:
        assert any(f"--target {target} " in line for line in calls)
        assert any("--privileged" in line for line in calls)
    else:
        assert not any("--privileged" in line or "buildx" in line for line in calls)
        assert any(line.startswith(f"compose {expected_files} build --pull ") for line in calls)
    # An existing but unused storage override must never be added by the updater.
    assert "compose.storage.yaml" not in up


@pytest.mark.parametrize("label", ["", "<no value>", "/missing/compose.yaml"])
def test_update_refuses_unknown_deployment_files(tmp_path, label):
    result, calls, _root = _run_update(tmp_path, [], label=label)
    assert result.returncode != 0
    assert not any(" up " in line or " build " in line or "--privileged" in line for line in calls)


def _run_update(tmp_path, selected, *, label=None):
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "checkout with spaces"
    for directory in (root / ".git", root / "scripts", root / "bin"):
        directory.mkdir(parents=True, exist_ok=True)
    for name in ("update-from-git.sh", "docker-build-lxc.sh"):
        shutil.copy2(repo / "scripts" / name, root / "scripts" / name)
    for name in ("compose.yaml", "compose.intel-gpu.yaml", "compose.lxc.yaml", "compose.storage.yaml", "custom storage.yaml"):
        (root / name).write_text("services: {}\n")
    log = root / "commands.log"
    (root / "bin" / "git").write_text('''#!/bin/sh
case "$*" in
  'rev-parse --abbrev-ref HEAD') echo v1.2;;
  'rev-list --left-right --count HEAD...origin/v1.2') printf '0\\t1\\n';;
  'rev-parse --short HEAD'|'rev-parse --short origin/v1.2'|'rev-parse HEAD') echo 1111111;;
  *) exit 0;;
esac
''')
    (root / "bin" / "docker").write_text('''#!/bin/sh
printf '%s\\n' "$*" >> "$UPDATE_TEST_LOG"
case "$*" in
  'compose -f compose.yaml ps --status running -q survng') echo test-container;;
  'inspect --format '*) printf '%s\\n' "$UPDATE_TEST_FILES";;
  'container inspect survng-buildkit') exit 1;;
  *) exit 0;;
esac
''')
    for executable in (root / "bin").iterdir():
        executable.chmod(0o700)
    environment = dict(os.environ, PATH=f"{root / 'bin'}:/usr/bin:/bin", UPDATE_TEST_LOG=str(log),
                       UPDATE_TEST_FILES=label if label is not None else ",".join(str(root / name) for name in selected))
    environment.pop("SURVNG_UPDATE_BRANCH", None)
    environment.pop("SURVNG_UPDATE_REMOTE", None)
    result = subprocess.run(["/bin/bash", str(root / "scripts" / "update-from-git.sh")],
                            cwd=root, env=environment, capture_output=True, text=True, timeout=20)
    return result, log.read_text().splitlines(), root
