from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from survng.app.ffmpeg_process import named_ffmpeg_executable
from survng.app.runtime_directory import prepare_private_directory


def test_private_runtime_creation_and_legacy_permission_repair(tmp_path: Path) -> None:
    directory = tmp_path / "survng"
    previous_umask = os.umask(0o022)
    try:
        prepare_private_directory(directory, repair_mode=True)
    finally:
        os.umask(previous_umask)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    marker = directory / "existing"
    marker.write_text("preserve")
    directory.chmod(0o755)
    prepare_private_directory(directory, repair_mode=True)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert marker.read_text() == "preserve"


def test_runtime_setup_refuses_symlink_without_changing_target(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    target.chmod(0o755)
    alias = tmp_path / "survng"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError):
        prepare_private_directory(alias, repair_mode=True)
    assert alias.is_symlink()
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
    assert named_ffmpeg_executable("sh", "survng-test", runtime_dir=alias) == "sh"
    assert not (target / "survng-test").exists()


def test_runtime_setup_refuses_foreign_owner_without_chmod(tmp_path: Path, monkeypatch) -> None:
    directory = tmp_path / "survng"
    directory.mkdir()
    directory.chmod(0o755)
    monkeypatch.setattr("survng.app.runtime_directory.os.geteuid", lambda: directory.stat().st_uid + 1)
    with pytest.raises(PermissionError, match="not owned by this service"):
        prepare_private_directory(directory, repair_mode=True)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o755
    assert named_ffmpeg_executable("sh", "survng-test", runtime_dir=directory) == "sh"
    assert not (directory / "survng-test").exists()


def test_runtime_setup_does_not_replace_regular_file(tmp_path: Path) -> None:
    directory = tmp_path / "survng"
    directory.write_text("keep")
    with pytest.raises(OSError):
        prepare_private_directory(directory, repair_mode=True)
    assert directory.read_text() == "keep"
