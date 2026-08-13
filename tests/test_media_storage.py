from pathlib import Path

import pytest

from survng.app.config import MediaStorageConfig, MediaStorageLocationConfig
from survng.app.media_storage import MediaStorageRegistry


def test_empty_config_preserves_legacy_storage_root(tmp_path: Path) -> None:
    registry = MediaStorageRegistry(tmp_path, MediaStorageConfig())

    assert registry.location_ids == ("default",)
    assert registry.directory("snapshots", "gate", "gate") == tmp_path / "snapshots" / "gate"


def test_role_selection_uses_only_eligible_locations(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings-disk"
    evidence = tmp_path / "evidence-disk"
    recordings.mkdir()
    evidence.mkdir()
    registry = MediaStorageRegistry(tmp_path, MediaStorageConfig(locations=[
        MediaStorageLocationConfig(id="recordings", path=str(recordings), roles=["recordings"]),
        MediaStorageLocationConfig(id="evidence", path=str(evidence), roles=["snapshots", "clips"]),
    ]))

    assert registry.directory("recordings", "gate:main:2026-08-13") == recordings / "recordings"
    assert registry.directory("snapshots", "gate", "gate") == evidence / "snapshots" / "gate"


def test_required_missing_mount_is_never_selected(tmp_path: Path) -> None:
    path = tmp_path / "ordinary-directory"
    path.mkdir()
    registry = MediaStorageRegistry(tmp_path, MediaStorageConfig(locations=[
        MediaStorageLocationConfig(id="nfs", path=str(path), require_mount=True),
    ]))

    assert registry.status("nfs").state == "not_mounted"
    with pytest.raises(OSError, match="no writable media location"):
        registry.choose("recordings", "gate")


def test_configuration_rejects_duplicate_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="paths must be unique"):
        MediaStorageConfig(locations=[
            MediaStorageLocationConfig(id="one", path=str(tmp_path)),
            MediaStorageLocationConfig(id="two", path=str(tmp_path / ".")),
        ])
