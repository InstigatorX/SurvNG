import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from survng.app.config import (
    AppConfig,
    MediaStorageConfig,
    MediaStorageLocationConfig,
    primary_media_storage,
)
from survng.app.media_storage import MediaStorageRegistry


def test_primary_location_uses_configured_media_root(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    registry = MediaStorageRegistry(tmp_path, primary_media_storage(str(media_root)))

    assert registry.location_ids == ("primary",)
    assert registry.directory("snapshots", "gate", "gate") == media_root / "snapshots" / "gate"


def test_empty_locations_are_rejected() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        MediaStorageConfig(locations=[])


def test_app_config_seeds_primary_location_from_storage_dir(tmp_path: Path) -> None:
    config = AppConfig(storage_dir=str(tmp_path))

    assert len(config.media_storage.locations) == 1
    assert config.media_storage.locations[0].id == "primary"
    assert config.media_storage.locations[0].path == str(tmp_path)


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


def test_required_mount_accepts_configured_directory_below_mountpoint(tmp_path: Path) -> None:
    mountpoint = tmp_path / "media1"
    path = mountpoint / "SurvNG"
    path.mkdir(parents=True)
    registry = MediaStorageRegistry(tmp_path, MediaStorageConfig(locations=[
        MediaStorageLocationConfig(id="nfs", path=str(path), require_mount=True),
    ]))

    with patch(
        "survng.app.media_storage.os.path.ismount",
        side_effect=lambda candidate: Path(candidate) == mountpoint,
    ):
        assert registry.status("nfs").state == "online"


def test_required_mount_rejects_path_backed_only_by_root_filesystem(tmp_path: Path) -> None:
    path = tmp_path / "SurvNG"
    path.mkdir()
    registry = MediaStorageRegistry(tmp_path, MediaStorageConfig(locations=[
        MediaStorageLocationConfig(id="nfs", path=str(path), require_mount=True),
    ]))

    with patch(
        "survng.app.media_storage.os.path.ismount",
        side_effect=lambda candidate: Path(candidate) == Path(candidate).anchor,
    ):
        assert registry.status("nfs").state == "not_mounted"


def test_selection_error_explains_unwritable_location(tmp_path: Path) -> None:
    path = tmp_path / "media1"
    path.mkdir()
    registry = MediaStorageRegistry(tmp_path, MediaStorageConfig(locations=[
        MediaStorageLocationConfig(id="media1", name="Media 1", path=str(path)),
    ]))

    with patch("survng.app.media_storage.os.access", return_value=False):
        with pytest.raises(OSError, match="Media 1: directory is not writable"):
            registry.choose("snapshots", "gate")


def test_configuration_rejects_duplicate_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="paths must be unique"):
        MediaStorageConfig(locations=[
            MediaStorageLocationConfig(id="one", path=str(tmp_path)),
            MediaStorageLocationConfig(id="two", path=str(tmp_path / ".")),
        ])


def test_static_topology_lookups_do_not_probe_capacity(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    registry = MediaStorageRegistry(tmp_path, MediaStorageConfig(locations=[
        MediaStorageLocationConfig(
            id="first",
            path=str(first),
            roles=["recordings", "snapshots"],
        ),
        MediaStorageLocationConfig(
            id="second",
            path=str(second),
            roles=["snapshots"],
        ),
    ]))

    with patch(
        "survng.app.media_storage.shutil.disk_usage",
        side_effect=AssertionError("static topology lookup probed storage health"),
    ):
        assert registry.configured_roots_for("recordings") == [first / "recordings"]
        assert registry.roots_for("snapshots") == [
            first / "snapshots",
            second / "snapshots",
        ]
        assert registry.contains(first / "recordings" / "gate" / "segment.mp4", "recordings")
        assert not registry.contains(second / "recordings" / "gate" / "segment.mp4", "recordings")
        assert registry.location_id_for(first / "snapshots" / "gate" / "event.webp", "snapshots") == "first"


def test_health_snapshot_reuses_samples_and_reports_shared_filesystem(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    registry = MediaStorageRegistry(tmp_path, MediaStorageConfig(locations=[
        MediaStorageLocationConfig(id="first", path=str(first)),
        MediaStorageLocationConfig(id="second", path=str(second)),
    ]))

    with patch(
        "survng.app.media_storage.shutil.disk_usage",
        wraps=shutil.disk_usage,
    ) as disk_usage:
        snapshot = registry.health_snapshot()
        assert disk_usage.call_count == 2

        assert registry.status("first", snapshot=snapshot).id == "first"
        assert [item.id for item in registry.statuses(snapshot=snapshot)] == ["first", "second"]
        assert registry.roots_for("recordings", writable_only=True, snapshot=snapshot) == [
            first / "recordings",
            second / "recordings",
        ]
        payload = registry.payload(snapshot=snapshot)
        assert disk_usage.call_count == 2

    assert snapshot.shared_filesystems == (("first", "second"),)
    assert payload["shared_filesystems"] == [["first", "second"]]
    assert all(item["filesystem_id"] for item in payload["locations"])


def test_balanced_placement_distributes_equal_cold_start_assignments(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    registry = MediaStorageRegistry(tmp_path, MediaStorageConfig(
        placement="balanced",
        locations=[
            MediaStorageLocationConfig(id="first", path=str(first)),
            MediaStorageLocationConfig(id="second", path=str(second)),
        ],
    ))

    selected = {
        registry.choose("recordings", f"camera-{index}").id
        for index in range(32)
    }

    assert selected == {"first", "second"}
