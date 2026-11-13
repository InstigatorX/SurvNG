"""Current writers round-trip without compatibility rewriting."""
import json
import os
import subprocess
import sys
from datetime import datetime

import pytest

from survng.app.config import (
    AppConfig, CameraMotionQualificationConfig, DetectorConfig,
    MotionQualificationConfig, ObjectTrackingConfig, PropertyZoneOverlay,
    load_config, save_config,
)
from survng.app.config_init import initialize_config
from survng.app.config_secrets import secrets_key_path, secrets_path
from survng.app.event_store import EventStore
from survng.app.incident_utils import portable_media_path
from survng.app.manager import validate_motion_pipeline_configuration
from survng.app.object_track.registry import build_builtin_object_tracker_registry
from survng.app.recorder import Recorder


@pytest.mark.parametrize('mode', ['off', 'audit', 'enforce'])
@pytest.mark.parametrize('model', [MotionQualificationConfig, CameraMotionQualificationConfig])
def test_retired_modes_are_rejected(model, mode):
    with pytest.raises(ValueError):
        model(mode=mode)


@pytest.mark.parametrize('model', [MotionQualificationConfig, CameraMotionQualificationConfig])
@pytest.mark.parametrize('value', [True, False, None])
def test_retired_mog2_field_is_rejected(model, value):
    with pytest.raises(ValueError, match='mog2_audit_enabled'):
        model(mog2_audit_enabled=value)


def test_removed_fields_and_rectangle_zones_are_rejected():
    with pytest.raises(ValueError, match='temporal_filter_threshold'):
        MotionQualificationConfig(temporal_filter_threshold=.5)
    with pytest.raises(ValueError):
        PropertyZoneOverlay(camera_id='gate', zone_name='yard', x=.5, y=.5, width=.2, height=.1)


@pytest.mark.parametrize('implementation', ['bytetrack', 'survng_hybrid_candidate', 'ultralytics_tracktrack', 'ultralytics_botsort', 'ultralytics_deepocsort', 'ultralytics_fasttrack'])
def test_retired_trackers_are_rejected(implementation):
    with pytest.raises(ValueError):
        ObjectTrackingConfig(implementation=implementation)
    with pytest.raises(ValueError):
        build_builtin_object_tracker_registry().require(implementation)


@pytest.mark.parametrize('implementation', ['survng_sparse_identity', 'survng_hybrid'])
def test_current_trackers_are_available(implementation):
    config = ObjectTrackingConfig(implementation=implementation)
    assert build_builtin_object_tracker_registry().create(implementation, config, .7)


@pytest.mark.parametrize('implementation', ['legacy_qualifier', 'legacy_motion_scorer', 'score_event_state', 'score_trigger', 'depth_object_evidence'])
def test_retired_graphs_are_not_rewritten(implementation):
    config = AppConfig.model_validate({'motion_qualification': {'pipeline': {'qualification': [{'stage_id': 'old', 'implementation': implementation}]}}})
    with pytest.raises(ValueError):
        validate_motion_pipeline_configuration(config)


def test_face_profile_rename_has_no_alias():
    assert DetectorConfig().face_embedding_profile == 'openvino_raw'
    with pytest.raises(ValueError):
        DetectorConfig(face_embedding_profile='legacy_openvino')


def test_initialize_creates_private_artifacts_and_applies_overrides(tmp_path):
    template = tmp_path / 'template.json'
    template.write_text(json.dumps({'mqtt': {'password': 'template-secret'}}))
    destination = tmp_path / 'installation' / 'config.json'
    initialize_config(template, destination, storage_dir=str(tmp_path / 'media'), ffmpeg_path='/opt/ffmpeg')
    config = load_config(destination)
    assert config.mqtt.password == 'template-secret'
    assert 'template-secret' not in destination.read_text()
    assert config.ffmpeg_path == '/opt/ffmpeg'
    assert config.storage_dir == config.media_storage.locations[0].path == str(tmp_path / 'media')
    artifacts = (destination, secrets_path(destination), secrets_key_path(secrets_path(destination)))
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in artifacts)
    before = [path.read_bytes() for path in artifacts]
    with pytest.raises(FileExistsError):
        initialize_config(template, destination)
    assert before == [path.read_bytes() for path in artifacts]


@pytest.mark.parametrize('name', ['config.json', 'secrets.json', 'secrets.key'])
def test_initializer_refuses_any_existing_artifact(tmp_path, name):
    template = tmp_path / 'template.json'
    template.write_text('{}')
    existing = tmp_path / name
    existing.write_text('keep')
    with pytest.raises(FileExistsError):
        initialize_config(template, tmp_path / 'config.json')
    assert existing.read_text() == 'keep'


def test_initializer_cli_honors_secret_path_overrides(tmp_path):
    destination = tmp_path / 'config.json'
    sidecar = tmp_path / 'private' / 'credentials.json'
    key = tmp_path / 'keys' / 'credentials.key'
    env = {**os.environ, 'SURVNG_SECRETS_PATH': str(sidecar), 'SURVNG_SECRETS_KEY_PATH': str(key)}
    subprocess.run([sys.executable, '-m', 'survng.app.config_init', '--template', 'docker/config.example.json', '--config', str(destination)], env=env, check=True)
    assert destination.is_file() and sidecar.is_file() and key.is_file()


def test_loader_rejects_inline_secrets_even_with_sidecar(tmp_path):
    destination = tmp_path / 'config.json'
    save_config(AppConfig(), destination)
    raw = json.loads(destination.read_text())
    raw['mqtt']['password'] = 'never-log-this'
    destination.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='inline secrets') as error:
        load_config(destination)
    assert 'never-log-this' not in str(error.value)


def test_current_database_reopens_without_migration_tables(tmp_path):
    store = EventStore(tmp_path)
    event = store.add_event(camera_id='gate', kind='motion', topic='motion', message='motion', snapshot_path='', recording_path='')
    reopened = EventStore(tmp_path)
    assert reopened.get(event['id'])
    with reopened._connect() as connection:
        tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
        assert not tables & {'scene_migrations', 'scene_acquisition_migrations', 'tracking_comparisons'}
    assert reopened.scene_incident(event_id=event['id'])


def test_current_recording_discovery_ignores_old_layout(tmp_path):
    recorder = Recorder('ffmpeg', tmp_path)
    for relative in ('gate/main/2026-11-12/10/current.mp4', 'gate/live/2026-11-12/10/live.mp4', 'gate/2026-11-12/10/old.mp4'):
        path = recorder.recordings_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'video')
    grouped = recorder._recording_files_by_source()
    assert {key: [path.name for path in paths] for key, paths in grouped.items()} == {('gate', 'main'): ['current.mp4'], ('gate', 'live'): ['live.mp4']}
    assert recorder._recording_search_dirs('gate', 'main') == [recorder.recordings_dir / 'gate' / 'main']


def test_portable_paths_do_not_guess_previous_mounts(tmp_path):
    current = tmp_path / 'snapshots' / 'gate' / 'image.jpg'
    current.parent.mkdir(parents=True)
    current.write_bytes(b'image')
    assert portable_media_path(tmp_path, current) == 'snapshots/gate/image.jpg'
    old = '/previous-mount/snapshots/gate/image.jpg'
    assert portable_media_path(tmp_path, old) == old


def test_comparison_routes_are_absent():
    from survng.app.main import app
    assert not any('tracking-comparison' in getattr(route, 'path', '') for route in app.routes)


def test_recent_recording_scan_ignores_old_layout(tmp_path):
    recorder = Recorder('ffmpeg', tmp_path)
    hour = datetime.now().strftime('%Y-%m-%d/%H')
    for relative in (f'gate/main/{hour}/current.mp4', f'gate/{hour}/old.mp4'):
        path = recorder.recordings_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'video')
    grouped = recorder._recent_recording_files_by_source(hours=1)
    assert [path.name for path in grouped[('gate', 'main')]] == ['current.mp4']


@pytest.mark.parametrize('schema_version', [None, 1, 2])
def test_notifications_reject_unsupported_versions(schema_version):
    from survng.app.manager import AppManager
    manager = object.__new__(AppManager)
    manager.config = AppConfig()
    payload = {'schema_version': schema_version, 'classes': ['person'],
               'alert_decisions': [{'objects': [{'eligible': True}]}]}
    assert not manager.incident_notification_allowed(payload)
    assert not manager.incident_notification_payload(payload)['notifications_enabled']


def test_initializer_preserves_explicit_camera_ids(tmp_path):
    template = tmp_path / 'template.json'
    template.write_text(json.dumps({'cameras': [{'id': 'gate', 'name': 'Entrance', 'stream_url': 'rtsp://camera/main'}]}))
    destination = tmp_path / 'config.json'
    initialize_config(template, destination)
    assert load_config(destination).cameras[0].id == 'gate'


def test_initializer_rejects_redacted_template_before_writing(tmp_path):
    template = tmp_path / 'template.json'
    template.write_text(json.dumps({'mqtt': {'password': '__SURVNG_SECRET_SET__'}}))
    destination = tmp_path / 'config.json'
    with pytest.raises(ValueError, match='redacted secrets'):
        initialize_config(template, destination)
    assert not destination.exists()
    assert not secrets_path(destination).exists()
    assert not secrets_key_path(secrets_path(destination)).exists()
