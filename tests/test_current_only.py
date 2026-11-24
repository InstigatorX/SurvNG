"""Current writers round-trip without compatibility rewriting."""
import json
import os
import sqlite3
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


def _schema(path):
    with sqlite3.connect(path) as conn:
        return sorted(conn.execute('select type, name, sql from sqlite_master'))


def test_event_database_older_than_1_2_0_stops_before_changing_it(tmp_path):
    from survng.app.main_database import UnsupportedDatabaseError

    # A current database whose scene_objects table lacks the column v1.2.0 added,
    # plus a table startup would otherwise drop.
    EventStore(tmp_path)
    database = tmp_path / 'survng.sqlite3'
    with sqlite3.connect(database) as conn:
        conn.execute('pragma foreign_keys=off')
        conn.execute('drop index scene_object_facet_label')
        conn.execute('alter table scene_objects drop column facet_label')
        conn.execute('create table scene_discovery_baselines(camera_id text primary key)')
    before = _schema(database)
    with pytest.raises(UnsupportedDatabaseError, match='scene_objects predates SurvNG 1.2.0'):
        EventStore(tmp_path)
    assert _schema(database) == before


def test_v1_2_0_event_index_leftover_is_dropped(tmp_path):
    EventStore(tmp_path)
    with sqlite3.connect(tmp_path / 'survng.sqlite3') as conn:
        conn.execute('create index idx_event_camera_epoch on events(camera_id, julianday(created_at), id)')
    reopened = EventStore(tmp_path)
    with reopened._connect() as conn:
        names = {row[0] for row in conn.execute("select name from sqlite_master where type='index'")}
    assert 'idx_event_camera_epoch' not in names


def test_recording_index_older_than_1_2_0_stops_before_changing_it(tmp_path):
    from survng.app.main_database import UnsupportedDatabaseError
    from survng.app.recording_process.recorder import Recorder

    with sqlite3.connect(tmp_path / 'recordings.sqlite3') as conn:
        conn.execute('create table recordings(path text primary key, camera_id text not null, source text not null, '
                     'name text not null, size_bytes integer not null, modified_at real not null, '
                     'start_epoch real not null, duration_seconds real not null, end_epoch real not null)')
    before = _schema(tmp_path / 'recordings.sqlite3')
    with pytest.raises(UnsupportedDatabaseError, match='recordings predates SurvNG 1.2.0'):
        Recorder('ffmpeg', tmp_path)
    assert _schema(tmp_path / 'recordings.sqlite3') == before


def test_manager_precheck_rejects_a_later_store_table_before_any_change(tmp_path):
    from survng.app.main_database import UnsupportedDatabaseError
    from survng.app.schema_minimum import require_minimum_databases
    from survng.app.semantic_search import SemanticIndex

    EventStore(tmp_path)
    database = tmp_path / 'survng.sqlite3'
    SemanticIndex(database)
    with sqlite3.connect(database) as conn:
        dependent = [name for (name,) in conn.execute(
            "select name from sqlite_master where type='index' and tbl_name='semantic_embeddings' "
            "and sql like '%evidence_revision%'")]
        for name in dependent:
            conn.execute(f'drop index {name}')
        conn.execute('alter table semantic_embeddings drop column evidence_revision')
        conn.execute('create table scene_discovery_baselines(camera_id text primary key)')
    before = _schema(database)
    with pytest.raises(UnsupportedDatabaseError, match='semantic_embeddings predates SurvNG 1.2.0'):
        require_minimum_databases(tmp_path, tmp_path)
    assert _schema(database) == before
    # The event store shares the file, so it refuses the same database itself.
    with pytest.raises(UnsupportedDatabaseError, match='semantic_embeddings predates SurvNG 1.2.0'):
        EventStore(tmp_path)
    assert _schema(database) == before
    require_minimum_databases(tmp_path / 'new', tmp_path / 'new')  # nothing yet: passes, creates nothing
    assert not (tmp_path / 'new').exists()


def test_manager_checks_databases_before_building_any_store(tmp_path):
    from unittest.mock import patch

    from survng.app.main_database import UnsupportedDatabaseError
    from survng.app.manager import AppManager

    stop = UnsupportedDatabaseError('old database')
    with (
        patch('survng.app.manager.require_minimum_databases', side_effect=stop) as precheck,
        patch('survng.app.manager.EventStore', side_effect=AssertionError('store built first')),
        pytest.raises(UnsupportedDatabaseError),
    ):
        AppManager(AppConfig(storage_dir=str(tmp_path)))
    precheck.assert_called_once_with(tmp_path, tmp_path)


def test_retired_scene_changes_history_is_kept_but_no_longer_written(tmp_path):
    EventStore(tmp_path)
    with sqlite3.connect(tmp_path / 'survng.sqlite3') as conn:
        conn.execute('create table scene_changes(id integer primary key autoincrement, incident_id text not null, '
                     'revision integer not null, created_at real not null)')
        conn.execute("insert into scene_changes(incident_id, revision, created_at) values ('old', 1, 0)")
    store = EventStore(tmp_path)
    event = store.add_event(camera_id='gate', kind='motion', topic='motion', message='motion', snapshot_path='', recording_path='')
    assert store.scene_incident(event_id=event['id'])
    with sqlite3.connect(tmp_path / 'survng.sqlite3') as conn:
        assert conn.execute('select incident_id from scene_changes').fetchall() == [('old',)]


def test_startup_reports_a_manager_failure_instead_of_a_telemetry_failure():
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from survng.app import main
    from survng.app.main_database import UnsupportedDatabaseError

    for failure in (UnsupportedDatabaseError('old database'), sqlite3.OperationalError('unreadable')):
        with patch.object(main, 'get_manager', side_effect=failure):
            with pytest.raises(type(failure)):
                main._record_process_lifecycle('startup_started')
    # A failed telemetry write alone stays non-fatal.
    telemetry = Mock()
    telemetry.record_lifecycle_event.side_effect = sqlite3.OperationalError('locked')
    with patch.object(main, 'get_manager', return_value=SimpleNamespace(telemetry=telemetry)):
        main._record_process_lifecycle('startup_started')


def test_event_store_refusing_an_old_jobs_database_creates_no_database(tmp_path):
    from survng.app.main_database import UnsupportedDatabaseError

    conn = sqlite3.connect(tmp_path / 'detection-jobs.sqlite3')
    conn.execute('pragma journal_mode=wal')
    conn.execute('create table detection_jobs(id text primary key, camera_id text not null)')
    conn.commit()
    conn.close()
    before = _schema(tmp_path / 'detection-jobs.sqlite3')
    with pytest.raises(UnsupportedDatabaseError, match='detection_jobs predates SurvNG 1.2.0'):
        EventStore(tmp_path)
    # SQLite may add WAL side files to the refused database; no new database appears.
    assert {path.name for path in tmp_path.iterdir()} <= {
        'detection-jobs.sqlite3', 'detection-jobs.sqlite3-wal', 'detection-jobs.sqlite3-shm',
    }
    assert _schema(tmp_path / 'detection-jobs.sqlite3') == before
