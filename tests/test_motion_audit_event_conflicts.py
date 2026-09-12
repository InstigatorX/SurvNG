"""Several decisions can resolve to one incident without losing audit identity."""
import pytest
from survng.app.events import EventStore


@pytest.mark.parametrize('existing_decision', [False, True])
@pytest.mark.parametrize('owner_decision', ['', 'other-decision'])
def test_refined_decision_links_to_already_audited_event(tmp_path, existing_decision, owner_decision):
    store = EventStore(tmp_path)
    base = dict(camera_id='gate', snapshot_path='', created_at='2026-09-12T12:00:00+00:00',
                mode='camera_rescue', sensitivity='balanced', score=.8, threshold=.5,
                reason='visual_backup_trigger', object_detected=True, trigger_count=1,
                features={}, category='visual_backup')
    original = None
    if existing_decision:
        original = store.add_motion_audit(**{**base, 'object_detected': None}, decision_id='qualified')
    event = store.add_event(camera_id='gate', kind='motion', objects_json='[]')
    owner = store.add_motion_audit(**base, event_id=event['id'], decision_id=owner_decision)
    completed = store.add_motion_audit(**base, event_id=event['id'], decision_id='qualified')
    retry = store.add_motion_audit(**base, event_id=event['id'], decision_id='qualified')
    assert completed['id'] == retry['id']
    if original:
        assert completed['id'] == original['id']
    assert completed['id'] != owner['id']
    assert completed['decision_id'] == 'qualified'
    assert completed['event_id'] is None
    assert completed['related_event_id'] == event['id']
    assert completed['object_detected'] == 1
    assert store.get_motion_audit(owner['id'])['event_id'] == event['id']
    assert [a['id'] for a in store.motion_audits_for_related_events([event['id']])] == [completed['id']]
    rows, total = store.motion_audits(camera_id='gate', include_incident_activity=True)
    assert total == 2
