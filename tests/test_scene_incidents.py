"""Behavioral contract for complete, durable scene histories."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from survng.app.events import EventStore
from survng.app.event_store.scenes import SceneConflict


def person(x=10, **extra):
    return {"label":"person", "confidence":.751, "temporal_observations":5,
            "box":{"x1":x,"y1":10,"x2":x+10,"y2":40}, **extra}


class SceneIncidentTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.store=EventStore(self.root)

    def add(self,seconds=0,camera="upper-garage",objects=None,**kwargs):
        return self.store.add_event(camera,"motion",created_at=datetime.fromtimestamp(1780000000+seconds,timezone.utc).isoformat(),
                                    objects_json=json.dumps(objects if objects is not None else [person()]),**kwargs)

    def test_80913_two_people_in_roster_and_alert_separate(self):
        event=self.add(objects=[person(confidence=.941,incident_eligible=True),person(50,incident_eligible=False,incident_ineligible_reasons=["outside_incident_zone"])])
        incident=self.store.scene_incident(event_id=event["id"])
        self.assertEqual(incident["summary"],"Person observed")
        self.assertEqual(len(incident["scene_objects"]),2)
        self.assertEqual(incident["labels"],["person"])
        self.assertFalse(incident["alert_decisions"][0]["objects"][1]["eligible"])
        self.assertEqual(self.store.scene_pending_notifications()[0]["payload"]["scene_objects"],incident["scene_objects"])

    def test_cover_refinement_preserves_previous_object_and_source_image(self):
        snapshots=self.root/"snapshots";snapshots.mkdir()
        first=snapshots/"first.webp";first.write_bytes(b"evidence")
        second=snapshots/"second.webp";second.write_bytes(b"later evidence")
        event=self.add(snapshot_path=str(first))
        self.store.refine_event_evidence(event["id"],snapshot_path=str(second),recording_path="",
                                        objects_json=json.dumps([person(60,label="car")]))
        incident=self.store.scene_incident(event_id=event["id"])
        self.assertEqual(incident["labels"],["car","person"])
        self.assertNotIn("snapshot_observation_id", incident)
        self.assertTrue(first.exists())
        observed=next(s for s in incident["scene_objects"] if s["label"]=="person")
        self.assertEqual(self.store.scene_observation(observed["observations"][0]["id"])["snapshot_path"],"snapshots/first.webp")

    def test_empty_refined_cover_uses_retained_observation_image(self):
        from survng.app.incident_presenter import _incident_list_payload, _recording_grid_incident_payload

        snapshots = self.root / "snapshots"
        snapshots.mkdir()
        first = snapshots / "first.webp"
        first.write_bytes(b"original evidence")
        empty = snapshots / "empty.webp"
        empty.write_bytes(b"empty later frame")
        event = self.add(snapshot_path=str(first), objects=[person(
            label="car", detection_frame_width=640, detection_frame_height=366)])
        self.store.refine_event_evidence(event["id"], snapshot_path=str(empty),
                                        recording_path="", objects_json="[]")
        incident = self.store.scene_incident(event_id=event["id"])
        observation_id = incident["scene_objects"][0]["observations"][0]["id"]
        self.assertEqual(incident["snapshot_observation_id"], observation_id)
        self.assertEqual(incident["snapshot_url"], f"/api/incidents/observations/{observation_id}/snapshot")
        self.assertEqual(incident["objects"][0]["detection_frame_width"], 640)
        self.assertTrue(incident["objects"][0]["snapshot_visible"])
        self.assertIsNone(incident["object_tracking"])
        with self.store._connect() as conn:
            stored_path = conn.execute("select snapshot_path from events where id=?", (event["id"],)).fetchone()[0]
        self.assertEqual(stored_path, "snapshots/empty.webp")
        self.assertNotIn("snapshot_url", incident["events"][0])
        for projection in (self.store.list_scene_incidents()[0], _incident_list_payload(incident),
                           _recording_grid_incident_payload(incident)):
            self.assertEqual(projection["snapshot_url"], incident["snapshot_url"])
        card = self.store.list_scene_incident_cards()[0]
        self.assertEqual(card["snapshot_observation_id"], observation_id)
        self.assertEqual(card["snapshot_url"], incident["snapshot_url"])
        self.assertEqual(card["representative_event_id"], event["id"])
        self.assertEqual(card["objects"][0]["label"], "car")
        self.assertEqual(card["events"][0]["objects"], card["objects"])
        self.assertTrue(first.exists())
        self.assertTrue(empty.exists())

    def test_promoted_main_cover_is_not_replaced_by_a_smaller_discovery_image(self):
        snapshots = self.root / "snapshots"
        snapshots.mkdir()
        discovery = snapshots / "discovery.webp"
        discovery.write_bytes(b"low resolution")
        main = snapshots / "main.webp"
        main.write_bytes(b"high resolution")
        event = self.add(snapshot_path=str(discovery), objects=[person(
            frame_source="live_discovery", detection_frame_width=1280, detection_frame_height=720)])
        self.store.refine_event_evidence(event["id"], snapshot_path=str(main), recording_path="", objects_json=json.dumps([
            {"status": "scene_observations", "observations": [person(
                frame_source="recorded_main", detection_frame_width=2560, detection_frame_height=1440,
                snapshot_visible=False)]},
        ]))
        incident = self.store.scene_incident(event_id=event["id"])
        self.assertNotIn("snapshot_observation_id", incident)
        self.assertNotIn("/observations/", incident.get("snapshot_url") or "")

    def test_card_and_detail_choose_the_same_multi_event_cover(self):
        snapshots = self.root / "snapshots"
        snapshots.mkdir()
        first_path = snapshots / "first.webp"
        second_path = snapshots / "second.webp"
        first_path.write_bytes(b"first")
        second_path.write_bytes(b"second")
        first = self.add(snapshot_path=str(first_path), objects=[person(
            confidence=.7, detection_frame_width=640, detection_frame_height=360,
        )])
        second = self.add(10, camera="gate", snapshot_path=str(second_path), objects=[person(
            confidence=.95, detection_frame_width=1920, detection_frame_height=1080,
        )])
        a = self.store.scene_incident(event_id=first["id"])
        b = self.store.scene_incident(event_id=second["id"])
        self.store.correct_scene_incident(
            a["id"], a["revision"],
            {"operation": "merge", "incident_ids": [b["id"]],
             "expected_revisions": {b["id"]: b["revision"]}},
        )

        detail = self.store.scene_incident(event_id=first["id"])
        card = self.store.list_scene_incident_cards()[0]

        self.assertEqual(card["representative_event_id"], detail["representative_event_id"])
        self.assertEqual(card.get("snapshot_observation_id"), detail.get("snapshot_observation_id"))
        self.assertEqual(card.get("snapshot_url"), detail.get("snapshot_url"))
        self.assertEqual(card["camera_id"], detail["camera_id"])
        self.assertEqual(card["camera_id"], "gate")

    def test_existing_scene_object_schema_adds_facet_index_after_column(self):
        with self.store._connect() as connection:
            connection.execute("drop index if exists scene_object_facet_label")
            connection.execute("pragma foreign_keys=off")
            connection.execute(
                "create table scene_objects_legacy("
                "id text primary key,episode_id text not null references scene_episodes(id),"
                "label_override text,association_locked integer not null default 0)"
            )
            connection.execute(
                "insert into scene_objects_legacy select id,episode_id,label_override,association_locked "
                "from scene_objects"
            )
            connection.execute("drop table scene_objects")
            connection.execute("alter table scene_objects_legacy rename to scene_objects")

        restored = EventStore(self.root)
        with restored._connect() as connection:
            columns = {row[1] for row in connection.execute("pragma table_info(scene_objects)")}
            indexes = {row[1] for row in connection.execute("pragma index_list(scene_objects)")}
        self.assertIn("facet_label", columns)
        self.assertIn("scene_object_facet_label", indexes)

    def test_bounded_facets_include_whole_overlapping_incident(self):
        first = self.add(objects=[person(zones=["front"])])
        self.add(20, objects=[person(70, label="dog", zones=["side"])])
        self.add(100, camera="gate", objects=[person(label="car", zones=["road"])])
        incident = self.store.scene_incident(event_id=first["id"])
        subject = next(s for s in incident["scene_objects"] if s["label"] == "person")
        self.store.correct_scene_incident(incident["id"], incident["revision"],
            {"operation":"label", "object_id":subject["id"], "label":"cat"})
        self.assertEqual(self.store.scene_incident_facets(start_epoch=1780000010, end_epoch=1780000030),
                         {"camera_ids":["upper-garage"], "labels":["cat", "dog"], "zones":["front", "side"]})
        self.assertEqual(self.store.scene_incident_facets(start_epoch=1780000200),
                         {"camera_ids":[], "labels":[], "zones":[]})
        self.assertEqual(self.store.scene_incident_facets(),
                         {"camera_ids":["gate", "upper-garage"], "labels":["car", "cat", "dog"], "zones":["front", "road", "side"]})

    def test_incident_cards_skip_observation_history(self):
        event = self.add(snapshot_path="snapshots/cover.webp", objects=[person(zones=["front"])])
        incident = self.store.scene_incident(event_id=event["id"])
        card = self.store.list_scene_incident_cards()[0]
        self.assertEqual(card["id"], incident["id"])
        self.assertEqual(card["labels"], ["person"])
        self.assertEqual(card["revision"], incident["revision"])
        self.assertNotIn("scene_objects", card)
        self.assertEqual(card["events"][0]["id"], event["id"])
        self.assertEqual(card["snapshot_path"], "available")
        self.assertEqual(card["objects"], [{
            "label": "person",
            "box": {"x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 40.0},
            "confidence": 0.751,
            "zones": ["front"],
        }])
        self.assertEqual(card["events"][0]["objects"], card["objects"])
        again = self.store.scene_incident_facets(start_epoch=1780000000, end_epoch=1780000100)
        self.assertEqual(again["labels"], ["person"])
        self.assertEqual(again["zones"], ["front"])

    def test_temporal_batch_keeps_objects_absent_from_cover(self):
        event=self.add(objects=[person(),{"status":"scene_observations","observations":[person(),person(80,label="dog",offset_seconds=3,scene_track_key="t2"),person(82,label="dog",offset_seconds=4,scene_track_key="t2")]}])
        incident=self.store.scene_incident(event_id=event["id"])
        self.assertEqual(incident["labels"],["dog","person"])
        dog=next(s for s in incident["scene_objects"] if s["label"]=="dog")
        self.assertEqual(len(dog["observations"]),2)
        self.assertFalse(dog["observations"][0]["snapshot_available"])

    def test_replay_is_idempotent_and_restart_keeps_identity(self):
        event=self.add()
        before=self.store.scene_incident(event_id=event["id"])
        self.store.update_objects(event["id"],event["objects_json"])
        after=self.store.scene_incident(event_id=event["id"])
        self.assertEqual(before,after)
        restarted=EventStore(self.root)
        self.assertEqual(restarted.scene_incident(event_id=event["id"]),after)

    def test_legacy_event_links_resolve_whole_persistent_episode(self):
        first=self.add();second=self.add(20,objects=[person(60,label="dog")])
        incident=self.store.scene_incident(event_id=first["id"])
        self.assertEqual(incident["event_ids"],[first["id"],second["id"]])
        self.assertEqual(self.store.scene_incident(f"incident-upper-garage-{first['id']}")["id"],incident["id"])
        self.assertEqual(self.store.list_scene_incidents(start_epoch=1780000010)[0]["event_ids"],incident["event_ids"])

    def test_incident_metadata_batches_canonical_search_fields(self):
        first=self.add();second=self.add(20,objects=[person(60,label="dog")])
        incident=self.store.scene_incident(event_id=first["id"])

        metadata=self.store.scene_incident_metadata([second["id"],first["id"],999999])

        self.assertEqual(set(metadata),{first["id"],second["id"]})
        self.assertEqual(metadata[first["id"]]["incident_id"],incident["id"])
        self.assertEqual(metadata[second["id"]]["incident_id"],incident["id"])
        self.assertEqual(metadata[first["id"]]["establishment"],incident["establishment"])

    def test_merge_split_conflicts_and_history(self):
        first=self.add();second=self.add(20,camera="gate")
        a=self.store.scene_incident(event_id=first["id"]);b=self.store.scene_incident(event_id=second["id"])
        merged=self.store.correct_scene_incident(a["id"],a["revision"],{"operation":"merge","incident_ids":[b["id"]],"expected_revisions":{b["id"]:b["revision"]}})
        self.assertEqual(len(merged["camera_ids"]),2)
        self.assertEqual(self.store.scene_incident(b["id"])["id"],a["id"])
        with self.assertRaises(SceneConflict):
            self.store.correct_scene_incident(a["id"],a["revision"],{"operation":"split","episode_ids":[merged["episodes"][0]["id"]]})
        split=self.store.correct_scene_incident(a["id"],merged["revision"],{"operation":"split","episode_ids":[merged["episodes"][1]["id"]]})
        self.assertEqual(len(split["episodes"]),1)
        self.assertNotEqual(self.store.scene_incident(event_id=second["id"])["id"],a["id"])

    def test_label_correction_preserves_original_model_result(self):
        event=self.add();incident=self.store.scene_incident(event_id=event["id"])
        result=self.store.correct_scene_incident(incident["id"],incident["revision"],{"operation":"label","object_id":incident["scene_objects"][0]["id"],"label":"dog"})
        self.assertEqual(result["labels"],["dog"])
        self.assertEqual(result["scene_objects"][0]["observations"][0]["label"],"person")

    def test_unconfirmed_discovery_does_not_create_incident(self):
        event=self.add(objects=[person(), {"status":"motion_qualification", "motion_qualification":{"scene_discovery":True}}])
        self.assertIsNone(self.store.scene_incident(event_id=event["id"]))
        self.assertEqual(self.store.scene_pending_notifications(), [])
        restarted=EventStore(self.root)
        self.assertIsNone(restarted.scene_incident(event_id=event["id"]))

    def test_history_migration_is_silent_and_marks_missing_coverage(self):
        event=self.add(objects=[person(50,incident_eligible=False)])
        with self.store._connect() as conn:
            for table in ("acquired_sample_episodes", "scene_activity_admissions", "scene_event_establishment", "scene_event_membership", "scene_notification_outbox", "scene_observations", "scene_objects", "scene_episodes", "scene_incidents", "scene_aliases"):
                conn.execute("delete from " + table)
        restarted=EventStore(self.root)
        incident=restarted.scene_incident(event_id=event["id"])
        self.assertEqual(restarted.scene_pending_notifications(),[])
        self.assertEqual(incident["coverage"]["state"],"historical")

    def test_uncertain_single_frame_is_not_asserted_as_confirmed(self):
        event=self.add(objects=[person(confidence=.3,temporal_observations=1)])
        incident=self.store.scene_incident(event_id=event["id"])
        self.assertEqual(incident["scene_objects"][0]["certainty"],"possible")
        self.assertIn("possible person",incident["summary"].lower())

    def test_association_split_owns_independent_corrections(self):
        first=self.add();second=self.add(10,camera="gate")
        a=self.store.scene_incident(event_id=first["id"]);b=self.store.scene_incident(event_id=second["id"])
        merged=self.store.correct_scene_incident(a["id"],a["revision"],{"operation":"merge","incident_ids":[b["id"]],"expected_revisions":{b["id"]:b["revision"]}})
        associated=self.store.correct_scene_incident(merged["id"],merged["revision"],{"operation":"associate","object_ids":[s["id"] for s in merged["scene_objects"]]})
        split=self.store.correct_scene_incident(associated["id"],associated["revision"],{"operation":"split","episode_ids":[associated["episodes"][1]["id"]]})
        other=self.store.scene_incident(event_id=second["id"])
        self.assertNotEqual(split["scene_objects"][0]["id"],other["scene_objects"][0]["id"])
        self.store.correct_scene_incident(other["id"],other["revision"],{"operation":"label","object_id":other["scene_objects"][0]["id"],"label":"dog"})
        unchanged=self.store.scene_incident(split["id"])
        self.assertEqual(unchanged["labels"],["person"])
        self.assertEqual(unchanged["revision"],split["revision"])

    def test_notification_outbox_coalesces_to_latest_revision(self):
        event=self.add()
        incident=self.store.scene_incident(event_id=event["id"])
        self.store.scene_pending_notifications()
        with self.store._lock,self.store._connect() as conn:
            for _ in range(3):
                self.store._scene_changed(conn,incident["id"])
            rows=conn.execute("select revision,payload_json from scene_notification_outbox where incident_id=?",(incident["id"],)).fetchall()
        self.assertEqual([(r["revision"],r["payload_json"]) for r in rows],[(incident["revision"]+3,"")])
        pending=self.store.scene_pending_notifications()
        self.assertEqual([(p["incident_id"],p["revision"]) for p in pending],[(incident["id"],incident["revision"]+3)])
        self.assertEqual(pending[0]["payload"]["revision"],incident["revision"]+3)
        with self.store._lock,self.store._connect() as conn:
            self.store._scene_changed(conn,incident["id"])
        # Acknowledging the older snapshot keeps the change made after it.
        self.store.acknowledge_scene_notification(incident["id"],pending[0]["revision"])
        self.assertEqual([p["revision"] for p in self.store.scene_pending_notifications()],[incident["revision"]+4])
        self.store.acknowledge_scene_notification(incident["id"],incident["revision"]+4)
        self.assertEqual(self.store.scene_pending_notifications(),[])

    def test_legacy_outbox_rows_are_retired_and_unfinished_incidents_requeued(self):
        active=self.store.scene_incident(event_id=self.add()["id"])
        done=self.store.scene_incident(event_id=self.add(5000,camera="gate")["id"])
        legacy=json.dumps({"legacy":"x"*1000})
        with self.store._lock,self.store._connect() as conn:
            conn.execute("update scene_incidents set state='active' where id=?",(active["id"],))
            conn.execute("update scene_incidents set state='complete' where id=?",(done["id"],))
            conn.execute("delete from scene_notification_outbox")
            for incident in (active,done):
                for revision in range(1,incident["revision"]+1):
                    conn.execute("insert into scene_notification_outbox values(?,?,?)",(incident["id"],revision,legacy))
            conn.execute("delete from scene_migrations where name='notification_outbox_coalesce_v1'")
        restarted=EventStore(self.root)
        pending=restarted.scene_pending_notifications()
        self.assertEqual([(p["incident_id"],p["revision"]) for p in pending],[(active["id"],active["revision"])])
        self.assertNotIn("legacy",pending[0]["payload"])
        while restarted.purge_legacy_scene_notifications(limit=2):
            pass
        with restarted._connect() as conn:
            remaining=[tuple(r) for r in conn.execute("select incident_id,payload_json from scene_notification_outbox")]
        self.assertEqual(remaining,[(active["id"],"")])

    def test_sole_highest_legacy_row_cannot_hide_requeued_marker(self):
        active=self.store.scene_incident(event_id=self.add()["id"])
        with self.store._lock,self.store._connect() as conn:
            conn.execute("update scene_incidents set state='active' where id=?",(active["id"],))
            conn.execute("delete from scene_notification_outbox")
            conn.execute("insert into scene_notification_outbox values(?,?,?)",
                         (active["id"],active["revision"],json.dumps({"legacy":True})))
            conn.execute("delete from scene_migrations where name='notification_outbox_coalesce_v1'")
        restarted=EventStore(self.root)
        pending=restarted.scene_pending_notifications()
        self.assertEqual([(item["incident_id"],item["revision"]) for item in pending],
                         [(active["id"],active["revision"])])
        self.assertEqual(restarted.purge_legacy_scene_notifications(),0)
        self.assertEqual(len(restarted.scene_pending_notifications()),1)

    def test_merged_alias_never_publishes_mismatched_revision(self):
        first=self.add();second=self.add(10,camera="gate")
        a=self.store.scene_incident(event_id=first["id"]);b=self.store.scene_incident(event_id=second["id"])
        self.store.correct_scene_incident(a["id"],a["revision"],{"operation":"merge","incident_ids":[b["id"]],"expected_revisions":{b["id"]:b["revision"]}})
        self.store.settle_scene_incidents(1780000100)
        for entry in self.store.scene_pending_notifications():
            self.assertEqual(entry["incident_id"],entry["payload"]["id"])
            self.assertEqual(entry["revision"],entry["payload"]["revision"])

    def test_confirmed_moving_identity_connects_cameras_but_stationary_does_not(self):
        first=self.add(objects=[person(temporal_center_displacement_ratio=.1)])
        second=self.add(15,camera="gate",objects=[person(temporal_center_displacement_ratio=.1)])
        from tests.scene_activity_helpers import record_person_motion
        record_person_motion(self.store,first['id'])
        record_person_motion(self.store,second['id'])
        identity=[{"identity_id":7,"status":"confirmed","name":"Resident"}]
        self.store.update_scene_identities(first["id"],identity)
        self.store.update_scene_identities(second["id"],identity)
        linked=self.store.scene_incident(event_id=first["id"])
        self.assertEqual(set(linked["camera_ids"]),{"upper-garage","gate"})
        stationary=self.add(30,camera="foyer")
        self.store.update_scene_identities(stationary["id"],identity)
        self.assertNotEqual(self.store.scene_incident(event_id=stationary["id"])["id"],linked["id"])

    def test_stationary_discovery_keeps_one_object_and_inactivity_boundary(self):
        event=self.add();initial=self.store.scene_incident(event_id=event["id"])
        self.store.record_scene_observations(event["id"],[person(captured_at_epoch=1780000010)],activity=False)
        after=self.store.scene_incident(event_id=event["id"])
        self.assertEqual(len(after["scene_objects"]),1)
        self.assertEqual(after["episodes"][0]["last_activity_at"],initial["episodes"][0]["last_activity_at"])



if __name__=="__main__":
    unittest.main()
