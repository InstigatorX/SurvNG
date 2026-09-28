"""All observed people participate in identity ambiguity checks."""
from survng.app.identity_projection import apply_event_identity


def test_excluded_second_person_prevents_guessing_face_to_body_assignment():
    people = [
        {"label": "person", "confidence": .94, "incident_eligible": True},
        {"label": "person", "confidence": .75, "incident_eligible": False},
    ]
    event = {"objects": people, "faces": [{"identity_id": 7, "name": "Alex", "status": "confirmed"}]}
    apply_event_identity(event)
    assert event["identities"][0]["name"] == "Alex"
    assert all("identity" not in person for person in people)


def test_single_observed_person_can_receive_identity_even_if_not_alert_eligible():
    person = {"label": "person", "incident_eligible": False}
    event = {"objects": [person], "faces": [{"identity_id": 7, "name": "Alex", "status": "confirmed"}]}
    apply_event_identity(event)
    assert person["identity"]["name"] == "Alex"
