export function storylineIncidentId(incident) {
  return String(incident.incident_id || incident.id);
}

export function toggleStorylineIncident(selected, incident) {
  const id = storylineIncidentId(incident);
  if (selected.some((item) => storylineIncidentId(item) === id)) return selected.filter((item) => storylineIncidentId(item) !== id);
  return selected.length < 64 ? [...selected, incident] : selected;
}

export function storylineMembers(selected) {
  return selected.map((incident) => ({ incident_id: storylineIncidentId(incident), subject_ids: [], relationship: "related_event", note: "" }));
}
