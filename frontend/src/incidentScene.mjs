// Scene membership is independent of notification eligibility and the selected frame.
export function sceneObjectLabel(object) {
  const label = object?.label || "object";
  return ["possible", "uncertain"].includes(object?.certainty) ? `Possible ${label}` : label;
}

export function sceneObjectDetection(object) {
  const label = String(object?.label || "object").replaceAll("_", " ");
  const direct = Number(object?.confidence);
  const observed = (object?.observations || []).map((item) => Number(item?.confidence)).filter(Number.isFinite);
  const confidence = Number.isFinite(direct) && direct > 0 ? direct : Math.max(0, ...observed);
  return { label, confidence: Number.isFinite(confidence) ? confidence : 0 };
}

function median(values) {
  const sorted = values.filter(Number.isFinite).sort((left, right) => left - right);
  if (!sorted.length) return 0;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function detectionConfidences(object) {
  const observed = (object?.observations || []).map((item) => Number(item?.confidence)).filter(Number.isFinite);
  if (observed.length) return observed;
  const direct = Number(object?.confidence);
  return Number.isFinite(direct) ? [direct] : [];
}

export function observedObjectSummaries(objects) {
  const groups = [];
  const byLabel = new Map();
  for (const object of objects || []) {
    const label = String(object?.label || "object").replaceAll("_", " ");
    let group = byLabel.get(label);
    if (!group) {
      group = { label, confidences: [] };
      byLabel.set(label, group);
      groups.push(group);
    }
    group.confidences.push(...detectionConfidences(object));
  }
  return groups.map((group) => ({ label: group.label, confidence: median(group.confidences) }));
}

export function sceneEstablishmentText(establishment) {
  const status = establishment?.status;
  if (!status) return null;
  const states = {
    established: ["Activity established", "The retained evidence supports an episode of activity."],
    unverified: ["Historical activity is unverified", "The retained evidence has not established whether an episode of activity occurred."],
    not_established: ["Activity was not established", "This link preserves the earlier incident record and its observations. These observations do not establish an incident."],
    incomplete: ["Activity analysis is incomplete", "Available evidence is insufficient to finish evaluating activity. This does not establish that nothing happened."],
  };
  const [title, explanation] = states[status] || states.unverified;
  return { title, explanation, summary: establishment.summary || "", status };
}

export function observationImageDescription(image) {
  const width = Number(image?.width), height = Number(image?.height);
  const dimensions = width > 0 && height > 0 ? `${width} × ${height}` : "Dimensions unknown";
  const source = image?.source ? String(image.source).replaceAll("_", " ") : "Source unknown";
  const role = image?.analyzed_frame === true ? "Analyzed frame" : image?.analyzed_frame === false ? "Additional image; not the analyzed frame" : "Analysis provenance unknown";
  return `${dimensions} · ${source} · ${role}`;
}

export function incidentSceneObjects(incident) {
  if (Array.isArray(incident?.scene_objects)) return incident.scene_objects;
  const events = incident?.events?.length ? incident.events : incident ? [incident] : [];
  return events.flatMap((event) => (event.objects || []).map((object, index) => object?.label ? ({
    ...object,
    id: `historical-${event.id}-${index}`,
    certainty: object.certainty || "uncertain",
    source_event_id: event.id,
    object_index: index,
    first_seen_at: event.created_at,
    last_seen_at: event.created_at,
    observations: [{ ...object, event_id: event.id, camera_id: event.camera_id, captured_at: event.created_at, snapshot_available: false }],
  }) : null).filter(Boolean));
}

export function sceneCoverageText(coverage) {
  if (!coverage || coverage.state === "historical") return "Historical evidence: analysis coverage is unknown.";
  if (coverage.state === "incomplete" || coverage.gaps?.length) return "Analysis is incomplete. Objects may be missing from the available evidence.";
  return "Sampled footage: observations do not establish everything that was present.";
}

export function observationBoxStyle(observation) {
  const width = Number(observation?.detection_frame_width);
  const height = Number(observation?.detection_frame_height);
  const box = observation?.box;
  if (!(width > 0 && height > 0) || !box) return null;
  const values = [box.x1, box.y1, box.x2, box.y2].map(Number);
  if (!values.every(Number.isFinite) || values[2] <= values[0] || values[3] <= values[1]) return null;
  const [x1, y1, x2, y2] = values;
  return { left: `${100 * x1 / width}%`, top: `${100 * y1 / height}%`, width: `${100 * (x2 - x1) / width}%`, height: `${100 * (y2 - y1) / height}%` };
}

export function canonicalIncidentHref(incident) {
  const id = incident?.incident_id;
  return id ? `/review?incident_id=${encodeURIComponent(id)}` : `/review?event_ids=${encodeURIComponent(incident?.representative_event_id || "")}`;
}

export function incidentEpisodeClips(incident) {
  if (!incident) return [];
  const events = incident.events?.length ? incident.events : [incident];
  const episodes = incident.episodes?.length ? incident.episodes : [{
    id: `episode-${incident.id}`, camera_id: incident.camera_id,
    start_at: incident.start_at || (Number.isFinite(incident.start_epoch) ? new Date(incident.start_epoch * 1000).toISOString() : incident.created_at),
    end_at: incident.end_at || (Number.isFinite(incident.last_epoch) ? new Date(incident.last_epoch * 1000).toISOString() : incident.created_at),
    event_ids: events.map((event) => event.id), coverage: incident.coverage,
  }];
  const canonical = Boolean(incident.episodes?.length);
  return [...episodes].sort((left, right) => Date.parse(left.start_at) - Date.parse(right.start_at)).flatMap((episode) => {
    const start = Date.parse(episode.start_at) / 1000;
    const end = Math.max(start + 1, Date.parse(episode.end_at) / 1000);
    const members = events.filter((event) => (
      (event.camera_id || incident.camera_id) === episode.camera_id
      && (episode.event_ids?.length ? episode.event_ids.some((id) => String(id) === String(event.id))
        : Date.parse(event.created_at) / 1000 >= start && Date.parse(event.created_at) / 1000 <= end)
    )).sort((left, right) => Date.parse(left.created_at) - Date.parse(right.created_at));
    const anchor = members[0];
    if (!anchor || !Number.isFinite(start) || !Number.isFinite(end)) return [{ ...episode, episode_id: episode.id, clip: null, part_index: 0 }];
    const anchorEpoch = Date.parse(anchor.created_at) / 1000;
    if (!canonical && (anchorEpoch - start > 3600 || end - anchorEpoch > 3600)) return [{ ...episode, episode_id: episode.id, clip: null, part_index: 0, playback_note: "This historical episode is available in the recording timeline." }];
    const count = canonical ? Math.ceil((end - start) / 900) : 1;
    return Array.from({ length: count }, (_, index) => {
      const clipStart = canonical ? start + index * 900 : start;
      const clipEnd = canonical ? Math.min(end, clipStart + 900) : end;
      return {
        ...episode, id: index === 0 ? episode.id : `${episode.id}:part-${index}`, episode_id: episode.id, part_index: index,
        clip: {
          ...anchor, camera_id: episode.camera_id, representative_event_id: Number(anchor.id), events: members,
          start_at: new Date(clipStart * 1000).toISOString(), end_at: new Date(clipEnd * 1000).toISOString(), start_epoch: clipStart, last_epoch: clipEnd,
          scene_clip_window: { ...(canonical ? { episode_id: episode.id } : {}), start: clipStart, end: clipEnd },
        },
      };
    });
  }).sort((left, right) => (left.clip?.start_epoch ?? Date.parse(left.start_at) / 1000) - (right.clip?.start_epoch ?? Date.parse(right.start_at) / 1000));
}

export function incidentEpisodeMediaUrl(url, event) {
  const bounds = event?.scene_clip_window;
  if (!bounds?.episode_id) return url;
  const params = new URLSearchParams({ episode_id: bounds.episode_id, start_epoch: String(bounds.start), end_epoch: String(bounds.end) });
  return `${url}${url.includes("?") ? "&" : "?"}${params}`;
}

export function sceneActivityText(activity) {
  const label = activity?.label || "Object";
  if (activity.kind === "appeared") return `${label} first observed`;
  if (activity.kind === "last_seen") return `${label} last observed`;
  if (activity.kind === "changed_position") return `${label} changed position`;
  if (activity.kind === "zone_changed") return `${label} zone changed: ${(activity.from_zones || []).join(", ") || "outside named zones"} → ${(activity.to_zones || []).join(", ") || "outside named zones"}`;
  return `${label} observed`;
}

export function sceneNotificationSummary(decisions) {
  if (!Array.isArray(decisions) || !decisions.length) return "Notification policy has not been evaluated for this incident.";
  return decisions.some((decision) => decision.eligible || decision.objects?.some((object) => object.eligible))
    ? "Some observations met the notification criteria."
    : "No observations met the notification criteria.";
}
