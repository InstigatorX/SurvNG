// Incident detail ("story") read models: title, storyline lanes, object
// cards, cross-camera paths and review-mode keys. Pure functions over the
// scene payload, so the view and its tests share one source of truth.
import { incidentSceneObjects } from "./incidentScene.mjs";

const VEHICLES = new Set(["car", "truck", "bus", "motorcycle", "van", "vehicle", "boat", "bicycle", "suv", "pickup"]);

export function epochOf(value) {
  if (typeof value === "number") return value;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed / 1000 : NaN;
}

export function isUncertainObject(object) {
  return ["possible", "uncertain"].includes(object?.certainty);
}

// Color family for bars, boxes and cards. Uncertainty wins over the label.
export function objectKind(object) {
  if (isUncertainObject(object)) return "uncertain";
  const label = String(object?.label || "").toLowerCase();
  if (label === "person") return "person";
  if (VEHICLES.has(label)) return "vehicle";
  return "other";
}

export function objectName(object) {
  const label = String(object?.label || "object").replaceAll("_", " ");
  return label.charAt(0).toUpperCase() + label.slice(1);
}

export function formatSpan(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  if (total < 60) return `${total} s`;
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  if (hours) return `${hours} h${minutes ? ` ${minutes} min` : ""}`;
  return `${minutes} min${rest ? ` ${rest} s` : ""}`;
}

// Server intervals when present. Older payloads (and historical event-only
// incidents) fall back to one observed span per camera; that is coarser, so
// it never claims a stationary stretch.
export function objectIntervals(object) {
  if (Array.isArray(object?.visible_intervals) && object.visible_intervals.length) {
    return object.visible_intervals.map((interval) => ({ ...interval, start: epochOf(interval.start_at), end: epochOf(interval.end_at) }))
      .filter((interval) => Number.isFinite(interval.start) && Number.isFinite(interval.end));
  }
  const byCamera = new Map();
  for (const observation of object?.observations || []) {
    const epoch = epochOf(observation.captured_at);
    if (!Number.isFinite(epoch)) continue;
    const key = observation.camera_id || "";
    const current = byCamera.get(key) || { camera_id: key, episode_id: null, start: epoch, end: epoch, state: "observed", observation_id: observation.id, last_observation_id: observation.id, zones: [] };
    if (epoch < current.start) Object.assign(current, { start: epoch, observation_id: observation.id });
    if (epoch > current.end) Object.assign(current, { end: epoch, last_observation_id: observation.id });
    for (const zone of observation.zones || []) if (!current.zones.includes(zone)) current.zones.push(zone);
    byCamera.set(key, current);
  }
  if (!byCamera.size && Number.isFinite(epochOf(object?.first_seen_at))) {
    return [{ camera_id: object.camera_id || "", episode_id: null, start: epochOf(object.first_seen_at), end: epochOf(object.last_seen_at || object.first_seen_at), state: "observed", zones: [] }];
  }
  return [...byCamera.values()].sort((left, right) => left.start - right.start);
}

const KIND_ORDER = { person: 0, vehicle: 1, other: 2, uncertain: 3 };

// Card order: people, vehicles, other objects, then uncertain ones; each by first sighting.
export function cardOrder(objects) {
  return [...objects].sort((left, right) => (KIND_ORDER[left.kind] - KIND_ORDER[right.kind])
    || ((left.intervals[0]?.start ?? Infinity) - (right.intervals[0]?.start ?? Infinity)));
}

export function storyObjects(incident) {
  return incidentSceneObjects(incident)
    .filter((object) => String(object?.label || "").toLowerCase() !== "face")
    .map((object) => ({ ...object, kind: objectKind(object), intervals: objectIntervals(object) }))
    .sort((left, right) => (left.intervals[0]?.start ?? Infinity) - (right.intervals[0]?.start ?? Infinity));
}

// Lanes follow camera episodes. A second episode on the same camera gets its
// own lane, labelled with its start so the two can be told apart.
export function storyline(incident, objects = storyObjects(incident)) {
  const episodes = [...(incident?.episodes || [])]
    .map((episode) => ({ ...episode, start: epochOf(episode.start_at), end: epochOf(episode.end_at) }))
    .filter((episode) => Number.isFinite(episode.start))
    .sort((left, right) => left.start - right.start);
  const lanes = episodes.map((episode) => ({ key: episode.id, episodeId: episode.id, cameraId: episode.camera_id, start: episode.start, end: Number.isFinite(episode.end) ? episode.end : episode.start, coverage: episode.coverage || null, bars: [] }));
  const laneFor = (interval) => {
    let lane = lanes.find((candidate) => interval.episode_id && candidate.episodeId === interval.episode_id)
      || lanes.find((candidate) => candidate.cameraId === interval.camera_id && interval.start >= candidate.start - 1 && interval.start <= candidate.end + 1)
      || lanes.find((candidate) => candidate.cameraId === interval.camera_id);
    if (!lane) {
      lane = { key: `camera-${interval.camera_id}`, episodeId: null, cameraId: interval.camera_id, start: interval.start, end: interval.end, coverage: null, bars: [] };
      lanes.push(lane);
    }
    return lane;
  };
  for (const object of objects) {
    for (const interval of object.intervals) {
      laneFor(interval).bars.push({ objectId: object.id, kind: object.kind, label: object.label, uncertain: object.kind === "uncertain", ...interval });
    }
  }
  const times = [
    epochOf(incident?.start_at), epochOf(incident?.end_at),
    ...lanes.flatMap((lane) => [lane.start, lane.end, ...lane.bars.flatMap((bar) => [bar.start, bar.end])]),
  ].filter(Number.isFinite);
  const start = times.length ? Math.min(...times) : 0;
  const end = times.length ? Math.max(start + 1, ...times) : 1;
  const sameCamera = new Map();
  for (const lane of lanes) sameCamera.set(lane.cameraId, (sameCamera.get(lane.cameraId) || 0) + 1);
  for (const lane of lanes) {
    lane.repeated = sameCamera.get(lane.cameraId) > 1;
    lane.bars.sort((left, right) => left.start - right.start);
    // Stack bars of different objects that overlap in time into rows.
    const rows = [];
    for (const bar of lane.bars) {
      let row = rows.findIndex((items) => items.every((item) => item.objectId === bar.objectId || item.end + 0.5 < bar.start || bar.end + 0.5 < item.start));
      if (row < 0) { rows.push([]); row = rows.length - 1; }
      rows[row].push(bar);
      bar.row = row;
    }
    lane.rows = Math.max(1, rows.length);
  }
  lanes.sort((left, right) => left.start - right.start);
  return { start, end, lanes };
}

export function timelineTicks(start, end, count = 5) {
  if (!(end > start)) return [{ epoch: start, fraction: 0 }];
  return Array.from({ length: count }, (_, index) => ({ epoch: start + (end - start) * index / (count - 1), fraction: index / (count - 1) }));
}

export function fraction(epoch, start, end) {
  if (!(end > start) || !Number.isFinite(epoch)) return 0;
  return Math.max(0, Math.min(1, (epoch - start) / (end - start)));
}

function cameraLabel(cameraId, cameraName) {
  return cameraName?.(cameraId) || cameraId || "camera";
}

// Steps of one object across cameras: consecutive intervals on one camera
// collapse into one step; association notes describe the hand-off.
export function objectPath(object, cameraName) {
  const steps = [];
  for (const interval of object?.intervals || objectIntervals(object)) {
    const last = steps[steps.length - 1];
    if (last && last.cameraId === interval.camera_id) {
      last.end = Math.max(last.end, interval.end);
      for (const zone of interval.zones || []) if (!last.zones.includes(zone)) last.zones.push(zone);
      last.intervals.push(interval);
      continue;
    }
    steps.push({ cameraId: interval.camera_id, camera: cameraLabel(interval.camera_id, cameraName), start: interval.start, end: interval.end, zones: [...(interval.zones || [])], intervals: [interval] });
  }
  return steps.map((step, index) => {
    const next = steps[index + 1];
    let note = "";
    if (next) {
      const delta = next.start - step.end;
      const timing = delta <= 0 ? `${formatSpan(-delta)} overlap` : `${formatSpan(delta)} gap`;
      note = `${timing} · ${object?.association_corrected ? "associated by a correction" : "associated automatically"}`;
      if (object?.continuity_uncertain) note += " · may be another sighting";
    } else if (steps.length > 1 || index === 0) {
      note = steps.length > 1 ? "Left view. Not seen on another camera." : "Not seen on another camera.";
    }
    return { ...step, number: index + 1, note };
  });
}

function spanSeconds(intervals) {
  if (!intervals.length) return 0;
  return Math.max(...intervals.map((item) => item.end)) - Math.min(...intervals.map((item) => item.start));
}

// Identities are recorded per incident, not per object, so names go on a
// person card only when there is exactly one person. With none recognized,
// every person card can say so.
export function cardIdentities(objects, identities) {
  const list = Array.isArray(identities) ? identities : [];
  const people = (objects || []).filter((object) => object.kind === "person").length;
  return people === 1 || !list.length ? list : null;
}

export function objectSummary(object, { cameraName, identities = null } = {}) {
  const intervals = object?.intervals || objectIntervals(object);
  const cameras = [...new Set(intervals.map((item) => item.camera_id))];
  const where = cameras.length === 1 ? cameraLabel(cameras[0], cameraName) : `${cameras.length} cameras`;
  const states = intervals.map((item) => item.state);
  const seen = `seen ${formatSpan(spanSeconds(intervals))}`;
  if (object?.kind === "person" || (!object?.kind && object?.label === "person")) {
    const names = (identities || []).map((item) => item?.name).filter(Boolean);
    const who = names.length ? `Recognized: ${names.join(", ")}` : Array.isArray(identities) ? "Not recognized" : "";
    return [who, who ? seen : `Seen ${formatSpan(spanSeconds(intervals))}`].filter(Boolean).join(" · ");
  }
  let motion = "";
  if (states.length > 1 && states[0] === "moving" && states[states.length - 1] === "stationary") motion = "Arrived, then stationary";
  else if (states.length > 1 && states[0] === "stationary" && states[states.length - 1] === "moving") motion = "Stationary, then moved";
  else if (states.length && states.every((state) => state === "stationary")) motion = "Stationary throughout";
  return [motion || `Seen ${formatSpan(spanSeconds(intervals))}`, where].join(" · ");
}

function clause(object, cameraName, person = "") {
  const intervals = object.intervals;
  const label = object.kind === "person" && person ? person : String(object.label || "object").replaceAll("_", " ");
  const cameras = [...new Set(intervals.map((item) => item.camera_id))];
  const states = intervals.map((item) => item.state);
  const lastCamera = cameraLabel(cameras[cameras.length - 1], cameraName);
  const zones = [...new Set(intervals.flatMap((item) => item.zones || []))];
  if (cameras.length > 1) return `${label} ${object.kind === "person" ? "walks" : object.kind === "vehicle" ? "drives" : "moves"} to ${lastCamera}`;
  if (states[0] === "moving" && states[states.length - 1] === "stationary") return `${label} arrives`;
  if (states.length && states.every((state) => state === "stationary")) return `${label} stationary at ${lastCamera}`;
  return zones.length ? `${label} in ${zones[0]}` : `${label} at ${lastCamera}`;
}

// Deterministic plain-language title from objects, cameras and zones.
// A single recognized name stands in for the one person; an existing summary
// is kept for incidents without observed objects (motion-only, historical).
export function incidentStoryTitle(incident, { cameraName, objects = storyObjects(incident), people = [], fallback = "" } = {}) {
  const certain = objects.filter((object) => object.kind !== "uncertain" && object.intervals.length);
  const persons = certain.filter((object) => object.kind === "person");
  const name = people.length === 1 && persons.length === 1 ? String(people[0]) : "";
  let text;
  if (certain.length) {
    text = certain.slice(0, 2).map((object) => clause(object, cameraName, name)).join(", ");
    if (certain.length > 2) text += ` and ${certain.length - 2} more`;
  } else if (objects.length) {
    const object = objects[0];
    const zone = object.intervals.flatMap((item) => item.zones || [])[0];
    text = `possible ${String(object.label || "object").replaceAll("_", " ")} ${zone ? `in ${zone}` : `at ${cameraLabel(object.intervals[0]?.camera_id || incident?.camera_id, cameraName)}`}`;
  } else if (fallback) {
    return fallback;
  } else {
    text = `motion at ${cameraLabel(incident?.camera_id, cameraName)}`;
  }
  return text.charAt(0).toUpperCase() + text.slice(1);
}

// Playback plan that follows one object: one segment per path step,
// padded a little so the hand-off is visible, clamped to the incident.
export function followSegments(object, { start, end, pad = 2 } = {}) {
  return objectPath(object).map((step) => ({
    cameraId: step.cameraId,
    start: Math.max(Number.isFinite(start) ? start : -Infinity, step.start - pad),
    end: Math.min(Number.isFinite(end) ? end : Infinity, Math.max(step.end + pad, step.start + 1)),
  }));
}

// The observation of an object closest to a moment, optionally on one camera.
export function nearestObservation(object, epoch, cameraId = null) {
  let best = null;
  for (const observation of object?.observations || []) {
    if (cameraId && observation.camera_id !== cameraId) continue;
    const distance = Math.abs(epochOf(observation.captured_at) - epoch);
    if (!Number.isFinite(distance)) continue;
    const rank = [distance, observation.snapshot_available === false ? 1 : 0];
    if (!best || rank[0] < best.rank[0] - 0.01 || (Math.abs(rank[0] - best.rank[0]) <= 0.01 && rank[1] < best.rank[1])) best = { observation, rank };
  }
  return best?.observation || null;
}

export function objectBestObservation(object) {
  const observations = object?.observations || [];
  return [...observations].sort((left, right) => (Number(right.snapshot_available !== false) - Number(left.snapshot_available !== false))
    || (Number(right.confidence || 0) - Number(left.confidence || 0)))[0] || null;
}

// Every object's box in the same captured frame as ``anchor``.
export function frameBoxes(objects, anchor) {
  if (!anchor) return [];
  const epoch = epochOf(anchor.captured_at);
  return objects.flatMap((object) => (object.observations || [])
    .filter((observation) => observation.camera_id === anchor.camera_id && Math.abs(epochOf(observation.captured_at) - epoch) < 0.05 && observation.box)
    .map((observation) => ({ object, observation })));
}

export function findObservation(objects, observationId) {
  for (const object of objects) {
    const observation = (object.observations || []).find((item) => String(item.id) === String(observationId));
    if (observation) return { object, observation };
  }
  return null;
}

export const REVIEW_KEYS = Object.freeze({ e: "reviewed", f: "keep", x: "nothing_here", j: "next", k: "previous" });

// Review shortcuts never fire while typing or with a modifier held.
export function reviewKeyAction(event) {
  if (!event || event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.repeat) return null;
  const target = event.target;
  const tag = String(target?.tagName || "").toLowerCase();
  if (target?.isContentEditable || ["input", "textarea", "select"].includes(tag) || target?.closest?.("dialog[open]")) return null;
  return REVIEW_KEYS[String(event.key || "").toLowerCase()] || null;
}

// Position and next targets in the day queue (list order, newest first).
export function reviewPosition(queue, incidentId) {
  const items = queue?.items || [];
  const index = items.findIndex((item) => String(item.incident_id) === String(incidentId));
  const decided = items.filter((item) => item.verdict).length;
  const afterCurrent = index >= 0 ? [...items.slice(index + 1), ...items.slice(0, index)] : items;
  return {
    index, total: items.length, decided,
    remaining: items.filter((item) => !item.verdict).length,
    previous: index > 0 ? items[index - 1].incident_id : null,
    next: index >= 0 && index < items.length - 1 ? items[index + 1].incident_id : null,
    nextUndecided: afterCurrent.find((item) => !item.verdict && String(item.incident_id) !== String(incidentId))?.incident_id || null,
  };
}

export function establishmentReasonText(incident, cameraName) {
  const establishment = incident?.establishment;
  const zones = [...new Set((incident?.zones || []).map(String))];
  const cameras = [...new Set((incident?.episodes || []).map((episode) => episode.camera_id))];
  const first = cameraLabel(cameras[0] || incident?.camera_id, cameraName);
  if (!establishment?.status) return "This incident predates activity establishment; its retained observations are shown as recorded.";
  if (establishment.status !== "established") return null;
  const where = zones.length ? `in the ${zones[0]} zone on ${first}` : `on ${first}`;
  const joined = cameras.slice(1).map((id) => cameraLabel(id, cameraName));
  const tail = joined.length ? ` ${joined.join(", ")} joined because ${joined.length > 1 ? "their" : "its"} activity overlapped in time.` : "";
  return `${establishment.summary || "Physical activity confirmed"}${establishment.summary ? "" : ` ${where}`}.${tail}`.replace("..", ".");
}

export function observationSnapshotUrl(observation) {
  if (!observation) return "";
  if (observation.snapshot_url) return observation.snapshot_url;
  return observation.id && observation.snapshot_available !== false ? `/api/incidents/observations/${encodeURIComponent(observation.id)}/snapshot` : "";
}

// Background crop of a detection, expanded to the thumbnail's aspect so the
// subject is not stretched. Returns null when the box cannot be placed.
export function cropBackgroundStyle(observation, aspect = 1, padding = 0.25) {
  const width = Number(observation?.detection_frame_width), height = Number(observation?.detection_frame_height);
  const box = observation?.box;
  if (!(width > 0 && height > 0) || !box) return null;
  const [x1, y1, x2, y2] = [box.x1, box.y1, box.x2, box.y2].map(Number);
  if (![x1, y1, x2, y2].every(Number.isFinite) || x2 <= x1 || y2 <= y1) return null;
  let w = (x2 - x1) * (1 + 2 * padding), h = (y2 - y1) * (1 + 2 * padding);
  if (w / h < aspect) w = h * aspect; else h = w / aspect;
  w = Math.min(w, width); h = Math.min(h, height);
  const left = Math.max(0, Math.min(width - w, (x1 + x2) / 2 - w / 2));
  const top = Math.max(0, Math.min(height - h, (y1 + y2) / 2 - h / 2));
  const fw = w / width, fh = h / height;
  return {
    backgroundSize: `${100 / fw}% ${100 / fh}%`,
    backgroundPosition: `${fw < 1 ? (left / width) / (1 - fw) * 100 : 0}% ${fh < 1 ? (top / height) / (1 - fh) * 100 : 0}%`,
  };
}
