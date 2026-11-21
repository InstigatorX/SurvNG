// Pure helpers for the vertical Event Rail, the 24 h edge rail, camera lanes
// and the condensed catch-up story. Times are epoch seconds.

export const RAIL_SPANS = [
  { minutes: 15, label: "15m" },
  { minutes: 60, label: "1h" },
  { minutes: 360, label: "6h" },
  { minutes: 1440, label: "24h" },
];

export const TIMELINE_VIEWS = ["rail", "cameras", "catchup", "classic"];

export function normalizedRailMinutes(value) {
  const minutes = Number(value);
  return RAIL_SPANS.some((span) => span.minutes === minutes) ? minutes : 60;
}

export function normalizedTimelineView(value) {
  return TIMELINE_VIEWS.includes(value) ? value : "rail";
}

const VEHICLE_LABELS = new Set(["vehicle", "car", "truck", "bus", "motorcycle", "bicycle", "van"]);

/** "person", "vehicle", "object" (another detection) or "motion" (no detection). */
export function railEventKind(event) {
  if (!event?.has_objects) return "motion";
  const labels = (Array.isArray(event.labels) ? event.labels : []).map((label) => String(label).toLocaleLowerCase());
  if (labels.some((label) => label === "person" || label === "people")) return "person";
  if (labels.some((label) => VEHICLE_LABELS.has(label))) return "vehicle";
  return "object";
}

function titleCase(value) {
  const text = String(value || "").replace(/[_-]+/g, " ").trim();
  return text ? text[0].toLocaleUpperCase() + text.slice(1) : "";
}

/** A short human title: identity name, else the distinct labels, else Motion. */
export function railEventTitle(event) {
  const identity = event?.primary_identity;
  const name = identity && typeof identity === "object" ? (identity.display_name || identity.name || identity.label) : null;
  if (name) return String(name);
  const labels = [...new Set((Array.isArray(event?.labels) ? event.labels : []).map((label) => String(label).toLocaleLowerCase()))];
  if (!event?.has_objects || !labels.length) return "Motion";
  return labels.slice(0, 2).map(titleCase).join(" · ") + (labels.length > 2 ? ` +${labels.length - 2}` : "");
}

export function railEventDurationSeconds(event) {
  const start = Number(event?.incident_epoch ?? event?.start_epoch);
  const end = Number(event?.last_epoch);
  return Number.isFinite(start) && Number.isFinite(end) && end > start ? end - start : 0;
}

/** Tick spacing that keeps labelled ticks at least `minPixels` apart. */
export function railTickSeconds(pixelsPerSecond, minPixels = 44) {
  const steps = [60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600];
  const pps = Number(pixelsPerSecond);
  if (!(pps > 0)) return 3600;
  return steps.find((step) => step * pps >= minPixels) ?? steps.at(-1);
}

/**
 * Lay out event cards from the top of the rail downwards (newest first).
 * `y(epoch)` maps a time to a rail position. Cards that would overlap the card
 * above join it as a cluster so every card stays at its own time.
 */
export function layoutRailCards(events, y, cardHeight = 60, gap = 8) {
  const ordered = [...events]
    .filter((event) => Number.isFinite(Number(event?.incident_epoch)))
    .sort((left, right) => Number(right.incident_epoch) - Number(left.incident_epoch));
  const cards = [];
  for (const event of ordered) {
    const top = y(Number(event.incident_epoch)) - cardHeight / 2;
    const previous = cards.at(-1);
    if (previous && top < previous.top + cardHeight + gap) {
      previous.events.push(event);
      continue;
    }
    cards.push({ top, event, events: [event] });
  }
  return cards;
}

/** Quarter-hour activity buckets for the 24 h edge rail. */
export function railDensityBuckets(dayStart, dayEnd, events, recordings, bucketSeconds = 900) {
  const start = Number(dayStart);
  const count = Math.max(1, Math.ceil((Number(dayEnd) - start) / bucketSeconds));
  const buckets = Array.from({ length: count }, (_, index) => ({
    start: start + index * bucketSeconds, end: start + (index + 1) * bucketSeconds,
    objects: 0, motion: 0, recorded: false,
  }));
  for (const range of recordings || []) {
    const first = Math.max(0, Math.floor((Number(range.start_epoch) - start) / bucketSeconds));
    const last = Math.min(count - 1, Math.floor((Number(range.end_epoch) - start - 0.001) / bucketSeconds));
    for (let index = first; index <= last; index++) buckets[index].recorded = true;
  }
  for (const event of events || []) {
    const index = Math.floor((Number(event.incident_epoch ?? event.start_epoch) - start) / bucketSeconds);
    if (index < 0 || index >= count) continue;
    if (event.has_objects) buckets[index].objects += 1; else buckets[index].motion += 1;
  }
  const peak = Math.max(1, ...buckets.map((bucket) => bucket.objects * 2 + bucket.motion));
  return buckets.map((bucket) => ({
    ...bucket,
    level: (bucket.objects * 2 + bucket.motion) / peak,
    kind: bucket.objects ? "object" : bucket.motion ? "motion" : bucket.recorded ? "recorded" : "empty",
  }));
}

/**
 * The condensed story: events in time order with quiet stretches longer than
 * `quietSeconds` collapsed into single gap entries.
 */
export function condenseStory(events, { quietSeconds = 300, startEpoch = null, endEpoch = null } = {}) {
  const ordered = [...events]
    .filter((event) => Number.isFinite(Number(event?.incident_epoch)))
    .sort((left, right) => Number(left.incident_epoch) - Number(right.incident_epoch));
  const items = [];
  let cursor = Number.isFinite(Number(startEpoch)) ? Number(startEpoch) : null;
  for (const event of ordered) {
    const start = Number(event.incident_epoch);
    const end = Math.max(start, Number(event.last_epoch) || start);
    if (cursor !== null && start - cursor > quietSeconds) items.push({ type: "quiet", start: cursor, end: start, seconds: start - cursor });
    items.push({ type: "event", event, start, end });
    cursor = cursor === null ? end : Math.max(cursor, end);
  }
  const stop = Number(endEpoch);
  if (cursor !== null && Number.isFinite(stop) && stop - cursor > quietSeconds) items.push({ type: "quiet", start: cursor, end: stop, seconds: stop - cursor });
  return items;
}

export function formatQuietDuration(seconds) {
  const minutes = Math.round(Math.max(0, Number(seconds) || 0) / 60);
  if (minutes < 60) return `${minutes} m`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest ? `${hours} h ${rest} m` : `${hours} h`;
}

/** The story event to continue with after `epoch`, for auto-advance. */
export function nextStoryEvent(items, epoch, graceSeconds = 2) {
  return items.find((item) => item.type === "event" && item.start > Number(epoch) + graceSeconds)?.event ?? null;
}

export function storyEventAt(items, epoch, graceSeconds = 3, preferredId = null) {
  const value = Number(epoch);
  const covering = items.filter((item) => item.type === "event" && value >= item.start - graceSeconds && value <= item.end + graceSeconds);
  // Overlapping events: keep the one the user picked when it covers the time.
  const preferred = preferredId === null || preferredId === undefined ? null : covering.find((item) => String(item.event.id) === String(preferredId));
  return (preferred || covering[0])?.event ?? null;
}

/**
 * Pairs of incidents on different cameras that share a tracked identity and
 * start within `withinSeconds`: the swimlanes draw these as one person.
 */
export function crossCameraLinks(events, withinSeconds = 120) {
  const keyed = events
    .map((event) => {
      const identity = event?.primary_identity;
      const key = identity && typeof identity === "object" ? identity.id ?? identity.identity_id ?? identity.key : null;
      return key === null || key === undefined ? null : { event, key: String(key), epoch: Number(event.incident_epoch) };
    })
    .filter((item) => item && Number.isFinite(item.epoch))
    .sort((left, right) => left.epoch - right.epoch);
  const links = [];
  for (let index = 0; index < keyed.length; index++) {
    for (let next = index + 1; next < keyed.length && keyed[next].epoch - keyed[index].epoch <= withinSeconds; next++) {
      if (keyed[next].key === keyed[index].key && keyed[next].event.camera_id !== keyed[index].event.camera_id) {
        links.push({ from: keyed[index].event, to: keyed[next].event });
        break;
      }
    }
  }
  return links;
}
