// The approved storyboard's sample incident, in the shape the scene API
// returns: a car arrives on Front Driveway and stays, a person walks from the
// driveway to Front Door, and a possible dog appears once in Side Yard.
// 10:24:20 – 10:26:02 America/New_York on 2026-11-25.
export const BASE = Date.parse("2026-11-25T15:24:20Z") / 1000;
const iso = (offset) => new Date((BASE + offset) * 1000).toISOString().replace(".000Z", "+00:00");
const frame = { detection_frame_width: 1280, detection_frame_height: 720 };

function observation(id, camera, offset, label, box, extra = {}) {
  return { id, event_id: 900 + Math.round(offset), camera_id: camera, captured_at: iso(offset), label, confidence: .82, snapshot_available: true,
    box: { x1: box[0], y1: box[1], x2: box[2], y2: box[3] }, zones: extra.zones || [], ...frame, ...extra,
    image: { width: 1280, height: 720, source: "recorded_main", captured_at: iso(offset), analyzed_frame: true } };
}

const carBox = [154, 288, 486, 533];
const car = {
  id: "object-car", label: "car", certainty: "observed", confidence: .9, source_event_id: 900, object_index: 0,
  first_seen_at: iso(0), last_seen_at: iso(102), association_corrected: false, continuity_uncertain: false,
  observations: [
    observation("obs-car-0", "front-driveway", 0, "car", [40, 300, 330, 520], { zones: ["Driveway"], activity_role: "active" }),
    observation("obs-car-10", "front-driveway", 10, "car", [110, 290, 440, 530], { zones: ["Driveway"], activity_role: "active" }),
    observation("obs-car-20", "front-driveway", 20, "car", carBox, { zones: ["Driveway"], activity_role: "active" }),
    observation("obs-car-30", "front-driveway", 30, "car", carBox, { zones: ["Driveway"], activity_role: "scene_context" }),
    observation("obs-cover-car", "front-driveway", 30.5, "car", carBox, { zones: ["Driveway"], activity_role: "scene_context" }),
    observation("obs-car-102", "front-driveway", 102, "car", carBox, { zones: ["Driveway"], activity_role: "scene_context" }),
  ],
  visible_intervals: [
    { camera_id: "front-driveway", episode_id: "episode-driveway", start_at: iso(0), end_at: iso(20), state: "moving", observation_id: "obs-car-0", last_observation_id: "obs-car-20", observation_count: 3, zones: ["Driveway"] },
    { camera_id: "front-driveway", episode_id: "episode-driveway", start_at: iso(20), end_at: iso(102), state: "stationary", observation_id: "obs-car-20", last_observation_id: "obs-car-102", observation_count: 4, zones: ["Driveway"] },
  ],
};
car.observation_count = car.observations.length;

const person = {
  id: "object-person", label: "person", certainty: "observed", confidence: .91, source_event_id: 921, object_index: 1,
  first_seen_at: iso(21), last_seen_at: iso(102), association_corrected: false, continuity_uncertain: false,
  observations: [
    observation("obs-person-21", "front-driveway", 21, "person", [520, 230, 640, 600], { zones: ["Driveway"], activity_role: "active" }),
    observation("obs-cover-person", "front-driveway", 30.5, "person", [563, 202, 704, 605], { zones: ["Driveway"], activity_role: "active" }),
    observation("obs-person-49", "front-driveway", 49, "person", [820, 190, 950, 560], { zones: ["Driveway"], activity_role: "active" }),
    observation("obs-person-door-47", "front-door", 47, "person", [470, 180, 700, 690], { zones: ["Entry"], activity_role: "active" }),
    observation("obs-person-door-62", "front-door", 62, "person", [512, 144, 742, 662], { zones: ["Entry"], activity_role: "active", depth_m: 3 }),
    observation("obs-person-door-102", "front-door", 102, "person", [540, 150, 760, 660], { zones: ["Entry"], activity_role: "active" }),
  ],
  visible_intervals: [
    { camera_id: "front-driveway", episode_id: "episode-driveway", start_at: iso(21), end_at: iso(49), state: "moving", observation_id: "obs-person-21", last_observation_id: "obs-person-49", observation_count: 3, zones: ["Driveway"] },
    { camera_id: "front-door", episode_id: "episode-door", start_at: iso(47), end_at: iso(102), state: "moving", observation_id: "obs-person-door-47", last_observation_id: "obs-person-door-102", observation_count: 3, zones: ["Entry"] },
  ],
};
person.observation_count = person.observations.length;

const dog = {
  id: "object-dog", label: "dog", certainty: "possible", confidence: .41, source_event_id: 974, object_index: 0,
  first_seen_at: iso(74), last_seen_at: iso(74), association_corrected: false, continuity_uncertain: false,
  observations: [observation("obs-dog-74", "side-yard", 74, "dog", [742, 403, 922, 576], { confidence: .41, zones: ["Lawn"] })],
  visible_intervals: [
    { camera_id: "side-yard", episode_id: "episode-side", start_at: iso(69), end_at: iso(82), state: "observed", observation_id: "obs-dog-74", last_observation_id: "obs-dog-74", observation_count: 1, zones: ["Lawn"] },
  ],
};
dog.observation_count = 1;

export function storyIncident() {
  return {
    id: "incident-story-1", incident_id: "incident-story-1", revision: 7, state: "complete", historical: false,
    camera_id: "front-driveway", representative_event_id: 930, snapshot_observation_id: "obs-cover-car",
    snapshot_url: "/api/incidents/observations/obs-cover-car/snapshot",
    start_at: iso(0), end_at: iso(102), start_epoch: BASE, last_epoch: BASE + 102, duration_seconds: 102,
    camera_ids: ["front-driveway", "front-door", "side-yard"], labels: ["car", "dog", "person"], zones: ["Driveway", "Entry", "Lawn"], has_objects: true,
    summary: "Car, possible dog and person observed",
    events: [{ id: 930, camera_id: "front-driveway", created_at: iso(30), labels: ["car", "person"] }, { id: 974, camera_id: "side-yard", created_at: iso(74), labels: ["dog"] }],
    episodes: [
      { id: "episode-driveway", camera_id: "front-driveway", start_at: iso(0), end_at: iso(102), event_ids: [930], coverage: { state: "sampled" } },
      { id: "episode-door", camera_id: "front-door", start_at: iso(45), end_at: iso(102), event_ids: [947], coverage: { state: "sampled" } },
      { id: "episode-side", camera_id: "side-yard", start_at: iso(68), end_at: iso(84), event_ids: [974], coverage: { state: "incomplete" } },
    ],
    scene_objects: [person, car, dog].map((item) => structuredClone(item)),
    establishment: { status: "established", reason: "supported_movement", summary: "", supporting_observation_ids: ["obs-car-10"], policy_version: 1 },
    coverage: { state: "incomplete", gaps: [] },
    alert_decisions: [{ event_id: 947, eligible: true, objects: [{ label: "person", eligible: true, zones: ["Entry"], object_id: "object-person" }] }],
    activity: [], identities: [],
  };
}

export const CAMERAS = [
  { id: "front-driveway", name: "Front Driveway" },
  { id: "front-door", name: "Front Door" },
  { id: "side-yard", name: "Side Yard" },
];

const SCENES = {
  "front-driveway": ["#1d2622", "#2b3530", "Front Driveway"],
  "front-door": ["#221f1d", "#3a3129", "Front Door"],
  "side-yard": ["#18241a", "#26372a", "Side Yard"],
};

// A plain drawn scene for one camera (no real footage in tests).
export function sceneSvg(cameraId) {
  const [sky, ground, name] = SCENES[cameraId] || SCENES["front-driveway"];
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720"><rect width="1280" height="720" fill="${sky}"/><rect y="380" width="1280" height="340" fill="${ground}"/><rect x="900" y="120" width="300" height="260" fill="#2c2f33"/><rect x="1010" y="220" width="70" height="160" fill="#3d4146"/><text x="24" y="700" fill="#6b7480" font-family="sans-serif" font-size="22">${name} · recorded frame</text></svg>`;
}

export function queueFor(incidentId, total = 25, decided = 4) {
  const items = Array.from({ length: total }, (_, index) => ({ incident_id: index === decided ? incidentId : `incident-q-${index}`, start_at: iso(-600 * index), end_at: iso(-600 * index + 60), verdict: index < decided ? "reviewed" : null, changed_since_review: false }));
  return { items, total, truncated: false, reviewed: decided, kept: 0, remaining: total - decided, day: "2026-11-25", time_zone: "America/New_York" };
}
