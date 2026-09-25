# People: visits and recognition evidence

People → Visits reconstructs a person's visit from durable camera tracks and face
observations. It distinguishes confirmed faces, automatic face recognition,
sightings linked to an identified visit, and unresolved people. Linked sightings
never enter the trusted face gallery.

## Visit review

Select a UTC start and a 1, 6, or 24-hour window, or leave the start empty for
recent sightings. Open each sighting's incident/recording to inspect its body
evidence. Confirm suggested links, or select two sightings to mark them as the
same person/visit or keep them separate. Reset removes your relationship decision.
Separating a link also prevents the two sightings from reconnecting through a
third sighting. Existing face review controls continue to own names/enrollment.

Visits are a deterministic projection of retained evidence and persisted link
decisions, computed on demand. They are not another mutable copy of face names.
Face correction, deletion, model/evidence replacement, and retention are reflected
on the next read. Changed source evidence invalidates old relationship decisions;
renaming or correcting a face does not erase a rejection. Contradictory identities
block a previously accepted link. No visit-based alert or gallery learning runs.

The display is bounded to 400 sightings and 200 suggestions per query. Truncation
is reported and automatic association is suspended when sightings are incomplete.
Membership and the representative visit ID are scoped to the selected event-time
window; widening the window can reveal earlier anchors and change the visit ID.
These IDs must not be used as permanent person identifiers. The API supports any
positive window up to 24 hours. Existing event retention removes related decisions.

Body ReID and camera transition routes are required for automatic suggestions.
Without them, existing person detections and faces remain visible and can be
linked manually. Historical face tracks do not share body tracker IDs: automatic
ownership requires one body track, one face, one detected person and an overlapping
timestamp. Ambiguous multi-person events retain separate face/body sightings;
an operator can link the face to the correct body after inspecting the recording.

## Automatic association

The following `detector.tracking` settings control visits:

| Setting | Default | Meaning |
| --- | --- | --- |
| `visit_auto_link_enabled` | `false` | Review suggestions before enabling calibrated automatic links |
| `visit_match_threshold` | `0.85` | Minimum cosine similarity, also respecting each embedding's threshold |
| `visit_top_two_margin` | `0.08` | Minimum lead over competing candidates in each camera direction |
| `visit_min_quality` | `0.4` | Minimum producer-reported appearance quality (some producers use detector confidence) |
| `visit_max_seconds` | `1200` | Maximum whole-visit duration |

These are conservative initial settings, not calibrated accuracy guarantees.
Automatic edges require matching embedding versions/dimensions, at least two
observations, a configured directional route and plausible travel time. They
reject identity conflicts, same-camera overlap and distinct tracks in one event.
Every cross-group appearance pair must pass the threshold to prevent chained
appearance drift. Similarity is displayed as a score, never a probability.

## Enrollment and evaluation

Use confirmed, varied examples from the actual cameras and different days.
Keep a person's persistent identity separate from clothing during a visit.
Never promote inferred visit links into enrollment examples automatically.

Build a labeled corpus containing unknown visitors, lookalikes, clothing changes,
groups, crossings, night/IR and poor faces. Split by whole visit/day, including
people absent from enrollment. Tune on calibration data; evaluate on held-out
days. Measure wrong names, false visit merges, identified and unresolved visits,
review burden, latency and resource contention. Keep coverage alongside accuracy.
Start with pretrained models; fine-tune only if held-out evidence justifies it.
