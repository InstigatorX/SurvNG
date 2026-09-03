# Matched GStreamer live-box admission

This draft specifies the safety boundary required before `gvadetect` boxes may
provisionally admit an EMA-pinned event.

## Contract

- A detector snapshot contains the GStreamer source PTS, live capture-session
  generation, detector dimensions, inference order, and an object list.
- An empty list is authoritative only when inference actually ran. This rollout
  uses `inference-interval=1`; skipped frames must not be emitted as empty.
- The child emits the actual `Gst.Buffer.pts` for both qualifier frames and
  detector snapshots. Receipt time is never used for matching.
- Parent caches a bounded per-live-session history. Lookup requires the same
  session generation and a PTS delta no greater than one detection period plus
  a small tolerance. It never selects a future result.
- A matched empty snapshot clears older positives. Missing, stale, invalid, or
  mismatched snapshots yield no fast object and continue to recorded-main
  refinement unchanged.

## Geometry

Detector boxes are rescaled from snapshot dimensions to the selected live EMA
frame before normal zone handling. Live-to-main calibration remains required
for provisional zone admission; untrusted geometry remains fail-closed.

## Explicitly deferred

No sidecar-backed tracking, no wall-clock matching, no unbounded cache, no
database persistence for ephemeral snapshots, and no `inference-interval > 1`
until skipped-vs-empty semantics are represented.

## Acceptance tests

1. Positive and empty snapshots round trip with PTS, dimensions, and session.
2. A positive followed by an authoritative empty cannot resurrect the positive.
3. Future, stale, invalid-PTS, old-session, and reconnect snapshots are rejected.
4. A matched EMA evidence frame can provisionally use a positive snapshot;
   unmatched/empty results retain recorded-main refinement.
5. Rescaling occurs before zone logic; untrusted FOV remains non-admitting.
6. A slow detector does not reduce EMA cadence.
