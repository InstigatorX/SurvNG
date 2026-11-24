# Backend dead-code cleanup — 2026-11

Baseline: `v1.2` at `6813834` (after #365–#368). The earlier reports
[`dead-code-audit.md`](dead-code-audit.md), [`dead-code-cleanup.md`](dead-code-cleanup.md)
and [`dead-code-audit-2026-10-04.md`](dead-code-audit-2026-10-04.md) remain
historical records. This campaign supersedes their "retain" decisions for the
items below because [`current-only.md`](current-only.md) has since retired
prior-generation configuration, databases and tracking comparisons.

Each change was made on a stacked branch so the steps can be tested together:
`cleanup/dead-code-t1` → `t2` → `t3` → `t4` → `t5`.

## Removed

| Branch | Change |
| --- | --- |
| t1 | `scripts/replay-detector-postprocess.py` (called a parser removed in #350); a telemetry test that could not fail; `botsort_*` keys in the example configs; the `OPENCV_FFMPEG_CAPTURE_OPTIONS` admin note; 11 definitions and 3 constants with no reference anywhere. |
| t2 | 11 undocumented face diagnostic, benchmark and dedupe endpoints and the face-store code only they reached. The `scene_discovery_baselines` table and three redundant indexes (dropped at startup). The `detector.depth.motion_evidence_enabled` and `cameras[].video_backend` config fields. |
| t3 | `tracking_evaluation.py`, the `HybridCandidateObjectTracker` alias and `ObjectTrackingLifecycle.bind_for_compatibility`; the uncalled `GET /api/cameras/{id}/recordings/mobile-window.mp4`. |
| t4 | 102 route-handler aliases in `main.py` (tests use the router bundles); the derived "legacy" retention budget and the tuple live-frame fallback, both reachable only from tests. |
| t5 | Upgrades that 1.2.0 already applied (see below). Incident revisions no longer write the never-read `scene_changes` table. |
| t6 | With a single install on 1.2 data, the remaining pre-1.2.0 support: the startup schema check added during review, the three drop-index cleanups for pre-1.2.0 indexes, the plaintext `secrets.json` read path, the retired-MOG2 guard (unsupported fusion sources and stages are still rejected) and the ZIP export media type. Startup drops the retired `scene_changes`, `tracking_comparisons`, `scene_migrations` and `scene_acquisition_migrations` tables. |

## Decisions

- **Minimum database:** one that SurvNG 1.2.0 or later has opened. Each removed
  upgrade was present in 1.2.0 and its column was already in that table's
  CREATE statement. There is no startup check: the only install runs on 1.2
  data. Startup drops retired tables no current build reads and v1.2.0's unused
  `idx_event_camera_epoch`, which taxed every event insert.
- **Face rows linked by the retired dedupe** (`duplicate_of_observation_id`)
  can remain on installs; pruning of those trees is unchanged.
- **Documented API stays** even without an in-repo caller, including
  `/api/events/{id}/stream.m3u8`, whose frontend caller #367 removed.

## Retained on purpose

- Column additions that fresh databases still need (columns not in their
  CREATE), and drop-index cleanups added since 1.2.0.
- `config_init` and `face_evaluation` (command-line tools), the operator shell
  scripts, and framework or protocol callbacks.

## Deferred

- No dead code was found in `media_exports.py` or
  `recording_media_runtime.py`; re-check them after #364 merges, since it
  rewrites much of both.
- Write-only columns that need a table rebuild to drop
  (`face_detection_samples` crop geometry, `route_incident_admissions.admitted_at`,
  `scene_alert_entries.last_decision`, `appearance_backfill_jobs.scene_backfill_version`).
