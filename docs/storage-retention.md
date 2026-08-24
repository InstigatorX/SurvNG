# Storage retention

SurvNG separates continuous recordings from protected incident media and
regenerable playback data. Configure at least one media location under
`media_storage.locations` (often the same path as `storage_dir`). Admin can
add additional role-aware locations for recordings, snapshots, motion-audit
evidence, event clips, and exports. Databases, indexes, model packages, and
transient playback work remain on local application storage.

Each location has an immutable ID, filesystem path, accepted media roles,
reserve percentage, priority, enabled state, and optional mount requirement.
`balanced` placement chooses the eligible location with the greatest weighted
usable space; `priority` placement prefers the highest-priority eligible
location. Placement is sticky for a camera/source or media role while that
location remains writable, so SurvNG does not bounce an active stream between
filesystems. New writes fail over when a location becomes unavailable or
reaches its reserve. Existing indexed media remains readable from every
configured location.

The retention worker removes indexed MP4 files inside configured recording-role
directories and clean incident images inside configured snapshot-role
directories. Capacity and watermarks are evaluated separately for recording
filesystems so one full location can be cleaned without treating free space on
another disk as interchangeable. Snapshot deletion is age-only.

It never removes:

- event clips, pinned face-reference images, motion-audit images, or databases;
- a continuous segment still referenced by an incident;
- an active recording or any segment from the newest five minutes; or
- files outside the configured recordings directory.

Retention combines three limits. A main or substream segment becomes eligible
when it exceeds that source's age limit. Cleanup also runs when indexed
continuous recordings exceed the SurvNG storage quota, or when free space on
the complete storage filesystem falls below the minimum watermark. Free-space
cleanup continues toward the higher target watermark to avoid repeatedly
starting and stopping around one threshold.

The global defaults are seven days for main streams, 21 days for substreams,
1,095 days for incident snapshots, a 13 TiB continuous-recording quota, cleanup
below 15% free, a 20% free-space target, and a 5% emergency warning. Cameras
inherit the stream age limits and can override either value independently.
Snapshot age is currently one global evidence policy.

Automatic deletion is disabled by default. Admin > General > Storage displays
the index-only dry-run projection, current growth rate, estimated headroom,
eligible bytes, and per-camera recording/snapshot usage. **Recalculate** refreshes the plan without
deleting anything. **Clean Up Now** requires confirmation and starts bounded,
oldest-first cleanup. Enabling automatic cleanup applies the same guarded plan
in the background.

The worker uses local SQLite recording and event indexes and does not walk
media directories during normal operation. Legacy event rows without a cached
image size are reconciled from their known paths in bounded batches. It performs
the complete storage projection once per day,
when retention settings change, when an operator requests recalculation, or
when the free-space watermark is crossed. Lightweight index-driven expiration
runs every 15 minutes between those projections.

Cleanup deletes a bounded batch, removes successfully deleted or already-missing
paths from the index, cleans empty date directories, and waits 10 seconds before
the next batch when a backlog remains. These bounded batches do not block a
SurvNG restart; shutdown stops between individual files and commits the index
updates for files already handled. Admin reports the states as `planning`,
`cleaning`, `waiting`, and `idle`. Storage or database
errors stop the cycle and are reported in Admin rather than causing an unbounded
retry loop.

## Operational guidance for multiple locations

- Mount every filesystem on the host before starting SurvNG. Enable **Require
  mount** for network or removable paths so a missing mount cannot silently
  write into its empty local mountpoint.
- Give each location a stable path. Existing database rows store verified
  absolute paths for media outside the legacy root, so changing a mount path
  requires moving or reconciling that media first.
- A location may accept one role or several. Keeping recordings on large disks
  while snapshots and clips use faster storage is supported, as is allowing
  every role on every location.
- Removing or disabling a location prevents new placement there. It does not
  migrate or delete existing media. Keep the path configured until its indexed
  media has expired or been moved deliberately.
- Full Maintenance scans inspect snapshot and motion-audit roots across all
  configured locations. Export cleanup remains governed by export retention;
  recording cleanup remains governed by recording retention.
