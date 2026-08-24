# Multi-disk media storage

SurvNG can place media across two or more independently mounted filesystems.
The storage pool is role-aware: each location can accept all media or only
selected categories such as continuous recordings, incident snapshots, motion
audit evidence, event clips, or exports.

This is a placement and indexing system, not a RAID layer or union filesystem.
SurvNG does not stripe one file across disks, mirror files, or automatically
move existing media merely to equalize free space. Each media file lives on one
normal filesystem and its location is retained in SurvNG's local indexes.

## What stays local

Multi-disk placement applies only to media. The following state continues to
use `database_dir`, `recording_index_dir`, or other existing local application
paths:

- the event, face, appearance, and semantic-search SQLite data;
- the continuous-recording index;
- model packages and OpenVINO caches;
- temporary export work and regenerable playback caches; and
- application configuration and logs.

Keeping these small, latency-sensitive files on local storage prevents a slow
or unavailable media mount from blocking the application database.

## Media roles and directory layout

Every configured location has a filesystem root. SurvNG creates fixed
subdirectories beneath that root according to the enabled roles:

| Role | Directory | Contents |
|---|---|---|
| `recordings` | `recordings/` | Main and substream continuous MP4 segments |
| `snapshots` | `snapshots/` | Clean incident evidence images |
| `motion_audits` | `motion_samples/` | Sampled EMA and motion-audit evidence |
| `clips` | `event_clips/` | Generated incident playback clips |
| `exports` | `exports/` | Recording exports and timelapses |

Example with a location rooted at `/mnt/survng-a`:

```text
/mnt/survng-a/
├── recordings/
│   └── gate/main/2026-08-13/14/...
├── snapshots/
│   └── gate/...
├── motion_samples/
│   └── gate/...
├── event_clips/
│   └── gate/main/...
└── exports/
    ├── recording/
    ├── timelapse/
    └── manifests/
```

If only one media location is configured, SurvNG still uses the same placement
and health checks as a multi-disk pool. Point that location at your media
filesystem (often the same path as `storage_dir` for portable relative paths).
Empty `media_storage.locations` is invalid.

## Configuration

Configure locations under **Admin → General → Storage → Media locations**.
Saving a change to the media-location topology or placement mode causes SurvNG
to transactionally replace its application-manager generation. Camera workers,
recorders, and media services are rebuilt against the new pool; the SurvNG
process itself does not require a separate restart. The save can be rejected
while protected storage work is active, in which case wait for that work to
finish or cancel it from Maintenance and save again.

Equivalent JSON:

```json
{
  "storage_dir": "/var/lib/survng/media",
  "database_dir": "/var/lib/survng/database",
  "recording_index_dir": "/var/lib/survng/index",
  "media_storage": {
    "placement": "balanced",
    "locations": [
      {
        "id": "array_a",
        "name": "Recording Array A",
        "path": "/mnt/survng-a",
        "enabled": true,
        "roles": [
          "recordings",
          "snapshots",
          "motion_audits",
          "clips",
          "exports"
        ],
        "reserve_percent": 15,
        "priority": 100,
        "require_mount": true
      },
      {
        "id": "array_b",
        "name": "Recording Array B",
        "path": "/mnt/survng-b",
        "enabled": true,
        "roles": ["recordings", "snapshots", "motion_audits"],
        "reserve_percent": 15,
        "priority": 100,
        "require_mount": true
      }
    ]
  }
}
```

### Location fields

`id`
: Stable identifier stored with indexed recordings. It may contain letters,
  numbers, underscores, and hyphens. Treat it as immutable after media has
  been written.

`name`
: Human-readable label shown in Admin and telemetry.

`path`
: Absolute filesystem root. Each configured location must use a unique path.
  The directory must already exist; SurvNG creates the role subdirectories.

`enabled`
: Controls whether the location accepts new media. Disabling a location does
  not move or delete its existing files. Keep the location configured and
  mounted while SurvNG still needs to read those files.

`roles`
: Media categories the location is permitted to hold. At least one role is
  required.

`reserve_percent`
: Percentage of the complete filesystem that is withheld from new placement.
  This is calculated from total filesystem capacity, not from the amount of
  SurvNG media on the disk.

`priority`
: Placement weight from 1 through 1000. Larger values have greater precedence.
  In balanced mode it weights usable space; in priority mode it is the primary
  selection criterion.

`require_mount`
: When enabled, the configured path must be a real mountpoint. This should be
  enabled for NFS, SMB, removable disks, and dedicated block filesystems. It
  prevents SurvNG from writing into an empty local directory when the intended
  mount is absent.

## Location health

Before accepting a new placement, SurvNG evaluates each location:

| State | Meaning | New writes |
|---|---|---|
| `online` | Enabled, present, writable, mounted when required, and above reserve | Allowed |
| `unavailable` | Disabled, missing, or filesystem status cannot be read | Rejected |
| `not_mounted` | `require_mount` is enabled but the path is not a mountpoint | Rejected |
| `read_only` | Directory is not writable/searchable | Rejected |
| `full` | Free space is at or below the configured reserve | Rejected |

The health check uses the filesystem containing the configured path. Two
different directory paths on the same underlying filesystem therefore report
the same capacity and should not be treated as independent disks.

## Usable space and reserve

SurvNG calculates placement capacity as:

```text
reserve bytes = filesystem total bytes × reserve percent ÷ 100
usable bytes  = max(0, filesystem free bytes − reserve bytes)
```

For a 10 TB filesystem with 2 TB free and a 15% reserve:

```text
reserve = 1.5 TB
usable  = 2.0 TB − 1.5 TB = 0.5 TB
```

When free space falls to 1.5 TB, usable space becomes zero and the location no
longer accepts a new assignment.

The reserve is an admission boundary, not currently a hard quota on an FFmpeg
process that is already recording. An active recorder chooses its output root
when it starts and can continue writing after that filesystem crosses its
reserve. Recording retention is expected to reclaim space before exhaustion.
If cleanup cannot reclaim eligible media, the filesystem can still fill and
the recorder will eventually report write failures. Operators should keep
automatic retention enabled and set its free-space threshold above, or with
adequate margin relative to, the placement reserve.

## Balanced placement

Balanced placement uses deterministic weighted rendezvous assignment. Each
camera/workload key receives a stable hash score per eligible location, and
usable bytes plus priority weight that score:

```text
weight = usable bytes × priority ÷ 100
score  = weight ÷ -ln(stable_hash(workload, location))
```

Example:

| Location | Free | Reserve | Usable | Priority | Placement score |
|---|---:|---:|---:|---:|---:|
| A | 5 TB | 1 TB | 4 TB | 100 | 4 TB |
| B | 3 TB | 0.5 TB | 2.5 TB | 200 | 5 TB |

The hash term prevents an equal cold-start pool from assigning every camera to
the same disk, while the weight still favors locations with more usable space
or a higher priority. Use the same priority when capacity should be the only
weighting difference.

Balanced placement does not alternate every file or every 10-second recording
segment. Assignments are sticky for the life of the running manager generation.
Whenever a workload asks the registry for placement again, SurvNG reuses the
selected location while it remains eligible. Similar free space on two disks
therefore does not cause oscillation.

Assignment is deterministic for the same configuration and workload key, so a
restart does not randomly reshuffle an otherwise unchanged pool.

## Priority placement

Priority placement selects the eligible location using this order:

1. highest `priority` value;
2. greatest usable space; and
3. location ID as a deterministic tie-breaker.

It continues using that location until the assignment loses eligibility. Only
then does it choose the next eligible location. Priority mode is appropriate
when one filesystem should normally receive all new media and another should
act primarily as overflow.

Changing between priority and balanced mode is non-destructive. Existing media
stays where it is, and all configured locations remain searchable. The new
mode affects only placement decisions made by the newly loaded manager
generation.

## Sticky assignment scope

SurvNG uses different assignment keys for different media workloads:

| Media | Assignment scope |
|---|---|
| Continuous recordings | Camera and source, such as `gate:main` or `gate:live` |
| Incident snapshots | Camera |
| Motion-audit evidence | Camera |
| Event clips | Camera and recording source |
| Exports and timelapses | Shared export workload |

This keeps a camera/source recording sequence together during ordinary
operation and avoids repeated cross-filesystem decisions. A location becoming
disabled, unavailable, read-only, unmounted, or full invalidates its cached
assignment the next time that workload asks the registry for placement.

Placement is requested at different lifecycle boundaries:

- a recorder chooses its directory when that camera/source recorder starts;
- an incident-snapshot writer chooses its directory when the camera worker is
  constructed;
- a motion-audit sample checks placement when it is written;
- an event clip checks placement when the clip path is created; and
- the export worker chooses its root when the export manager is constructed.

Consequently, an already-running recorder or camera snapshot writer does not
move mid-generation solely because its filesystem crosses the reserve. A new
selection occurs after its owning service is rebuilt or restarted. Motion-audit
and event-clip writes re-enter placement more frequently and can fail over on
their next request.

## Reading existing media

The recording index stores absolute paths and a location ID for each segment.
Playback, timeline discovery, export generation, event clips, maintenance, and
retention search every configured recording root.

Snapshots outside `storage_dir` retain verified absolute paths.
Incident viewing, face recognition, appearance indexing, semantic search,
model evaluation, AI analysis, and training-image APIs validate those paths
against the configured media-location registry before reading them.

This means a location may be disabled for new placement while its existing
media remains readable. Do not remove the location from configuration or
unmount it until that media has expired, been migrated, or is no longer needed.

## Retention across disks

Recording retention remains index-driven and evaluates recording filesystems
independently:

- age limits apply per camera and source regardless of location;
- the configured total recording quota applies to indexed recordings across
  the complete pool;
- minimum and target free-space watermarks are evaluated for each recording
  location;
- when a location is below its minimum, capacity cleanup targets oldest
  eligible recordings on the pressured location; and
- protected incident recordings, active output, and the newest protected
  recording window remain guarded.

The placement reserve and retention watermarks are separate controls:

- `reserve_percent` determines whether a location may accept a new placement;
- `minimum_free_percent` determines when retention pressure begins; and
- `target_free_percent` determines how far cleanup attempts to recover.

For proactive cleanup, configure the retention minimum at or above the media
location reserve and set the retention target several percentage points higher.
For example:

```text
media reserve:            15%
retention minimum free:   15% or higher
retention target free:    20%
```

If automatic cleanup is disabled, SurvNG can calculate a dry-run plan but will
not delete recordings. If all eligible locations reach their reserves, new
placements fail until space is reclaimed or another eligible location becomes
available.

Incident snapshots use a separate age policy within the same serialized
retention worker. The default is 1,095 days (three 365-day years), and snapshot
age is never shortened merely because recording storage is under pressure.
Pinned face-reference images remain protected. Export retention is handled by
the export subsystem, while motion-audit sample pruning remains independently
bounded per camera. Event clips, databases, and model data are not deleted by
recording or snapshot retention.

See [Storage retention](storage-retention.md) for detailed deletion policy.

## What happens when a disk fails

For new media:

- a failed location is excluded from selection;
- cached assignments to it are invalidated when next checked; and
- SurvNG selects another eligible location for that role.

For existing media:

- database and index entries are preserved;
- requests for unavailable files fail rather than silently substituting a
  different file; and
- media becomes readable again if the same filesystem returns at the same path.

SurvNG does not reconstruct missing media and does not mirror it automatically.
Multi-disk placement improves capacity management and write failover, but it is
not a backup strategy.

When every location eligible for a role is unavailable or at reserve, SurvNG
raises a clear `no writable media location supports <role>` error for new
placement. Existing indexed media on reachable locations remains available.

## Maintenance and repair

A full **Admin → Maintenance** storage scan:

- reconciles continuous recordings across every configured recording root;
- checks snapshot and motion-audit references across configured locations;
- identifies missing references and old unreferenced evidence samples;
- reports health and capacity for each media location; and
- preserves user media unless an explicit repair operation safely clears a
  database reference to a missing file.

Maintenance does not rebalance media and does not delete arbitrary orphaned
files. It also does not turn a missing filesystem into an empty replacement.
Use `require_mount` to ensure the intended mount exists before allowing writes.

## Adding a location safely

1. Format or provision the filesystem outside SurvNG.
2. Mount it at a stable host path.
3. For Docker, bind-mount that exact host path into the container at a stable
   container path.
4. Create the configured root and grant the SurvNG process read, write, and
   directory-search permissions.
5. Add the location in Admin with a permanent ID and the desired roles.
6. Enable **Require a real mount** for dedicated or network filesystems.
7. Set reserve and priority values.
8. Save the configuration; SurvNG performs a controlled manager-generation
   reload, including its camera and recorder workers.
9. Verify that Admin reports the location as `online` before relying on it.

No existing files are moved when a location is added.

## Disabling or removing a location safely

To stop new writes while retaining access, clear **Accept new media** but leave
the location configured and mounted. This is the safest drain state.

Before removing a location entirely:

1. stop new placement to it;
2. allow its recordings to expire or deliberately migrate its media;
3. verify that no event, face, export, or recording index references remain;
4. run a full Maintenance scan; and
5. only then remove the location and unmount the filesystem.

There is currently no automatic drain/rebalance operation. Moving files by
hand without updating their database or recording-index paths makes those
files appear missing.

## Docker considerations

All configured media paths are interpreted inside the SurvNG container. Mount
each host filesystem explicitly:

```yaml
services:
  survng:
    volumes:
      - /srv/survng/database:/config/database
      - /mnt/survng-a:/media-a
      - /mnt/survng-b:/media-b
```

The matching SurvNG paths would be `/media-a` and `/media-b`, not the host-side
paths. The container must see each as a separate mounted filesystem if
`require_mount` is enabled. Keep the container paths stable across upgrades.

## Performance characteristics

Placement checks use normal filesystem capacity and permission calls and are
small compared with decoding, recording, and inference work. Continuous
recordings do not recalculate placement for every segment. Normal retention is
SQLite-index driven and does not repeatedly walk every media filesystem.

Full Maintenance scans intentionally enumerate configured evidence roots and
can produce substantial I/O on large or remote filesystems. Run full scans
during quieter periods. Ordinary playback accesses only the indexed files
needed for the selected camera and time range.

## Current boundaries

The initial multi-disk implementation deliberately has several conservative
boundaries:

- placement assignments are in memory and may be recalculated after restart;
- existing media is not automatically rebalanced or migrated;
- no mirroring or redundancy is provided;
- an active FFmpeg recorder is not forcibly moved merely because its disk
  crosses the placement reserve;
- recording retention does not reclaim non-recording roles; and
- removing a configured location does not rewrite historical paths.

These properties make placement predictable and preserve existing media, but
they mean monitoring and retention remain essential protections against a
completely full filesystem.
