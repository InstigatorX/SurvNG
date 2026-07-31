# Storage retention

SurvNG separates continuous recordings from protected incident media and
regenerable playback data. The retention worker only removes indexed MP4 files
inside `storage_dir/recordings`.

It never removes:

- event clips, snapshots, faces, motion-audit images, or databases;
- a continuous segment still referenced by an incident;
- an active recording or any segment from the newest five minutes; or
- files outside the configured recordings directory.

Retention combines three limits. A main or substream segment becomes eligible
when it exceeds that source's age limit. Cleanup also runs when indexed
continuous recordings exceed the SurvNG storage quota, or when free space on
the complete storage filesystem falls below the minimum watermark. Free-space
cleanup continues toward the higher target watermark to avoid repeatedly
starting and stopping around one threshold.

The global defaults are seven days for main streams, 21 days for substreams, a
13 TiB continuous-recording quota, cleanup below 15% free, a 20% free-space
target, and a 5% emergency warning. Cameras inherit the stream age limits and
can override either value independently.

Automatic deletion is disabled by default. Admin > General > Storage displays
the index-only dry-run projection, current growth rate, estimated headroom,
eligible bytes, and per-camera usage. **Recalculate** refreshes the plan without
deleting anything. **Clean Up Now** requires confirmation and starts bounded,
oldest-first cleanup. Enabling automatic cleanup applies the same guarded plan
in the background.

The worker uses the local SQLite recording index and does not walk NFS during
normal operation. It performs the complete storage projection once per day,
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
