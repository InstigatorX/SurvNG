# Recordings & storage

SurvNG can keep continuous video, incident pictures, short clips, and exports. Storage settings decide how long each kind of media stays and when cleanup runs.

## What gets stored

| Kind | Everyday meaning |
| --- | --- |
| Main recordings | High-quality continuous history |
| Live/sub recordings | Lighter continuous history when configured |
| Snapshots | Incident evidence pictures |
| Motion audits | Diagnostic samples that were not incidents |
| Event clips | Short generated clips around incidents |
| Exports | Files you explicitly saved for download |

## How retention works

Under **Admin → Storage** you set limits such as:

- How many days of main vs live recordings to keep
- How long snapshots stay
- Minimum free disk space SurvNG should protect
- Whether cleanup runs automatically

Protected incident media and protected exports are treated more carefully than ordinary continuous video.

Deep detail: [Storage retention](../storage-retention.md) and [Multi-disk media storage](../storage.md).

## Multiple disks

Advanced installs can place media on more than one mounted disk and assign roles (recordings on disk A, exports on disk B, and so on). Databases and configuration stay on fast local storage.

## Evidence image format

New incident pictures default to WebP. You can switch format/quality under Storage. Changing this affects new images only; old files remain readable.

## Maintenance

**Admin → Maintenance** can run storage cleanup or repair index/media links after disk problems. Prefer planned maintenance windows for large cleanups.

## Example starter policy

For a home with a 4 TB disk and a few cameras:

- Keep about a week of main video
- Keep longer for incident snapshots
- Leave automatic cleanup off until you have watched free space for a few days
- Turn automatic cleanup on once you trust the watermarks

Exact numbers depend on camera count, resolution, and how busy the scenes are.

## Related

- [Timeline & exports](timeline.md)
- [Admin](admin.md)
- [HTTP API](api.md) (retention and maintenance endpoints)
