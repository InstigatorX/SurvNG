# Changelog

## Unreleased

- Incidents opens on a gallery in place of the viewer. It loads as it is
  scrolled, from the selected day into the 30 days before it, with a slider for
  incidents per row. Each picture shows its camera and time, object icons and a
  review mark, and opens the incident overlay used by Live. Filters move to a
  left pane: cameras, object types, zones and known people are multi-select, and
  a confidence threshold is new. The phone layout uses the same gallery and
  filters. `/api/incidents/search` gains `min_confidence` and `days_back`, takes
  comma-separated `camera_id`, `object_label`, `zone` and `person_id`, and
  returns `review_verdict` and `image_version` per item. See
  `docs/guide/incidents.md`.
- Incident detail: unfinished camera views are named again in "Why this is an
  incident", and a followed object keeps a box, drawn as an estimate, on
  recording frames between analysed frames.
- Faster incident gallery: pictures are requested near their shown size and
  kept by the browser when their URL names the file version (`s=`); other image
  requests answer 304 when unchanged. Large snapshots are decoded at reduced
  size for thumbnails, the interface's scripts and styles are served compressed
  with long-lived caching, and the periodic refresh re-reads only the newest
  incidents.
- Admin → General has a per-browser **Font** setting beside Text size: Inter
  (default), Figtree, Manrope, DM Sans, Source Sans or the system font. All are
  bundled; none is fetched from another site.
- New incident page: a plain-language title, a storyline with one lane per
  camera episode and a bar per object's visible interval, object cards that
  focus the player and follow an object across cameras, per-object
  corrections, and a review mode with Reviewed / Keep for later / Nothing here
  verdicts (keys E, F, X; J and K move without deciding). Scene objects now
  carry `visible_intervals`; verdicts are stored per incident and served by
  `GET /api/incidents/review-queue` and `GET`/`PUT /api/incidents/{id}/review`.
  A verdict is feedback only and never changes evidence. See
  `docs/guide/incidents.md`.
- Optional AI incident titles (Admin → AI Provider, off by default) word the
  incident page title with the configured provider from scene facts only, and
  fall back to the built-in title when off or on failure. See
  `docs/guide/assistant.md`.
- Database cleanup runs daily at `database_retention.run_at` (host local time,
  default `03:00`) instead of 10 minutes after every restart. A restart only
  starts a catch-up run when the last completed run began more than
  `interval_hours` ago. See `docs/guide/admin.md`.
- Only data written by `v1.2.0` or later is supported. Startup no longer
  upgrades older databases, `secrets.json` must be the encrypted format, and
  completed exports are always served as MP4. Startup drops retired tables that
  no current build reads (`scene_changes`, `tracking_comparisons`, `scene_migrations`,
  `scene_acquisition_migrations`, `scene_discovery_baselines`) and unused
  indexes. See `docs/current-only.md`.
- Removed endpoints with no caller and no documentation: the
  `/api/faces/` diagnostics, benchmark (`/production`, `/camera-pairs`,
  `/by-identity`), `gallery-optimization`, `camera-suitability`,
  `unknown-cluster-health`, `representation-health`, `gallery-candidates` and
  `maintenance/dedupe` routes, and
  `/api/cameras/{id}/recordings/mobile-window.mp4`.
- Removed config fields `detector.depth.motion_evidence_enabled` and
  `cameras[].video_backend`; files that still contain them load unchanged.
  See `docs/dead-code-cleanup-2026-11.md`.
- Timeline: step the paused player one frame back or forward (buttons beside
  the ±10 s skips, or `,` and `.`), with exact server-decoded frames and their
  time to the millisecond. New endpoints `recordings/frames.json` and
  `recordings/frame.jpg`; see `docs/recording-frame-api.md`.

## SurvNG v1.2

Released as `v1.2.0`.

- Recording indexes and snapshot links stay when the expected volume is
  unmounted or cannot be identified, including after a mountpoint directory
  is left behind. A replacement disk is not enrolled automatically.
- Restoring a camera to the site setting no longer changes what that camera
  does today. Rollback puts the previous override back without the forward
  step limit. A dismissed system-review suggestion cannot be applied.
- One failed semantic retry no longer blocks historical indexing. Database
  cleanup keeps a durable claim on motion-audit files until they are actually
  removed, and stops between batches when cleanup is turned off.
- Incident covers stay on the subject that established the incident, gallery
  times match the selected frame, and detection reserves evidence slots before
  tracking can take them.
- Live camera frames for integrations no longer hold the reload lock, and the
  status stream no longer runs git status. Docker installs disable OpenCV
  OpenCL the same way the systemd unit does.
- Added optional per-camera speech detection (Silero VAD on OpenVINO) with
  local English transcription (Whisper via OpenVINO GenAI; `base.en` on CPU,
  `small.en` on GPU with CPU fallback). Off by default globally and per camera;
  transcripts are searchable through admin-scoped APIs, a Home Assistant
  speech sensor is published over MQTT, and transcript text reaches MQTT only
  when explicitly enabled. Install models with
  `install-docker-models.sh --speech`. See `docs/speech-detection.md`.
- Camera motion that no detector class explains now establishes a motion-only
  incident when the qualifier located that motion inside an eligible zone, so
  animals and other unmodelled subjects appear under **Incidents → Motion**.
  Motion placed only in an Ignore zone, or outside a required incident zone,
  still stays unconfirmed.
- Added frame-crop and object-based **Find similar** searches with persistent
  forensic results, nearby-evidence navigation, and Timeline integration.
- Added optional monocular-depth enrichment for incident objects, depth replay
  overlays, distance policies, and decision-scoped depth health telemetry.
- Added **Admin → Health → Detection at a glance** with site and per-camera
  guidance for admission sources, visual-analysis capacity, detector workers,
  and tracking load.
- Reduced detection contention by deferring tracking until recorded refinement
  completes, reserving detector capacity for tracking, and limiting simultaneous
  EMA analysis fairly across cameras.
- Incident cover images now show the subject that established the incident.
  A subject sampled too few times to confirm no longer loses the cover to a
  sharper empty frame, and cover choice weighs subject sharpness with size.
- Incident evidence thumbnails now show when each cover frame was captured,
  not when its event was triggered, and are ordered by that time.
- Incidents can keep up to four additional evidence images, up from two.
  Object tracking now fills the extra slots with distinct full-detail views
  from across the incident once its session completes.
- Added browser sign-in, local users and API tokens, TLS certificate management,
  and trusted reverse-proxy configuration.
- Added privacy-safe support bundle downloads for remote troubleshooting.
- Added selectable product-update branches for branch-based installations.
- Removed native Reolink Baichuan (`reolink://` / port 9000) ingest; use RTSP
  URLs instead. Legacy `baichuan` config keys are ignored.
- Added MIT `LICENSE`, `NOTICE`, and `THIRD_PARTY.md` for redistribution.

## SurvNG 1.0.0

First stable release of SurvNG: local-first RTSP/ONVIF surveillance with live
viewing, continuous recording, motion qualification, OpenVINO detection,
incidents, timeline investigation, people/face workflows, and an installable
progressive web app shell.

## Highlights

- Live workspace with mobile-first composition and installable PWA shell
- Timeline investigation, incidents, Smart Search, and People review
- Recording retention, multi-disk media storage, and export center
- ONVIF/Reolink camera onboarding and Admin configuration
- Optional AI assistant and motion-audit analysis

Requires a network connection for live cameras and API access. The service
worker caches hashed static assets only.
