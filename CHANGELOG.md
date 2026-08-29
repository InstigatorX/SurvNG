# Changelog

## SurvNG v1.2

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
