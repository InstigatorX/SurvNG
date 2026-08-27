# SurvNG

Local-first video surveillance for RTSP/ONVIF cameras. SurvNG records video,
turns meaningful motion into reviewable incidents, and provides a browser
workspace for live monitoring, playback, search, and administration.

The in-product guide is available through **Help**, or at `/help` on a running
server.

## What SurvNG does

- Connects to RTSP, RTMP, HTTP, and file streams; supports ONVIF camera events.
- Records continuous video with FFmpeg and manages retention across storage
  locations.
- Uses OpenVINO-compatible object detection, motion qualification, zones, and
  optional object tracking to create incidents.
- Provides live view, incident review, timeline playback, exports, search,
  people/face recognition, MQTT, Home Assistant, and a browser-based admin
  workspace.

## Documentation

- [User guide](docs/guide/index.md) — start here for everyday use.
- [Getting started](docs/guide/getting-started.md) — open SurvNG, add a camera,
  and confirm recording.
- [Admin](docs/guide/admin.md), [Cameras](docs/guide/cameras.md), [Motion &
  detection](docs/guide/motion-detection.md), and [Storage](docs/guide/storage.md).
- [Integrations](docs/guide/integrations.md) and the [HTTP API](docs/guide/api.md).
- [Access](docs/guide/access.md) and [Reverse proxy](docs/guide/reverse-proxy.md)
  — sign-in, HTTPS, trusted proxies, and internet exposure.
- [AI assistant](docs/guide/assistant.md), [People](docs/guide/people.md), and
  [Timeline & exports](docs/guide/timeline.md).

Technical references: [video pipeline](VIDEO_PIPELINE.md), [motion triggers and
validation](docs/adaptive-motion.md), [incident evidence data
path](docs/incident-evidence-data-path.md), [configuration application
boundaries](docs/configuration-application.md), and [Docker deployment](docs/docker.md).

## Installation

Installation, upgrades, Docker deployment, native systemd setup, and GPU host
guidance are maintained in [README.install](README.install). Choose either the
[Docker guide](README.install.docker.md) or the [native systemd
guide](README.install.systemd.md); do not run both against the same cameras.

## Deployment and security

For internet-facing use, enable browser sign-in and terminate HTTPS at a trusted
reverse proxy. Keep SurvNG's raw port off the public internet; use loopback,
LAN/VPN firewalling, or the reverse-proxy topology in the [reverse-proxy
guide](docs/guide/reverse-proxy.md).

## Support bundle

For a remote support case, use **Admin → Diagnostics → Download support bundle**.
The redacted JSON bundle contains the safe system context needed for diagnosis;
see [Access](docs/guide/access.md#support-bundle) for the handoff workflow.

## Development

Python 3.12+, Node.js 20+, and FFmpeg are required for local development. Run
the focused tests for the area you change; the frontend production build is:

```bash
npm --prefix frontend run build
```

## License

SurvNG is licensed under the [MIT License](LICENSE). Third-party notices are in
[NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md).
