# Admin

**Admin** answers: how is SurvNG configured and performing?

![Camera settings in Admin](images/admin-cameras.png)

Admin is grouped into practical jobs.

## Configure

| Area | What you do there |
| --- | --- |
| **Cameras** | Add, edit, clone, and order cameras; motion behavior; zones |
| **Detection** | Object detection, face recognition, Smart Search, AI analysis models |
| **Storage** | Retention days, free-space watermarks, evidence image format |
| **Integrations** | MQTT and Home Assistant discovery |
| **Access** | Browser users, admin/viewer roles, HTTPS, and trusted reverse proxies |
| **Server** | General service options, appearance, updates |

## Observe

| Area | What you do there |
| --- | --- |
| **Health** | System status, camera availability, detector load |
| **Audit** | Motion samples that did not become incidents |
| **Logs** | Recent application messages |

## Act

| Area | What you do there |
| --- | --- |
| **Tune-Up** | Guided detection calibration with before/after monitoring |
| **Diagnostics** | Deeper telemetry captures |
| **Maintenance** | Storage cleanup and repair tools |
| **Camera Advisor** | Review a camera’s recent samples and confirm bounded setting changes |

## Suggested first Admin pass

1. **Cameras** — add devices and confirm streams
2. **Storage** — set retention you can afford on disk
3. **Detection** — enable only what you need (objects, faces, Smart Search)
4. **Health** — confirm cameras are online and recording
5. **Audit** — after a day, skim rejected motion to see if zones or sensitivity need a nudge

## Configuration saves

Some settings apply immediately. Others restart camera workers or heavier services. SurvNG tells you when a change needs a broader reload. Technical detail: [Configuration application boundaries](../configuration-application.md).

## Host-local runtime status

For host-side troubleshooting, SurvNG exposes a read-only runtime snapshot over
an owner-only Unix socket. It is not an HTTP endpoint and does not expose
passwords, API tokens, private keys, stream URLs, or raw errors.

After the service starts, run this on the SurvNG host:

```bash
./survngctl status
```

The snapshot includes effective tracking limits and current capacity, camera
health, detector activity, storage status, and a recent in-memory log tail.
The log tail contains only timestamp, level, logger, and message fields; it is
size-limited, credential-redacted, and has URLs removed. Tracebacks and
structured logging extras are not exposed. The default socket is
`/run/survng/observability.sock`; it is accessible only to the SurvNG service
owner (or root). Use `--observability-socket /absolute/path.sock` when starting
SurvNG if your service needs a different runtime directory.

## Depth estimation

**Detection → Depth Estimation** configures optional monocular depth enrichment
for representative incident frames. Enable it only after installing a compatible
OpenVINO depth model and setting its container or host path. The default model
installer supplies `yolo26n-depth_openvino_model/yolo26n-depth.xml`.

- **Minimum/Maximum Distance** calibrate the range represented by the model.
- **Ignore Incidents Beyond** optionally makes objects past that estimated
  distance ineligible.
- **Store representative depth heatmap** retains a small heatmap with incident
  evidence for later inspection.

Monocular distance is an estimate, not a physical measurement. Validate it
against known distances before using a distance limit for incident decisions.
Under **Health → Telemetry → Object activity**, **Depth-shadow health** is
informational: it shows how depth would have affected motion attribution and
does not itself change admission.

## Detection at a glance

Open **Health → Detection at a glance** for a seven-day site summary and
per-camera findings. The cards combine incident history with live detector
health:

- **Admission** explains whether incidents started from camera notices or visual
  backup. A high visual-backup share does not by itself mean ONVIF is dead; the
  card also checks whether camera notices are still arriving and correlating.
- **Detection engine** reports configured and running detector workers, queue
  pressure, failures, and whether tracking retains a detector lane.
- **Tracking** reports sessions that waited or were skipped when detector
  capacity was busy.
- **Visual analysis** reports whether the
  `motion_qualification.max_concurrent_analysis` limit is delaying EMA analysis.

Use each card's settings link to adjust the owning control, then watch the
summary again before making another change. For deeper raw counters, switch to
**Health → Telemetry**.

## Updates and support

Under **Server Preferences**, choose the release branch used by **Check for
updates** and product updates. A branch can be selected only when the
installation is a Git checkout with that branch available; Docker installations
update by pulling a new image instead.

For remote troubleshooting, open **Diagnostics** and choose **Download support
bundle**. The JSON bundle includes bounded configuration, runtime, telemetry,
and recent-log context with secrets redacted. Review it before sharing it.

## Related

- [Cameras](cameras.md)
- [Motion & detection](motion-detection.md)
- [Recordings & storage](storage.md)
- [Integrations](integrations.md)
- [Access](access.md)
- [Reverse proxy](reverse-proxy.md)
