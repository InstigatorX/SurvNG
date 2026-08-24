# Admin

**Admin** answers: how is SurvNG configured and performing?

Admin is grouped into practical jobs.

## Configure

| Area | What you do there |
| --- | --- |
| **Cameras** | Add, edit, clone, and order cameras; motion behavior; zones |
| **Detection** | Object detection, face recognition, Smart Search, AI analysis models |
| **Storage** | Retention days, free-space watermarks, evidence image format |
| **Integrations** | MQTT and Home Assistant discovery |
| **Server** | General service options, appearance, updates, API tokens |

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

## Related

- [Cameras](cameras.md)
- [Motion & detection](motion-detection.md)
- [Recordings & storage](storage.md)
- [Integrations](integrations.md)
