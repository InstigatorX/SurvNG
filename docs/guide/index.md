# Welcome to SurvNG

SurvNG is software that watches your cameras for you. It keeps a continuous video history, notices when something important happens, and helps you find that moment later — in a browser on your computer or phone.

This guide is written for a first visit. You do not need prior experience with video surveillance systems, object detection, or networking jargon.

## Table of contents

### Start here

1. [Basic ideas](concepts.md) — plain-language words SurvNG uses
2. [First-time setup](getting-started.md) — open the app, add a camera, confirm recording

### Everyday screens

3. [Live view](live.md) — what is happening right now
4. [Incidents](incidents.md) — important events SurvNG kept for you
5. [Timeline & exports](timeline.md) — scrub recorded video and save clips
6. [Search](search.md) — find activity with filters or a short description
7. [People](people.md) — faces SurvNG has seen and people you name
8. [Admin](admin.md) — configure cameras, detection, storage, and health

### How SurvNG decides what matters

9. [Cameras](cameras.md) — connect streams and camera motion notices
10. [Motion & detection](motion-detection.md) — when SurvNG looks closer at a frame
11. [Zones](zones.md) — watch only the parts of the picture that matter
12. [Recordings & storage](storage.md) — how long video is kept and where it lives
13. [AI assistant](assistant.md) — ask questions about health and incidents
14. [Integrations](integrations.md) — Home Assistant, MQTT, and API tokens

### For developers and automation

15. [HTTP API](api.md) — endpoints, authentication, and examples

## Deeper technical notes

These pages go into more detail when you need it. They are still available from the online help under **Reference** links in each topic.

- [Motion triggers and validation](../adaptive-motion.md)
- [Multi-disk media storage](../storage.md)
- [Storage retention](../storage-retention.md)
- [Smart Search model packages](../semantic-search.md)
- [Docker installation](../docker.md)
- [Configuration application boundaries](../configuration-application.md)
- [Training samples API](../training-api.md)
- [Recording frame API](../recording-frame-api.md)
- [Video pipeline overview](../incident-evidence-data-path.md)

## Open this guide in SurvNG

When SurvNG is running, open **Help** in the left navigation, or go to:

```text
http://YOUR-SERVER:8088/survng/help
```

Replace the host and path if you use a different address or `base_path`.
