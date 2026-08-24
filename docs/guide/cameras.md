# Cameras

Cameras are the heart of SurvNG. Each camera needs at least a video stream SurvNG can read. Motion notices and a second live stream are optional but helpful.

## What to collect from the camera

| Setting | Why it matters |
| --- | --- |
| Name | Human label in Live, Incidents, and Timeline |
| Main stream URL | Continuous recording and detailed evidence |
| Live / sub stream URL | Smooth live view and lighter motion analysis |
| ONVIF host/port/user/password | Camera-sent motion and person/vehicle notices |
| Time zone awareness | SurvNG displays times in your configured zone |

SurvNG accepts RTSP, RTSPS, HTTP, and similar FFmpeg-readable URLs. Native proprietary camera URLs that are not RTSP are outside the normal path — use the camera’s RTSP main/sub links.

## Add a camera (GUI)

1. **Admin → Cameras →** add camera
2. Fill name and stream URLs
3. Optionally fill ONVIF fields
4. Save and wait for the tile to come online on **Live**

You can also clone an existing camera when several devices share similar settings.

## Probe before you commit

Admin can probe camera capabilities. Use that when you are unsure which ONVIF port or stream path a vendor expects.

## Reolink notes

Many Reolink cameras expose RTSP and ONVIF after you enable them in the camera’s network settings. Event topic names vary by model and firmware. SurvNG logs raw ONVIF topics and treats topics containing words like `motion`, `person`, or `vehicle` as useful triggers.

If ONVIF notices are missing or flaky, set that camera’s motion behavior to **EMA only**.

## Main vs live streams

- Record the **main** stream for history.
- Watch the **sub** stream live when it looks good enough.
- After motion, SurvNG may use a fresh live frame for a quick check, then refine with high-resolution frames from the main recording.

Technical path: [Incident evidence data path](../incident-evidence-data-path.md).

## Camera power and recording toggles

You can start/stop a camera’s capture or recording from Admin controls and, when MQTT is enabled, from Home Assistant. Stopping recording pauses history for that camera; it does not delete what was already saved.

## Related

- [Getting started](getting-started.md)
- [Motion & detection](motion-detection.md)
- [Zones](zones.md)
- [Integrations](integrations.md)
