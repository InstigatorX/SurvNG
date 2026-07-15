# SurvNG

Local-first surveillance app for RTSP/RTMP/ONVIF cameras with recording, camera motion-event ingestion, OpenVINO object detection, and a browser GUI.

## What This MVP Does

- Connects to RTSP, RTMP, HTTP, or file streams through OpenCV/FFmpeg.
- Records streams to segmented MP4 files with FFmpeg.
- Listens for ONVIF pull-point events when the camera supports them.
- Runs an OpenVINO detection pass on the latest frame when motion is reported.
- Serves a React bento-style GUI for live streams, event history, recordings, and camera controls.

## Reolink / ONVIF Notes

ONVIF includes event handling in its network interface specifications, and many Reolink cameras expose ONVIF and RTSP when enabled in the camera network settings. In practice, Reolink event topic names and support vary by model and firmware, so this app logs raw ONVIF event topics and treats events containing `motion`, `cellmotion`, `person`, `vehicle`, `animal`, or `alarm` as detection triggers by default.

If ONVIF events are not available on your exact camera, the app can still record and view streams; the next fallback to add is periodic frame-difference motion detection.

## Requirements

- Python 3.11+
- Node.js 20+ for building the React UI
- FFmpeg available on `PATH`
- A camera with RTSP/RTMP URL, or an ONVIF-capable camera
- Optional: OpenVINO-readable model files, such as `.onnx` or `.xml` + `.bin`

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
npm --prefix frontend install
npm --prefix frontend run build
```

The React frontend in `frontend/` is the default interface. Its production build is emitted into `survng/static/`, which FastAPI serves for `/` and `/recordings`.

## Run

```bash
uvicorn survng.app.main:app --reload --host 0.0.0.0 --port 8088
```

Open http://127.0.0.1:8088/ on this machine, or use the Mac's LAN address from another device, for example `http://192.168.82.12:8088/`.

Pages:

- `/` live camera grid and recent events
- `/recordings` continuous recording review with per-camera navigation
- `/config` camera inventory, add/clone/edit workflow, and ONVIF/Reolink capability detection

## Reolink Baichuan Video

SurvNG can read Reolink Baichuan video directly from the camera on port `9000`.
Set the camera backend to native Baichuan:

```json
{
  "video_backend": "baichuan_native",
  "baichuan": {
    "enabled": true,
    "host": "CAMERA_IP",
    "port": 9000,
    "username": "USER",
    "password": "PASSWORD",
    "channel": 0
  }
}
```

The native reader logs into the camera with Baichuan, starts the main or sub
video stream, parses the Reolink media frames, and pipes H264 into FFmpeg for
HLS, MSE, and recording.

Keep `stream_url` and `live_stream_url` configured as RTSP fallbacks for
snapshots, detection frames, and quick rollback. To disable native Baichuan,
set `video_backend` back to `url`.

The native Baichuan implementation was ported from Neolink.NET protocol behavior
and should be treated as AGPL-derived code if this app is redistributed.

## Configure A Camera

Copy `config.example.json` to `config.json` for a clean start, or copy `config.reolink.example.json` to `config.json` and replace the placeholders.

Common Reolink RTSP shapes are:

```text
rtsp://USER:PASSWORD@CAMERA_IP:554/h264Preview_01_main
rtsp://USER:PASSWORD@CAMERA_IP:554/h264Preview_01_sub
```

For ONVIF, use the camera host, ONVIF port, username, and password. Many Reolink cameras use ONVIF port `8000`, but verify the camera model's network settings.

### Stream Source Strategy

Use `stream_url` for the main/high-resolution stream and `live_stream_url` for the optional low-latency live preview/substream. SurvNG records the main stream continuously. ONVIF motion events trigger object detection by sampling five high-resolution frames from the recorded main stream around the event timestamp, then saving the best annotated frame as the event snapshot.

## Object Detection

Set `model_path` to an OpenVINO-readable model, such as `best.onnx` or an OpenVINO IR `.xml` file. Set `labels_path` to a newline-delimited class file such as `classes.txt`.

The detector supports YOLO-style ONNX output shaped like `[1, 4 + classes, anchors]` and SSD-style output shaped like `[image_id, label, confidence, xmin, ymin, xmax, ymax]`.

On macOS, the detector can also use Core ML. Set `detector.backend` to `coreml` and `coreml_model_path` to a `.mlmodel` or `.mlpackage` detection model. If Core ML is unavailable or the Core ML model cannot load, the detector falls back to the configured OpenVINO/ONNX model.

The detector is optional. If OpenVINO or the model is missing, the app records a motion event and reports `detector_unavailable` instead of failing the camera loop.

## Face Recognition

SurvNG stores detected face observations separately from object detections. Install the default OpenVINO embedding model with:

```bash
scripts/install-face-model.sh
```

Then enable face recognition under Config > General > Object Detection. Set the embedding model to `face_model/face-recognition-resnet100-arcface-onnx.xml` and the landmark model to `face_model/landmarks-regression-retail-0009.xml`. SurvNG aligns each face from five landmarks before generating its 512-dimensional ArcFace embedding. Enroll several clear observations for each person in Faces. New observations are matched asynchronously and remain suggestions until they are confirmed. Model binaries are intentionally excluded from Git; the installer restores them on a new server.

The default ArcFace cosine-similarity threshold is `0.40`. Raise it to reduce false matches or lower it cautiously to recognize more difficult views; thresholds from the previous embedding model are not directly comparable.

## MQTT

Enable MQTT under Config > General and set the broker connection and topic prefix. The default prefix is `survng`.

Published topics:

```text
survng/status
survng/camera/CAMERA_ID/state
survng/camera/CAMERA_ID/motion
survng/camera/CAMERA_ID/object
```

Availability and camera state are retained. Motion payloads include the camera, timestamp, and event source. Object payloads include the event ID, camera, timestamp, aggregate `classes` and `zones`, plus each object's confidence, box, zone match, and zone test point.

Turn a camera and its configured recorders on or off by publishing `ON` or `OFF` to:

```text
survng/camera/CAMERA_ID/power/set
```

The power command also accepts JSON such as `{"state":"OFF"}`. Replace `survng` with the configured topic prefix.

When Home Assistant Discovery is enabled, SurvNG publishes retained entity configuration under `homeassistant/` by default. Each camera appears as a Home Assistant device with a power switch, camera-wide motion and object binary sensors, and a last-object sensor. Every enabled detection zone appears as its own device with an any-object binary sensor and one binary sensor for each class configured on that zone. A zone with no class filter receives sensors for every class in the active detection model. Change the discovery prefix if the Home Assistant MQTT integration uses a non-default prefix.

Zone object events are published under `survng/zone/CAMERA_ID/ZONE/object`; per-class events use `survng/zone/CAMERA_ID/ZONE/class/CLASS`. Camera topics contain every object detected for that camera event and are not filtered by incident-zone eligibility.

## Recording Playback Test

Install the Playwright browser once, then run the recording scrub/soak test against a day with sufficient history:

```bash
npx --prefix frontend playwright install chromium
DATE=2026-07-11 CAMERA=front-door SOURCE=main SOAK_SECONDS=90 SCRUBS=12 \
  npm --prefix frontend run test:recordings-soak
```

The test fails on media errors, stalled playback, failed segment requests, unstable positional fragment URLs, scrubber/video misalignment, or inadequate timeline advancement. Expected request cancellation caused by moving between scrub positions is ignored.

To also verify that today's timeline discovers newly finalized segments while the page remains open:

```bash
CHECK_GROWTH=1 CAMERA=front-door SOURCE=main SOAK_SECONDS=30 SCRUBS=8 \
  npm --prefix frontend run test:recordings-soak
```

Real Safari validation must run on a Mac because Linux Chromium does not provide Safari's HEVC pipeline. Open the same camera/day in Safari, play through several segment boundaries, scrub to widely separated times, and confirm playback resumes without a black frame. Use an H.265 camera such as Upper Garage to exercise `hvc1` playback.
