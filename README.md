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
