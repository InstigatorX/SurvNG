# SurvNG

Local-first surveillance app for RTSP/RTMP/ONVIF cameras with recording, camera motion-event ingestion, OpenVINO object detection, and a browser GUI.

For the end-to-end ingest, motion qualification, inference, incident, recording,
and playback architecture, see [VIDEO_PIPELINE.md](VIDEO_PIPELINE.md). Keep that
document synchronized with changes to any video-processing stage.

## What This MVP Does

- Connects to RTSP, RTMP, HTTP, or file streams through OpenCV/FFmpeg.
- Records streams to segmented MP4 files with FFmpeg.
- Listens for ONVIF pull-point events when the camera supports them, with explicit camera-triggered or adaptive visual-triggered operation.
- Runs an OpenVINO detection pass on the latest frame when motion is reported.
- Serves a React bento-style GUI for live streams, event history, recordings, and camera controls.

## Reolink / ONVIF Notes

ONVIF includes event handling in its network interface specifications, and many Reolink cameras expose ONVIF and RTSP when enabled in the camera network settings. In practice, Reolink event topic names and support vary by model and firmware, so this app logs raw ONVIF event topics and treats events containing `motion`, `cellmotion`, `person`, `vehicle`, `animal`, or `alarm` as detection triggers by default.

If ONVIF motion events are missing or unreliable, select **Visual-triggered** so adaptive visual analysis becomes the sole automatic trigger.

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

### Intel GPU runtime upgrade

OpenVINO's Intel GPU plug-in depends on the host OpenCL runtime and Intel
Graphics Compiler (IGC). Ubuntu 24.04's standard repository may provide an
older compute stack than the installed OpenVINO release. The following upgrade
uses Intel's Ubuntu graphics PPA for a coordinated compute and QSV/media stack.
See Intel's [Ubuntu graphics package guide](https://dgpu-docs.intel.com/installation-guides/installing-packages-from-the-intel-ppa.html)
for the upstream repository procedure.

These instructions are written for the SurvNG Ubuntu 24.04 host using a
Proxmox kernel. Keep the kernel's existing `i915` driver. Do **not** install
`intel-i915-dkms` or replace the Proxmox kernel as part of this procedure.

Capture the current package and repository state:

```bash
sudo -i
cd /root/SurvNG

stamp=$(date +%Y%m%d-%H%M%S)
backup="/root/intel-gpu-backup-$stamp"
mkdir -p "$backup"
dpkg-query -W > "$backup/packages.txt"
apt-mark showmanual > "$backup/manual-packages.txt"
cp -a /etc/apt/sources.list /etc/apt/sources.list.d "$backup/"
clinfo > "$backup/clinfo.txt"
vainfo --display drm --device /dev/dri/renderD128 > "$backup/vainfo.txt" 2>&1
```

Add Intel's repository and inspect the new candidates:

```bash
apt-get update
apt-get install -y software-properties-common
add-apt-repository -y ppa:kobuk-team/intel-graphics
apt-get update

apt-cache policy intel-opencl-icd libigc2 libze-intel-gpu1 libze1 \
  libigdgmm12 intel-media-va-driver-non-free libmfx-gen1.2 libvpl2
```

Simulate the coordinated userspace upgrade before changing the host:

```bash
apt-get --simulate install \
  intel-opencl-icd libze-intel-gpu1 libze1 clinfo \
  intel-media-va-driver-non-free libmfx-gen1.2 libvpl2 libvpl-tools vainfo \
  | tee "$backup/apt-simulation.txt"

grep -E '^(Inst|Remv)|linux-image|pve|proxmox|i915-dkms' \
  "$backup/apt-simulation.txt"
```

Stop if the simulation proposes removing Proxmox packages or installing
`intel-i915-dkms`. Otherwise, install the userspace packages together:

```bash
systemctl stop survng.service
apt-get install -y \
  intel-opencl-icd libze-intel-gpu1 libze1 clinfo \
  intel-media-va-driver-non-free libmfx-gen1.2 libvpl2 libvpl-tools vainfo

cd /root/SurvNG
if [ -d .cache/openvino ]; then
  mv .cache/openvino ".cache/openvino-before-intel-$stamp"
fi
mkdir -p .cache/openvino
reboot
```

After reconnecting, verify OpenCL, VA-API/QSV, OpenVINO, and clean service
shutdown. The first model load recompiles the OpenVINO cache and is expected to
take longer than subsequent starts.

```bash
cd /root/SurvNG
clinfo -l
vainfo --display drm --device /dev/dri/renderD128
/etc/frigate/custom-ffmpeg/bin/ffmpeg -hide_banner -hwaccels
systemctl status survng.service --no-pager
curl -s http://127.0.0.1:8088/api/system/status | .venv/bin/python -m json.tool

systemctl restart survng.service
sleep 15
journalctl -u survng.service --since "5 minutes ago" --no-pager \
  | grep -E 'shutdown complete|SEGV|core-dump|left-over|SIGKILL'
```

The detector status should report `loaded_device: GPU`, `openvino_loaded: true`,
and an empty `warmup_error`. To roll back the PPA packages:

```bash
sudo -i
systemctl stop survng.service
apt-get install -y ppa-purge
ppa-purge ppa:kobuk-team/intel-graphics
cd /root/SurvNG
rollback_stamp=$(date +%Y%m%d-%H%M%S)
mv .cache/openvino ".cache/openvino-failed-$rollback_stamp"
mkdir -p .cache/openvino
reboot
```

## Run

```bash
uvicorn survng.app.main:app --reload --host 0.0.0.0 --port 8088
```

Open http://127.0.0.1:8088/survng/ on this machine, or use the server's LAN address from another device, for example `http://192.168.82.12:8088/survng/`.

SurvNG's HTTP API is an administrative interface and does not provide its own
user authentication. Keep port `8088` limited to trusted LAN/VPN clients with a
host firewall, or place it behind an authenticated reverse proxy. Do not expose
the port directly to the public internet. SurvNG rejects cross-origin state
changes and masks stored credentials in API responses, but those protections do
not replace network access control and authentication.

### Reverse proxy subpath

`base_path` controls the browser-visible path and defaults to `/survng`. Set it to an empty string to serve browser URLs from `/` instead. SurvNG continues to accept unprefixed backend routes for local API clients.

An nginx proxy only needs to preserve the configured prefix:

```nginx
location = /survng {
    return 301 /survng/;
}

location /survng/ {
    proxy_pass http://127.0.0.1:8088;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection $connection_upgrade;
    proxy_buffering off;
    proxy_read_timeout 3600s;
}
```

The `proxy_pass` URL intentionally has no trailing slash so nginx forwards `/survng`. SurvNG strips the configured prefix internally for HTTP, SSE, HLS, and WebSocket routes.

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

### Motion Qualification

SurvNG learns each scene from a bounded, latest-frame analysis worker that is isolated from the live capture loop. Trigger selection and visual validation are separate decisions.

The GUI exposes two trigger modes:

| GUI option | Configuration | What starts object detection? | Filtering behavior |
| --- | --- | --- | --- |
| **Camera-triggered (Recommended)** | `camera` | Camera ONVIF or manual notices only | Adaptive and MOG2 validation are optional; priority semantic notices bypass validation |
| **Visual-triggered** | `adaptive` | Accepted adaptive visual motion or manual tests | MOG2 may corroborate adaptive motion; ordinary ONVIF notices are diagnostics only |

Camera-triggered mode never allows adaptive analysis or MOG2 to create an event. Visual-triggered mode never allows ordinary ONVIF notices or MOG2 to create an event. If a selected validator is unavailable or still warming, camera-triggered events fail open so object detection still runs. Each camera can inherit or override the global mode, sensitivity, and configured qualification, observation, or fusion stage graph. Empty global graph lists retain the built-in pipeline.

Adaptive analysis uses the camera's live feed: `live_stream_url` when a substream is configured, otherwise `stream_url`. Frames are downscaled to `frame_width` (320 px by default), converted to grayscale, and sampled at `sample_fps` (5 FPS by default). Object detection after a trigger uses high-resolution frames from the main recording.

All cameras remain eligible for analysis, but a shared application semaphore permits at most two cameras to execute the CPU-heavy visual pipeline simultaneously. Each camera has a bounded latest-frame queue, so stale pending analysis is replaced instead of accumulating. This limit does not restrict capture, continuous recording, live view, or the number of cameras with motion detection enabled.

The built-in decision graph uses adaptive validation without MOG2. Guided settings can disable validation, use MOG2 alone, require adaptive and MOG2, or—in camera-triggered mode—allow either validator. Incomplete graphs are rejected before config is saved.

Use the guided **Motion decision** panel under **Config > Detection** to select the trigger source and validators without editing stage graphs. Each camera can inherit the global choice or create a camera-specific policy. Advanced stage graphs created outside the GUI remain protected until explicitly replaced with guided settings.

For the normal setup, select **Camera-triggered with adaptive validation** and click **Use this setup**. ONVIF remains the only automatic trigger, adaptive analysis validates ordinary notices, and MOG2 stays off for lower CPU use. See [Motion triggers and validation](docs/adaptive-motion.md) for the complete data flow and performance model.

The adjacent **Motion analysis method** selector is populated from the runtime stage catalog at `GET /api/motion/pipeline/catalog`. It offers only presets whose implementations are registered and available. Camera Settings shows the effective analysis, observation, and decision graphs with live cycle counts, failures, and per-stage timing.

Stages with the same non-empty `parallel_group` run concurrently when they are adjacent in a graph. Each branch receives an isolated `MotionContext`; the pipeline merges only artifacts declared by the stage registrations. Conflicting non-mergeable outputs are rejected during configuration validation. The built-in MOG2 and ONVIF stages handle different observation kinds, so only the relevant stage runs for each frame or camera event.

Camera Settings also includes an on-demand **Motion Diagnostics** viewer for the original and processed frame, difference image, threshold and cleaned masks, blob/track overlay, decision score, and stage timings. Diagnostics are limited to one selected camera and one capture per second, retain only the latest encoded snapshot in memory, stop when the viewer closes, and expire automatically if the browser disconnects.

MOG2 frame evidence and ONVIF event evidence are independent observation stages backed by the same per-camera, thread-safe repository. ONVIF evidence is never presented as a visual validator in the guided configuration.

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
