# SurvNG

Local-first surveillance app for RTSP/RTMP/ONVIF cameras with recording, camera motion-event ingestion, OpenVINO object detection, and a browser GUI.

See the [SurvNG guide](docs/guide/index.md).
When the app is running, the same guide is available at
`/help` (for example `http://127.0.0.1:8088/survng/help`), also linked from
**Help** in the navigation.

Continuous-recording quotas, age limits, free-space watermarks, and protected
incident media are described in [Multi-disk media storage](docs/storage.md) and
[Storage retention](docs/storage-retention.md).
Configuration saves and their runtime restart boundaries are described in
[Configuration application boundaries](docs/configuration-application.md).
For the optional container installation, persistent-volume layout, migration,
and Intel GPU/QSV support, see the [Docker README](docker/README.md). Generic
deployment and upgrade guidance is in [Docker installation](docs/docker.md).
Local visual-language search setup is documented in [Smart Search model packages](docs/semantic-search.md).
Importing original incident images and model-generated boxes into an external
annotation workflow is documented in the [Training samples API](docs/training-api.md).
The production Hybrid object tracker, optional FastTrack comparison engine, and
evidence-based selection process are covered in [Object tracking](README.tracking.md).

SurvNG is licensed under the [MIT License](LICENSE). Third-party notices for
bundled and notable dependencies are in [NOTICE](NOTICE) and
[THIRD_PARTY.md](THIRD_PARTY.md).

For the end-to-end ingest, motion qualification, inference, incident, recording,
and playback architecture, see [VIDEO_PIPELINE.md](VIDEO_PIPELINE.md). Keep that
document synchronized with changes to any video-processing stage.

## What SurvNG 1.0 Does

- Connects to RTSP, RTMP, HTTP, or file streams through OpenCV/FFmpeg.
- Records streams to segmented MP4 files with FFmpeg.
- Listens for ONVIF pull-point events when the camera supports them, with explicit camera-triggered or adaptive visual-triggered operation.
- Runs an OpenVINO detection pass on the latest frame when motion is reported.
- Serves a React bento-style GUI for live streams, event history, recordings, and camera controls.
- Provides an optional read-only AI assistant for health checks, configuration explanations, incident investigation, and structured incident search.

## AI analysis and assistant

SurvNG reuses one configured AI provider, API key, and optional base URL for
Motion Audit image analysis and the global assistant. Under **Admin → Object
Detection → AI analysis & assistant**, two model roles can be configured:

- **Everyday AI model** is the existing Motion Audit image-analysis model.
  It also handles assistant routing, incident searches, system status, and
  straightforward factual questions.
- **Detailed analysis model** is the optional second model for incident diagnosis, ambiguous
  timelines, comparisons, and tuning advice. Leave it blank to use the fast
  model.

The assistant is the sparkle button at the lower-right of every screen. Normal
chat is read-only: it can query bounded, typed SurvNG tools, but cannot run
commands, restart services, delete media, or send notifications. Evidence
sent to the provider is bounded and strips credentials, stream URLs, web URLs,
and filesystem paths. Responses link back to the camera or incident used as
evidence. Assistant conversation history expires from browser storage after
24 hours of inactivity. Configuration saves for these model fields use the
existing hot AI configuration path and do not restart camera workers.

When incident evidence has a retained snapshot, the assistant response includes
the actual incident image in its source card. Images are served through SurvNG's
existing authenticated event-image endpoint rather than embedded in conversation
storage or returned as provider-hosted URLs. System-health and configuration
answers do not attach unrelated camera images.

When an incident is selected, asking the assistant to **visually analyze this
incident** sends its representative retained image and bounded incident
telemetry through the same configured multimodal AI transport. The deeper model
is used when configured. The review distinguishes likely misses,
misclassifications, false positives, and uncertain single-frame evidence while
keeping detector and tracking assessments separate.

Visual reviews may propose only the existing allowlisted, bounded,
camera-scoped motion settings. SurvNG calculates the displayed before/after
values itself; the model does not. Applying requires **Allow confirmed changes**
to be enabled plus an explicit confirmation in the drawer. A configuration
fingerprint rejects stale proposals if motion settings changed after analysis.
Each apply request also carries a one-hour server-issued proof binding it to the
exact reviewed incident, camera, settings, values, and explanation; edited or
replayed recommendations are rejected. Active AI analysis prevents a camera
manager reload from tearing down resources that the review is still using.
No object thresholds, zones, tracking settings, models, trigger topology, or
global settings can be changed through this incident-review path.

Under **Admin → Camera Intelligence**, a manual per-camera review looks across
up to 100 records from the last 24 hours, 3 days, or 7 days. It balances likely
misses, visual backup triggers, filtered motion, motion-only incidents, and
recognized incidents so frequent routine successes do not hide rarer problems.
Only 4–24 representative images are sent to the provider (12 by default), which
bounds both cost and review time. The report shows the actual reviewed images in
SurvNG, uses plain-language verdicts, and recommends a camera setting only when
multiple samples independently support the same value. Suggestions remain
camera-scoped, configuration-fingerprinted, server-validated, and require an
explicit confirmation before SurvNG applies them.

When a recommendation is applied, SurvNG can measure it after either 24 hours
or 7 days. The original balanced review and exact applied values become the
baseline. Once the observation period ends, **Run follow-up review** analyzes a
new bounded sample and compares likely misses, nuisance alerts, wrong labels,
and results that look correct within like-for-like review categories. A result
without enough category-matched evidence remains inconclusive. The GUI
labels the outcome improved, worsened, or inconclusive and retains the report
across restarts. Follow-up analysis remains manually initiated so it cannot
silently incur provider costs; small changes are explicitly described as
directional evidence rather than proof.

The assistant can also build a bounded cross-camera investigation timeline from
the incident currently open in the viewer. Ask **Trace this incident across
cameras** or name a recognized face and time range. Confirmed face recognition
is treated as a strong identity link, possible face recognition remains
uncertain, and nearby incidents sharing only an object class are labeled as
context candidates rather than the same person or vehicle. Each timeline result
includes its incident image and link. For newly tracked incidents, SurvNG also
stores normalized person and vehicle appearance signatures in a durable local
index. Comparisons are restricted to signatures produced by the exact same ReID
model version. A strong match is reported as visual similarity—not confirmed
identity—because camera angle, lighting, occlusion, and similar-looking subjects
can affect the score. Raw appearance vectors remain server-side and are never
returned by the API or assistant.

## Reolink / ONVIF Notes

ONVIF includes event handling in its network interface specifications, and many Reolink cameras expose ONVIF and RTSP when enabled in the camera network settings. In practice, Reolink event topic names and support vary by model and firmware, so this app logs raw ONVIF event topics and treats events containing `motion`, `cellmotion`, `person`, `vehicle`, `animal`, or `alarm` as detection triggers by default.

If ONVIF motion events are missing or unreliable, select **EMA only** so Enhanced Motion Analysis becomes the sole automatic trigger.

## Requirements

- Python 3.12+
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

Native installs can pull product updates from Git under **Admin → General →
Check for Updates / Update**, or by running `scripts/update-from-git.sh` on the
host. Docker upgrades still rebuild the image from the host checkout; see
[README.install](README.install) section 10.

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
python -m survng.app --reload --host 0.0.0.0 --port 8088
```

`python -m survng.app` reads TLS settings from config and attaches certificate files when HTTPS is enabled. Direct `uvicorn survng.app.main:app` still works for HTTP-only development.

Open http://127.0.0.1:8088/survng/ on this machine, or use the server's LAN address from another device, for example `http://192.168.82.12:8088/survng/`.

### Optional Docker installation

Docker is an alternative deployment path; the native virtualenv/systemd path
remains supported. The production image builds the React UI, installs FFmpeg,
runs SurvNG as a configurable non-root UID/GID, and keeps configuration,
databases, models, and recordings outside the image.

```bash
cp .env.example .env
docker compose build
docker compose up -d
docker compose ps
```

On first start, a private camera-free config is created under
`docker-data/config/config.json`. Open `http://SERVER:8088/survng/` and configure
SurvNG normally. Do not start the container alongside the systemd service: both
would record the same cameras and consume duplicate ONVIF connections. See
[docs/docker.md](docs/docker.md) before migrating an existing installation.

SurvNG can be published on the internet when **browser sign-in is enabled** and
a reverse proxy terminates HTTPS. Keep port `8088` on localhost (or a private
network) and follow [docs/guide/reverse-proxy.md](docs/guide/reverse-proxy.md).
Do not expose the raw SurvNG port. Cross-origin API calls are rejected and
stored credentials are masked in API responses; those controls do not replace
sign-in and a trusted proxy.

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

### Progressive web app

SurvNG 1.0 ships as an installable online shell. Mobile browsers can Add to Home
Screen / Install for a standalone experience. Live cameras and APIs still require
a network connection; the service worker only caches hashed `/static/assets/`
files and never caches `/api/` or media.

Installability works best over HTTPS behind the reverse proxy above
(`X-Forwarded-Proto` must remain accurate so WebRTC uses `wss`).

Pages:

- `/` live camera grid and recent events
- `/recordings` continuous recording review with per-camera navigation
- `/config` camera inventory, add/clone/edit workflow, and ONVIF/Reolink capability detection

## Testing

Run tests through the bounded wrapper so an interrupted or wedged native worker
cannot remain resident after a review campaign:

```bash
scripts/run-tests.sh -q
```

The complete suite has a three-minute default limit. Override it for an
intentional long-running campaign with `SURVNG_TEST_TIMEOUT_SECONDS`.

## Configure A Camera

Copy `config.example.json` to `config.json` for a clean start, or copy `config.reolink.example.json` to `config.json` and replace the placeholders.

SurvNG ingests cameras over RTSP/RTSPS/HTTP/HTTPS URLs (typically via go2rtc restreams). Native Reolink Baichuan (`reolink://` / port 9000) support has been removed; use the camera’s RTSP main/sub URLs instead.

Common Reolink RTSP shapes are:

```text
rtsp://USER:PASSWORD@CAMERA_IP:554/h264Preview_01_main
rtsp://USER:PASSWORD@CAMERA_IP:554/h264Preview_01_sub
```

For ONVIF, use the camera host, ONVIF port, username, and password. Many Reolink cameras use ONVIF port `8000`, but verify the camera model's network settings.

### Stream Source Strategy

Use `stream_url` for the main/high-resolution stream and `live_stream_url` for
the optional low-latency live preview/substream. SurvNG records the main stream
continuously. After a detection request is admitted, SurvNG first checks a
strictly fresh frame already held by live capture so it does not wait for the
open main-recording segment to finalize. This is provisional evidence and is
normally the substream when one is configured. A durable delayed pass then
samples high-resolution main recordings at several timestamps, combines
spatially consistent detections, uses median confidence rather than the highest
outlier, and applies the configured confirmation count. Cover selection is
independent from incident admission: a compatible, materially better main frame
can replace the provisional cover without changing the security decision, and
ambiguous scenes remain unchanged for later tracking-based verification. See
[Incident evidence data path](docs/incident-evidence-data-path.md) for the full
source, timing, geometry, refinement, and cover-promotion contract.

Live camera tiles use equal 16:9 display frames. Admin > Cameras > Settings >
Live view framing can independently center or zoom the Main and Sub streams,
or fit the full source without cropping. These settings affect browser
presentation only: they do not crop recordings, snapshots, object detection,
motion analysis, or exported media. Detection Zones remain independent.

Recorder timestamp handling is discontinuity-aware. Persistent regressing or rebased FFmpeg timestamps are coalesced into one diagnostic event and cause only the affected camera/source to begin a new recording epoch. The old recorder is finalized before its replacement starts, replacement health is verified, and recovery is rate-limited to prevent restart loops. Per-source discontinuity, rollover, failure, and rate-limit counters are exposed with camera status and under Admin > Telemetry.

### Evidence image storage

New incident snapshots and rejected motion-audit samples default to WebP at quality `95`. Change the format or quality under **Admin → Storage → Evidence image storage**, or configure `image_storage.format` (`webp` or `jpeg`) and `image_storage.quality` (`1` through `100`). The change applies immediately without restarting camera workers and affects only newly written evidence; existing JPEG, PNG, and WebP files remain readable.

Evidence is written to a temporary file, flushed, and atomically moved into place so API readers never observe a partial image. File access follows the target storage directory's read permissions without granting group or world write access. If a particular OpenCV deployment cannot encode WebP, SurvNG logs one warning and safely stores JPEG for that image instead of losing incident evidence. Snapshot APIs return the actual media type, and the incident download action preserves the stored file extension.

## Object Detection

Set `model_path` to an OpenVINO-readable model, such as `best.onnx` or an OpenVINO IR `.xml` file. Set `labels_path` to a newline-delimited class file such as `classes.txt`.

`object_worker_count` controls how many independent OpenVINO object-detector processes SurvNG starts (`1` through `4`, default `1`). Requests are sent to the least-busy process and can fail over when another process is unavailable. Two workers can reduce inference queue delay when several cameras trigger together, at the cost of loading the model twice and consuming additional accelerator and application memory. Core ML always uses one object worker. Configure this under **Admin → Object Detection → Parallel detectors** and confirm the online count, aggregate response time, queue depth, GPU activity, and worker memory under **Admin → Telemetry** before increasing it further.

`event_confirmation_frames` controls the global temporal confirmation requirement from one to five frames. `event_class_confirmation_frames` maps individual model labels to optional overrides, for example `{"robot_lawnmower": 3}`. `event_class_confidence_thresholds` provides the matching per-label confidence override, for example `{"robot_lawnmower": 0.75}`. SurvNG runs inference at the lowest applicable global, class, or zone threshold and then applies the correct threshold to each result, so class overrides may safely be either higher or lower than the global setting. Once confirmation is met, recorded refinement stops early within the current stage instead of inferring every remaining offset. `event_refinement_stages` and `event_refinement_retry_seconds` control the recorded sample window and how long refinement may wait for finalized segments; tighter budgets free the shared detector sooner after each event.

The detector supports YOLO-style ONNX output shaped like `[1, 4 + classes, anchors]` and SSD-style output shaped like `[image_id, label, confidence, xmin, ymin, xmax, ymax]`.

On macOS, the detector can also use Core ML. Set `detector.backend` to `coreml` and `coreml_model_path` to a `.mlmodel` or `.mlpackage` detection model. If Core ML is unavailable or the Core ML model cannot load, the detector falls back to the configured OpenVINO/ONNX model.

The detector is optional. If OpenVINO or the model is missing, the app records a motion event and reports `detector_unavailable` instead of failing the camera loop.

### Motion Qualification

SurvNG learns each scene from a bounded, latest-frame analysis worker that is isolated from the live capture loop. Trigger selection and visual validation are separate decisions.

The guided editor exposes four motion behaviors. The first two share the
`camera` trigger mode and differ only in whether EMA validates ordinary camera
notices:

| GUI option | Configuration | What starts object detection? | Filtering behavior |
| --- | --- | --- | --- |
| **Camera only** | `camera` + validation bypass | Camera ONVIF or manual notices only | Every ordinary camera notice proceeds directly to object detection |
| **Camera + EMA validation** | `camera` + EMA validation | Camera ONVIF or manual notices only | EMA validates ordinary camera notices but cannot trigger detection independently |
| **Camera + EMA backup (default)** | `camera_rescue` | Camera ONVIF notices, or exceptionally strong persistent EMA motion without an admitted camera request | ONVIF stays primary; one episode owner merges both sources and bounds detector work; an eligible object must overlap that motion or move across detector samples |
| **EMA only** | `adaptive` | Accepted EMA motion or manual tests | Ordinary ONVIF notices are diagnostics only |

Camera only and Camera + EMA validation never allow EMA to create an event. Camera + EMA backup preserves the camera as primary but permits a tightly bounded EMA backup attempt. After the configured warmup, backup triggering waits for a quiet scene baseline; an eligible object must then overlap the credible EMA motion region or demonstrate movement across detector samples before an incident is created. This prevents a parked vehicle or other stationary object elsewhere in the frame from validating unrelated shadows, vegetation, or exposure changes. EMA only never allows ordinary ONVIF notices to create an event. If EMA validation is unavailable or still warming, camera-triggered events fail open so object detection still runs. Each camera can inherit or override the global mode, sensitivity, and configured qualification, observation, or fusion stage graph. Empty global graph lists retain the built-in pipeline.

Adaptive analysis uses the camera's live feed: `live_stream_url` when a
substream is configured, otherwise `stream_url`. Frames are downscaled to
`frame_width` (320 px by default), converted to grayscale, and sampled at
`sample_fps` (5 FPS by default). After a trigger, a fresh live frame supplies
the provisional low-latency check; durable temporal refinement and normal cover
selection use high-resolution frames from the main recording.

All cameras remain eligible for analysis, but a shared application semaphore permits at most two cameras to execute the CPU-heavy visual pipeline simultaneously. Each camera has a bounded latest-frame queue, so stale pending analysis is replaced instead of accumulating. This limit does not restrict capture, continuous recording, live view, or the number of cameras with motion detection enabled.

The built-in decision graph uses EMA validation. The guided selector configures validation as part of the chosen behavior; Camera + EMA backup and EMA only require EMA. Incomplete or retired stage graphs are rejected before config is saved.

Use the guided **Motion behavior** panel under **Admin > Camera Settings > Motion/Object** to select the complete trigger and validation behavior in one control. Each camera can inherit the global choice or select a camera-specific behavior. Advanced stage graphs created outside the GUI remain protected until explicitly replaced with a guided behavior.

The default and recommended setup is **Camera + EMA backup**. ONVIF remains primary, EMA validates ordinary notices, and tightly bounded EMA rescue can recover an incident when the camera fails to notify SurvNG. See [Motion triggers and validation](docs/adaptive-motion.md) for the complete data flow and performance model.

When the runtime stage catalog at `GET /api/motion/pipeline/catalog` exposes two or more supported motion-analysis presets, the GUI automatically displays a **Motion analysis method** selector. With only the standard EMA pipeline available, SurvNG hides the redundant selector. Externally configured custom pipelines remain active and protected and receive a compact read-only notice. Camera Settings shows the effective analysis, observation, and decision graphs with live cycle counts, failures, and per-stage timing.

Stages with the same non-empty `parallel_group` run concurrently when they are adjacent in a graph. Each branch receives an isolated `MotionContext`; the pipeline merges only artifacts declared by the stage registrations. Conflicting non-mergeable outputs are rejected during configuration validation. Built-in ONVIF evidence is event-based, while future registered observation plugins may independently consume their supported observation kinds.

Camera Settings also includes an on-demand **Motion Diagnostics** viewer for the original and processed frame, difference image, threshold and cleaned masks, blob/track overlay, decision score, and stage timings. Diagnostics are limited to one selected camera and one capture per second, retain only the latest encoded snapshot in memory, stop when the viewer closes, and expire automatically if the browser disconnects.

ONVIF event evidence is stored in the per-camera, thread-safe evidence repository for diagnostics. It is never presented as a visual validator in the guided configuration.

## Face Recognition

SurvNG stores detected face observations separately from object detections. Install the default OpenVINO embedding model with:

```bash
scripts/install-face-model.sh
```

Then enable face recognition under Admin > General > Object Detection. Set the embedding model to `models/face_model/face-recognition-resnet100-arcface-onnx.xml`, the landmark model to `models/face_model/landmarks-regression-retail-0009.xml`, and the detector model to `models/face_detector/face-detection-retail-0004.xml`. The dedicated detector supplies accurate face boxes without creating face-only incidents. SurvNG selects the clearest face from the recorded temporal samples, aligns five landmarks, and generates a 512-dimensional ArcFace embedding.

Confirmed observations form each person's trusted reference gallery. SurvNG selects a quality-weighted, identity-consistent, camera-diverse subset rather than simply using the newest images. A pinned reference is always retained. New matches remain reviewable suggestions by default; optional automatic identification requires a higher score, a clear lead over the next person, at least three supporting references, and a sufficiently good input image. Automatically identified observations do not become references until a person confirms them, preventing recognition errors from teaching the gallery. Model binaries are intentionally excluded from Git; the installer restores them on a new server.

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

### API authentication

SurvNG supports optional scoped long-lived bearer tokens for integrations and
automation clients. Authentication is disabled by default for compatibility
with trusted-LAN and existing reverse-proxy deployments. Create a Home
Assistant token from the repository root:

```bash
.venv/bin/python scripts/create-api-token.py \
  --id home-assistant \
  --name "Home Assistant" \
  --scope read \
  --scope camera:control \
  --enable
```

The command prints the bearer token once and stores only its SHA-256 digest in
the active SurvNG configuration. Send it on API requests as:

```text
Authorization: Bearer <SURVNG_TOKEN>
```

List configured credentials (metadata only; secrets and hashes are never
shown), or delete one by its stable ID:

```bash
.venv/bin/python scripts/create-api-token.py list
.venv/bin/python scripts/create-api-token.py delete --id home-assistant
```

Tokens can also be created and deleted from **Admin → General → API**.
New secrets are shown once. Creating a token does not enable enforcement.

Available scopes are `read`, `camera:control`, and `admin`. The `admin` scope
includes the other scopes. Camera power, recording, and detection controls need
`camera:control`; other mutations need `admin`. `GET /api/health` intentionally
remains unauthenticated for service supervision.

When authentication is enabled, the SurvNG browser UI also needs the bearer
header for API and WebSocket requests. A reverse proxy can inject that header
after completing its own user authentication. Keep the native SurvNG port on a
trusted network so clients cannot bypass that proxy.

Do not use `--enable` until that header injection is in place. Enabling bearer
authentication without it leaves the page shell reachable but causes its API
requests to return HTTP 401, making the browser interface appear offline.

### Integration stream sources

`GET /api/cameras/CAMERA_ID/stream-source?source=live` returns a versioned,
FFmpeg-readable go2rtc RTSP descriptor for integrations such as Home Assistant.
Use `source=main` for the configured main stream. The descriptor preserves the
configured go2rtc host, port, stream name and native codec while removing all
embedded user information. It returns `404` for an unknown camera and `503`
when the camera or go2rtc stream is unavailable. Stream URLs are operational
secrets and should not be exposed as Home Assistant entity attributes or
diagnostic fields even when they contain no credentials.

SurvNG also publishes a separate `SurvNG Server` device. Its entities report system status (`starting`, `running`, `stopping`, or `restarting`), overall health, current maintenance activity, uptime, camera and recorder counts, CPU and memory load, cached storage capacity, and object-detector status. Server state and metrics are retained at `survng/server/state` and `survng/server/metrics`; state transitions are published without retention at `survng/server/event`. The existing `survng/status` topic remains the availability and Last Will topic for both server and camera entities. An unexpected process or network failure therefore becomes `offline`; graceful shutdown is reported as `stopping` before disconnect.

Server status publishing can be disabled independently of camera messages. The metrics interval defaults to 30 seconds and can be adjusted from 10 to 3600 seconds under Admin > MQTT. Storage metrics reuse the most recent daily retention plan instead of scanning the recording filesystem on every MQTT update. MQTT reports daily projection work as `planning` and collapses bounded cleanup and its short inter-batch waits into the stable `cleaning` activity.

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
