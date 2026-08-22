# SurvNG Docker deployment

This directory contains the container entrypoint and a camera-free example
configuration. The repository root contains the `Dockerfile` and Compose files.
Docker is optional; the native virtualenv/systemd deployment remains supported.

Do not run the native service and Docker container simultaneously. Doing so
opens duplicate video streams, recorders, MQTT clients, and ONVIF subscriptions.

## Files

| File | Purpose |
| --- | --- |
| `Dockerfile` | Multi-stage frontend, Python runtime, and Intel GPU image |
| `compose.yaml` | Base CPU-compatible runtime |
| `compose.intel-gpu.yaml` | Intel OpenVINO GPU and `/dev/dri` override |
| `compose.lxc.yaml` | Explicit AppArmor compatibility override for nested Docker |
| `.env.example` | Non-secret host path, identity, timezone, and GPU group settings |
| `scripts/docker-build-lxc.sh` | Persistent, cached BuildKit workflow for this LXC host |
| `scripts/install-docker-models.sh` | Download detector/ReID/Smart Search models and patch `config.json` |
| `.github/workflows/docker-publish.yml` | Build and push both image targets to GHCR on `v1.0` commits and `v*` tags |
| `docker/config.example.json` | Camera-free initial configuration |
| `docker/go2rtc.example.yaml` | Seeded go2rtc config for the bundled restreamer |

## Persistent storage

The image contains no live configuration, credentials, databases, recordings,
or model binaries. Compose supplies these locations:

| Container path | Contents |
| --- | --- |
| `/config` | Private `config.json` |
| `/data` | SQLite databases, indexes, and OpenVINO cache |
| `/media` | Recordings, snapshots, clips, HLS, and playback cache |
| `/models` | Read-only OpenVINO, face, and ReID models |

Keep `/config` and `/data` on local storage. Mount NFS on the Docker host and
bind that mount to `/media`. Never put camera, MQTT, or AI credentials in `.env`
or the image; they remain in the mode-`0600` configuration bind mount.

## Model packages

The GHCR image does not include detector, ReID, or Smart Search weights.
Download them on the Docker host and bind-mount the directory at `/models`:

```bash
scripts/install-docker-models.sh --device GPU
```

Omit `--device GPU` on CPU-only hosts. The installer is idempotent: existing
files are left in place unless you pass `--force`. It writes container paths
into `docker-data/config/config.json` (creating that file from
`docker/config.example.json` when needed) and does not replace cameras or
other settings.

| Host path under `SURVNG_MODELS_DIR` | Container path | License |
| --- | --- | --- |
| `yolo26s_openvino_model/yolo26s.xml` | `/models/yolo26s_openvino_model/yolo26s.xml` | Ultralytics AGPL-3.0 |
| `person_reid_model/person-reidentification-retail-0286.xml` | `/models/person_reid_model/person-reidentification-retail-0286.xml` | Intel OMZ Apache-2.0 |
| `vehicle_reid_model/vehicle-reid-0001.onnx` | `/models/vehicle_reid_model/vehicle-reid-0001.onnx` | MIT |
| `mobileclip2-b-openvino-fp16/` | `/models/mobileclip2-b-openvino-fp16` | Apple ML Research (non-commercial) |

The detector folder name must end in `_openvino_model`. Use `--skip-semantic`
to omit MobileCLIP2-B, `--skip-detector` if you already have a custom OpenVINO
detector, and `--no-enable` to write paths without turning the features on.
Face models remain optional via `scripts/install-face-model.sh`.

## Build

On a normal Docker host, build the Intel image with:

```bash
docker compose -f compose.yaml -f compose.intel-gpu.yaml build
```

On this Proxmox LXC, Docker's embedded BuildKit cannot apply its inner
`docker-default` AppArmor profile. Use the supplied helper instead:

```bash
cd /root/SurvNG
scripts/docker-build-lxc.sh
```

The helper builds the `runtime-intel` target as `survng:local`. On first use it
creates a persistent privileged, AppArmor-unconfined BuildKit worker named
`survng-buildkit`, registers the `survng-lxc` builder, and retains build cache in
the `survng-buildkit-state` volume. Later builds reuse that worker and cache.
Only run it against trusted source. The resulting SurvNG application container
is not privileged.

### FFmpeg

Both image targets install FFmpeg **8.1.2** from `ppa:ubuntuhandbook1/ffmpeg8`
(`ffmpeg=10:8.1.2-0build1~ubuntu24.04`), held so apt cannot silently roll back
to Noble's 6.1 package. Override `FFMPEG_VERSION` at build time only when
intentionally qualifying a new build.

```bash
docker exec survng ffmpeg -version | head -1
docker exec survng ffmpeg -hide_banner -hwaccels
```

### go2rtc

The image ships **go2rtc v1.9.14** at `/usr/local/bin/go2rtc`. The entrypoint
starts it by default before SurvNG, using `/config/go2rtc.yaml` (seeded from
`docker/go2rtc.example.yaml` on first boot).

- API listens on `127.0.0.1:1984` (SurvNG's default adapter port; not published)
- RTSP listens on `:8554` for restreams
- Point SurvNG camera `stream_url` / `live_stream_url` at
  `rtsp://127.0.0.1:8554/<stream_name>`
- Set `SURVNG_GO2RTC=0` to disable the bundled process and use an external
  go2rtc instead
- Override the config path with `SURVNG_GO2RTC_CONFIG` when needed

```bash
docker exec survng go2rtc -version
docker exec survng wget -qO- http://127.0.0.1:1984/api/streams || true
```

### Intel GPU userspace

The Intel target uses Ubuntu 24.04 and pins the GPU userspace versions verified
on the prototype host: Intel compute runtime **26.27.39122.14**, IGC **2.38.5**,
Level Zero **1.32.0**, media driver **26.2.2**, and oneVPL **2.16**. The kernel
driver still comes from the Docker host through `/dev/dri`; it is never installed
in the image. Update the version build arguments together and rebuild when
intentionally qualifying a new Intel stack.

After rebuilding, verify the installed packages and OpenVINO device discovery:

```bash
docker exec survng dpkg-query -W \
  ffmpeg intel-opencl-icd libigc2 libze-intel-gpu1 libze1 \
  intel-media-va-driver-non-free libmfx-gen1.2 libvpl2

docker exec survng /opt/survng-venv/bin/python -c \
  'from openvino import Core; core = Core(); print(core.available_devices)'
```

To build the CPU image in this LXC:

```bash
scripts/docker-build-lxc.sh runtime
```

## Map the current native installation

The current systemd unit runs as root from `/root/SurvNG`. Its recordings are on
the root-squashed NFS mount `/mnt/frigate/SurvNG`, its databases are under
`runtime`, and the Intel devices use video GID 44 and render GID 993. Use this
`.env` for a compatibility-first migration:

```dotenv
TZ=America/New_York

SURVNG_UID=0
SURVNG_GID=0

SURVNG_CONFIG_DIR=/root/SurvNG/docker-data/config
SURVNG_DATA_DIR=/root/SurvNG/runtime
SURVNG_MODELS_DIR=/root/SurvNG/docker-data/models
SURVNG_MEDIA_DIR=/mnt/frigate/SurvNG

SURVNG_VIDEO_GID=44
SURVNG_RENDER_GID=993
```

UID/GID `0:0` initially matches the existing root-run service and the NFS
root-squash behavior. The Docker container is still not privileged. Converting
to a dedicated non-root identity should be a separate migration that also
changes the NFS ownership or ACLs.

Prepare isolated configuration and model mounts:

```bash
cd /root/SurvNG

install -d -m 700 docker-data/config docker-data/models
install -m 600 config.json docker-data/config/config.json

cp -a \
  072926_openvino_model \
  face_model \
  person_reid_model \
  vehicle_reid_model \
  docker-data/models/
```

Change only filesystem paths in the Docker configuration copy:

```bash
.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

path = Path("docker-data/config/config.json")
config = json.loads(path.read_text())

config["storage_dir"] = "/media"
config["database_dir"] = "/data/database"
config["recording_index_dir"] = "/data/recording-index"
config["ffmpeg_path"] = "/usr/bin/ffmpeg"

detector = config["detector"]
detector["model_path"] = "/models/072926_openvino_model/best.xml"
detector["cache_dir"] = "/data/openvino-cache"
detector["face_embedding_model_path"] = (
    "/models/face_model/face-recognition-resnet100-arcface-onnx.xml"
)
detector["face_landmark_model_path"] = (
    "/models/face_model/landmarks-regression-retail-0009.xml"
)

tracking = detector["tracking"]
tracking["reid_model_path"] = (
    "/models/person_reid_model/person-reidentification-retail-0286.xml"
)
tracking["vehicle_reid_model_path"] = (
    "/models/vehicle_reid_model/vehicle-reid-0001.onnx"
)

path.write_text(json.dumps(config, indent=2) + "\n")
os.chmod(path, 0o600)
PY
```

This maps the existing databases directly:

- `/root/SurvNG/runtime/database` to `/data/database`
- `/root/SurvNG/runtime/recording-index` to `/data/recording-index`
- `/mnt/frigate/SurvNG` to `/media`

OpenVINO creates a container-specific compiled cache under
`/root/SurvNG/runtime/openvino-cache`; old compiled blobs do not need to be
copied.

## Validate and switch

Render and validate the merged configuration before stopping the native service:

```bash
docker compose \
  -f compose.yaml \
  -f compose.intel-gpu.yaml \
  -f compose.lxc.yaml \
  config
```

Then switch implementations cleanly:

```bash
systemctl stop survng.service

docker compose \
  -f compose.yaml \
  -f compose.intel-gpu.yaml \
  -f compose.lxc.yaml \
  up -d --no-build
```

Verify service health, inference devices, and FFmpeg acceleration:

```bash
curl -fsS http://127.0.0.1:8088/api/health

docker exec survng python -c \
  'from openvino import Core; print(Core().available_devices)'

docker exec survng ffmpeg -hide_banner -hwaccels
```

OpenVINO should report both `CPU` and `GPU`. FFmpeg should list `qsv`, `vaapi`,
`opencl`, and `vulkan`. Then verify Admin > Telemetry, live video, recordings,
object detection, ONVIF events, and MQTT before disabling the native unit.

## Rollback

```bash
docker compose \
  -f compose.yaml \
  -f compose.intel-gpu.yaml \
  -f compose.lxc.yaml \
  down

systemctl start survng.service
```

The Docker config copy is independent, while the native and Docker deployments
reuse the same database and recording directories sequentially. Never start both
at once.

## Operations

```bash
docker compose \
  -f compose.yaml \
  -f compose.intel-gpu.yaml \
  -f compose.lxc.yaml \
  ps

docker compose \
  -f compose.yaml \
  -f compose.intel-gpu.yaml \
  -f compose.lxc.yaml \
  logs --tail=200 survng
```

Docker sends `SIGTERM` with a 60-second grace period. The entrypoint first asks
SurvNG to release ONVIF PullPoint subscriptions, then performs normal shutdown.
Avoid `docker kill`, which bypasses that cleanup.

See [`docs/docker.md`](../docs/docker.md) for the generic installation,
backup/upgrade guidance, Docker Desktop differences, and building on another
host.
