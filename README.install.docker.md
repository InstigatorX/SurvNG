# SurvNG Docker installation

This path pulls the published image. You do **not** need a Git checkout, Python,
Node.js, OpenVINO, or FFmpeg on the host. You need Docker Engine with the
Compose plugin, and the host must reach your cameras, MQTT broker, and media
disk.

The container uses `network_mode: host`, so port 8088 is on the host. Do not
run this next to `survng.service`.

Do **not** set `SURVNG_UID=0`. The image starts as root only to remap its
internal user, then drops to the UID/GID below.

## 1. Set these values, then paste the rest

Edit the four assignments if you need to. `1500` is an example; it must not
already be a UID or GID on the host.

```bash
SURVNG_UID=1500
SURVNG_GID=1500
SURVNG_TZ=America/New_York
SURVNG_MEDIA_DIR=/srv/survng/media
SURVNG_IMAGE=ghcr.io/instigatorx/survng:v1.2

getent passwd "$SURVNG_UID" || true
getent group "$SURVNG_GID" || true
docker compose version
```

Intel iGPU (optional). If `/dev/dri` is missing, skip GPU and keep the CPU
image:

```bash
ls -l /dev/dri
getent group video render
SURVNG_VIDEO_GID="$(getent group video | cut -d: -f3)"
SURVNG_RENDER_GID="$(getent group render | cut -d: -f3)"
printf 'SURVNG_VIDEO_GID=%s SURVNG_RENDER_GID=%s\n' "$SURVNG_VIDEO_GID" "$SURVNG_RENDER_GID"
```

If the GHCR package is private:

```bash
docker login ghcr.io
```

## 2. Create the SurvNG user and directories

This host user owns bind-mounted files. Leave it **out** of the `docker` group.

```bash
getent group survng >/dev/null \
  || sudo groupadd --system --gid "$SURVNG_GID" survng
getent passwd survng >/dev/null \
  || sudo useradd --system --uid "$SURVNG_UID" --gid survng \
       --home-dir /var/lib/survng --create-home \
       --shell /usr/sbin/nologin survng

sudo mkdir -p /opt/survng \
  /var/lib/survng/{config,data,models,model-cache} \
  "$SURVNG_MEDIA_DIR"

sudo chown -R survng:survng /var/lib/survng "$SURVNG_MEDIA_DIR"
sudo chmod 700 /var/lib/survng/config /var/lib/survng/data
```

Mount NFS on the host first if recordings live there. Do not put config or
SQLite data on NFS. `SURVNG_UID` must be able to write `$SURVNG_MEDIA_DIR`.

## 3. Write Compose files

CPU image (always start from this file):

```bash
cat > /tmp/survng.env <<EOF
TZ=${SURVNG_TZ}
SURVNG_UID=${SURVNG_UID}
SURVNG_GID=${SURVNG_GID}
SURVNG_VIDEO_GID=${SURVNG_VIDEO_GID:-44}
SURVNG_RENDER_GID=${SURVNG_RENDER_GID:-993}
EOF
sudo mv /tmp/survng.env /opt/survng/.env

cat > /tmp/survng.compose.yaml <<EOF
name: survng

services:
  survng:
    image: ${SURVNG_IMAGE}
    container_name: survng
    restart: unless-stopped
    network_mode: host
    stop_grace_period: 60s
    environment:
      TZ: \${TZ:-UTC}
      SURVNG_CONFIG_PATH: /config/config.json
      SURVNG_UID: \${SURVNG_UID:-1500}
      SURVNG_GID: \${SURVNG_GID:-1500}
      MALLOC_ARENA_MAX: \${MALLOC_ARENA_MAX:-16}
    volumes:
      - /var/lib/survng/config:/config
      - /var/lib/survng/data:/data
      - "${SURVNG_MEDIA_DIR}:/media"
      - /var/lib/survng/models:/models:ro
    security_opt:
      - no-new-privileges:true
EOF
sudo mv /tmp/survng.compose.yaml /opt/survng/compose.yaml
```

Intel GPU override (skip on CPU-only hosts):

```bash
cat > /tmp/survng.intel.yaml <<'EOF'
services:
  survng:
    image: ghcr.io/instigatorx/survng:v1.2-intel
    devices:
      - /dev/dri:/dev/dri
    environment:
      SURVNG_VIDEO_GID: ${SURVNG_VIDEO_GID:-44}
      SURVNG_RENDER_GID: ${SURVNG_RENDER_GID:-993}
EOF
sudo mv /tmp/survng.intel.yaml /opt/survng/compose.intel-gpu.yaml
```

Proxmox LXC / nested Docker: the daemon often cannot load `docker-default`.
Add this override (weaker isolation; keep `no-new-privileges`):

```bash
cat > /tmp/survng.lxc.yaml <<'EOF'
services:
  survng:
    security_opt:
      - no-new-privileges:true
      - apparmor=unconfined
EOF
sudo mv /tmp/survng.lxc.yaml /opt/survng/compose.lxc.yaml
```

Pick the Compose file list for later commands:

```bash
cd /opt/survng
COMPOSE=( -f compose.yaml )
[ -f compose.intel-gpu.yaml ] && COMPOSE+=( -f compose.intel-gpu.yaml )
[ -f compose.lxc.yaml ] && COMPOSE+=( -f compose.lxc.yaml )
docker compose "${COMPOSE[@]}" config
docker compose "${COMPOSE[@]}" config | grep -A2 security_opt
```

Optional extra media disks: write `compose.storage.yaml` and append
`-f compose.storage.yaml` to `COMPOSE`. Example:

```yaml
services:
  survng:
    volumes:
      - /srv/survng-a:/media-a
      - /srv/survng-b:/media-b
```

Then set those container paths under **Admin → General → Storage**. Enable
**Require mount** for NFS/SMB. Details: [docs/storage.md](docs/storage.md).

`.env` holds paths and numeric IDs only. Camera, MQTT, and AI secrets stay in
`/var/lib/survng/config/config.json`.

## 4. Optional: install models

The runtime image has no detector, ReID, face, or Smart Search weights. Skip
this to boot with detection off.

On LXC, keep `--security-opt apparmor=unconfined`. On a normal VM, you can omit
that flag.

```bash
docker pull ghcr.io/instigatorx/survng:v1.2-model-installer

docker run --rm --user "${SURVNG_UID}:${SURVNG_GID}" \
  --security-opt apparmor=unconfined \
  --security-opt no-new-privileges:true \
  -v /var/lib/survng/models:/models-out \
  -v /var/lib/survng/config:/config-out \
  -v /var/lib/survng/model-cache:/cache \
  -e SURVNG_INSTALLER_IN_CONTAINER=1 \
  ghcr.io/instigatorx/survng:v1.2-model-installer \
  --device GPU
```

Use `--device CPU` without an Intel iGPU. Add `--skip-semantic`, `--skip-face`,
or `--skip-depth` to omit those packages (YOLO26s and YOLO26n-depth are
AGPL-3.0; MobileCLIP2-B is Apple research/non-commercial). A successful depth
install writes and enables its model path unless `--no-enable` is supplied.
Licenses:
[docker/model-installer/THIRD_PARTY_MODELS.md](docker/model-installer/THIRD_PARTY_MODELS.md).

Admin model paths must be container paths such as
`/models/yolo26s_openvino_model/yolo26s.xml`, never host paths.

## 5. Start

```bash
cd /opt/survng
COMPOSE=( -f compose.yaml )
[ -f compose.intel-gpu.yaml ] && COMPOSE+=( -f compose.intel-gpu.yaml )
[ -f compose.lxc.yaml ] && COMPOSE+=( -f compose.lxc.yaml )
[ -f compose.storage.yaml ] && COMPOSE+=( -f compose.storage.yaml )

docker compose "${COMPOSE[@]}" pull
docker compose "${COMPOSE[@]}" up -d
docker compose "${COMPOSE[@]}" ps
docker compose "${COMPOSE[@]}" logs --tail=100 survng
curl -fsS http://127.0.0.1:8088/api/health
```

First start writes `/var/lib/survng/config/config.json` (mode `0600`) and seeds
`go2rtc.yaml` if missing. Detection stays off until models are installed and
enabled in Admin.

## 6. Open SurvNG

```text
http://NEW-SERVER-IP:8088/survng/
```

Restrict port 8088 to LAN/VPN or an authenticated reverse proxy.

Edit `/var/lib/survng/config/go2rtc.yaml`, then set each camera `stream_url` /
`live_stream_url` to `rtsp://127.0.0.1:8554/<stream_name>`. Set
`SURVNG_GO2RTC=0` in `.env` only when an external go2rtc already restreams.

API bearer tokens (optional) are under **Admin → General → API**. The health
endpoint stays unauthenticated. Do not turn tokens on until the browser and
WebSockets will send `Authorization: Bearer …`.

For a remote support case, use **Admin → Diagnostics → Download support
bundle**. It creates a redacted JSON report suitable for sharing; see the
[support-bundle guide](README.md#support-bundle).

## 7. Verify Intel acceleration (GPU image only)

```bash
docker exec survng python -c \
  'from openvino import Core; print(Core().available_devices)'
docker exec survng ffmpeg -hide_banner -hwaccels
docker exec survng ffmpeg -version | head -1
```

Expect `CPU` and `GPU` from OpenVINO, and `qsv` / `vaapi` from FFmpeg. Then
select `GPU` or `AUTO` in Admin. Passthrough does not force every workload onto
the GPU. The image ships FFmpeg **8.1.2**.

## 8. Functional checks

```bash
curl -fsS http://127.0.0.1:8088/api/health
```

In the UI: Telemetry, live view, recordings, an object-detection incident,
ONVIF, MQTT, and each media location online and writable.

## 9. Upgrade (no Git)

Pin a build with `ghcr.io/instigatorx/survng:sha-<7chars>` (add `-intel` for
GPU) if you do not want the moving `v1.2` tip.

```bash
cd /opt/survng
COMPOSE=( -f compose.yaml )
[ -f compose.intel-gpu.yaml ] && COMPOSE+=( -f compose.intel-gpu.yaml )
[ -f compose.lxc.yaml ] && COMPOSE+=( -f compose.lxc.yaml )
[ -f compose.storage.yaml ] && COMPOSE+=( -f compose.storage.yaml )

docker compose "${COMPOSE[@]}" pull
docker compose "${COMPOSE[@]}" up -d --remove-orphans
curl -fsS http://127.0.0.1:8088/api/health
```

Config, databases, models, and media stay on the host. Back them up before an
upgrade that changes schema or storage.

## 10. Backup

```bash
cd /opt/survng
COMPOSE=( -f compose.yaml )
[ -f compose.intel-gpu.yaml ] && COMPOSE+=( -f compose.intel-gpu.yaml )
[ -f compose.lxc.yaml ] && COMPOSE+=( -f compose.lxc.yaml )
[ -f compose.storage.yaml ] && COMPOSE+=( -f compose.storage.yaml )

docker compose "${COMPOSE[@]}" down
sudo tar -C /var/lib/survng -czf /root/survng-state-backup.tgz config data
docker compose "${COMPOSE[@]}" up -d
```

Do not add `-v` to `docker compose down`. Back up models separately if they are
hard to re-download. Recordings under `/media` follow retention; copy protected
incident media on their own schedule.

## Alternative: build from a Git checkout

Use this when you cannot pull GHCR, or you want a local `survng:local` image.

```bash
sudo mkdir -p /opt/survng
sudo chown "$(id -u):$(id -g)" /opt/survng
git clone https://github.com/InstigatorX/SurvNG.git /opt/survng
cd /opt/survng
cp .env.example .env
```

Set `SURVNG_UID` / `SURVNG_GID` / `SURVNG_MEDIA_DIR` in `.env` to the identity
from section 2. Then:

```bash
# Intel GPU + QSV
docker compose -f compose.yaml -f compose.intel-gpu.yaml build
docker compose -f compose.yaml -f compose.intel-gpu.yaml up -d

# CPU only
docker compose build
docker compose up -d

# Nested Docker in Proxmox LXC (BuildKit cannot load docker-default)
scripts/docker-build-lxc.sh
docker compose -f compose.yaml -f compose.intel-gpu.yaml -f compose.lxc.yaml \
  up -d --no-build
```

Host-side model installer from a checkout:

```bash
SURVNG_MODELS_DIR=/var/lib/survng/models \
SURVNG_CONFIG_DIR=/var/lib/survng/config \
scripts/install-docker-models.sh --device GPU --lxc
```

Upgrade a checkout with `scripts/update-from-git.sh` (rebuilds the running
Compose deployment). In-app **Admin → Update** does not rebuild Docker images
unless you mount a helper and set `SURVNG_UPDATE_HELPER`.
