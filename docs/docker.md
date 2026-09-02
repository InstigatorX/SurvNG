# Docker installation

Pasteable first-install commands (published image, dedicated user, no Git
checkout) are in [README.install.docker.md](../README.install.docker.md). Native
systemd install: [README.install.systemd.md](../README.install.systemd.md).

Docker is an optional SurvNG deployment. It does not replace the native
virtualenv and systemd installation.

## Deployment contract

The image is immutable. All installation-specific state lives outside it:

| Container path | Contents | Storage recommendation |
| --- | --- | --- |
| `/config` | Writable `config.json`, including camera and service credentials | Local filesystem, private and backed up |
| `/data` | SQLite databases, recording index, image cache, ONVIF cache, and OpenVINO cache | Local SSD/filesystem |
| `/media` | Recordings, snapshots, event clips, HLS, and playback cache | Existing host-mounted NFS or local recording disk |
| `/models` | Detection, face, and ReID models | Read-only local bind mount |

Do not place `/config` or `/data` on NFS. SQLite and frequently updated caches
belong on local storage. Mount NFS on the Docker host first, then bind the
mounted directory to `/media`; SurvNG does not need NFS credentials inside the
container.

Linux host networking is the default because it gives cameras, ONVIF event
subscriptions, go2rtc, MQTT, and WireGuard/LAN routes the same reachability as
the native service. Port 8088 is therefore bound directly on the host. On
Docker Desktop, remove `network_mode: host` from `compose.yaml` and add:

```yaml
ports:
  - "8088:8088"
```

Camera discovery and callback behavior can differ behind Docker Desktop's VM,
so Linux host networking is recommended for an NVR.

## Confidential configuration

The repository, Docker build context, and image exclude the live config,
databases, media, certificates, environment files, and model binaries. The
checked-in Docker config contains no cameras or credentials. On first start the
container creates `/config/config.json` with mode `0600`.

SurvNG must be able to read the plaintext camera, MQTT, and AI credentials in
order to connect to those services. Protect them by limiting host access to the
config directory, using an encrypted host disk or secret-backed filesystem when
required, and backing the file up only to an encrypted destination. Do not put
credentials in `.env`: Docker environment variables are easier to expose
through process and container inspection. The Admin API masks stored secrets,
and SurvNG redacts recognized secret values from its in-memory logs.

External FFmpeg processes necessarily receive their input URL as a command-line
argument. Keep camera credentials out of those arguments by defining the
credentialed upstream in `go2rtc.yaml` and configuring SurvNG with the
credential-free local restream URL
(`rtsp://127.0.0.1:8554/<stream_name>`). SurvNG logs a bounded warning when a
credentialed URL is passed directly to capture, but cannot hide it from a
privileged host process inspector.

Treat SurvNG as a private app even in Docker. Prefer signing in (Admin →
Access), binding `8088` to loopback, and putting TLS on nginx. See
[Internet, TLS, and reverse proxies](guide/reverse-proxy.md). Do not publish
the raw SurvNG port to the public internet.

To inspect the build context before publishing an image:

```bash
docker build --check .
git status --ignored --short docker-data config.json .env
```

Never push a locally exported image if credentials or runtime files were added
with an ad-hoc Dockerfile or `docker commit`.

## GitHub Container Registry

Pushing to a release branch (`v1.0`, `v1.1`, or `v1.2`) or a version tag such as
`v1.2.0` runs `.github/workflows/docker-publish.yml` on the self-hosted runner
and publishes all Dockerfile targets to GHCR. Pushing to `gstreamer` publishes
only the Intel runtime.

| Image | When | Target |
| --- | --- | --- |
| `ghcr.io/instigatorx/survng:v1.2` | Every push to `v1.2` | `runtime` (CPU tip) |
| `ghcr.io/instigatorx/survng:v1.2-intel` | Every push to `v1.2` | `runtime-intel` tip |
| `ghcr.io/instigatorx/survng:v1.2-model-installer` | Every push to `v1.2` | `model-installer` tip |
| `ghcr.io/instigatorx/survng:gstreamer-intel` | Every push to `gstreamer` | `runtime-intel` tip |
| `ghcr.io/instigatorx/survng:sha-<7chars>` | Every push to a release branch | Immutable CPU commit |
| `ghcr.io/instigatorx/survng:sha-<7chars>-intel` | Every push to a release branch | Immutable Intel commit |
| `ghcr.io/instigatorx/survng:sha-<7chars>-model-installer` | Every push to a release branch | Immutable model-installer commit |
| `ghcr.io/instigatorx/survng:v1.2.0` | Git tag `v1.2.0` | `runtime` release |
| `ghcr.io/instigatorx/survng:v1.2.0-intel` | Git tag `v1.2.0` | `runtime-intel` release |
| `ghcr.io/instigatorx/survng:v1.2.0-model-installer` | Git tag `v1.2.0` | `model-installer` release |

The same pattern applies to older release branches.

Day-to-day deploys can track the branch tip without cutting a release tag:

```bash
docker pull ghcr.io/instigatorx/survng:v1.2
```

Pin a specific build with the `sha-…` tag. Use a `v*` tag when you want a
stable release number. Pull requires a GitHub account with read access to the
package (or a public package).

## Self-hosted runner disk maintenance

Docker image builds run on the repository's self-hosted GitHub Actions runner.
Each publish builds three large targets sequentially, but CI and prior builds
still accumulate Docker layer cache, old images, and tool caches under the runner
account.

SurvNG automates cleanup in four layers:

1. **Per publish job** — `scripts/github-runner-cleanup.sh --publish` reclaims
   dangling images and containers only. It does not wipe the Docker build
   cache. Sequential `v1.2` matrix targets keep the moving tip locally so
   `runtime-intel` can reuse `runtime-base`. Do not `--cache-from` a pulled
   GHCR image on the legacy builder; that restore fails on this multi-stage
   Dockerfile.
2. **Per CI test job** — `--light` cleanup before/after focused tests.
3. **Nightly** — `.github/workflows/runner-maintenance.yml` runs at 05:00 UTC
   with `--standard` cleanup (build cache, images older than 24h, npm/pip cache).
4. **Auto-escalation** — when root filesystem free space drops below 15%,
   `--light` and `--standard` escalate to a stronger mode. `--publish` never
   escalates to a cache wipe.

Manual cleanup on the runner host (or via **Actions → Runner maintenance → Run
workflow**):

```bash
scripts/github-runner-cleanup.sh --standard   # normal maintenance
scripts/github-runner-cleanup.sh --aggressive # when disk is tight
```

Optional host-level safety net (outside GitHub Actions), e.g. daily cron as the
runner user:

```cron
0 4 * * * /path/to/SurvNG/scripts/github-runner-cleanup.sh --standard >>/var/log/survng-runner-cleanup.log 2>&1
```

Do not prune Docker volumes from CI unless the runner host has no other services;
SurvNG production containers on the same machine would be affected.

Point Compose at the published image instead of building locally by setting
`image:` under `survng` (and omit `build:` when you want a registry-only pull).
For Intel hosts, use the `-intel` tag with `compose.intel-gpu.yaml` device mounts.

## New installation

1. Copy the non-secret environment template and set host paths and the UID/GID
   that should own local data. The UID/GID should also have write permission on
   the host's media mount.

   ```bash
   cp .env.example .env
   id -u
   id -g
   getent group video render
   ```

2. Edit `.env`. In particular, set `SURVNG_MEDIA_DIR` to the host's existing
   recording mount; the checked-in `/srv/survng/media` value is only a generic
   example. `.env` is ignored by Git and the Docker build, but it should contain
   paths and numeric IDs only—not passwords.

3. Download detector, depth, ReID, and optional Smart Search models into the host
   models directory and patch `config.json`. These weights are not in the
   GHCR runtime image (YOLO26s is AGPL-3.0; MobileCLIP2-B is Apple research/non-commercial).
   No SurvNG Git checkout is required. The script uses the
   `survng-model-installer` container by default (see
   `docker/model-installer/THIRD_PARTY_MODELS.md` for attributions):

   ```bash
   SURVNG_MODELS_DIR=/docker-data/models \
   SURVNG_CONFIG_DIR=/docker-data/config \
   scripts/install-docker-models.sh --device GPU
   ```

   Pass `--native` on a dev checkout to run without Docker. On Proxmox/LXC
   nested Docker hosts, pass `--lxc` to run the installer with
   `apparmor=unconfined` (opt-in; same trade-off as `compose.lxc.yaml`).
   Drop `--device GPU` on CPU-only hosts. Add `--skip-semantic` to skip the
   MobileCLIP2-B export, `--skip-face` to skip ArcFace / landmarks / face
   detector, or `--skip-depth` to skip monocular depth. Successful depth
   installation writes and enables the model path unless `--no-enable` is used.
   The script preserves any cameras already in
   `docker-data/config/config.json` and still patches paths for models that
   installed successfully even when another step fails.

4. Build and start the CPU-compatible image:

   ```bash
   docker compose build
   docker compose up -d
   docker compose ps
   docker compose logs --tail=100 survng
   ```

5. Open `http://SERVER:8088/survng/`. If you skipped the model installer, the
   generated configuration starts with no cameras and object detection
   disabled. Configure cameras and optional services through Admin.

6. Edit `/config/go2rtc.yaml` (bind-mounted under `SURVNG_CONFIG_DIR`) to define
   go2rtc streams, then point each SurvNG camera `stream_url` /
   `live_stream_url` at `rtsp://127.0.0.1:8554/<stream_name>`. The container
   starts bundled go2rtc automatically; set `SURVNG_GO2RTC=0` only when an
   external go2rtc already provides restreams.

The health check calls `/api/health`. It deliberately does not inspect cameras,
FFmpeg, NFS, databases, or models, so a slow or unavailable external dependency
cannot make Docker repeatedly restart an otherwise functioning service. Use
Admin > Telemetry for subsystem health.

## Bundled go2rtc

The image includes go2rtc **v1.9.14**. On first start the entrypoint seeds
`/config/go2rtc.yaml` from `docker/go2rtc.example.yaml` and launches go2rtc
before SurvNG. With Compose `network_mode: host`, SurvNG reaches the API at
`127.0.0.1:1984` and restreams at `rtsp://127.0.0.1:8554/...`. Stream ownership
stays in go2rtc; SurvNG does not invent transcoding aliases.

## Intel OpenVINO GPU and QSV/VA-API

The base image supports CPU inference and FFmpeg 8.1.2. The Intel override
selects an image target with the Intel OpenCL and media userspace packages,
passes `/dev/dri`, and adds the configured video/render groups to the runtime
user:

```bash
docker compose -f compose.yaml -f compose.intel-gpu.yaml up -d --build
docker compose exec survng ffmpeg -version | head -1
docker compose exec survng python -c \
  'from openvino import Core; print(Core().available_devices)'
docker compose exec survng ffmpeg -hide_banner -hwaccels
```

Set `SURVNG_VIDEO_GID` and `SURVNG_RENDER_GID` in `.env` from
`getent group video render` on the host. Then select `GPU` or `AUTO` for the
detector and `auto`, `qsv`, or the appropriate hardware setting in Admin. Device
pass-through does not force hardware acceleration; it only makes it available.
If the host Intel compute stack is newer than Debian's container packages, pin
or extend the `runtime-intel` target rather than installing GPU packages on the
host into the container.

## Migrating an existing native installation

Never run the systemd and Docker instances together. They would open duplicate
camera streams, recorder processes, MQTT clients, and ONVIF subscriptions—an
especially important constraint for cameras with small ONVIF connection limits.

1. Record the configured paths, then stop the native service cleanly:

   ```bash
   systemctl stop survng.service
   systemctl is-active survng.service
   ```

2. Create the persistent directories and copy local state. Preserve the source
   until the container has been verified.

   ```bash
   install -d -m 700 docker-data/config docker-data/data docker-data/models
   install -m 600 config.json docker-data/config/config.json
   cp -a runtime/database docker-data/data/database
   cp -a runtime/recording-index docker-data/data/recording-index
   chown -R "$(id -u):$(id -g)" docker-data/config docker-data/data
   ```

3. Edit the copied config, not the native one. Change container-visible paths:

   - `storage_dir`: `/media`
   - `database_dir`: `/data/database`
   - `recording_index_dir`: `/data/recording-index`
   - `ffmpeg_path`: `/usr/bin/ffmpeg`
   - `detector.cache_dir`: `/data/openvino-cache`
   - model and label paths: locations below `/models`

   Copy each configured model directory into `docker-data/models` and keep each
   OpenVINO `.xml`, matching `.bin`, labels, and ReID model together. Do not copy
   old OpenVINO compiled cache blobs between materially different OpenVINO or
   GPU runtimes; the container will rebuild them under `/data/openvino-cache`.

4. Confirm `.env` maps the same host recording mount to `/media`, then start and
   verify the container:

   ```bash
   docker compose config
   docker compose up -d --build
   docker compose ps
   curl -fsS http://127.0.0.1:8088/api/health
   docker compose logs --tail=200 survng
   ```

5. Check Admin > Telemetry, live view, a recording, object detection, ONVIF, and
   MQTT before disabling the native unit permanently.

Rollback is intentionally simple:

```bash
docker compose down
systemctl start survng.service
```

The original native config and runtime directories remain untouched.

## Upgrades and backups

Back up `/config` and `/data` while SurvNG is stopped or with a SQLite-aware
backup process. Recordings under `/media` follow the configured retention policy
and normally need no second Docker-specific copy.

```bash
docker compose down
tar -C docker-data -czf survng-state-backup.tgz config data
docker compose build --pull
docker compose up -d
```

`docker compose down` does not delete bind-mounted state. Do not add `-v` unless
you have separately verified every mounted path and intend to remove managed
volumes. Image rollback is done by tagging a known-good image and setting the
`image:` value; the config and database remain independent of the image.

Docker sends `SIGTERM` with a 60-second grace period. The entrypoint first sends
SurvNG `SIGUSR1` to release ONVIF PullPoint subscriptions, waits one second, and
then begins the normal application shutdown. Avoid `docker kill` except for a
truly wedged process because it bypasses that cleanup.

## Nested Docker and AppArmor

If Docker itself runs inside an LXC container, validate the host before blaming
the SurvNG image:

```bash
docker run --rm hello-world
```

An error saying Docker cannot load or apply the `docker-default` AppArmor
profile means the outer LXC host has not granted the nesting/AppArmor support
the inner Docker daemon expects. Docker containers can still run successfully
with an explicit AppArmor override while Docker's embedded BuildKit fails before
the first Dockerfile instruction. The Compose service's `security_opt` applies
at runtime; it does not configure BuildKit's worker containers.

Proxmox officially recommends a QEMU VM, rather than LXC, for application
containers such as Docker. At minimum, an LXC intended to host Docker needs the
`nesting=1,keyctl=1` features. Check this on the Proxmox host, replacing `CTID`:

```bash
pct config CTID
pct set CTID -features nesting=1,keyctl=1
```

Those features do not give an unprivileged LXC permission to manage the host's
AppArmor profiles. A VM remains the strongest isolation boundary, but SurvNG
also provides a narrowly scoped local-build path for an LXC where unconfined
Docker workloads are already an accepted policy.

### Build locally in the LXC

The helper creates a dedicated privileged, AppArmor-unconfined BuildKit worker
named `survng-buildkit`, connects to its Unix socket through Docker's
`docker-container://` transport, and loads the completed image into the local
Docker image store. The worker uses the `survng-buildkit-state` volume and an
`unless-stopped` restart policy so later builds retain their cache:

```bash
# Intel OpenVINO GPU/QSV image (default)
scripts/docker-build-lxc.sh

# CPU image
scripts/docker-build-lxc.sh runtime
```

Only run this builder against trusted source: build instructions execute with
elevated access and anyone able to submit builds to it effectively has root
authority within the LXC. The resulting SurvNG application container is not
privileged. Start it with the explicit LXC runtime override:

```bash
docker compose \
  -f compose.yaml \
  -f compose.intel-gpu.yaml \
  -f compose.lxc.yaml \
  up -d --no-build
```

Do not use plain `docker compose ... build` on this host; that selects the
embedded BuildKit worker and reproduces the AppArmor failure.

### Build elsewhere and run in LXC

Building does not require the GPU. A practical migration path is to build on
any trusted x86-64 Linux Docker host or VM, then import the image into this LXC:

```bash
# On the build host, from the SurvNG repository:
docker compose -f compose.yaml -f compose.intel-gpu.yaml build
docker save survng:local | gzip > survng-local.tar.gz

# Copy survng-local.tar.gz to the SurvNG LXC, then:
gzip -dc survng-local.tar.gz | docker load
```

The imported image can run with the Intel device override and an explicit LXC
AppArmor compatibility override:

```bash
docker compose \
  -f compose.yaml \
  -f compose.intel-gpu.yaml \
  -f compose.lxc.yaml \
  up -d --no-build
```

`compose.lxc.yaml` disables Docker's inner AppArmor profile for only the SurvNG
container. The outer Proxmox LXC boundary and `no-new-privileges` remain, but
this is still weaker isolation than running Docker in a VM. Never start it
while the native `survng.service` is running.
