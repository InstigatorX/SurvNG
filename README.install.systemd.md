# SurvNG systemd installation

Native virtualenv on Linux, run by a dedicated `survng` user and a systemd
unit. The host needs Python 3.12+, Node.js 20+ (frontend build), Git, FFmpeg,
and (for Intel GPU) a working `/dev/dri`.

Do **not** run this next to the Docker container. Do **not** run SurvNG as
root on a new install.

Commands are meant to be pasted in order. Set the variables in section 1, then
paste each later block as-is.

## 1. Set these values, then paste the rest

`1500` is an example; it must not already be a UID or GID on the host.

```bash
SURVNG_UID=1500
SURVNG_GID=1500
SURVNG_TZ=America/New_York
SURVNG_ROOT=/opt/survng
SURVNG_MEDIA_DIR=/srv/survng/media
SURVNG_GIT_URL=https://github.com/InstigatorX/SurvNG.git
SURVNG_GIT_BRANCH=v1.2

getent passwd "$SURVNG_UID" || true
getent group "$SURVNG_GID" || true
```

## 2. Install host packages (Ubuntu 24.04)

```bash
sudo apt-get update
sudo apt-get install -y \
  ca-certificates curl git python3 python3-venv python3-pip \
  ffmpeg pkg-config build-essential \
  libgl1 libglib2.0-0 libgomp1

curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
python3 --version
node --version
ffmpeg -version | head -1
```

Python must be 3.12+. Node must be 20+.

## 3. Create the SurvNG user and directories

Leave this user **out** of the `docker` group. Add `video` / `render` so an
Intel iGPU is usable.

```bash
getent group survng >/dev/null \
  || sudo groupadd --system --gid "$SURVNG_GID" survng
getent passwd survng >/dev/null \
  || sudo useradd --system --uid "$SURVNG_UID" --gid survng \
       --home-dir /var/lib/survng --create-home \
       --shell /usr/sbin/nologin survng

getent group video >/dev/null && sudo usermod -aG video survng || true
getent group render >/dev/null && sudo usermod -aG render survng || true

sudo mkdir -p "$SURVNG_ROOT" "$SURVNG_MEDIA_DIR" /var/lib/survng
sudo chown -R survng:survng "$SURVNG_ROOT" "$SURVNG_MEDIA_DIR" /var/lib/survng
sudo chmod 700 /var/lib/survng
```

Mount NFS on the host first if recordings live there. Keep SQLite/config on
local disk; only media should be on NFS.

## 4. Clone and build

```bash
sudo -u survng git clone --branch "$SURVNG_GIT_BRANCH" --single-branch \
  "$SURVNG_GIT_URL" "$SURVNG_ROOT"
cd "$SURVNG_ROOT"

sudo -u survng python3 -m venv "$SURVNG_ROOT/.venv"
sudo -u survng "$SURVNG_ROOT/.venv/bin/pip" install --upgrade pip
sudo -u survng "$SURVNG_ROOT/.venv/bin/pip" install -r "$SURVNG_ROOT/requirements.txt"

sudo -u survng bash -lc "cd '$SURVNG_ROOT/frontend' && npm ci --no-audit --no-fund && npm run build"
```

The production UI lands in `survng/static/`.

## 5. Private config and media path

```bash
cd "$SURVNG_ROOT"
if [ ! -f config.json ]; then
  sudo -u survng cp config.example.json config.json
fi
sudo chmod 600 config.json
sudo chown survng:survng config.json

sudo -u survng "$SURVNG_ROOT/.venv/bin/python" - <<PY
import json, os
from pathlib import Path
path = Path("$SURVNG_ROOT/config.json")
config = json.loads(path.read_text())
config["storage_dir"] = "$SURVNG_MEDIA_DIR"
locations = (config.get("media_storage") or {}).get("locations") or []
if locations:
    locations[0]["path"] = "$SURVNG_MEDIA_DIR"
config["ffmpeg_path"] = "/usr/bin/ffmpeg"
path.write_text(json.dumps(config, indent=2) + "\n")
os.chmod(path, 0o600)
print("Wrote", path)
PY
```

Camera, MQTT, and AI credentials belong in this file (or Admin), not in the
systemd unit.

## 6. Optional: go2rtc restreamer

SurvNG expects camera URLs to be go2rtc restreams
(`rtsp://127.0.0.1:8554/<name>`). Docker bundles go2rtc; systemd does not.

```bash
GO2RTC_VERSION=1.9.14
GO2RTC_SHA256=32d616af226bd731678ffde328b94cfb94e30339bfefc469cfb76323144615a6
curl -fsSL -o /tmp/go2rtc \
  "https://github.com/AlexxIT/go2rtc/releases/download/v${GO2RTC_VERSION}/go2rtc_linux_amd64"
printf '%s  %s\n' "$GO2RTC_SHA256" /tmp/go2rtc | sha256sum -c -
sudo install -m 755 /tmp/go2rtc /usr/local/bin/go2rtc
rm -f /tmp/go2rtc
go2rtc -version

sudo -u survng tee /var/lib/survng/go2rtc.yaml >/dev/null <<'EOF'
api:
  listen: "127.0.0.1:1984"
rtsp:
  listen: ":8554"
webrtc:
  listen: ":8555"
ffmpeg:
  bin: "/usr/bin/ffmpeg"
streams: {}
EOF
sudo chmod 600 /var/lib/survng/go2rtc.yaml

sudo tee /etc/systemd/system/go2rtc.service >/dev/null <<EOF
[Unit]
Description=go2rtc restreamer for SurvNG
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=survng
Group=survng
NoNewPrivileges=true
ExecStart=/usr/local/bin/go2rtc -config /var/lib/survng/go2rtc.yaml
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now go2rtc.service
sudo systemctl --no-pager --full status go2rtc.service
```

Edit `/var/lib/survng/go2rtc.yaml` to add streams, then point SurvNG cameras at
`rtsp://127.0.0.1:8554/<stream_name>`.

## 7. Optional: detection models

```bash
cd "$SURVNG_ROOT"
sudo -u survng mkdir -p "$SURVNG_ROOT/models"
sudo -u survng env \
  SURVNG_MODELS_DIR="$SURVNG_ROOT/models" \
  SURVNG_CONFIG_DIR="$SURVNG_ROOT" \
  SURVNG_HOST_CONFIG_PATH="$SURVNG_ROOT/config.json" \
  "$SURVNG_ROOT/scripts/install-docker-models.sh" --native --device GPU
```

Use `--device CPU` without an Intel iGPU. Add `--skip-semantic` or `--skip-face`
if you do not want those packages. YOLO26s is AGPL-3.0; MobileCLIP2-B is Apple
research/non-commercial.

The installer writes Docker paths (`/models/...`). Rewrite them for the host:

```bash
sudo -u survng "$SURVNG_ROOT/.venv/bin/python" - <<PY
import json, os
from pathlib import Path
root = Path("$SURVNG_ROOT")
models = str(root / "models")
path = root / "config.json"
text = path.read_text()
text = text.replace('"/models/', f'"{models}/')
text = text.replace('"/data/openvino-cache"', f'"{root / "runtime" / "openvino-cache"}"')
path.write_text(text)
os.chmod(path, 0o600)
(root / "runtime" / "openvino-cache").mkdir(parents=True, exist_ok=True)
print("Rewrote model paths in", path)
PY
```

## 8. Install and start survng.service

```bash
sudo tee /etc/systemd/system/survng.service >/dev/null <<EOF
[Unit]
Description=SurvNG network video recorder
Wants=network-online.target
After=network-online.target go2rtc.service
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User=survng
Group=survng
NoNewPrivileges=true
WorkingDirectory=${SURVNG_ROOT}
ExecStart=${SURVNG_ROOT}/.venv/bin/uvicorn survng.app.main:app --host 0.0.0.0 --port 8088 --loop asyncio --timeout-graceful-shutdown 30
ExecStop=-/bin/kill -USR1 \$MAINPID
ExecStop=/bin/sleep 1
ExecStop=/bin/kill -TERM \$MAINPID
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONFAULTHANDLER=1
Environment=TZ=${SURVNG_TZ}
Environment=SURVNG_REPO_ROOT=${SURVNG_ROOT}
Environment=SURVNG_CONFIG_PATH=${SURVNG_ROOT}/config.json
Environment=MALLOC_ARENA_MAX=16
KillSignal=SIGTERM
KillMode=mixed
TimeoutStopSec=60
SendSIGKILL=yes
Restart=on-failure
RestartSec=5
LimitNOFILE=65536
LimitCORE=infinity

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now survng.service
sudo systemctl --no-pager --full status survng.service
curl -fsS http://127.0.0.1:8088/api/health
```

`ExecStop` sends `SIGUSR1` first so ONVIF PullPoint slots are released before
Uvicorn drains.

The checked-in `deploy/survng.service` is the historical root/`/root/SurvNG`
unit. New installs should use the unit written above.

## 9. Open SurvNG

```text
http://NEW-SERVER-IP:8088/survng/
```

Restrict port 8088 to LAN/VPN or an authenticated reverse proxy.

Configure cameras under **Admin**. Optional API tokens: **Admin → General →
API**. The health endpoint stays unauthenticated.

For a remote support case, use **Admin → Diagnostics → Download support
bundle**. It creates a redacted JSON report suitable for sharing; see the
[support-bundle guide](README.md#support-bundle).

## 10. Verify Intel GPU (optional)

Host OpenVINO GPU needs a current Intel userspace stack. Ubuntu 24.04's default
packages may be older than the OpenVINO wheel. Intel's procedure:
[Ubuntu graphics packages](https://dgpu-docs.intel.com/installation-guides/installing-packages-from-the-intel-ppa.html).
Do **not** install `intel-i915-dkms` on a Proxmox kernel.

```bash
ls -l /dev/dri
sudo -u survng "$SURVNG_ROOT/.venv/bin/python" -c \
  'from openvino import Core; print(Core().available_devices)'
ffmpeg -hide_banner -hwaccels
```

Expect `CPU` and `GPU`. Then select `GPU` or `AUTO` in Admin.

Coordinated PPA upgrade (stops SurvNG; reboot after):

```bash
sudo systemctl stop survng.service
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt-get update
sudo apt-get --simulate install \
  intel-opencl-icd libze-intel-gpu1 libze1 clinfo \
  intel-media-va-driver-non-free libmfx-gen1.2 libvpl2 libvpl-tools vainfo
sudo apt-get install -y \
  intel-opencl-icd libze-intel-gpu1 libze1 clinfo \
  intel-media-va-driver-non-free libmfx-gen1.2 libvpl2 libvpl-tools vainfo
sudo -u survng mv "$SURVNG_ROOT/.cache/openvino" \
  "$SURVNG_ROOT/.cache/openvino-before-intel-$(date +%Y%m%d-%H%M%S)" 2>/dev/null || true
sudo -u survng mkdir -p "$SURVNG_ROOT/.cache/openvino"
sudo reboot
```

After reboot:

```bash
clinfo -l
vainfo --display drm --device /dev/dri/renderD128
sudo systemctl start survng.service
curl -fsS http://127.0.0.1:8088/api/health
```

## 11. Functional checks

```bash
curl -fsS http://127.0.0.1:8088/api/health
journalctl -u survng.service -n 100 --no-pager
```

In the UI: Telemetry, live view, recordings, an object-detection incident,
ONVIF, MQTT, and each media location online and writable.

## 12. Upgrade

In-app **Admin → General → Check for Updates / Update** fast-forwards Git
(optionally after selecting a remote branch), reinstalls Python deps, rebuilds the
frontend when `npm` is on `PATH`, and restarts `survng.service`. The checkout must
be clean and fast-forwardable. `SURVNG_REPO_ROOT` is already set in the unit.

`scripts/update-from-git.sh` runs `systemctl restart` as the current user. For
this dedicated account, use the manual block (git/pip/npm as `survng`, restart
as root):

```bash
cd "$SURVNG_ROOT"
sudo -u survng git status --short
sudo -u survng git fetch --prune origin
sudo -u survng git pull --ff-only origin "$SURVNG_GIT_BRANCH"
sudo -u survng "$SURVNG_ROOT/.venv/bin/pip" install -r requirements.txt
sudo -u survng bash -lc "cd '$SURVNG_ROOT/frontend' && npm ci --no-audit --no-fund && npm run build"
sudo systemctl restart survng.service
curl -fsS http://127.0.0.1:8088/api/health
```

## 13. Backup

```bash
sudo systemctl stop survng.service
sudo tar -C "$SURVNG_ROOT" -czf /root/survng-native-backup.tgz config.json models
sudo tar -C /var/lib/survng -czf /root/survng-go2rtc-backup.tgz go2rtc.yaml 2>/dev/null || true
sudo systemctl start survng.service
```

Include `$SURVNG_MEDIA_DIR` (or protected incident media) on a separate
schedule. Native databases default under the checkout (`runtime/` or paths in
`config.json`); add those directories to the archive if you use them.

## 14. Rollback to the previous Git commit

```bash
sudo systemctl stop survng.service
cd "$SURVNG_ROOT"
sudo -u survng git log --oneline -5
sudo -u survng git checkout '<previous-sha>'
sudo -u survng "$SURVNG_ROOT/.venv/bin/pip" install -r requirements.txt
sudo -u survng bash -lc "cd '$SURVNG_ROOT/frontend' && npm ci --no-audit --no-fund && npm run build"
sudo systemctl start survng.service
```

Newer config or database files are not guaranteed to work with an older
checkout.
