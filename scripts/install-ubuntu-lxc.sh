#!/usr/bin/env bash
# Install SurvNG as a systemd service on a fresh Ubuntu LXC.
#
# Prerequisite: a root shell on Ubuntu with systemd and a network connection.
# The survng user, packages, checkout, config, go2rtc, and unit are created
# here. Nothing else is expected to exist.
#
#   curl -fsSL https://raw.githubusercontent.com/InstigatorX/SurvNG/v1.2/scripts/install-ubuntu-lxc.sh | bash
#   bash scripts/install-ubuntu-lxc.sh --gpu --media-dir /srv/survng/media
set -euo pipefail

SURVNG_UID="${SURVNG_UID:-1500}"
SURVNG_GID="${SURVNG_GID:-1500}"
SURVNG_TZ="${SURVNG_TZ:-America/New_York}"
SURVNG_ROOT="${SURVNG_ROOT:-/opt/survng}"
SURVNG_MEDIA_DIR="${SURVNG_MEDIA_DIR:-/srv/survng/media}"
SURVNG_GIT_URL="${SURVNG_GIT_URL:-https://github.com/InstigatorX/SurvNG.git}"
SURVNG_GIT_BRANCH="${SURVNG_GIT_BRANCH:-v1.2.0}"
DEVICE="CPU"
SKIP_MODELS=0
SKIP_GO2RTC=0
SKIP_START=0

GO2RTC_VERSION="1.9.14"
GO2RTC_SHA256="32d616af226bd731678ffde328b94cfb94e30339bfefc469cfb76323144615a6"

usage() {
  cat <<'EOF'
Usage: install-ubuntu-lxc.sh [options]

Install SurvNG from Git onto a fresh Ubuntu LXC and start survng.service.
Run as root. The script creates the survng user. The service does not run as root.

Options:
  --uid N            survng uid (default: 1500, or $SURVNG_UID)
  --gid N            survng gid (default: 1500, or $SURVNG_GID)
  --tz ZONE          Service timezone (default: America/New_York)
  --root PATH        Checkout directory (default: /opt/survng)
  --media-dir PATH   Recording and snapshot storage (default: /srv/survng/media)
  --url URL          Git remote (default: https://github.com/InstigatorX/SurvNG.git)
  --branch REF       Branch or tag to check out (default: v1.2.0)
  --gpu              Install models for an Intel GPU and require /dev/dri
  --skip-models      Boot with detection off
  --skip-go2rtc      Do not install the local go2rtc restreamer
  --no-start         Write the unit but do not start it
  -h, --help         Show this help
EOF
}

log() { printf '==> %s\n' "$*"; }
die() { printf 'install-ubuntu-lxc: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --uid) SURVNG_UID="$2"; shift 2 ;;
    --gid) SURVNG_GID="$2"; shift 2 ;;
    --tz) SURVNG_TZ="$2"; shift 2 ;;
    --root) SURVNG_ROOT="$2"; shift 2 ;;
    --media-dir) SURVNG_MEDIA_DIR="$2"; shift 2 ;;
    --url) SURVNG_GIT_URL="$2"; shift 2 ;;
    --branch) SURVNG_GIT_BRANCH="$2"; shift 2 ;;
    --gpu) DEVICE="GPU"; shift ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    --skip-go2rtc) SKIP_GO2RTC=1; shift ;;
    --no-start) SKIP_START=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ "$(id -u)" -eq 0 ]] || die "run this script as root"
[[ "$SURVNG_UID" != "0" && "$SURVNG_GID" != "0" ]] || die "refusing to install SurvNG as uid 0 or gid 0"
[[ "$SURVNG_UID" =~ ^[0-9]+$ && "$SURVNG_GID" =~ ^[0-9]+$ ]] || die "uid and gid must be numeric"
[[ -d /run/systemd/system ]] || die "systemd is not running. Use an Ubuntu LXC template with systemd."

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
fi
[[ "${ID:-}" == "ubuntu" ]] || die "this installer targets Ubuntu (found ${ID:-unknown})"

if command -v docker >/dev/null 2>&1; then
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'survng'; then
    die "Docker container survng is running on port 8088. Stop it first: docker stop survng"
  fi
fi

as_survng() { runuser -u survng -- "$@"; }

log "Installing host packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  ca-certificates curl gnupg sudo git python3 python3-venv python3-pip \
  ffmpeg pkg-config build-essential \
  libgl1 libglib2.0-0 libgomp1

python3 - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit(f"Python 3.12 or newer is required, found {sys.version.split()[0]}")
PY

node_major=0
if command -v node >/dev/null 2>&1; then
  node_major="$(node -p 'process.versions.node.split(".")[0]')"
fi
if [[ "$node_major" -lt 20 ]]; then
  log "Installing Node.js 20"
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi
node_major="$(node -p 'process.versions.node.split(".")[0]')"
[[ "$node_major" -ge 20 ]] || die "Node.js 20 or newer is required"

if [[ "$DEVICE" == "GPU" && ! -d /dev/dri ]]; then
  die "--gpu needs /dev/dri inside this LXC. Pass the host GPU through, or omit --gpu."
fi

if getent passwd survng >/dev/null; then
  log "Using the existing survng account"
else
  log "Creating the survng user (it is not expected to exist on a fresh LXC)"
fi
if getent group survng >/dev/null; then
  existing_gid="$(getent group survng | cut -d: -f3)"
  [[ "$existing_gid" == "$SURVNG_GID" ]] || die "group survng exists with gid $existing_gid, not $SURVNG_GID"
elif getent group "$SURVNG_GID" >/dev/null; then
  die "gid $SURVNG_GID is already in use. Pass a free --gid."
else
  groupadd --system --gid "$SURVNG_GID" survng
fi

if getent passwd survng >/dev/null; then
  existing_uid="$(id -u survng)"
  [[ "$existing_uid" == "$SURVNG_UID" ]] || die "user survng exists with uid $existing_uid, not $SURVNG_UID"
elif getent passwd "$SURVNG_UID" >/dev/null; then
  die "uid $SURVNG_UID is already in use. Pass a free --uid."
else
  useradd --system --uid "$SURVNG_UID" --gid survng \
    --home-dir /var/lib/survng --create-home \
    --shell /usr/sbin/nologin survng
fi

id survng >/dev/null || die "the survng user was not created"
getent group video >/dev/null && usermod -aG video survng || true
getent group render >/dev/null && usermod -aG render survng || true
log "survng is $(id survng)"

mkdir -p "$SURVNG_ROOT" "$SURVNG_MEDIA_DIR" /var/lib/survng
chown survng:survng "$SURVNG_ROOT" "$SURVNG_MEDIA_DIR" /var/lib/survng
chmod 700 /var/lib/survng

log "Checking out $SURVNG_GIT_BRANCH"
if [[ ! -d "$SURVNG_ROOT/.git" ]]; then
  as_survng git clone --branch "$SURVNG_GIT_BRANCH" --single-branch \
    "$SURVNG_GIT_URL" "$SURVNG_ROOT"
else
  if [[ -n "$(as_survng git -C "$SURVNG_ROOT" status --porcelain --untracked-files=no)" ]]; then
    die "tracked changes in $SURVNG_ROOT. Commit or stash them before re-running."
  fi
  as_survng git -C "$SURVNG_ROOT" fetch --prune --tags origin
  as_survng git -C "$SURVNG_ROOT" checkout "$SURVNG_GIT_BRANCH"
  if as_survng git -C "$SURVNG_ROOT" show-ref --verify --quiet "refs/heads/$SURVNG_GIT_BRANCH"; then
    as_survng git -C "$SURVNG_ROOT" pull --ff-only origin "$SURVNG_GIT_BRANCH"
  fi
fi
chown -R survng:survng "$SURVNG_ROOT"

log "Building the virtualenv and frontend"
if [[ ! -x "$SURVNG_ROOT/.venv/bin/python" ]]; then
  as_survng python3 -m venv "$SURVNG_ROOT/.venv"
fi
as_survng "$SURVNG_ROOT/.venv/bin/pip" install --upgrade pip
as_survng "$SURVNG_ROOT/.venv/bin/pip" install -r "$SURVNG_ROOT/requirements.txt"
as_survng bash -lc "cd '$SURVNG_ROOT/frontend' && npm ci --no-audit --no-fund && npm run build"

config_path="$SURVNG_ROOT/config.json"
if [[ ! -f "$config_path" ]]; then
  log "Writing $config_path"
  as_survng cp "$SURVNG_ROOT/config.example.json" "$config_path"
  as_survng "$SURVNG_ROOT/.venv/bin/python" - "$config_path" "$SURVNG_MEDIA_DIR" <<'PY'
import json, os, sys
from pathlib import Path
path, media = Path(sys.argv[1]), sys.argv[2]
config = json.loads(path.read_text())
config["storage_dir"] = media
locations = (config.get("media_storage") or {}).get("locations") or []
if locations:
    locations[0]["path"] = media
config["ffmpeg_path"] = "/usr/bin/ffmpeg"
path.write_text(json.dumps(config, indent=2) + "\n")
os.chmod(path, 0o600)
print("Wrote", path)
PY
fi
chown survng:survng "$config_path"
chmod 600 "$config_path"

if [[ "$SKIP_GO2RTC" -eq 0 ]]; then
  log "Installing go2rtc $GO2RTC_VERSION"
  tmp_go2rtc="$(mktemp)"
  curl -fsSL -o "$tmp_go2rtc" \
    "https://github.com/AlexxIT/go2rtc/releases/download/v${GO2RTC_VERSION}/go2rtc_linux_amd64"
  printf '%s  %s\n' "$GO2RTC_SHA256" "$tmp_go2rtc" | sha256sum -c -
  install -m 755 "$tmp_go2rtc" /usr/local/bin/go2rtc
  rm -f "$tmp_go2rtc"

  go2rtc_config="/var/lib/survng/go2rtc.yaml"
  if [[ ! -f "$go2rtc_config" ]]; then
    as_survng tee "$go2rtc_config" >/dev/null <<'EOF'
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
  fi
  chown survng:survng "$go2rtc_config"
  chmod 600 "$go2rtc_config"

  tee /etc/systemd/system/go2rtc.service >/dev/null <<EOF
[Unit]
Description=go2rtc restreamer for SurvNG
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=survng
Group=survng
NoNewPrivileges=true
ExecStart=/usr/local/bin/go2rtc -config ${go2rtc_config}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable --now go2rtc.service
fi

if [[ "$SKIP_MODELS" -eq 0 ]]; then
  log "Installing detection models for $DEVICE"
  as_survng mkdir -p "$SURVNG_ROOT/models"
  as_survng env \
    SURVNG_MODELS_DIR="$SURVNG_ROOT/models" \
    SURVNG_CONFIG_DIR="$SURVNG_ROOT" \
    SURVNG_HOST_CONFIG_PATH="$config_path" \
    "$SURVNG_ROOT/scripts/install-docker-models.sh" \
      --native --device "$DEVICE" --python "$SURVNG_ROOT/.venv/bin/python"
  as_survng "$SURVNG_ROOT/.venv/bin/python" - "$SURVNG_ROOT" <<'PY'
import os, sys
from pathlib import Path
root = Path(sys.argv[1])
models = str(root / "models")
path = root / "config.json"
text = path.read_text().replace('"/models/', f'"{models}/')
text = text.replace('"/data/openvino-cache"', f'"{root / "runtime" / "openvino-cache"}"')
path.write_text(text)
os.chmod(path, 0o600)
(root / "runtime" / "openvino-cache").mkdir(parents=True, exist_ok=True)
print("Rewrote model paths in", path)
PY
fi

after_go2rtc=""
if [[ "$SKIP_GO2RTC" -eq 0 ]]; then
  after_go2rtc=" go2rtc.service"
fi

log "Installing survng.service"
tee /etc/systemd/system/survng.service >/dev/null <<EOF
[Unit]
Description=SurvNG network video recorder
Wants=network-online.target
After=network-online.target${after_go2rtc}
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User=survng
Group=survng
NoNewPrivileges=true
RuntimeDirectory=survng
RuntimeDirectoryMode=0700
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
Environment=SURVNG_OBSERVABILITY_SOCKET=/run/survng/observability.sock
Environment=MALLOC_ARENA_MAX=16
Environment=OPENBLAS_NUM_THREADS=1
Environment=OPENCV_OPENCL_RUNTIME=disabled
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

systemctl daemon-reload
systemctl enable survng.service

if [[ "$SKIP_START" -eq 0 ]]; then
  log "Starting SurvNG"
  systemctl restart survng.service
  ok=0
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
    if curl -fsS http://127.0.0.1:8088/api/health >/dev/null; then
      ok=1
      break
    fi
    sleep 2
  done
  if [[ "$ok" -ne 1 ]]; then
    systemctl --no-pager --full status survng.service >&2 || true
    journalctl -u survng.service -n 80 --no-pager >&2 || true
    die "SurvNG did not answer http://127.0.0.1:8088/api/health"
  fi
  curl -fsS http://127.0.0.1:8088/api/health
  printf '\n'
fi

log "SurvNG is installed at $SURVNG_ROOT"
printf 'Open http://<this-lxc-ip>:8088/survng/\n'
printf 'Add camera streams in /var/lib/survng/go2rtc.yaml, then use rtsp://127.0.0.1:8554/<name>\n'
