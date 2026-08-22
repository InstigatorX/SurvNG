#!/usr/bin/env bash
# Download SurvNG Docker model packages into the host models directory and
# point docker-data/config/config.json at the matching container paths.
#
# Default: runs the published survng-model-installer container so the host
# does not need PyTorch/Ultralytics venvs. Pass --native to run inline (dev).
#
# Standalone: does not require a SurvNG Git checkout. Set SURVNG_MODELS_DIR and
# SURVNG_CONFIG_DIR (or pass --models-dir / --config) for Docker host paths.
#
# None of these weights ship in the public MIT GHCR image. YOLO26s is
# Ultralytics AGPL-3.0; MobileCLIP2-B is Apple ML Research (non-commercial);
# person ReID is Intel Open Model Zoo Apache-2.0; vehicle ReID is MIT.
# License summaries: docker/model-installer/THIRD_PARTY_MODELS.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS_DIR="${SURVNG_MODELS_DIR:-}"
CONFIG_DIR="${SURVNG_CONFIG_DIR:-}"
CONFIG_PATH="${SURVNG_HOST_CONFIG_PATH:-}"
CACHE_DIR=""
DEVICE="CPU"
YOLO_NAME="yolo26s"
ENABLE=1
WRITE_CONFIG=1
FORCE=0
NATIVE=0
LXC_APPARMOR=0
DO_DETECTOR=1
DO_PERSON_REID=1
DO_VEHICLE_REID=1
DO_SEMANTIC=1
PYTHON_BIN=""
SEMANTIC_EXPORTER_REF="${SURVNG_INSTALLER_REF:-v1.0}"
IN_CONTAINER="${SURVNG_INSTALLER_IN_CONTAINER:-0}"
INSTALLER_IMAGE="${SURVNG_MODEL_INSTALLER_IMAGE:-ghcr.io/instigatorx/survng:v1.0-model-installer}"
INSTALL_FAILURES=0

PERSON_NAME="person-reidentification-retail-0286"
PERSON_URL="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-reidentification-retail-0286/FP16"
VEHICLE_URL="https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.onnx"
VEHICLE_SHA256="4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f"
SEMANTIC_EXPORTER_URL="https://raw.githubusercontent.com/InstigatorX/SurvNG/${SEMANTIC_EXPORTER_REF}/scripts/export-mobileclip2-openvino.py"

usage() {
  cat <<'EOF'
Usage: install-docker-models.sh [options]

Download object-detection, person/vehicle ReID, and Smart Search models into
the Docker models directory, then patch config.json with container paths
under /models. Existing cameras and unrelated settings are left untouched.

No SurvNG Git checkout is required. Typical Docker host layout:

  SURVNG_MODELS_DIR=/docker-data/models
  SURVNG_CONFIG_DIR=/docker-data/config

Options:
  --models-dir DIR     Host models directory (default: ./docker-data/models)
  --config PATH        Host config.json (default: ./docker-data/config/config.json)
  --cache-dir DIR      Download/export cache (default: MODELS_DIR/.download-cache)
  --device CPU|GPU     Inference device written to config.json (default: CPU)
  --yolo NAME          Ultralytics detect weights to export (default: yolo26s)
  --python PATH        Python used for venvs and config patching (--native only)
  --native             Run on this host (requires python3 + venvs); skip installer image
  --installer-image IMG  Override installer image (default: ghcr.io/.../survng:v1.0-model-installer)
  --lxc                Proxmox/LXC nested Docker: pass --security-opt apparmor=unconfined
                       to the installer container. Weakens isolation; opt-in only
                       (same trade-off as compose.lxc.yaml for the SurvNG service).
  --force              Re-download / re-export even when files already exist
  --no-enable          Write model paths but leave enabled flags unchanged
  --skip-config        Download only; do not create or patch config.json
  --skip-detector      Skip YOLO26 OpenVINO export
  --skip-reid          Skip person and vehicle ReID downloads
  --skip-person-reid   Skip person ReID only
  --skip-vehicle-reid  Skip vehicle ReID only
  --skip-semantic      Skip MobileCLIP2-B Smart Search export
  -h, --help           Show this help

Environment:
  SURVNG_MODELS_DIR, SURVNG_CONFIG_DIR, SURVNG_HOST_CONFIG_PATH
  SURVNG_INSTALLER_REF          Git ref for bundled exporter download (default: v1.0)
  SURVNG_MODEL_INSTALLER_IMAGE    Installer image (default: ghcr.io/instigatorx/survng:v1.0-model-installer)
  SURVNG_INSTALLER_IN_CONTAINER   Set by the installer image entrypoint (do not set on host)

Licenses (not SurvNG MIT; not baked into GHCR images):
  YOLO26s              Ultralytics AGPL-3.0
  MobileCLIP2-B        Apple ML Research Model terms (research/non-commercial)
  person-reid-0286     Intel Open Model Zoo Apache-2.0
  vehicle-reid-0001    MIT (OSNet / Open Model Zoo public)

Examples:
  # Default: installer container (no host PyTorch/Ultralytics)
  SURVNG_MODELS_DIR=/docker-data/models \
  SURVNG_CONFIG_DIR=/docker-data/config \
  ./install-docker-models.sh --device GPU

  # Proxmox/LXC nested Docker (AppArmor unconfined; intentional isolation trade-off)
  ./install-docker-models.sh --lxc --device GPU

  # Dev checkout / air-gapped: run inline
  ./install-docker-models.sh --native --device CPU

Default layout under the models directory (container /models):
  yolo26s_openvino_model/yolo26s.xml
  yolo26s_openvino_model/classes.txt
  person_reid_model/person-reidentification-retail-0286.xml
  vehicle_reid_model/vehicle-reid-0001.onnx
  mobileclip2-b-openvino-fp16/
EOF
}

log() { printf '%s\n' "$*"; }
err() { printf '%s\n' "$*" >&2; }

note_step_failure() {
  local step="$1"
  err "FAILED: $step (continuing; config.json will reflect any models that did install)"
  INSTALL_FAILURES=1
}

run_installer_container() {
  need_cmd docker
  local models_host config_host cache_host
  models_host="$(absolute_path "$MODELS_DIR")"
  config_host="$(absolute_path "$CONFIG_DIR")"
  cache_host="$(absolute_path "$CACHE_DIR")"
  mkdir -p "$models_host" "$config_host" "$cache_host"

  local -a docker_run=(docker run --rm --user "$(id -u):$(id -g)")
  if [[ "$LXC_APPARMOR" -eq 1 ]]; then
    docker_run+=(--security-opt apparmor=unconfined)
  fi
  docker_run+=(
    -v "$models_host:/models-out"
    -v "$config_host:/config-out"
    -v "$cache_host:/cache"
    -e "SURVNG_INSTALLER_IN_CONTAINER=1"
    -e "SURVNG_INSTALLER_REF=${SEMANTIC_EXPORTER_REF}"
  )

  local -a inner=(--native --device "$DEVICE" --yolo "$YOLO_NAME")
  [[ -n "$CACHE_DIR" ]] && inner+=(--cache-dir /cache)
  [[ "$FORCE" -eq 1 ]] && inner+=(--force)
  [[ "$ENABLE" -eq 0 ]] && inner+=(--no-enable)
  [[ "$WRITE_CONFIG" -eq 0 ]] && inner+=(--skip-config)
  [[ "$DO_DETECTOR" -eq 0 ]] && inner+=(--skip-detector)
  [[ "$DO_PERSON_REID" -eq 0 && "$DO_VEHICLE_REID" -eq 0 ]] && inner+=(--skip-reid)
  [[ "$DO_PERSON_REID" -eq 0 ]] && inner+=(--skip-person-reid)
  [[ "$DO_VEHICLE_REID" -eq 0 ]] && inner+=(--skip-vehicle-reid)
  [[ "$DO_SEMANTIC" -eq 0 ]] && inner+=(--skip-semantic)

  log "Running installer container: $INSTALLER_IMAGE"
  log "  models mount: $models_host -> /models-out"
  log "  config mount: $config_host -> /config-out"
  log "  cache mount:  $cache_host -> /cache"
  if [[ "$LXC_APPARMOR" -eq 1 ]]; then
    log "  security:   apparmor=unconfined (Proxmox/LXC nested Docker; weakens isolation)"
  fi
  log "License notices for downloaded models: docker/model-installer/THIRD_PARTY_MODELS.md"
  if ! docker pull "$INSTALLER_IMAGE"; then
    err "Could not pull $INSTALLER_IMAGE"
    err "The installer image is published as ghcr.io/instigatorx/survng:v1.0-model-installer"
    err "If the package is private, run: docker login ghcr.io"
    err "Or run inline on this host without Docker: install-docker-models.sh --native [options]"
    return 1
  fi
  "${docker_run[@]}" "$INSTALLER_IMAGE" "${inner[@]}"
}

normalize_path() {
  local path="$1"
  while [[ "$path" == *//* ]]; do
    path="${path//\/\//\/}"
  done
  printf '%s\n' "$path"
}

absolute_path() {
  local raw="$1"
  raw="$(normalize_path "$raw")"
  if [[ "$raw" != /* ]]; then
    raw="$(normalize_path "$(pwd)/${raw#./}")"
  fi
  mkdir -p "$(dirname "$raw")"
  if [[ -d "$raw" ]]; then
    normalize_path "$(cd "$raw" && pwd)"
  else
    local parent child
    parent="$(cd "$(dirname "$raw")" && pwd)"
    child="$(basename "$raw")"
    normalize_path "$parent/$child"
  fi
}

container_path() {
  local abs="$1"
  local rel="${abs#"$MODELS_DIR"/}"
  if [[ "$rel" == "$abs" ]]; then
    err "Refusing to map $abs; it is not under $MODELS_DIR"
    exit 1
  fi
  printf '/models/%s\n' "$rel"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    err "Required command not found: $1"
    exit 1
  fi
}

pick_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    return
  fi
  if [[ -x "$(pwd)/.venv/bin/python" ]]; then
    PYTHON_BIN="$(pwd)/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    err "python3 is required to patch config.json and export models"
    exit 1
  fi
}

download_file() {
  local url="$1"
  local dest="$2"
  local tmp="${dest}.part"
  mkdir -p "$(dirname "$dest")"
  curl -fL --retry 5 --retry-delay 2 -o "$tmp" "$url"
  mv -f "$tmp" "$dest"
}

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

require_openvino_ir() {
  local xml="$1"
  if ! grep -q '<net ' "$xml"; then
    err "Downloaded model is not a valid OpenVINO IR: $xml"
    exit 1
  fi
}

# Host-native and container runs must not share venvs: shebangs and symlinks
# break across environments. Keep HF/download caches on CACHE_DIR; put venvs
# in a local path when running inside the installer image.
venv_root() {
  if [[ "$IN_CONTAINER" -eq 1 ]]; then
    printf '/var/tmp/survng-install-venvs\n'
  else
    printf '%s\n' "$CACHE_DIR"
  fi
}

venv_usable() {
  local venv_dir="$1"
  [[ -x "$venv_dir/bin/python" ]] || return 1
  "$venv_dir/bin/python" -c 'import sys' 2>/dev/null || return 1
}

install_cpu_torch() {
  local venv_dir="$1"
  # Export needs torch for Ultralytics/OpenCLIP, not CUDA. Default PyPI torch
  # pulls multi-GB NVIDIA wheels; pin the CPU index instead.
  "$venv_dir/bin/python" -m pip install --upgrade pip >/dev/null
  "$venv_dir/bin/python" -m pip install \
    'torch>=2.5,<3' \
    'torchvision>=0.20,<1' \
    --index-url https://download.pytorch.org/whl/cpu
}

# Ultralytics may pull CUDA torch or opencv-python 5.x / headless mixes.
# Reassert CPU torch and a single full opencv-python 4.x build. Ultralytics
# imports cv2.imshow at module load; headless has no imshow. System libs in
# Dockerfile.model-installer cover the .so deps (libxcb, libGL, …).
finalize_export_venv() {
  local venv_dir="$1"
  install_cpu_torch "$venv_dir"
  "$venv_dir/bin/python" -m pip uninstall -y opencv-python-headless >/dev/null 2>&1 || true
  "$venv_dir/bin/python" -m pip install 'opencv-python>=4.8,<5'
}

ensure_venv() {
  local venv_dir="$1"
  shift
  pick_python
  if ! venv_usable "$venv_dir"; then
    if ! "$PYTHON_BIN" -c 'import venv' 2>/dev/null; then
      err "Python venv support is required. Install python3-venv and retry."
      exit 1
    fi
    rm -rf "$venv_dir"
    mkdir -p "$(dirname "$venv_dir")"
    "$PYTHON_BIN" -m venv "$venv_dir"
  fi
  if ! venv_usable "$venv_dir"; then
    err "Failed to create a usable Python venv at $venv_dir"
    exit 1
  fi
  "$venv_dir/bin/python" -m pip install --upgrade pip >/dev/null
  if [[ "$#" -gt 0 ]]; then
    "$venv_dir/bin/python" -m pip install "$@"
  fi
}

semantic_exporter_path() {
  local sibling="$SCRIPT_DIR/export-mobileclip2-openvino.py"
  local cached="$CACHE_DIR/export-mobileclip2-openvino.py"
  if [[ -f "$sibling" ]]; then
    printf '%s\n' "$sibling"
    return 0
  fi
  if [[ -f "$cached" && "$FORCE" -eq 0 ]]; then
    printf '%s\n' "$cached"
    return 0
  fi
  need_cmd curl
  log "Fetching MobileCLIP exporter (${SEMANTIC_EXPORTER_REF})..."
  download_file "$SEMANTIC_EXPORTER_URL" "$cached"
  printf '%s\n' "$cached"
}

write_classes_from_metadata() {
  local model_dir="$1"
  local metadata="$model_dir/metadata.yaml"
  local classes="$model_dir/classes.txt"
  pick_python
  "$PYTHON_BIN" - "$metadata" "$classes" <<'PY'
import re
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
classes_path = Path(sys.argv[2])
names: list[str] = []
if metadata_path.is_file():
    text = metadata_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text) or {}
        raw = payload.get("names") or []
        if isinstance(raw, dict):
            names = [str(raw[key]) for key in sorted(raw, key=lambda item: int(item))]
        elif isinstance(raw, list):
            names = [str(item) for item in raw]
    except Exception:
        names = []
    if not names:
        in_names = False
        for line in text.splitlines():
            if re.match(r"^names:\s*$", line):
                in_names = True
                continue
            if in_names:
                match = re.match(r"^\s+\d+:\s*(.+)$", line)
                if not match:
                    break
                names.append(match.group(1).strip().strip("'\""))
if names:
    classes_path.write_text("\n".join(names) + "\n", encoding="utf-8")
PY
}

install_detector() {
  local model_dir="$MODELS_DIR/${YOLO_NAME}_openvino_model"
  local xml="$model_dir/${YOLO_NAME}.xml"
  local bin="$model_dir/${YOLO_NAME}.bin"
  if [[ -f "$xml" && -f "$bin" && "$FORCE" -eq 0 ]]; then
    log "Detector already present: $model_dir"
    [[ -f "$model_dir/classes.txt" ]] || write_classes_from_metadata "$model_dir"
    return
  fi
  need_cmd curl
  log "Exporting ${YOLO_NAME} to OpenVINO FP16 (Ultralytics AGPL-3.0)..."
  local venv
  venv="$(venv_root)/yolo-venv"
  local work="$CACHE_DIR/yolo-export"
  mkdir -p "$work"
  ensure_venv "$venv"
  install_cpu_torch "$venv"
  # Full opencv-python: Ultralytics requires cv2.imshow at import time.
  # Pin <5 to stay on the manylinux builds exercised with libxcb in the image.
  ensure_venv "$venv" \
    'opencv-python>=4.8,<5' \
    'ultralytics>=8.4,<9' \
    'openvino>=2025.1'
  finalize_export_venv "$venv"
  (
    cd "$work"
    "$venv/bin/python" - "$YOLO_NAME" <<'PY'
import sys
from pathlib import Path

from ultralytics import YOLO

name = sys.argv[1]
exported = YOLO(f"{name}.pt").export(
    format="openvino",
    imgsz=640,
    quantize=16,
    nms=False,
)
print(Path(exported))
PY
  )
  local exported
  exported="$(find "$work" -maxdepth 1 -type d -name "${YOLO_NAME}_openvino_model" -print -quit)"
  if [[ -z "$exported" || ! -d "$exported" ]]; then
    err "YOLO OpenVINO export did not produce ${YOLO_NAME}_openvino_model"
    exit 1
  fi
  rm -rf "$model_dir"
  mkdir -p "$(dirname "$model_dir")"
  mv "$exported" "$model_dir"
  if [[ ! -f "$xml" ]]; then
    local found_xml
    found_xml="$(find "$model_dir" -maxdepth 2 -name '*.xml' -print -quit)"
    if [[ -n "$found_xml" && "$(basename "$found_xml")" != "${YOLO_NAME}.xml" ]]; then
      err "Expected $xml after export; found $found_xml"
      exit 1
    fi
  fi
  write_classes_from_metadata "$model_dir"
  if [[ ! -f "$model_dir/classes.txt" ]]; then
    err "YOLO export succeeded but classes.txt could not be generated from metadata.yaml"
    exit 1
  fi
  log "Installed detector: $model_dir"
}

install_person_reid() {
  local model_dir="$MODELS_DIR/person_reid_model"
  local xml="$model_dir/${PERSON_NAME}.xml"
  local bin="$model_dir/${PERSON_NAME}.bin"
  if [[ -f "$xml" && -f "$bin" && "$FORCE" -eq 0 ]]; then
    log "Person ReID already present: $model_dir"
    return
  fi
  need_cmd curl
  log "Downloading person ReID ${PERSON_NAME} (Intel OMZ Apache-2.0)..."
  mkdir -p "$model_dir"
  download_file "${PERSON_URL}/${PERSON_NAME}.xml" "$xml"
  download_file "${PERSON_URL}/${PERSON_NAME}.bin" "$bin"
  require_openvino_ir "$xml"
  log "Installed person ReID: $xml"
}

install_vehicle_reid() {
  local model_dir="$MODELS_DIR/vehicle_reid_model"
  local onnx="$model_dir/vehicle-reid-0001.onnx"
  if [[ -f "$onnx" && "$FORCE" -eq 0 ]]; then
    log "Vehicle ReID already present: $onnx"
    return
  fi
  need_cmd curl
  log "Downloading vehicle ReID vehicle-reid-0001.onnx (MIT)..."
  log "  OMZ publishes vehicle-reid-0001 as ONNX only; OpenVINO loads it directly."
  log "  Person ReID uses pre-built IR (xml+bin). Face models (optional) are IR too."
  mkdir -p "$model_dir"
  download_file "$VEHICLE_URL" "$onnx"
  local actual
  actual="$(sha256_of "$onnx")"
  if [[ "$actual" != "$VEHICLE_SHA256" ]]; then
    err "vehicle-reid-0001.onnx SHA-256 mismatch:"
    err "  expected $VEHICLE_SHA256"
    err "  actual   $actual"
    rm -f "$onnx"
    exit 1
  fi
  log "Installed vehicle ReID: $onnx"
}

install_semantic() {
  local model_dir="$MODELS_DIR/mobileclip2-b-openvino-fp16"
  local exporter
  if [[ -f "$model_dir/semantic_model.json" && -f "$model_dir/image_encoder.xml" && "$FORCE" -eq 0 ]]; then
    log "Smart Search package already present: $model_dir"
    return
  fi
  exporter="$(semantic_exporter_path)"
  log "Exporting MobileCLIP2-B OpenVINO package (Apple ML Research terms)..."
  local venv
  venv="$(venv_root)/semantic-venv"
  ensure_venv "$venv"
  install_cpu_torch "$venv"
  ensure_venv "$venv" \
    'open_clip_torch>=3.2,<4' \
    'timm>=1.0.20,<2' \
    'huggingface_hub>=0.34,<2' \
    'Pillow>=10,<13' \
    'openvino>=2025.1'
  finalize_export_venv "$venv"
  local force_flag=()
  if [[ "$FORCE" -eq 1 ]]; then
    force_flag=(--force)
  fi
  "$venv/bin/python" "$exporter" \
    --output "$model_dir" \
    --cache-dir "$CACHE_DIR/hf" \
    "${force_flag[@]}"
  log "Installed Smart Search package: $model_dir"
}

# Model packages must be readable by the SurvNG container user, which may differ
# from the installer UID (e.g. root installer, SURVNG_UID=1000). mkdtemp-based
# exports start as 0700 and would otherwise cause PermissionError on /models.
ensure_models_readable() {
  pick_python
  "$PYTHON_BIN" - "$MODELS_DIR" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.is_dir():
    raise SystemExit(0)
for path in [root, *root.rglob("*")]:
    name = path.name
    if name == ".download-cache" or ".download-cache" in path.parts:
        continue
    try:
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o644)
    except OSError as exc:
        print(f"Warning: could not chmod {path}: {exc}", file=sys.stderr)
print(f"Made model files under {root} world-readable (dirs 0755, files 0644)")
PY
}

patch_config() {
  pick_python
  local detector_xml="$MODELS_DIR/${YOLO_NAME}_openvino_model/${YOLO_NAME}.xml"
  local labels="$MODELS_DIR/${YOLO_NAME}_openvino_model/classes.txt"
  local person_xml="$MODELS_DIR/person_reid_model/${PERSON_NAME}.xml"
  local vehicle_onnx="$MODELS_DIR/vehicle_reid_model/vehicle-reid-0001.onnx"
  local semantic_dir="$MODELS_DIR/mobileclip2-b-openvino-fp16"

  local detector_container="" labels_container="" person_container=""
  local vehicle_container="" semantic_container=""
  [[ -f "$detector_xml" ]] && detector_container="$(container_path "$detector_xml")"
  [[ -f "$labels" ]] && labels_container="$(container_path "$labels")"
  [[ -f "$person_xml" ]] && person_container="$(container_path "$person_xml")"
  [[ -f "$vehicle_onnx" ]] && vehicle_container="$(container_path "$vehicle_onnx")"
  if [[ -f "$semantic_dir/semantic_model.json" ]]; then
    semantic_container="$(container_path "$semantic_dir")"
  fi

  if [[ -z "$detector_container$person_container$vehicle_container$semantic_container" ]]; then
    err "No installed models found under $MODELS_DIR; not writing config.json"
    exit 1
  fi

  mkdir -p "$(dirname "$CONFIG_PATH")"
  "$PYTHON_BIN" - "$CONFIG_PATH" "$DEVICE" "$ENABLE" \
    "$detector_container" "$labels_container" "$person_container" \
    "$vehicle_container" "$semantic_container" <<'PY'
import json
import os
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
device = sys.argv[2]
enable = sys.argv[3] == "1"
detector_xml, labels, person_xml, vehicle_onnx, semantic_dir = sys.argv[4:9]

DOCKER_DEFAULT = {
    "base_path": "/survng",
    "storage_dir": "/media",
    "database_dir": "/data/database",
    "recording_index_dir": "/data/recording-index",
    "ffmpeg_path": "/usr/bin/ffmpeg",
    "hardware_acceleration": "auto",
    "recording_segment_seconds": 10.0,
    "recording_cache_prewarm": True,
    "image_storage": {"format": "webp", "quality": 95},
    "retention": {
        "enabled": True,
        "automatic_cleanup": False,
        "storage_limit_tb": 1.0,
        "minimum_free_percent": 15.0,
        "target_free_percent": 20.0,
        "emergency_free_percent": 5.0,
        "main_days": 7,
        "live_days": 21,
        "cleanup_batch_files": 2000,
    },
    "motion_qualification": {
        "mode": "camera",
        "sensitivity": "balanced",
        "frame_width": 320,
        "sample_fps": 5.0,
        "temporal_filter_threshold": 0.005,
        "window_seconds": 1.6,
        "post_trigger_seconds": 2.5,
        "burst_quiet_seconds": 0.5,
        "rejected_sample_rate": 1.0,
        "borderline_rescue_enabled": True,
        "borderline_margin": 0.03,
        "pipeline": {"qualification": [], "observation": [], "fusion": []},
    },
    "audit_ai": {
        "enabled": False,
        "provider": "openai",
        "api_key": "",
        "base_url": "",
        "model": "",
        "timeout_seconds": 45.0,
        "allow_apply_recommendations": False,
    },
    "mqtt": {
        "enabled": False,
        "host": "",
        "port": 1883,
        "username": "",
        "password": "",
        "client_id": "survng",
        "topic_prefix": "survng",
        "qos": 0,
        "tls": False,
        "discovery_enabled": True,
        "discovery_prefix": "homeassistant",
        "incident_events_enabled": True,
    },
    "detector": {
        "enabled": False,
        "model_path": "/models/detector/model.xml",
        "labels_path": "/models/detector/classes.txt",
        "device": "CPU",
        "cache_enabled": True,
        "cache_dir": "/data/openvino-cache",
        "confidence_threshold": 0.45,
        "nms_threshold": 0.45,
        "event_confirmation_frames": 2,
        "event_class_confirmation_frames": {},
        "event_class_confidence_thresholds": {},
        "labels": [],
        "tracking": {
            "enabled": True,
            "implementation": "survng_hybrid",
            "excluded_labels": ["face"],
            "sample_fps": 2.0,
            "max_session_seconds": 15.0,
            "lost_timeout_seconds": 3.0,
            "min_confirmations": 2,
            "low_confidence_threshold": 0.25,
            "match_iou_threshold": 0.2,
            "match_center_distance_ratio": 0.65,
            "max_active_cameras": 2,
            "max_tracks_per_session": 100,
            "reid_enabled": False,
            "reid_model_path": "",
            "reid_device": "AUTO",
            "reid_match_threshold": 0.7,
            "reid_max_age_seconds": 30.0,
            "reid_max_embeddings_per_frame": 8,
            "reid_refresh_interval_frames": 8,
            "botsort_match_threshold": 0.8,
            "botsort_proximity_threshold": 0.1,
            "botsort_fuse_score": True,
        },
    },
    "cameras": [],
}

if config_path.exists():
    payload = json.loads(config_path.read_text(encoding="utf-8"))
else:
    payload = json.loads(json.dumps(DOCKER_DEFAULT))

if not isinstance(payload, dict):
    raise SystemExit("config.json must contain a JSON object")

def section(*keys: str) -> dict:
    current = payload
    for key in keys:
        value = current.get(key)
        if not isinstance(value, dict):
            value = {}
            current[key] = value
        current = value
    return current

if detector_xml:
    detector = section("detector")
    detector["model_path"] = detector_xml
    if labels:
        detector["labels_path"] = labels
    detector["device"] = device
    detector.setdefault("cache_dir", "/data/openvino-cache")
    detector.setdefault("cache_enabled", True)
    if enable:
        detector["enabled"] = True

if person_xml or vehicle_onnx:
    tracking = section("detector", "tracking")
    if person_xml:
        tracking["reid_model_path"] = person_xml
        tracking.setdefault("reid_device", "AUTO")
        if enable:
            tracking["reid_enabled"] = True
    if vehicle_onnx:
        tracking["vehicle_reid_model_path"] = vehicle_onnx
        tracking.setdefault("vehicle_reid_device", "AUTO")
        if enable:
            tracking["vehicle_reid_enabled"] = True

if semantic_dir:
    semantic = section("semantic_search")
    semantic["model_dir"] = semantic_dir
    semantic["device"] = device
    semantic.setdefault("implementation", "mobileclip2_openvino")
    if enable:
        semantic["enabled"] = True

temporary = config_path.with_name(config_path.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, config_path)
os.chmod(config_path, 0o600)
PY
  log "Updated $CONFIG_PATH (mode 0600). Cameras and unrelated keys were preserved."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-dir) MODELS_DIR="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --device)
      DEVICE="${2^^}"
      if [[ "$DEVICE" != "CPU" && "$DEVICE" != "GPU" ]]; then
        err "--device must be CPU or GPU"
        exit 2
      fi
      shift 2
      ;;
    --yolo) YOLO_NAME="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --native) NATIVE=1; shift ;;
    --installer-image) INSTALLER_IMAGE="$2"; shift 2 ;;
    --lxc) LXC_APPARMOR=1; shift ;;
    --force) FORCE=1; shift ;;
    --no-enable) ENABLE=0; shift ;;
    --skip-config) WRITE_CONFIG=0; shift ;;
    --skip-detector) DO_DETECTOR=0; shift ;;
    --skip-reid) DO_PERSON_REID=0; DO_VEHICLE_REID=0; shift ;;
    --skip-person-reid) DO_PERSON_REID=0; shift ;;
    --skip-vehicle-reid) DO_VEHICLE_REID=0; shift ;;
    --skip-semantic) DO_SEMANTIC=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      err "Unknown option: $1"
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "$MODELS_DIR" ]] || MODELS_DIR="./docker-data/models"
[[ -n "$CONFIG_DIR" ]] || CONFIG_DIR="./docker-data/config"
[[ -n "$CONFIG_PATH" ]] || CONFIG_PATH="$CONFIG_DIR/config.json"

if [[ "$NATIVE" -eq 0 && "$IN_CONTAINER" -eq 0 ]]; then
  if [[ -z "${CACHE_DIR}" ]]; then
    CACHE_DIR="$MODELS_DIR/.download-cache"
  fi
  run_installer_container
  exit $?
fi

MODELS_DIR="$(absolute_path "$MODELS_DIR")"
CONFIG_PATH="$(absolute_path "$CONFIG_PATH")"
if [[ "$IN_CONTAINER" -eq 1 ]]; then
  MODELS_DIR="/models-out"
  CONFIG_PATH="/config-out/config.json"
  [[ -z "$CACHE_DIR" ]] && CACHE_DIR="/cache"
fi
if [[ -z "$CACHE_DIR" ]]; then
  CACHE_DIR="$MODELS_DIR/.download-cache"
fi
CACHE_DIR="$(absolute_path "$CACHE_DIR")"
mkdir -p "$MODELS_DIR" "$CACHE_DIR"

log "SurvNG Docker model installer"
log "  models: $MODELS_DIR  (container /models)"
log "  config: $CONFIG_PATH"
log "  device: $DEVICE"
log "  mode:   $([[ "$IN_CONTAINER" -eq 1 ]] && echo container || echo native)"
log "YOLO26s is AGPL-3.0. MobileCLIP2-B is Apple research/non-commercial."
log "Person ReID is Apache-2.0. Vehicle ReID is MIT. These are not in GHCR."
log "Attributions: docker/model-installer/THIRD_PARTY_MODELS.md"

if [[ "$DO_DETECTOR" -eq 1 ]]; then
  install_detector || note_step_failure "detector export"
fi
if [[ "$DO_PERSON_REID" -eq 1 ]]; then
  install_person_reid || note_step_failure "person ReID download"
fi
if [[ "$DO_VEHICLE_REID" -eq 1 ]]; then
  install_vehicle_reid || note_step_failure "vehicle ReID download"
fi
if [[ "$DO_SEMANTIC" -eq 1 ]]; then
  install_semantic || note_step_failure "MobileCLIP2-B export"
fi

ensure_models_readable || note_step_failure "model permission fix"

if [[ "$WRITE_CONFIG" -eq 1 ]]; then
  if ! patch_config; then
    note_step_failure "config.json patch"
  fi
fi

if [[ "$INSTALL_FAILURES" -ne 0 ]]; then
  err "One or more install steps failed. config.json was updated for models that exist."
  err "Fix errors above and re-run, or pass --skip-* for steps you do not need."
  exit 1
fi

log "Done. Restart the container if it is already running so it reloads config.json:"
log "  docker compose restart survng"
