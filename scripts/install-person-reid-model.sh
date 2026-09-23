#!/usr/bin/env bash
# Native/dev helper to install torchreid OSNet-x1.0 MSMT17 as OpenVINO IR.
# Intel OMZ person-reidentification-retail-0286 remains the fallback package
# installed by scripts/install-docker-models.sh.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${1:-${root_dir}/models/person_reid_model}"
output_xml="${model_dir}/osnet_x1_0_msmt17.xml"
cache_dir="${SURVNG_OSNET_CACHE_DIR:-${root_dir}/models/.osnet-download-cache}"

if [[ -x "${root_dir}/.venv/bin/python" ]]; then
  python_bin="${root_dir}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
else
  printf 'python3 was not found.\n' >&2
  exit 1
fi

if [[ -x "${root_dir}/.venv/bin/ovc" ]]; then
  ovc_bin="${root_dir}/.venv/bin/ovc"
elif command -v ovc >/dev/null 2>&1; then
  ovc_bin="$(command -v ovc)"
else
  printf 'OpenVINO converter (ovc) was not found. Install OpenVINO first.\n' >&2
  exit 1
fi

mkdir -p "$model_dir" "$cache_dir"
"$python_bin" "${root_dir}/scripts/export-osnet-openvino.py" \
  --output "$output_xml" \
  --cache-dir "$cache_dir" \
  --ovc "$ovc_bin"

if ! grep -q '<net ' "$output_xml"; then
  printf 'Downloaded model is not a valid OpenVINO IR: %s\n' "$output_xml" >&2
  exit 1
fi

printf 'Installed OSNet-x1.0 MSMT17 person ReID in %s\n' "$model_dir"
printf 'Point detector.tracking.reid_model_path at %s\n' "$output_xml"
