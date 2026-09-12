#!/usr/bin/env bash
# Native capture dependencies for Ubuntu 24.04 amd64. Safe to rerun.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep matched with Dockerfile: older native OpenVINO produces NaN YOLO26 scores.
dlstreamer_version=2026.2.0

if [[ $# -ne 0 ]]; then
    echo "Usage: sudo $0 (Ubuntu 24.04 amd64)" >&2
    exit 2
fi
if [[ $EUID -ne 0 ]]; then
    echo "Run with sudo to install the native GStreamer / DL Streamer runtime." >&2
    exit 1
fi
source /etc/os-release
if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != 24.04 || "$(dpkg --print-architecture)" != amd64 ]]; then
    echo "This installer supports Ubuntu 24.04 amd64 only." >&2
    exit 1
fi

apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg
runtime_tmp="$(mktemp -d)"
trap 'rm -rf "$runtime_tmp"' EXIT
curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    -o "$runtime_tmp/openvino.asc"
gpg --batch --dearmor --output "$runtime_tmp/openvino.gpg" "$runtime_tmp/openvino.asc"
curl -fsSL https://apt.repos.intel.com/edgeai/dlstreamer/GPG-PUB-KEY-INTEL-DLS.gpg \
    -o "$runtime_tmp/dls.gpg"
install -m 644 "$runtime_tmp/openvino.gpg" /usr/share/keyrings/intel-gpg-archive-keyring.gpg
install -m 644 "$runtime_tmp/dls.gpg" /usr/share/keyrings/dls-archive-keyring.gpg
printf '%s\n' \
    'deb [signed-by=/usr/share/keyrings/dls-archive-keyring.gpg] https://apt.repos.intel.com/edgeai/dlstreamer/ubuntu24 ubuntu24 main' \
    > /etc/apt/sources.list.d/intel-dlstreamer.list
printf '%s\n' \
    'deb [signed-by=/usr/share/keyrings/intel-gpg-archive-keyring.gpg] https://apt.repos.intel.com/openvino ubuntu24 main' \
    > /etc/apt/sources.list.d/intel-openvino.list
apt-get update
apt-get install -y --no-install-recommends \
    python3 python3-gi python3-numpy \
    gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-libav gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-tools \
    "intel-dlstreamer=$dlstreamer_version"
apt-mark hold intel-dlstreamer
/usr/bin/python3 "$repo_dir/scripts/check-native-runtime.py"
