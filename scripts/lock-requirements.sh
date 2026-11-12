#!/usr/bin/env bash
# Regenerate requirements.lock (fully pinned, hash-locked runtime dependencies)
# from requirements.txt, which remains the human-edited source of version ranges.
#
# The lock is universal (all platforms, Python >= 3.12), so the Darwin-only
# entries in requirements.txt keep their environment markers.
#
# Requires uv (https://docs.astral.sh/uv/), e.g. `python3 -m pip install uv`.
# Pass --upgrade to move every pin to the newest version the ranges allow;
# without it, existing pins are kept where they still satisfy requirements.txt.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if command -v uv >/dev/null 2>&1; then
  uv_cmd=(uv)
elif python3 -m uv --version >/dev/null 2>&1; then
  uv_cmd=(python3 -m uv)
else
  echo "uv is required: python3 -m pip install uv" >&2
  exit 2
fi

UV_CUSTOM_COMPILE_COMMAND="scripts/lock-requirements.sh" \
  "${uv_cmd[@]}" pip compile \
  --universal \
  --generate-hashes \
  --python-version 3.12 \
  --no-emit-index-url \
  --quiet \
  "$@" \
  requirements.txt \
  -o requirements.lock
