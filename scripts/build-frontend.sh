#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir/frontend"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to build the SurvNG frontend" >&2
  exit 1
fi

if [[ ! -d node_modules ]]; then
  npm ci --no-audit --no-fund
fi

npm run build
echo "Frontend built to survng/static/"
