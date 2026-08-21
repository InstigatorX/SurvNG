#!/usr/bin/env bash
# Upgrade a SurvNG checkout from Git, then rebuild native or Docker deployments.
set -euo pipefail

REMOTE="${SURVNG_UPDATE_REMOTE:-origin}"
BRANCH="${SURVNG_UPDATE_BRANCH:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .git ]]; then
  echo "SurvNG update requires a Git checkout at $ROOT" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked local changes are present. Commit or stash them before updating." >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

git fetch --prune "$REMOTE"

if [[ -z "$BRANCH" ]]; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  if [[ "$BRANCH" == "HEAD" ]]; then
    BRANCH=main
  fi
fi

UPSTREAM="$REMOTE/$BRANCH"
if ! git rev-parse --verify "$UPSTREAM" >/dev/null 2>&1; then
  echo "Remote branch $UPSTREAM was not found after fetch." >&2
  exit 1
fi

COUNTS="$(git rev-list --left-right --count "HEAD...$UPSTREAM")"
AHEAD="${COUNTS%%[$'\t' ]*}"
BEHIND="${COUNTS##*[$'\t' ]}"
if [[ "${AHEAD:-0}" -gt 0 ]]; then
  echo "Local commits prevent a fast-forward update ($AHEAD ahead of $UPSTREAM)." >&2
  exit 1
fi
if [[ "${BEHIND:-0}" -le 0 ]]; then
  echo "Already up to date with $UPSTREAM ($(git rev-parse --short HEAD))."
  exit 0
fi

echo "Updating $(git rev-parse --short HEAD) -> $(git rev-parse --short "$UPSTREAM") ($BEHIND commit(s))"
git pull --ff-only "$REMOTE" "$BRANCH"
echo "Now at $(git rev-parse --short HEAD)"

compose_files=(-f compose.yaml)
if [[ -f compose.intel-gpu.yaml ]] && docker compose -f compose.yaml -f compose.intel-gpu.yaml ps --status running --services 2>/dev/null | grep -qx survng; then
  compose_files+=(-f compose.intel-gpu.yaml)
fi
if [[ -f compose.lxc.yaml ]] && docker compose -f compose.yaml -f compose.lxc.yaml ps --status running --services 2>/dev/null | grep -qx survng; then
  compose_files+=(-f compose.lxc.yaml)
fi
if [[ -f compose.storage.yaml ]]; then
  compose_files+=(-f compose.storage.yaml)
fi

if [[ -z "${SURVNG_GIT_SHA:-}" ]] && command -v git >/dev/null 2>&1; then
  SURVNG_GIT_SHA="$(git rev-parse HEAD 2>/dev/null || true)"
  export SURVNG_GIT_SHA
fi

if command -v docker >/dev/null 2>&1 && docker compose "${compose_files[@]}" ps --status running --services 2>/dev/null | grep -qx survng; then
  echo "Rebuilding running Docker deployment"
  if [[ " ${compose_files[*]} " == *" compose.lxc.yaml "* ]]; then
    SURVNG_GIT_SHA="$SURVNG_GIT_SHA" scripts/docker-build-lxc.sh
    docker compose "${compose_files[@]}" up -d --no-build --remove-orphans
  else
    docker compose "${compose_files[@]}" build --pull --build-arg "SURVNG_GIT_SHA=${SURVNG_GIT_SHA}"
    docker compose "${compose_files[@]}" up -d --remove-orphans
  fi
  echo "Docker update complete"
  exit 0
fi

if [[ -x .venv/bin/pip ]]; then
  echo "Installing Python dependencies"
  .venv/bin/pip install -r requirements.txt
fi

if [[ -f frontend/package.json ]] && command -v npm >/dev/null 2>&1; then
  echo "Building frontend"
  (
    cd frontend
    npm ci --no-audit --no-fund
    npm run build
  )
fi

if command -v systemctl >/dev/null 2>&1 && systemctl cat survng.service >/dev/null 2>&1; then
  echo "Restarting survng.service"
  systemctl restart survng.service
fi

echo "Native update complete"
