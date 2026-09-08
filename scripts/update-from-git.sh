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

git fetch --prune "$REMOTE" "+refs/heads/*:refs/remotes/$REMOTE/*"

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

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
  if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    echo "Checking out $BRANCH"
    git switch "$BRANCH"
  else
    echo "Creating local branch $BRANCH from $UPSTREAM"
    git switch --create "$BRANCH" --track "$UPSTREAM"
  fi
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

# `compose ps` identifies project/service containers, regardless of the -f files
# passed to that command. Recover the deployment's actual file list instead of
# treating every available override as enabled.
compose_files=(-f compose.yaml)
running_container=""
use_lxc=false
if command -v docker >/dev/null 2>&1; then
  running_container="$(docker compose -f compose.yaml ps --status running -q survng 2>/dev/null || true)"
fi
if [[ -n "$running_container" ]]; then
  if [[ "$running_container" == *$'\n'* ]]; then
    echo "Multiple SurvNG containers found; update the intended Compose deployment explicitly." >&2
    exit 1
  fi
  recorded_files="$(docker inspect --format '{{index .Config.Labels "com.docker.compose.project.config_files"}}' "$running_container")"
  if [[ -z "$recorded_files" || "$recorded_files" == "<no value>" ]]; then
    echo "Cannot determine the running deployment's Compose files; refusing to guess update modes." >&2
    exit 1
  fi
  IFS=',' read -r -a deployment_files <<< "$recorded_files"
  compose_files=()
  for compose_file in "${deployment_files[@]}"; do
    if [[ ! -f "$compose_file" ]]; then
      echo "Running deployment references a missing Compose file: $compose_file" >&2
      exit 1
    fi
    compose_files+=(-f "$compose_file")
    case "${compose_file##*/}" in
      compose.lxc.yaml) use_lxc=true ;;
    esac
  done
fi

if [[ -z "${SURVNG_GIT_SHA:-}" ]] && command -v git >/dev/null 2>&1; then
  SURVNG_GIT_SHA="$(git rev-parse HEAD 2>/dev/null || true)"
  export SURVNG_GIT_SHA
fi

if [[ -n "$running_container" ]]; then
  echo "Rebuilding running Docker deployment"
  if [[ "$use_lxc" == true ]]; then
    SURVNG_GIT_SHA="$SURVNG_GIT_SHA" scripts/docker-build-lxc.sh --compose "${compose_files[@]}"
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
