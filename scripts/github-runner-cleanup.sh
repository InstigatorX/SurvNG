#!/usr/bin/env bash
# Reclaim disk on the self-hosted GitHub Actions runner.
# Safe for scheduled maintenance and pre/post Docker publish jobs.
set -euo pipefail

MODE="standard"
RUNNER_ROOT="${GITHUB_RUNNER_ROOT:-/home/github-runner/actions-runner}"
MIN_FREE_PCT="${RUNNER_MIN_FREE_PCT:-15}"

usage() {
  cat <<'EOF'
Usage: github-runner-cleanup.sh [--light | --publish | --standard | --aggressive]

Modes:
  --light       Dangling Docker layers only; no age-based image deletion.
  --publish     Dangling images/containers only. Keeps the Docker build cache
                used by GHCR publish. Never escalates to a cache wipe.
  --standard    Default. Prune build cache and images older than 24h.
  --aggressive  Prune all unused Docker images, stale runner temp dirs, tool caches.

When root filesystem free space falls below RUNNER_MIN_FREE_PCT (default 15%),
the script escalates automatically to the next stronger mode.

Environment:
  GITHUB_RUNNER_ROOT   Runner install path (default: /home/github-runner/actions-runner)
  RUNNER_MIN_FREE_PCT  Escalate when free space is below this percent (default: 15)
EOF
}

log() {
  printf '[runner-cleanup] %s\n' "$*"
}

disk_free_pct() {
  df -P / | awk 'NR==2 {gsub(/%/,"",$5); print 100 - $5}'
}

report_disk() {
  log "Disk usage:"
  df -h /
  if command -v docker >/dev/null 2>&1; then
    docker system df || true
  fi
}

maybe_escalate_mode() {
  local free_pct="$1"
  if [[ "$MODE" == "publish" ]]; then
    if (( free_pct < MIN_FREE_PCT )); then
      log "Free space ${free_pct}% < ${MIN_FREE_PCT}% during publish; keeping build cache"
    fi
    return
  fi
  if (( free_pct < MIN_FREE_PCT )); then
    case "$MODE" in
      light) MODE="standard"; log "Free space ${free_pct}% < ${MIN_FREE_PCT}%; escalating to standard" ;;
      standard) MODE="aggressive"; log "Free space ${free_pct}% < ${MIN_FREE_PCT}%; escalating to aggressive" ;;
    esac
  elif (( free_pct < MIN_FREE_PCT + 10 )) && [[ "$MODE" == "light" ]]; then
    MODE="standard"
    log "Free space ${free_pct}% is low; escalating to standard"
  fi
}

cleanup_docker_publish() {
  # Keep layer cache. Do not builder prune -af. The next GHCR publish reuses
  # local layers from the moving tip left on the runner.
  command -v docker >/dev/null 2>&1 || return 0
  docker image prune -f || true
  docker container prune -f || true
}

cleanup_docker_light() {
  command -v docker >/dev/null 2>&1 || return 0
  docker builder prune -f || true
  docker image prune -f || true
  docker container prune -f || true
}

cleanup_docker_standard() {
  command -v docker >/dev/null 2>&1 || return 0
  docker builder prune -af || true
  docker image prune -af --filter "until=24h" || true
  docker container prune -f || true
}

cleanup_docker_aggressive() {
  command -v docker >/dev/null 2>&1 || return 0
  docker builder prune -af || true
  docker image prune -af || true
  docker container prune -f || true
}

cleanup_runner_temp() {
  local temp_dir="${RUNNER_ROOT}/_work/_temp"
  [[ -d "$temp_dir" ]] || return 0
  log "Removing runner temp dirs older than 48h under ${temp_dir}"
  find "$temp_dir" -mindepth 1 -maxdepth 1 -type d -mtime +2 -print -exec rm -rf {} + 2>/dev/null || true
}

cleanup_tool_caches() {
  if command -v npm >/dev/null 2>&1; then
    npm cache clean --force >/dev/null 2>&1 || true
  fi
  if command -v pip >/dev/null 2>&1; then
    pip cache purge >/dev/null 2>&1 || true
  fi
  if command -v pip3 >/dev/null 2>&1; then
    pip3 cache purge >/dev/null 2>&1 || true
  fi
}

run_mode() {
  case "$MODE" in
    light)
      cleanup_docker_light
      ;;
    publish)
      cleanup_docker_publish
      ;;
    standard)
      cleanup_docker_standard
      cleanup_tool_caches
      ;;
    aggressive)
      cleanup_docker_aggressive
      cleanup_runner_temp
      cleanup_tool_caches
      ;;
    *)
      echo "Unknown cleanup mode: $MODE" >&2
      exit 2
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --light) MODE="light"; shift ;;
    --publish) MODE="publish"; shift ;;
    --standard) MODE="standard"; shift ;;
    --aggressive) MODE="aggressive"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

report_disk
maybe_escalate_mode "$(disk_free_pct)"
log "Running ${MODE} cleanup"
run_mode
report_disk
log "Cleanup complete (${MODE})"
