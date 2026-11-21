#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

usage() {
    cat <<'USAGE'
Usage: scripts/verify-architecture.sh [static|lifecycle|recording|media]

  static     Check Codex guidance, CI test assignment, and dependency lock (default).
  lifecycle  Check manager reload, worker startup, and motion admission.
  recording  Check recorder lifecycle, retention, and recording route leases.
  media      Exercise fragmented MP4 remuxing with local FFmpeg and ffprobe.

Runtime suites require the CI Python dependencies in .venv.
For lifecycle, build the frontend with npm --prefix frontend ci and
npm --prefix frontend run build.
USAGE
}

case "${1:-static}" in
    -h|--help)
        usage
        exit 0
        ;;
esac
if (( $# > 1 )); then
    usage >&2
    exit 2
fi

case "${1:-static}" in
    static)
        python3 scripts/check-codex-config.py
        python3 scripts/check-ci-test-coverage.py
        python3 scripts/check-requirements-lock.py
        exit 0
        ;;
    lifecycle)
        if [[ ! -d survng/static ]]; then
            echo 'Build the frontend with npm --prefix frontend ci && npm --prefix frontend run build.' >&2
            exit 2
        fi
        tests=(
            tests/test_app_lifespan.py
            tests/test_config_reload.py
            tests/test_manager_reload.py
            tests/test_worker_topology.py
            tests/test_inference_lifecycle.py
            tests/test_camera_fleet.py
            tests/test_motion_runtime.py
        )
        ;;
    recording)
        tests=(
            tests/test_recording_lifecycle.py
            tests/test_recorder.py
            tests/test_recording_routes.py
        )
        ;;
    media)
        for program in ffmpeg ffprobe; do
            if ! command -v "$program" >/dev/null; then
                printf 'The media suite requires %s on PATH.\n' "$program" >&2
                exit 2
            fi
        done
        tests=(tests/test_recording_fmp4_integration.py)
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

if [[ ! -x .venv/bin/pytest ]]; then
    echo 'Install the CI test dependencies in .venv before running runtime suites.' >&2
    exit 2
fi

exec scripts/run-tests.sh "${tests[@]}"
