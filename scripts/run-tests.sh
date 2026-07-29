#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

timeout_seconds="${SURVNG_TEST_TIMEOUT_SECONDS:-180}"
if [[ ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "SURVNG_TEST_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 2
fi

# GNU timeout owns and reaps the pytest process group. TERM permits normal
# fixture cleanup; KILL guarantees that a wedged native worker cannot remain
# resident after the bounded grace period.
exec timeout --signal=TERM --kill-after=10s "${timeout_seconds}s" \
    env PYTHONPATH="${PYTHONPATH:-.}" .venv/bin/pytest "$@"
