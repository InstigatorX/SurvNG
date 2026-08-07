#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

timeout_seconds="${SURVNG_TEST_TIMEOUT_SECONDS:-180}"
if [[ ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "SURVNG_TEST_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 2
fi

# Serialize campaigns so one wrapper can distinguish its own descendants from
# a pre-existing test run. The advisory lock is released automatically if the
# wrapper is interrupted or killed.
exec 9>"${SURVNG_TEST_LOCK_FILE:-/tmp/survng-pytest.lock}"
flock 9

pytest_processes() {
    local process pid cwd command
    for process in /proc/[0-9]*; do
        pid="${process##*/}"
        cwd="$(readlink "$process/cwd" 2>/dev/null || true)"
        [[ "$cwd" == "$repo_dir" ]] || continue
        command="$(tr '\0' ' ' < "$process/cmdline" 2>/dev/null || true)"
        [[ "$command" == *".venv/bin/pytest"* ]] || continue
        printf '%s\n' "$pid"
    done
}

declare -A baseline=()
while read -r pid; do
    [[ -n "$pid" ]] && baseline["$pid"]=1
done < <(pytest_processes)

cleanup_found=0
cleanup_test_processes() {
    local pid pgid self_pgid
    local -a survivors=()
    declare -A process_groups=()
    self_pgid="$(ps -o pgid= -p "$$" | tr -d ' ')"
    while read -r pid; do
        [[ -n "$pid" && -z "${baseline[$pid]:-}" ]] || continue
        survivors+=("$pid")
        pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
        if [[ -n "$pgid" && "$pgid" != "$self_pgid" ]]; then
            process_groups["$pgid"]=1
        fi
    done < <(pytest_processes)
    ((${#survivors[@]})) || return 0

    cleanup_found=1
    echo "cleaning test processes that escaped the bounded runner: ${survivors[*]}" >&2
    for pgid in "${!process_groups[@]}"; do
        kill -TERM -- "-$pgid" 2>/dev/null || true
    done
    kill -TERM "${survivors[@]}" 2>/dev/null || true
    sleep 1
    for pgid in "${!process_groups[@]}"; do
        kill -KILL -- "-$pgid" 2>/dev/null || true
    done
    kill -KILL "${survivors[@]}" 2>/dev/null || true
}

# Also clean up if the wrapper itself is interrupted while waiting for pytest.
trap cleanup_test_processes EXIT

# GNU timeout owns the pytest process group. TERM permits normal fixture
# cleanup; KILL guarantees that a wedged native worker cannot remain resident
# after the bounded grace period.
set +e
timeout --signal=TERM --kill-after=10s "${timeout_seconds}s" \
    env PYTHONPATH="${PYTHONPATH:-.}" .venv/bin/pytest "$@"
status=$?
set -e

cleanup_test_processes
trap - EXIT
[[ $cleanup_found -eq 0 || $status -ne 0 ]] || status=1

exit "$status"
