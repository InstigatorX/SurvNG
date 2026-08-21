#!/bin/sh
set -eu

umask 077

config_path="${SURVNG_CONFIG_PATH:-/config/config.json}"
config_dir=$(dirname "$config_path")
go2rtc_config="${SURVNG_GO2RTC_CONFIG:-/config/go2rtc.yaml}"
go2rtc_enabled="${SURVNG_GO2RTC:-1}"
runtime_uid="${SURVNG_UID:-1000}"
runtime_gid="${SURVNG_GID:-1000}"
go2rtc_pid=""
child_pid=""

case "$runtime_uid:$runtime_gid" in
    *[!0-9:]*|:*|*:)
        echo "SURVNG_UID and SURVNG_GID must be numeric" >&2
        exit 2
        ;;
esac
if { [ "$runtime_uid" = "0" ] && [ "$runtime_gid" != "0" ]; } \
    || { [ "$runtime_uid" != "0" ] && [ "$runtime_gid" = "0" ]; }
then
    echo "SURVNG_UID and SURVNG_GID must either both be zero or both be non-zero" >&2
    exit 2
fi

mkdir -p "$config_dir" /data

if [ "$(id -u)" = "0" ]; then
    if [ "$runtime_gid" != "0" ] && [ "$(id -g survng)" != "$runtime_gid" ]; then
        primary_group=$(getent group "$runtime_gid" | cut -d: -f1 || true)
        if [ -z "$primary_group" ]; then
            groupmod --gid "$runtime_gid" survng
            primary_group=survng
        fi
        usermod --gid "$primary_group" survng
    fi
    if [ "$runtime_uid" != "0" ] && [ "$(id -u survng)" != "$runtime_uid" ]; then
        usermod --uid "$runtime_uid" survng
    fi
    for device_group_spec in \
        "video:${SURVNG_VIDEO_GID:-}" \
        "render:${SURVNG_RENDER_GID:-}"
    do
        device_group=${device_group_spec%%:*}
        device_gid=${device_group_spec#*:}
        if [ -n "$device_gid" ]; then
            case "$device_gid" in
                *[!0-9]*)
                    echo "SURVNG device group IDs must be numeric" >&2
                    exit 2
                    ;;
            esac
            group_name=$(getent group "$device_gid" | cut -d: -f1 || true)
            if [ -z "$group_name" ]; then
                group_name="survng-$device_group"
                groupadd --gid "$device_gid" "$group_name"
            fi
            usermod -aG "$group_name" survng
        fi
    done
fi

if [ ! -e "$config_path" ]; then
    install -m 600 /usr/share/survng/config.docker.example.json "$config_path"
    echo "Created private SurvNG configuration at $config_path" >&2
fi

chmod 600 "$config_path"

if [ "$go2rtc_enabled" != "0" ] && [ ! -e "$go2rtc_config" ]; then
    install -m 600 /usr/share/survng/go2rtc.example.yaml "$go2rtc_config"
    echo "Created go2rtc configuration at $go2rtc_config" >&2
fi

if [ "$(id -u)" = "0" ] && [ "$runtime_uid:$runtime_gid" != "0:0" ]; then
    chown "$runtime_uid:$runtime_gid" "$config_dir" /data "$config_path"
    if [ -e "$go2rtc_config" ]; then
        chown "$runtime_uid:$runtime_gid" "$go2rtc_config"
        chmod 600 "$go2rtc_config"
    fi
    run_as="gosu survng"
else
    run_as=""
    if [ -e "$go2rtc_config" ]; then
        chmod 600 "$go2rtc_config"
    fi
fi

stop_pid() {
    target_pid=$1
    signal=$2
    if [ -n "$target_pid" ] && kill -0 "$target_pid" 2>/dev/null; then
        kill "-$signal" "$target_pid" 2>/dev/null || true
    fi
}

# Keep SurvNG as a child so Docker TERM can first release scarce ONVIF
# PullPoint subscriptions, matching the native systemd shutdown ordering.
# Stop SurvNG before go2rtc so live clients drain cleanly.
shutdown_children() {
    if [ -n "${child_pid:-}" ] && kill -0 "$child_pid" 2>/dev/null; then
        stop_pid "$child_pid" USR1
        sleep 1
        stop_pid "$child_pid" TERM
    fi
    stop_pid "${go2rtc_pid:-}" TERM
}
trap shutdown_children TERM INT

start_background() {
    # Redirect child stdio to the entrypoint's streams so PID capture via $!
    # is not polluted when this helper runs inside command substitution.
    if [ -n "$run_as" ]; then
        gosu survng "$@" >&2 &
    else
        "$@" >&2 &
    fi
}

if [ "$go2rtc_enabled" != "0" ]; then
    if ! command -v go2rtc >/dev/null 2>&1; then
        echo "go2rtc binary is missing; set SURVNG_GO2RTC=0 to disable" >&2
        exit 1
    fi
    start_background go2rtc -config "$go2rtc_config"
    go2rtc_pid=$!
    echo "Started go2rtc (pid $go2rtc_pid) with $go2rtc_config" >&2
fi

start_background "$@"
child_pid=$!

status=0
go2rtc_failed=0
while kill -0 "$child_pid" 2>/dev/null; do
    if [ -n "$go2rtc_pid" ] && ! kill -0 "$go2rtc_pid" 2>/dev/null; then
        echo "go2rtc exited; shutting down SurvNG" >&2
        if wait "$go2rtc_pid"; then
            status=0
        else
            status=$?
        fi
        go2rtc_pid=""
        go2rtc_failed=1
        shutdown_children
        break
    fi
    sleep 1
done

if [ -n "${child_pid:-}" ]; then
    if wait "$child_pid"; then
        child_status=0
    else
        child_status=$?
    fi
    child_pid=""
    if [ "$go2rtc_failed" = "0" ]; then
        status=$child_status
    fi
fi
stop_pid "${go2rtc_pid:-}" TERM
if [ -n "${go2rtc_pid:-}" ]; then
    wait "$go2rtc_pid" 2>/dev/null || true
    go2rtc_pid=""
fi
exit "$status"
