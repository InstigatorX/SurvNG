#!/bin/sh
set -eu

umask 077

config_path="${SURVNG_CONFIG_PATH:-/config/config.json}"
config_dir=$(dirname "$config_path")
runtime_uid="${SURVNG_UID:-1000}"
runtime_gid="${SURVNG_GID:-1000}"

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

if [ "$(id -u)" = "0" ] && [ "$runtime_uid:$runtime_gid" != "0:0" ]; then
    chown "$runtime_uid:$runtime_gid" "$config_dir" /data "$config_path"
    run_as="gosu survng"
else
    run_as=""
fi

# Keep the command as a child so Docker TERM can first release scarce ONVIF
# PullPoint subscriptions, matching the native systemd shutdown ordering.
shutdown_child() {
    if [ -n "${child_pid:-}" ] && kill -0 "$child_pid" 2>/dev/null; then
        kill -USR1 "$child_pid" 2>/dev/null || true
        sleep 1
        kill -TERM "$child_pid" 2>/dev/null || true
    fi
}
trap shutdown_child TERM INT

if [ -n "$run_as" ]; then
    gosu survng "$@" &
else
    "$@" &
fi
child_pid=$!

status=0
while kill -0 "$child_pid" 2>/dev/null; do
    if wait "$child_pid"; then
        status=0
    else
        status=$?
    fi
done
exit "$status"
