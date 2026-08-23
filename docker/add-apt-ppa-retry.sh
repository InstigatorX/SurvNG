#!/bin/sh
# Retry add-apt-repository when Launchpad returns transient GPG/API errors.
set -eu

if [ "$#" -lt 1 ]; then
  echo "usage: add-apt-ppa-retry.sh ppa:owner/name" >&2
  exit 2
fi

PPA="$1"
attempt=1
max_attempts=6
delay=15

while [ "$attempt" -le "$max_attempts" ]; do
  if add-apt-repository -y "$PPA"; then
    echo "Added ${PPA} on attempt ${attempt}"
    exit 0
  fi
  echo "add-apt-repository ${PPA} failed on attempt ${attempt}/${max_attempts}" >&2
  if [ "$attempt" -eq "$max_attempts" ]; then
    exit 1
  fi
  sleep "$delay"
  attempt=$((attempt + 1))
  delay=$((delay * 2))
  if [ "$delay" -gt 120 ]; then
    delay=120
  fi
done
