#!/usr/bin/env bash
set -euo pipefail

# Build one Dockerfile target, push tags, and keep the moving tip locally so
# the next sequential matrix job can reuse layers. Do not pass --cache-from:
# the legacy builder crashes on this multi-stage Dockerfile when restoring a
# pulled GHCR image ("failed to read diff archive: InvalidArgument").
# Required environment: GITHUB_REPOSITORY, GIT_SHA, REF_TYPE, REF_NAME, TARGET,
# DOCKERFILE. Optional: SUFFIX.

image="ghcr.io/$(printf '%s' "${GITHUB_REPOSITORY:?}" | tr '[:upper:]' '[:lower:]')"
short_sha="$(printf '%.7s' "${GIT_SHA:?}")"
suffix="${SUFFIX:-}"
ref_type="${REF_TYPE:?}"
ref_name="${REF_NAME:?}"
target="${TARGET:?}"
dockerfile="${DOCKERFILE:?}"

tags=()
tags+=("${image}:${ref_name}${suffix}")
if [ "${ref_type}" != "tag" ]; then
  tags+=("${image}:sha-${short_sha}${suffix}")
fi
primary="${tags[0]}"

echo "Building ${target} as ${primary}"
docker build \
  --file "${dockerfile}" \
  --target "${target}" \
  --build-arg "SURVNG_GIT_SHA=${GIT_SHA}" \
  --tag "${primary}" \
  .

for tag in "${tags[@]:1}"; do
  docker tag "${primary}" "${tag}"
done
for tag in "${tags[@]}"; do
  docker push "${tag}"
  echo "Pushed ${tag}"
done

# Keep the moving tip for local layer cache. Drop only the immutable sha pin.
if [ "${ref_type}" != "tag" ]; then
  docker rmi "${image}:sha-${short_sha}${suffix}" 2>/dev/null || true
fi
