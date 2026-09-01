#!/usr/bin/env bash
set -euo pipefail

# Build one Dockerfile target, reuse the moving GHCR tip as --cache-from, push
# tags, and keep the tip image locally for the next sequential matrix job.
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

echo "Pulling cache source ${primary}"
docker pull "${primary}" || true

echo "Building ${target} as ${primary}"
docker build \
  --file "${dockerfile}" \
  --target "${target}" \
  --cache-from "${primary}" \
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
