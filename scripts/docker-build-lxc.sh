#!/usr/bin/env bash
set -euo pipefail

# Docker's embedded BuildKit may be unable to apply docker-default when Docker
# runs inside an AppArmor-confined LXC. Reuse a dedicated, explicitly
# unconfined BuildKit worker with persistent cache instead.
readonly BUILDER_NAME="survng-lxc"
readonly BUILDER_CONTAINER="survng-buildkit"
readonly BUILDER_ENDPOINT="docker-container://${BUILDER_CONTAINER}"
readonly BUILDER_VOLUME="survng-buildkit-state"
readonly IMAGE_NAME="${SURVNG_IMAGE:-survng:local}"
readonly TARGET="${1:-runtime-intel}"
readonly REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$TARGET" in
  runtime|runtime-intel) ;;
  *)
    echo "usage: $0 [runtime|runtime-intel]" >&2
    exit 2
    ;;
esac

if docker container inspect "$BUILDER_CONTAINER" >/dev/null 2>&1; then
  privileged="$(docker container inspect --format '{{.HostConfig.Privileged}}' "$BUILDER_CONTAINER")"
  apparmor="$(docker container inspect --format '{{.AppArmorProfile}}' "$BUILDER_CONTAINER")"
  if [[ "$privileged" != "true" || "$apparmor" != "unconfined" ]]; then
    echo "$BUILDER_CONTAINER exists without the required privileged/AppArmor-unconfined settings" >&2
    exit 1
  fi
  docker start "$BUILDER_CONTAINER" >/dev/null
else
  docker run --detach \
    --name "$BUILDER_CONTAINER" \
    --restart unless-stopped \
    --privileged \
    --security-opt apparmor=unconfined \
    --volume "$BUILDER_VOLUME:/var/lib/buildkit" \
    moby/buildkit:buildx-stable-1 \
    --oci-worker-gc-keepstorage 4096,8192,24576 >/dev/null
fi

if ! docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
  docker buildx create \
    --name "$BUILDER_NAME" \
    --driver remote \
    "$BUILDER_ENDPOINT" >/dev/null
fi

for attempt in {1..30}; do
  if docker buildx inspect "$BUILDER_NAME" --bootstrap >/dev/null 2>&1; then
    break
  fi
  if [[ "$attempt" -eq 30 ]]; then
    echo "BuildKit did not become ready" >&2
    exit 1
  fi
  sleep 1
done

GIT_SHA="${SURVNG_GIT_SHA:-$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)}"

docker buildx inspect "$BUILDER_NAME" --bootstrap
docker buildx build \
  --builder "$BUILDER_NAME" \
  --target "$TARGET" \
  --build-arg "SURVNG_GIT_SHA=${GIT_SHA}" \
  --load \
  --tag "$IMAGE_NAME" \
  "$REPO_DIR"

echo "Built $IMAGE_NAME ($TARGET) with persistent builder $BUILDER_NAME."
