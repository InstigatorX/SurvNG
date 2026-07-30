#!/usr/bin/env bash
set -euo pipefail

# Docker's embedded BuildKit may be unable to apply docker-default when Docker
# runs inside an AppArmor-confined LXC. Use a short-lived, explicitly
# unconfined BuildKit worker instead; remove it regardless of build outcome.
readonly BUILDER_NAME="survng-lxc-builder"
readonly BUILDER_CONTAINER="survng-buildkit"
readonly BUILDER_ENDPOINT="tcp://127.0.0.1:12345"
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

cleanup() {
  docker buildx rm "$BUILDER_NAME" >/dev/null 2>&1 || true
  docker stop "$BUILDER_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

if docker container inspect "$BUILDER_CONTAINER" >/dev/null 2>&1; then
  echo "refusing to replace existing container: $BUILDER_CONTAINER" >&2
  exit 1
fi
if docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
  echo "refusing to replace existing builder: $BUILDER_NAME" >&2
  exit 1
fi

docker run --detach --rm \
  --name "$BUILDER_CONTAINER" \
  --privileged \
  --security-opt apparmor=unconfined \
  --network host \
  moby/buildkit:buildx-stable-1 \
  --addr "$BUILDER_ENDPOINT" >/dev/null

docker buildx create \
  --name "$BUILDER_NAME" \
  --driver remote \
  "$BUILDER_ENDPOINT" >/dev/null

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

docker buildx inspect "$BUILDER_NAME" --bootstrap
docker buildx build \
  --builder "$BUILDER_NAME" \
  --target "$TARGET" \
  --load \
  --tag "$IMAGE_NAME" \
  "$REPO_DIR"

echo "Built $IMAGE_NAME ($TARGET); temporary builder removed."
