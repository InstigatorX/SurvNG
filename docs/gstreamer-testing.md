# Isolated GStreamer testing

Work from the `gstreamer` checkout. Do not restart the v1.2 service, reuse its
configuration/database/media directory, or combine this file with `compose.yaml`.
The project name, loopback port, model mount and writable directories below are
separate. Camera connections and GPU compute would still share hardware if you
later run both instances on the same machine.

## 1. Native plumbing smoke test (no cameras or GPU)

```bash
docker compose -f compose.gstreamer-test.yaml run --rm smoke
```

This uses the digest-pinned Intel 2026.1.0 image, a read-only source mount, a
temporary synthetic OpenVINO model, no network, 2 CPUs and 2 GiB RAM. Expect:

- 5 FPS, 320-wide grayscale qualification, with approximately half as many
  inference snapshots at 2.5 FPS.
- Positive snapshots labeled `car`, then a separate all-empty run.
- 640-wide BGR main frames with no live detection snapshots.
- Two simultaneous live streams and one main stream in a shared supervisor,
  each receiving its own correctly framed messages.

On the current nested-container development host, Docker's default AppArmor
profile cannot be applied. The native test was run successfully with this
**per-container** exception; it is not needed/recommended on a normal host:

```bash
docker run --rm --network none --cpus 2 --memory 2g \
  --cap-drop ALL --security-opt apparmor=unconfined --security-opt no-new-privileges \
  --entrypoint /bin/bash -v "$PWD:/work:ro" -w /work \
  intel/dlstreamer:2026.1.0-ubuntu24@sha256:355435d2bdb986fe1d51f443d366da3ac1eb0c7175aa03ce2b31e4485f36b3bd \
  -lc 'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/work:$PYTHONPATH python3 scripts/gstreamer-smoke.py'
```

Do not disable host AppArmor or alter production Docker configuration to make
these tests pass. The complete application image build is **not verified on
this host** because it fails at container initialization for that reason.

## 2. Regression suite

Use Python 3.12 and Node 22 in a development environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt pytest anyio
npm --prefix frontend ci
npm --prefix frontend run build
scripts/run-tests.sh -q
node frontend/tests/run-tests.mjs unit
```

Tests create temporary configuration/storage and their own observability
socket. Native GStreamer coverage is the separate smoke test, not a skipped
host-GI test. CI runs both suites and the native smoke job.

## 3. Application test on an Intel GPU host

This requires a working `/dev/dri`, supported drivers, and a normal Docker build
environment. Prepare **new** directories; the commands below assume the test
runtime UID/GID are 1000. Adjust ownership and `GSTREAMER_TEST_UID/GID` together
if needed. They do not copy or alter production configuration.

```bash
sudo install -d -m 700 -o 1000 -g 1000 \
  docker-data/gstreamer-test/config docker-data/gstreamer-test/data \
  docker-data/gstreamer-test/media docker-data/gstreamer-test/models
stat -c '%n %g' /dev/dri/renderD* /dev/dri/card*
```

Set `GSTREAMER_TEST_RENDER_GID` and `GSTREAMER_TEST_VIDEO_GID` to the appropriate
device group IDs, then:

```bash
docker compose -f compose.gstreamer-test.yaml --profile app build app
docker compose -f compose.gstreamer-test.yaml --profile app up -d app
docker compose -f compose.gstreamer-test.yaml logs --tail 100 app
```

Open `http://127.0.0.1:18088/survng/` locally (or forward that loopback port over
SSH). The generated default config has no cameras and detection/MQTT disabled.
Use only `docker-data/gstreamer-test/config/config.json` for this instance.
Its media, database and cache paths must remain `/media` and `/data/...`.
Keep MQTT, outgoing AI calls and ONVIF disabled for the first replay. Do not
import production credentials/runtime state wholesale. The test profile does
not expose go2rtc ports or promise WebRTC connectivity; UI/media transport is a
separate check from the detector pipeline.

Put a **copy** of the intended model's XML/BIN, labels and compatible model-proc
or model metadata under the dedicated `models` directory, readable by the test
UID. Configure its `/models/...` path. First verify the existing model's live
and recorded output parity; the synthetic smoke model is not a detector to
deploy. Configure one test camera before adding more.

To read this container's owner-only status without contacting the host service:

```bash
docker compose -f compose.gstreamer-test.yaml exec app python -c \
  'import json; from survng.app.local_observability import request_runtime_status; print(json.dumps(request_runtime_status("/data/observability.sock"), indent=2))'
```

This uses the same client as `survngctl`, which is the preferred host-local
command for a native installation. Do not use the host's default socket to
assess the test container.

### VA-memory and cold-start checks

The CPU smoke test does not validate Intel VA negotiation. On real hardware,
check pipeline status for `decoded_memory=memory:VAMemory`,
`detection_memory=memory:VAMemory`, `qualifier_memory=memory:SystemMemory`,
and `preprocess_backend=va` on a VA-enabled live inference stream. Detection
keeps VA surfaces; the rate-limited EMA branch uses `vapostproc` to resize and
download square-pixel NV12, then converts the small host frame to GRAY8.
The rate-limited JPEG branch also has an explicit VA download boundary.
Do not substitute direct VA-to-GRAY8 conversion based on advertised caps alone:
the test host's driver negotiates it but produces no frames.

Require actual grayscale frames, JPEG previews and timestamp/session-matched
inference snapshots, not just successful negotiation or EOS. Main evidence
must remain aspect-correct BGR with no live inference snapshots. Repeat with
two live streams sharing the model, then close/reopen a stream and verify a
new session with no old snapshots. Cold model compilation took about 16 seconds
on the test host; inference-enabled capture uses at least 30 seconds for
startup in both parent and child. Normal read timeouts remain unchanged.

## 4. Scene and regression acceptance

Replay synchronized main/live clips through a disposable RTSP source, or use a
dedicated test camera. Include quiet background before the missed Gate vehicle
and time after it leaves. For recorded clips, original event wall time is not
the replay's time: compare positions in the source clip and resulting evidence,
not today's clock time. An RTSP replay alone does not recreate ONVIF events.

Run these checks with unchanged model/thresholds before trying a faster model:

| Case | Required observation |
| --- | --- |
| Vehicle enters/exits Gate | Correct class, event admitted, useful main-frame cover, boxes overlap the moving subject; no stale positive after it leaves. |
| Empty scene / headlights / foliage | No new false admissions relative to the baseline; empty inference is visible in matching counters. |
| Different main/live crop | Zone admission agrees after trusted calibration; untrusted geometry waits for recorded evidence. |
| Stop/restart or PTS reset | Old boxes never enter the new session; capture recovers and history stays bounded. |
| Skipped/duplicate inference | No fabricated confirmations or rapid track expiry merely because a lookup lacks a result. |
| Two similar vehicles crossing | Stable IDs and cover ownership; compare `survng_hybrid` with the retained `bytetrack` alternative. |
| Model/device/cadence change | New native graph actually takes effect; no surviving retired supervisor. Expect a brief manager restart. |
| Recording and notifications | Event IDs, recording references, covers and MQTT counts remain correct; test MQTT only against an isolated broker/topic. |

Record baseline and candidate: labeled vehicles detected/missed, false events,
ID switches, first-event latency, EMA and detection FPS, CPU/GPU utilization,
RSS, frame age and recorder gaps. Require no regression on the labeled Gate
scene and no stale/cross-session admissions. A 30-minute one-camera soak comes
before a multi-camera test. Choose performance targets from those measurements;
no percentage improvement is claimed in advance.

Stop only this test project:

```bash
docker compose -f compose.gstreamer-test.yaml --profile app down
```

Its bind-mounted test data is retained. No production container, service or data
is stopped or removed by this command.
