# HKSV3 protocol acceptance lab

This is **milestone 1**, a synthetic-camera protocol worker for the native HKSV3
implementation. It is not connected to SurvNG cameras, configuration, motion,
recording, or the admin UI. Passing its automated tests does not prove that Apple
Home can record from it. Production integration is gated on the real-device
acceptance checklist below.

The lab implements separately controlled local SRTP, remote WebRTC/SFrame,
snapshots, and HDS recording with a bounded synthetic HEVC prebuffer. It does not
advertise direct CMAF upload or talkback. The camera name is **SurvNG HKSV3 Lab**.

## Build

Requires Linux, Python 3.12+, Node.js **24**, npm, FFmpeg with `libx265`, `libopus`,
AAC and MJPEG encoders, and `ffprobe`. The existing SurvNG service need not stop.

From this directory:

```sh
python3 scripts/bootstrap.py
npm test
HKSV_MEDIA_TESTS=1 npm test
```

If the host has another Node version, use an isolated Node 24 runtime:

```sh
npm exec --yes --package=node@24 -- python3 scripts/bootstrap.py
npm exec --yes --package=node@24 -- npm test
HKSV_MEDIA_TESTS=1 npm exec --yes --package=node@24 -- npm test
```

Bootstrap verifies the upstream archive SHA-256, installs upstream build tools
from its lockfile, compiles HAP, and installs worker dependencies from the worker
lockfile. Downloaded source, packages, build output, and media are excluded from
Git. HAP's revision is pinned; an upstream update requires deliberate review and
a lockfile update. Third-party notices are retained in `licenses/` and
`src/vendor/`.

## Run the synthetic camera

Choose an absolute directory owned by the account running the lab, mode 0700.
Keep it for pairing persistence. Choose the host IPv4 address on the same LAN as
the Apple TV and iPhone. Use a free HAP port (default 51826).

```sh
node dist/src/lab.js \
  --state-dir /absolute/private/homekit-lab \
  --bind YOUR_LAN_IPV4 \
  --allow-synthetic-encoding
```

With an isolated Node runtime, prefix the command with
`npm exec --yes --package=node@24 --`.

The explicit encoding flag permits one-time generation of three short CPU-encoded
test patterns. Subsequent runs reuse those fixtures. No camera credentials or
stream URLs are accepted. Streaming copies HEVC from these fixtures and encodes
Opus audio; HDS copies HEVC/AAC. This lab does not implement production hardware
encoder qualification or shared producer admission.

The lab uses one accessory, up to five local and six remote sessions, and a
single HDS recording stream. Abandoned viewer sessions terminate after ten
minutes. Recording sessions terminate after motion plus four seconds or after
five minutes. Media buffers retain twelve seconds within 64 MiB. Consumers that
fall behind fail explicitly.

Only an owner-only Unix socket exposes controls:

```sh
python3 scripts/control.py --state-dir /absolute/private/homekit-lab status
python3 scripts/control.py --state-dir /absolute/private/homekit-lab pairing
python3 scripts/control.py --state-dir /absolute/private/homekit-lab motion --seconds 10
```

`pairing` explicitly displays the secret setup code and setup URI. Enter the code
in Apple's Add Accessory flow; do not paste it into issues or logs. Status excludes
pairing material, crypto keys, SDP, and stream contents. No unauthenticated HTTP
control API is added.

For a paired accessory with no picture, status includes snapshot requests,
completions and failures, HAP request totals, and per-characteristic read, value
write, and subscription counts. These counters reset on worker restart and do
not retain request values. Check `localAllowed` and `remoteAllowed` before media
diagnostics. A reachable paired accessory with no snapshot or stream setup
requests has not yet exercised the media producers; verify the exact iPhone and
active home hub OS versions and repeat live view before attributing the failure
to an encoder or VLAN routing.

HomeKit privacy stops synthetic delivery and buffering. Synthetic motion has no
spatial evidence, so requests are rejected when Apple motion zones are active;
disable zones for the initial protocol test. Production spatial qualification is
a later integration milestone, not simulated here.

Stop with Ctrl-C or SIGTERM. The lab reaps its FFmpeg children and closes sessions.
Pairing records remain. An exclusive `lab.lock` prevents overlapping instances.
After an unclean crash, verify the recorded process is gone before removing that
lock and the stale `control.sock`. Never remove pairing data to repair a runtime
lock.

## Required hardware acceptance

Record the device model and exact OS build for both the iPhone and Apple TV 4K.
Both must support the HKSV3 path; the target is iOS/tvOS 27. Use a Home with an
iCloud+ plan that supports recording.

1. Pair the lab camera and confirm its snapshot appears.
2. Open local live view and confirm moving video and the test tone.
3. Disable Wi-Fi on the iPhone and verify cellular live view and audio.
4. Set the camera to Stream & Allow Recording. Wait until `status` reports the
   recording buffer ready, then trigger ten seconds of synthetic motion.
5. Confirm a playable event appears in Home with footage before the trigger.
6. Disable recording audio and confirm subsequent clips have no audio track.
7. Set HomeKit Off: viewers must stop and buffer readiness must become false.
8. Restart the lab using the same state directory; pairing must survive.
9. Record pass/fail and sanitized status for each step. Do not record setup codes,
   SDP, SRTP keys, SFrame keys, or full HAP diagnostic dumps.

Do not move production cameras onto this worker until these checks pass. The
remaining native-integration work is tracked in `docs/homekit-implementation.md`
at the repository root.
