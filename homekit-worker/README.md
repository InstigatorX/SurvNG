# HKSV3 protocol acceptance lab

This optional worker now has two entry points: the isolated synthetic protocol
lab (`lab.js`) and SurvNG's experimental per-camera runtime (`camera.js`).
Real-camera HomeKit is disabled by default. The QSV candidate is work in
progress with a failing opt-in hardware regression; software fallback requires
per-camera opt-in. The complete hardware acceptance gate remains unverified.
See [implementation status](../docs/homekit-implementation.md) for architecture,
validation and remaining qualification.

The lab implements separately controlled local SRTP, remote WebRTC/SFrame,
snapshots, and HDS recording with a bounded synthetic HEVC prebuffer. It does not
advertise direct CMAF upload. A temporary, user-authorized hardware experiment
advertises a Speaker service (`twoWayAudio: true`) to test Home's live-view
selection. The talk button may appear, but talkback is unimplemented. Revert this
advertisement after the experiment if it does not restore stream setup.
The camera name is **SurvNG HKSV3 Lab**.

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
Git. HAP is pinned to secure-video `d0359d529de18ef318b09a74b71380a879023ec4`
(HAP-NodeJS 2.3.0 merged on 2026-11-03). An upstream update requires deliberate
review and a lockfile update. Third-party notices are retained in `licenses/` and
`src/vendor/`.

## Enable real cameras in SurvNG

After building the pinned dependencies, open **Admin → HomeKit** under
Integrations. Set the server's LAN IPv4 address and the service account's Node.js
24 executable path. Enable the HomeKit service, select cameras, opt into software
HEVC encoding, and save. The worker reports missing build/runtime/encoder
requirements in the same panel; startup never downloads packages.

Choose the main or live camera source and enable camera audio only when that
source has audio. Use **Pair with Apple Home** to reveal the setup code and QR.
In Apple Home, choose Add Accessory, scan the QR, and allow the experimental
accessory when prompted. Pair every camera separately. Configure recording and
privacy in Home. Object-triggered motion is opt-in in SurvNG and requires the
camera's Detection switch; keep Home activity zones off for these unlocated
motion events.

Each camera's identity persists in the private database/storage `homekit`
directory. Disable/enable preserves pairing. **Reset pairing** removes only the
selected camera's HomeKit identity; remove its old Home tile and pair again.
Backing up this directory preserves pairing across moves/reinstalls. The paired
synthetic lab's separate state is never reused or modified.

The runtime is supervised by SurvNG and must not be launched manually alongside
it. Its configuration and media admission use an inherited versioned pipe;
private Unix sockets carry status/control and shared encoded media. Same-tier
viewers and recording reuse a demand-driven HEVC encoder. Snapshots have separate
JPEG admission. Resource exhaustion fails visibly rather than bypassing limits.
HomeKit Off closes its pipelines and clears its ring without changing SurvNG's
own recording or detection settings. Talkback remains unavailable.

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

The `services` section separates cached operating-mode and streaming flags by
service, including HAP status codes and linked service IDs. It reads only an
allowlist of flags and the capabilities version; transport values remain excluded.
Microphone/speaker mute and volume are included. `receiverFeedback` summarizes
authenticated RTCP receiver reports separately for local/remote video and audio:
report count, maximum reported packet loss, fractional loss, jitter in
milliseconds, and keyframe requests. Local packets that cannot be parsed or
authenticated increment an invalid-packet counter; raw feedback is never kept.

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

The September 2026 attempt paired and served HAP reads, then showed “No Response”
with no snapshot or stream setup. That Apple TV was tvOS 18.6. Repeat the
checklist on an iOS/tvOS 27 hub. Record the device model and exact OS build for
both the iPhone and Apple TV 4K. Use a Home with an iCloud+ plan that supports
recording. Keep the existing state directory so the paired accessory is reused.

Hardware progress (2026-11-13–14; gate still unverified): the user reports iOS 27.2
and tvOS 27.2.1 on the active Family Room Apple TV. Exact OS builds are unknown.

| Check | Evidence and result |
| --- | --- |
| Snapshot/setup requests | Restored after the user-authorized Speaker advertisement; JPEG requests complete and local SRTP/remote WebRTC start. Talkback is unimplemented. |
| Local video/audio | **Pass (2026-11-14 user retest):** smooth video and audible tone after filtering live SEI metadata. |
| Cellular video/audio | **Pass (2026-11-14 user retest):** smooth video and tone after omitting the on-wire RID extension; 1080p tier and signaling metadata retained. |
| Direct Apple TV live playback | **Pass (2026-11-14):** user confirmed it was playing before the privacy check. |
| HDS recording | Earlier session's handoff reports a playable 1080p30/four-second-fragment clip, ending on the library's end-of-stream ACK timeout. Pre-roll unverified. |
| Pairing across restart | Preserved with the same state directory; hub reselects recording configuration. |
| HomeKit Off/privacy | **Pass (2026-11-14):** active TV viewer stopped; runtime shows both streaming paths disabled, zero viewers/children, recording inactive/not ready, and zero buffer bytes/seconds. |
| Audio-off recording | Motion triggered 2026-11-14 11:13:32 Eastern after readiness; producer maps video without audio, and user confirmed the Home clip plays silently. Exported clip track inspection remains pending. |
| Pre-roll | Exported clip timing inspection pending. |
| 24-hour soak | Read-only health capture started 2026-11-14 11:18:05 Eastern, sampling once a minute. Results need review after the window; not yet passed. |

Powering off the older Alex Room Apple TV did not restore LIVE. Cached mode flags
show streaming/snapshots enabled and microphone unmuted at volume 100.

Producer probes found missing in-band HEVC parameter sets, dropped loopback RTP
fragments, 48 kHz capture, and CRA frames in fixtures intended to have closed
GOPs. Corrections are `hevc_mp4toannexb`, a 1 MiB feed receive buffer, 16 kHz
capture, and `open-gop=0`. The supplied HKSV Compatibility Guide (June 2026
preview), section 2/page 6, distinguishes capture at 16/24 kHz from the advertised
48 kHz Opus transmission clock. The four-second GOP interval is retained.

Regression coverage verifies 300 decodable frames across a fixture loop, RTP
sequence continuity, parameter sets, no CRA frames, Opus bandwidth, and 960-tick
steps for 20 ms audio packets. Real-media tests pass. An independent FFmpeg SRTP
receiver decoded 299 frames and about ten seconds of non-silent audio without
authentication or decode errors using the accessory's returned parameters. Those
results do not establish negotiation compatibility with Apple's controller.

Home sends repeated video keyframe requests with little cumulative packet loss.
Feedback counters normalize signed 24-bit loss and match both media SSRCs across
compound reports on either socket; earlier audio-report absence was inconclusive
because the counters originally classified reports by socket. Safe START
negotiation summaries and sent-packet counters are available under `localMedia`.
The direct Apple TV comparison reproduced the iPhone failure. Comparing Scrypted
at `fb679504ce2d8bc150bb504a7098826644627100` identified a negotiation difference:
its multi-tier RTP delegate uses the controller-provided SRTP parameters for both
directions, while this lab generated replacements for outbound media. The lab now
retains the controller parameters; regression coverage decrypts video and audio
using the original controller offer. Prepared accessory SSRCs are still retained,
which agrees with Scrypted even when START supplies different SSRCs. Hardware
validation of the SRTP-parameter correction failed: playback still buffers with
no sound. See [upstream investigation notes](UPSTREAM-NOTES.md) for the Scrypted
PR review, applicability, and remaining sender/signaling differences.
The subsequent Apple Console capture reports an oversized 2,048-byte parameter
set and decoder-format creation failures. The fixture's x265 SEI metadata is
2,403 bytes although VPS/SPS/PPS are small. Live output now filters SEI NAL units
39/40 while preserving decoder parameter sets and picture data; the filtered
real-media suite passes. The user then confirmed smooth local LIVE video and
tone. Cellular playback and the remaining recording/privacy/soak checks still
require acceptance.

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
