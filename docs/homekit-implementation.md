# Native HKSV3 implementation status

Target: two cameras, Linux/systemd, iOS/tvOS 27, 1080p/720p/360p HEVC,
HDS recording. Hardware encoding is preferred; software encoding requires
per-camera opt-in. Apple Home privacy affects HomeKit only. HomeKit motion follows
SurvNG detection enablement. Direct CMAF uploads, legacy HKSV, and talkback are
out of scope.

## Protocol lab: implementation and acceptance

The isolated [synthetic worker](../homekit-worker/README.md) uses Node.js 24,
HAP-NodeJS `d0359d529de18ef318b09a74b71380a879023ec4`, and werift 0.24.4.
It has no production camera input. Pairing state and controls are private, and
status exposes allowlisted state without credentials, SDP, or transport keys.

Hardware validation on 2026-11-14 used an iPhone reporting iOS 27.2 and the
Family Room Apple TV reporting tvOS 27.2.1. Exact OS builds were not captured.
The user confirmed local and cellular LIVE with smooth video and the test tone,
direct Apple TV playback, and privacy shutdown. HomeKit Off stopped the active TV
viewer; runtime showed disabled streaming, zero media children/viewers, and an
empty, inactive recording buffer. Pairing survived repeated same-directory
restarts. A motion event at 11:13:32 Eastern with recording audio disabled produced
a playable silent Home clip; the producer mapped video without audio.

The main interoperability fixes are:

- Advertise the Speaker service required by the tested Home setup to request
  snapshots and streams. Return-audio playback/talkback remains unimplemented.
- Supply HEVC parameter sets in-band and omit live prefix/suffix SEI metadata.
  Apple's Console capture reported a 2,048-byte oversized parameter set and
  compressed-frame creation failures; the synthetic x265 SEI was 2,403 bytes.
  Local video and sound passed after filtering that metadata.
- Preserve HomeKit RID/bitrate signaling, but omit the on-wire RID RTP extension.
  Cellular video and sound passed after this separate change.
- Use closed four-second GOPs, advertise compatible recording fragment lengths
  and 15/30 fps resolutions, and allow same-size recording fixture fallback.
- Capture Opus at 16 kHz while preserving its advertised 48 kHz transmission
  clock; retain controller-negotiated SRTP parameters and prepared accessory SSRCs.
- Increase live feed receive buffers to avoid loopback keyframe burst loss, and
  add bounded receiver loss/jitter/keyframe counters plus per-service mode/audio
  flags. Compound feedback is matched by media SSRC and signed loss is normalized.

Regression validation includes decoding 300 HEVC frames across a fixture loop,
parameter-set/SEI/GOP checks, RTP continuity and audio bandwidth/clock checks,
controller-offer SRTP decryption, authenticated compound receiver feedback,
SFrame/DTLS-SRTP peers without wire RID, and privacy teardown. The real-media
suite and protocol tests passed. An independent FFmpeg SRTP receiver also decoded
299 frames and about ten seconds of non-silent audio; peer tests alone did not
establish Apple compatibility. See the [upstream investigation](../homekit-worker/UPSTREAM-NOTES.md)
for the Scrypted PR comparison and the limitations of generic peer fixtures.

Remaining lab acceptance: inspect an exported clip for absent audio tracks and
pre-roll timing, review the 24-hour health capture started at 11:18:05 Eastern on
2026-11-14, and resolve the earlier session's reported end-of-stream ACK timeout.
The complete hardware gate remains unverified. Runtime state, captures, fixtures,
setup codes, and keys are excluded from Git.

The user authorized real-camera implementation on 2026-11-14 while these checks
remain open. Build that integration disabled by default and retain its
experimental designation until real-camera qualification and acceptance pass.

## Real-camera integration (experimental, disabled by default)

QSV low-delay qualification and Gate Wi-Fi LIVE passed on 2026-11-14.
The startup probe now encodes a complete four-second GOP plus five frames,
checks every packet's decode/presentation order, verifies zero signaled reorder
depth and no future-reference entries, requires repeated closed IDRs without CRA,
and decodes the result. Real hardware tests apply this contract at all three
tiers. Explicit software fallback must pass the same checks.

The original B-picture assertion was an incorrect proxy for reordering. This
Intel runtime reports GPB ON even when P pictures are requested. Its GPB slices
are labeled B by ffprobe, but their packet PTS equals DTS, VPS/SPS reorder depth
is zero, and their references are to past pictures. These are low-delay generalized
P/B pictures, not temporally reordered B pictures. See Intel's
[driver discussion](https://github.com/intel/vpl-gpu-rt/issues/336). The regression
now rejects actual temporal reordering, future references and open GOPs, including
negative tests for each; it does not reject an otherwise valid GPB picture label.

Playback also had a separate, reproduced pacing bug. FFmpeg's demuxer probe
buffered video and then dumped roughly eight seconds of media into RTP in 2.93
seconds. Shared-source consumers now use timestamp-based real-time reads, a
one-millisecond initial burst and no accelerated catch-up. An exact zero burst
would restore FFmpeg's default half-second burst, so it is deliberately avoided.
The same 120-frame diagnostic then spanned about 7.84 seconds for eight seconds
of media, with no local sequence gaps. The automated real-input test checks
pacing over a four-second window, allowing bounded codec-startup latency.

Gate's paired worker was restarted without resetting its identity. The user
confirmed continuous Wi-Fi LIVE for 45 seconds without buffering/freezing.
A subsequent active-session snapshot showed QSV at 29.54 fps and 1.0x speed,
zero media failures, zero current video loss and 26.6 ms receiver jitter.
Before the fix, receiver maxima included 71.9% fractional loss, 3.49 seconds
of jitter and repeated keyframe requests. Maxima span startup and reconnects;
latest loss/jitter are also exposed to distinguish history from current behavior.
Full real-camera cellular/Apple TV, recording and soak acceptance remain open.

The synthetic soak monitor was stopped at the user's request. Its partial
capture was preserved; future soak validation will use real cameras.

The admin sidebar now groups HomeKit under Integrations. The HomeKit section
contains service enablement, LAN bind address and Node.js 24 path, camera
enrollment, main/live source selection, camera audio, software encoder opt-in,
object motion, runtime status, pairing code/QR, and an explicit pairing reset.
Existing Cameras setup remains the place to add camera connections.

`homekit.enabled` and each `camera.homekit.enabled` default to false. Software
fallback requires `camera.homekit.allow_software_encoding`; it also defaults to false.
The candidate runtime prefers QSV, with libx265 fallback, AAC and libopus.
Broader QSV hardware acceptance and HEVC passthrough qualification remain future work. Enabling camera
audio requires an audio track in the selected source. Otherwise the worker
supplies silence for Home's live audio transport. Home's Record Audio Off omits
the recording audio track entirely.

The first paired RTSP camera exposed an input-option mismatch: this FFmpeg
build rejects the generic `rw_timeout` option on its RTSP demuxer. RTSP/RTSPS
inputs now use the demuxer's `timeout` option; HTTP keeps `rw_timeout`. A bounded
one-frame probe against the configured camera failed before the correction and
decoded successfully afterward. Intentional child shutdown is also distinguished
from encoder failure, including FFmpeg's exit code 255 on SIGTERM, so closing a
viewer does not create a misleading source-failure warning. Regression tests cover
the option selection and shutdown classification.

The Python manager owns one optional Node worker process group per enabled
camera. Camera URLs travel over an inherited pipe, not the Node command line or
persisted worker configuration. A versioned admission protocol reserves media
capacity before each child starts, records its actual process PID, and releases
its lease after child closure. Crashed workers and their FFmpeg descendants are
terminated as a group before reservations are released. Disabled/off cameras stop
workers; restart, disable/enable and independent camera changes preserve identity.

Each demanded video tier has one shared live HEVC/AAC MPEG-TS producer. Recording
and viewers using that tier reuse it. Media travels over mode-0600 Unix sockets
inside the camera's private state directory, with no network media endpoint.
Slow consumers are disconnected at a bounded queue size without blocking other
consumers. The producer ends when its last consumer closes. Source failures and
fifteen-second stalls terminate dependent streams; recording retries on a
five-second reconciliation interval. Encoders consume the existing transcode
budget, packetizing consumers the relay budget, and snapshots the JPEG budget.
Two same-tier recording prebuffers therefore consume two encoders; live viewers
can join them without starting two more encoders. A different requested tier
still requires available encoder capacity. No indexed recording is used as a
pretend live source.

The recording ring is bounded by time and bytes. Coverage and pre-roll now use
MP4 sample durations, including AAC alignment, rather than an assumed two-second
fragment. Timestamp gaps fail explicitly. MPEG-TS AAC is converted with
`aac_adtstoasc` before fMP4 muxing. Live HEVC retains the lab's parameter-set/SEI
fix and remote RID fix. HomeKit Off closes viewers, snapshots and the prebuffer;
it does not disable SurvNG recording or detection.

Object-triggered motion is selectable per camera and defaults off. Only recent
persisted detector object events are admitted, while camera power and Detection
must both be on. Historical events, manual reviews and stale detections are
excluded. Each event extends a ten-second monotonic motion window. A one-second
supervisor reconciliation sends retained state after reconnect; the worker
expires motion after five seconds without a heartbeat. HomeKit motion permissions
are enforced again in the worker. Home activity zones suppress this unlocated
motion, with a visible admin explanation; spatial Home zone qualification remains
future work. SurvNG incident zones continue to govern the existing detector path.

State is under `<database_dir or storage_dir>/homekit/<camera-id>` with private
permissions and an exclusive owner lock. Stable automatic ports derive from the
camera ID; collisions require a custom port. Backup this state with the database.
Only an explicit reset removes HomeKit identity and HAP state for that camera.
Disablement, camera removal, startup failure and service shutdown never reset it.

All `/api/homekit` routes require admin scope. Status is allowlisted; pairing uses
an explicit POST and no-store response. QR generation uses existing OpenCV,
without an external QR service or another dependency. The UI clears setup
material on backgrounding, navigation, connection failure, successful pairing,
or after two minutes. Reset requires camera-specific confirmation.

Validation covers real H.264/AAC input over a local HTTP fixture, shared live and
recording normalization to HEVC, Opus RTP, decoded fMP4 with/without audio,
recording durations, snapshot independence, media admission/privacy races,
private media socket permissions, and actual Node worker lifecycle/identity/reset.
The admin browser tests exercise loading, empty, pairing/hiding, object-motion
editing, reset confirmation, connection recovery, and narrow-screen layout.

## Remaining qualification

- Gate is paired and QSV Wi-Fi LIVE is confirmed; complete real-camera cellular/Apple TV,
  recording, audio-off, motion on/off, camera power, privacy and reconnect acceptance.
- Complete exported Home clip/pre-roll inspection, end-of-stream ACK diagnosis,
  and the 24-hour soak review described above.
- HEVC passthrough qualification remains future work; retain the QSV temporal and pacing regressions.
- Qualify spatial Home activity zones before emitting motion with zones active.

Worker dependencies must be built using the pinned bootstrap process. SurvNG's
service account needs Node.js 24, the documented FFmpeg encoders, private writable
state, and LAN access to Apple devices. The worker is an optional child of the
existing SurvNG service; no separate systemd unit is needed. Loading this feature
into an already-running service requires the normal application deployment. This
implementation work does not restart the production service or the paired lab.
