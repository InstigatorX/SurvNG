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

## Remaining implementation after the protocol gate

1. Add independent ActivityEventBus subscriptions with atomic snapshots, ordered
   acknowledged replay, duplicate suppression, and explicit replay gaps.
2. Add HomeKit leases and separate encoder/bandwidth budgets to MediaSessionManager;
   count shared producers once and release capacity only after resource teardown.
3. Add a Python-owned live EncodedFragmentSource and demand-driven profiles,
   hardware qualification, HEVC passthrough/transcoding, bounded pre-roll,
   monotonic timing, and explicit reconnect discontinuities. Preserve indexed
   recording playback behavior.
4. Connect the TypeScript protocol delegates to Python over a private versioned
   control socket and a separate media channel. Replace synthetic process ownership
   with Python-owned shared producers and integrate camera/mode/generation teardown.
5. Qualify spatial motion for HomeKit zones without changing SurvNG incident zones;
   unlocated ONVIF events need visual qualification when HomeKit zones are active.
6. Integrate stable per-camera identities, persistent pairing, authenticated admin
   enablement/qualification/pairing/reset APIs, and the HomeKit settings panel.
7. Package the optional worker under systemd and expose sanitized observability.
8. Complete two-camera hardware acceptance and the 24-hour soak; retain the
   experimental designation until these pass.

No production configuration migration or service restart is required for the lab.
