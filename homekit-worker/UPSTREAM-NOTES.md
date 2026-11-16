# HKSV3 upstream investigation — 2026-11-14

Current result: the user confirmed smooth local LIVE video and the tone after
filtering oversized live SEI metadata (see receiver evidence below), then smooth
cellular video and tone after omitting the on-wire RID extension. The remaining
recording/privacy/soak checks are open. Before those fixes, both iPhone and
direct Apple TV viewing showed a picture/loading spinner without sound, even
though automated peers and an independent FFmpeg receiver decoded the streams.
The controller-parameter SRTP experiment alone did not resolve that failure.
Keep the full hardware gate unverified.

## Pull requests and their applicability

| Source | Finding | Relevance to this lab |
| --- | --- | --- |
| [Scrypted #2186](https://github.com/koush/scrypted/pull/2186) | Draft, unmerged startup fix: publish session control before an optional playback hint; hint rejection must not prevent reoffers and receive-key updates. No discussion comments at review time. Its source tests pass, but the proposed source refactor was not deployed by the author. | The lab uses a direct peer connection and has no `setPlayback` hint or equivalent awaited session-control publication. Do not copy this patch as a local-playback fix. Test remote reoffer/key-update ordering on our actual path. |
| [Scrypted #2152, detailed device trace](https://github.com/koush/scrypted/issues/2152#issuecomment-6282843553) | A contributor reports three distinct remote failures on iOS/tvOS 27.2: an unacceptable high-resolution/bitrate description, an early DTLS ClientHello race, and a reoffer/control race. Local playback worked in that report. | Useful receiver-side evidence, not proof of our local root cause. Our remote offer and producer always select 1080p/1800 kbps. A cellular comparison should use matching 720p/800 kbps or 360p/190 kbps media and metadata. Inspect DTLS startup before adding a buffering workaround. |
| [Scrypted #2174](https://github.com/koush/scrypted/pull/2174), merged | Fixes a demonstrated FFmpeg native Opus decoding defect in the live audio transcode path by selecting the libopus decoder for Opus input. | Our fixture input is AAC and the output encoder is already libopus. Adding an Opus input-decoder flag here would target the wrong boundary. Revisit for future production Opus inputs. |
| [Scrypted #2158](https://github.com/koush/scrypted/pull/2158) | Documents the corresponding native-decoder failure for Opus-to-AAC recording. | Our recording fixture already contains AAC, which is copied. Does not explain this lab's missing live tone. |
| [Scrypted #2098](https://github.com/koush/scrypted/pull/2098), unmerged | Broader audio proposal includes duration-aware Opus repacketization: a packet containing 60 ms of audio cannot be treated as a 20 ms packet just by changing timestamps. | Verify payload duration as well as RTP timestamp deltas. Do not transplant legacy negotiated-rate timestamp scaling into HKSV3: this path uses a fixed 48 kHz Opus RTP clock. Our encoder is configured for 20 ms; no Ring/60 ms source is involved. |
| [Scrypted #2181](https://github.com/koush/scrypted/pull/2181), unmerged | Makes RTSP-source teardown idempotent and prevents forwarding after client disconnection. | No upstream RTSP client exists in this synthetic lab. The ownership/cleanup principle applies; our local/remote stop paths already own FFmpeg, sockets, and peer teardown. Not evidence for changing media formats. |
| [Scrypted #2166](https://github.com/koush/scrypted/pull/2166), unmerged | Cleans up sessions if fetching a camera stream fails before an inactivity watchdog is armed. | Our source is a generated fixture, with start-failure cleanup and bounded sessions. Relevant to later production integration, not a demonstrated cause of the current spinner. |

The changes in #2098, #2174, and #2158 overlap but are not interchangeable.
Distinguish an observed decoder bug from proposals about clipping, payload
duration, and legacy timestamp handling.

## Concrete sender differences in current Scrypted

Reviewed at [fb679504ce2d8bc150bb504a7098826644627100](https://github.com/koush/scrypted/tree/fb679504ce2d8bc150bb504a7098826644627100).

- [Multi-tier RTP](https://github.com/koush/scrypted/blob/fb679504ce2d8bc150bb504a7098826644627100/plugins/homekit/src/types/camera/camera-multitier-rtp.ts)
  preserves prepared accessory SSRCs even if START supplies different SSRCs.
  This confirms the lab's existing SSRC choice. It uses controller-provided SRTP
  parameters; matching that behavior in the lab did not restore hardware playback.
- The same sender defaults to a monotonic local clock for video timestamps,
  preserving one timestamp across all fragments of a picture and keeping adjacent
  pictures distinct. Our lab currently forwards FFmpeg timestamps.
- [The stream sender](https://github.com/koush/scrypted/blob/fb679504ce2d8bc150bb504a7098826644627100/plugins/homekit/src/types/camera/camera-streaming-srtp-sender.ts)
  normalizes sequence numbers, repacketizes HEVC, generates audio timestamps from
  output packet count/duration, and retains corresponding RTP/NTP pairs for RTCP.
  Our producer continuity tests do not establish equality with that wire format.
- [WebRTC signaling](https://github.com/koush/scrypted/blob/fb679504ce2d8bc150bb504a7098826644627100/plugins/homekit/src/types/camera/camera-webrtc.ts)
  retains SDP RID constraints but explicitly documents that sending the RID RTP
  extension broke remote playback. The lab originally advertised and sent it;
  after the local SEI fix, cellular still failed. The next isolated test retains
  RID signaling but omits the on-wire extension, with a peer test checking that
  it is absent from received video packets. The 1080p tier is unchanged for this
  comparison. The user confirmed cellular video and tone work with this change.
  Stock werift treats RID metadata
  as a mandatory receive-demultiplexing rule, so the peer fixture removes
  RID/simulcast metadata for its SSRC receiver while retaining the extension
  negotiation. The test checks decrypted packets for absence of the wire
  extension; protocol tests separately check the real HomeKit offer. The lab's
  actual offer, including its extmap and simulcast metadata, remains unchanged
  for this one-variable comparison.

## Scope and next evidence

[HAP-NodeJS #1132](https://github.com/homebridge/HAP-NodeJS/pull/1132) is the draft
upstream proposal from which the lab's pinned revision comes; its reviewed head
is the same `d0359d529de18ef318b09a74b71380a879023ec4`. Its discussion includes
both working-device reports and unresolved audio reports. There is no new pin
established by that review. CMAF and full talkback remain outside this lab task.

Next distinguish the local sender's wire timing/packetization from a receiver
rejection of its advertised description. The linked Scrypted device trace used
`avconferenced`/`homed` error messages to identify the rejected quality field;
successful ICE, stream-start counters, and FFmpeg decoding alone did not identify
it. Capture only the relevant error text/allowlisted metrics, never setup codes,
SDP, keys, full HAP dumps, or raw sysdiagnose bundles in chat or Git.

## Apple receiver capture — 2026-11-14

The supplied Console capture contains a concrete compressed-frame failure at
10:49:21.919 Eastern, repeated around subsequent keyframes:

- `ParameterSet size=2048 too large`
- `Can't recreate avcC from SPS/PPS -12902`
- `VideoDecoder_NewFormat_SPSPPS failed`

The receiver explicitly selects HEVC and payload type 99. The `avcC` wording
alone therefore does not establish that it selected H.264. Repeated frame-creation
failures explain its continuous keyframe recovery requests.

Inspection of the synthetic fixture found VPS/SPS/PPS lengths of 24/43/6 bytes,
but x265 prefix SEI metadata of 2,403 bytes. The next controlled test omits
prefix/suffix SEI on the shared live RTP feed with FFmpeg `filter_units`, while
retaining parameter sets and picture data. Recording output is unchanged. Tests
check that actual live parameter sets remain below 2 KiB, SEI is absent, and the
reconstructed video still decodes across a fixture loop. The oversized-SEI
explanation is supported by the hardware retest: with this filter, the user
confirmed smooth local LIVE video and the test tone. Cellular initially still
failed and passed after the separate on-wire RID correction. The full protocol
gate remains unverified. Raw Console logs and transport material
are not copied into the repository.
