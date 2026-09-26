# Native HKSV3 implementation status

Target: two cameras, Linux/systemd, iOS/tvOS 27, 1080p/720p/360p HEVC,
HDS recording. Hardware encoding is preferred; software encoding requires
per-camera opt-in. Apple Home privacy affects HomeKit only. HomeKit motion follows
SurvNG detection enablement. Direct CMAF uploads, legacy HKSV, and talkback are
out of scope.

## Gate 1: protocol acceptance

The [synthetic worker](../homekit-worker/README.md) provides an isolated protocol
test. HAP is pinned to `d81fba565ee26e82170d5f4f8cd358c2fd773f6c`, with
`werift` 0.24.4 and retained MIT-licensed media helpers. It intentionally has no
production camera input and no integration-enable setting.

Automated tests exercise generated media and local peers. Real Apple pairing,
cellular viewing, and Home recording acceptance remain required. The original
implementation plan explicitly requires resolving this gate before production
camera integration. A lab build is not a finished native HomeKit feature.

Initial validation: the Node 24 build, repeat bootstrap from locked dependencies,
and 16 tests pass, including real FFmpeg HEVC/AAC fragments, Opus conversion,
SRTP decryption, SFrame HEVC/audio reception through a local WebRTC peer, and
startup/restart/shutdown.

Hardware check (2026-09-26): the synthetic accessory paired successfully and
received HAP accessory reads across VLANs, but Apple Home displayed “No Response”.
No snapshot handler or local/remote stream setup was invoked. The Apple TV was
then confirmed to run tvOS 18.6, below the planned tvOS 27 target. Protocol
acceptance is blocked pending the hub upgrade and a repeat test; this is not a
successful video or recording acceptance result. Retain the existing pairing
for the retry. The 24-hour soak has not been run.

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
