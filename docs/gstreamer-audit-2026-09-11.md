# GStreamer audit and first modernization increment

Date: 2026-09-11. Scope: the `gstreamer` branch, not the running v1.2 service.

## Assessment

The branch had a useful direction—shared native inference, a small grayscale
EMA stream, and recorded-main refinement—but its evidence plumbing was not
ready for dependable live admission. The largest risks were incorrect metadata,
reused observations, and mixing frames/coordinates across streams. Increasing
inference speed alone would not have corrected those failures.

This increment fixes those contracts, ports selected newer tracking/motion
improvements, and adds real Intel-runtime integration coverage. It is a
candidate for isolated testing, **not a production accuracy/performance sign-off**.
No Gate recording was replayed and no Intel GPU was assigned to this test run.

Baseline: remote `gstreamer` at `7bf30b7`; #144 had already been reverted by #145.
PR #146 (`879ceb0`, matched-evidence implementation) was incorporated locally
and reviewed rather than assumed complete because its CI was green. Review
covered native capture/protocol, EMA evidence, alignment, live/recorded
detection, zone/admission policy, tracking, reload/shutdown, observability,
packaging, and their regression tests. This is a subsystem audit, not a claim
that every unrelated application line was inspected.

## Findings addressed

Severity describes the original defect. P1 = correctness/runtime risk;
P2 = efficiency, maintainability, or operational risk.

| Priority | Finding and downstream impact | Implemented remedy / evidence |
| --- | --- | --- |
| P1 | Detection code interpreted mapped video pixels as JSON. A healthy decoder did not imply usable detections. | Read attached JSON metadata with `gstgva.VideoFrame.messages()`. Native positive-result test passes. |
| P1 | No-object inference was not authoritative; older positive boxes could survive. | Enable `add-empty-results`, distinguish missing from empty snapshots, validate payloads, and cap both metadata queues at 32. Native empty-result test passes. |
| P1 | PTS alone was reusable after reconnect; capture generation did not identify every native session. | Carry a reconnect/session ID through inbox, capture, EMA evidence, and matching. Backward PTS flushes old evidence. Reject wrong-session, future, reordered, and stale snapshots. |
| P1 | Harvesting queued messages could change the identity of an already returned frame. | Bind pixels/sequence/PTS/session in one frame packet. Regression test exercises harvesting a newer frame. |
| P1 | PR #146 changed a positional argument and bypassed an existing capture stop guard. | Restore `stop_event` compatibility; PTS/session are keyword-only. The original failing capture test now passes. |
| P1 | Tracking could apply latest live boxes to a main frame and repeatedly count one inference. | Explicit `TrackingFrame` provenance. Main frames use their own detector; matched live snapshots are consumed once. Missing results do not count as misses; completed empty results do. |
| P1 | Main capture inherited the live grayscale/detection graph. This lost color/detail and launched duplicate `gvadetect`. | Separate source roles: bounded BGR main frames, no main-side live detector or JPEG branch. Native main and shared-supervisor tests verify this. |
| P1 | Auto alignment assumed BGR although native EMA input is grayscale; first-incident alignment might never warm. | Accept grayscale; probe main for up to 12 seconds, at most once per 300 seconds while untrusted. Reset learned trust on session/generation/resolution change. |
| P1 | Live boxes were tested against main zones without projection; EMA overlap could then apply that projection again. | Project copies for zone evaluation only; retain live image coordinates for crops and same-stream EMA correlation. Focused shifted-zone/correlation tests pass. |
| P1 | JPEG previews were stamped with qualifier times, manufacturing repeated temporal/color observations. | Do not feed identity-less preview images into tracking catch-up. Use main history/finalized recordings; current matched-live fallback remains available for identical FOV. |
| P1 | The labels filename was assigned to the `labels` class-list property. Both accept strings, so no exception exposed it. | Use `labels-file` for paths and `labels` for configured inline names. Native smoke verifies the actual output label. |
| P1 | Native confidence filtering could discard low-confidence candidates before class/zone policy or two-pass association saw them. | Derive the native threshold floor from downstream candidate/class/zone/tracking requirements. Final admission thresholds remain unchanged. |
| P1 | Reconfiguring OpenVINO workers left the shared GStreamer model/cadence graph running old settings. | Relevant graph changes use the existing transactional manager rebuild. Unrelated hot updates retain their existing path. |
| P1 | Library/plugin/scanner selection mixed Ubuntu and Intel GStreamer ABIs. Native plugins failed to load. | Select matching Intel libraries, typelibs, Python bindings and scanner, re-executing the child before GI imports when necessary. Native tests use GStreamer 1.28.2/OpenVINO 2026.1.0. |
| P1 | One source frame decoded for multiple refinement offsets could count as multiple confirmations. Depth matching could also undo an earlier admission veto. | Reuse inference for exact duplicate source frames and count each once; preserve prior eligibility vetoes. Ported regression tests cover both. |
| P2 | `survng_hybrid` still selected the older greedy association implementation. | Port the newer global one-to-one assignment, stable-size center prediction, and high-confidence appearance recovery order. Keep the `bytetrack` alternative and existing output/lifecycle contract. |
| P2 | EMA repeated mask reductions and expensive per-frame statistics work. | Port exact histogram-based uint8 median/MAD, reuse reductions, and short-circuit empty component masks. No new motion threshold or neural motion model. |
| P2 | Shared capture could outlive manager shutdown; an alive process with a dead reader could be reused. | Close the supervisor after camera workers drain; fail inboxes on reader failure and recreate an unhealthy supervisor. |
| P2 | Configuration and CI did not reveal the native evidence failures. | Add bounded matching counters, full Python CI, real CPU Intel-runtime smoke CI, and an isolated Compose test project. Pin `intel-dlstreamer` to 2026.1.0. |

The native metadata behavior is documented by Intel's
[gvametaconvert reference](https://docs.openedgeplatform.intel.com/2026.1/edge-ai-libraries/dlstreamer/elements/gvametaconvert.html).
The class-list/file distinction and confidence filtering are documented in
[gvadetect](https://docs.openedgeplatform.intel.com/2026.1/edge-ai-libraries/dlstreamer/elements/gvadetect.html).

Selected committed v1.2 improvements were copied/ported into this worktree only:
`4de76b6`, `704bc2c`, `323b805`, `f9c219f`, `2fe7b6b`, `70589fa`, and the committed
hybrid/assignment modules with their tests. This was not a wholesale v1.2 merge;
its unrelated changes and runtime were not modified.

## Resulting ownership and evidence flow

```text
Live decode (one graph per camera, shared model pool)
  ├─ bounded grayscale frames ── EMA ── pinned frame identity ──────┐
  ├─ bounded gvadetect ── versioned snapshots ── session/PTS join ──┤
  └─ 1 FPS JPEG ── display only                                  │
                                                                v
                     live zone projection + same-view correlation
                                      │ provisional admission
Recorded main ── exact-frame refinement ── final policy/confirmation
                                      │
                                      v
                        tracking, events, snapshots, MQTT
                        ├─ main BGR / recorded: own inference, color evidence
                        └─ identical-FOV live: matched boxes once; no color ReID
```

PTS is media time, not UTC. A permitted live association is the newest prior
snapshot in the **same session**, within one detector period plus 50 ms, capped
at 500 ms. This is deliberately bounded association, not a claim that every
qualifier and inference used identical pixels. At 5 FPS the bound is 250 ms;
at 2.5 FPS it is 450 ms. No match means recorded refinement must supply evidence.
One inference cannot manufacture repeated tracking confirmations.

The tracker remains SurvNG's wall-clock-based implementation, with its existing
event persistence and selective ReID. This is not advertised as the reference
ByteTrack implementation. Preserving lower-confidence association candidates
is consistent with the principle in the
[ByteTrack paper](https://arxiv.org/abs/2110.06864).

## Intel optimization: what is justified now

Update: the [full-application replay findings](gstreamer-replay-2026-09-12.md)
supersede the runtime and cadence settings below.

Keep independent bounded branches, an output queue after `gvadetect`, model
sharing, and hardware decode/preprocessing where supported. This increment
chooses `batch-size=1`, `nireq=2`, and latency scheduling as a conservative
starting point for low-rate event detection—not as a universal optimum. Intel
documents these controls in
[gvadetect](https://docs.openedgeplatform.intel.com/2026.1/edge-ai-libraries/dlstreamer/elements/gvadetect.html).

`inference-interval=2` means every other input frame, not 2 FPS. We retain
`inference-interval=1` after the detector branch's own `videorate drop-only`
limit: this avoids running inference on full camera FPS while keeping skipped
input distinct from completed empty inference. The smoke test explicitly
exercises 5 FPS EMA alongside 2.5 FPS detection. The application's current rate
policy remains `max(EMA FPS, enabled tracking FPS)`; there is no new user-facing
independent detection-FPS control in this increment.

Intel's [optimizer](https://docs.openedgeplatform.intel.com/2026.1/edge-ai-libraries/dlstreamer/dev_guide/optimizer.html)
can help benchmark candidate settings. Its example throughput is not evidence
of our camera/model performance. Optimize end-to-end evidence latency, recall,
and total resource usage—not merely decoded FPS. GPU zero-copy behavior and
`va-surface-sharing` remain target-machine measurements, not verified claims.

## Remaining work, in priority order

1. **P1: actual model/export parity.** Compare live `gvadetect` and recorded
   OpenVINO on the same labeled frames: class indices, resize/letterbox,
   confidence, boxes, and NMS. The native element has no universal
   `nms-threshold` property; converter/model metadata owns this, while the
   application has its own NMS policy. This increment does not assert parity or
   silently rewrite model metadata. Validate the current model before changing
   architectures or precision.
2. **P1: real Gate replay and event lifecycle.** Retest the missed vehicle with
   its original zones and aligned main/live clips, including background before
   entry and after exit. Check ONVIF-only and EMA-only triggers separately. Then
   check MQTT/event counts, recording links, covers, and ID continuity. No raw
   ONVIF incident log or original scene was analyzed in this run.
3. **P1: deployment/GPU validation.** The native CPU image passes; the complete
   application Docker build is blocked on this host before the first RUN by
   nested-container AppArmor (`unable to apply apparmor profile`). No host
   security policy was changed. Build on a normal Docker host, then verify
   decoder selection, GPU usage, memory negotiation and reconnects. The new
   application Compose profile has not been started here.
4. **P2: historical cross-stream tracking.** Same-perspective crop/scale is
   supported for motion/zone admission. Track seeding/history across different
   crops is not yet transformed; live tracking fallback is restricted to
   trusted near-identity alignment (2% tolerance). Main/recorded tracking is
   preferred; cropped live events also wait for a main-refinement tracking seed,
   including when the refinement queue is full. Preview-to-history removal can delay catch-up while main warms;
   it intentionally does not invent timestamped evidence.
5. **P2: motion modernization by measurement.** EMA remains a good inexpensive
   candidate generator. Before replacing it, label failures involving shadows,
   foliage, headlights, rain and small/far objects. Evaluate alternative motion
   algorithms in shadow mode against those cases; don't add an always-on
   expensive model without an observed gain. Main tracking still performs CPU
   inference, and the application still has frame-copy boundaries.
6. **P2: operational tuning.** Full graph reloads can briefly interrupt camera
   activity. Start with one test camera. Measure CPU/GPU, RSS, frame age,
   matching-counter deltas, first-event latency and recorder continuity over
   30 minutes before increasing camera count. Counters count lookup attempts
   (including status reads), not unique frames or recall. Shared supervisor
   failure can affect all cameras; automatic recovery is not fault isolation.

Avoid adding a second independent tracker behind `gvatrack`, a new message bus,
or a model zoo now. The next useful increment is measured model parity and
scene evaluation on this corrected evidence contract, followed by one targeted
model/precision or motion change at a time.

## Validation and handoff

See [the test guide](gstreamer-testing.md) for exact commands and acceptance
checks. Native tests use a generated deterministic SSD-shaped model: they prove
plumbing, cadence, labels, empty results and multistream operation—not vehicle
recognition accuracy. Frontend build and all 52 frontend unit test files pass.
The full Python suite passed 2,120 tests (one skipped, 212 subtests). After the
final cropped-seed guard, all 191 focused detection, tracking, lifecycle and
incident tests passed. Native smoke was rerun after bounding the full-video
metadata queue to four buffers (the parent's metadata-only history stays 32).
