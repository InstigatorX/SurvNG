# Incident lifecycle correctness and acceleration follow-up

Base: `v1.3-gstreamer`, merge `d36e549` (PR #199). This follow-up does not
modify `v1.2`, database schemas, incident acceptance policy, model precision,
EMA/live sampling rates, or the native VA surface-sharing path.

## Lifecycle contract

Compressed recording and native live inference run alongside EMA/ONVIF
monitoring. A qualified trigger reserves durable work; matched live metadata
can provide provisional object evidence. Recorded-main refinement, tracking,
ReID, faces, cover selection and semantic indexing strengthen that evidence.
They need not all finish before an incident is persisted.

Camera-primary requests may retain motion-only incidents. Ordinary adaptive
motion and EMA backup/followup requests have different object/correlation
requirements. This change preserves those rules rather than silently turning
an acceleration fix into a stricter admission policy.

## Implemented remedies

### Durable merged trigger authority

`MotionTrigger.admitted_sources` and frozen camera semantic reports are
additive payload fields in the existing durable job. The coordinator orders
camera merges against initial enqueue, the store updates queued/running jobs
transactionally, and retry checkpoints union authority instead of erasing
notices received after claim. Decision start refreshes this evidence, making
normal delivery and recovery agree even after the episode controller is gone.

Reports retain the ingress model-label interpretation; reconstruction does
not reinterpret them against a newly loaded label catalog. The change does
not persist additional raw SOAP or alter frame identity/occurrence time.

The processing cutoff remains decision start. Later notices survive retries,
but do not retroactively change a completed result. Metadata already lost by
an older build cannot be reconstructed, and legacy payloads remain readable.

### Pixel provenance for deferred ReID and tracker seeds

Native live fast-path/fallback evidence is EMA luma, even when represented as
three BGR channels. A shared provenance check prevents those pixels from
entering deferred appearance backfill or color-dependent tracking seed work.
Deferred jobs remain retryable rather than publishing an early vector that
would suppress later attempts through `has_event`.

Cover promotion deliberately preserves original detection/admission metadata.
Its existing `snapshot_source` therefore takes precedence: a recorded-main or
verified tracking cover remains usable even if the original detection is
marked live/provisional. No channel-equality heuristic rejects legitimate
nighttime main footage; legacy unknown-source evidence remains compatible.
Existing appearance-index records are not destructively purged.

### Bounded device-aware optional inference

One proven CPU auxiliary request can finish alongside GPU object-security
work. Optional work still yields to pending/active security, unknown or mixed
devices remain conservative, and offline work remains exclusive within the
Python supervisor. Current worker device/configuration is checked at the
request boundary so a CPU-to-GPU reconfiguration cannot reuse a CPU exemption.

Semantic image/text compilation uses the shared OpenVINO latency and GPU
compilation-thread policy. Production startup and encoder calls acquire
cooperative offline admission. Image indexing yields between individual
images/crops, and transient admission failure is not a compiler failure or a
reason to discard queued semantic work.

This is cooperative scheduling, not GPU preemption. Native live inference
continues independently; an already-started native request or cold compilation
cannot be interrupted for a newly arriving incident. Existing worker timeouts
still bound failure recovery, not guaranteed incident latency. No arbitrary
CPU/BLAS thread caps or extra inference workers were introduced.

### Actual-worker observability

The owner-only socket retains `detector.device` as configured-device output
and adds reported loaded devices, per-worker status, fallback, restart and
crash counters, plus auxiliary-role devices. Stopped workers do not present
stale loaded-device values. Mixed GPU/CPU pools no longer look GPU-only.
Model paths, raw errors and credentials remain excluded.

### Select recorded evidence before GPU download

Explicit VA-API/QSV batch extraction now applies timestamp selection before
`hwdownload,format=nv12`; discarded frames stay on the device. `showinfo`
continues to describe the converted output, and CPU fallback is unchanged.
`hardware_acceleration=auto` behavior is deliberately unchanged; changing its
decode default requires broader hardware/corrupt-stream measurements.

## Validation

Final Python suite: **2,710 passed, 1 skipped, 275 subtests passed**. All **67
frontend unit-test files** passed. The first full run exposed one test fixture
matching the old retry SQL projection; its synchronization hook was updated
without weakening the lease/CAS assertions. The final run passed. The existing
Starlette/httpx deprecation warning remains unrelated to these changes.

Two bounded domain reviews used `gpt-6-astra`: durable trigger authority and
acceleration/scheduling, with cross-review of consequential boundaries. Review
found and resolved stale CPU-admission reconfiguration and semantic timeout
retention issues before handoff.

Three fresh-cache GPU startups each loaded two workers and completed four
inferences without crashes, restarts or CPU fallback. Saved incident snapshots
3212/3216 (person), 3214 (cat) and 3213 (empty Gate) retained their expected
eligible-label results using the end-to-end model on CPU. These snapshot checks
are not a substitute for live incident admission or long-duration accuracy.

Two 90-second full-application scene replays passed, using existing `auto`
and explicit `vaapi` recorded decode respectively. Each saved a person incident
with an existing snapshot and recording, reported loaded GPU devices, and had
zero inference failures, worker crashes/restarts or fallback. They completed
26 and 32 model inferences respectively. The VA-API run also reported a face;
these are asynchronous scene replays, not identical sample-by-sample schedules.

Replays use
disposable databases/media, private loopback video sources, no external network,
and read-only source/config/model mounts; the running service is not restarted.
ONVIF and semantic model execution are disabled in those footage replays:
their new recovery/admission contracts are covered by deterministic regressions,
not a claimed live vendor or real semantic-model GPU soak.

Regression coverage includes merge/enqueue/claim/checkpoint ordering,
reconstruction, lifecycle boundaries, frozen reports, live-to-main cover
promotion, GPU/CPU admission, reconfiguration, semantic admission and
cancellation, loaded-device projection, and exact recorded-frame timestamps.

The retained main-stream clip was checked on the test host with both VA-API
and QSV. Download-before versus download-after selection returned byte-identical
73,728,270-byte BMP output and PTS `0.5, 1, 2.033, 4.032, 8.031` for five
requested samples. Single measurements were 0.790→0.541 seconds (VA-API) and
0.872→0.494 seconds (QSV); these establish compatibility on this clip, not a
general throughput claim.

## Deliberately separate decisions

- Whether ordinary adaptive motion must confirm an object before retaining
  an incident, and whether provisional route watches are desired.
- Guaranteed external alarm delivery: current MQTT publication is best
  effort while disconnected; a durable outbox is a separate contract.
- Automatic recorded HW-backend selection, input seeking, CPU thread tuning,
  and VA preprocessing A/B tests across devices.
- Live vendor-specific ONVIF testing and labeled long-duration accuracy/
  contention measurements. Unit recovery tests are not a live ONVIF claim.
