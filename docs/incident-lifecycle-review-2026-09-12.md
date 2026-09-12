# Incident lifecycle re-audit and PR #200 comparison

## Scope and comparison baseline

Re-audited September 12, 2026 against `v1.3-gstreamer` at `0418987`, including
qualification-driven inference (`45ee776`), audit-event ownership (`2ce7124`)
and the application process-name change (`0418987`). The PR incorporates these
commits by merge; published history is preserved. The re-audit was performed
by the primary agent without delegated reviewers.

The original PR remains relevant. Most of its changes were absent from the
current target. Its assumption that every live fast-path image was luma became
incorrect after production capture changed to color frames. The original
historical review is retained below for comparison.

## Findings and implemented changes

### F1: live color snapshots were incorrectly excluded from appearance work

Original PR `snapshot_has_luma_only_pixels()` classified `live_fast_path` and
`live_fallback` by source name alone. Current production capture supplies BGR,
but still uses those source names. The old check therefore deferred valid
snapshot ReID and excluded valid tracker-seed appearance/cover work.

The capture adapter now supplies original `pixel_format`. EMA retains that
format before expanding luma to three channels; the timestamped evidence
adapter passes it into `frame_pixel_format` on initial and live-fallback object
metadata. Serialization preserves it for deferred work after reopening the
store. Known BGR is usable, GRAY8 is deferred, and legacy live metadata without
format remains conservative. Recorded/verified tracking cover provenance takes
precedence over the original detection provenance. No channel-equality heuristic
rejects legitimate nighttime BGR imagery.

Regression evidence: color backfill cases failed against the original helper.
Tests now follow real preprocessing, both camera evidence adapters, detector
results, incident persistence, PNG storage, reopened event state and deferred
appearance indexing. Six cases cover notice, qualified EMA and recorded-live
fallback, each with BGR and luma input. Both inputs have equal channel values
once stored, proving that the decision uses provenance. Tracker seed and cover
promotion tests cover known color, known luma and legacy metadata.

### F2: unavailable refinement could strand an admitted incident's tracking

`MotionIncidentService.process()` defers tracking while recorded refinement is
available. An exception during refinement already handed the initial incident
to tracking. A nonterminal `object_detected=None` result, including
`refinement_unavailable_preserved`, only retried the durable job. It never made
the initial handoff, even when attempts later exhausted.

That result path now makes the same idempotent initial handoff as the exception
path while retaining durable refinement retries. A regression reproduced the
missing handoff before the fix and verifies it afterwards. Incident admission,
recorded retry policy and tracking capacity are unchanged. Tracking remains
optional: a declined handoff is observable, not a promise of eventual execution.

### F3: the evidence-path documentation described superseded inference

Updated `incident-evidence-data-path.md` to describe qualified initial inference
on color frames, shared Python inference workers, post-inference freshness and
identity checks, explicit continuous-native integrations, pixel provenance and
unavailable-refinement handoff. This changes documentation, not cadence or policy.

## End-to-end ownership audit

| Boundary | Current contract and comparison with original PR | Evidence / disposition |
| --- | --- | --- |
| ONVIF subscription and ingress | Create, PullMessages, renew, unsubscribe and reconnect remain separate, generation-bound operations. Raw SOAP notifications are authoritative when captured; Zeep objects are a fallback. Adaptive mode keeps camera notices diagnostic. | Reviewed `onvif_events.py`, ingress and existing ONVIF/clock/semantic regressions. No new vendor-specific capture was taken and no topic/dialect behavior was changed. |
| EMA qualification and selected pixels | Qualified evidence retains camera/capture generation, sequence, session and time. Color capture now serves initial inference; EMA preprocessing still uses luma. | Rechecked selected-frame and post-inference invalidation tests. F1 preserves format through the same buffer. |
| Trigger merge and recovery | Camera/EMA admitted-source authority and frozen label interpretation must survive initial enqueue, merge-after-claim, retry checkpoints and restart. Decision start remains the cutoff. | Original PR transaction/authority lock and additive payload merge retained; reconstruction, concurrent enqueue/checkpoint, completed-job and generation regressions pass. |
| Initial inference and incident admission | Initial workload takes priority; stale or wrong-session results do not attach to new pixels. Missing/negative initial evidence cannot cancel recorded discovery. Camera-primary and EMA correlation rules remain distinct. | Demand-inference, decision-handler and detection-path replay coverage. No threshold, route-watch or incident policy change. |
| Recording process, index and files | Independent FFmpeg processes copy video into MP4 segments. Start reservations prevent duplicate owners; shutdown checks ownership; timestamp warnings trigger bounded recovery. Finalized recording samples supply stronger evidence. | Reviewed recorder lifecycle and warning handling. Full suite exercises real FFmpeg MP4/HLS timing and recorded-frame integration; footage replay checks saved evidence and referenced files. |
| Refinement and completion | Durable jobs preserve occurrence identity, owner leases, result checkpoints and completion context. Recovery can replay completion without repeating successful inference. Audit-event uniqueness respects the latest base fix. | Existing event-store/refinement/audit-conflict regressions retained. F2 closes the unavailable-result handoff gap. |
| Tracking, faces and ReID | Tracking is bounded optional work. Recorded/qualified color can seed appearance; legacy/luma cannot. One proven CPU auxiliary can finish while GPU security proceeds; unknown/mixed devices remain exclusive. | Original device admission, CPU reconfiguration and actual-device worker checks retained. New provenance tests cover saved snapshot and tracker seeds. |
| Cover replacement and indexes | Promotion updates display geometry and snapshot provenance atomically, preserving admission. Semantic revision invalidation prevents old queued/in-flight evidence from replacing the new cover. | Existing promotion, geometry, semantic refresh and stale-revision tests. Recorded/verified tracking cover provenance overrides original luma metadata. |
| Semantic GPU work | Startup and encoder calls use shared compile policy and cooperative admission; images/crops yield between calls. Admission contention retains the in-flight queue item. | Original semantic admission/cancellation/queue-pressure regressions retained. Production color capture now removes continuous native inference from the default path; explicit native integrations remain outside Python admission. |
| Notifications and shutdown | Incident updates refresh clients/indexes without another object alarm. MQTT is best effort. Durable security jobs recover; optional tracking and semantic queues are not exactly-once external delivery. | Reviewed manager dispatch and lifecycle tests. No durable MQTT outbox or stronger delivery promise added. |
| Runtime observability | Configured device remains compatible; actual loaded devices, mixed workers, fallback and restart/crash counters are added to the owner-only allowlist. | Original socket tests retained. Running baseline snapshot showed 13 connected cameras and two alive object workers; it did not expose the PR's loaded-device fields. |
| Recorded hardware extraction | Explicit VA-API/QSV selects by decoded PTS before GPU download; CPU fallback and automatic-backend selection remain unchanged. | Original extraction-order and exact-PTS regressions retained. Earlier byte-identical hardware comparison below is historical, not a repeated benchmark. |

## Current validation

- Full Python suite on the updated implementation: **2,781 passed, 1 skipped,
  275 subtests passed** in 65.44 seconds. The existing Starlette/httpx
  deprecation warning remains. This run included four new provenance cases;
  the subsequent extension to all six notice/EMA/fallback cases passed separately.
- First focused lifecycle run: **286 passed, 2 subtests passed**.
- Fresh full-application comparison: two sequential **140-second** replays,
  baseline `0418987` and the updated PR implementation, with **28 runtime
  snapshots each**. Both persisted **two person incidents**, completed **two
  durable refinement jobs**, and retained every referenced snapshot/recording.
  All four referenced recordings fully decoded with FFmpeg: exit 0 and no
  error output. Both cameras stayed connected and reported zero inference
  failures. The candidate reported two loaded GPU workers, zero crashes,
  zero restarts and no CPU fallback. Baseline lacks these new socket fields;
  its log showed exactly two GPU worker-ready messages and no ERROR lines.
- `git diff --check` passed.
- No frontend code changed; frontend tests and the original GPU cold-start /
  VA-API-versus-QSV measurements were not repeated in this re-audit.

## Fresh footage comparison

The same retained Downstairs clips fed private loopback RTSP publishers:
H.264 video in MP4, 896×672 live and 2560×1920 main. Replay publication and
recording used video stream copy; recording output was segmented MP4. Explicit
CPU FFmpeg decoding verified the four referenced output files. This check
found no decoder or timestamp/muxer error output; it does not certify every
recording produced during the run.

Each run used separate configuration, database, recording index, media, model
cache and observability socket. MQTT, ONVIF, audit AI and semantic-model
execution were disabled in the disposable configuration. Existing model,
threshold, tracking settings and 5 FPS capture were retained. Runtime tracing
wrapped initial detection and tracking only in the copied replay sources; no
instrumentation is committed or installed in the running service.

| Observation | Current target | Updated PR |
| --- | ---: | ---: |
| Person incidents / completed refinement jobs | 2 / 2 | 2 / 2 |
| Referenced snapshots / recordings present | 2 / 2 | 2 / 2 |
| Total detector inferences / failures | 53 / 0 | 93 / 0 |
| Completed tracking sessions | 1 (39 frames) | 2 (43 and 42 frames) |
| Initial inference calls, duration in ms | 45.28, 40.92 | 37.52, 46.74 |
| Live initial objects explicitly marked BGR | No | Yes |
| Loaded-device socket field | Not available | GPU |
| Object-worker crashes / restarts / fallback | Not exposed by baseline socket | 0 / 0 / false |

These are asynchronous runs of a looping scene, not identical frame schedules
or an accuracy/throughput benchmark. Different refined/cover decisions and
tracking sessions account for different work volumes. Neither the extra
tracking session nor a timing difference is proof that F2 caused an improvement:
the unavailable-refinement defect is established by its deterministic regression.
The final candidate incidents used recorded-main evidence; the replay trace
verified BGR provenance on their earlier live initial results. Persistence of
live-format metadata is covered by the six cross-boundary tests.

Host-local replay evidence is retained at
`/tmp/survng-pr200-reaudit-2vo4xp8j` (per-run runtime samples, traces, disposable
databases/media and `summary.json`); its harness is
`/tmp/survng-pr200-reaudit-replay.py`. These are local audit artifacts, not
repository assets or a portable benchmark. Generated replay configurations
were removed on shutdown. The original 90-second auto/VA-API comparison below
used the previous architecture and remains historical evidence only.

## Limits and remaining decisions

The re-audit does not reconstruct previously lost trigger metadata or purge
already-indexed weak appearance vectors. Unknown historical live snapshots stay
conservative. Cooperative admission cannot preempt a native call or compilation
already in progress. No latency or accuracy guarantee follows from a short
single-scene replay. Tracking handoff is optional and not exactly-once across a
process crash. Vendor-specific ONVIF operation and a real semantic-model GPU soak
still need separate evidence; deterministic tests are not those measurements.

No v1.2 changes, database schema migration, model/precision/cadence change,
new dependency, automatic decode-selection change or live-service deployment
is included. Route/significance policy, guaranteed MQTT delivery, broader
accuracy evaluation and hardware/thread tuning remain separate decisions.

## Original PR #200 review (historical)

The sections below describe commit `07885ad` against `d36e549`. Their model,
review and validation claims belong to that earlier run; they are not new
measurements or a description of the current capture architecture.

Base: `v1.3-gstreamer`, merge `d36e549` (PR #199). This follow-up does not
modify `v1.2`, database schemas, incident acceptance policy, model precision,
EMA/live sampling rates, or the native VA surface-sharing path.

### Lifecycle contract

Compressed recording and native live inference run alongside EMA/ONVIF
monitoring. A qualified trigger reserves durable work; matched live metadata
can provide provisional object evidence. Recorded-main refinement, tracking,
ReID, faces, cover selection and semantic indexing strengthen that evidence.
They need not all finish before an incident is persisted.

Camera-primary requests may retain motion-only incidents. Ordinary adaptive
motion and EMA backup/followup requests have different object/correlation
requirements. This change preserves those rules rather than silently turning
an acceleration fix into a stricter admission policy.

### Implemented remedies

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

### Validation

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

### Deliberately separate decisions

- Whether ordinary adaptive motion must confirm an object before retaining
  an incident, and whether provisional route watches are desired.
- Guaranteed external alarm delivery: current MQTT publication is best
  effort while disconnected; a durable outbox is a separate contract.
- Automatic recorded HW-backend selection, input seeking, CPU thread tuning,
  and VA preprocessing A/B tests across devices.
- Live vendor-specific ONVIF testing and labeled long-duration accuracy/
  contention measurements. Unit recovery tests are not a live ONVIF claim.
