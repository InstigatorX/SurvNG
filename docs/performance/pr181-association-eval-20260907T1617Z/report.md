# PR #181 saved-detection association evaluation

## Executive verdict

**NO_OBSERVED_EFFECT on confirmed association outputs.** All 13 executed pairs produced literally identical confirmed frame observations and final track summaries. Cue scores changed in 7 profiles across 3 distinct incidents, but no changed confirmed assignment or continuity decision was observed. This does not establish identity accuracy or justify production promotion.

Execution/output-agreement confidence: **high**. Scene-category confidence: **low to moderate**. Selected-case timing confidence: **moderate**. Identity accuracy: **unknown**.

Completed **13/14 requested paired evaluations**, across six distinct incidents and four cameras: all six at fixed_2fps and fixed_075fps, plus one sparse_gaps case. The second suitable gap case was not established; it was not replaced by an inactive clip. There were six sequential capture attempts, all successful, no retries or reserve captures. The approved scratch-only test-package marker resolved the import collision: **64 tests and 64 subtests passed**. The original failed log and blocked-run archive remain unchanged.

## Experiment and runtime provenance

Candidate and experimental baseline are both from isolated source at `67e67ad73b7dc950dd55f15cabd4cb6d8f35531d` (PR #181 current open head, verified twice). Baseline is corrected `survng_hybrid` / `HybridObjectTracker`; candidate is `survng_hybrid_multicue` / `HybridMultiCueObjectTracker`. Parent `b131688` merges PR #180. No tracking math, weights, assertions, production registry or new-track policy was changed. No TAI was implemented.

Live `survng.service` PID **3257018**, owner instance `4aafae2f5f4ffd06d624088b`, cwd `DEPLOYMENT`, uses `SERVICE_VENV/bin/python` and Uvicorn. Start: **2026-09-07 15:26:57 UTC**, after the checkout update. The clean deployed checkout has the same SHA; an in-process build hash is unavailable, so this is deployment evidence, not independent loaded-byte proof. Local API is `/survng` on `127.0.0.1:8088`; owner-only observability socket `/run/survng/observability.sock` worked. Deployment/camera time is America/New_York (UTC-04:00); measurements use UTC. Camera timezone overrides were unset; viewed timestamps corroborate the deployment offset.

Python 3.12.3, NumPy 2.4.6, OpenCV 5.0.0.93, Pydantic 2.13.4, pytest 9.1.1; dependency metadata is in the manifest. Existing OpenCV default: 16 threads, unchanged. Imports resolved into scratch, not the deployed checkout. No packages or optional runtimes were installed. The service's optional upstream trackers were unavailable, but that did not invalidate the full replay bundles or either Hybrid engine.

The running Compare capture serialized identical effective tracking configurations into all six replays. Relevant settings: nominal 3fps, adaptive stable 0.75fps, lost timeout 4s, min confirmations 2, person and vehicle ReID enabled; detailed allowlisted values are in the manifest. Both engines received identical selected detections, timestamps, dimensions, settings, initialization and supplied embeddings. The replay does not measure live selective-ReID scheduling or total accelerator savings.

## Locked corpus and selection limits

Selection was locked at 2026-09-07T16:34:56.850810+00:00, before capture or paired outcome inspection. Metadata selected candidates; modest local image inspection checked scene plausibility. It did not create identity labels. “Crossing” slots are suspected close interactions, not proven geometric crossings. The occlusion slot has partial car/door obstruction; a true same-identity return is unproven. Historical tracking sessions were **terminal/interrupted**, commonly stale handoff, not successfully completed tracking coverage. The selected incidents are days old, with no active tracking-job rows and finalized retained media.

| Slot | Camera / actual child event | UTC anchor | Rationale / confidence |
|---|---|---|---|
| crossing_1 | upper-garage / 63156 | 2026-09-03T21:53:41.823575+00:00 | Two people simultaneously leaving the porch toward a shared narrow image edge; close-interaction candidate, not a proven crossing. (moderate) |
| crossing_2 | front-door / 63346 | 2026-09-04T00:48:07.532451+00:00 | Two people approaching the front porch together at night; simultaneously visible and approaching a shared entrance. (moderate) |
| occlusion_return | upper-garage / 64897 | 2026-09-05T18:08:17.325783+00:00 | Two people move around a parked car; later frames show partial obstruction at the car/door. Suspected occlusion, return identity NOT established. Earlier actual child anchors interaction inside the next 30 seconds without overwriting existing comparison 70 on child 64898. (low) |
| moving_vehicle | back-middle / 65548 | 2026-09-06T22:25:54.513231+00:00 | Saved car boxes and sparse frames show a vehicle traversing the curved road with changing apparent position/size. (moderate) |
| small_distant | back-middle / 64911 | 2026-09-05T18:09:50.845785+00:00 | Saved car box about 55x29 pixels in a 2688x1520 frame; sparse frames show distant road traffic. Daylight, not a low-light claim. (moderate) |
| single_person_control | gate / 64006 | 2026-09-04T20:21:58.503983+00:00 | One person in the saved event, seven consecutive historical observations; single-person control remains limited by sparse visual review. (moderate) |

The original 46 shortlisted details were reused; 11 additional detail requests included one 422 from omitting intermediate child IDs, corrected using the actual full child list. Two 100-item API summary pages were read; the missing categories were widened from 72 hours to seven days, not 30 days. No arbitrary unlimited search occurred. Gate event 65540 was excluded after its “two people” proved to be truck occupants. Child 64898 already had comparison 70 and was not overwritten; earlier child 64897 anchors the same incident's interaction inside the following 30 seconds. Earlier candidate 66002 had an indexed one-second gap and was not used. No post-result substitutions occurred.

All six requested 30-second main-stream windows have indexed continuous coverage and retained files. Each normal Compare capture decoded **90 source-PTS frames** (nominal raw sampling 3fps, not 2fps); all timestamps are strictly increasing and checksums valid. There were 540 captured frames, 632 detections, and 438 supplied embeddings. Actual raw intervals span approximately 0.15–0.49s; per-case cadence and digests are in the manifest. Selected profiles have 60 frames at about 1.99fps, 23 at about 0.74–0.75fps, and 40 at 1.315fps for the one gap case. Sampling only selects existing observations; it cannot manufacture frames. The source-PTS label refers to the exported clip; sparse original-segment image seeks used for scene selection are approximate and were not treated as frame-exact identity labels.

Gap case 64897 contains person observations before, inside and after both imposed gaps, and beyond second 21. At elapsed [21,24) there are 9 raw frames and 18 person detections; [27,30) contains 8 frames and 8 person detections. Profiles drop [5,9) and [15,21), rather than inserting empty detections. Gaps exceeding retention can legitimately create different IDs. This offline harness continues beyond the normal lost-track predicate and is not the complete production lifecycle.

## Per-case/profile results

Each paired value below is **baseline / candidate**. Tracks are confirmed-summary count; births are all new-track diagnostic allocations, including tentative tracks. Observations count confirmed outputs. Fragmentation is the existing count-minus-maximum-simultaneous proxy, not true fragmentation. ReID counts supplied-embedding recoveries, not independent identities. All engines executed successfully. The CSV contains all fields, errors, exact timing deltas/ratios and agreement counts.

| Event / profile | Frames | Tracks | Births | Observations | Fragment proxy | ReID recoveries | Embeddings | Tracker ms/frame B/C | Delta ms / ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 63156 / fixed_2fps | 60 | 3/3 | 4/4 | 97/97 | 0/0 | 0/0 | 97/97 | 0.185/0.245 | +0.060 / 1.32x |
| 63156 / fixed_075fps | 23 | 3/3 | 4/4 | 37/37 | 0/0 | 0/0 | 37/37 | 0.193/0.188 | -0.005 / 0.97x |
| 63346 / fixed_2fps | 60 | 2/2 | 2/2 | 22/22 | 0/0 | 0/0 | 23/23 | 0.075/0.091 | +0.016 / 1.21x |
| 63346 / fixed_075fps | 23 | 2/2 | 2/2 | 9/9 | 0/0 | 0/0 | 10/10 | 0.082/0.084 | +0.002 / 1.02x |
| 64897 / fixed_2fps | 60 | 3/3 | 3/3 | 153/153 | 0/0 | 0/0 | 126/126 | 0.262/0.25 | -0.012 / 0.95x |
| 64897 / fixed_075fps | 23 | 3/3 | 3/3 | 58/58 | 0/0 | 0/0 | 48/48 | 0.254/0.244 | -0.010 / 0.96x |
| 64897 / sparse_gaps | 40 | 5/5 | 5/5 | 97/97 | 2/2 | 3/3 | 82/82 | 0.245/0.297 | +0.052 / 1.21x |
| 65548 / fixed_2fps | 60 | 3/3 | 3/3 | 19/19 | 2/2 | 0/0 | 12/12 | 0.065/0.068 | +0.003 / 1.05x |
| 65548 / fixed_075fps | 23 | 3/3 | 5/5 | 5/5 | 2/2 | 0/0 | 6/6 | 0.089/0.081 | -0.008 / 0.91x |
| 64911 / fixed_2fps | 60 | 3/3 | 3/3 | 18/18 | 2/2 | 0/0 | 8/8 | 0.044/0.039 | -0.005 / 0.89x |
| 64911 / fixed_075fps | 23 | 2/2 | 5/5 | 4/4 | 1/1 | 1/1 | 4/4 | 0.085/0.076 | -0.009 / 0.89x |
| 64006 / fixed_2fps | 60 | 1/1 | 1/1 | 25/25 | 0/0 | 0/0 | 22/22 | 0.071/0.074 | +0.003 / 1.04x |
| 64006 / fixed_075fps | 23 | 1/1 | 1/1 | 10/10 | 0/0 | 0/0 | 10/10 | 0.097/0.084 | -0.013 / 0.87x |

| Event / profile | Valid cue pairs | Ambiguous pairs | Adjusted pairs | Direction available | Confidence available | Changed confirmed frames |
|---|---:|---:|---:|---:|---:|---:|
| 63156 / fixed_2fps | 101 | 14 | 10 | 4 | 8 | 0 |
| 63156 / fixed_075fps | 36 | 4 | 2 | 1 | 1 | 0 |
| 63346 / fixed_2fps | 37 | 36 | 34 | 13 | 32 | 0 |
| 63346 / fixed_075fps | 13 | 13 | 12 | 4 | 9 | 0 |
| 64897 / fixed_2fps | 158 | 17 | 17 | 7 | 16 | 0 |
| 64897 / fixed_075fps | 59 | 6 | 4 | 0 | 4 | 0 |
| 64897 / sparse_gaps | 97 | 13 | 11 | 4 | 9 | 0 |
| 65548 / fixed_2fps | 18 | 0 | 0 | 0 | 0 | 0 |
| 65548 / fixed_075fps | 2 | 0 | 0 | 0 | 0 | 0 |
| 64911 / fixed_2fps | 17 | 0 | 0 | 0 | 0 | 0 |
| 64911 / fixed_075fps | 2 | 0 | 0 | 0 | 0 | 0 |
| 64006 / fixed_2fps | 24 | 0 | 0 | 0 | 0 | 0 |
| 64006 / fixed_075fps | 9 | 0 | 0 | 0 | 0 | 0 |

These counters cover evaluated pairs across the replay, not necessarily selected assignments. Unambiguous strict-geometry matches keep their baseline scores. HMIoU, confidence and direction penalties were exercised where available; no cue adjustment changed a confirmed decision here. Direction/confidence availability uses elapsed timestamps, becomes neutral for stale gaps and can be absent with insufficient motion/history. The one sparse case's five tracks / fragmentation proxy two and three ReID recoveries are identical in both engines; they are not proven errors.

The two vehicle cases and single-person control have **zero ambiguous/adjusted cue pairs** in both regular profiles. These are untouched controls for association-score changes, not evidence about ambiguous-cue quality. Tentative tracks are absent from frame_observations. Whole-track alignment used shared exact input observations and one-to-one maximum-weight matching, preserving split/merge disagreements. Same-class boxes with IoU >=0.95 were marked ambiguous rather than forced; none occurred in these outputs. All outputs and track summaries were also literally equal before ID normalization. No changed timestamps or material disagreement episodes exist, so no observer reruns or diagnostic overlays were manufactured. The analyzer's row-max tie flag is not a general uniqueness test for global assignments; literal equality makes that limitation immaterial here.

**IDF1: NOT MEASURED. True identity switches: NOT MEASURED. True false merges: NOT MEASURED.** There are no independent exact-replay labels. Fewer tracks, equal baseline IDs, stored histories and embeddings cannot establish accuracy. No detection-recall conclusion is drawn from these unlabelled recordings.

## Timing: small selected-case overhead, not fleet CPU

Initial timing is one baseline-then-candidate observation per profile, including detection deepcopy and tracker update wall time, excluding initialization. Ratios are unstable at these small absolute costs. The largest positive initial delta was event 63156/fixed_2fps: +0.060 ms/frame. It was checked with one excluded warmup and three bounded measured repetitions, alternating engine order C/B, B/C, C/B:

| Round | Baseline ms/frame | Candidate ms/frame | Paired delta ms |
|---|---:|---:|---:|
| 1 | 0.168 | 0.185 | +0.017 |
| 2 | 0.167 | 0.176 | +0.009 |
| 3 | 0.167 | 0.187 | +0.020 |

Baseline median **0.167**, candidate median **0.185 ms/frame**: difference of medians **+0.018 ms**, about +10.8%. Median paired delta is **+0.017 ms**; paired range +0.009 to +0.020 ms. Baseline range 0.167–0.168, candidate 0.176–0.187. All repeated observations remained equal. This supports a small observed cost in this selected case, not a statistically established/general regression or a fleet CPU/iGPU change. There is no measured association benefit to offset it in this corpus. Apparent speedups in other one-shot rows are not claimed as improvements. No long soak or inference rerun was performed.

## Live safety, health and side effects

14 before/after snapshots show 13 connected cameras and 11 enabled recordings active throughout sampled checks; detector queue depth and failed-inference count remained zero. The sampled minimum available memory was approximately **10.68 GiB**, above the 1.6 GiB floor. CPU PSI avg10 peaked at 4.05%; memory PSI avg10 stayed 0.00%. PSI is waiting pressure, not CPU utilization. No continuous service-CPU percentage series was collected, and snapshot health cannot rule out every transient issue.

No new recorder/detector failures appeared in the available recent-log snapshots. Old startup stream-open failures around 15:27 UTC and historical interrupted tracking predate this experiment. New log entries show the six normal QSV event-clip builds, not service failures. All six captures completed with source-PTS frames and zero appearance failures. Optional upstream tracker errors did not trigger runtime installation or input fabrication.

Only authorized capture effects were observed: new comparisons **77–82**, six normal 30-second event-clip cache files, and the associated normal temporary clip work. Full pre-existing comparison content/verdict hashes were preserved; no row was overwritten or evicted, including row 70. The normal running database continued changing; it was not treated as static. Cache file sizes/timestamps and comparison provenance are in the manifest. Experiment processes ran nice 10 / idle IO where supported; that does not throttle service-side capture or guarantee iGPU isolation. No streams were added, auth bypassed, debug enabled, priorities changed on the service, packages installed, or independent inference worker started.

The service PID, instance and clean checkout SHA remained unchanged. Configuration preservation needs an explicit caveat: its fingerprint changed from the original blocked run's `5cad7961…` to `49b26dd9…`, with file mtime **16:26:09 UTC**, before the resumed snapshot and first capture. This coincides with the operator token being supplied. No task code wrote configuration; the saved tracking subtree is unchanged. The full original config was intentionally not retained, so not every external field difference can be reconstructed. The resumed observed fingerprint matches the final one. This is an external change, not an assertion that all live configuration stayed byte-identical across the user's authorization interval.

## Recommendation and artifacts

Keep the candidate offline; do not promote or tune it on this evidence. The smallest justified next evaluation is a corpus selected independently for association ambiguity and identity-labelled before comparing tracker outcomes, without selecting cases based on candidate wins or disagreements. No runtime fix, architectural expansion, or TAI work is justified by this test. The current sample demonstrates successful execution and confirmed-output agreement on these replays, not superior association quality.

Raw inputs and full Compare responses remain private locally. The ZIP contains sanitized report, summaries, manifest, per-pair and repeated-timing results, agreement records, bounded logs and scripts—no raw embeddings, full configs, DBs, model weights or full recordings. `review/` records why there are no disagreement overlays. Reproduction commands are in `commands.md`; raw replays must be supplied privately to reproduce. Astra (`gpt-6-astra`, agent `astra_association_review`) directly reviewed the relevant source, analyzer, all 13 result/agreement files and repeat-timing evidence. No GitHub write or PR was made. The API-session process was closed, and its in-memory token was not persisted; revoke the token shared in chat when finished.
