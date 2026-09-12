# GStreamer cross-subsystem review campaign

Branch: `fix/gstreamer-deep-review`, based on `v1.3-gstreamer` at `aba69f5`
plus PR #197 (`198ec65`). PR #197 was still open during this campaign; its
detector fixes and tracking deadline fix remain part of the intended baseline.
No v1.2 source changes or deployment changes are included.

## Scope and review cycles

Reviewed native graph construction, GPU/host memory boundaries, sampling and
NMS policy, shared-process recovery, per-stream inboxes, PTS/session matching,
capture publication, EMA evidence retention, spatial alignment and live
admission, tracking seed/history/cover/ReID handoffs, recorded refinement,
configuration replacement, owner-only observability, and preview consumers.
Existing event/database and recording formats are unchanged.

1. Traced live evidence into tracking; reproduced and fixed seed provenance,
   duplicate inference consumption, and history-priority defects. Ran focused
   integration tests, then replayed native detections from retained footage.
2. Re-reviewed recovery and presentation paths. Reproduced an RTSP-resume false
   watchdog trip and stale JPEG freshness. Fixed both and tested genuine
   inference stalls, preview expiration, and unchanged camera/media consumers.
3. Ran the full suite, investigated its optional-comparator fixture failure,
   corrected that fixture with a default-policy control, and reran the suite.
   Reviewed the consequential diffs and added finalized-recording priority
   coverage. Checked retained incident snapshots through recorded refinement
   and ran native GPU recognition and worker startup/teardown checks.
4. Full-application replay reproduced a native OpenVINO startup crash that
   short/warm-cache checks missed. Reduced it to camera-free cold compilation,
   captured a native debugger backtrace, and tested allocator, cache, package,
   and compilation-concurrency differences. Applied a bounded GPU compilation
   policy across object, auxiliary, and native live inference; repeated the
   full suite, cold starts, recognition, and application replay.

## Confirmed findings and fixes

| Finding | Effect | Remedy |
| --- | --- | --- |
| Live/EMA tracking seed treated as color | Grayscale expanded to BGR could feed ReID and cover selection | Preserve the existing live-source provenance; skip color-only seed enrichment while retaining geometric tracking |
| Seed inference not marked consumed | The same native result could count again on a later qualifier | Carry session/sequence/PTS through live admission; initialize the tracker inference high-water mark and reject repeated/older results in that session |
| Near-tie priority implemented as timestamp order | An earlier live frame could displace available main or recorded evidence | Explicitly prefer finalized recordings, then main history, then live history within the existing duplicate window |
| Camera pause counted toward shared inference timeout | A resumed video frame could restart all camera graphs before its asynchronous detection completed | Start the normal bounded inference grace at video resumption, without changing the actual last-inference timestamp or weakening continuous-video stall detection |
| JPEG freshness borrowed from EMA video | A stalled preview branch could display an old JPEG indefinitely | Track preview receipt age independently and clear preview state on stream teardown |
| Parallel cold GPU compilation crashes natively | A detector worker can segfault before becoming ready, delaying or preventing recorded refinement | Use OpenVINO's `COMPILATION_NUM_THREADS=1` for GPU compilation; preserve inference streams, request pools, precision, cadence, and VA surface sharing |

The compilation policy is an evidence-backed containment of a native compiler
fault, not a claim to repair Intel's heap implementation. A GDB backtrace showed
`malloc`/`posix_memalign` beneath `libopencl-clang.so.17` during `compile_model`.
Cold starts also emitted invalid-SPIR-V/allocator failures. Loaded library sets
before compilation were identical in standalone and application workers.
Changing `MALLOC_ARENA_MAX` from 4 to the service's 16 reduced but did not remove
the failure; an alternate official IGC library overlay also failed. Neither is
shipped as a remedy. The exact upstream memory-corruption cause is unproven.

Single-thread compilation passed five fresh-cache differential cycles and the
final regression checks below. It can modestly increase cold model-loading time;
it does not serialize runtime inference. CPU/NPU compilation settings remain
unchanged, virtual devices scope the property to GPU, and a CPU fallback does
not inherit a GPU-only property. Intel documents compilation thread limits as a
supported [model-compilation memory control](https://docs.openvino.ai/nightly/openvino-workflow/running-inference/optimize-inference/optimizing-memory-usage.html).
Re-evaluate the policy only after a future compiler stack passes repeated cold
starts and full-application replay, not just a warm-cache smoke check.

The optional Deep OC-SORT test named “when proximity is disabled” left the
adapter's nonzero proximity gate enabled. The test now disables that gate
explicitly; a new control verifies that the unchanged default rejects a distant
appearance-only recovery. No production comparison threshold was changed.

## Validation

- Final full local Python suite: **2,655 passed, 272 subtests passed, one skipped**.
  No test was deselected. One existing Starlette/httpx deprecation warning.
- Frontend build passed; all **67 frontend unit-test files** passed.
- Intel DL Streamer 2026.2 native CPU smoke passed: independent 5 FPS qualifier
  and 2.5 FPS inference, empty results, shared multistream capture, per-camera
  widths, BGR main frames, raw NMS policy, and unsuppressed final YOLO26 outputs.
- On the test server's newer GPU, the preserved Downstairs walk-through with
  its current `e2e_openvino_model/best.xml` produced **81 sampled frames, 19
  person-positive frames, and zero person detections in 36 empty-window
  frames**. Playback reached EOS in about 13 seconds. This is accelerated
  recorded playback at one inference sample per media second, not live latency.
- Replayed all 81 exported native results through live admission: 19 person
  admissions, no duplicate model calls. Two tracking windows completed with
  zero coverage gaps: nine processed frames through the occupied window and
  five through the departure window. Grayscale seed/live frames caused zero
  ReID or cover-selection calls. Missing/duplicate evidence remained distinct
  from completed empty detections.
- Existing saved incidents replayed through the actual recorded-refinement
  detection/enrichment boundary on CPU: Downstairs **3212** and Foyer **3216**
  retained person detections; Downstairs **3214** retained cat; Gate **3213**
  remained empty. These checks use stored snapshots, not manually relabeled
  ground truth or a complete temporal-refinement accuracy benchmark.
- Final isolated GPU object-pool check: three fresh-cache startup/teardown
  cycles, each with two workers and four refinement requests. Every worker was
  verified GPU-loaded, with zero crashes, restarts, or fallback. Cold-cycle time
  was about 13 seconds under the 3-CPU/6-GiB test-container limit.
- Real OpenVINO `AUTO:CPU` compilation passed with the GPU-scoped policy.
- Final native GPU recognition after the compilation policy retained exactly
  **81 frames / 19 person-positive / 0 false positives in 36 empty frames**.
- A 90-second full-application replay with the formerly failing allocator
  setting (4 arenas), fresh application/model-cache state, preserved main/live
  footage, and the candidate code passed without a native startup crash.
  Both object workers loaded on GPU; the application saved a person incident
  with its snapshot and recording. It completed 23 recorded inferences, zero
  failed inferences, averaging 28.25 ms. The test asserts startup-log cleanliness
  separately because final `failed_inferences=0` does not count startup crashes.
- A second 90-second replay using the service's 16-arena setting also passed:
  both GPU workers started once, a person incident/snapshot/recording was saved,
  and 24 recorded inferences completed with zero failures (37.05 ms average).
  Both application runs shut down cleanly, in 1.21 and 0.98 seconds respectively.
- Replayed the final native results through admission/tracking again: unchanged
  19 person admissions, zero duplicate inference/color-enrichment calls, and
  zero coverage gaps in both temporal windows.

The initial full-suite attempt lacked generated frontend assets in the new
worktree; building the frontend resolved collection. The first local Docker
smoke could not load the host's AppArmor profile; the isolated container was
rerun with the test environment's existing `apparmor=unconfined` convention.
No service security settings were changed. A native plugin-scanner PyGObject
warning occurred during GPU recognition; the actual pipeline completed and
passed its assertions. It is not presented as a warning-free runtime.

## Replay tools and evidence boundaries

`scripts/gstreamer-model-check.py --results-json <new-file>` can now export
validated native detections without overwriting an existing file.
`scripts/replay-gstreamer-evidence.py --video <recording> --results-json <file>`
replays those results through admission and tracking without touching a
database, publishing notifications, or requesting another inference.
Its clock is synthetic and its video seeks supply pixels for provenance/shape
checks; it does not claim exact-frame color inference or real-time latency.
`scripts/openvino-gpu-startup-check.py --model <model.xml> --iterations 3`
checks fresh-cache GPU worker startup, inference, and shutdown without using
application configuration or touching existing caches. Run the native model
check with the Intel image's system Python and distro GI path (the application
venv alone does not provide the complete PyGObject package).

Original recordings, models, and incident databases were not modified. Test
artifacts live outside the repository in task-specific temporary directories.
The normal test container remained on `sha-aba69f5-intel`; candidate code was
mounted only into disposable test containers.
After testing, the normal service's owner-only runtime snapshot reported all
three cameras connected, detector ready, and zero failed recorded inferences.

## Remaining confidence limits

The earlier VA-API synchronization failure was not reproduced. The distinct
OpenVINO cold-compilation startup crash was reproduced and contained as described
above; its upstream native root cause is not established. No speculative shared
VA-context rewrite, system-memory inference fallback, threshold tuning, or cadence
reduction was introduced. A deployment soak and repeated full-fleet reconnect/
configuration-reload testing remain necessary for long-duration GPU stability
confidence. Passing this campaign is not an exhaustive accuracy or driver
reliability guarantee.
