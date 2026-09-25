# Production hardening audit

Baseline: `347d6d2`. This is an execution-path audit, not proof of 30-day stability. No production fault injection or camera disconnection is performed during the audit.

## Ownership and concurrency map

| Owner | Execution contexts and resources | Boundaries / overload behavior |
| --- | --- | --- |
| FastAPI lifespan / AppManager | HTTP event loop and sync worker pool; manager generation; startup admission; orderly teardown | Reload RLock, generation leases, AI-operation drain; camera dependencies retained if camera shutdown is incomplete |
| CameraLifecycleService | Per-camera capture sources, observer dispatcher, motion analysis/decision threads, ONVIF listener and object tracking | Operation lock serializes lifecycle; state lock protects short transitions; generation checks reject stale work |
| CameraCaptureService / MotionAnalysisService | FFmpeg/OpenCV decode, NumPy frames and derived gray/color frames | Latest pending frame per source; bounded analysis mailbox replaces obsolete work; bounded frame histories and copy/latency telemetry |
| InferenceLifecycle / supervisor | Isolated detector/face/depth/re-ID processes, shared frame memory and IPC | Prioritized workload admission; optional work sheds; per-worker request lock and deadlines; terminate/kill escalation |
| Recorder / retention | Independent FFmpeg recording processes, directory keepers, watchdog, index and retention threads | Recording continues independently of inference; index/retention retry loops; playback leases and deletion claims protect files |
| Event/face/semantic stores | Shared SQLite main database; camera, projection, recognition and API callers | WAL reads, shared writer RLock and store locks; short-lived connection contexts; durable motion/evidence outboxes |
| Semantic / face / evidence workers | Model bootstrap/backfill/indexing, face recognition, derived notifications | Bounded queues; weak revision tokens; bounded projection retry/delivery maps; durable pending rows survive failed enqueue |
| MediaExportManager / recording media | One export worker, bounded job queue, FFmpeg remux/transcode sessions and temporary files | Queue size 100; admission budgets; weak per-cache-key locks; age/size cache cleanup |
| StateEventBroker / browser media | Bounded history and per-client SSE queues; WebSocket relays; browser players/listeners/timers | Slow SSE clients drop oldest work; disconnect cleanup; media effects close sockets, clear timers and revoke URLs |
| RuntimeMonitor / local observability | Monitor thread, retained telemetry DB, owner-only Unix socket | Bounded diagnostic samples, retained rollups, allowlisted snapshots; no secret-bearing runtime dumps required |

No GStreamer pipeline implementation was found on the inspected production media paths; they use FFmpeg/OpenCV and browser/go2rtc transports. There is no GStreamer lifecycle to validate here.

## Critical findings — confirmed and fixed

1. **P1 CONFIRMED — AI cancellation releases ownership too soon.** `intelligence_routes.py`, `assistant_chat` and `motion_audit_ai_analyze`: an async route registers work and acquires a semaphore, awaits `asyncio.to_thread`, then releases both in the coroutine's `finally`. Cancelling the request does not stop the thread. A new request can exceed the limit, and reload can retire the manager still used by assistant tools. Trigger: cancellation/shutdown during provider I/O. Fix: execute admission, blocking work and cleanup in the same thread, retaining the async route interface. Risk: cancellation intentionally does not terminate provider I/O; provider deadlines remain necessary. Test with a blocked provider, cancel its caller, verify admission remains held until actual completion and another request is rejected.

2. **P1 CONFIRMED — export worker can die permanently on transient storage failure.** `media_exports.py`, `_run`: job lookup, cancellation/failure persistence and cleanup execute outside effective exception containment. SQLite busy/I/O or filesystem errors escape the only worker; new jobs still enqueue. Trigger: transient DB or storage failure during normal export/maintenance. Fix: interruptible retry for job bookkeeping and isolated, logged cleanup failure. Risk: do not repeat a completed transcode or lose the dequeued job. Test injected lookup, failure-persistence and cleanup faults followed by successful work and stop during retry.

3. **P1 CONFIRMED — face queue refill can kill recognition.** `face_store/recognition.py`, `_recognition_loop`: `_queue_pending_recognition` can raise from SQLite or model status both on idle refill and after a job. It runs outside the per-observation handler, after clearing the refill flag. Queue saturation followed by a transient DB failure permanently kills the thread and loses the refill signal. Fix: retain the flag on failure and retry with stop-aware delay and scoped logging. Test failure on both refill paths, subsequent recovery and shutdown.

4. **P1 CONFIRMED — reload rollback can start overlapping generations.** `manager_reload.py`, `ManagerGenerationLifecycle.reload`: failed candidate cleanup is logged, then recovery starts regardless. If candidate start succeeded but config persistence failed, incomplete camera shutdown leaves the candidate alive while recovery opens the same cameras/recorders. Recovery startup failure also lacks explicit cleanup. Fix: refuse recovery when candidate teardown is incomplete, retain a reachable owner for supervisor cleanup, and clean up failed recovery. Risk: availability is intentionally sacrificed when exclusive ownership cannot be restored. Test cleanup failure after persistence failure and recovery startup failure.

5. **P1 CONFIRMED — detector replay blocks the event loop and releases ownership on cancellation.** `detection_routes.py`, `detect_debug_frame`, through `manager_access.py`'s async guard: acquiring the synchronous reload RLock on the event loop blocks every HTTP/SSE/WebSocket coroutine while a reload holds it. Cancelling replay releases its lease while `to_thread` inference continues. A held-lock reproduction delayed a 20ms heartbeat by **1.0006 seconds**; cancellation reduced active leases to zero before inference completed. Fix: stream the bounded upload first, then decode/detect/depth-process in one synchronous, leased worker. Tests now keep the heartbeat below 500ms under the same one-second lock hold and retain the lease through cancellation. Risk: model I/O still needs its existing deadlines; request cancellation deliberately does not kill an inference process.

The reload finding also includes `AppManager.start_all`/`stop_all_with_runtime_preferences` marking failed teardown as closed and `InferenceLifecycle.close` discarding failed cleanup callbacks. Both prevented an effective retry. Failed owners now remain reachable; inference closes admission but retains only failed cleanup operations for retry. If teardown cannot complete, reload refuses overlapping generations and requires supervisor restart rather than pretending recovery succeeded.

No P0 was established. Each finding above has a failing-before/passing-after regression. Worker/storage failures were injected at real service boundaries using temporary databases or controlled dependencies, not by corrupting the live server.

## Important findings — remaining

**P2 CONFIRMED — SQLite connections depend on cyclic GC for release.** `main_database.py:MainDatabaseConnection.__exit__`, `media_exports.py:MediaExportStore._connect`, `telemetry_store.py:TelemetryStore._connect`, `recording_process/index.py:RecordingIndexMixin._index_connection`, and the analogous jobs connection factory return SQLite connections used with transaction contexts. `with connection` commits/rolls back but does not close. Repeated production query/write calls therefore retain handles until Python GC collects connection cycles. With GC deliberately delayed, **300 temporary-database calls raised open FDs from 4 to 304; explicit collection returned them to 4**. This is avoidable FD/native-memory pressure, not evidence of monotonically growing memory under normal GC. A low FD limit, heavy query load or delayed collections could cause open-file errors. Proposed follow-up: explicitly close owned per-call connections after commit/rollback, accounting for reused connections and streaming cursors; test commit/rollback, cursor lifetimes, concurrent readers/writers and FD plateaus. Deferred because this audit prioritizes P1 defects and changing global connection semantics needs a separate call-site migration.

## Scope and remaining investigation

Repository-wide thread/task/queue and frontend timer/socket searches were followed into the owners above. Existing camera stress tests, durable outbox/admission tests, inference lifecycle tests and file-retention tests will be used rather than inventing replacement subsystems. Unbounded-looking caches inspected so far use TTLs, weak values or fixed histories; no claim is made that every allocation is leak-free. Live initial snapshot: 13 cameras healthy, detector ready, storage idle.

## Performance and resource conclusions

- The confirmed event-loop stall was removed from detector replay and blocking AI request setup. No claim of higher inference throughput is made; no comparable sustained-load benchmark was run.
- Camera frame capture/analysis coalesces pending frames rather than building an uptime-sized frame queue. Recorded tracking catchup buffers are bounded by `max_catchup_frames_per_tick`; these historical frames are intentional work, distinct from live stale-frame accumulation.
- Inference admission sheds optional work in favor of security work. Motion notifications and evidence indexing can accumulate **durable database work** while dependencies are unavailable even though in-memory wake queues are bounded. Disk/backlog growth under a multi-day outage remains a soak concern.
- Main DB writers share an RLock; recording index, job, telemetry and export databases have separate contention domains. Store-level locks plus the main writer can delay unrelated event/face/semantic writes during slow storage. No measured long-transaction bottleneck or missing-index defect was established in this pass.
- ONVIF retries use bounded backoff and cancellation; recorder/index/retention loops contain failures. Capture and inference expose frame age, replacements, queue/admission and inference timings. Frontend SSE subscriptions and media effects have disconnect/unmount cleanup; MSE input buffering has an 8MiB fallback limit.
- Existing process diagnostics already report RSS, threads and FDs. The owner-only socket now additionally reports task count and a 10ms event-loop scheduling probe; it starts no persistent timer or worker. The probe is an occasional sample, not a continuous loop-lag histogram or CPU profiler.

## Changes and validation

- AI ownership: cancellation regressions for assistant and motion audit, shutdown admission rejection, existing provider/error/assistant behavior tests.
- Export worker: lookup, cleanup, failure-persistence and cancellation-persistence faults, recovery without duplicate execution, and shutdown during failed lookup retry; 35 export-focused tests passed, including terminal cancellation persistence during shutdown.
- Face worker: startup/idle/post-job refill failures retain the pending flag and recover with stop-aware retry and rate-limited logging; 59 worker/face tests passed.
- Reload/manager/inference ownership: failed candidate cleanup cannot start recovery; failed previous/recovery generations remain owned; failed shutdown can be retried; 138 related tests and 2 subtests passed after the final inference cleanup correction.
- Detector replay: reproduced event-loop blocking and premature lease release, then passed both regressions and existing detection/depth response tests.
- Camera lifecycle stress now performs **100 start/stop cycles**, including concurrent status reads, late input rejection and worker teardown. These use controlled capture/ONVIF producers and real motion services; they do not demonstrate 100 hardware reconnects.
- Final full backend suite after all code corrections: **2,692 passed, 247 subtests passed**, 82.41 seconds. One existing Starlette/httpx deprecation warning.
- Changed Python unused imports/redefinitions/locals and whitespace checks passed. No frontend source changed; browser visual retesting was not needed for these fixes.
- The read-only soak collector completed a four-sample live trial and a six-sample post-restart trial, both with zero snapshot errors and owner-only output permissions; summary mode parsed the trial successfully. After restart, all 13 cameras were connected in every sample, asyncio task count stayed at 6, and sampled loop delay ranged from 0.100 to 1.045ms. The old main PID exited. These short samples verify instrumentation/startup, not leak freedom. The collector records counter-only `/proc` data, never process arguments/environment/FD targets, and omits snapshot logs.

## Remaining risks

A short audit cannot establish 30-day stability. Native OpenCV/OpenVINO/FFmpeg memory retention, GPU/driver allocations, actual camera firmware reconnects, filesystem stalls, multi-day durable-backlog growth and multi-client browser behavior still need workload testing. The observed live process tree varied with active work; that alone is not proof of a leak. No throughput optimization or mature-subsystem rewrite was made without measurements. Existing per-DB transaction latency and lock-retry metrics are incomplete; investigate them if soak samples show rising API latency or storage errors. Live fault injection and the full 12–24 hour soak were not run. One later full-suite run intermittently failed `test_product_update_start_checks_out_selected_branch` when Git switch exited 128 in a temporary checkout. The focused retry, 30 isolated repeats and 30 repeats with eight concurrent status pollers all passed. Root cause is **PLAUSIBLE BUT UNPROVEN** (status/check-out contention is a candidate); updater code was not changed on speculation. This remains a test/runtime investigation item, not a confirmed regression from these patches.

## Recommended 24-hour soak

After deploying/restarting this revision, run as the service owner or root from the repository:

```bash
python3 scripts/production-soak.py --hours 24 --interval 60 --output /var/tmp/survng-soak-24h.jsonl
python3 scripts/production-soak.py --summarize /var/tmp/survng-soak-24h.jsonl
```

Use a new filename for each run; the collector refuses overwrites and creates mode-0600 files. For 12 hours use `--hours 12` and halve the workload periods below. Stop with Ctrl-C if needed; completed lines remain valid. Snapshot failures are recorded while `/proc` sampling continues for the last verified process identity. Restarts are separated by instance ID, and summary trends exclude each process's first hour.

| Hours | Workload |
| --- | --- |
| 0–1 | Warm up all configured cameras and enabled models under normal recording/detection. Keep the ordinary camera/retention configuration. |
| 1–4 | Keep one desktop Live view and one second client open. Every 15 minutes use Timeline seek/playback, People and incident search; close both clients for five minutes at the end of each hour to compare idle resource counts. |
| 4–6 | On a designated test camera whose recording interruption is acceptable, perform ten disconnect/reconnect cycles: disable for 30 seconds, re-enable, and allow at least two minutes for recovery using Admin. Confirm the remaining cameras keep recording and their frame ages recover normally. Do not cycle the entire fleet. |
| 6–12 | Combine normal detections with repeated detector replay and a short recording export each hour. Stop the extra workload for five minutes hourly; queues should drain and worker/process counts should return near the warmed baseline. Use existing admission limits. |
| 12 | During an acceptable brief recording interruption, run `systemctl restart survng` once with playback active. Verify old service processes exit and the new snapshot reports healthy cameras. The collector stays running independently. |
| 12–24 | Repeat the normal mixed workload, including quiet and busy camera periods; leave the last hour at the same workload as the post-warmup baseline. |

Review the summary and raw per-camera samples, comparing equivalent workloads and each instance separately:

- Investigate RSS that rises across successive hourly idle windows without flattening (for example >200MiB or >10% above the warmed baseline); native caches can legitimately retain a plateau. Compare main-process and child totals separately; summed RSS includes shared pages.
- After clients close or camera cycles complete, sustained growth in FDs, tasks, threads or subprocesses is a failure signal. Inspect any zombie (`state: "Z"`) or old-generation process. Allow model startup and temporary export/decode workers to settle first.
- Flag three consecutive samples with frame age >5 seconds on a normally enabled camera, disconnected cameras that do not recover, or a stopped analysis worker when analysis is demanded. Inspect per-camera mailbox replacements alongside frame age: dropping obsolete work is expected under pressure; increasingly stale processing is not.
- Flag queue depth/refinement age that keeps increasing for ten minutes or does not drain within five minutes after extra load stops; compare inference latency with the same workload's baseline. Counters reset on generation change.
- Investigate repeated socket timeouts, snapshot latency >1 second, or sampled event-loop delay >100ms under ordinary load. These can expose a deadlock or storage/CPU stall, but occasional probes can miss short stalls.
- Review Admin telemetry and its redacted diagnostics for worker failures, DB busy/I/O errors and reconnect bursts at the recorded times. Do not infer DB lock latency from CPU alone. Confirm retention keeps disk use within policy and export temporary files disappear after their cleanup window.

Success means stable workload-relative plateaus, recovering cameras, draining queues and bounded latency—not merely that the process stayed alive. Retain the JSONL with workload/restart timestamps for diagnosis.



## Commits

- `a5d2e83`: AI request ownership and cancellation.
- `dfe04f6`: export bookkeeping/cleanup recovery.
- `9ce2767`: face queue refill recovery.
- `83da2a8`: manager/reload/inference cleanup ownership and 100-cycle stress.
- `2398fa9`: detector replay event-loop and lease ownership.
- `2ba33b6`: owner-local task/loop metrics and read-only soak collector.

The final code diff was reviewed after committing. No further audit-introduced defect was identified; the explicit P2 and unproven risks above remain.
