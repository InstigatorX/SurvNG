# Local inference latency investigation

The local native deployment requested five live inference frames per second on
13 cameras. A 30-second baseline completed 613 live results (about 20/sec),
while a nearby GPU sample measured 94.8% render occupancy. Recorded inference
averaged 125.5 ms at the end of that baseline. Its displayed metric excludes
the continuous DL Streamer workload. Comparisons with v1.2 also differ in scope:
initial event detection used the Python inference pool in that branch.

Both CPU and integrated GPU share the host package power budget. The user
raised PL1 from 45 W to 50 W before this investigation. No further power-limit
changes were made here. Workload and power contention vary over time, so these
are operational samples, not controlled throughput or recognition benchmarks.

## Retained changes

The temporary local 1 FPS experiment was rejected: lower kernel latency did not
establish acceptable event latency or coverage. The server was restored to 5 FPS
before the qualification-driven implementation described below. No reduced-FPS
production setting is retained.

The metadata appsink now uses `async=false`: sparse inference output must not
hold up qualifier/video startup. A real GStreamer control reproduced no qualifier
frame when a fully withheld metadata branch waited for preroll, and immediate
frame delivery when async preroll was disabled. See the GStreamer
[BaseSink documentation](https://gstreamer.freedesktop.org/documentation/base/gstbasesink.html).
The graph regression checks the production property, and the native smoke script
now includes the sparse-branch control and an optional `--shared-device GPU`.

## Rejected experiment

A custom global admission scheduler was implemented and tested, then removed.
It used one outstanding live request, fair camera submission counts and a 70%
request-time duty budget. Unit and native synthetic checks passed. Its initial
live trial reset the supervisor during startup and was rolled back. Fixing the
sparse metadata preroll dependency allowed a second trial to remain connected,
but longer observation showed worse detection freshness and higher recorded
inference latency; an observability request also timed out. It was rolled back
again. Neither the scheduler nor its configuration field remains in the code.

An isolated test of that rejected candidate used 13 synthetic HTTP MPEG-TS
streams, VA H.264 decoding, VAMemory inference and the deployed model. All streams
produced results and the process exited cleanly after main-stream removal. That
was useful lifecycle evidence, but did not predict acceptable production
performance. The live trials were decisive.

## Validation

- Retained source change: affected suites passed, 106 tests and one skipped.
- Native GPU smoke passed with two live streams and an independent main stream.
  The generated model checks deterministic result plumbing, not recognition.
- The sparse metadata control reproduced the before/after startup behavior.

- Final full suite: **2,675 passed, 272 subtests passed, one skipped** in
  93.38 seconds. One existing Starlette/httpx deprecation warning.
- Final native sparse-branch smoke helper passed from the retained script.
- `git diff --check` passed.

## Historical observation of the rejected 1 FPS experiment

The earlier code and temporary local 1 FPS setting were loaded by the service restart at
17:12:02 UTC. After all tests and isolated GPU benchmarks stopped, the final
observation ran from **17:24:09 to 17:27:13 UTC** (184 seconds, 37 snapshots).

- All 13 cameras remained connected. Every camera produced 183–184 additional
  detection snapshots, with zero capture-session resets.
- 81 additional recorded inferences completed; the detector reported no failed
  inferences. There were no observability request failures or warnings/errors
  in the available runtime log window.
- Median of fresh sampled last-inference readings: **46.8 ms**, versus
  **113.7 ms** in the earlier baseline. There were 15 distinct readings in the
  final window and five in the baseline; these are sampled values, not the
  distribution of every inference.
- The displayed rolling average was **80.75 ms**, versus **125.5 ms** at the
  end of the baseline. It retains each worker's last 100 calls, including earlier
  expensive startup/test calls until displaced.
- Median observed live detection PTS lag: **0.599 seconds**, versus **1.2
  seconds** in the baseline. Freshness checks were not loosened.

These are different operational time windows with varying recorded-event load,
not a controlled comparison or proof of unchanged recognition recall. GPU render
utilization remained high in the earlier 1 FPS sample; this change should not be
presented as eliminating GPU saturation. No alternative preprocessing backend,
model-priority policy, model replacement or additional power adjustment was
retained.

## Qualification-driven implementation

Production GStreamer capture now emits color frames without a `gvadetect`
branch. Qualification selects a timestamped evidence frame and submits exactly
those pixels to `detect_initial` in the existing inference supervisor. Initial
requests retain priority over refinement and bounded tracking work. Recording,
qualification cadence, tracking rate/capacity, model and power settings are
unchanged. The native continuous-detection path remains available to explicit
integrations and its contract tests.

Initial results retain the selected frame sequence, capture/lifecycle generation,
source session and source PTS. Before returning a usable initial result, the
caller checks elapsed freshness and current capture/lifecycle/session identity.
Expired or invalidated results leave recorded refinement pending. A newer frame
in the same session does not replace the selected pixels.

Tracking frames explicitly distinguish demand-driven pixels from missing native
metadata. The former use the existing tracking workload, including deferral and
bounded retry of the retained frame; the latter remain unknown evidence. The
bounded history admits uninferred color frames without polling for nonexistent
native results, while retaining its timestamp and continuity boundaries.

### Same-footage native app replay

Baseline HEAD `c3996fe` and the candidate each ran for 140 seconds against a
loopback RTSP replay of the retained Downstairs live/main footage, at the same
5 FPS capture setting. Each app had its own configuration, recordings, database,
cache and owner-only status socket; ONVIF and external integrations were disabled.
The deployed model ran on GPU. Production continued running during both trials,
so these sequential observations are not a controlled accelerator benchmark.
The footage contains a roughly 20-second person appearance, repeated twice;
this does not measure recall of subsecond appearances or diverse scenes.

| Observation | Continuous baseline | Qualification-driven candidate |
| --- | --- | --- |
| Walkthrough passes detected | 2/2 | 2/2 |
| Qualified evidence to initial result, pass 1 | 537.3 ms | 633.2 ms |
| Qualified evidence to initial result, pass 2 | 530.8 ms | 608.7 ms |
| Completed tracking sessions | 1 | 2 |
| Frames in completed sessions | 36 | 41, 37 |
| Reported coverage gaps in completed sessions | 0 | 0, 0 |

Latency here includes qualification/admission delay plus initial processing,
measured against the event's evidence timestamp. It is not time from a manually
annotated first visible pixel. Demand-driven inference added 78–96 ms in this
small replay compared with already-computed boxes; **this trial does not establish
lower event latency**. Both candidate tracking sessions completed their configured
windows. A missing baseline tracking session is not counted as gap-free coverage.
Server-wide contention must be assessed separately after removing continuous
inference from all cameras.

### Implementation validation

- Full suite: **2,684 passed, 272 subtests passed, one skipped** in 68.64 seconds.
  One existing Starlette/httpx deprecation warning.
- Native smoke exited successfully, including GPU shared inference compatibility
  and a new frames-only live case: 16 BGR frames at 5 FPS, zero native detection
  messages. Sparse metadata startup and existing output/NMS checks also passed.
- Both isolated replays ended with the detector ready, camera connected and zero
  reported failed inferences.
- Regression coverage includes exact evidence pixels, later frames in the same
  session, delayed results, disconnect/reconnect and lifecycle invalidation,
  tracking workload selection, retained-frame retry after deferral, and bridging
  recorded gaps using uninferred live history.
- `git diff --check` passed. Temporary replay instrumentation is outside the
  repository and is not loaded by the production service.

The first rollout exposed an incidental timeout change: disabling native inference
also reduced first-frame startup from 30 seconds to 3 seconds. Steve-garage then
repeatedly timed out while other cameras connected. Native capture now keeps a
30-second minimum startup allowance in both parent and child, independently of
whether a model is attached; explicit caller deadlines still take precedence.
The follow-up capture/lifecycle suites passed **102 tests, one skipped**.

### Final native-server observation

After the startup-timeout correction and restart, all 13 cameras stayed connected
through the 90-second observation. Capture-session reset deltas were zero, both
object workers were alive, and the final pending request count was zero. The
available runtime log window contained no capture/DL Streamer warnings. The native
command used `--no-detect` and `--fps 5.000000`.

Sampled GPU render occupancy was **93.4% before** (12 samples) and **0.36% after**
(29 samples, all cameras online). These are different operational windows, not
matched event-load benchmarks. The final detector rolling average was
**45.55 ms** with **4** total calls since restart
and zero reported inference failures. The small call count is insufficient for a
latency distribution; the continuous idle-camera model workload has been removed.
Host package power remained around 55 W during the final sample, so lower render
usage must not be presented as resolving the host's shared power-limit condition.

A pre-existing `back-middle` refinement-completion error remains:
`UNIQUE constraint failed: motion_audits.event_id`. It was observed before these
changes and still retries. This rollout does not claim to fix that database issue.
Temporary replay hooks and disposable source/media/cache/database directories
were removed after extracting comparison evidence.
