# Deployment and effective-setting comparison

Supporting raw captures, JSON evidence, collector sources, and metric definitions referenced below are retained on the measurement host under `/root/survng-measurements/`; they are not included in this documentation-only change.

The earlier capture is preserved under `pre-deployment/capture-20260905T0210Z`,
outside temporary storage. `preservation.json` verifies every preserved raw file,
collector source file, metric definition and checklist against the originals.
Neither capture is overwritten. The launcher log is operational, not part of the
frozen measurement manifest. Collection code is byte-identical in both runs.

| Property | Earlier | Post-deployment |
|---|---|---|
| Inferred loaded commit | 74297328e58db72e0becf0407e6cd2f649e5cb07 | 69c50ba5ef76b6fc4ffc7630f6363c6a8f6b9bf0 |
| Main PID | 3072885 | 3762068 |
| Service start, September 4 EDT | 14:07:07 | 23:02:25 |
| Capture start, September 4 EDT | 22:04:58 | 23:04:18 |
| Process age at first snapshot | Approximately 7 h 58 m | 112.2 seconds |
| Working directory / editor checkout | /root/SurvNG | /root/SurvNG |
| Executable | /usr/bin/python3.12 via .venv uvicorn | Recorded in post metadata; same resolved executable |

The post checkout was pulled at 23:01:46 EDT, before service startup, and was clean
at inspection. Both commit identities are high-confidence inferences, not runtime
build-ID attestations. PRs #150, #152, #153, #154, #155 and #151 are ancestors of
the new checkout. It also contains #157 (independent route deliveries), #158
(refinement recovery/checkpoint completion), and #159 (healthy-worker failover),
plus #156 documentation. Performance changes cannot be assigned solely to the six
original PRs. Exact ancestry evidence is in `deployment-comparison.json`.

## Persisted and exported settings

The full persisted configuration file is byte-identical, including fields omitted
from the safe export. Its SHA-256 in both captures is:
`0444474a52f5eaf09e2330c2bdb338640774a134caf3cd633bed9acb0f4fa6b0`.
The file still predates both startups. No credentials or full configuration copy
were saved. Identical file hashes prove persisted equality, not all in-memory
override or environment equality.

Exported runtime settings are equal: 13 enabled/connected cameras; 11 with
detection and recording enabled; `sherry-garage` and `steve-garage` with both
disabled. OpenVINO/GPU enabled, two configured object workers; tracking baseline
3, adaptive burst enabled with limit 5, capacity wait 8 s, deferred ReID enabled;
recorded-decode process limit 3. Flags can change during capture; raw snapshots
preserve each observation rather than assuming permanent equivalence.

Persisted settings resolve the same unless code behavior changed below: inherited
camera_rescue, balanced sensitivity/stationary tolerance, 640-pixel maximum
analysis dimension, 5 fps preparation and 2 fps background qualification,
2 concurrent analysis slots, 4 concurrent refinements, model basename best.xml.
The socket omits these full effective settings and loaded model hash; the baseline
did not capture model-file content, so exact loaded model equivalence is unproven.
The global qualification/observation/fusion stage lists are empty, using defaults;
camera stage overrides are null, inheriting the global selection.

The config schema diff introduces no new scalar field or changed scalar default;
its only change is a compatibility comment. Source tracing identifies these
effective behavior/default changes despite the unchanged JSON:

| Setting or policy | Before | After | Interpretation |
|---|---|---|---|
| motion_qualification.temporal_filter_threshold=0.005 | Used before qualification to skip low-pixel-change scenes | Still accepted/validated in config, but no longer consumed by MotionAnalysisService | Newly retired functional setting. Do not claim identical effective filtering merely because JSON matches. Quiet scenes now reach the learning pipeline. |
| temporal_filter_skips telemetry | Count of executions skipped by that gate | Retained for compatibility, remains zero because the gate was removed | A drop is a semantic change, not evidence of faster execution or fewer misses. |
| Default adaptive scorer stationary_displacement_ratio/path_ratio | Default option entries 0.01/0.025; process used the selected stationary policy | Default option entries removed; absent means selected policy, explicit custom options now override it | For this unchanged balanced/no-custom configuration, effective ratios remain 0.01/0.025. No numerical tuning change should be claimed here. |
| Continuous color/processed evidence window | Last 3 frames | Last 4 frames | Code-owned default; slightly different storage/compute context, not a user config edit. |
| Classic continuous scorer insufficient_frames | Could fail open | Rejected while waiting for a scoreable window | Startup behavior and trigger volume may differ; not a measured recall improvement by itself. |
| Checkpointed refinement completion retention | Subject to ordinary freshness expiry | Separate 24-hour completion horizon | Code-owned durability policy. Bookkeeping retries must not be counted as new inference. Raw inference freshness remains 20 s for probe jobs / 60 s for event jobs. |
| Ledger I/O recovery | Exceptions could stop refinement processing | New 2-second store retry interval for retryable errors | Additional code-owned recovery policy. Completion retry spacing (2 s) and numeric maximum attempts (2147483647) already existed; they are not newly changed settings. |

The MOG2 compatibility rejection already existed before these fixes; it is not a
newly retired setting in this deployment. Config examples now omit the temporal
threshold, but the live configuration was not rewritten.

## Adaptive state is not a configuration edit

At the first snapshots, recorded-decode memory budget changed from 2,244,160,512
to 2,491,416,576 bytes; estimated per-process memory changed from 748,053,504 to
830,472,192 bytes. The new snapshot had no observed frame bytes yet, whereas the
old process had learned 34,002,432 bytes. Treat these as cold/learned runtime budget
state, not a changed configured process limit. Later snapshots show its evolution.
Alignment confidence/transforms, counters, readiness and queue occupancy also reset
or evolve after restart and must not be listed as user configuration differences.

All identifiable persisted/exported differences and changed default semantics are
recorded above. A claim that *all* effective runtime configuration is proven equal
would exceed the available baseline: its full environment, model content and
unexported overrides were never saved. No attempt was made to recover credentials
or modify observability to fill that gap.

## Measurement comparability

The local socket serializer, runtime monitor, persisted telemetry store, camera
status assembly, detector statistics and supervisor aggregation are unchanged
between commits. The original METRICS.md remains authoritative, with the retired
temporal counter caveat above. Episode decisions themselves were fixed, so their
counts cannot serve as independent scene ground truth.

Same collection policy: processes every ~5 s; application snapshots initially
every ~15 s with the same >250 ms adaptive backoff; DRM every ~15 s; indexed saved
telemetry every ~60 s. The old capture backed off to 30 s. New actual intervals
are recorded rather than forcibly matched to the old backoff outcome. GPU/go2rtc
sampling began later in the old capture; comparisons must retain their shorter
coverage. One-minute database rows are estimates, not measured per-frame latency.
