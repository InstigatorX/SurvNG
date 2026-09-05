> Completed capture: 2026-09-05 05:45:44–06:30:44 UTC (45 minutes).
> See [Final Astra assessment](FINAL-ASTRA-ASSESSMENT.md) for conclusions and limits.
> Referenced raw captures, JSON evidence and collector helpers remain host-local
> under `/root/survng-measurements/`; they are not published in this documentation PR.

# Final Astra assessment — completed 45-minute capture

Reviewed 2026-09-05 after capture completion: final `LIVE-CAPTURE-REPORT.md`/`live-comparison.json` generated at 06:31:27.946 UTC; the reporter/accounting source; original aligned pre/post evidence; raw resource/application/telemetry samples; all three collector completion files; and `recording-checks.json`. Source mechanisms and precise code references are in `ASTRA-SOURCE-REVIEW.md`. No runtime, settings, application code, packages, debug session, inference replay or profiler was changed by this review. A separate preliminary documentation PR does not alter the running service.

## Finding

The elevated CPU signal persists hours after the restart. The completed capture averages **3.584 service CPU cores** over 539 measured intervals/2695.836 seconds; 100% of one CPU core corresponds to 1.0 here. Main-process CPU averages 2.658 cores. This establishes observed resource consumption, not the cause or detection quality.

For the same capture-relative selection, seconds 360–1740, the comparison is:

| Capture | Actual covered seconds | Service cores | Main cores | Capture-worker cores | Recorder-worker cores |
|---|---:|---:|---:|---:|---:|
| Original pre-deployment | 1375.0 | 2.417 | 1.732 | 0.491 | 0.077 |
| Original post-deployment | 1376.3 | 3.143 | 2.439 | 0.496 | 0.079 |
| New capture, same post-deployment process hours later | 1375.0 | 3.527 | 2.607 | 0.609 | 0.082 |

Each selection contains 275 intervals. Slight covered-duration differences come from actual polling times; the rates use their own elapsed denominators. This is a shared measurement window, **not matched real-world activity** or an identical process-age window.

Original pre→post service CPU rose 30.1% in this aligned selection. Approximately 97% of that increment lies in the main process. Original-post→new service CPU is another 12.2% higher; main-process CPU rises 6.9%, and capture/observed FFmpeg work and unattributed residual also increase. The 97% figure must not be transferred to this later comparison. For that later pair, about 44% of the additional service CPU lies in the main process. The residual remains unattributed; short-lived children and asynchronous process/cgroup reads prevent naming it as decoder work. go2rtc is a separately accounted dependency: 0.216→0.225 cores in the shared original-post/new window, not part of the service total.

## Process age, stabilization and observed health

The same main instance and the same two object-worker PID/start-time identities appear throughout the new saved samples. The first application snapshot reports process age 9797.5 seconds (163.3 minutes), and the last reports 12413.1 seconds (206.9 minutes). The reporter's resource-time extrapolation extends to approximately 208.2 minutes. No arbitrary age-300 cutoff was applied.

New five-minute service CPU means are 3.452, 3.620, 3.784, 3.905, 3.419, 3.172, 3.567, 3.935 and 3.303 cores (the last bin is 4.9 minutes). This does not show a sustained decline toward the original pre-deployment level. It also does not establish a converged CPU plateau: activity is unmarked, the bins fluctuate, and scene-learning indicators are absent. Five-minute cgroup-memory means fluctuate from 9.018 GiB initially through 9.822 GiB to 8.065 GiB in the last bin. That is not evidence of a memory leak or proof of completed learning. Eleven reliable spatial-alignment counts and a constant recorded-decode memory budget of 2,244,160,512 bytes are separate runtime states; neither measures EMA scene convergence.

All 67 application snapshots report detector readiness. All 13 cameras are enabled, running and connected at those samples. Eleven retain detection/recording enabled and recording=true; sherry-garage and steve-garage retain detection/recording disabled. There are no observed changes to those sampled flag combinations. Maximum sampled detector queue depth is **1**, not zero; recorded-decode waiting is zero at sampled instants. These gauges cannot exclude short bursts between polls.

With stable sampled main/object-worker identities and no observed counter decreases, detector completions rise 4885→6311, a delta of 1426 across application endpoint coverage. Failed-inference counters remain zero. These completions are not distinct scene events, refinement attempts, recall observations, or a pure workload-equivalence measure. The persisted per-camera summary has 40 qualifying adjacent minute intervals: sherry-garage preparation is approximately 4.370 fps, steve-garage 3.997 fps, with no object-check admissions on either. Four superseded preparation frames are observed in those summarized rows (one back-left, three boiler), with zero recorded object-check failures. Neither supersession nor absence of admissions establishes a missed subject.

## Coverage and measurement limits

Primary capture completed normally at **06:30:44.378 UTC**, elapsed 2700.346 seconds, `stopped_early=false`; supplement and dependency collectors completed shortly afterward. There are 540 resource snapshots/539 intervals and 67 successful application snapshots, with no application error rows. All 2695.836 covered resource seconds remain **unmarked**: there are no completed scene observations or detection outcomes. No quiet/normal/busy ground truth, recall, missed-event rate or day/night quality comparison can be inferred.

The unchanged application sampler backed off through scheduled intervals of 15, 30, 60 and finally 120 seconds. Its last 15 actual intervals are approximately 60 seconds. Request duration ranges 39.1–570.4 ms, median 58.4 ms; the last 271.1 ms request schedules 120 seconds. The final application sample at 06:29:19.839 UTC leaves about **84.5 seconds of unobserved application-state tail** before capture completion. Do not describe the completed run as uniformly sampled every 15 or 30 seconds. Resource sampling continues through approximately 06:30:40 UTC. Minute telemetry has its separate lag/gap/whole-interval selection, and missing tail minutes are not zero activity.

The start metadata records unchanged persisted configuration fingerprint and checkout evidence. It does not attest every live override or loaded model/build identity. The source report and earlier configuration comparison retain those limitations. Primary collector self CPU is 3.016 seconds and supplement self CPU 1.17 seconds over this run; these do not include dependency collector CPU or service-side snapshot construction/IPC/database cost. Parent final preservation/overhead validation may add the complete collector total; do not present partial self-CPU accounting as total measurement overhead.

## Verified mechanisms and remaining hypotheses

Source verification establishes two added-work mechanisms: the retired pre-pipeline temporal gate now permits actual qualification/learning on formerly skipped quiet cycles, and the continuous history now has four frames/three transitions rather than three frames/two transitions. Preparation already occurred before the old gate. Background statistical work, thresholding, morphology, connected components, tracking/scoring and policy/result handling are now reached. Cached blur avoids re-preprocessing all historical frames.

Historical transitions still compute robust statistics and update the invocation's noise accumulator before the stale-transition guard; they skip background/persistence-image updates. The accumulator can later be persisted. Therefore eliminating repeated historical work requires preserving or deliberately testing those semantics. Removing a transition or restoring the old quiet gate is not an established safe optimization. The committed stationary-foreground regression demonstrates why quiet image-model updates matter; it was inspected, not run in this read-only investigation.

Persisted illumination filtering is **false**, with null inherited camera overrides. Expensive illumination work is therefore not expected under the captured persisted configuration and is not a leading explanation for this run; its live effective flag remains unexported. Source cost is not measured stage CPU. No saved diagnostic samples exist in either original measurement window to recover the omitted stage timers. Main-process localization makes qualification a reasonable hypothesis to measure, not a verified attribution of the CPU increment. Greater detector/decoder/capture work in the newer unmarked window also prevents calling process age the sole changed variable.

## Disabled-camera necessity and smallest justified next change

The garage cameras continue preparation/qualification because admission depends on camera_rescue policy, custom frame-observation capability or debug demand, and does not check detection eligibility. Their actual current consumers include image-model learning, qualification/evidence rings and policy bookkeeping. Optional consumers include debug visualization and accepted route evidence. The default frame observation graph is ONVIF-only, hence does not need the gray-frame observation call. Live viewing, recording, raw tracking history and stream alignment have separate capture paths and do not require these motion derivatives.

These facts justify investigating disabled-camera demand; they do not establish measured savings or a safe blanket skip. An active debug lease is unverified; pending intents/history and re-enable behavior must be handled. Disabled EMA conditioners clear candidates and do not advance their readiness observations, even while the image model is maintained. Pausing qualification is consequently a lifecycle-policy change requiring explicit demand, deferred-slot cancellation and defined warmup/history behavior on re-enable.

The smallest justified next change is a narrow owner-only numeric observability extension for **already-collected** prep and qualification counts/totals, completed continuous-result counts, per-stage calls/failures/raw totals, analysis limiter waits/deferrals, and effective mode/debug/demand state. Reuse the already-built camera snapshot and existing cached refinement gauges; do not add per-frame imaging, inference or another database scan. This can discriminate prep cost from qualification-stage and policy cost before changing detection behavior. The source report's engineering estimate of 1–10 ms incremental projection/serialization CPU per fleet snapshot remains unmeasured and excludes existing status collection. Preserve/adapt backoff based on measured request cost; this run already needed backoff to 120 seconds.

True durable event latency and refinement retry attribution remain separate missing measurements. Existing workflow completion timing occurs before event persistence, and a durable job claim can represent checkpoint completion rather than new inference. Do not substitute inference compute, model completion count or `created_at` for the absent end-to-end timing.

## Recording verification

The post-capture checker finished at 06:31 UTC with **zero eligible marked scenarios**. Its recorded result states that no recording files were read or decoded. Recording continuity and decodability are therefore **unverified**, not passed or pending decoding. The reviewed checker enforces bounded finalized/validated/playable local-file selection, one low-priority decoder, quarter-realtime pacing and a total budget if scenarios exist; those checks did not execute here. The result's scope text retains an obsolete “per scenario/source” phrase, but the reviewed implementation's actual selection cap is three per scenario; with zero scenarios this wording has no execution effect. Parent should correct the description in any final published summary.

This assessment accepts the measurement/report as evidence of persistent adverse resource use and limited sampled operational health. It does not approve a performance fix, claim completed scene learning, or certify detection/recording quality.
