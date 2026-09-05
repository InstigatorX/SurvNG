> Completed measurement of PR #170, published after execution.
> Raw captures, recordings, decoded arrays, worker logs and full host-local provenance
> remain under `/root/survng-measurements/pr170-20260905T180430Z/`.
> Statements about no GitHub writes refer to the measurement, before this documentation PR.

# Astra method review before execution

Reviewed source only during the live measurement; no pipeline or decoder was executed before the completion supervisor releases replay.

PR170 changes only exact-zero handling in `morphology_motion_masks` and `ConnectedComponentBlobStage`, plus the focused regression suite. The morphology path returns independently owned copies for supported empty masks; the component path emits the same empty `MotionFrameBlobs` observation that native labeling would produce. Preserving those observations matters because continuous tracking/scoring must still process misses and time progression. Nonzero masks retain the native operations and incur the added morphology zero check. That scan can offset savings in predominantly nonempty scenes.

The replay follows the production qualification contract in `MotionAnalysisService._preprocess_frame`, `analyze_continuous`, `MotionQualificationService.continuous_primary_due`, and `run_pipeline`: cached resize/gray/blur; four overlapping frame windows; due cadence from effective settings; one persistent pipeline runtime; no per-window reset; identical preroll and state advancement for both versions. Full qualification CPU time and elapsed latency are measured directly with process CPU and monotonic wall clocks. They exclude already-cached preprocessing and production scheduling/bookkeeping.

Separate diagnostic passes count exact nonzero pixels at the actual morphology input after exclusion, and at the component input after its inclusion-mask application. These counts classify timed invocations afterward; diagnostic overhead cannot contaminate the timed pairs. Full outputs and every semantic runtime field, including background arrays, noise and threshold accumulators, historical statistics, persistent change age, zone cache, tracks, and scoring history, are fingerprinted at every diagnostic invocation. Locks and timing metrics are excluded.

The source versions, installed dependencies, default OpenCV settings, service affinity and allowlisted environment are recorded and checked across workers. Read-only native-library mappings strengthen dependency provenance. In-process CV thread settings and all effective zone/options fields are not exposed by the socket; those limits are explicit. Production learned state and original capture/admission timestamps cannot be reconstructed from recordings. This is a controlled recorded-footage qualification comparison, not an exact replay of the live service or a recall/continuity evaluation.

The runner enforces a single 900-second budget including source extraction, selection, decoding, all pairs, diagnostics and regression checks. Recording queries are bounded by camera/source/time, a 1500-row limit and a 2-second query deadline. Only validated/playable, stable finalized local MP4s inside configured recording roots are used. Decode happens once per selected segment, and immutable shared input arrays are reused. Material gaps or partial clips are rejected. Missing footage triggers the existing focused regression suite when the remaining budget permits, with an explicit gap instead of fabricated timing results.

The parent live collection keeps application polling independent of 1-second resource sampling and adapts application cadence to actual request cost. Stage elapsed counters are differenced only for stable process/component owners and generations; they must not be labeled CPU shares or summed with overlapping qualification/worker counters. The parent applied finite-value, clock-tick and missing-counter rejection checks requested during review.

Final acceptance and the recommendation will be based on actual saved results after execution. Component-only performance cannot be established from the bundled before/after experiment without an additional ablation; per-stage elapsed changes are supporting observations, not stage CPU measurements.
