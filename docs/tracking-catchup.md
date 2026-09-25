# Recorded tracking catch-up

Tracking resumes from the last successfully analyzed media timestamp, in bounded
batches. A delayed incident is not rejected just because its first frame is more
than 12 seconds behind live. Tracking queries the recorder's maintained index;
it does not synchronously discover recordings across several calendar days.

Each batch bounds both decoding and inference, including the valid one-frame
batch setting. Providers receive the continuity cursor separately from the next
sampling timestamp, so recorder/capture boundaries between samples are checked.
Live fallback checks continuity through its own timestamp with a point read.
Deferred inference retains and retries the exact recorded frame and the remaining
decoded batch, or the pending live frame. Each session retains at most
`max_catchup_frames_per_tick` recorded samples. Provider iterators are closed
after bounded materialization, so retries retain images without holding decoder
resources. Successful processing advances the cursor and releases that sample;
a missing-media gap discards the pending batch so newly readable coverage can be
queried again. Recording boundaries remain attached to their readable prefix,
including when a provider returns more samples than the retained-frame cap.
Missing recording coverage gets a bounded opportunity to become available. A live
or recorded frame more than the object lost timeout ahead of the cursor cannot silently age out the existing track. Recorder/capture
continuity boundaries terminate incomplete coverage after their readable prefix.

For legacy sessions without an explicit recorded window, `max_session_seconds`
bounds the media window from the selected initial detection frame and remains
the independent processing-time safety budget after capacity admission.
Production recorded-window sessions use the policy described below. This deadline is cooperative: blocking decoding,
inference, or persistence can overrun it. Reaching the media horizon or observing object
expiry completes tracking; stopping, missing media, or exhausting processing time
reports interruption. Cancellation and budget are checked after provider work,
and known continuity interruptions take precedence over completion. Only after
checking the remaining tail for available samples and boundaries does the final
horizon permit 1.5 sample intervals of timestamp jitter (capped to half the
window); this tolerance does not widen interior-gap handling. `analyzed_through` and the compatible `updated_at` field retain the exact
last analyzed media timestamp. `persisted_at` records wall-clock persistence time.
Processing latency therefore does not extend the incident replay duration.

Stored tracking `processing` diagnostics include decode-batch calls, buffered
frame count, recorded-frame retry count, actual inference-deferral count, decode
time, object-detection request time, tracker-update time (including synchronous
appearance matching), and elapsed session processing time. Timings are in
milliseconds.
Counters are per session and cumulative, not per-frame logs. Elapsed time also
includes retry waits and other session work; the named timers are not an
exhaustive breakdown. These snapshots are taken before their persistence call.
An outstanding deferred sample remains `inference_unavailable` at the processing
deadline. Sustained higher-priority inference can still exhaust the unchanged
budget; retaining the batch prevents repeated decoding from adding to that load.

This is resumable catch-up within a running session, not durable recovery after
a service restart. Replacement incidents still follow the existing per-camera
lifecycle. Initial detection queue priority and latency are unchanged. A longer
replay does not imply a longer configured tracking window.

## Validation on September 11, 2026

A separate process used the stored detector model on CPU, the actual recordings,
the revised frame timeline/session, and in-memory event updates. It did not
modify either original incident. Optional appearance/ReID encoders were not
instantiated, so this validates frame delivery and geometry association, not
production-load performance or appearance recovery.

| Incident | Original stored tracking | Separate validation |
| --- | --- | --- |
| Back Left, event 68701 | One seed observation; zero follow-up frames | 23 analyzed frames, 14 car observations; natural expiry after analyzed frames no longer detected it; 5.9 seconds processing |
| Front Side, event 67993 | 17 observations across 5.14 seconds; interrupted at a recording gap | 39 analyzed frames, 40 person observations across 14.883 seconds; completed the configured 15-second window; 10.1 seconds processing |

Regression coverage includes a 50-second delayed handoff over multiple bounded
batches, deferred inference beyond the media-settle timeout, irregular timestamps,
interior gaps, continuity boundaries after readable prefixes, processing-budget
exhaustion, explicit stop, and a sub-sample tail at the media horizon.

## Astra HIGH review

An independent `gpt-6-astra` review with `high` reasoning found and verified fixes
for one-frame batch reads, boundaries between batches and before live fallback,
unread tails, retained deferred live frames, cancellation during provider calls,
and boundary priority at the media horizon. The final revised patch was approved
for this scope with no remaining blockers. The chronology/coverage suite includes
18 regressions, with actual timeline integration for the relevant boundary cases.
The measurements above were refreshed after the batching and tail corrections.

## Full recorded incident window

Production sessions receive an explicit window from the current incident clip
settings and trigger evidence. The start includes the motion persistence leading
up to the trigger plus configured pre-roll; motion lookback is capped at 120
seconds. The end includes `max_session_seconds` after the trigger plus configured
post-roll. The actual representative-frame timestamp is included if it falls
outside these bounds. These requested bounds are stored as `window_start_epoch`
and `window_end_epoch` and used directly for incident playback, without adding
post-roll a second time. Playback begins at the window start rather than seeking
to the representative event.

This pass runs in the existing background tracking worker after confirmation;
it does not delay the initial incident notification. It begins with an empty
tracker and processes frames chronologically. The confirmed snapshot detections
join the tracker at their actual timestamp. Their resulting IDs are matched back
to the original snapshot objects by exact label and box, so an earlier vehicle
cannot silently take the person's displayed ID. The pass continues through
empty frames and expired tracks to discover later activity within the window.
A boundary or unavailable recording remains an incomplete result, not permission
to jump over missing footage.

Recorded-window sessions use a separate cooperative processing budget,
`recorded_processing_budget_seconds` (default 60 seconds), while legacy sessions
without a window retain their existing budget. Recorded batches and inference
priorities remain bounded and unchanged. The worker is not a durable restart
queue: a restart or replacement can still interrupt a pass. The requested window,
`analyzed_from`, and `analyzed_through` distinguish requested footage from actual
coverage. No analyzed-through timestamp is claimed before the first frame.
The UI displays progress and marks footage outside the analyzed range, and does
not hold estimated boxes after their last observation in a full-window replay.
