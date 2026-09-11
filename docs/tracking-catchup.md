# Recorded tracking catch-up

Tracking resumes from the last successfully analyzed media timestamp, in bounded
batches. A delayed incident is not rejected just because its first frame is more
than 12 seconds behind live. Tracking queries the recorder's maintained index;
it does not synchronously discover recordings across several calendar days.

Each batch bounds both decoding and inference, including the valid one-frame
batch setting. Providers receive the continuity cursor separately from the next
sampling timestamp, so recorder/capture boundaries between samples are checked.
Live fallback checks continuity through its own timestamp with a point read.
Deferred inference retains and retries the same recorded cursor or pending live
frame, and missing recording coverage gets a bounded opportunity to become
available. A live or recorded frame more than the object lost timeout ahead of
the cursor cannot silently age out the existing track. Recorder/capture
continuity boundaries terminate incomplete coverage after their readable prefix.

The configured `max_session_seconds` bounds the media window from the selected
initial detection frame. It also remains the independent processing-time safety
budget after capacity admission. This deadline is cooperative: blocking decoding,
inference, or persistence can overrun it. Reaching the media horizon or observing object
expiry completes tracking; stopping, missing media, or exhausting processing time
reports interruption. Cancellation and budget are checked after provider work,
and known continuity interruptions take precedence over completion. Only after
checking the remaining tail for available samples and boundaries does the final
horizon permit 1.5 sample intervals of timestamp jitter (capped to half the
window); this tolerance does not widen interior-gap handling. `analyzed_through` and the compatible `updated_at` field retain the exact
last analyzed media timestamp. `persisted_at` records wall-clock persistence time.
Processing latency therefore does not extend the incident replay duration.

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
