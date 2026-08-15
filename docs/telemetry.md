# Telemetry architecture

SurvNG separates always-on operational telemetry from temporary diagnostics.
Operational telemetry answers whether the service, cameras, detector, motion
analysis, tracking, database, and storage are healthy. It must remain compact,
bounded, and inexpensive enough to leave enabled in production.

Operational samples are summarized into one-minute intervals, retained at
one-minute resolution for 48 hours, at fifteen-minute resolution for 30 days,
and hourly for one year. Operational state transitions are retained for 90
days. Counters are stored as interval deltas; cumulative runtime dictionaries
are not persisted repeatedly.

Detailed stage timings, queue internals, copy reasons, worker details, and
episode transitions are diagnostic data. Diagnostics are scoped to a subsystem
or camera, automatically expire after a selected duration, and have a separate
storage budget. Raw images, video, credentials, tokens, and secret-bearing URLs
are never telemetry.

The operational and diagnostic stores are independent of the incident/event
database. Telemetry failure must not delay recording, incident creation, or
object detection.

## Operational metric catalog

System metrics cover CPU, application and worker memory, inference latency,
GPU utilization, detector demand/failures/capacity delays, and database writer
contention. Camera metrics cover availability, live/main FPS, capture
interruptions, EMA coverage and credible episodes, object-check outcomes,
tracking outcomes, and incident creation.

An operational event records important transitions such as a controlled
restart, unexpected interruption, camera outage/recovery, detector failure,
queue saturation, database contention, or retention failure. Repeated
equivalent events are coalesced.

## Legacy removal requirement

The migration may temporarily compare old and new results, but the final
runtime has one writer, one reader, and no compatibility fallback. The old
`runtime_telemetry_samples` table, compressed JSON codec, legacy readers, and
shadow plumbing are removed after the one-time versioned migration succeeds.

