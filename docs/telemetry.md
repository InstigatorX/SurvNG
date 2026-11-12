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
and detector demand and failures. Current GPU health remains available in the
live telemetry response without being duplicated into history. Camera metrics cover availability, live/main FPS, capture
interruptions, EMA coverage and credible episodes, object-check outcomes,
and tracking capacity outcomes.

An operational event records important transitions such as a controlled
restart, unexpected interruption, camera outage/recovery, detector failure,
sustained detector backlog, or retention failure. Repeated
equivalent events are coalesced.

## Host-local process metrics

These live only in process memory and are exposed in the owner-only
observability snapshot (`./survngctl status`); they are not persisted in the
telemetry store. Each uses a fixed-bucket latency histogram
(`survng/app/runtime_metrics.py`): memory is fixed at start-up, and a record
costs one bisect plus a few integer updates under a small lock.

Every histogram has the same shape, all durations in milliseconds:
`count`, `total_ms`, `max_ms`, `mean_ms` (since process start);
`bucket_upper_bounds_ms` (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500,
5000, 10000) with `bucket_counts` holding one more entry for values above the
last bound; and `recent` covering the last 270–300 seconds (`window_seconds`,
`count`, `max_ms`, `max_at_epoch_seconds` — Unix time of the worst value —
and approximate `p50_ms`/`p95_ms`/`p99_ms`, each the upper bound of the
bucket holding that rank, capped at the window maximum).

- `event_loop.lag`: how late a 100 ms asyncio sleep woke, measured
  continuously by one task owned by the observability server (started and
  cancelled by the app lifespan). Adds `running` and `interval_ms`. Stalls
  longer than the interval are always observed, as lag of at least the
  stall minus the interval. A stall of 1 s or more is logged at most once per
  five minutes. `event_loop.probe_lag_ms` (one 10 ms probe per request) and
  `event_loop.tasks` are unchanged.
- `databases.domains.<name>`: per SQLite contention domain (`main`, `jobs`,
  `recording_index`, `telemetry`, `exports`, `face_dataset`; `other` for an
  unlabelled `ClosingDatabaseConnection`). `transaction_ms` is the duration
  of each `with connection:` block from enter to commit/rollback and close;
  `transactions_failed` counts blocks that ended in an exception;
  `busy_errors` counts those that ended in a SQLite "locked"/"busy"
  `OperationalError` after SQLite's own busy timeout. Connections used
  without a `with` block are not timed.
- `databases.main_write_lock`: the process-wide main-database writer lock.
  `acquisitions` and `contended` are counters; `wait_ms` records contended
  acquisitions only (uncontended ones wait ~0); `hold_ms` records every
  acquisition until release at commit, rollback or close. Nested reentrant
  acquisitions are recorded separately. `available` is false when the lock is
  not instrumented.
- `databases.metric_failures` and `main_write_lock.metric_failures` count
  metric recording errors; these never affect the transaction or lock.

The production soak collector (`scripts/production-soak.py`) records these
fields in each sample and flags threshold breaches in `--summarize`.

## Legacy removal requirement

The migration may temporarily compare old and new results, but the final
runtime has one writer, one reader, and no compatibility fallback. The old
`runtime_telemetry_samples` table, compressed JSON codec, legacy readers, and
shadow plumbing are removed after the one-time versioned migration succeeds.
