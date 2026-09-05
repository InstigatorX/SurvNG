> Completed capture: 2026-09-05 05:45:44–06:30:44 UTC (45 minutes).
> See [Final Astra assessment](FINAL-ASTRA-ASSESSMENT.md) for conclusions and limits.
> Referenced raw captures, JSON evidence and collector helpers remain host-local
> under `/root/survng-measurements/`; they are not published in this documentation PR.

# Motion CPU investigation — completed capture

The unchanged original collector started at 2026-09-05 05:45:44 UTC
(01:45:44 EDT), for 2700 seconds. Finished: 06:30:44 UTC
(02:30:44 EDT). The collectors have exited; do not start another capture without authorization.

Capture directory:
`/root/survng-measurements/investigation-20260905/capture-20260905T054543Z`

Primary/supplement/dependency PIDs: 4027454 / 4027460 / 4027461.
Read-only report updater PID: 4041905. Post-capture recording checker: 4041906.

## Exact commands

Start a subsequent capture, only when requested:

```bash
python3 -B /root/survng-measurements/investigation-20260905/launch.py
```

Status and early stop (only collectors, never SurvNG):

```bash
python3 -B /root/survng-measurements/pre-deployment/baseline.py status /root/survng-measurements/investigation-20260905/capture-20260905T054543Z
python3 -B /root/survng-measurements/pre-deployment/baseline.py stop /root/survng-measurements/investigation-20260905/capture-20260905T054543Z
```

The report updater and recording checker have exited. PIDs above are historical;
do not signal them because they may be reused. Commands describe the original
host-local workflow, not active jobs.

## Human ground truth

Use a unique scenario name per camera/attempt. Camera IDs or display names
are accepted. Start/end record actual activity; outcomes are a separate file.
Optional `--at '2026-09-05T02:00:00-04:00'` records an approximate past time.

```bash
python3 -B /root/survng-measurements/investigation-20260905/mark.py start --camera Gate --scenario walk-1 --activity 'Person walking close to camera'
python3 -B /root/survng-measurements/investigation-20260905/mark.py end --camera Gate --scenario walk-1
python3 -B /root/survng-measurements/investigation-20260905/mark.py outcome --camera Gate --scenario walk-1 --result 'Observed outcome, or not checked'
```

For quiet observations use `start --kind quiet` and describe what you actually
observed. Quiet on one camera does not establish quiet across the fleet.

Checklist, where safe and practical:

- Quiet scene with start/end and camera identified.
- Person walking close, then far from the camera (separate attempts).
- Slow movement, stopping, then moving again.
- Person entering a scene containing parked vehicles.
- Simultaneous movement on several cameras, marking each camera.
- Day/night examples when practical; this capture itself is at night.

Unmarked periods remain unmarked, not automatically quiet or normal.

## Sampling and evidence

Original scripts and original captures are unchanged. Process resources are
sampled about every five seconds. The original application collector begins
at fifteen seconds and backs off after requests exceeding 250 ms or errors.
This completed run scheduled intervals of 15, 30, 60 and finally 120 seconds.
There were 67 successful application snapshots; the final state sample preceded
capture completion by 84.5 seconds. Actual timestamps/request durations are saved. Existing minute telemetry is read with the original lag/bounds.
Do not treat missing final minute rows as zero activity.

Metadata records checkout/deployment evidence, process identity/age and the
configuration fingerprint. The process began before this capture and is the
same instance as the original post-deployment run. Loaded build/model identity
not exposed by the runtime remains explicitly unverified.

- `ASTRA-SOURCE-REVIEW.md`: source mechanisms, consumers and telemetry limits.
- `aligned-original-cpu.json`: aligned original pre/post process attribution.
- `original-evidence-sha256.json`: original raw evidence preservation manifest.
- `LIVE-CAPTURE-REPORT.md` / `live-comparison.json`: refreshed saved-data analysis.
- Capture `*.jsonl`: timestamped raw samples; scene/outcome markers are separate.
- `recording-checks.json`: post-capture verification result, when finished.

Five-minute bins describe process-age trends, not a warmup completion rule.
Scene-learning convergence is not currently exported. Same elapsed coverage
does not prove equivalent camera activity. No causal performance improvement
is claimed without that evidence.

## Bounded recording checks

Only completed human-marked scenarios wholly within this capture and on
recording-enabled cameras are eligible. No markers means no media reads and
recording continuity/decodability remains unverified. Decoding starts only
after capture completes, uses existing local ffprobe/ffmpeg, one decoder
thread, low CPU/I/O priority and quarter-realtime pacing. No inference occurs.

At most three selected segments per scenario and twelve total are decoded;
main stream is preferred before live. Maximum file size 256 MiB, duration
30 seconds, total wall budget 900 seconds. Validated/playable rows and a
conservative finalization guard (start + 302 seconds and stale mtime/end) are
required. Index queries are camera/source/time bounded, at most 201 rows with
a 300-second start lookback. Truncation means incomplete coverage. Index gaps,
decoder errors, timeouts, and unverified portions are reported separately.
Successful samples do not establish whole-scenario frame continuity.

No service, settings, application code, packages or profiler changes were made.

## Completed outcome

All measured periods were unmarked. The recording checker completed without
reading or decoding media because no eligible scenarios were marked. Recording
continuity, decodability and detection recall remain unverified. Final assessment
and full measurement tables are included alongside this historical runbook.
