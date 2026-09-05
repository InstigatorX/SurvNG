> Completed capture: 2026-09-05 05:45:44–06:30:44 UTC (45 minutes).
> See [Final Astra assessment](FINAL-ASTRA-ASSESSMENT.md) for conclusions and limits.
> Referenced raw captures, JSON evidence and collector helpers remain host-local
> under `/root/survng-measurements/`; they are not published in this documentation PR.

# 45-minute motion CPU investigation

Updated 2026-09-05T06:31:27.946477+00:00; capture finished.

## CPU and memory over process age

No warmup cutoff is imposed. These are five-minute reporting bins; changing activity can also move the means. The original post-deployment run used the same process near startup, whereas this run observes it hours later.

| Capture | Elapsed minutes | Process age minutes | CPU cores | Main cores | Capture cores | Memory GiB | Queue max | CPU intervals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original post | 0.0–5.0 | 1.9–6.9 | 3.316 | 2.494 | 0.539 | 7.518 | 0 | 59 |
| original post | 5.0–10.0 | 6.9–11.9 | 2.765 | 2.214 | 0.417 | 9.762 | 0 | 59 |
| original post | 10.0–15.0 | 11.9–16.9 | 2.944 | 2.373 | 0.425 | 10.633 | 0 | 59 |
| original post | 15.0–20.0 | 16.9–21.9 | 2.829 | 2.266 | 0.430 | 11.898 | 0 | 59 |
| original post | 20.0–25.0 | 21.9–26.9 | 4.107 | 2.933 | 0.718 | 10.644 | 0 | 59 |
| original post | 25.0–29.9 | 26.9–31.8 | 2.894 | 2.293 | 0.461 | 11.604 | 0 | 59 |
| new | 0.0–5.0 | 163.3–168.3 | 3.452 | 2.651 | 0.572 | 9.018 | 0 | 59 |
| new | 5.0–10.0 | 168.3–173.3 | 3.620 | 2.691 | 0.605 | 9.193 | 1 | 59 |
| new | 10.0–15.0 | 173.3–178.3 | 3.784 | 2.736 | 0.674 | 9.407 | 0 | 59 |
| new | 15.0–20.0 | 178.3–183.3 | 3.905 | 2.803 | 0.691 | 9.822 | 0 | 59 |
| new | 20.0–25.0 | 183.3–188.3 | 3.419 | 2.553 | 0.591 | 8.547 | 0 | 59 |
| new | 25.0–30.0 | 188.3–193.3 | 3.172 | 2.394 | 0.548 | 8.359 | 0 | 59 |
| new | 30.0–35.0 | 193.3–198.3 | 3.567 | 2.690 | 0.601 | 8.565 | 0 | 59 |
| new | 35.0–40.0 | 198.3–203.3 | 3.935 | 2.833 | 0.734 | 8.377 | 0 | 59 |
| new | 40.0–44.9 | 203.3–208.2 | 3.303 | 2.525 | 0.545 | 8.065 | 0 | 59 |

## Matched measurement coverage

The fixed elapsed-capture window starts at minute 6 and ends at minute 29 (or the latest common sample while running). It does not imply equivalent activity or completed warmup.

| Capture | Covered seconds | CPU cores | Main cores | Unattributed CPU seconds |
|---|---:|---:|---:|---:|
| original_post | 1376.3 | 3.143 | 2.439 | 149.160 |
| new | 1375.0 | 3.527 | 2.607 | 256.119 |

## Marked activity

| Category | Seconds | CPU cores | Intervals |
|---|---:|---:|---:|
| unmarked | 2695.8 | 3.584 | 539 |

Quiet on one camera is not quiet for the whole fleet. Marker-boundary intervals are separated; unfinished scenarios are not retrospectively classified. Actual activity and detection outcomes are stored in separate files. No marked scenes means no recall conclusions.

## Per-camera workload

| Camera | Scene category | Minute rows | Preparation fps estimate | Admissions | Failures | Superseded |
|---|---|---:|---:|---:|---:|---:|
| back-left | unmarked | 40 | 4.026 | 28 | 0 | 1 |
| back-middle | unmarked | 40 | 2.957 | 11 | 0 | 0 |
| back-right | unmarked | 40 | 4.221 | 18 | 0 | 0 |
| boiler | unmarked | 40 | 3.957 | 17 | 0 | 3 |
| downstairs | unmarked | 40 | 3.957 | 0 | 0 | 0 |
| foyer | unmarked | 40 | 3.863 | 2 | 0 | 0 |
| front-door | unmarked | 40 | 4.019 | 0 | 0 | 0 |
| front-side | unmarked | 40 | 3.993 | 8 | 0 | 0 |
| gate | unmarked | 40 | 3.991 | 0 | 0 | 0 |
| lower-garage | unmarked | 40 | 2.732 | 0 | 0 | 0 |
| sherry-garage | unmarked | 40 | 4.370 | 0 | 0 | 0 |
| steve-garage | unmarked | 40 | 3.997 | 0 | 0 | 0 |
| upper-garage | unmarked | 40 | 2.912 | 1 | 0 | 0 |

Full per-camera capture/queue/failure evidence, process-age bins, alignment/budget indicators, and separate human outcomes are in live-comparison.json. Preparation fps is not stage duration or completed qualification cadence.

## Verified causes versus hypotheses

The aligned original pre→post evidence localizes ~97% of that service CPU increase to the main process. This does not describe the later original-post→new comparison, where about 44% of the additional CPU lies in the main process. Astra verified that the retired quiet gate now allows qualification/learning to execute and that the four-frame window processes an additional transition. Neither identifies stage CPU shares. See ASTRA-SOURCE-REVIEW.md for consumers, mechanisms and missing timers.

No plateau in process CPU alone proves EMA warmup is complete. EMA learning-state indicators are absent from the allowlisted snapshot; alignment state must not be substituted. No existing diagnostic samples were found for the earlier captures.

## Recording checks

The recording checker completed with zero eligible marked scenarios and no media reads or decodes. Recording continuity and decodability remain unverified, not passed. The host-local result is recording-checks.json.

## Smallest next measurement

Use already-collected preparation/qualification/stage totals and demand/debug state through a narrowly extended owner-only snapshot, if separately authorized. Astra estimates 1–10 ms incremental projection CPU per snapshot; that estimate is unmeasured and excludes existing IPC/database work. No code, settings, profiler or package changes have been made. Source-based optimization candidates are not verified performance fixes.
