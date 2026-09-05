> Preliminary, frozen investigation snapshot for the follow-up to PR #167.
> The 45-minute capture was still running when these files were prepared;
> final analysis and recording checks are pending. Referenced JSON/raw captures
> and helper scripts remain host-local under `/root/survng-measurements/` and
> are intentionally not published here. Commands and PIDs describe the original
> capture, not a newly started job; verify process identity before using them.

# 45-minute motion CPU investigation

Updated 2026-09-05T05:56:55.195146+00:00; capture running.

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
| new | 10.0–11.2 | 173.3–174.5 | 4.227 | 2.908 | 0.780 | 9.193 | 0 | 14 |

## Matched measurement coverage

The fixed elapsed-capture window starts at minute 6 and ends at minute 29 (or the latest common sample while running). It does not imply equivalent activity or completed warmup.

| Capture | Covered seconds | CPU cores | Main cores | Unattributed CPU seconds |
|---|---:|---:|---:|---:|
| original_post | 305.0 | 2.905 | 2.303 | 20.431 |
| new | 310.0 | 3.572 | 2.645 | 59.653 |

## Marked activity

| Category | Seconds | CPU cores | Intervals |
|---|---:|---:|---:|
| unmarked | 670.0 | 3.625 | 134 |

Quiet on one camera is not quiet for the whole fleet. Marker-boundary intervals are separated; unfinished scenarios are not retrospectively classified. Actual activity and detection outcomes are stored in separate files. No marked scenes means no recall conclusions.

## Per-camera workload

| Camera | Scene category | Minute rows | Preparation fps estimate | Admissions | Failures | Superseded |
|---|---|---:|---:|---:|---:|---:|
| back-left | unmarked | 7 | 4.036 | 5 | 0 | 0 |
| back-middle | unmarked | 7 | 2.938 | 1 | 0 | 0 |
| back-right | unmarked | 7 | 4.183 | 4 | 0 | 0 |
| boiler | unmarked | 7 | 3.938 | 5 | 0 | 0 |
| downstairs | unmarked | 7 | 3.907 | 0 | 0 | 0 |
| foyer | unmarked | 7 | 3.848 | 0 | 0 | 0 |
| front-door | unmarked | 7 | 4.010 | 0 | 0 | 0 |
| front-side | unmarked | 7 | 4.002 | 1 | 0 | 0 |
| gate | unmarked | 7 | 4.005 | 0 | 0 | 0 |
| lower-garage | unmarked | 7 | 2.695 | 0 | 0 | 0 |
| sherry-garage | unmarked | 7 | 4.371 | 0 | 0 | 0 |
| steve-garage | unmarked | 7 | 4.005 | 0 | 0 | 0 |
| upper-garage | unmarked | 7 | 2.895 | 0 | 0 | 0 |

Full per-camera capture/queue/failure evidence, process-age bins, alignment/budget indicators, and separate human outcomes are in live-comparison.json. Preparation fps is not stage duration or completed qualification cadence.

## Verified causes versus hypotheses

The aligned original evidence localizes ~97% of the measured service CPU increase to the main process. Astra verified that the retired quiet gate now allows qualification/learning to execute and that the four-frame window processes an additional transition. Neither identifies stage CPU shares. See ASTRA-SOURCE-REVIEW.md for consumers, mechanisms and missing timers.

No plateau in process CPU alone proves EMA warmup is complete. EMA learning-state indicators are absent from the allowlisted snapshot; alignment state must not be substituted. No existing diagnostic samples were found for the earlier captures.

## Recording checks

Recording flags are not proof of decodability. Only completed marked scenarios on recording-enabled cameras are eligible for saved-segment checks. Results, when run, are in recording-checks.json; absent markers mean unverified recording continuity, not a clean bill of health.

## Smallest next measurement

Use already-collected preparation/qualification/stage totals and demand/debug state through a narrowly extended owner-only snapshot, if separately authorized. Astra estimates 1–10 ms incremental projection CPU per snapshot; that estimate is unmeasured and excludes existing IPC/database work. No code, settings, profiler or package changes have been made. Source-based optimization candidates are not verified performance fixes.
