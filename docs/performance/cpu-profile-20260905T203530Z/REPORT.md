# Bounded live CPU profile — 2026-09-05

**The strongest resolved computation hotspot is NumPy’s float selection routine, called by motion background median/MAD and threshold statistics. Full main-process CPU attribution remains incomplete: threads absent from the profile consumed 37.25% of main CPU.** There is not enough coverage to name the largest total-process function or recommend an implementation change.

**One highest-value follow-up:** obtain a permitted CPU-clock native profile of the main process, including the unrepresented worker-thread cohort, before choosing an optimization. The current profile identifies the relevant application paths, but the missing workers are too large to assign to OpenCV or any other library by assumption. This follow-up was not run because `perf` is unavailable; no profiler/toolchain/kernel installation or setting change was attempted beyond the authorized isolated py-spy installation.

## Execution and boundaries

- Target: running SurvNG PID **651962**, started 19:03:28 UTC; clean checkout `8177935464ed3c398cad01082c7dfab47b2d2e05`.
- Tool: **py-spy 0.4.2**, installed only in this directory’s `profiler-env`. Local `record --help` was read and saved before attachment.
- One requested **60-second, 25 Hz native** sample window. Options included `--native --threads --full-filenames --format speedscope`; no GIL-only filter, subprocess capture, idle inclusion or locals capture.
- Wrapper interval: **20:36:55.882650–20:37:58.306166 UTC**, 62.432 seconds. It encloses the requested 60 seconds plus attachment/output work; the speedscope format does not provide exact wall timestamps for each retained sample. Independent CPU uses 61 whole one-second intervals inside these bounds, covering **61.005 seconds**.
- Successful exit; **2,726 sampled thread stacks**, **86 thread profiles**, and **11 errors reported by py-spy**. The error count is retained as a count, not converted into an unsupported error percentage.
- No second window: native coverage warranted considering one, but `perf` was absent from PATH, `/usr/bin/perf`, `/usr/local/bin/perf`, and standard `/usr/lib/linux-tools*` locations. No profiler remains active.

This is current operational evidence. It does not establish why CPU differed in any earlier capture, replay or deployment.

## Independently measured CPU and coverage

| Scope | CPU seconds | Average cores | Share of main CPU |
|---|---:|---:|---:|
| Entire service cgroup | 141.223 | 2.3149 | — |
| Main process | 115.410 | 1.8918 | 100% |
| Main threads represented in py-spy | 71.820 | 1.1773 | 62.23% |
| Main threads absent from py-spy | 42.990 | 0.7047 | 37.25% |
| Main/thread counter difference | 0.600 | 0.0098 | 0.52% |
| Capture child processes | 20.830 | 0.3414 | — |
| Recorder child process | 4.420 | 0.0725 | — |
| Object-inference child process | 0.170 | 0.0028 | — |

The main process accounts for 81.72% of measured service CPU. Cgroup and process numbers overlap; they must not be added. The 0.600-second main/thread discrepancy is consistent with counter quantization and non-atomic reads, not assigned function work.

Thread-identity attribution locates **38.400 CPU seconds (33.27% of main CPU)** on the named motion-analysis threads, **9.750 seconds (8.45%)** on recording-indexer, and **2.750 seconds (2.38%)** on recording-index-maintenance. These are measured thread CPU totals, not estimates derived from stack percentages. The remaining represented threads consumed 20.920 seconds.

The largest unrepresented TIDs were 652968, 652969, 652970 and 652971, consuming 5.26, 5.19, 5.04 and 4.68 CPU seconds respectively. Many absent workers share a start time and fall in the 652968–652983 cohort. Their `comm` is `uvicorn`; that inherited name and creation timing do **not** identify their owning library. `--native` successfully captured extension frames on represented threads, but it did not provide stacks for these CPU-consuming workers.

Thus **62.23% is thread-identity coverage, not the fraction of main CPU precisely explained by functions**. No defensible exact median/MAD, copying or OpenCV share of total main CPU can be calculated from this capture.

## Observed self and inclusive stacks

“Self” means the retained stack’s leaf. “Inclusive” means the function/path occurs anywhere in that stack, counted once per function per stack. The denominator is 2,726 sampled thread stacks, **not process CPU time**. Inclusive counts overlap and must not be summed.

| Self function / library | Self samples | Inclusive samples | Self share of sampled stacks |
|---|---:|---:|---:|
| `introselect_<npy::float_tag, false, float>` / NumPy native extension | 668 | 668 | 24.50% |
| Unresolved libc `0x72450e4c6e51` | 608 | 608 | 22.30% |
| `pread64` / libc | 300 | 300 | 11.01% |
| `select` / libc | 230 | 230 | 8.44% |
| Unresolved libc `0x72450e4c7050` | 107 | 107 | 3.93% |
| `pthread_cond_signal` / libc | 88 | 88 | 3.23% |
| `fstatat64` / libc | 50 | 50 | 1.83% |
| Unresolved libc `0x72450e5b6f07` | 48 | 48 | 1.76% |
| `_strptime` / Python standard library | 23 | 29 | 0.84% |

The unresolved libc addresses are retained rather than guessed into function names. Stack ancestry places the first large address mainly under `pthread_cond_timedwait`, the second under mutex-lock paths, and examples of the third under NumPy flat-copy paths. `pthread_cond_signal` is synchronization activity, not itself proof of a blocked thread.

| Inclusive application/function path | Sampled stacks |
|---|---:|
| Motion analysis service `_run` | 1,361 |
| Motion pipeline `_process_stage_active` | 1,259 |
| `AdaptiveEmaBackgroundStage` | 848 |
| NumPy `PyArray_Partition` | 827 |
| NumPy `median` | 787 |
| Camera capture `_next_frame` | 402 |
| Recording index `refresh_recording_index` | 397 |
| `AdaptiveStatisticalThresholdStage` | 329 |

A conservative stack-path classification found 668 NumPy-selection computation leaves, 702 synchronization/wait stacks, 624 I/O/polling stacks, and 732 other/unresolved stacks. Wait or I/O ancestry does not measure time actually asleep or prove those calls consume no CPU. The retained weights sum to 109.04 sampled thread-seconds; that is **not** a process CPU measurement and exceeds measured CPU on the represented threads, illustrating why sample weights cannot be treated as CPU seconds here.

## Hypotheses checked against the actual profile

Source inspection followed the completed profile and its observed call sites.

| Hypothesis | Evidence and judgment |
|---|---|
| Background median/MAD | Supported as the largest resolved computation path. Background has 848 inclusive stacks. NumPy selection is the leaf in 471 of them: 220 at `adaptive_stages.py:266` (`np.median(delta)`) and 251 at line 267 (`np.median(np.abs(delta - median))`). Another 250 background stacks are synchronization/wait paths, so all 848 must not be labeled computation. |
| Threshold statistics | Supported as a secondary resolved computation path. Threshold has 329 inclusive stacks, including 196 NumPy-selection leaves: 51 at line 434 (median), 61 at line 435 (MAD), and 84 at line 437 (80th percentile). The source already caches statistics by timestamp; this profile does not establish how much further work could safely be removed. |
| Array copies | Present, not dominant among resolved sampled leaves. Explicit copy APIs occur in 52 inclusive stacks; `PyArray_NewCopy` has 29 and `PyArray_CopyAsFlat` 23. Native unresolved copy paths and memory traffic inside other routines prevent treating 52 as a complete copy-cost count. |
| OpenCV / thread contention | Incomplete. CV2 library frames occur in 261 inclusive stacks, largely as unresolved native addresses. Synchronization is visible, including around NumPy and OpenCV callers, but this does not separate GIL reacquisition, allocator/library locks, useful parallel work, or spinning. The missing 37.25% of main CPU cannot be assigned to OpenCV without a native CPU-clock profile. |

Recording-index work is also real: its named thread consumed 8.45% of main CPU, and observed stacks include `_recording_rows_for_files`, path resolution and `media_storage.location_id_for` at lines 297/305, plus SQLite and filesystem calls. Its prominence in stack samples includes I/O and lock paths; it is not evidence that it dominates total main computation.

The background source still computes median/MAD and updates noise statistics before handling stale transitions. Any future change there must preserve historical-noise and learning semantics. This report proposes no code change or additional optimization experiment.

## Confidence, overhead and preservation

Confidence is high that NumPy float selection is the largest **resolved computation leaf observed** and that the background/threshold call-site mapping is correct. Confidence is moderate in broad thread attribution and low in a complete function-level explanation of main CPU because of missing worker stacks, unresolved symbols, wait/I/O occupancy, 11 reported sampling errors, and the wrapper/sampling boundary mismatch.

Profiler overhead was not isolated by a controlled baseline. Py-spy’s normal process-pausing sampler can perturb scheduling. The independent collector used 1.759 CPU seconds over its approximately 199.827-second lifetime; that does not measure py-spy overhead or service-side perturbation.

Independent before/after verification reports service PID/start, configuration checksum and clean checkout unchanged. No service restart, deployment, runtime/thread setting change, inference/replay/unit-test workload, new stream, production code edit or GitHub write occurred. Only py-spy was installed, in the dedicated profiling environment. The loaded application revision remains inferred from checkout/deployment timing; no runtime build identifier is exposed.

Artifacts: [native speedscope profile](pyspy-native.speedscope.json), [profile analysis](profile-analysis.json), [aligned CPU summary](cpu-summary.json), [profile window and exact command](profile-window.json), `pyspy-record.log`, `py-spy-record-help.txt`, `profiler-packages.txt`, `cpu.jsonl`, `metadata.jsonl`, `completion.jsonl`, and before/after socket snapshots. All are local to this measurement directory.
