# Nightly benchmarks

The `benchmarks` job in `.github/workflows/nightly-tests.yml` runs after the
nightly test job (whether it passed or not) on the same self-hosted runner and
records a performance baseline. It is **record only**: nothing fails on a slow
number. Set regression thresholds only after about two weeks of results show
how noisy the shared runner is. It is not part of PR CI.

## What runs

`scripts/run-benchmarks.py` runs each benchmark script as a subprocess with
`--json` and fixed parameters, three times (`--repetitions 3`), and reports
the median, min and max of each metric across those runs.

| Benchmark | Script and fixed parameters | Metrics |
| --- | --- | --- |
| `capture_transport_640x360`, `capture_transport_1280x720` | `benchmark_capture_transport.py --width W --height H --fps 20 --duration 20 --frame-rate 10 --repeats 3` | FFmpeg and Python CPU ms per delivered frame, read p95, elapsed, frames, FFmpeg RSS, for the `bmp` and `rawvideo` transports (each value is the script's own median of 3 runs) |
| `motion_color_reuse` | `benchmark_motion_color_reuse.py --seed 725` | ms per frame for the five preprocessing cases |

Not run, because they cannot run unattended and deterministically on a
CPU-only runner:

- `benchmark-depth.py` needs the YOLO26-depth OpenVINO model, which is not in
  the repository and would need a download.
- `benchmark_motion_polling.py` compares against functions extracted from a
  historical commit (`git show 0c119dd:...`), which needs full Git history
  instead of the shallow checkout, and the extracted code is executed against
  today's module namespace, so it breaks as those modules change.

## Fixed input

No video is committed. Each capture run generates its clip with FFmpeg:
`lavfi testsrc` at the given size, 20 fps, 20 seconds, `libx264`, `yuv420p`,
GOP 20 frames. The capture backend selects 10 fps from it, which gives about
160 delivered frames per run. The 20-second clip, longer than the script's
4-second default, means the 10 ms `/proc` CPU tick does not swamp the
per-frame CPU numbers. The motion benchmark uses NumPy `RandomState(725)`
frames (640x360 and 1280x720 BGR). The JSON records the FFmpeg, OpenCV,
NumPy and Python versions; compare numbers only across runs with the same
versions and CPU.

If `ffmpeg` is not on the runner's `PATH`, the capture benchmarks are
reported as `skipped` and the job continues. A benchmark that exits non-zero
or prints unparseable output is reported as `failed`, and the runner exits 1
after it has written the result.

## Results

- `benchmark-results/nightly-benchmarks.json` is uploaded as the
  `nightly-benchmarks-<run number>` artifact and kept for 90 days. Fields:
  `schema_version`, `generated_at`, `repetitions`, `wall_seconds`,
  `environment` (commit, Python, OpenCV, NumPy, FFmpeg, platform, CPU model,
  CPU count, runner name), and `benchmarks.<name>` with `status`,
  `wall_seconds` and `metrics.<metric>` = `unit`, `median`, `min`, `max`,
  `repetitions`, `values`.
- A Markdown table is printed and appended to the job summary.

Run locally with
`python scripts/run-benchmarks.py --output /tmp/benchmarks.json`
(add `--only NAME` to run one benchmark). The whole set takes under a
minute on a 4-core container. Numbers from a desktop or container cannot be
compared with the runner's.

## Noise

The runner is shared and self-hosted. Other jobs, thermal state, and CPU
frequency all move these numbers. CPU per frame comes from 10 ms `/proc`
ticks. Read trends over many nights, not single results.
