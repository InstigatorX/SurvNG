# Video pipeline optimization

Phase 0 was skipped. Recording stream-copy and go2rtc/WebRTC are unchanged.

| Phase | Change | CPU | p95 latency | Copies | RSS | Result |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Raw BGR instead of BMP on the live capture pipe (FFmpeg 8.1.2) | Lower at 720p | Read p95 lower at 720p | One pipe fill into the NumPy frame | Similar | Keep |
| 2 | Hardware decode on the one live capture process, then the Phase 1 software path | Not measured | Not measured | Same bgr24 pipe after hwdownload | Not measured | Revert the live wiring. QSV stays on recorded evidence frames |

## Phase 1 — raw BGR transport

### Current state

`FfmpegCaptureBackend` in `survng/app/camera_capture.py` decoded the live/substream in software FFmpeg, then encoded BMP into `image2pipe`. Python parsed the BMP and copied it into a contiguous BGR array. `CaptureHandle.read()` still returns a caller-owned frame. Motion, snapshots, and live detection were not changed.

### Change

Default `capture_frame_transport` is `rawvideo`:

`FFmpeg -> bgr24 -> rawvideo -> one read into a new uint8 array`

Each output frame is preceded by a `showinfo@capture=checksum=0` line. The reader does not consume stdout until that line provides the width and height, so a resolution change starts a new fixed-size read instead of shifting frame boundaries. The geometry queue is bounded at 64 entries and stores only dimensions. Stderr is read with `read1` so a large frame cannot block the geometry line behind a full-buffer read.

`capture_frame_transport=bmp` keeps the previous command and parser. Applying it requires a process restart.

### Risks

- Showinfo is required for safe framing. `-loglevel info` stays on for this transport.
- A geometry-index jump or an oversized frame fails the capture process and uses the existing reconnect path.
- Saturated file decode is slower in wall time because each frame waits for its stderr line. Live sample intervals are much larger than that wait.

### Tests

`tests/test_camera_capture.py` covers the raw command, the BMP fallback command, owned frames across a resolution change, alignment failure, the oversized-geometry reject, a real FFmpeg BGR decode, and a 640x360 file that previously deadlocked.

### Benchmark

Measured with FFmpeg 8.1.2, the same build the image pins in the Dockerfile. `scripts/benchmark_capture_transport.py` runs the production backend against a local H.264 file as fast as decode allows. It is not a paced RTSP session. Clock resolution on this machine is 10 ms, so the smaller deltas are coarse. Median of 3 runs:

640x360, 90 delivered frames:

| Transport | FFmpeg CPU / frame | Python CPU / frame | Read p95 | FFmpeg RSS |
| --- | --- | --- | --- | --- |
| bmp | 1.67 ms | 0.67 ms | 1.00 ms | 79 MB |
| rawvideo | 1.44 ms | 0.22 ms | 0.90 ms | 83 MB |

1280x720, 60 delivered frames:

| Transport | FFmpeg CPU / frame | Python CPU / frame | Read p50 | Read p95 | FFmpeg RSS |
| --- | --- | --- | --- | --- | --- |
| bmp | 5.00 ms | 3.17 ms | 3.20 ms | 4.56 ms | 119 MB |
| rawvideo | 3.83 ms | 1.00 ms | 1.25 ms | 2.59 ms | 112 MB |

Payload bytes match: both paths deliver the same BGR arrays. At 720p, FFmpeg CPU and the read tail both drop, and Python CPU drops because the BMP parse and extra contiguous copy are gone. Saturated 720p wall time was about 0.28 s (BMP) versus 0.26 s (raw). At 640x360 the raw run still spends more wall time waiting on the showinfo line (about 0.21 s versus 0.14 s). Both are far under an 8 fps live interval.

### Recommendation

Keep for the FFmpeg 8.1.2 runtime. Roll back with `capture_frame_transport` set to `bmp` if a camera cannot tolerate the stderr framing. Further capture CPU gains still depend on avoiding software decode.

## Phase 2 — hardware decode on the live capture process

### Current state

Phase 1 still decodes the live/substream in software inside the one persistent `FfmpegCaptureBackend` process, then writes bgr24. Recorded-frame decode is a separate FFmpeg invocation and is unchanged: `auto` and `off` stay on CPU there.

### Change

Live capture reads the existing `hardware_acceleration` setting (`auto`, `qsv`, `vaapi`, `off`). `auto` tries Intel QSV, then VAAPI, then the Phase 1 software command. An explicit mode tries that device and then software. `off` is software only.

A plan is used only when a real `/dev/dri/renderD*` node exists and `ffmpeg -hwaccels` lists that method. The default render-node path is not treated as present. The hwaccel list is cached per FFmpeg binary. The probe runs outside that cache lock. `off`, and every mode when no render node exists, do not probe FFmpeg.

The hardware filter is `select`, then `hwdownload,format=nv12`, then `format=bgr24`, then the existing rawvideo `showinfo` line. Device arguments are input options, before `-i`. Downstream frames stay caller-owned bgr24. There is still one FFmpeg process per source: a failed plan is closed before the next plan starts, inside the same open, so reconnect backoff is not the hardware fallback.

The capture backend is created when the manager starts. It does not receive `hardware_acceleration`. That setting remains on the recorded-evidence decoder. The live command is the Phase 1 software command, including when the configured mode is `qsv`.

### Risks

- If `select` cannot run on hardware frames, that plan exits and the open continues with software decode.
- `hwdownload` plus the nv12-to-bgr24 conversion can erase the decode savings. That was not measured here.
- A hardware plan that hangs until the open timeout delays the software attempt by that timeout. A plan that exits is closed immediately and the next plan starts.

### Tests

Plan order, the skipped probe when no render node exists, and the QSV command shape are unit-tested. A real FFmpeg 8.1.2 file open with QSV pointed at a missing device falls back to software and reports `decode_plan=cpu`. `auto` with no render node stays on the Phase 1 command.

### Benchmark

This machine has no `/dev/dri` render node, so hardware decode and the hwdownload copy were not timed. With `auto`, the command that actually runs is the Phase 1 software command. Do not treat that as a GPU result.

### Recommendation

Do not put this on the persistent live capture process. On a host whose `hardware_acceleration` is `qsv`, those live processes hold the render node that recorded evidence frames also use, and the incident picture falls back to the substream. The manager leaves live capture on the Phase 1 software command. `hardware_acceleration=qsv` still applies to recorded evidence frames.
