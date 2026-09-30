# Video pipeline optimization

Phase 0 was skipped. Recording stream-copy and go2rtc/WebRTC are unchanged.

| Phase | Change | CPU | p95 latency | Copies | RSS | Result |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Raw BGR instead of BMP on the live capture pipe | Python lower; FFmpeg flat | Saturated read similar or lower | One pipe fill into the NumPy frame | FFmpeg lower at 720p | Keep |

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

`scripts/benchmark_capture_transport.py` runs the production backend against a local H.264 file as fast as decode allows. It is not a paced RTSP session. Clock resolution on this machine is 10 ms, so small FFmpeg deltas are coarse. Median of 3 runs:

640x360, 90 delivered frames:

| Transport | FFmpeg CPU / frame | Python CPU / frame | Read p95 | FFmpeg RSS |
| --- | --- | --- | --- | --- |
| bmp | 1.33 ms | 0.67 ms | 1.05 ms | 65 MB |
| rawvideo | 1.33 ms | 0.33 ms | 1.31 ms | 61 MB |

1280x720, 60 delivered frames:

| Transport | FFmpeg CPU / frame | Python CPU / frame | Read p50 | Read p95 | FFmpeg RSS |
| --- | --- | --- | --- | --- | --- |
| bmp | 3.83 ms | 2.50 ms | 2.74 ms | 4.45 ms | 97 MB |
| rawvideo | 3.67 ms | 1.00 ms | 1.94 ms | 3.28 ms | 81 MB |

Payload bytes match: both paths deliver the same BGR arrays. FFmpeg CPU did not drop. H.264 decode dominates BMP encode here, and showinfo replaces that encode work. Python CPU dropped because the BMP header parse and the extra contiguous copy are gone. Raw read p95 at 720p is lower. Wall time for the saturated 720p run was about 0.22 s (BMP) versus 0.28 s (raw). Both are far under an 8 fps live interval.

### Recommendation

Keep. Roll back with `capture_frame_transport` set to `bmp` if a camera cannot tolerate the stderr framing. The next gain on capture CPU has to come from avoiding software decode, not from the pipe format.
