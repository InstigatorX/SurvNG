# Native video datapath review — 2026-09-16

Reviewed the current native-first execution path, its introduction at `63d8b92`,
and the GStreamer lineage beginning at `47efc6b`. This review covers native decode,
the evidence/detection split, process transport, capture ownership, adaptive
admission and exclusions, tracking/incident state, verification, recording lookup,
and evidence promotion. It is not a claim that every historical implementation or
every camera scene has been exhaustively tested.

## Findings and fixes

| Area | Finding | Change |
| --- | --- | --- |
| Camera polling | Every 50 ms poll copied native BGR pixels just to inspect frame timestamps/session, even without new video. | Native polling borrows the published immutable frame. Writable callers retain the existing copying API. |
| Observation polling | Each poll handed the entire recent detection history back to incident processing and timestamp mapping. | A session/sequence cursor returns only unseen observations without draining history needed by live overlays. Reconnects still admit the new session. |
| GStreamer transport | Frame encoding concatenated the large pixel payload into several successive headers; receive-side slicing also copied full payloads. | Send header and pixels under one writer lock, preserving the wire format and short-write/failure handling. Receive header slices use views until the owned NumPy allocation. The reader copies out of its mutable buffer once. |
| JPEG previews | Capture decoded every preview JPEG even though native evidence uses BGR frames and encoded-preview callers need only JPEG bytes. | Decode preview pixels on demand and cache until the next JPEG. |
| Verification memory | Separate object nominations copied the same immutable source frame. | Share capture-owned immutable frames; copy writable inputs to maintain ownership. |
| Verification work | The verifier examined remaining samples after a clear positive already satisfied admission. Cancellation did not stop subsequent sample work. | Stop at the first clear positive; keep the three-negative rejection rule. Check cancellation between stages and discard canceled results. |
| Recording extraction | Each temporary local frame was PNG-compressed by FFmpeg and decompressed by OpenCV. | Use lossless uncompressed BGR BMP for local IPC. Archived image formats remain configured separately. |
| Incident history | Every fresh observation copied the accumulated episode lists. | Append to the single-owner histories; keep deep copies at publication/persistence boundaries and existing compaction. |
| Health policy | Reading effective FPS repeatedly serialized and validated all motion settings. | Resolve the two already-validated inherited fields directly, without a stale cache. |
| Evidence scoring | Empty/ineligible object lists still triggered full-frame image-quality work. | Return before touching pixels. |
| Cover correctness | Later cover promotion accepted same-class fragments with a weaker overlap rule than initial admission. | Share the admission extent rule: detected-box overlap at least 50%, and IoU at least 0.3. |

## Decisions retained after review

- VA evidence downloads keep their separate postprocessing surface. Earlier fixes
  established that mapping decoder surfaces shared with inference is unsafe.
- The evidence branch stays independent of adaptive inference. Idle inference
  must not remove source frames needed for time-matched evidence and live viewing.
- Native tracking predictions remain display context, never fresh incident proof.
- Motion exclusions remain polygon-based and override approach margins without
  suppressing fresh object evidence. No new pixel masking or resolution loss was
  introduced.
- Recorders already use compressed video stream copy, with independent lifecycle
  and timestamp repair. Recording lookup already uses an indexed query and defers
  missing-edge refreshes; no request-path media scan was added.
- The native CPU verifier already reuses its process and graph. Per-sample model
  process startup was not present.

## Measurements

Local synthetic measurements compare the pre-review code at `d2af72d` with the
review changes. They are not fleet CPU/GPU savings or scene-recall measurements.

| Measurement | Before | After |
| --- | --- | --- |
| Package/write one 1920×1080 BGR frame to a discard sink, 100 iterations with tracemalloc | 6.109 ms/frame; 11.87 MiB peak extra allocation | 0.007 ms/frame; under 0.01 MiB peak extra allocation |
| Extract a synthetic 1920×1080 FFV1 frame through FFmpeg/OpenCV, six iterations | PNG: 121.6 ms wall / 102.7 ms child CPU per frame | BMP: 74.9 ms wall / 71.2 ms child CPU per frame |

PNG and BMP extraction produced identical BGR arrays. The tradeoff is local pipe
bandwidth: this synthetic PNG was 212,953 bytes versus 6,220,854 bytes for BMP.
Frame-packaging measurements exclude actual pipe I/O, native decode, and model
inference; tracemalloc also affects absolute timings.

## Validation

The final focused Python run passed **434 tests**, with one skipped generated
capture test because the virtualenv lacked GStreamer `videotestsrc`. The real
native checks below ran separately using `/usr/bin/python3` and passed.

Regression coverage includes immutable ownership and staleness, cursor/reconnect
semantics, lazy preview caching, interleaved multipart messages and short writes,
failure after a sent header, borrowed payload lifetime, shared nomination frames,
early-positive/cancellation behavior, policy inheritance, history publication,
uncompressed frame decoding, and rejection of fragmentary cover replacements.

Real native checks passed:

- `scripts/gstreamer-smoke.py`: shared supervisor, frames-only branches, fresh and
  empty results, interval/prediction provenance, native IDs, and NMS variants.
- `scripts/gstreamer-spatial-check.py --budget`: four adaptive/fixed scenarios.
- `scripts/gstreamer-spatial-check.py --va`: 12 fresh results with VA surface sharing.
- `scripts/gstreamer-spatial-check.py --verify-crops`: three odd-width native crops.

The smoke run emitted a GStreamer plugin-scanner format assertion during plugin
discovery but completed successfully. It was not suppressed or treated as proof
of a pipeline failure. Native checks validate plumbing and geometry, not object
recognition accuracy on camera scenes.

After restarting with the reviewed code, the owner-only status snapshot at
15:35:39 UTC reported 13/13 cameras connected, all nine enabled detectors healthy,
zero invalid native-evidence counters, and recording active on all nine cameras
with recording enabled. Adaptive active/idle modes were present. This is a
post-startup operational check, not a long-duration soak or a measured fleet CPU
comparison.
