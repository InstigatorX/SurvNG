# Native incident imagery proposal

## Current failure modes

Native capture exposes a 640-pixel preview. `NativeCameraWorker._evidence` saves that frame only when an episode starts. Resizing it to detection coordinates does not recover main-stream detail. Missing exact-session/PTS preview frames yield an empty snapshot path.

Meanwhile, `NativeActivity.persist` updates incident objects with later track coordinates, but leaves the original snapshot unchanged. Object-focused thumbnails can therefore crop a location the subject has already left. Incident representative selection ranks object confidence ahead of image availability, and does not assess image quality. These are code-level findings; individual gray-image examples still need pixel-level inspection to distinguish bad capture from bad cropping.

## Recommended pipeline

1. **Keep immutable image evidence.** Store the image timestamp, source/session/PTS, dimensions, and the objects observed in that exact frame together. Separate snapshot annotations from latest live-track state. A cover change updates all of these atomically and increments `evidence_revision` so clients invalidate thumbnails.
2. **Maintain a bounded shortlist during the episode.** Evaluate at most one eligible candidate per second and retain only the best three timestamps/metadata records. Score the actual incident subject: visible area, confidence, distance from frame edges, sharpness, exposure, and scene context. Reject missing, near-uniform gray/black, grossly corrupted, or substantially clipped frames. Do not reject a useful night image merely because it is dark.
3. **Extract full-resolution images from existing main recordings.** A bounded background job waits for the relevant recording segment to become readable, then extracts around the shortlisted timestamps. This avoids a permanent second inference pipeline or raw-frame copying on every stream. Keep a maximum of three candidates per event and a small system-wide extraction concurrency limit. Cancel obsolete work; retry incomplete recording segments within a bounded deadline.
4. **Verify alignment before cropping.** Main/substream field-of-view compatibility must be confirmed. Use recording timestamps to account for stream delay and check nearby frames when necessary. Matching field of view alone does not prove time alignment. If a subject cannot be located reliably in the main image, use a full-scene main image without projected boxes, or retain the exact live evidence for the subject crop. Never present an upscaled substream as high-resolution evidence.
5. **Promote the best valid cover.** A clear subject view outranks a higher-confidence but blank, blurred, occluded, or poorly timed image. Keep the previous valid cover until replacement succeeds. Use the full-resolution original for zoom, a context-preserving crop for the incident card, and separately sized thumbnails for grids. Preserve the other shortlisted frames in Evidence.

## Defaults to test

- Candidate interval: 1 second; shortlist: 3 frames; initial selection horizon: 15 seconds, with later replacement only for a materially better view.
- Background extraction concurrency: 2 jobs system-wide; bounded queue with per-event coalescing.
- Full-resolution original retained; 1280-pixel focus rendition and display-sized thumbnails generated from that original.
- Diagnostic reasons: frame missing, recording pending, uniform image, blur, subject clipped, alignment unverified, and candidate promoted.

## Acceptance checks

Walk-by, approach/departure, stationary person, parked vehicle with passing person, night/IR, reconnect, missing recording, and cropped main/substream views. Verify image/box timestamp pairing, no blank cover promotion, no incorrect subject crop, cache refresh after promotion, bounded work under simultaneous incidents, and no inference FPS regression. Compare chosen covers with the available evidence by inspection.

This document proposes the imagery implementation; the current integration fixes do not yet implement high-resolution extraction or automatic cover selection.
