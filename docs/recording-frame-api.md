# Recording frame API

This endpoint is the public arbitrary-time extraction contract. It is separate
from the incident pipeline's live provisional check and recorded temporal
refinement. See [Incident evidence data path](incident-evidence-data-path.md)
for how SurvNG selects and promotes incident evidence internally.

SurvNG can return a clean recording frame near an arbitrary UTC epoch:

```text
GET /api/cameras/{camera_id}/recordings/preview.jpg?epoch={unix_seconds}&source=main&width=1280&exact=true
```

Use `exact=true` when the decoded frame time matters. The JPEG response then
includes:

- `X-SurvNG-Requested-Timestamp`: requested Unix epoch, with microsecond
  precision.
- `X-SurvNG-Actual-Timestamp`: Unix epoch of the frame FFmpeg actually decoded.
- `X-SurvNG-Timestamp-Source: source_pts`: confirms that the actual timestamp
  came from the decoded frame PTS.

If FFmpeg cannot report source timing, the image can still be returned with
`X-SurvNG-Timestamp-Source: requested_offset`; the actual-timestamp header is
then omitted. Consumers must not claim exact alignment in that case.

`source` accepts `main` or `live`. `width` accepts 320 through 1920 and keeps
the recording's aspect ratio. The endpoint returns `404` when the local
recording index has no segment covering the requested epoch. API authentication,
when enabled, requires a bearer token with the `read` scope.

Ordinary scrub previews omit `exact=true` and may be reused from a five-second
cache bucket. Their requested timestamp is reported, but they intentionally do
not make an exact decoded-time guarantee.

## Scrub preview strips

```http
GET /api/cameras/{camera_id}/recordings/preview-strip.jpg?epoch={unix_seconds}&source=main&width=320
```

Returns one JPEG holding a grid of frames from the recording segment that
contains `epoch`: one frame for each second (a segment longer than 60 s is
thinned to a tile every few seconds, about 60 in all). The Timeline shows these in the player while the
playhead is dragged, so moving within a loaded segment needs no request.

- The camera's sub stream is used when it was recorded at that time, whatever
  `source` says: it shows the same scene and decodes several times faster.
  Otherwise, or when that file can no longer be served, the requested source
  is used. `X-SurvNG-Strip-Source` reports which.
- `X-SurvNG-Strip-Start` and `X-SurvNG-Strip-End` are the wall-clock bounds of
  that segment. Tile `i` (row-major, `X-SurvNG-Strip-Columns` by
  `X-SurvNG-Strip-Rows`) is the last frame at or before
  `start + i * X-SurvNG-Strip-Step`; if the video starts a little into the
  file, tile 0 is its first frame. Only the first `X-SurvNG-Strip-Count` tiles
  hold a frame; the rest of the grid is blank, and a time past the last tile
  is shown by the last tile.
- A frame larger than about one megapixel is decoded at its keyframes only, so
  on a camera with a keyframe interval longer than a second, a tile can show
  the keyframe before its second, and the tiles end at the last keyframe.
  Smaller streams are decoded in full.
- `width` is the tile width, 160 to 640.
- Such a large frame is decoded and scaled on the GPU through VAAPI where
  hardware acceleration is not `off` and VAAPI works (measured at about 0.2 s
  against 0.5 s on the CPU for 2560x1920). If that fails or takes more than
  5 s, the CPU builds the strip and the GPU is not tried again, for any
  camera, for five minutes.
- A strip is built on first request and kept in the preview cache. Two builds
  run at once. Each holds a preview-process media session, a resource of its
  own: not one of the two transcode slots that event clips, exports and
  HomeKit share, and not a JPEG encoder, which live MJPEG streams hold. A busy
  generator answers `429` with `Retry-After`.
- A time with no recording is `404`, with `X-SurvNG-Gap-Start` and
  `X-SurvNG-Gap-End` saying how far the gap reaches on both streams, looking
  up to five minutes either side and not past the present (or one second past
  the time asked about, if that is later).

The page runs two requests at a time with no queue: when one finishes, the
next is for where the playhead is then, and after that for up to two segments
ahead in the direction of the scrub. While the playhead's own segment loads,
the nearest loaded strip within ten minutes stands in. A segment whose request
failed is asked for again after a second, then two, four and eight; that wait
applies within a segment's length of the time that failed and holds back
nothing further away. A request with no answer is
given up after ten seconds. A gap is remembered for ten seconds, over the range the
server stated, or for that one second if it stated none.
It keeps at most 24 strips in memory; strips from the sub and main streams,
which are cut at different times, are kept side by side. Every Timeline layout reports its scrub position: the rail,
the camera columns and the day strip's loupe. After a
scrub that ends in a seek, the released time keeps loading and its frame stays
up until the player has seeked there and shows the exact full-resolution
frame.

## Frame stepping

The Timeline's frame-by-frame controls use two endpoints. "One frame" means
the next frame the file holds: cameras run at rates such as 19.84 or
20.07 fps, so frame times are not a fixed grid.

```text
GET /api/cameras/{camera_id}/recordings/frames.json?epoch={unix_seconds}&source=main
```

Lists every frame of the recording segment that holds `epoch`:

```json
{
  "source": "main",
  "start": 1788955200.0,
  "end": 1788955209.98,
  "frames": [1788955200.013, 1788955200.063403, "…"],
  "previous": 1788955199.999,
  "next": 1788955209.98
}
```

- `frames` are wall-clock times (Unix epoch, to the microsecond) in display
  order, from the file's packet index; nothing is decoded. Only frames inside
  `[start, end)` are listed: a frame the file holds past the indexed end
  belongs to the next segment's span. A video track that starts after the
  audio simply has its first frame later than `start`.
- `previous` and `next` are times to ask about for the segment just before
  and just after this one, or `null` where there is no recording to step
  into: a neighbour counts only when its index times are within one second
  of this segment's. A `next` of `null` on the newest segment can change once
  the following segment is indexed; the response is not cached.
- A time with no recording is `404`. Two lists are read at a time; a busy
  server answers `429` with `Retry-After`.

```text
GET /api/cameras/{camera_id}/recordings/frame.jpg?epoch={unix_seconds}&source=main&width=1280
```

Returns the exact frame on screen at `epoch` (the last frame at or before it;
the first frame if `epoch` precedes it), normally a time taken from
`frames.json`. Headers as for `preview.jpg?exact=true`:
`X-SurvNG-Requested-Timestamp`, `X-SurvNG-Actual-Timestamp` (the decoded
frame's own time) and `X-SurvNG-Timestamp-Source: source_pts`. `width` is 320
to 1920 and never enlarges the recording. A time with no recording is `404`.
If the decoder does not produce that frame (a damaged stream), the answer is
`500`: another frame is never served in its place.

- Frames are decoded a run at a time: the GOP holding the frame (from its
  keyframe to the next, at most 30 frames; a longer GOP is cut into slices of
  30, each decoded from the keyframe) in one FFmpeg process, through the shared
  recorded-frame decoder. The whole run is kept in the preview cache, so the
  other frames of that GOP are served without decoding.
- As for preview strips, a run is decoded and scaled on the GPU through VAAPI
  where hardware acceleration is not `off` and VAAPI works. If the GPU fails
  or takes more than 5 s, the CPU decodes the run, and neither runs nor strips
  try the GPU again, for any camera, for five minutes. A failure that is about
  the file (missing, unreadable, a damaged stream), or a GPU run with frames
  missing, also goes to the CPU but leaves the GPU in use. A frame of unknown
  size is always decoded on the CPU.
- One run is built at a time. Each holds a preview-process media session
  (shared with preview strips), not one of the transcode slots that event
  clips, exports and HomeKit use. A busy generator answers `429` with
  `Retry-After`.
- Runs count against the preview cache's size and age limits like any other
  preview image; every 32 MB of new runs brings the cache's maintenance pass
  forward. Serving any frame of a run refreshes all of its files, so the
  oldest-first eviction tends to take a run as a whole; if part of a run is
  evicted anyway, the next request for it decodes the run again.
