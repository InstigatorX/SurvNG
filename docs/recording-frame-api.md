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
  up to five minutes either side.

The page runs two requests at a time with no queue: when one finishes, the
next is for where the playhead is then, and after that for up to two segments
ahead in the direction of the scrub. While the playhead's own segment loads,
the nearest loaded strip within ten minutes stands in. A failed request is
tried again after a second, then two, four and eight; one with no answer is
given up after ten. A gap is remembered for ten seconds, over the range the
server stated, or for that one second if it stated none.
It keeps at most 24 strips in memory; strips from the sub and main streams,
which are cut at different times, are kept side by side. Every Timeline layout reports its scrub position: the rail,
the camera columns, the day strip's loupe and the classic scrubber. After a
scrub that ends in a seek, the released time keeps loading and its frame stays
up until the player has seeked there and shows the exact full-resolution
frame.
