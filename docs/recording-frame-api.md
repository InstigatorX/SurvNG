# Recording frame API

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
