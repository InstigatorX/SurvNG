# Timeline visual contract

The Timeline workspace is a playback and investigation tool organized around a
selected hero camera, linked companions, and a continuously scrollable temporal
viewport. Existing playback, indexing, exact-time linking, export, and incident
semantics remain the product contract.

## Desktop hierarchy

1. A 52px command bar owns camera selection, date navigation, Today, Main/Sub,
   playback speed, and Export. The selected camera name is a searchable picker,
   not a permanent camera rail. Timeline/Export-Center navigation must not
   consume the primary command row.
2. The media stage is hero-dominant: the selected camera uses approximately
   70–80% of available desktop width. Linked companions from camera-route logic
   occupy a compact vertical rail on the remaining width. About three companions
   are visible without scrolling; additional linked cameras scroll vertically.
   Selecting a companion promotes it to hero without changing the playhead epoch.
3. A compact filter/scale row precedes a continuously pannable timeline. Object
   and Motion lanes stay distinct. The teal playhead is the authoritative epoch.
   An optional thumbnail rail remains available.
4. The lower investigation region is contextual. With no selected event it
   collapses to a short prompt so video and timeline keep the height. With a
   selected event it reveals Details / AI / Related without leaving Timeline.

There is no permanent desktop camera rail and no camera-page navigation. Other
cameras are reached through the searchable command-bar picker.

## Time navigation

`windowHours` is the visible time scale (1, 2, 4, 8, 12, or 24 hours), not a
page size. Horizontal pan/scroll moves `viewportStartEpoch` / `viewportEndEpoch`
only. `playheadEpoch` changes only from explicit seek, scrub, or event
selection. While playback is following, the playhead stays in the middle region
of the viewport; a manual pan suspends follow and offers Return to playhead.

## Responsive hierarchy

- Tablet keeps a large hero. Companions may become a horizontal strip. The
  inspector must not crush primary evidence.
- Mobile uses the searchable camera selector, one hero, a horizontal companion
  strip, a horizontally navigable timeline, and stacked event details when an
  event is selected.
- Export Center may still use a camera list; that rail is not part of Timeline
  playback.

## Geometry tokens

Timeline geometry is owned in `frontend/src/styles.css` and refined in
`frontend/src/timeline/timeline.css`:

- `--timeline-command-height: 52px`
- `--timeline-inspector-width: 280px`
- `--timeline-evidence-height: 188px`
- `--timeline-media-gap: 4px`
- companion rail: `clamp(148px, 22vw, 280px)` on desktop

`--timeline-camera-rail-width: 244px` remains for Export Center only. It is no
longer a Timeline playback requirement.

Panels use one-pixel dividers and four-pixel control corners. The media stage is
near-black. Inter remains the primary typeface; teal means current selection and
time, while blue/orange distinguish Object and Motion lanes.

## Preservation gates

- Camera selection, incident refresh, filter changes, and inspector changes do
  not recreate an unaffected active media element.
- Exact URL camera/date/source/epoch/filter/window/lanes/speed state, indexed-day
  playback, open-segment updates, Main/Sub fallback, synchronized companion
  playback, scrub preview, range export, timelapse, and Export Center remain
  functional.
- Legacy `camera=all` resolves to a concrete hero camera and does not crash.
- The playhead is the single authoritative epoch for hero, companions, lanes,
  thumbnails, selected evidence, deep links, and assistant context.
- Changing selected evidence stops prior playback before replacing it. All
  internal media/navigation links remain base-path safe.
- All core actions remain keyboard operable and mobile targets remain at least
  44 CSS pixels.
