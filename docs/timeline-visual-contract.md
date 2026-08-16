# Timeline visual contract

The approved Timeline mockup is the visual source of truth. Existing playback,
indexing, exact-time linking, export, and incident semantics remain the product
contract; the campaigns below reorganize those capabilities around the target
operator workflow.

## Desktop hierarchy

1. A 244px camera rail owns scope, search, camera groups, online state, and
   favorites. It spans the full Timeline workspace height.
2. A 52px command bar owns the selected camera, date navigation, Today, Main/Sub,
   playback speed, and Export. Timeline/Export-Center navigation must not consume
   the primary command row.
3. The media stage uses one selected-camera hero with synchronized companion
   cameras. Selecting a companion changes the hero without changing the epoch.
4. A compact filter row precedes a full-day timeline with distinct Object and
   Motion lanes, an authoritative teal playhead, and an optional thumbnail rail.
5. The lower investigation region follows `selected incident → related events →
   inspector`. Details, AI, and Related are inspector tabs, not separate pages.

## Responsive hierarchy

- Tablet retains camera scope plus the media/timeline surface; the inspector is
  a drawer or stacked panel and never crushes primary evidence.
- Mobile uses date and camera selectors, one primary media surface, a horizontal
  companion rail, the same temporal lanes, and a compact selected-incident card.
- Desktop rails become horizontal selectors rather than duplicated data trees.

## Geometry tokens

Timeline geometry is owned in `frontend/src/styles.css`:

- `--timeline-camera-rail-width: 244px`
- `--timeline-command-height: 52px`
- `--timeline-inspector-width: 280px`
- `--timeline-evidence-height: 188px`
- `--timeline-media-gap: 4px`

Panels use one-pixel dividers and four-pixel control corners. The media stage is
near-black. Inter remains the primary typeface; teal means current selection and
time, while blue/orange distinguish Object and Motion lanes.

## Preservation gates

- Camera selection, incident refresh, filter changes, and inspector changes do
  not recreate an unaffected active media element.
- Exact URL camera/date/source/epoch state, indexed-day playback, open-segment
  updates, Main/Sub fallback, synchronized all-camera playback, scrub preview,
  range export, timelapse, and Export Center remain functional.
- The playhead is the single authoritative epoch for hero, companions, lanes,
  thumbnails, selected evidence, deep links, and assistant context.
- Changing selected evidence stops prior playback before replacing it. All
  internal media/navigation links remain base-path safe.
- All core actions remain keyboard operable and mobile targets remain at least
  44 CSS pixels.
