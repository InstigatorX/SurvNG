# Timeline visual contract

The Timeline workspace is a playback and investigation tool organized around a
selected hero camera, linked companions, and a vertical time rail that scrubs
the hero. Existing playback, indexing, exact-time linking, export, and incident
semantics remain the product contract.

## Desktop hierarchy

1. A 52px command bar owns camera selection and the view switch (Timeline,
   Catch up, Cameras). The selected camera name is a searchable picker, not a
   permanent camera rail. Timeline/Export-Center navigation must not consume the
   primary command row.
2. The media stage is hero-dominant: the selected camera uses approximately
   70–80% of available desktop width. Linked companions from camera-route logic
   occupy a compact vertical rail on the remaining width. About three companions
   are visible without scrolling; additional linked cameras scroll vertically.
   Selecting a companion promotes it to hero without changing the playhead epoch.
3. A player HUD under the hero carries transport (back/forward 10 s, play,
   mute, AI overlay), Today, playback speed, the day picker with previous/next
   day, Find similar, previous/next event, Nearby, Export, and Main/Sub.
4. The view beside or below the hero is one of:
   - **Timeline**: the Event Rail, a vertical time axis (newest at the top) with
     a fixed teal playhead line, event cards, a zoom (15m, 1h, 6h, 24h), an
     event-type filter (All, People, Vehicles, Motion) and a 24 h edge strip
     with a loupe for long jumps.
   - **Catch up**: the day's events as a condensed story, with quiet stretches
     collapsed.
   - **Cameras**: one lane per camera over the visible window (rows on wider
     screens, scrubbable columns on phones).
5. The lower investigation region (Nearby) is contextual and opened from the
   HUD. It shows nearby evidence for the selected event without leaving
   Timeline. All cameras has no Nearby region.

There is no permanent desktop camera rail and no camera-page navigation. Other
cameras are reached through the searchable command-bar picker.

## Time navigation

The Event Rail scrubs: scrolling, dragging, the wheel, or the arrow, Page and
Home/End keys move time under the playhead line, and a scroll that settles
commits a seek. The rail zoom (`rail` in the URL, in minutes) sets how much time
one rail height shows. While scrubbing, the player shows the frame under the
line (see [Recording frame API](recording-frame-api.md)).

`windowHours` (1, 2, 4, 8, 12, or 24 hours) is the visible window around the
playhead used by the Cameras rows, the export bounds and the Nearby radius. It
follows the playhead during playback.

## Responsive hierarchy

- Tablet keeps a large hero. Companions may become a horizontal strip. The
  Nearby region must not crush primary evidence.
- Mobile uses the same compact native camera select as Live and a native date
  field in the command bar, one hero with the HUD, and the Event Rail below it
  with the edge strip and loupe at its side. The rail views hide the companion
  strip on phones, and phones do not offer All cameras. Desktop keeps the
  searchable picker.
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
time, while blue/orange distinguish object and motion events.

## Preservation gates

- Camera selection, incident refresh, filter changes, and Nearby changes do
  not recreate an unaffected active media element.
- Exact URL camera/date/source/epoch/filter/window/speed/view/rail state, indexed-day
  playback, open-segment updates, Main/Sub fallback, synchronized companion
  playback, scrub frames, range export, timelapse, and Export Center remain
  functional.
- Retired URL state is ignored rather than honoured: `view=classic` opens the
  rail, and the old `objects` / `motion` lane parameters show every event.
- Legacy `camera=all` resolves to a concrete hero camera and does not crash.
- The playhead is the single authoritative epoch for hero, companions, the rail,
  selected evidence, deep links, and assistant context.
- Changing selected evidence stops prior playback before replacing it. All
  internal media/navigation links remain base-path safe.
- All core actions remain keyboard operable and mobile targets remain at least
  44 CSS pixels.
