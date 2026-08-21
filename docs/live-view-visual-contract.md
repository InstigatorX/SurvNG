# Live view visual contract

The approved Live mockups are the source of truth for hierarchy, density, and
responsive behavior. Sample camera names and imagery are illustrative; the
following geometry and interaction rules are product requirements.

## Desktop composition

- A 176px navigation rail and 58px system bar frame an edge-to-edge command
  center. The page does not use floating dashboard-card margins.
- A 52px Live command bar owns camera scope, density, layout mode, fullscreen,
  and workspace actions.
- The primary camera surface receives all remaining width after a 306px Recent
  Activity rail. Camera tiles use four-pixel gutters and compact black media
  chrome.
- Automatic density choices are `Fit`, `4`, `6`, `9`, `16`, and `25`. Every
  automatic tile uses the same 16:9 frame; `Fit` chooses the largest equal-size
  grid that shows the current camera set, while numbered choices establish a
  predictable camera count.
- Recent Activity uses `All`, `Person`, and `Vehicle` quick filters, a separate
  advanced-filter action, one incident per row, and an explicit View all action.
- Selecting recent activity is local. Opening an incident is always an explicit
  action and never remounts a live camera.

## Camera framing

- Equal tile geometry never changes recording, detection, snapshot, or source
  resolution. Media is presentation-cropped inside the tile.
- Each camera stores independent Main and Sub framing. `fit` is `cover` (fill
  and crop) or `contain` (show the entire frame); `focal_x` and `focal_y` are
  normalized percentages; `zoom` is bounded from 1× through 3×.
- The same framing is applied to the idle poster, hover-promoted live stream,
  fallback transport, and full Live overlay. Source fallback uses the framing
  belonging to the source actually displayed.
- Framing is edited under Admin > Cameras > Settings using a still preview. It
  is a scoped configuration save and never restarts a camera or recorder.
- Detection zones are not framing controls. The UI may reuse their normalized
  geometry conventions, but changing security policy must never silently
  change Live presentation (or the reverse).

## Mobile composition

- Live starts with one primary camera, then a compact secondary-camera rail,
  followed by Recent Activity. One stable keyed camera tree owns all media.
- Camera selection changes focus without changing the meaning of tapping the
  video, which always opens the existing live overlay.
- Press-and-hold on a camera (touch or stylus) opens a transient live preview
  with enter motion; releasing closes it with exit motion. A normal tap still
  opens the sticky live overlay. Mouse pointers keep click-to-open only.
- Mobile uses the same names and status colors as desktop. Controls are at least
  44 CSS pixels and clear the bottom navigation and safe area.

## Visual tokens

The implementation owns these values in `frontend/src/styles.css`:

- `--workspace-rail-width: 176px`
- `--workspace-topbar-height: 58px`
- `--live-command-height: 52px`
- `--live-activity-width: 306px`
- `--live-tile-gap: 4px`
- `--workspace-control-radius: 4px`

Inter is the primary typeface. Teal is selection and action; green, amber, and
red are operational state only. Workspace separation uses one-pixel rules,
not elevated cards or large rounded containers.

## Preservation gates

- Polling, activity selection, density changes, and camera focus do not remount
  an unaffected live media element.
- Automatic/custom layout state, camera order, measured aspect ratios, stream
  source, transport fallback memory, and server-instance reset behavior remain
  compatible with existing stored state.
- Desktop, tablet, and mobile share one 760px composition boundary until the
  product-wide 768px migration is completed as a separate compatibility change.
- Base-path-safe links and all existing incident, overlay, and assistant context
  behavior remain intact.
