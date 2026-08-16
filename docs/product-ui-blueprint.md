# SurvNG product interface blueprint

Status: implemented product contract and release-certification reference.

This document defines the operator workflows, navigation, shared interaction
rules, compatibility requirements, and responsive behavior that must remain
stable while SurvNG adopts the hybrid Command Center and Timeline design.

The visual targets are:

- [Hybrid Command Center](ui-mockups/hybrid-command-center.png)
- [Timeline investigation workspace](ui-mockups/timeline-workspace.png)
- [Mobile Live and Timeline](ui-mockups/mobile-live-timeline.png)

These mockups define hierarchy and interaction density, not literal sample
camera names, dates, or generated imagery.

## Visual system

The mockups are the source of truth for visual hierarchy. SurvNG uses a compact,
monitoring-oriented system rather than a collection of floating dashboard cards:

- Dark is the fresh-install default. Existing explicit theme choices remain
  respected, and the light theme uses the same hierarchy and density.
- The canvas is near-black blue; application chrome and workspaces are separated
  primarily by one-pixel dividers rather than shadows.
- Teal identifies selection, the current time, and primary actions. Green,
  amber, and red are reserved for operational state.
- The type stack is `Inter`, `SF Pro Text`, `Segoe UI`, then the platform
  sans-serif. Normal interface text is 13–14px; metadata is 9–11px; workspace
  headings are 14–20px. Weight, not oversized text, establishes hierarchy.
- Controls use four-pixel corners. Segmented controls share one outer boundary
  and internal dividers. Pills are reserved for compact state and labels, not
  general navigation.
- Desktop workspaces are edge-to-edge within the shell. Rails are 226–306px,
  inspectors are about 278px, and media/evidence receives the remaining width.
- Mobile keeps the same terminology and order of operations while replacing
  vertical rails with horizontal selectors and inspectors with sheets.

The standard visual sequence is **scope → controls → primary evidence → nearby
evidence → details/actions**. Live, Timeline, Incidents, Search, People, Admin,
and operational views all follow this sequence where their workflow permits it.

## Product model

Each primary workspace owns one operator question:

| Workspace | Operator question | Canonical route | Compatibility route |
| --- | --- | --- | --- |
| Live | What is happening now? | `/` | — |
| Incidents | What important event occurred? | `/incidents` | — |
| Timeline | What happened around this time? | `/timeline` | `/recordings` |
| Search | Where did matching activity appear? | `/search` | `/recordings/search` |
| People | Who was seen? | `/people` | `/faces` |
| Admin | How is SurvNG configured and performing? | `/admin` | `/config` |

Live contains two workspaces, **Command Center** and **Overview**. Overview is
not a new global destination. Export Center remains a Timeline subsection.
Motion Audit remains diagnostic evidence in Admin and is not an incident list.

## Shared page anatomy

Every primary workspace uses the same hierarchy:

1. Global application shell and system status.
2. Workspace-specific command bar.
3. Primary workspace.
4. Optional rail or inspector.
5. Context-aware AI assistant.
6. Toasts and persistent background-task status.

The shared component contracts are:

- `WorkspaceHeader`: title-free contextual actions, health, and current scope.
- `FilterBar`: URL-backed filters with one Reset action and visible active state.
- `CameraRail`: camera/group selection; selection never implicitly changes date.
- `ActivityRail`: independently updating event summaries that do not remount media.
- `EvidenceViewer`: progressive clean image, annotations, zoom, and playback lifecycle.
- `Timeline`: authoritative selected epoch, coverage, markers, and range selection.
- `InspectorPanel`: Details, AI, and Related contextual tabs.
- `StatusChip`: consistent healthy, caution, failure, disabled, and pending semantics.
- `BackgroundTaskStatus`: resumable progress for exports, tune-ups, maintenance, and migrations.
- `EmptyState`: explains why content is absent and offers the next valid action.

## Interaction contract

### Selection and navigation

- A single click or tap changes selection and updates the current workspace.
- Navigation occurs only through an explicitly named action such as **Open incident**
  or **View in Timeline**.
- The selected camera, event, epoch, filters, and inspector tab are URL state where
  they must survive a copied link or browser navigation.
- Browser Back restores selection, filters, and scroll position.
- Related evidence changes the inspected item without replacing its anchor incident.
- Changing selection stops prior media playback before showing the next poster.

### Media

- Polling or rail updates must not remount a live or recorded player.
- Posters remain visible until replacement video has decoded a usable frame.
- Incident images load progressively and promote to original media for zoom.
- Annotation geometry becomes visible only after the displayed media dimensions
  and coordinate transform are known.
- Timeline and recording deep links carry an exact requested epoch and expose the
  actual decoded epoch when available.

### Loading, errors, and background work

- Preserve useful content while refreshing; do not replace it with a blank spinner.
- Loading indicators identify the item or operation being loaded.
- Recoverable errors keep the user's scope and offer Retry.
- A disabled action either explains its prerequisite or remains actionable and
  reports the missing prerequisite when invoked.
- Long-running work is resumable, observable outside its originating page, and
  never represented as a modal that must remain open.

### Terminology

- **Incident**: the enclosing security occurrence.
- **Event**: one constituent camera observation.
- **Detection**: one object observation in a frame.
- **Motion Audit**: diagnostic evidence that may not have produced an incident.
- **Timeline**: recorded media and temporal evidence, not a storage file browser.

## Workspace preservation inventory

### Live

Preserve automatic and custom layouts, camera order, mixed-aspect sizing, saved
tile geometry, snapshot-to-live continuity, per-overlay Main/Sub defaults,
WebRTC/MSE fallback memory, active-motion borders, and the non-date-specific
recent incident feed. Server-instance resets must continue to restore live tiles
to Sub + snapshot without changing desktop incident-overlay Main defaults.

### Incidents

Preserve compact paging, semantic search, camera/object/zone/source filters,
Focus/Mosaic session preference, progressive evidence loading, full-resolution
zoom, clean/AI/tracks replay, related incident anchor context, exact event links,
and mobile previous/next navigation.

### Timeline

Preserve single/all-camera playback, Main/Sub selection, indexed-day playback,
fine scrub previews, incident thumbnail windows, object/motion filters, range
export and timelapse, Export Center, exact-time links, and playback retry state.

### Search

Preserve the last five semantic queries, active results when returning from an
incident, camera refinement, reset, result-to-incident links, result-to-Timeline
links, and model/index status.

### People

Preserve People list, all/suggested/unknown review, reference pinning, automatic
advance after review, calibration results, observation paging, and deletion.

### Admin

Preserve every setting, scoped configuration application, camera subnavigation,
Motion Audit, Detection Tune-Up resume state, Telemetry subviews, Maintenance,
Logs, storage progress, and background migrations.

### AI assistant

Preserve conversation history, open state, camera/event/epoch context, exact
evidence links, follow-up suggestions, export/timelapse actions, and guarded
configuration previews.

## Browser-state compatibility

The redesign must migrate or retain these current state families:

- `survng.live*`, `survng.streamMode.*`, and `survng.webrtcFailure.*`
- `survng.incident*`
- `survng.semanticSearch*`
- `survng.detectionTuneup.*`
- `survng.telemetry*`
- `survng.motionAudit*`
- `survng.assistant*`
- `survng.configTab`, `survng.generalSection.v1`, and `survng.cameraSection.v1`
- `survng.theme` and `survng.timeZone`

New state keys require a version suffix and a one-time migration. Compatibility
routes preserve query strings. Base-path installations must use `appUrl()` (or
its eventual shared equivalent) rather than root-relative browser navigation.

## Responsive contract

| Range | Navigation | Main composition | Secondary content |
| --- | --- | --- | --- |
| `>= 1280px` | persistent compact left rail | multi-region desktop workspace | fixed or resizable inspector/activity rail |
| `768–1279px` | collapsible rail | reduced grid or two-region workspace | drawer/overlay inspector |
| `< 768px` | bottom navigation | one primary media surface | horizontal rails and bottom sheets |

Mobile primary navigation is Live, Incidents, Timeline, Search, and More. More
contains People, Admin, status, and appearance controls. The assistant floats
above the safe area and opens full-screen. Arbitrary tile resizing is desktop
only; mobile allows ordering and camera focus. All interactive targets are at
least 44 CSS pixels.

## Accessibility and performance budgets

- WCAG AA color contrast for text, controls, and non-color status cues.
- Complete keyboard operation and visible focus for core desktop workflows.
- Focus moves to the new workspace heading after navigation and returns to the
  invoking control when drawers close.
- No polling response may recreate an active media element.
- Long rails are paged or virtualized.
- Thumbnail and scrub-preview caches are bounded and revocable.
- Drag/resize feedback targets a smooth 60 fps on the supported desktop.
- Mobile layouts are certified against dynamic Safari viewport and safe-area changes.

## Campaign acceptance gate

Every implementation campaign must finish with:

1. Feature-preservation check against this inventory.
2. Focused unit tests for extracted state and layout logic.
3. Production frontend build.
4. Desktop browser workflow verification.
5. Mobile viewport and Safari-safe-area verification for changed surfaces.
6. `git diff --check` and a discrete commit.
7. UX review against the same navigation, media, feedback, and terminology rules.

The final campaign removes superseded components, styles, state keys, and route
branches after compatibility migration is verified. It must not leave a second
legacy interface hidden beside the new one.
