# Storylines & Story Replay

A **Storyline** groups selected incidents into one saved account of a real-world
event. It can include several people, vehicles, cameras, and related actions.
Original incidents and their evidence remain authoritative and independently
accessible. A **Story Replay** plays or exports the selected footage as one sequence.

## Create and edit

Open **Storylines** from the desktop navigation or the mobile More menu. The
Incidents workspace also has **Build Storyline** for its focused incident.

1. Choose a day and optional camera in the evidence browser.
2. Select incidents, including across browser pages, then **Create from selected**.
3. Give the Storyline a title and context. Add more selected incidents with
   **Add to current**.
4. Review each member's relationship: related event, operator-confirmed same
   subject, or context only. Optional notes explain the connection. **Focus**
   chooses one observed subject for replay crops; the default keeps all subjects.
5. Save your edits. Use the up/down controls with **My sequence** to arrange
   a manual replay. **Split out** separates one member into a new Storyline;
   the merge controls combine saved Storylines.

Edits use revision checks. A conflicting edit is rejected instead of overwriting
another operator's work. Refresh to load current evidence before retrying.
Deleted or expired source incidents stay listed as unavailable. Corrected incident
aliases resolve to the current incident without duplicating replay footage.
Deleting a Storyline does not delete original incidents or exported videos.

## Suggested connections

**Suggest related incidents** reuses the existing cross-camera trace, durable
appearance index and configured camera transition routes. It searches at most
eight selected anchors, their bounded trace windows, and returns up to 48 candidates.

- **High:** a confirmed identity link and compatible configured camera route.
- **Moderate:** stronger identity or appearance evidence requiring review.
- **Low:** weak identity or nearby context, which does not establish sameness.

These are qualitative evidence-strength labels, not calibrated probabilities.
Each suggestion shows its reason and optional route. **Add as related event**
adds it only after operator review. **Dismiss** persists for the evidence version;
new evidence can bring the candidate back. No transitive or AI-generated link
silently merges identities. Distinct subjects can belong to the same real-world event.

## Configured AI

**AI title & context** uses the existing enabled **AI analysis & assistant**
provider, API key, base URL and detailed analysis model, falling back to the
configured everyday model. No separate provider settings are needed.

The review sends a labeled montage of at most 12 retained images total, including
incident covers and useful additional evidence images. Selection prioritizes one
usable image per incident before extra views fill remaining slots. Missing covers
fall back to retained gallery images; unavailable or duplicate frames are skipped. Each tile
includes its source camera and actual capture time. Bounded timestamps, labels,
activity and operator context accompany the montage. Gallery detector hints can be
below admission thresholds and do not establish confirmed incident subjects. It
does not send stream URLs, raw appearance vectors or recording files. AI results include evidence links,
observed/possible action labels and proposed connections. These proposals do not
change incident membership or identity. **Use AI title and summary** copies the
text into the editable fields; save to adopt it. Review is requested explicitly,
not incurred continuously in the background.

If source evidence changes, the review is marked stale. Membership edits invalidate
it. With AI unavailable, all manual editing and replay functions still work.

## Replay and export

Save edits, choose options, then **Prepare Story Replay**:

- **Chronological:** a single source-time timeline. Automatic direction favors
  cameras with supported crop observations; up to two simultaneous cameras can
  appear side by side. Remaining source incidents stay accessible separately.
- **My sequence:** follows the saved member order, with explicit cards for source
  time jumps, including backward jumps. Each incident's cameras remain chronological.
- **Automatic zoom:** uses normalized recorded object observations, unions
  simultaneous subjects by default, caps digital zoom at 2x, and interpolates
  supported crop positions. Single-frame evidence or observation gaps longer than
  five seconds use the full view. The beginning establishes the scene before zooming.
- **Full frame:** toggles the browser replay back to the uncropped recording.
- **Split-screen overlap:** shows up to two recorded views during simultaneous activity.

The replay uses retained main recordings. It labels cameras and source UTC time,
shows elapsed-time cards between selected intervals, and lists missing coverage.
The replay duration can be shorter than the original event's wall-clock duration.
It is muted: mixing audio from different cameras is not attempted.

**Export MP4** queues a 1280 × 720, 25 FPS H.264 replay in the existing **Exports**
workspace. It renders camera/time captions, zooms, split-screen views and time-jump
cards. Rendering runs through the existing bounded cancellable export worker,
with protected recording leases, atomic output publication, a frozen replay-plan
manifest, download/playback, protection, labels and ordinary export retention.
Story rendering uses CPU encoding to support the crop and composition filters.
Missing or changed recording coverage causes a clear job failure instead of
silently compressing unexplained gaps; refresh and retry.

A Storyline accepts at most 64 incidents within six hours. Chronological replay
allows up to 256 video shots and one hour of selected footage. Individual episodes
must fit within one hour; use shorter or split incidents for longer investigations.
Manual sequences allow up to 512 total chapters and 70 minutes including time cards.
These limits bound query, rendering and inference work. Browser viewers can read
Storylines and prepare/play replay; editing, AI requests and exports require admin scope.

## HTTP API

- `GET/POST /api/storylines`: list or create.
- `GET/PUT/DELETE /api/storylines/{id}`: detail, replace editable fields, or delete.
- `POST /api/storylines/{id}/merge` and `/split`: atomic membership edits.
- `GET/POST /api/storylines/{id}/suggestions`: evidence candidates and decisions.
- `POST /api/storylines/{id}/ai`: configured AI review.
- `POST /api/storylines/{id}/replay`: generate a bounded replay plan.
- `POST /api/storylines/{id}/export`: queue its frozen plan in Exports.

Every mutation and replay request supplies the current `revision`; deletion takes
it as a query parameter. Suggestions also require their evidence fingerprint.
Replay options are `directed`, `split_screen`, `padding` (0–10 seconds), and
`order` (`chronological` or `member`). No API accepts client-selected file paths or
FFmpeg commands. Storage is an additive `storylines` table in the existing main DB.
