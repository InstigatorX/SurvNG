# Concepts

These are the main terms SurvNG uses in the interface and this guide.

## Camera

A camera is a device that sends video to SurvNG over your network. SurvNG works with standard stream URLs (usually RTSP) and, when available, camera motion notices such as ONVIF events.

**Example:** “Front Door” and “Driveway” are two cameras.

## Stream

A stream is the live video feed from a camera. Many cameras offer two streams:

- **Main** — higher quality; SurvNG records this for history and detailed review
- **Sub** (live) — lighter feed; SurvNG prefers this for live viewing and quick checks

Main is the archive copy; sub is usually the “watching right now” copy.

## Recording

SurvNG can save video continuously, not only when something happens. That history is what you scrub in **Timeline**. Short clips can also be exported for sharing or evidence.

## Motion

Motion means something in the picture changed. Cameras often send a motion notice themselves. SurvNG can also watch the video and decide that motion is worth a closer look. Motion alone is not yet a finished security incident.

## Object detection

When detection is enabled, SurvNG looks for known shapes — for example a person,
car, or animal — on the live stream. Detection is optional. Without it, SurvNG
still records video and can store motion-related notes.

## Observation

An **observation** is one camera sample at a moment in time: the set of labeled
boxes SurvNG saw on that frame (and their zone membership). Observations do not
require tracker IDs.

## Participant

A **participant** is a labeled object role inside an incident (person, car, and
so on). Several participants can share one incident for its whole time range.

## Incident

An **incident** is a stretch of related activity SurvNG kept for review. It has
a start and end time, the participants that appeared during that span, an ordered
sequence of observations, a representative picture, and links into recorded video.
Incidents are stored as first-class records; they are not invented by grouping
nearby events after the fact.

## Event

An **event** is the compatibility storage/API row behind an incident while older
clients still expect event IDs. Prefer **incident** and **observation** when
talking about product behavior.

## Zone

A **zone** is a shape you draw on the camera picture. Use zones to focus on a doorway or driveway, or to ignore foliage and other nuisance motion.

## Timeline

**Timeline** is the recorded-video workspace. Pick a camera and a time, then scrub through what was saved. It is a time-based player, not a storage file browser.

## People

When face recognition is enabled, SurvNG can group face sightings and let you name a person. Matches stay reviewable until you confirm them.

## Admin

**Admin** is where you add cameras, turn detection on, set storage limits, connect Home Assistant, and check whether the system is healthy.

## Assistant

The sparkle button opens a read-only helper. It can explain status, search incidents, and review evidence. It cannot delete recordings or change settings unless you explicitly confirm a recommendation SurvNG itself calculated.
