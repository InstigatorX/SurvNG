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

When motion qualifies, SurvNG can run a model that looks for known shapes — for example a person, car, or animal. Detection is optional. Without it, SurvNG still records video and can store motion-related notes.

## Incident

An **incident** is a stretch of activity SurvNG kept for review. It groups related camera observations, keeps a representative picture, and links you to nearby recorded video.

## Event

An **event** is one camera observation inside an incident — one moment from one camera. The Incidents workspace is built from these observations.

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
