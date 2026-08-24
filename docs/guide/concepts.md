# Basic ideas

A few everyday words make SurvNG easier to use. None of them require a technical background.

## Camera

A camera is a device that sends video to SurvNG over your network. SurvNG does not need a special brand, but it does need a video address (usually an RTSP link) and, when available, a way for the camera to announce motion.

**Example:** “Front Door” and “Driveway” are two cameras.

## Stream

A stream is the live video feed from a camera. Many cameras offer two streams:

- **Main** — higher quality; SurvNG records this for history and detailed review
- **Sub** (live) — lighter feed; SurvNG prefers this for live viewing and quick checks

Think of main as the archive copy and sub as the “watching right now” copy.

## Recording

SurvNG can save video all the time, not only when something happens. That continuous history is what you scrub in **Timeline**. Short clips can also be exported for sharing or evidence.

## Motion

Motion means “something in the picture changed.” Cameras often send a motion notice themselves. SurvNG can also watch the video and decide that motion is worth a closer look. Motion alone is not yet a finished security incident.

## Object detection

When motion looks interesting, SurvNG can run a model that looks for known shapes — for example a person, car, or animal. That step is optional. Without it, SurvNG can still record video and store motion notes.

## Incident

An **incident** is SurvNG’s way of saying “this stretch of activity mattered.” It groups related camera observations, keeps a representative picture, and links you to nearby recorded video.

## Event

An **event** is one camera observation inside an incident — one moment from one camera. Most people browse **incidents**; events are the building blocks underneath.

## Zone

A **zone** is a shape you draw on the camera picture. You can say “only create incidents when a person appears near the gate,” or “ignore the tree that waves in the wind.”

## Timeline

**Timeline** is the recorded-video workspace. Pick a camera and a time, then scrub through what was saved. It is not a file browser; it is a time-based player.

## People

When face recognition is enabled, SurvNG can group face sightings and let you name a person. Matches stay reviewable until you confirm them.

## Admin

**Admin** is where you add cameras, turn detection on, set storage limits, connect Home Assistant, and check whether the system is healthy.

## Assistant

The sparkle button opens a read-only helper. It can explain status, search incidents, and review evidence. It cannot delete recordings or change settings unless you explicitly confirm a narrow recommendation SurvNG itself calculated.
