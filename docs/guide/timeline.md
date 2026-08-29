# Timeline & exports

**Timeline** answers: what happened around this time on this camera?

![Timeline workspace with scrubber and related events](images/timeline-workspace.png)

## Scrub recorded video

1. Open **Timeline**.
2. Pick a camera (or an all-camera grid view when available).
3. Choose the day.
4. Drag the scrubber to the time you care about.
5. Switch between **main** and **live/sub** recordings when both exist.

Fine scrubbing may show preview frames while you drag. Playback then resumes from the selected moment.

## Exact time links

You can share or bookmark a Timeline moment. SurvNG carries the requested time in the link and reports the actual decoded time when the player knows it.

## Filters

Object and motion markers on the timeline help you jump between interesting points instead of watching hours of empty driveway.

## Find similar from a frame

When Smart Search is enabled, pause near the subject you want to investigate
and choose **Find similar**. SurvNG freezes the current frame so you can crop the
person, vehicle, or object precisely, then searches indexed incident evidence.
Results remain in an evidence rail while you inspect matches or move to nearby
evidence. Use the previous/next controls to follow the trail, or **Exit** to
return to ordinary Timeline playback.

This search uses the selected crop as visual evidence; it does not create or
modify incidents. See [Search](search.md) for object-based and text searches.

## Exports

**Exports** is the place to save a clip or timelapse for download.

Typical flow:

1. Select a time range on Timeline.
2. Start an export.
3. Watch progress in the background-task status (you can leave the page while it runs).
4. Open **Exports** to download, protect, or delete finished jobs.

Protected exports are kept even when ordinary retention cleanup runs.

### Example

You want a 45-second clip of a package drop:

1. Timeline → camera `Porch` → today → scrub to 14:02.
2. Select from 14:02:00 to 14:02:45.
3. Export as a normal recording clip.
4. Download from **Exports** when the job finishes.

## Related

- [Recordings & storage](storage.md)
- [Incidents](incidents.md)
- [Recording frame API](../recording-frame-api.md)
