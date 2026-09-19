# Native incident imagery

## Implemented flow

Native capture exposes a 640-pixel preview. Initial incident evidence still uses an exact-session/PTS preview when available. A missing preview can leave the initial snapshot empty; background selection now fills it from recordings when a verified subject image is available.

1. **Nominate frames from native tracks.** Evaluate at most one preview per second per event and retain the best three, spaced at least one second apart. Score subject area, confidence, edge clearance, sharpness, and contrast. Reject near-uniform frames and subject crops. Dark but useful night images remain eligible.
2. **Extract existing main recordings at full resolution.** Two background workers share a queue of at most 32 event jobs. Initial work waits 20 seconds for recording availability; completed events can also request selection. Missing recordings retry within a 120-second deadline. No main-stream inference runs continuously per camera.
3. **Verify geometry and subject presence.** Reuse the existing ORB/RANSAC alignment estimator, then local template correspondence. These checks establish spatial correspondence, not proof that an object is present: a single shared native `gvadetect` CPU subprocess confirms the same class at an overlapping location in the actual main image. It processes only nominated images, uses one inference request and two inference threads, and applies configured class confidence thresholds. The live GPU pipeline stays separate.
4. **Commit paired evidence atomically.** Choose a materially better valid cover, retaining its exact main-frame dimensions, timestamp, and verified boxes together. Preserve track history independently. Increment `evidence_revision` and publish an incident update. Archive up to three shortlisted images; preserve the prior cover through the source-observation archive. Never upscale a preview and label it high resolution. Failed verification keeps the existing cover.
5. **Prefer usable incident representatives.** Image availability and verified cover quality take precedence over confidence alone. Existing focus/thumbnail rendering uses the full-resolution original and its own subject coordinates.

The earlier claim that routine track updates overwrite snapshot boxes was incorrect: `EventStore.update_object_tracking` preserves the snapshot objects and updates the separate tracking diagnostic.

## Historical reprocessing

`scripts/reprocess-native-evidence.py --config config.json --hours 2 --report /tmp/native-evidence-report.json` selects native events in an explicit two-hour window and records per-event results. It samples up to twelve timestamps across each saved track history, reads recorded live/main frames, and applies the same verification and promotion rules. It does not create new events or reconstruct discarded track history. Missing recordings and unverified subjects are reported without replacing the old cover.

## Alignment and replay

The prior ORB calibration implementation remains shared and tested. Native cover selection uses it per image pair; this does not certify main-stream replay alignment globally. Tracks mode uses recorded live/substream footage unless compatibility is explicitly known. Replay windows include all saved track timestamps, and bounded history compaction preserves the beginning and end of new long episodes rather than retaining only the latest 150 observations.

## Validation and limits

Regression coverage checks uniform-image rejection, registration, explicit subject verification, atomic cover/history preservation, archived-file retention, and bounded history. Browser coverage plays a 24-second incident through a 10-second recording boundary with track overlays.

Visual inspection remains necessary: spatial registration alone accepted an empty background in an initial pilot, so native main-image detection is mandatory. This is conservative selection, not guaranteed recovery: missing recordings, truncated historical tracks, large timing offsets, or failed registration can leave an event without an upgraded cover. The verifier adds bounded CPU work and one lazily loaded native CPU model; GPU live inference metrics do not include this offline verification work.
