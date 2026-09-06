# Evaluate identity stability before changing production

Production remains SurvNG Hybrid. Compare evaluates current Hybrid, an offline
Hybrid candidate, TrackTrack and BoT-SORT against the same detector output. The
candidate changes corner-size prediction, strict/relaxed confidence ordering and
one-to-one geometry assignment. It does not redesign appearance disambiguation
or extend production session lifetime.

## Collect a representative set

Choose 20–30 incidents across relevant cameras. Include crossings, partial
occlusions, 3–10 second returns, small/distant objects, box-size jitter, vehicles,
night scenes, and uncomplicated controls. Reserve about one third as a held-out
set; do not tune thresholds against that set. Keep camera, day/night, scenario,
and train/held-out classification in a separate corpus manifest.

In the incident viewer, open Compare, run a profile and **Download replay inputs**.
Each bundle contains exact timestamps, frame dimensions, eligible/continuation
flags, detections and available appearance embeddings. It excludes camera stream
URLs, image pixels and incident mutations. It includes capture configuration and
model paths for provenance; keep the download with the corresponding recording.
The digest identifies exact input content, not proof of origin or authenticity.

The first capture performs detection/appearance work. Downloaded replay requires
no recording decoder or inference worker. History keeps compact evidence, not
full embeddings, and a rerun replaces the prior result for that incident. Use
**Download replay inputs** and **Download results** to preserve every trial. Capture is bounded to 30
seconds in the UI, 600 frames, 100 detections/frame and 32 MiB per bundle.

The optional engines require:

```sh
.venv/bin/pip install -r requirements-ultralytics-tracking.txt
```

Missing engines get an unavailable result. Baseline capture still works. A zero
appearance-input count means this is a geometry-only experiment; enabling an
engine's ReID code does not manufacture useful embeddings. Capture again after
configuring the intended person/vehicle encoder if a ReID comparison is needed.

## Replay without rerunning detection

From the SurvNG checkout and its normal Python environment:

```sh
python -m survng.app.tracking_evaluation gate-replay.json --profile fixed_2fps --output gate-2fps.json
python -m survng.app.tracking_evaluation gate-replay.json --profile fixed_075fps --output gate-slow.json
python -m survng.app.tracking_evaluation gate-replay.json --profile sparse_gaps --output gate-gaps.json
```

`recorded` uses every saved sample. Other profiles select the first saved frame
at or after each target timestamp. A 0.75 FPS target derived from 2 FPS input has
quantized intervals, not exact 0.75 FPS spacing. `sparse_gaps` uses a 2 FPS target
and omits samples in elapsed seconds [5,9) and [15,21). These are dropped samples,
not fabricated empty detections. True empty detections remain empty observations.
Deferred or failed inference aborts capture rather than being labeled no-object.

Ultralytics motion prediction advances per update; its timestamp-based retention
adapter does not make its Kalman model elapsed-time aware. Evaluate sparse-gap
results as a stress test of that integration. They are not proof of equivalence
to Hybrid's elapsed-time motion model. The full production adaptive policy,
capacity waits, shutdown and session lifetime are not simulated by this tool.

Every result includes replay digest, evaluation-source digest, package version,
profile, per-frame confirmed observations and backend diagnostics. Runtime timing
is local tracker wall time and varies between runs; it is not fleet CPU, peak
memory or end-to-end latency. No automatic winner is selected.

## Label and score

```sh
python -m survng.app.tracking_evaluation gate-replay.json --label-template --output gate-labels.json
```

The template is a starting point, not ground truth. Replace each null `identity`
with a stable string such as `person-A`. Inspect the recording, correct boxes,
remove false detections and add objects missed by the detector. Empty `objects`
means the frame was reviewed and has no visible ground-truth objects. Every
selected frame must be labeled, with each identity appearing at most once.
Use the same identity after a true occlusion; distinguish a different entrant.
Frame indexes/timestamps in the replay locate the sampled source observations.

```sh
python -m survng.app.tracking_evaluation gate-replay.json --labels gate-labels.json --profile fixed_2fps --output gate-scored.json
```

The label file must name the replay digest. Metrics are versioned as
`survng_identity_v1_iou_0.5`:

- Class-aware IoU ≥ 0.5 defines valid ground-truth/prediction overlap.
- IDF1 uses a global one-to-one identity assignment and counts unmatched ground
  truth and predictions. It is a fraction from 0 to 1; an entirely empty scene
  has an undefined (`null`) IDF1, not a perfect score.
- ID switches compare successive matched tracker IDs for one identity, including
  matches separated by gaps.
- Fragmentations count tracked → untracked → tracked transitions while the
  ground-truth object is present. A true ground-truth absence is not itself a
  tracker fragment.
- False merges count predicted IDs associated with multiple ground-truth
  identities. This is a diagnostic count, not a standardized MOTChallenge metric.

These explicitly defined metrics are not an official TrackEval benchmark. Keep
identity metrics separate from the existing `fragmentation_proxy`, which can be
nonzero simply because different objects entered at different times. Review
per-frame matches and footage when results disagree with visual evidence.

## Promotion gate and live follow-up

Compare current Hybrid, repaired Hybrid and both alternatives on the same labeled
inputs, then repeat on the held-out clips without retuning. Count errors by clip
and scenario; aggregate IDF1 from summed IDTP/IDFP/IDFN, not a mean of clip scores.
Reject changes that reduce fragmentation by merging distinct objects.

Select a candidate only after identity errors improve across relevant scenes
without unacceptable processing cost. A subsequent live shadow test needs access
to the deployed server, effective settings, and a resource budget. This change
adds no background shadow worker and changes no live incidents or alerts.
Measure CPU, memory, detector contention and recovery latency there before
activation. The outer session's lost-track termination and per-incident ID reset
remain separate changes regardless of the winning tracker.
