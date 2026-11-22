# Motion and object detection pipeline review (2026-11-22)

Scope: motion qualification (EMA), trigger and decision handling, live and
recorded object detection, recorded-frame decode, and the tracking entry
points. Code and git history are the source of truth; `VIDEO_PIPELINE.md` was
treated as a hint only and is partly out of date (see "Documentation drift").

Not reviewed in depth: tracking-session internals, inference-supervisor
scheduling, recorded temporal consensus, scene/zone admission, face and ReID.
This review was performed by a single agent without an independent reviewer.

## How the pipeline runs today

Default mode is `camera_rescue` (`config.py`, `MotionQualificationConfig.mode`).

- An ONVIF notice skips visual validation (`motion_decisions.py`,
  `camera_primary_fast_path`). It goes to a fast object check on the live frame
  and then a durable pass over the main recording.
- EMA runs continuously as a backup trigger: about 2 Hz in `camera_rescue`
  (`camera_mode_background_fps`) and at `sample_fps` in `adaptive`. Qualified
  EMA motion goes through the episode controller and then the same object
  checks.
- Each incident passes through three durable queues: motion trigger, detection
  job, scene tracking job. Tracking defaults to `survng_sparse_identity`.

## Findings

### Performance (measured)

1. **EMA re-processed its whole color window on every cycle.**
   `MotionAnalysisService.analyze_continuous` passed the entire color ring to
   the stateful pipeline each cycle. That ring holds 4 frames
   (`color_frames`, `maxlen=4`); the 36-entry gray ring is not passed. The
   background, threshold, morphology and connected-component stages
   recomputed the older frames, which the tracker then discarded, and the
   illumination filter recomputed LAB/Sobel evidence for every pair. Older
   frames were also compared against the *current* background rather than the
   one they were first seen against.

   Synthetic 320x180 benchmark (4-core container, one moving object, 4-frame
   ring):

   | Configuration | Whole ring | Anchor plus new frame |
   | --- | ---: | ---: |
   | Default (illumination filter off) | 4.65 ms per cycle | 3.60 ms per cycle |
   | Illumination filter on, sustained motion | 23.9 ms per cycle | 15.0 ms per cycle |

   An earlier draft of this review modelled a 36-frame ring and reported
   6–14x savings; that overstated the real cost. The measured saving is about
   23% by default and 37% with the illumination filter on.

2. **EMA-triggered fast checks use the 320-pixel motion frame.** The evidence
   ring stores the downscaled analysis frame, and EMA triggers require it
   (`evidence_frame_required`). Small subjects therefore wait for the
   main-recording pass. Whether this is intended is not documented.

3. **Main-recording evidence waits for the segment to close.** Recordings are
   plain 10-second MP4 segments, which are unreadable until finalized. This
   adds roughly 1–11 s before refinement can start and compounds finding 2.

### Weaknesses and convoluted design

- **Evidence fusion and observation graphs are effectively unreachable.**
  Validation rejects ONVIF as a fusion source in every mode and rejects
  `depth_object`, while the stage catalog still offered `["depth_object"]` as
  the default sources. The UI only writes `audit` or `bypass` with no sources,
  so the layer acts as one validation on/off switch. Setting `any`, `all` or
  `weighted` by hand left no usable sources, and the fail-open default then
  accepted every motion with score 1.0. Nothing reads the ONVIF evidence the
  observation graph stores.
- **The pipeline framework is heavier than its workload.** It supports
  parallel stage groups, a thread pool, context fork/merge and per-thread
  lifecycle tracking, but runs one fixed sequence of about ten stages.
- **Trigger kinds are encoded as topic strings** (`adaptive/visual_backup`,
  `adaptive/active_followup`, `manual...`) and expanded into boolean flags in
  `MotionDecisionOrchestrator._process_batch`.
- **Collaborators are patched after construction.**
  `CameraWorker._activity_attributor` assigns attributes on four objects that
  are later read with `getattr(..., None)`.
- **Production code adapts to test doubles**: call-time
  `inspect.signature` checks for "legacy mocks", `hasattr` probing of stores
  with an in-memory job-store fallback, and `getattr(config, name, default)`
  for declared configuration fields.
- **Four separate FFmpeg decoders read recorded video**, each with its own
  hardware fallback, timeout and stderr handling (`camera.py`,
  `motion_pipeline/object_detection.py` twice, `recorded_video_frames.py`).
- **Oversized functions**: `ObjectTrackingSession._run` (875 lines),
  `MotionDecisionHandler._handle_with_provider` (661),
  `RecordedMotionObjectDetector._detect_with_budget` (449),
  `MotionAnalysisService.consider_visual_backup` (433),
  `MotionIncidentService._run_refinements_until_error` (390),
  `_temporal_consensus` (377).
- **Small defects**: an equal capture timestamp triggered a full motion-state
  reset (strict `>` in the clock-discontinuity check); the ring size was
  computed in two places; detection calls pass through four layers that add
  nothing.

### Dead or test-only code

- The fixed-threshold motion graph (`default_motion_stage_configs`, its
  stages in `image_stages.py`, and `qualify_motion` / `score_motion_masks` in
  `motion.py`) is reachable only from tests or a hand-edited configuration.
  `build_default_motion_pipeline` had no callers.
- `MotionEventState`, `MotionEventPhase`, `TriggerDecision` and the context
  fields `event_state` and `decision` are never written.
- `update_guided_fusion` and `analysis_preset_selections` are test-only.
- The ignored `motion_evidence` argument of `RecordedMotionObjectDetector`
  and the gray-frame fallback in `analyze_continuous` do nothing.
- `OpenVinoDetector._parse_yolo_output_scalar_reference` is a test oracle
  shipped in production.

### Documentation drift

- `VIDEO_PIPELINE.md` (last reviewed 2026-08-24) describes a fixed-threshold
  rollback preset and `legacy_qualifier`, describes the classic frame-difference
  scorer, omits `camera_rescue` from the trigger modes, and describes Hybrid as
  the production tracker plus Compare/Deep OC-SORT, which were removed on
  2026-11-12.
- `object_track/sparse_identity.py` described itself as offline-only although
  it is the default tracker.

## Proposal

| Tranche | Scope | Risk |
| --- | --- | --- |
| T1 | Process each EMA frame once: pass only new frames plus one anchor, keep per-frame illumination evidence instead of recomputing it, size caches to the ring. Validate with a parity test against the full-ring path and the benchmark. | Low–medium; contained to motion qualification |
| T2 | Remove dead and vestigial code: the fixed-threshold graph, dead context types, test-only helpers, the YOLO test oracle, ignored arguments. Restrict evidence fusion to the supported audit/bypass shape so hand edits cannot silently disable validation. | Low; current-only configuration policy applies |
| T3 | Run EMA fast checks on the full-resolution live frame for the same capture sequence. | Behavior change; needs recall checks |
| T4 | Evaluate fragmented or shorter recording segments for earlier refinement; consolidate the FFmpeg decoders. | Cross-subsystem; needs independent review |
| T5 | Typed trigger kinds, constructor injection, direct config access, remove pass-through layers, split oversized functions. | Incremental refactors |
| T6 | Rewrite the motion, tracking and trigger-mode sections of `VIDEO_PIPELINE.md` from code. | Documentation |

Suggested order: T1 and T2, then T3, T5 and T6, with T4 last.

## Implementation status

- **T1 (done):** continuous analysis passes the newest processed frame as an
  anchor plus newer frames, tied to the qualification runtime generation so a
  reset replays the retained window. The illumination filter keeps per-frame
  history and computes each pair's evidence once. An equal capture timestamp
  no longer resets motion state. `tests/test_motion_incremental_qualification.py`
  replays moving-object, lighting-ramp and static scenes through the old and
  new paths and requires identical decisions.
- **T2:** not started.
- **T3 (done):** each EMA evidence frame keeps a reference to the read-only
  live capture it was derived from (no copy), bounded per camera by
  `EVIDENCE_SOURCE_FRAME_BUDGET_BYTES` (64 MiB; the newest entry always keeps
  it). The EMA fast object check now runs on that full-resolution frame and
  falls back to the analysis-width frame only for entries beyond the budget.
  Detection recall on real footage has not been measured.
- **T5 (partly done):**
  - The motion ring size is computed once
    (`MotionQualificationConfig.motion_ring_size`).
  - Detection providers share one declared `(event_at, qualification)`
    contract; call-time `inspect.signature` checks are gone. They were
    covering a real case: the live fast-path provider was wired with one
    argument.
  - `MotionTriggerKind` / `motion_trigger_kind()` classify trigger topics.
    SurvNG's own topics are defined once, and manual-topic detection uses
    one rule everywhere.
  - The activity attributor is passed through constructors to the recorded
    detector, the tracking lifecycle and every session it creates. The
    write-only `scene_context_memory` attributes were removed.
  - `CameraWorker` reads detector configuration fields directly, including
    on the per-frame capture path.
  - Remaining:
    - Splitting the oversized functions.
    - Removing the pass-through detection layers.
    - Converting the roughly 37 `getattr(config, ...)` reads in
      `motion_pipeline/object_detection.py`. Several of those fallbacks
      (`event_refinement_stages`, `event_route_refinement_stages`,
      `tracking`, `depth`) differ from the `DetectorConfig` defaults, so test
      doubles that supply partial configs exercise non-production refinement
      settings. Converting them requires migrating those doubles first.
