# Native DL Streamer Deep SORT experiment

`experiment/native-first` keeps `gvatrack tracking-type=short-term-imageless`
as the default. Deep SORT is an opt-in, person-only experiment that inserts a
MARS appearance embedding stage in the existing GStreamer metadata path:

```text
gvadetect -> gvainference(MARS, person ROI) -> gvatrack(deep-sort) -> gvaanalytics
```

Configure the existing tracking fields:

```json
{
  "detector": {
    "native": {
      "inference_interval": 1,
      "tracking_classes": ["person"]
    },
    "tracking": {
      "implementation": "dlstreamer_deep_sort",
      "reid_enabled": true,
      "reid_model_path": "/root/dlstreamer_models/public/mars-small128/mars_small128_fp32.xml",
      "reid_device": "GPU"
    }
  }
}
```

`tracking_classes` may be omitted in Deep SORT mode; SurvNG then constrains
the native tracker to `person`. Other class combinations are rejected so a
person-trained MARS model is never silently used for vehicles or animals.

Initial Deep SORT parameters deliberately match the DL Streamer 2026.2
baseline: `max_iou_distance=0.7,max_age=30,n_init=3,`
`max_cosine_distance=0.2,nn_budget=100`.

The adaptive inference budget remains ahead of `gvadetect`; Deep SORT still
requires `detector.native.inference_interval=1` for every frame admitted to
the detector. The ReID model is shared across supervisor streams through a
stable `model-instance-id`.

The existing authoritative-fresh-detection provenance check remains strict.
If Deep SORT changes detector ROI geometry, `native_evidence_invalid` will
expose it rather than silently promoting tracker predictions as detections.


## Live ReID metadata correction

Live diagnostics on 2026-09-17 showed every Deep SORT person detection falling back to a 128-dimensional zero vector: `No feature tensor found ... using zero feature (motion-only tracking)`. The native-first detector probe had been rebuilding ROI metadata before `gvainference`, which breaks the metadata attachment identity needed by the per-ROI raw ReID tensor. Deep SORT now preserves the original `gvadetect` ROIs, relies on `object-class=person` / `object_class=person` for person filtering, adds the reference queue after `gvainference`, and reports valid/missing tracker-compatible 128D MARS tensors in live pipeline status.
