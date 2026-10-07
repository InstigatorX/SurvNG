# Camera-specific face detector

This loop labels the upper-body crop SurvNG already sends to the face detector.
The cloud job is outside the API. Live frames stay on the NVR. Labels are not
names, and they never enter the face gallery or automatic identification.

People review cannot supply the set. Those rows are padded crops of faces the
current detector already found. Misses are person observations with no face box.

## What the worker accepts

`OpenVinoFaceDetector` stretch-resizes with `cv2.INTER_AREA` to the IR's static
`H×W`, then sends BGR NCHW float32 in the range 0–255. It does not letterbox
and it does not divide by 255. The output must be SSD rows of seven floats,
preferably shaped `[1, 1, N, 7]`:

```text
image_id, class_id, confidence, x1, y1, x2, y2
```

Confidence is index 2. The box is indices 3–6, normalized to the stretched
crop. RGB conversion, normalization, box decode, and NMS have to be baked into
the graph. A raw YOLO or SCRFD head will not parse. A static 320 or 384 input
loads without a worker change, because the loader reads the shape from the IR.

The crop is the person box plus 8% of its width on each side, 5% of its height
above the top, and down to 68% of its height. Crops under 24 pixels on either
side are skipped. Only the `person` label is used.

## Label, export, score

Open **People → Face finding**.

1. **Materialize** reads recorded `person` observations with an exact frame
   time, cuts the upper-body window, and runs the current face detector at 0.60
   when a model path is configured. Crops with no baseline face are queued
   first. A portion of crops that already have a baseline face stay in the
   queue so false boxes can be marked.
2. Draw a face, click an orange baseline box to mark it as not a face, choose
   **No face**, or **Drop** with a reason. `no_face` cannot sit on a crop that
   has a face box.
3. **Freeze splits** assigns whole calendar days, about 70% train, 15%
   calibration, and 15% held out. At least three labeled days are required.
   The same decoded pixels cannot land in two splits. Frozen days stay put.
4. **Export** writes `manifest.json` and `crops/*.png` under the database
   directory. The manifest stores crop hashes and boxes. It does not store
   person names.
5. Train outside SurvNG. **Score IR** loads both OpenVINO XML files through
   the same detector class and applies the promotion gate.

Set **Admin → Detection → Face Detector Model** to a passing XML. Compare the
embedding fingerprint shown by the scorer before and after the face worker
reloads. It must be unchanged. Keep the previous detector on disk so the path
can be switched back.

A better box still has to pass `face_min_size` and the landmark check before it
can be named. A person the object detector never boxed never becomes a crop.

## External training job

The job is a batch script, not a SurvNG service.

1. Verify every crop sha256 in the manifest.
2. Fine-tune a one-class pretrained face detector whose license allows this
   deployment. Stretch-resize to the training square the same way the worker
   does. Do not letterbox.
3. Treat `no_face` crops and `not_a_face` regions as background. `face` boxes
   are the only positive class. Skip `drop` samples; they are not in the export.
4. Export an ONNX wrapper that accepts BGR float32 0–255 and emits
   `[1, 1, N, 7]` SSD rows with normalized coordinates. Bake channel order,
   normalization, decode, and NMS inside that wrapper.
5. Convert with OpenVINO `compress_to_fp16=True` and a static input
   `[1, 3, H, W]`.
6. Compare ONNX and IR outputs on a few manifest crops.
7. Return the XML, BIN, file hashes, and the manifest hash. Delete the
   uploaded crops after the IR is downloaded.

Score the IR on the NVR before pointing `face_detection_model_path` at it.
The gate, at IoU 0.5, requires:

- held-out miss-set recall at least 0.80, on faces retail-style baseline
  detections at 0.60 did not match, and at least 0.10 above that baseline
- overall held-out face recall not below the baseline
- false-positive rate on `no_face` crops not above the baseline
- a threshold other than 0.60 chosen only from the calibration day, then frozen
- estimated p95 of a 12-call face pass at or under 2 seconds, and of a
  44-call evidence budget at or under 4 seconds

`survng.app.face_evaluation` stays the identity tool. It does not measure
whether a face box was found.
