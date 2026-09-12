# Automatic detector model settings

OpenVINO model loading reads the input size, FP16/FP32 input type, and NCHW/NHWC
layout. Floating-point constant types are reported separately: compressed FP16
weights do not imply FP16 inputs or FP16 execution on the selected device.
Execution precision remains under OpenVINO's device policy.

The output parser uses `metadata.yaml` beside the model, OpenVINO `model_info`
runtime metadata, and NMS operations in the graph. Shape inference is the
fallback. Explicit labels and label files retain priority over embedded labels.
ONNX files loaded through OpenVINO receive the same graph inspection; export
metadata that OpenVINO does not retain needs the adjacent `metadata.yaml`.

Admin → object detection → **Model paths and startup options** shows the loaded
properties, errors, and warnings. `detector.model_output_format` defaults to
`auto`; overrides are `yolo`, `yolo-e2e`, `yolo-seg`, `yolo-seg-e2e`, and `ssd`.
The `e2e` choices mean final detections, including embedded-NMS exports. SurvNG
does not apply NMS again to those outputs. Its NMS threshold affects raw outputs
only; filtering already performed inside an export cannot be undone at runtime.

`detector.model_input_layout` defaults to `auto`, with `NCHW` and `NHWC` overrides
for OpenVINO. Saving either override rebuilds the object detector workers and
refreshes tracking through the existing configuration application boundary.

Six-column outputs can also describe raw two-class models. When only shapes
support the final-detection interpretation, Admin displays an inference warning
and provides the override. Unsupported output contracts produce a model-settings
error instead of silently being decoded as SSD.

The supported preprocessing contract remains RGB, division by 255, and
letterboxing with padding 114. Arbitrary normalization cannot be inferred from
tensor shapes. Static three-channel single-image FP16/FP32 inputs are supported;
dynamic image inputs and integer image inputs require a compatible re-export.
Segmentation requires 32-channel mask prototypes. Core ML and OpenCV use the
shared output resolver; input precision/layout inspection is OpenVINO-specific.
OpenCV fallback requires FP32 NCHW input.
