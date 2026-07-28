# Vehicle Re-identification Model

SurvNG can optionally use Open Model Zoo's `vehicle-reid-0001` model to
reconnect vehicle tracks after geometry matching becomes ambiguous. The model
artifact is intentionally excluded from Git.

Install the official ONNX model from OpenVINO's model storage:

```bash
mkdir -p vehicle_reid_model
curl --retry 5 -L \
  https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.onnx \
  -o vehicle_reid_model/vehicle-reid-0001.onnx
```

Expected SHA-256:

```text
4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f
```

Configure `vehicle_reid_model/vehicle-reid-0001.onnx` as the vehicle ReID
model in Config → Object Detection → Advanced object tracking. The ONNX model
uses RGB input; SurvNG performs the conversion from OpenCV's BGR crops.
