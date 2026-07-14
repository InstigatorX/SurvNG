#!/usr/bin/env bash
set -euo pipefail

model_dir="${1:-face_model}"
arcface_name="face-recognition-resnet100-arcface-onnx"
arcface_source="$model_dir/arcfaceresnet100-8.onnx"
arcface_url="https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/face-recognition-resnet100-arcface-onnx/arcfaceresnet100-8.onnx"
landmark_name="landmarks-regression-retail-0009"
landmark_url="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/${landmark_name}/FP16"

mkdir -p "$model_dir"

if [[ -x ".venv/bin/ovc" ]]; then
  ovc_bin=".venv/bin/ovc"
elif command -v ovc >/dev/null 2>&1; then
  ovc_bin="$(command -v ovc)"
else
  printf 'OpenVINO converter (ovc) was not found. Install OpenVINO first.\n' >&2
  exit 1
fi

curl -fL --retry 3 -o "$arcface_source" "$arcface_url"
"$ovc_bin" "$arcface_source" \
  --output_model "$model_dir/${arcface_name}.xml" \
  --output fc1 \
  --compress_to_fp16=True
rm -f "$arcface_source"

curl -fL --retry 3 -o "$model_dir/${landmark_name}.xml" "$landmark_url/${landmark_name}.xml"
curl -fL --retry 3 -o "$model_dir/${landmark_name}.bin" "$landmark_url/${landmark_name}.bin"

for model_xml in "$model_dir/${arcface_name}.xml" "$model_dir/${landmark_name}.xml"; do
  if ! grep -q '<net ' "$model_xml"; then
    printf 'Downloaded model is not a valid OpenVINO IR: %s\n' "$model_xml" >&2
    exit 1
  fi
done

printf 'Installed ArcFace embedding and landmark alignment models in %s\n' "$model_dir"
