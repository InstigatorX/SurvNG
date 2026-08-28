# Third-party models installed by `survng-model-installer`

The SurvNG **runtime** GHCR image (`ghcr.io/instigatorx/survng`) ships no
detector, ReID, face, or semantic-search weights. This **installer** image
downloads them into a host bind mount at install time. None of the rows below
are SurvNG MIT code.

| Model | Artifact | License | Source / notes |
| --- | --- | --- | --- |
| YOLO26s (Ultralytics) | OpenVINO IR (`*.xml` / `*.bin`) | **AGPL-3.0** | Exported at install time via `ultralytics`. AGPL obligations apply if you run or modify this detector in production. |
| YOLO26n-depth (Ultralytics) | OpenVINO IR (`*.xml` / `*.bin`) | **AGPL-3.0** | Exported at install time via `ultralytics` (`yolo depth` task). Optional monocular depth enrichment for incident objects. |
| person-reidentification-retail-0286 | OpenVINO IR (`*.xml` / `*.bin`) | **Apache-2.0** | Intel Open Model Zoo pre-converted FP16 IR. |
| vehicle-reid-0001 (OSNet) | **ONNX** (single file) | **MIT** | Intel Open Model Zoo publishes ONNX only for this model; OpenVINO loads it directly. Not converted to IR because OMZ does not ship an IR build. |
| ArcFace embedding | OpenVINO IR after `ovc` | **OMZ / Apache-2.0** | ONNX from OMZ converted to FP16 IR at install time. |
| landmarks-regression-retail-0009 | OpenVINO IR | **Apache-2.0** | Intel OMZ pre-converted FP16 IR. |
| face-detection-retail-0004 | OpenVINO IR | **Apache-2.0** | Intel OMZ pre-converted FP16 IR. |
| MobileCLIP2-B | OpenVINO package dir | **Apple ML Research Model terms** | Research / non-commercial use; exported at install time via `open_clip_torch`. |

Pass `--skip-face` to omit the face stack. The legacy host helper
`scripts/install-face-model.sh` remains for native checkouts.

Person ReID uses OMZ **IR** because Intel publishes pre-built FP16 binaries.
Vehicle ReID stays **ONNX** because that is the only official OMZ artifact; SurvNG
and OpenVINO accept both formats via `read_model()`.

## Python packages used during install

Install-time venvs may pull PyTorch, Ultralytics, OpenCLIP, OpenVINO (`ovc`), and
transitive dependencies. See pinned ranges in `scripts/install-docker-models.sh`
and the exporter script. Their licenses (Apache-2.0, BSD, MIT, AGPL for
Ultralytics, etc.) apply to those packages inside this installer container only.
