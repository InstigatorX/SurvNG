#!/usr/bin/env python3
"""Export torchreid OSNet-x1.0 MSMT17 to OpenVINO IR for SurvNG person ReID."""
from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import urllib.request
from pathlib import Path

OSNET_SOURCE_URL = (
    "https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/"
    "master/torchreid/models/osnet.py"
)
WEIGHTS_URL = (
    "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
    "b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "SurvNG-osnet-export"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)


def _load_osnet(source_path: Path):
    spec = importlib.util.spec_from_file_location("survng_osnet_export", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the OSNet model definition")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_dict(raw: object) -> dict:
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "net"):
            nested = raw.get(key)
            if isinstance(nested, dict):
                raw = nested
                break
    if not isinstance(raw, dict):
        raise ValueError("OSNet checkpoint did not contain a state dict")
    return {
        str(key).removeprefix("module."): value
        for key, value in raw.items()
    }


def export(output_xml: Path, cache_dir: Path, ovc_bin: str) -> Path:
    import torch

    source_path = cache_dir / "osnet.py"
    weights_path = cache_dir / "osnet_x1_0_msmt17.pth"
    if not source_path.is_file():
        _download(OSNET_SOURCE_URL, source_path)
    if not weights_path.is_file():
        _download(WEIGHTS_URL, weights_path)

    module = _load_osnet(source_path)
    state = _state_dict(torch.load(weights_path, map_location="cpu", weights_only=False))
    classifier = state.get("classifier.weight")
    num_classes = int(classifier.shape[0]) if classifier is not None else 1000
    model = module.osnet_x1_0(num_classes=num_classes, pretrained=False, loss="softmax")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [name for name in missing if not str(name).startswith("classifier.")]
    if missing:
        raise RuntimeError(f"OSNet weights were incomplete: {missing[:8]}")
    if unexpected:
        raise RuntimeError(f"OSNet weights had unexpected keys: {unexpected[:8]}")
    model.eval()

    output_xml = output_xml.expanduser().resolve()
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, 256, 128)
    try:
        import openvino as ov

        converted = ov.convert_model(model, example_input=dummy)
        ov.save_model(converted, str(output_xml), compress_to_fp16=True)
    except Exception as direct_error:
        onnx_path = output_xml.with_suffix(".onnx")
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["data"],
            output_names=["reid_embedding"],
            opset_version=17,
            dynamo=False,
        )
        import subprocess

        result = subprocess.run(
            [
                ovc_bin,
                str(onnx_path),
                "--output_model",
                str(output_xml),
                "--compress_to_fp16=True",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        onnx_path.unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError(
                "OpenVINO conversion failed:\n"
                f"{direct_error}\n"
                f"{(result.stderr or result.stdout or '').strip()[-800:]}"
            ) from direct_error
    if not output_xml.is_file() or not output_xml.with_suffix(".bin").is_file():
        raise RuntimeError(f"OpenVINO IR was not written to {output_xml}")
    return output_xml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="models/person_reid_model/osnet_x1_0_msmt17.xml",
        help="OpenVINO IR xml path",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Download cache for the OSNet definition and weights",
    )
    parser.add_argument(
        "--ovc",
        default="",
        help="Path to the OpenVINO ovc converter",
    )
    args = parser.parse_args()
    ovc_bin = args.ovc.strip()
    if not ovc_bin:
        from shutil import which

        ovc_bin = which("ovc") or ""
    if not ovc_bin:
        print("OpenVINO converter (ovc) was not found.", file=sys.stderr)
        return 1
    cache = Path(args.cache_dir).expanduser() if args.cache_dir else Path(tempfile.mkdtemp(prefix="survng-osnet-"))
    try:
        path = export(Path(args.output), cache, ovc_bin)
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Installed OSNet-x1.0 MSMT17 person ReID at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
