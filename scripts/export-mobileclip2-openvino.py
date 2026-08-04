#!/usr/bin/env python3
"""Build a validated SurvNG MobileCLIP2-B OpenVINO FP16 package."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

MODEL_NAME = "MobileCLIP2-B"
OFFICIAL_REPOSITORY = "apple/MobileCLIP2-B"
CHECKPOINT_NAME = "mobileclip2_b.pt"
DEFAULT_OUTPUT = Path("models/mobileclip2-b-openvino-fp16")
VALIDATION_PROMPTS = [
    "a person in a red jacket",
    "a white delivery truck",
    "a dog walking outside",
]


def _dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import open_clip
        import openvino as ov
        import torch
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing export dependencies. Run: "
            ".venv/bin/pip install -r requirements-semantic-export.txt"
        ) from exc
    return open_clip, ov, torch, hf_hub_download


def _official_checkpoint(hf_hub_download: Any, cache_dir: str) -> Path:
    options: dict[str, Any] = {
        "repo_id": OFFICIAL_REPOSITORY,
        "filename": CHECKPOINT_NAME,
    }
    if cache_dir:
        options["cache_dir"] = cache_dir
    return Path(hf_hub_download(**options))


def _reparameterize(model: Any) -> Any:
    """Mirror Apple's official reparameterize_model implementation."""
    converted = copy.deepcopy(model)
    for module in converted.modules():
        callback = getattr(module, "reparameterize", None)
        if callable(callback):
            callback()
    return converted


def _normalized(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-9)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _friendly_names(model: Any, input_name: str, output_name: str) -> None:
    model.inputs[0].tensor.set_names({input_name})
    model.outputs[0].tensor.set_names({output_name})


def _validate_pair(
    reference: np.ndarray,
    candidate: np.ndarray,
    label: str,
    minimum_cosine: float,
) -> dict[str, float]:
    reference = _normalized(reference)
    candidate = _normalized(candidate)
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"{label} validation shape mismatch: {reference.shape} != {candidate.shape}"
        )
    cosine = np.sum(reference * candidate, axis=-1)
    minimum = float(np.min(cosine))
    maximum_error = float(np.max(np.abs(reference - candidate)))
    if not np.all(np.isfinite(cosine)) or minimum < minimum_cosine:
        raise RuntimeError(
            f"{label} OpenVINO parity failed: minimum cosine {minimum:.6f} "
            f"is below {minimum_cosine:.6f}"
        )
    return {
        "minimum_cosine": round(minimum, 8),
        "maximum_absolute_error": round(maximum_error, 8),
    }


def build_package(
    output_dir: Path,
    *,
    checkpoint: Path | None = None,
    cache_dir: str = "",
    force: bool = False,
    minimum_cosine: float = 0.995,
) -> Path:
    open_clip, ov, torch, hf_hub_download = _dependencies()
    output_dir = output_dir.resolve()
    if output_dir.exists() and not force:
        raise RuntimeError(f"output directory already exists: {output_dir}; use --force to replace it")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    supplied_checkpoint = checkpoint is not None
    checkpoint = checkpoint.resolve() if checkpoint else _official_checkpoint(hf_hub_download, cache_dir)
    if not checkpoint.is_file():
        raise RuntimeError(f"checkpoint not found: {checkpoint}")

    print(f"Loading {MODEL_NAME} from {checkpoint}", flush=True)
    model, _, _preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(checkpoint),
        device="cpu",
        image_mean=(0.0, 0.0, 0.0),
        image_std=(1.0, 1.0, 1.0),
    )
    model.eval()
    model = _reparameterize(model).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    image_size_value = getattr(model.visual, "image_size", 224)
    image_size = int(image_size_value[0] if isinstance(image_size_value, (tuple, list)) else image_size_value)

    class ImageEncoder(torch.nn.Module):
        def __init__(self, source: Any) -> None:
            super().__init__()
            self.source = source

        def forward(self, pixel_values: Any) -> Any:
            return self.source.encode_image(pixel_values)

    class TextEncoder(torch.nn.Module):
        def __init__(self, source: Any) -> None:
            super().__init__()
            self.source = source

        def forward(self, input_ids: Any) -> Any:
            return self.source.encode_text(input_ids)

    image_encoder = ImageEncoder(model).eval()
    text_encoder = TextEncoder(model).eval()
    generator = torch.Generator(device="cpu").manual_seed(20260803)
    image_example = torch.rand((2, 3, image_size, image_size), generator=generator)
    text_example = tokenizer(VALIDATION_PROMPTS)
    with torch.inference_mode():
        reference_image = image_encoder(image_example).detach().cpu().numpy()
        reference_text = text_encoder(text_example).detach().cpu().numpy()
    dimensions = int(reference_image.shape[-1])
    if reference_text.shape[-1] != dimensions:
        raise RuntimeError("image and text encoders produced different embedding dimensions")

    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        print("Converting image encoder to OpenVINO IR", flush=True)
        ov_image = ov.convert_model(image_encoder, example_input=image_example[:1])
        ov_image.reshape([-1, 3, image_size, image_size])
        _friendly_names(ov_image, "pixel_values", "image_features")
        ov.save_model(ov_image, temporary / "image_encoder.xml", compress_to_fp16=True)

        print("Converting text encoder to OpenVINO IR", flush=True)
        ov_text = ov.convert_model(text_encoder, example_input=text_example[:1])
        ov_text.reshape([-1, int(text_example.shape[1])])
        _friendly_names(ov_text, "input_ids", "text_features")
        ov.save_model(ov_text, temporary / "text_encoder.xml", compress_to_fp16=True)

        tokenizer_dir = temporary / "tokenizer"
        tokenizer_dir.mkdir()
        bpe_path = Path(open_clip.tokenizer.default_bpe())
        shutil.copy2(bpe_path, tokenizer_dir / bpe_path.name)

        core = ov.Core()
        compiled_image = core.compile_model(temporary / "image_encoder.xml", "CPU")
        compiled_text = core.compile_model(temporary / "text_encoder.xml", "CPU")
        candidate_image = compiled_image({"pixel_values": image_example.numpy()})["image_features"]
        candidate_text = compiled_text({"input_ids": text_example.numpy()})["text_features"]
        validation = {
            "image": _validate_pair(reference_image, candidate_image, "image encoder", minimum_cosine),
            "text": _validate_pair(reference_text, candidate_text, "text encoder", minimum_cosine),
        }

        manifest = {
            "schema_version": 1,
            "implementation": "mobileclip2_openvino",
            "model_name": MODEL_NAME,
            "source": {
                "repository": "local" if supplied_checkpoint else OFFICIAL_REPOSITORY,
                "checkpoint": checkpoint.name,
                "sha256": _sha256(checkpoint),
                "license": "Apple ML Research Model TOU",
            },
            "precision": "FP16",
            "dimensions": dimensions,
            "image_model": "image_encoder.xml",
            "text_model": "text_encoder.xml",
            "image": {
                "input": "pixel_values",
                "output": "image_features",
                "size": image_size,
                "interpolation": "bicubic",
                "resize_mode": "shortest_center_crop",
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
            "text": {
                "input": "input_ids",
                "output": "text_features",
                "tokenizer_kind": "openclip_bpe",
                "tokenizer_path": f"tokenizer/{bpe_path.name}",
                "max_length": int(text_example.shape[1]),
            },
            "validation": validation,
        }
        (temporary / "semantic_model.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "VALIDATION.json").write_text(
            json.dumps({"prompts": VALIDATION_PROMPTS, **validation}, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            license_path = Path(hf_hub_download(repo_id=OFFICIAL_REPOSITORY, filename="LICENSE"))
            shutil.copy2(license_path, temporary / "LICENSE")
        except Exception as exc:
            print(f"Warning: could not bundle upstream license file: {exc}", flush=True)

        del compiled_image, compiled_text, ov_image, ov_text
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        del model, image_encoder, text_encoder

    print(f"Validated SurvNG package created at {output_dir}", flush=True)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, help="Use an already downloaded official checkpoint")
    parser.add_argument("--cache-dir", default="", help="Optional Hugging Face download cache")
    parser.add_argument("--force", action="store_true", help="Replace an existing output package")
    parser.add_argument("--minimum-cosine", type=float, default=0.995)
    args = parser.parse_args()
    if not 0.9 <= args.minimum_cosine <= 1.0:
        parser.error("--minimum-cosine must be between 0.9 and 1.0")
    build_package(
        args.output,
        checkpoint=args.checkpoint,
        cache_dir=args.cache_dir,
        force=args.force,
        minimum_cosine=args.minimum_cosine,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
