#!/usr/bin/env python3
"""Build a validated SurvNG SigLIP2 Base OpenVINO FP16 package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

MODEL_ID = "google/siglip2-base-patch16-224"
DEFAULT_OUTPUT = Path("models/siglip2-base-patch16-224-openvino-fp16")
VALIDATION_PROMPTS = [
    "a white delivery truck",
    "a black pickup truck",
    "a person wearing red clothing",
    "a package on a porch",
    "a dog outside",
]


def _dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import openvino as ov
        import torch
        from huggingface_hub import hf_hub_download
        from PIL import Image
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise SystemExit(
            "Missing export dependencies. Run: "
            ".venv/bin/pip install -r requirements-semantic-export.txt"
        ) from exc
    return ov, torch, hf_hub_download, Image, (AutoModel, AutoProcessor)


def _normalized(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-9)


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


def _validate_cross_modal(
    reference_images: np.ndarray,
    reference_text: np.ndarray,
    candidate_images: np.ndarray,
    candidate_text: np.ndarray,
    maximum_error: float = 5e-4,
) -> dict[str, float]:
    source_scores = _normalized(reference_text) @ _normalized(reference_images).T
    candidate_scores = _normalized(candidate_text) @ _normalized(candidate_images).T
    difference = float(np.max(np.abs(source_scores - candidate_scores)))
    if not np.isfinite(difference) or difference > maximum_error:
        raise RuntimeError(
            "cross-modal OpenVINO parity failed: maximum cosine error "
            f"{difference:.6f} exceeds {maximum_error:.6f}"
        )
    return {"maximum_cosine_error": round(difference, 8)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _set_names(model: Any, inputs: list[str], output: str) -> None:
    if len(model.inputs) != len(inputs):
        raise RuntimeError(
            f"exported model has {len(model.inputs)} inputs; expected {len(inputs)}"
        )
    for port, name in zip(model.inputs, inputs, strict=True):
        port.tensor.set_names({name})
    model.outputs[0].tensor.set_names({output})


def _validation_images(image_type: Any) -> list[Any]:
    rng = np.random.default_rng(20260811)
    return [
        image_type.fromarray(rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)),
        image_type.fromarray(rng.integers(0, 256, (320, 180, 3), dtype=np.uint8)),
    ]


def build_package(
    output_dir: Path,
    *,
    cache_dir: str = "",
    force: bool = False,
    minimum_cosine: float = 0.995,
) -> Path:
    ov, torch, hf_hub_download, image_type, auto = _dependencies()
    auto_model, auto_processor = auto
    output_dir = output_dir.resolve()
    if output_dir.exists() and not force:
        raise RuntimeError(
            f"output directory already exists: {output_dir}; use --force to replace it"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    load_options: dict[str, Any] = {"local_files_only": False}
    if cache_dir:
        load_options["cache_dir"] = cache_dir

    print(f"Loading {MODEL_ID}", flush=True)
    processor = auto_processor.from_pretrained(MODEL_ID, use_fast=False, **load_options)
    model = auto_model.from_pretrained(MODEL_ID, **load_options).eval()
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    max_length = int(model.config.text_config.max_position_embeddings)
    images = _validation_images(image_type)
    image_inputs = image_processor(images=images, return_tensors="pt")
    # Use the model's official combined processor contract. SigLIP2's fixed
    # padded text tower intentionally omits attention_mask; supplying one
    # changes the embedding space even though the token IDs are identical.
    text_inputs = processor(
        text=VALIDATION_PROMPTS,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    text_input_names = [
        name for name in ("input_ids", "attention_mask") if name in text_inputs
    ]
    if "input_ids" not in text_input_names:
        raise RuntimeError("source processor did not produce input_ids")

    packed_images = all(
        name in image_inputs
        for name in ("pixel_values", "pixel_attention_mask", "spatial_shapes")
    )
    image_input_names = (
        ["pixel_values", "pixel_attention_mask", "spatial_shapes"]
        if packed_images
        else ["pixel_values"]
    )

    class PackedImageEncoder(torch.nn.Module):
        def __init__(self, source: Any) -> None:
            super().__init__()
            self.source = source

        def forward(
            self,
            pixel_values: Any,
            pixel_attention_mask: Any,
            spatial_shapes: Any,
        ) -> Any:
            return self.source.get_image_features(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            ).pooler_output

    class FixedImageEncoder(torch.nn.Module):
        def __init__(self, source: Any) -> None:
            super().__init__()
            self.source = source

        def forward(self, pixel_values: Any) -> Any:
            return self.source.get_image_features(
                pixel_values=pixel_values
            ).pooler_output

    class TextEncoder(torch.nn.Module):
        def __init__(self, source: Any) -> None:
            super().__init__()
            self.source = source

        def forward(self, input_ids: Any) -> Any:
            return self.source.get_text_features(input_ids=input_ids).pooler_output

    image_encoder = (
        PackedImageEncoder(model).eval()
        if packed_images
        else FixedImageEncoder(model).eval()
    )
    text_encoder = TextEncoder(model).eval()
    one_image = tuple(image_inputs[name][:1] for name in image_input_names)
    if text_input_names != ["input_ids"]:
        raise RuntimeError(
            "this SigLIP2 exporter expects the official input_ids-only text contract"
        )
    text_example = (text_inputs["input_ids"],)
    with torch.inference_mode():
        reference_images = np.concatenate([
            image_encoder(*(image_inputs[name][index:index + 1] for name in image_input_names))
            .detach().cpu().numpy()
            for index in range(len(images))
        ])
        reference_text = text_encoder(*text_example).detach().cpu().numpy()
    dimensions = int(reference_images.shape[-1])
    if reference_text.shape[-1] != dimensions:
        raise RuntimeError("image and text encoders produced different embedding dimensions")

    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        print("Converting fixed-batch image encoder to OpenVINO IR", flush=True)
        ov_image = ov.convert_model(image_encoder, example_input=one_image)
        _set_names(
            ov_image,
            image_input_names,
            "image_features",
        )
        ov.save_model(ov_image, temporary / "image_encoder.xml", compress_to_fp16=True)

        print("Converting dynamic-batch text encoder to OpenVINO IR", flush=True)
        ov_text = ov.convert_model(text_encoder, example_input=(text_example[0][:1],))
        if int(text_example[0].shape[1]) != max_length:
            raise RuntimeError("source tokenizer did not honor the model text context length")
        ov_text.reshape({
            "input_ids": [-1, max_length],
        })
        _set_names(ov_text, ["input_ids"], "text_features")
        ov.save_model(ov_text, temporary / "text_encoder.xml", compress_to_fp16=True)

        tokenizer_dir = temporary / "tokenizer"
        tokenizer.save_pretrained(tokenizer_dir)
        tokenizer_json = tokenizer_dir / "tokenizer.json"
        if not tokenizer_json.is_file():
            raise RuntimeError("exported tokenizer did not include tokenizer.json")

        from survng.app.semantic_search import (
            HuggingFaceJsonTokenizer,
            _prepare_fixed_pil_images,
            _prepare_siglip2_images,
        )

        if packed_images:
            image_spec = {
                "processor_kind": "siglip2_patches",
                "inputs": {
                    "pixel_values": "pixel_values",
                    "pixel_attention_mask": "pixel_attention_mask",
                    "spatial_shapes": "spatial_shapes",
                },
                "output": "image_features",
                "batch_size": 1,
                "patch_size": int(image_processor.patch_size),
                "max_num_patches": int(image_processor.max_num_patches),
                "interpolation": "bilinear",
                "mean": list(image_processor.image_mean),
                "std": list(image_processor.image_std),
            }
        else:
            image_size = int(
                image_processor.size.get("height")
                if isinstance(image_processor.size, dict)
                else image_processor.size.height
            )
            image_spec = {
                "processor_kind": "fixed_pil",
                "input": "pixel_values",
                "output": "image_features",
                "batch_size": 1,
                "size": image_size,
                "interpolation": "bilinear",
                "resize_mode": "fixed",
                "mean": list(image_processor.image_mean),
                "std": list(image_processor.image_std),
            }
        text_spec = {
            "inputs": {
                "input_ids": "input_ids",
            },
            "output": "text_features",
            "tokenizer_kind": "huggingface_tokenizer_json",
            "tokenizer_path": "tokenizer/tokenizer.json",
            "max_length": max_length,
            "padding_side": str(tokenizer.padding_side),
            "truncation_side": str(tokenizer.truncation_side),
            "pad_token": str(tokenizer.pad_token),
            "pad_token_id": int(tokenizer.pad_token_id),
        }
        bgr_images = [np.asarray(image)[:, :, ::-1].copy() for image in images]
        if packed_images:
            runtime_images: Any = _prepare_siglip2_images(bgr_images, image_spec)
            runtime_image_inputs = runtime_images
        else:
            runtime_images = _prepare_fixed_pil_images(bgr_images, image_spec)
            runtime_image_inputs = {"pixel_values": runtime_images}
        for name in image_input_names:
            np.testing.assert_allclose(
                runtime_image_inputs[name],
                image_inputs[name].detach().cpu().numpy(),
                rtol=1e-5,
                atol=2e-5,
                err_msg=f"runtime image preprocessing mismatch for {name}",
            )
        runtime_tokens = HuggingFaceJsonTokenizer(tokenizer_json, text_spec)(
            VALIDATION_PROMPTS
        )
        np.testing.assert_array_equal(
            runtime_tokens["input_ids"], text_inputs["input_ids"].numpy()
        )

        core = ov.Core()
        compiled_image = core.compile_model(temporary / "image_encoder.xml", "CPU")
        compiled_text = core.compile_model(temporary / "text_encoder.xml", "CPU")
        candidate_images = np.concatenate([
            np.asarray(compiled_image({
                name: runtime_image_inputs[name][index:index + 1]
                for name in image_input_names
            })["image_features"])
            for index in range(len(images))
        ])
        candidate_text = compiled_text({
            name: runtime_tokens[name] for name in text_input_names
        })["text_features"]
        cross_modal_validation = _validate_cross_modal(
            reference_images, reference_text, candidate_images, candidate_text
        )
        validation = {
            "image": _validate_pair(
                reference_images, candidate_images, "image encoder", minimum_cosine
            ),
            "text": _validate_pair(
                reference_text, candidate_text, "text encoder", minimum_cosine
            ),
            "runtime_preprocessing": "exact",
            "runtime_tokenizer": "exact",
            "cross_modal": cross_modal_validation,
        }
        manifest = {
            "schema_version": 2,
            "implementation": "siglip2_openvino",
            "model_name": MODEL_ID,
            "source": {
                "repository": MODEL_ID,
                "revision": str(getattr(model.config, "_commit_hash", "") or "main"),
                "license": "Apache-2.0",
            },
            "precision": "FP16",
            "dimensions": dimensions,
            "image_model": "image_encoder.xml",
            "text_model": "text_encoder.xml",
            "image": image_spec,
            "text": text_spec,
            "validation": validation,
        }
        (temporary / "semantic_model.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        try:
            options: dict[str, Any] = {"repo_id": MODEL_ID, "filename": "LICENSE"}
            if cache_dir:
                options["cache_dir"] = cache_dir
            shutil.copy2(hf_hub_download(**options), temporary / "LICENSE")
        except Exception:
            system_license = Path("/usr/share/common-licenses/Apache-2.0")
            if not system_license.is_file():
                raise RuntimeError(
                    "could not bundle the Apache-2.0 license for the model package"
                )
            shutil.copy2(system_license, temporary / "LICENSE")
        package_files = [
            path for path in temporary.rglob("*")
            if path.is_file() and path.name != "VALIDATION.json"
        ]
        validation_report = {
            "model": MODEL_ID,
            "prompts": VALIDATION_PROMPTS,
            "validation": validation,
            "files": {
                str(path.relative_to(temporary)): {
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(package_files)
            },
        }
        (temporary / "VALIDATION.json").write_text(
            json.dumps(validation_report, indent=2) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"Validated SurvNG package created at {output_dir}", flush=True)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", default="", help="Optional Hugging Face cache")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--minimum-cosine", type=float, default=0.995)
    args = parser.parse_args()
    if not 0.9 <= args.minimum_cosine <= 1.0:
        parser.error("--minimum-cosine must be between 0.9 and 1.0")
    build_package(
        args.output,
        cache_dir=args.cache_dir,
        force=args.force,
        minimum_cosine=args.minimum_cosine,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
