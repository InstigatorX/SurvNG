"""Optional local SigLIP 2 encoder package for Smart Search.

The weights are the fp16 ONNX export of ``google/siglip2-base-patch16-224``.
SurvNG downloads them the first time this provider starts and checks every
file against a pinned size and SHA-256. Pictures are embedded on this host.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOGGER = logging.getLogger("uvicorn.error")

SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP2_REPOSITORY = "upshift/siglip2-base-patch16-224-onnx"
SIGLIP2_REVISION = "c7a76f6f6e1244641930efa9116fb0bf563c55b9"
SIGLIP2_SOURCE_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SIGLIP2_DIMENSIONS = 768
SIGLIP2_IMAGE_SIZE = 224
SIGLIP2_TEXT_LENGTH = 64
SIGLIP2_IMAGE_BATCH_SIZE = 8
SIGLIP2_PACKAGE_DIRNAME = "siglip2-base-patch16-224"

# name, sha256, size. The digest is the Hugging Face LFS oid for this revision.
SIGLIP2_FILES: tuple[tuple[str, str, int], ...] = (
    (
        "vision_model_fp16.onnx",
        "41f242e203f1b8e96fb681c56dc162dc3991c65f74b952c3431261d17a22b0ce",
        226218,
    ),
    (
        "vision_model_fp16.onnx_data",
        "6770e253d8bde8eabf1f01b680d310c74a75fc8ff4a23d3aaaf1611b51aa80fd",
        201429536,
    ),
    (
        "text_model_fp16.onnx",
        "78465bbbda3a435a48d1eb4415e1a90580cf100fbcd7616fe0d52071114adab1",
        193721,
    ),
    (
        "text_model_fp16.onnx_data",
        "a157f51f52f5de2e0cee7456175d4f113384a624df280344ac44ba066c949535",
        564607488,
    ),
    (
        "tokenizer.json",
        "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
        34363039,
    ),
)


class Siglip2DownloadInterrupted(RuntimeError):
    """Package download stopped because Smart Search was shut down."""


def siglip2_manifest() -> dict[str, object]:
    """OpenVINO manifest for the pinned SigLIP 2 image and text towers."""
    return {
        "implementation": "siglip2",
        "embedding_space_id": SIGLIP2_MODEL_ID,
        "dimensions": SIGLIP2_DIMENSIONS,
        "image_model": "vision_model_fp16.onnx",
        "text_model": "text_model_fp16.onnx",
        "image": {
            "input": "pixel_values",
            "output": "image_embeds",
            "size": SIGLIP2_IMAGE_SIZE,
            "interpolation": "linear",
            "resize_mode": "fixed",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "text": {
            "input": "input_ids",
            "output": "text_embeds",
            "tokenizer_kind": "siglip2",
            "tokenizer_path": "tokenizer.json",
            "max_length": SIGLIP2_TEXT_LENGTH,
        },
    }


def siglip2_semantic_contract() -> tuple[object, str]:
    """Generation key for the pinned SigLIP 2 checkpoint.

    The cache directory is not part of the key. Moving the download does not
    mix it with MobileCLIP or EmbeddingGemma vectors, and it does not start
    another SigLIP 2 generation.
    """
    from .semantic_search import SemanticModelIdentity

    payload = {
        "implementation": "siglip2",
        "model": SIGLIP2_MODEL_ID,
        "weights": SIGLIP2_REPOSITORY,
        "revision": SIGLIP2_REVISION,
        "source_revision": SIGLIP2_SOURCE_REVISION,
        "dimensions": SIGLIP2_DIMENSIONS,
        "image_size": SIGLIP2_IMAGE_SIZE,
        "resize": "fixed-linear",
        "normalize": [0.5, 0.5],
        "text_length": SIGLIP2_TEXT_LENGTH,
        "text_normalization": "lowercase",
        "query_prefix": "",
        "files": [
            {"name": name, "sha256": digest, "size": size}
            for name, digest, size in SIGLIP2_FILES
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity = SemanticModelIdentity(
        "siglip2",
        digest[:24],
        digest[24:48],
        SIGLIP2_DIMENSIONS,
    )
    return identity, digest


def siglip2_package_dir(model_dir: str, storage_dir: Path) -> Path:
    """Return the configured cache, or the storage-dir default."""
    configured = str(model_dir or "").strip()
    if not configured:
        return Path(storage_dir) / "models" / SIGLIP2_PACKAGE_DIRNAME
    path = Path(configured)
    if path.is_absolute():
        return path
    return Path(storage_dir) / path


def ensure_siglip2_package(
    directory: Path,
    *,
    should_stop: Callable[[], bool] | None = None,
    files: Sequence[tuple[str, str, int]] | None = None,
    base_url: str | None = None,
) -> Path:
    """Download any missing pinned files and write the local manifest."""
    selected = tuple(files) if files is not None else SIGLIP2_FILES
    package = Path(directory)
    package.mkdir(parents=True, exist_ok=True)
    for name, digest, size in selected:
        if _stop_requested(should_stop):
            raise Siglip2DownloadInterrupted("SigLIP 2 download was interrupted")
        destination = package / name
        if _file_matches(destination, digest, size, should_stop):
            continue
        LOGGER.info("Downloading SigLIP 2 file %s (%d bytes)", name, size)
        _download_file(
            _file_url(name, base_url=base_url),
            destination,
            digest,
            size,
            should_stop,
        )
    manifest_path = package / "semantic_model.json"
    manifest_path.write_text(
        json.dumps(siglip2_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return package


def _file_url(name: str, *, base_url: str | None) -> str:
    if base_url:
        return f"{base_url.rstrip('/')}/{name}"
    return (
        f"https://huggingface.co/{SIGLIP2_REPOSITORY}/resolve/"
        f"{SIGLIP2_REVISION}/{name}"
    )


def _stop_requested(should_stop: Callable[[], bool] | None) -> bool:
    return bool(should_stop and should_stop())


def _file_matches(
    path: Path,
    digest: str,
    size: int,
    should_stop: Callable[[], bool] | None,
) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            if _stop_requested(should_stop):
                raise Siglip2DownloadInterrupted("SigLIP 2 download was interrupted")
            hasher.update(chunk)
    return hasher.hexdigest() == digest


def _download_file(
    url: str,
    destination: Path,
    digest: str,
    size: int,
    should_stop: Callable[[], bool] | None,
) -> None:
    partial = destination.with_name(destination.name + ".partial")
    hasher = hashlib.sha256()
    written = 0
    try:
        request = Request(url, headers={"User-Agent": "SurvNG"})
        with urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                if _stop_requested(should_stop):
                    raise Siglip2DownloadInterrupted("SigLIP 2 download was interrupted")
                written += len(chunk)
                if written > size:
                    raise RuntimeError(f"SigLIP 2 file {destination.name} was larger than expected")
                hasher.update(chunk)
                handle.write(chunk)
    except Siglip2DownloadInterrupted:
        partial.unlink(missing_ok=True)
        raise
    except HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"SigLIP 2 download failed for {destination.name}: HTTP {exc.code}"
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"SigLIP 2 download failed for {destination.name} ({type(exc).__name__})"
        ) from exc
    actual = hasher.hexdigest()
    if written != size or actual != digest:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"SigLIP 2 file {destination.name} did not match its pinned digest")
    os.replace(partial, destination)


class Siglip2Tokenizer:
    """Lowercase Gemma tokenization padded to SigLIP 2's 64-token contract.

    The exported tokenizer already adds the end-of-sequence token and right-pads
    with pad id 0. Lowercasing is the training normalization and is applied
    here because that file does not include it.
    """

    def __init__(self, path: Path) -> None:
        from tokenizers import Tokenizer
        from tokenizers import normalizers

        tokenizer = Tokenizer.from_file(str(path))
        current = tokenizer.normalizer
        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.Lowercase()] if current is None else [normalizers.Lowercase(), current]
        )
        tokenizer.enable_truncation(max_length=SIGLIP2_TEXT_LENGTH)
        self._tokenizer = tokenizer

    def __call__(self, texts: Sequence[str]) -> dict[str, object]:
        import numpy as np

        if not texts:
            raise RuntimeError("semantic text batch cannot be empty")
        rows = []
        for text in texts:
            ids = self._tokenizer.encode(str(text)).ids
            if len(ids) != SIGLIP2_TEXT_LENGTH:
                raise RuntimeError(
                    f"SigLIP 2 tokenizer returned {len(ids)} tokens; expected {SIGLIP2_TEXT_LENGTH}"
                )
            rows.append(ids)
        return {"input_ids": np.asarray(rows, dtype=np.int64)}
