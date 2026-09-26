from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import threading
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..config import DetectorConfig, detector_routing_payload
from .protocol import WorkerRole


MODEL_SYNC_CHUNK_BYTES = 1024 * 1024
MAX_MODEL_BUNDLE_FILES = 2048
MAX_MODEL_BUNDLE_BYTES = 16 * 1024 * 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024
_ALL_WORKER_ROLES: tuple[WorkerRole, ...] = ("object", "face", "reid", "depth")


class ModelSyncError(RuntimeError):
    pass


class ModelFileManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=4096)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=1, le=MAX_MODEL_BUNDLE_BYTES)

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts or any(
            part in {"", ".", ".."} for part in path.parts
        ):
            raise ValueError("model manifest paths must be safe relative paths")
        return path.as_posix()


class ModelBundleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: list[ModelFileManifest] = Field(
        default_factory=list,
        max_length=MAX_MODEL_BUNDLE_FILES,
    )
    bindings: dict[str, str] = Field(default_factory=dict, max_length=16)
    total_bytes: int = Field(ge=0, le=MAX_MODEL_BUNDLE_BYTES)

    @field_validator("bindings")
    @classmethod
    def validate_bindings(cls, value: dict[str, str]) -> dict[str, str]:
        allowed = {spec.config_key for spec in _MODEL_SPECS}
        normalized: dict[str, str] = {}
        for key, raw_path in value.items():
            if key not in allowed:
                raise ValueError(f"unsupported model binding: {key}")
            normalized[key] = ModelFileManifest.validate_relative_path(raw_path)
        return normalized


@dataclass(frozen=True, slots=True)
class _ModelSpec:
    role: WorkerRole
    config_key: str
    destination: str
    enabled_key: str


_MODEL_SPECS: tuple[_ModelSpec, ...] = (
    _ModelSpec("object", "model_path", "object/model", "enabled"),
    _ModelSpec("object", "model_xml", "object/model-legacy", "enabled"),
    _ModelSpec("object", "coreml_model_path", "object/coreml", "enabled"),
    _ModelSpec("object", "labels_path", "object/labels", "enabled"),
    _ModelSpec(
        "face",
        "face_embedding_model_path",
        "face/embedding",
        "face_recognition_enabled",
    ),
    _ModelSpec(
        "face",
        "face_landmark_model_path",
        "face/landmark",
        "face_recognition_enabled",
    ),
    _ModelSpec(
        "face",
        "face_detection_model_path",
        "face/detection",
        "face_recognition_enabled",
    ),
    _ModelSpec(
        "reid",
        "tracking.reid_model_path",
        "reid/person",
        "tracking.reid_enabled",
    ),
    _ModelSpec(
        "reid",
        "tracking.vehicle_reid_model_path",
        "reid/vehicle",
        "tracking.vehicle_reid_enabled",
    ),
    _ModelSpec("depth", "depth.model_path", "depth/model", "depth.enabled"),
)


@dataclass(frozen=True, slots=True)
class PreparedModelBundle:
    config: DetectorConfig
    config_generation: str
    manifest: ModelBundleManifest
    sources: dict[str, Path]


def _config_value(config: DetectorConfig, key: str) -> Any:
    value: Any = config
    for part in key.split("."):
        value = getattr(value, part)
    return value


def _set_config_value(config: DetectorConfig, key: str, value: Any) -> None:
    target: Any = config
    parts = key.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _base_config_generation(config: DetectorConfig) -> str:
    encoded = json.dumps(
        detector_routing_payload(config),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def worker_config_for_roles(
    config: DetectorConfig,
    roles: Iterable[WorkerRole],
) -> DetectorConfig:
    selected = frozenset(roles)
    worker_config = config.model_copy(deep=True)
    worker_config.inference_mode = "local"
    worker_config.inference_balance = "remote_first"
    worker_config.object_worker_count = 1
    if worker_config.backend != "coreml":
        worker_config.coreml_model_path = ""
    if "object" not in selected:
        worker_config.enabled = False
    if "face" not in selected:
        worker_config.face_recognition_enabled = False
    if "reid" not in selected:
        worker_config.tracking.reid_enabled = False
        worker_config.tracking.vehicle_reid_enabled = False
    if "depth" not in selected:
        worker_config.depth.enabled = False
    return worker_config


class ModelBundleCatalog:
    """Builds bounded content-addressed model bundles for remote workers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._hash_cache: dict[
            tuple[str, int, int, int, int], str
        ] = {}

    def config_generation(self, config: DetectorConfig) -> str:
        with self._lock:
            try:
                files, _bindings = self._collect(config, _ALL_WORKER_ROLES)
            except ModelSyncError as error:
                invalid_payload = (
                    f"{_base_config_generation(config)}\0invalid\0{error}"
                )
                return hashlib.sha256(
                    invalid_payload.encode("utf-8")
                ).hexdigest()
            if not files:
                return _base_config_generation(config)
            payload = {
                "config": _base_config_generation(config),
                "models": [
                    (relative, digest, size)
                    for relative, _source, digest, size in files
                ],
            }
            return hashlib.sha256(
                json.dumps(
                    payload,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()

    def prepare(
        self,
        config: DetectorConfig,
        roles: Iterable[WorkerRole],
    ) -> PreparedModelBundle:
        selected = tuple(dict.fromkeys(roles))
        with self._lock:
            files, bindings = self._collect(config, selected)
            total_bytes = sum(item[3] for item in files)
            manifest_payload = {
                "files": [
                    {
                        "path": relative,
                        "digest": digest,
                        "size": size,
                    }
                    for relative, _source, digest, size in files
                ],
                "bindings": bindings,
                "total_bytes": total_bytes,
            }
            generation = hashlib.sha256(
                json.dumps(
                    manifest_payload,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            manifest = ModelBundleManifest(
                generation=generation,
                **manifest_payload,
            )
            sources: dict[str, Path] = {}
            for _relative, source, digest, _size in files:
                sources.setdefault(digest, source)
            return PreparedModelBundle(
                config=worker_config_for_roles(config, selected),
                config_generation=self.config_generation(config),
                manifest=manifest,
                sources=sources,
            )

    def _collect(
        self,
        config: DetectorConfig,
        roles: Iterable[WorkerRole],
    ) -> tuple[list[tuple[str, Path, str, int]], dict[str, str]]:
        selected = frozenset(roles)
        files: list[tuple[str, Path, str, int]] = []
        bindings: dict[str, str] = {}
        for spec in _MODEL_SPECS:
            if spec.role not in selected or not bool(
                _config_value(config, spec.enabled_key)
            ):
                continue
            raw_path = str(_config_value(config, spec.config_key) or "").strip()
            if (
                spec.config_key == "model_xml"
                and str(config.model_path or "").strip()
            ):
                continue
            if not raw_path:
                continue
            if spec.config_key == "coreml_model_path" and config.backend != "coreml":
                continue
            source = Path(raw_path).expanduser()
            try:
                resolved = source.resolve(strict=True)
            except OSError as error:
                raise ModelSyncError(
                    f"configured {spec.config_key} does not exist"
                ) from error
            destination_root = PurePosixPath(spec.destination)
            if resolved.is_dir():
                directory_root = destination_root / resolved.name
                bindings[spec.config_key] = directory_root.as_posix()
                candidates = sorted(
                    path
                    for path in resolved.rglob("*")
                    if path.is_file()
                )
                if not candidates:
                    raise ModelSyncError(
                        f"configured {spec.config_key} directory is empty"
                    )
                for candidate in candidates:
                    if candidate.is_symlink() and not candidate.is_file():
                        raise ModelSyncError(
                            f"configured {spec.config_key} contains a directory symlink"
                        )
                    relative = (
                        directory_root
                        / PurePosixPath(
                            candidate.relative_to(resolved).as_posix()
                        )
                    )
                    files.append(self._file_entry(relative, candidate))
            elif resolved.is_file():
                relative = destination_root / resolved.name
                bindings[spec.config_key] = relative.as_posix()
                files.append(self._file_entry(relative, resolved))
                if resolved.suffix.casefold() == ".xml":
                    companion = resolved.with_suffix(".bin")
                    if not companion.is_file():
                        raise ModelSyncError(
                            f"configured {spec.config_key} XML has no BIN companion"
                        )
                    files.append(
                        self._file_entry(
                            destination_root / companion.name,
                            companion,
                        )
                    )
                if spec.config_key in {"model_path", "model_xml"}:
                    for sidecar_name in ("metadata.yaml", "classes.txt"):
                        sidecar = resolved.parent / sidecar_name
                        if not sidecar.exists():
                            continue
                        if not sidecar.is_file():
                            raise ModelSyncError(
                                f"configured object model has an unreadable {sidecar_name}"
                            )
                        files.append(
                            self._file_entry(
                                destination_root / sidecar.name,
                                sidecar,
                            )
                        )
            else:
                raise ModelSyncError(
                    f"configured {spec.config_key} is not a regular file or directory"
                )
            if len(files) > MAX_MODEL_BUNDLE_FILES:
                raise ModelSyncError("model bundle contains too many files")
            if sum(item[3] for item in files) > MAX_MODEL_BUNDLE_BYTES:
                raise ModelSyncError("model bundle is too large")
        files.sort(key=lambda item: item[0])
        return files, bindings

    def _file_entry(
        self,
        relative: PurePosixPath,
        source: Path,
    ) -> tuple[str, Path, str, int]:
        stat_result = source.stat()
        if stat_result.st_size <= 0:
            raise ModelSyncError(f"model artifact is empty: {source.name}")
        key = (
            str(source),
            int(stat_result.st_ino),
            int(stat_result.st_size),
            int(stat_result.st_mtime_ns),
            int(stat_result.st_ctime_ns),
        )
        digest = self._hash_cache.get(key)
        if digest is None:
            hasher = hashlib.sha256()
            with source.open("rb") as handle:
                while chunk := handle.read(_HASH_CHUNK_BYTES):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            self._hash_cache = {
                cached_key: cached_digest
                for cached_key, cached_digest in self._hash_cache.items()
                if cached_key[0] != str(source)
            }
            self._hash_cache[key] = digest
        return relative.as_posix(), source, digest, int(stat_result.st_size)


class WorkerModelCache:
    """Receives verified model blobs and atomically materializes a bundle."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.blob_dir = self.root / "blobs"
        self.bundle_dir = self.root / "bundles"

    def available_digests(self) -> list[str]:
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        available: list[str] = []
        for path in sorted(self.blob_dir.iterdir()):
            if (
                not path.is_file()
                or len(path.name) != 64
                or any(character not in "0123456789abcdef" for character in path.name)
            ):
                continue
            if self._digest(path) == path.name:
                available.append(path.name)
            else:
                path.unlink(missing_ok=True)
        return available

    def receive(
        self,
        websocket: Any,
        manifest: ModelBundleManifest,
    ) -> None:
        from .protocol import decode_packet

        self.blob_dir.mkdir(parents=True, exist_ok=True)
        expected = {
            item.digest: item.size
            for item in manifest.files
            if not self._valid_blob(item.digest, item.size)
        }
        temporary: dict[str, tuple[Path, Any, int]] = {}
        try:
            while True:
                incoming = websocket.recv(timeout=60.0)
                if isinstance(incoming, str):
                    control = json.loads(incoming)
                    if (
                        not isinstance(control, dict)
                        or control.get("type") != "model_sync_complete"
                        or control.get("generation") != manifest.generation
                    ):
                        raise ModelSyncError(
                            "primary sent an invalid model synchronization control"
                        )
                    if expected:
                        raise ModelSyncError(
                            "primary completed model synchronization early"
                        )
                    break
                if not isinstance(incoming, bytes):
                    raise ModelSyncError(
                        "primary sent an invalid model synchronization packet"
                    )
                message, chunk = decode_packet(incoming)
                if message.get("type") != "model_chunk" or not chunk:
                    raise ModelSyncError("primary sent an invalid model chunk")
                digest = str(message.get("digest") or "")
                if digest not in expected:
                    raise ModelSyncError("primary sent an unexpected model chunk")
                try:
                    offset = int(message.get("offset"))
                    total_size = int(message.get("total_size"))
                except (TypeError, ValueError) as error:
                    raise ModelSyncError("primary sent invalid model offsets") from error
                if total_size != expected[digest]:
                    raise ModelSyncError("primary sent a mismatched model size")
                entry = temporary.get(digest)
                if entry is None:
                    temporary_path = self.blob_dir / (
                        f".{digest}.{os.getpid()}.tmp"
                    )
                    handle = temporary_path.open("wb")
                    entry = (temporary_path, handle, 0)
                    temporary[digest] = entry
                temporary_path, handle, written = entry
                if offset != written or written + len(chunk) > total_size:
                    raise ModelSyncError("primary sent a non-contiguous model chunk")
                handle.write(chunk)
                written += len(chunk)
                temporary[digest] = (temporary_path, handle, written)
                if written == total_size:
                    handle.flush()
                    os.fsync(handle.fileno())
                    handle.close()
                    if self._digest(temporary_path) != digest:
                        raise ModelSyncError("model digest verification failed")
                    final_path = self.blob_dir / digest
                    os.chmod(temporary_path, 0o600)
                    os.replace(temporary_path, final_path)
                    temporary.pop(digest)
                    expected.pop(digest)
        finally:
            for temporary_path, handle, _written in temporary.values():
                handle.close()
                temporary_path.unlink(missing_ok=True)

    def materialize(
        self,
        config: DetectorConfig,
        manifest: ModelBundleManifest,
    ) -> DetectorConfig:
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        final_root = self.bundle_dir / manifest.generation
        if not final_root.is_dir():
            staging = self.bundle_dir / (
                f".{manifest.generation}.{os.getpid()}.tmp"
            )
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            try:
                for item in manifest.files:
                    blob = self.blob_dir / item.digest
                    if not self._valid_blob(item.digest, item.size):
                        raise ModelSyncError(
                            f"verified model blob is unavailable: {item.digest}"
                        )
                    destination = staging.joinpath(*PurePosixPath(item.path).parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(blob, destination)
                    except OSError:
                        shutil.copyfile(blob, destination)
                    os.chmod(destination, 0o600)
                try:
                    os.replace(staging, final_root)
                except OSError:
                    if not final_root.is_dir():
                        raise
                    shutil.rmtree(staging, ignore_errors=True)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        worker_config = config.model_copy(deep=True)
        for key, relative in manifest.bindings.items():
            _set_config_value(
                worker_config,
                key,
                str(final_root.joinpath(*PurePosixPath(relative).parts)),
            )
        self._prune(manifest, final_root)
        return worker_config

    def _prune(
        self,
        manifest: ModelBundleManifest,
        current_bundle: Path,
    ) -> None:
        for path in self.bundle_dir.iterdir():
            if path != current_bundle:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                elif path.name.startswith("."):
                    path.unlink(missing_ok=True)
        retained = {item.digest for item in manifest.files}
        for path in self.blob_dir.iterdir():
            if path.is_file() and path.name not in retained:
                path.unlink(missing_ok=True)

    def _valid_blob(self, digest: str, size: int) -> bool:
        path = self.blob_dir / digest
        return path.is_file() and path.stat().st_size == size

    @staticmethod
    def _digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                hasher.update(chunk)
        return hasher.hexdigest()
