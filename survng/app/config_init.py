"""Initialize a new configuration and encrypted secrets sidecar from a template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AppConfig, primary_media_storage, save_config
from .config_secrets import assert_secrets_materialized, secrets_key_path, secrets_path


def initialize_config(
    template: Path,
    destination: Path,
    *,
    storage_dir: str | None = None,
    ffmpeg_path: str | None = None,
) -> None:
    """Create fresh artifacts; never convert or overwrite an existing installation."""
    sidecar = secrets_path(destination)
    key = secrets_key_path(sidecar)
    artifacts = (destination, sidecar, key)
    if len({path.resolve() for path in artifacts}) != len(artifacts):
        raise ValueError("config, secrets, and key paths must be distinct")
    for path in artifacts:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing configuration artifact: {path}")
    config = AppConfig.model_validate(json.loads(template.read_text(encoding="utf-8")))
    if storage_dir is not None:
        config.storage_dir = storage_dir
        config.media_storage = primary_media_storage(storage_dir)
    if ffmpeg_path is not None:
        config.ffmpeg_path = ffmpeg_path
    assert_secrets_materialized(config)
    save_config(config, destination, assign_ids=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--storage-dir")
    parser.add_argument("--ffmpeg-path")
    args = parser.parse_args()
    initialize_config(args.template, args.config, storage_dir=args.storage_dir, ffmpeg_path=args.ffmpeg_path)


if __name__ == "__main__":
    main()
