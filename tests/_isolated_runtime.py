"""One temporary application configuration for every supported test runner."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from survng.app.config import AppConfig, primary_media_storage, save_config

_RUNTIME = tempfile.TemporaryDirectory(prefix='survng-test-')
_ROOT = Path(_RUNTIME.name)
_CONFIG_PATH = _ROOT / 'config.json'


def isolate_runtime():
    # Override inherited deployment settings: tests must opt into their own
    # fixtures, never fall through to the live application's configuration.
    os.environ['SURVNG_CONFIG_PATH'] = str(_CONFIG_PATH)
    os.environ.pop('SURVNG_SECRETS_PATH', None)
    os.environ.pop('SURVNG_SECRETS_KEY_PATH', None)
    if not _CONFIG_PATH.exists():
        save_config(AppConfig(
            storage_dir=str(_ROOT / 'storage'),
            database_dir=str(_ROOT / 'database'),
            recording_index_dir=str(_ROOT / 'recording-index'),
            media_storage=primary_media_storage(str(_ROOT / 'storage')),
            cameras=[],
            retention={'enabled': False},
        ), _CONFIG_PATH)
