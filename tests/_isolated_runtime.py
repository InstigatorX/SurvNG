"""One temporary application configuration for every supported test runner."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

_RUNTIME = tempfile.TemporaryDirectory(prefix='survng-test-')
_ROOT = Path(_RUNTIME.name)
_CONFIG_PATH = _ROOT / 'config.json'
_CONFIG_PATH.write_text(json.dumps({
    'storage_dir':str(_ROOT / 'storage'),
    'database_dir':str(_ROOT / 'database'),
    'recording_index_dir':str(_ROOT / 'recording-index'),
    'cameras':[],
    'retention':{'enabled':False},
}),encoding='utf-8')


def isolate_runtime():
    # Override inherited deployment settings: tests must opt into their own
    # fixtures, never fall through to the live application's configuration.
    os.environ['SURVNG_CONFIG_PATH'] = str(_CONFIG_PATH)
