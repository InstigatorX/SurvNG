"""Global test-process isolation from the live SurvNG runtime."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


_RUNTIME = tempfile.TemporaryDirectory(prefix="survng-pytest-")
_ROOT = Path(_RUNTIME.name)
_CONFIG_PATH = _ROOT / "config.json"
_CONFIG_PATH.write_text(
    json.dumps(
        {
            "storage_dir": str(_ROOT / "storage"),
            "database_dir": str(_ROOT / "database"),
            "recording_index_dir": str(_ROOT / "recording-index"),
            "cameras": [],
            "retention": {"enabled": False},
        }
    ),
    encoding="utf-8",
)

# Several route tests import survng.app.main during collection. main constructs
# its application manager at import time, so this must be set before any test
# module can accidentally initialize or migrate the production databases.
os.environ["SURVNG_CONFIG_PATH"] = str(_CONFIG_PATH)

