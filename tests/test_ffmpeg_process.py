from __future__ import annotations

import os
import shutil
from pathlib import Path

from survng.app.ffmpeg_process import named_ffmpeg_executable


def test_named_executable_preserves_path_lookup(tmp_path: Path) -> None:
    expected = shutil.which("sh")
    assert expected is not None

    executable = named_ffmpeg_executable(
        "sh",
        "survng-test",
        runtime_dir=tmp_path,
    )

    assert Path(executable) == tmp_path / "survng-test"
    assert os.path.realpath(executable) == os.path.realpath(expected)


def test_named_executable_leaves_missing_command_for_popen_error(
    tmp_path: Path,
) -> None:
    executable = named_ffmpeg_executable(
        "definitely-missing-ffmpeg",
        "survng-test",
        runtime_dir=tmp_path,
    )

    assert executable == "definitely-missing-ffmpeg"
    assert not (tmp_path / "survng-test").exists()
