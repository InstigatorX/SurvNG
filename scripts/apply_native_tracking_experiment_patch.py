"""One-shot patch helper for PR #206. Deleted after the experiment commit."""

from __future__ import annotations

from pathlib import Path


def replace(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace(
    "survng/app/config.py",
    (
        "class DetectorConfig(BaseModel):\n"
        "    enabled: bool = False\n"
        "    # Retained for native continuous-inference integrations; production capture\n"
        "    # is frames-only and qualification schedules inference in the shared pool.\n"
        "    live_sample_fps: float = Field(default=5.0, ge=0.5, le=10.0)\n"
        "    backend: Literal[\"openvino\", \"coreml\"] = \"openvino\"\n"
    ),
    (
        "class DetectorConfig(BaseModel):\n"
        "    enabled: bool = False\n"
        "    # Native live analytics is deliberately opt-in. SurvNG Hybrid remains the\n"
        "    # authoritative event tracker even when DL Streamer supplies short-term IDs.\n"
        "    live_sample_fps: float = Field(default=5.0, ge=0.5, le=10.0)\n"
        "    live_pipeline_inference_enabled: bool = False\n"
        "    live_pipeline_inference_interval: int = Field(default=1, ge=1, le=5)\n"
        "    live_pipeline_tracking: Literal[\"off\", \"short-term-imageless\"] = \"off\"\n"
        "    backend: Literal[\"openvino\", \"coreml\"] = \"openvino\"\n"
    ),
)

replace(
    "survng/app/manager.py",
    (
        "                inference_device=detector.device,\n"
        "                # Qualification submits evidence to the priority-aware pool.\n"
        "                detect_enabled=False,\n"
        "                labels_path=detector.labels_path,\n"
    ),
    (
        "                inference_device=detector.device,\n"
        "                # Optional native live analytics experiment. DL Streamer may\n"
        "                # fill skipped-frame ROIs, but SurvNG Hybrid remains authoritative.\n"
        "                detect_enabled=(\n"
        "                    detector.enabled and detector.live_pipeline_inference_enabled\n"
        "                ),\n"
        "                inference_interval=detector.live_pipeline_inference_interval,\n"
        "                native_tracking=detector.live_pipeline_tracking,\n"
        "                labels_path=detector.labels_path,\n"
    ),
)

replace(
    "survng/app/dlstreamer_capture.py",
    (
        "    inference_device: str = \"GPU\"\n"
        "    detect_enabled: bool = False\n"
        "    frame_width: int = 320\n"
    ),
    (
        "    inference_device: str = \"GPU\"\n"
        "    detect_enabled: bool = False\n"
        "    inference_interval: int = 1\n"
        "    native_tracking: str = \"off\"\n"
        "    frame_width: int = 320\n"
    ),
)

replace(
    "survng/app/dlstreamer_capture.py",
    (
        "        if self.options.decoder not in {\"auto\", \"va\"}:\n"
        "            raise ValueError(\"decoder must be auto or va\")\n"
        "        self._credential_warning_lock = threading.Lock()\n"
    ),
    (
        "        if self.options.decoder not in {\"auto\", \"va\"}:\n"
        "            raise ValueError(\"decoder must be auto or va\")\n"
        "        if not 1 <= int(self.options.inference_interval) <= 5:\n"
        "            raise ValueError(\"inference_interval must be between 1 and 5\")\n"
        "        if self.options.native_tracking not in {\"off\", \"short-term-imageless\"}:\n"
        "            raise ValueError(\"native_tracking must be off or short-term-imageless\")\n"
        "        self._credential_warning_lock = threading.Lock()\n"
    ),
)

replace(
    "survng/app/dlstreamer_capture.py",
    (
        "        if self.options.detect_enabled and model_path:\n"
        "            command.extend([\"--model\", model_path])\n"
        "            command.extend(\n"
    ),
    (
        "        if self.options.detect_enabled and model_path:\n"
        "            command.extend([\"--model\", model_path])\n"
        "            command.extend([\"--inference-interval\", str(int(self.options.inference_interval))])\n"
        "            command.extend([\"--native-tracking\", self.options.native_tracking])\n"
        "            command.extend(\n"
    ),
)

replace(
    "survng/dlstreamer_live.py",
    (
        "    parser.add_argument(\n"
        "        \"--detect-fps\",\n"
        "        type=float,\n"
        "        default=5.0,\n"
        "        help=\"maximum gvadetect input rate; independent from EMA qualification FPS\",\n"
        "    )\n"
        "    parser.add_argument(\"--open-timeout\", type=float, default=3.0)\n"
    ),
    (
        "    parser.add_argument(\n"
        "        \"--detect-fps\",\n"
        "        type=float,\n"
        "        default=5.0,\n"
        "        help=\"maximum gvadetect input rate; independent from EMA qualification FPS\",\n"
        "    )\n"
        "    parser.add_argument(\n"
        "        \"--inference-interval\",\n"
        "        type=int,\n"
        "        default=1,\n"
        "        help=\"run gvadetect every Nth detect-branch frame (1-5)\",\n"
        "    )\n"
        "    parser.add_argument(\n"
        "        \"--native-tracking\",\n"
        "        choices=(\"off\", \"short-term-imageless\"),\n"
        "        default=\"off\",\n"
        "        help=\"optional DL Streamer ROI tracking between detector frames\",\n"
        "    )\n"
        "    parser.add_argument(\"--open-timeout\", type=float, default=3.0)\n"
    ),
)

replace(
    "survng/dlstreamer_live.py",
    (
        "    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:\n"
        "        raise ValueError(\"detection threshold must be between 0 and 1\")\n"
        "    instance_id = (\n"
    ),
    (
        "    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:\n"
        "        raise ValueError(\"detection threshold must be between 0 and 1\")\n"
        "    if not 1 <= args.inference_interval <= 5:\n"
        "        raise ValueError(\"inference interval must be between 1 and 5\")\n"
        "    instance_id = (\n"
    ),
)

replace(
    "survng/dlstreamer_live.py",
    (
        "        try:\n"
        "            objects.append(\n"
        "                {\n"
        "                    \"label\": label or \"object\",\n"
        "                    \"confidence\": round(confidence, 4),\n"
        "                    \"box\": {\n"
        "                        \"x1\": int(x1),\n"
        "                        \"y1\": int(y1),\n"
        "                        \"x2\": int(x2),\n"
        "                        \"y2\": int(y2),\n"
        "                    },\n"
        "                }\n"
        "            )\n"
        "        except (TypeError, ValueError):\n"
        "            continue\n"
    ),
    (
        "        try:\n"
        "            normalized = {\n"
        "                \"label\": label or \"object\",\n"
        "                \"confidence\": round(confidence, 4),\n"
        "                \"box\": {\n"
        "                    \"x1\": int(x1),\n"
        "                    \"y1\": int(y1),\n"
        "                    \"x2\": int(x2),\n"
        "                    \"y2\": int(y2),\n"
        "                },\n"
        "            }\n"
        "            # gvatrack IDs are diagnostic input only. SurvNG Hybrid owns\n"
        "            # persisted/event track identity and ignores this field today.\n"
        "            native_track_id = item.get(\"object_id\")\n"
        "            if native_track_id is None:\n"
        "                native_track_id = detection.get(\"object_id\")\n"
        "            if native_track_id is not None:\n"
        "                native_track_id = int(native_track_id)\n"
        "                if native_track_id >= 0:\n"
        "                    normalized[\"native_track_id\"] = native_track_id\n"
        "            objects.append(normalized)\n"
        "        except (TypeError, ValueError):\n"
        "            continue\n"
    ),
)

replace(
    "survng/dlstreamer_live.py",
    (
        "    detector = None\n"
        "    detect_output_queue = None\n"
        "    meta_convert = None\n"
    ),
    (
        "    detector = None\n"
        "    detect_output_queue = None\n"
        "    native_tracker = None\n"
        "    meta_convert = None\n"
    ),
)

replace(
    "survng/dlstreamer_live.py",
    (
        "        detector.set_property(\"batch-size\", 1)\n"
        "        detector.set_property(\"nireq\", 1)\n"
        "        detector.set_property(\"inference-interval\", 1)\n"
        "        detector.set_property(\"threshold\", args.threshold)\n"
    ),
    (
        "        detector.set_property(\"batch-size\", 1)\n"
        "        detector.set_property(\"nireq\", 1)\n"
        "        detector.set_property(\"inference-interval\", args.inference_interval)\n"
        "        detector.set_property(\"threshold\", args.threshold)\n"
    ),
)

replace(
    "survng/dlstreamer_live.py",
    (
        "        detect_output_queue.set_property(\"max-size-time\", 0)\n"
        "        detect_output_queue.set_property(\"leaky\", 2)\n"
        "        elements.extend([detect_queue, detect_rate_el, detect_rate_caps, detector, detect_output_queue])\n"
        "        if preprocess.startswith(\"va\"):\n"
    ),
    (
        "        detect_output_queue.set_property(\"max-size-time\", 0)\n"
        "        detect_output_queue.set_property(\"leaky\", 2)\n"
        "        if args.native_tracking != \"off\":\n"
        "            if not _factory_available(Gst, \"gvatrack\"):\n"
        "                raise InferencePipelineError(\n"
        "                    \"native tracking requested but gvatrack is unavailable\"\n"
        "                )\n"
        "            native_tracker = _element(Gst, \"gvatrack\", \"native-track\")\n"
        "            native_tracker.set_property(\"tracking-type\", args.native_tracking)\n"
        "        elements.extend([detect_queue, detect_rate_el, detect_rate_caps, detector, detect_output_queue])\n"
        "        if native_tracker is not None:\n"
        "            elements.append(native_tracker)\n"
        "        if preprocess.startswith(\"va\"):\n"
    ),
)

replace(
    "survng/dlstreamer_live.py",
    (
        "        if detect_output_queue is None or not detector.link(detect_output_queue):\n"
        "            raise RuntimeError(\"could not link gvadetect output queue\")\n"
        "        if meta_convert is not None and meta_sink is not None:\n"
        "            if not detect_output_queue.link(meta_convert) or not meta_convert.link(meta_sink):\n"
        "                raise RuntimeError(\"could not link detection metadata branch\")\n"
        "        else:\n"
        "            fake = pipeline.get_by_name(\"detect-sink\")\n"
        "            if fake is None or not detect_output_queue.link(fake):\n"
        "                raise RuntimeError(\"could not link detection sink\")\n"
    ),
    (
        "        if detect_output_queue is None or not detector.link(detect_output_queue):\n"
        "            raise RuntimeError(\"could not link gvadetect output queue\")\n"
        "        metadata_source = detect_output_queue\n"
        "        if native_tracker is not None:\n"
        "            if not detect_output_queue.link(native_tracker):\n"
        "                raise RuntimeError(\"could not link gvatrack\")\n"
        "            metadata_source = native_tracker\n"
        "        if meta_convert is not None and meta_sink is not None:\n"
        "            if not metadata_source.link(meta_convert) or not meta_convert.link(meta_sink):\n"
        "                raise RuntimeError(\"could not link detection metadata branch\")\n"
        "        else:\n"
        "            fake = pipeline.get_by_name(\"detect-sink\")\n"
        "            if fake is None or not metadata_source.link(fake):\n"
        "                raise RuntimeError(\"could not link detection sink\")\n"
    ),
)

replace(
    "survng/dlstreamer_live.py",
    (
        "                    if detector is not None and message.src == detector:\n"
        "                        raise InferencePipelineError(error)\n"
    ),
    (
        "                    if (\n"
        "                        (detector is not None and message.src == detector)\n"
        "                        or (native_tracker is not None and message.src == native_tracker)\n"
        "                    ):\n"
        "                        raise InferencePipelineError(error)\n"
    ),
)

replace(
    "survng/dlstreamer_live.py",
    (
        "                            \"detect_fps\": float(detect_rate),\n"
        "                            \"inference_interval\": 1,\n"
        "                            \"jpeg_preview\": jpeg_sink is not None,\n"
    ),
    (
        "                            \"detect_fps\": float(detect_rate),\n"
        "                            \"inference_interval\": args.inference_interval if detect else None,\n"
        "                            \"effective_inference_fps\": (\n"
        "                                round(float(detect_rate) / args.inference_interval, 3)\n"
        "                                if detect else 0.0\n"
        "                            ),\n"
        "                            \"native_tracking\": args.native_tracking if detect else \"off\",\n"
        "                            \"native_tracking_authoritative\": False,\n"
        "                            \"native_tracking_memory\": _negotiated_memory(native_tracker),\n"
        "                            \"jpeg_preview\": jpeg_sink is not None,\n"
    ),
)

replace(
    "config.example.json",
    (
        "    \"enabled\": false,\n"
        "    \"live_sample_fps\": 5.0,\n"
        "    \"object_worker_count\": 2,\n"
    ),
    (
        "    \"enabled\": false,\n"
        "    \"live_sample_fps\": 5.0,\n"
        "    \"live_pipeline_inference_enabled\": false,\n"
        "    \"live_pipeline_inference_interval\": 1,\n"
        "    \"live_pipeline_tracking\": \"off\",\n"
        "    \"object_worker_count\": 2,\n"
    ),
)

replace(
    "scripts/gstreamer-smoke.py",
    (
        "def consume(model: Path, proc: Path | None, threshold: float, source_role=\"live\", *, nms_threshold=.45, expected_objects=None, detect=True) -> dict:\n"
        "    command = [sys.executable, \"-m\", \"survng.dlstreamer_live\", \"--test-source\",\n"
        "               \"--decoder\", \"auto\", \"--device\", \"CPU\", \"--fps\", \"5\", \"--detect-fps\", \"2.5\",\n"
        "               \"--model\", str(model), \"--threshold\", str(threshold), \"--nms-threshold\", str(nms_threshold),\n"
        "               \"--jpeg-fps\", \"0\", \"--open-timeout\", \"15\"]\n"
    ),
    (
        "def consume(model: Path, proc: Path | None, threshold: float, source_role=\"live\", *, nms_threshold=.45, expected_objects=None, detect=True, inference_interval=1, native_tracking=\"off\") -> dict:\n"
        "    command = [sys.executable, \"-m\", \"survng.dlstreamer_live\", \"--test-source\",\n"
        "               \"--decoder\", \"auto\", \"--device\", \"CPU\", \"--fps\", \"5\", \"--detect-fps\", \"2.5\",\n"
        "               \"--inference-interval\", str(inference_interval), \"--native-tracking\", native_tracking,\n"
        "               \"--model\", str(model), \"--threshold\", str(threshold), \"--nms-threshold\", str(nms_threshold),\n"
        "               \"--jpeg-fps\", \"0\", \"--open-timeout\", \"15\"]\n"
    ),
)

replace(
    "scripts/gstreamer-smoke.py",
    (
        "    if source_role == \"live\" and detect:\n"
        "        assert len(snapshots) >= 3 and matched > 0, (len(snapshots), matched, errors[-4000:])\n"
        "    else:\n"
    ),
    (
        "    if source_role == \"live\" and detect:\n"
        "        assert len(snapshots) >= 3 and matched > 0, (len(snapshots), matched, errors[-4000:])\n"
        "        assert status.get(\"inference_interval\") == inference_interval, status\n"
        "        assert status.get(\"native_tracking\") == native_tracking, status\n"
        "        assert status.get(\"native_tracking_authoritative\") is False, status\n"
        "    else:\n"
    ),
)

replace(
    "scripts/gstreamer-smoke.py",
    (
        "    return {\"threshold\": threshold, \"source_role\": source_role, \"frames\": len(frames), \"snapshots\": len(snapshots),\n"
        "            \"nms_threshold\": nms_threshold, \"objects_per_snapshot\": expected_objects,\n"
        "            \"positive_snapshots\": sum(bool(s.objects) for s in snapshots),\n"
        "            \"matched_at_receipt\": matched,\n"
        "            \"frame_fps\": round((len(frames)-1)/(frames[-1]-frames[0]), 2),\n"
        "            \"metadata_contract\": status.get(\"metadata_contract\")}\n"
    ),
    (
        "    return {\"threshold\": threshold, \"source_role\": source_role, \"frames\": len(frames), \"snapshots\": len(snapshots),\n"
        "            \"nms_threshold\": nms_threshold, \"objects_per_snapshot\": expected_objects,\n"
        "            \"positive_snapshots\": sum(bool(s.objects) for s in snapshots),\n"
        "            \"matched_at_receipt\": matched,\n"
        "            \"frame_fps\": round((len(frames)-1)/(frames[-1]-frames[0]), 2),\n"
        "            \"inference_interval\": inference_interval,\n"
        "            \"native_tracking\": native_tracking,\n"
        "            \"native_track_ids\": sum(\n"
        "                \"native_track_id\" in obj for snapshot in snapshots for obj in snapshot.objects\n"
        "            ),\n"
        "            \"metadata_contract\": status.get(\"metadata_contract\")}\n"
    ),
)

replace(
    "scripts/gstreamer-smoke.py",
    (
        "        results.extend(consume(model_path, proc_path, threshold) for threshold in (0.1, 1.0))\n"
        "        results.append(consume(model_path, proc_path, .1, source_role=\"main\"))\n"
    ),
    (
        "        results.extend(consume(model_path, proc_path, threshold) for threshold in (0.1, 1.0))\n"
        "        results.append(consume(\n"
        "            model_path, proc_path, .1, inference_interval=3,\n"
        "            native_tracking=\"short-term-imageless\",\n"
        "        ))\n"
        "        results.append(consume(model_path, proc_path, .1, source_role=\"main\"))\n"
    ),
)

Path("tests/test_dlstreamer_native_tracking_experiment.py").write_text(
    '''from __future__ import annotations

import pytest

from survng.app.camera_capture import CaptureOpenLimiter
from survng.app.config import DetectorConfig
from survng.app.dlstreamer_capture import DlStreamerCaptureBackend, DlStreamerCaptureOptions
from survng.dlstreamer_live import _normalize_gva_objects, _parser


def test_native_live_pipeline_defaults_are_off() -> None:
    config = DetectorConfig()
    assert config.live_pipeline_inference_enabled is False
    assert config.live_pipeline_inference_interval == 1
    assert config.live_pipeline_tracking == "off"


def test_backend_command_carries_native_tracking_policy() -> None:
    backend = DlStreamerCaptureBackend(
        CaptureOpenLimiter(1),
        DlStreamerCaptureOptions(
            detect_enabled=True,
            model_path="/models/yolo.xml",
            inference_interval=3,
            native_tracking="short-term-imageless",
        ),
    )
    command = backend.command()
    assert command[command.index("--inference-interval") + 1] == "3"
    assert command[command.index("--native-tracking") + 1] == "short-term-imageless"
    assert "--no-detect" not in command


def test_backend_rejects_interval_beyond_short_term_tracker_contract() -> None:
    with pytest.raises(ValueError, match="inference_interval"):
        DlStreamerCaptureBackend(
            CaptureOpenLimiter(1),
            DlStreamerCaptureOptions(inference_interval=6),
        )


def test_live_parser_accepts_native_tracking_policy() -> None:
    args = _parser().parse_args(
        ["--inference-interval", "3", "--native-tracking", "short-term-imageless"]
    )
    assert args.inference_interval == 3
    assert args.native_tracking == "short-term-imageless"


def test_gva_normalization_preserves_native_track_id_as_diagnostic_metadata() -> None:
    objects = _normalize_gva_objects(
        {
            "objects": [
                {
                    "x": 10,
                    "y": 20,
                    "w": 30,
                    "h": 40,
                    "object_id": 17,
                    "detection": {"label": "person", "confidence": 0.91},
                }
            ]
        }
    )
    assert objects == [
        {
            "label": "person",
            "confidence": 0.91,
            "box": {"x1": 10, "y1": 20, "x2": 40, "y2": 60},
            "native_track_id": 17,
        }
    ]
''',
    encoding="utf-8",
)

Path("docs/dlstreamer-native-tracking-experiment.md").write_text(
    '''# DL Streamer native tracking experiment

This experiment keeps SurvNG Hybrid as the authoritative event tracker while
allowing the live/substream GStreamer branch to run `gvadetect` and optionally
`gvatrack` before metadata crosses into SurvNG.

The experiment is **off by default**. It does not replace Hybrid IDs, recorded
main-stream refinement, ReID, incident persistence, or zone policy.

## Why test it

Intel DL Streamer `short-term-imageless` tracking can extrapolate ROI positions
on frames where object detection is skipped. That makes it possible to lower
live detector cadence with `inference-interval` while keeping a denser stream
of object boxes. `gvatrack` itself is CPU work; the expected trade is lower GPU
inference load for some additional CPU tracking work.

## Configuration

Under `detector`, start with:

```json
{
  "enabled": true,
  "live_pipeline_inference_enabled": true,
  "live_pipeline_inference_interval": 3,
  "live_pipeline_tracking": "short-term-imageless"
}
```

`live_pipeline_inference_interval` is intentionally bounded to 1-5, matching
the supported detector interval for DL Streamer's short-term imageless tracker.

Baselines worth comparing on the same cameras:

1. Current/default path: `live_pipeline_inference_enabled=false`.
2. Native detection only: enabled, interval `1`, tracking `off`.
3. Native tracking assist: enabled, interval `2` or `3`, tracking
   `short-term-imageless`.

## What remains authoritative

`native_track_id` is carried through when DL Streamer exposes an object ID, but
it is diagnostic metadata only. SurvNG Hybrid still performs event-scoped
association, high/low-confidence handling, appearance recovery, persisted track
IDs, cover selection, and recorded catch-up.

## What to measure

Camera pipeline status reports:

- `detect`
- `detect_fps`
- `inference_interval`
- `effective_inference_fps`
- `native_tracking`
- `native_tracking_authoritative` (always `false` in this experiment)
- `native_tracking_memory`

Compare Intel GPU utilization, CPU utilization, detector latency, incident recall,
duplicate/fragmented Hybrid tracks, and small/distant-object misses. Do not judge
the experiment on FPS alone.

## Rollback

Set `live_pipeline_inference_enabled` to `false` and restart. The default runtime
then returns to the existing frames-only live capture path.
''',
    encoding="utf-8",
)
