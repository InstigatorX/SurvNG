"""Native CPU integration check: real GStreamer + gvadetect, synthetic model.

Run in the Intel image; no cameras, model downloads, database or service needed.
The synthetic SSD-shaped model checks plumbing, not recognition accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survng.app.dlstreamer_protocol import MessageReader, TYPE_FRAME, TYPE_DETECTIONS, TYPE_STATUS, decode_frame_payload, decode_json_payload, decode_stream_payload
from survng.app.live_detections import DetectionHistory, DetectionSnapshot


def consume(model: Path, proc: Path | None, threshold: float, source_role="live", *, nms_threshold=.45, expected_objects=None, detect=True) -> dict:
    command = [sys.executable, "-m", "survng.dlstreamer_live", "--test-source",
               "--decoder", "auto", "--device", "CPU", "--fps", "5", "--detect-fps", "2.5",
               "--model", str(model), "--threshold", str(threshold), "--nms-threshold", str(nms_threshold),
               "--jpeg-fps", "0", "--open-timeout", "15"]
    command.extend(["--source-role", source_role])
    if not detect:
        command.append("--no-detect")
    if proc is not None:
        command.extend(["--model-proc", str(proc)])
    command.extend(["--labels", str(model.with_suffix(".txt"))])
    reader = MessageReader()
    frames = []
    snapshots = []
    status = {}
    history = DetectionHistory()
    history.reset("native")
    matched = 0
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr, stdin=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 20
            first = None
            while time.monotonic() < deadline:
                if not select.select([process.stdout], [], [], 0.2)[0]:
                    continue
                chunk = process.stdout.read1(65536)
                if not chunk:
                    break
                reader.feed(chunk)
                while (message := reader.pop()) is not None:
                    kind, payload = message
                    if kind == TYPE_FRAME:
                        width, height, sequence, pts, pixels = decode_frame_payload(payload)
                        assert width == (320 if source_role == "live" else 640)
                        assert len(pixels) == width * height * (1 if source_role == "live" and detect else 3)
                        assert math.isfinite(pts)
                        frames.append(pts)
                        matched += history.match(pts=pts, session="native", detect_fps=2.5) is not None
                        first = first or time.monotonic()
                    elif kind == TYPE_DETECTIONS:
                        snapshot = DetectionSnapshot.parse(decode_json_payload(payload), session="native")
                        snapshots.append(snapshot)
                        history.add(snapshot)
                    elif kind == TYPE_STATUS:
                        status.update(decode_json_payload(payload))
                if first and time.monotonic() - first >= 3 and (len(snapshots) >= 3 or source_role == "main" or not detect):
                    break
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()
        stderr.seek(0)
        errors = stderr.read().decode(errors="replace")
    assert status.get("ok") and status.get("detect") == (source_role == "live" and detect), (status, errors[-4000:])
    assert len(frames) >= 10, (len(frames), len(snapshots), errors[-4000:])
    if source_role == "live" and detect:
        assert len(snapshots) >= 3 and matched > 0, (len(snapshots), matched, errors[-4000:])
    else:
        assert snapshots == [], "frames-only capture must not run gvadetect"
    assert all(b > a for a, b in zip(frames, frames[1:]))
    assert all(b.source_pts > a.source_pts for a, b in zip(snapshots, snapshots[1:]))
    assert all(bool(item.objects) == (threshold < 1) for item in snapshots)
    assert all(obj["label"] == "car" for item in snapshots for obj in item.objects), "labels-file must override model-proc labels"
    if expected_objects is not None:
        assert all(len(item.objects) == expected_objects for item in snapshots), [len(item.objects) for item in snapshots]
    assert len(frames) > len(snapshots), "detector cadence must not throttle EMA"
    return {"threshold": threshold, "source_role": source_role, "frames": len(frames), "snapshots": len(snapshots),
            "nms_threshold": nms_threshold, "objects_per_snapshot": expected_objects,
            "positive_snapshots": sum(bool(s.objects) for s in snapshots),
            "matched_at_receipt": matched,
            "frame_fps": round((len(frames)-1)/(frames[-1]-frames[0]), 2),
            "metadata_contract": status.get("metadata_contract")}


def shared_supervisor(model: Path, proc: Path, *, device: str = "CPU") -> dict:
    """Exercise the production demultiplexer contract and shared model pool."""
    roles = {"gate-live": "live", "yard-live": "live", "gate-main": "main"}
    widths = {"gate-live": 320, "yard-live": 960, "gate-main": 640}
    counts = {key: {"frames": 0, "snapshots": 0, "ok": False} for key in roles}
    command = [sys.executable, "-m", "survng.dlstreamer_live", "--supervisor", "--test-source",
               "--decoder", "auto", "--device", device, "--fps", "5", "--detect-fps", "2.5",
               "--model", str(model), "--model-proc", str(proc), "--labels", str(model.with_suffix(".txt")),
               "--threshold", "0.1", "--nms-threshold", "0.45", "--jpeg-fps", "0", "--open-timeout", "15"]
    reader = MessageReader()
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=stderr)
        try:
            for stream_id, role in roles.items():
                process.stdin.write((json.dumps({"op": "add", "stream_id": stream_id,
                                                "source_role": role, "frame_width": widths[stream_id],
                                                "url": "rtsp://fixture.invalid/video"}) + "\n").encode())
            process.stdin.flush()
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if not select.select([process.stdout], [], [], .2)[0]:
                    continue
                chunk = process.stdout.read1(65536)
                if not chunk:
                    break
                reader.feed(chunk)
                while (message := reader.pop()) is not None:
                    kind, body = message
                    stream_id, body = decode_stream_payload(body)
                    entry = counts[stream_id]
                    if kind == TYPE_FRAME:
                        width, height, _seq, _pts, pixels = decode_frame_payload(body)
                        assert width == widths[stream_id]
                        assert len(pixels) == width * height * (3 if roles[stream_id] == "main" else 1)
                        entry["frames"] += 1
                    elif kind == TYPE_DETECTIONS:
                        snapshot = DetectionSnapshot.parse(decode_json_payload(body), session=stream_id)
                        assert roles[stream_id] == "live" and snapshot.objects
                        assert snapshot.objects[0]["label"] == "car"
                        entry["snapshots"] += 1
                    elif kind == TYPE_STATUS:
                        status = decode_json_payload(body)
                        assert status.get("ok"), status
                        entry["ok"] = True
                if all(e["frames"] >= 8 and e["ok"] for e in counts.values()) and all(
                    counts[key]["snapshots"] >= 4 for key, role in roles.items() if role == "live"
                ):
                    break
        finally:
            process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()
        stderr.seek(0)
        errors = stderr.read().decode(errors="replace")
    assert all(e["frames"] >= 8 and e["ok"] for e in counts.values()), (counts, errors[-4000:])
    assert counts["gate-live"]["snapshots"] >= 4 and counts["yard-live"]["snapshots"] >= 4, counts
    assert counts["gate-main"]["snapshots"] == 0
    return {"shared_supervisor": counts}


def sparse_metadata_preroll() -> dict:
    """An inference branch without a result must not block qualifier video."""
    from survng.dlstreamer_live import _load_gstreamer
    gst = _load_gstreamer()
    observed = {}
    for asynchronous in (True, False):
        pipeline = gst.parse_launch(
            "videotestsrc is-live=true ! tee name=t "
            "t. ! queue ! appsink name=frames sync=false max-buffers=1 drop=true "
            "t. ! queue ! valve drop=true ! appsink sync=false async="
            + str(asynchronous).lower()
        )
        try:
            pipeline.set_state(gst.State.PLAYING)
            frame = pipeline.get_by_name("frames").emit("try-pull-sample", gst.SECOND)
            observed[str(asynchronous)] = frame is not None
        finally:
            pipeline.set_state(gst.State.NULL)
    assert observed == {"True": False, "False": True}, observed
    return {"sparse_metadata_preroll": observed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-device", choices=("CPU", "GPU"), default="CPU")
    args = parser.parse_args()
    import numpy as np
    import openvino as ov
    from openvino import opset13 as ops

    with tempfile.TemporaryDirectory(prefix="survng-gst-smoke-") as directory:
        model_path = Path(directory) / "fixture.xml"
        proc_path = Path(directory) / "fixture.json"
        model_path.with_suffix(".txt").write_text("background\ncar\n")
        image = ops.parameter([1, 3, 64, 64], np.float32, name="image")
        # Preserve a real input-to-output path, with a deterministic SSD result.
        zero = ops.multiply(ops.reduce_mean(image, ops.constant([0, 1, 2, 3]), False), ops.constant(0, np.float32))
        boxes = ops.constant(np.array([[[[0, 1, .9, .1, .1, .6, .8], [-1, 0, 0, 0, 0, 0, 0]]]], dtype=np.float32))
        output = ops.add(boxes, zero, name="detection_out")
        output.output(0).get_tensor().set_names({"detection_out"})
        ov.save_model(ov.Model([output], [image]), model_path, compress_to_fp16=False)
        proc_path.write_text(json.dumps({
            "json_schema_version": "2.2.0",
            "input_preproc": [{"layer_name": "image", "format": "image"}],
            "output_postproc": [{"layer_name": "detection_out", "converter": "detection_output",
                                 "labels": ["background", "person"]}],
        }))
        results = [sparse_metadata_preroll()]
        results.extend(consume(model_path, proc_path, threshold) for threshold in (0.1, 1.0))
        results.append(consume(model_path, proc_path, .1, source_role="main"))
        results.append(consume(model_path, proc_path, .1, detect=False))
        results.append(shared_supervisor(model_path, proc_path, device=args.shared_device))
        # Two overlapping same-class raw YOLO boxes: native NMS must change
        # actual output, not merely accept a command-line/configuration value.
        raw = np.zeros((1, 5, 16), dtype=np.float32)
        raw[0, :, 0] = [32, 32, 32, 32, .9]
        raw[0, :, 1] = [34, 32, 32, 32, .8]
        output = ops.add(ops.constant(raw), zero, name="raw_boxes")
        output.output(0).get_tensor().set_names({"raw_boxes"})
        model = ov.Model([output], [image])
        model.set_rt_info("yolo_v8", ["model_info", "model_type"])
        model.set_rt_info("0.7", ["model_info", "iou_threshold"])
        raw_path = Path(directory) / "raw.xml"
        ov.save_model(model, raw_path, compress_to_fp16=False)
        raw_path.with_suffix(".txt").write_text("car\n")
        raw_proc = Path(directory) / "raw.json"
        raw_proc.write_text(json.dumps({
            "json_schema_version": "2.2.0", "input_preproc": [{"layer_name": "image", "format": "image"}],
            "output_postproc": [{"layer_name": "raw_boxes", "converter": "yolo_v8", "iou_threshold": .7}],
        }))
        for policy in (None, raw_proc):
            for iou, expected in ((.1, 1), (.95, 2)):
                results.append(consume(raw_path, policy, .1, nms_threshold=iou, expected_objects=expected))
        # YOLO26 already emits final boxes. Changing external NMS policy must
        # not suppress either result or lose adjacent exporter metadata.
        final = np.zeros((1, 300, 6), dtype=np.float32)
        final[0, 0] = [16, 16, 48, 48, .9, 0]
        final[0, 1] = [18, 16, 50, 48, .8, 0]
        output = ops.add(ops.constant(final), zero, name="final_boxes")
        model = ov.Model([output], [image])
        model.set_rt_info("YOLO", ["model_info", "model_type"])
        final_dir = Path(directory) / "end_to_end"
        final_dir.mkdir()
        final_path = final_dir / "final.xml"
        ov.save_model(model, final_path, compress_to_fp16=False)
        (final_dir / "metadata.yaml").write_text("description: YOLO26 synthetic fixture\ntask: detect\n")
        final_path.with_suffix(".txt").write_text("car\n")
        for iou in (.1, .95):
            results.append(consume(final_path, None, .1, nms_threshold=iou, expected_objects=2))
        print(json.dumps({"native_gstreamer_smoke": results}, indent=2))


if __name__ == "__main__":
    main()
