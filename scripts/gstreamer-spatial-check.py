"""Bounded real native spatial integration check; synthetic model, no cameras.

Run with the installed DL Streamer Python. Validates metadata and coordinates,
not recognition accuracy or throughput.
"""
from pathlib import Path
import json
import os
import select
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survng.dlstreamer_live import _apply_dlstreamer_env
_apply_dlstreamer_env()
if not os.environ.get("SURVNG_SPATIAL_CHECK_READY"):
    os.environ["SURVNG_SPATIAL_CHECK_READY"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

from survng.app.dlstreamer_protocol import MessageReader, TYPE_DETECTIONS, TYPE_STATUS, decode_json_payload, decode_stream_payload


def check(model, proc, *, interval=1, batch=1, threshold=.1, budget=None, idle_objects=False, motion_wake=False):
    plan = {"revision": "spatial-check", "zones": [
        {"name": "right", "behavior": "incident", "enabled": True, "points": [
            {"x": .5, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}, {"x": .5, "y": 1}]}],
        "roi": {"enabled": True, "padding": 0, "full_frame_interval": 3}}
    if budget is not None:
        plan["budget"] = budget
    if idle_objects:
        plan["zones"][0]["object_classes"] = ["person"]
    cmd = [sys.executable, '-m', 'survng.dlstreamer_live', '--supervisor', '--test-source',
           '--device', 'CPU', '--decoder', 'auto', '--model', str(model), '--model-proc', str(proc),
           '--labels', str(model.with_suffix('.txt')), '--native-tracking', 'short-term-imageless',
           '--fps', '5', '--detect-fps', '5', '--jpeg-fps', '0', '--inference-interval', str(interval),
           '--batch-size', str(batch), '--threshold', str(threshold), '--nms-threshold', '.45']
    snapshots = []
    other_snapshots = []
    budget_status = {}
    reader = MessageReader()
    with tempfile.TemporaryFile() as err:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err)
        try:
            process.stdin.write((json.dumps({"op": "add", "stream_id": "roi", "url": "rtsp://unused.invalid/live", "spatial_plan": plan})+'\n').encode())
            process.stdin.write((json.dumps({"op": "add", "stream_id": "full", "url": "rtsp://unused.invalid/live", "spatial_plan": {"revision": "full-check", "zones": [], "roi": {"enabled": False}}})+'\n').encode())
            process.stdin.flush()
            deadline = time.monotonic()+35
            while time.monotonic() < deadline and (len(snapshots) < 18 or len(other_snapshots) < 18):
                if not select.select([process.stdout], [], [], .2)[0]:
                    continue
                chunk = process.stdout.read1(65536)
                if not chunk:
                    break
                reader.feed(chunk)
                while (message := reader.pop()) is not None:
                    kind, raw = message
                    if kind not in (TYPE_DETECTIONS, TYPE_STATUS):
                        continue
                    stream_id, raw = decode_stream_payload(raw)
                    payload = decode_json_payload(raw)
                    if kind == TYPE_STATUS:
                        assert payload.get('ok'), payload
                        if stream_id == 'roi':
                            budget_status = payload.get('native_budget', {})
                        if payload.get('native_evidence_invalid', 0):
                            err.seek(0); raise AssertionError(err.read().decode(errors='replace')[-5000:])
                    else:
                        (snapshots if stream_id == "roi" else other_snapshots).append(payload)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
            err.seek(0)
            diagnostics = err.read().decode(errors='replace')
    assert len(snapshots) >= 12, (len(snapshots), diagnostics[-4000:])
    assert len(other_snapshots) >= 12, diagnostics[-4000:]
    assert all(s['zone_revision'] == 'full-check' for s in other_snapshots)
    assert all(o['native_zone_ids'] == [] for s in other_snapshots for o in s['objects'])
    assert all(abs(o['box']['x1'] / s['width'] - .1) < .01 for s in other_snapshots for o in s['objects'])
    assert 'GStreamer-CRITICAL' not in diagnostics, diagnostics[-4000:]
    fresh = [s for s in snapshots if s['provenance'] == 'native_fresh_detection']
    assert len(fresh) >= 4, snapshots
    assert all(s['zone_revision'] == plan['revision'] for s in snapshots)
    assert all(o['label'] != '__survng_inference_region__' for s in snapshots for o in s['objects'])
    if threshold == 1:
        assert all(not s['objects'] for s in snapshots), snapshots
    else:
        found = set()
        for s in fresh:
            objs = [o for o in s['objects'] if o['detection_provenance'] == 'native_fresh_detection']
            assert len(objs) == 1, s
            obj = objs[0]
            assert obj['label'] == 'car', obj
            assert isinstance(obj.get('native_track_id'), int), obj
            relative_x = obj['box']['x1'] / s['width']
            if abs(relative_x - .1) < .01:
                found.add('full')
                assert obj['native_zone_ids'] == [], obj
            else:
                assert abs(relative_x - .55) < .01, obj
                found.add('crop')
                assert obj['native_zone_ids'] == ['0'], obj
        assert found == {'full', 'crop'}, found
    if budget is not None and not motion_wake:
        assert budget_status.get("skipped_frames", 0) > 10, budget_status
        assert budget_status["mode"] == "idle", budget_status
        assert snapshots[-1]["source_pts"] - snapshots[0]["source_pts"] > 10
    if idle_objects:
        assert all(abs(o['box']['x1']/s['width']-.1) < .01 for s in snapshots[-8:] for o in s['objects'])
    if motion_wake:
        assert budget_status.get('motion_wakes', 0) > 0, budget_status
        assert budget_status['mode'] == 'active', budget_status
    if interval > 1 and budget is None:
        assert any(s['provenance'] == 'native_tracked_prediction' for s in snapshots)
    return {"interval": interval, "batch": batch, "threshold": threshold, "snapshots": len(snapshots), "fresh": len(fresh)}


def check_va(model):
    from survng.dlstreamer_live import (_load_gstreamer, _create_shared_va_context,
                                      _filter_tracking_regions, _NativeInferenceEvidence, _detection_metadata)
    from survng.dlstreamer_model import configured_model
    from gstgva import VideoFrame
    from gstgva.util import GST_PAD_PROBE_INFO_BUFFER
    Gst = _load_gstreamer()
    plan = {"revision": "va-check", "zones": [], "roi": {"enabled": True, "padding": 0, "full_frame_interval": 3}}
    plan['zones'] = [{"name": "right", "behavior": "incident", "points": [
        {"x": .5, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}, {"x": .5, "y": 1}]}]
    from survng.native_spatial import analytics_zones
    evidence = _NativeInferenceEvidence(1, tracking=True)
    with configured_model(model, "", .45) as (configured, _):
        graph = ('videotestsrc num-buffers=12 is-live=true ! video/x-raw,format=NV12,width=320,height=240,framerate=5/1 ! '
                 'vapostproc ! video/x-raw(memory:VAMemory),format=NV12 ! gvamotiondetect name=motion ! gvapython name=roi ! '
                 'gvadetect name=detect device=GPU pre-process-backend=va-surface-sharing inference-region=roi-list '
                 'batch-size=1 nireq=2 ie-config="PERFORMANCE_HINT=THROUGHPUT,ALLOW_AUTO_BATCHING=NO" ! '
                 'gvatrack tracking-type=short-term-imageless ! gvaanalytics name=zones evaluation-point=bottom-center '
                 'draw-zones=false draw-tripwires=false ! gvametaconvert add-empty-results=true ! appsink name=out sync=false')
        pipeline = Gst.parse_launch(graph)
        pipeline.set_context(_create_shared_va_context(Gst))
        roi = pipeline.get_by_name('roi')
        roi.set_property('module', str(Path(__file__).resolve().parents[1]/'survng/native_spatial.py'))
        roi.set_property('class', 'RoiInput')
        roi.set_property('kwarg', json.dumps({'plan': plan, 'interval': 1}))
        detector = pipeline.get_by_name('detect')
        detector.set_property('model', str(configured))
        detector.set_property('labels-file', str(model.with_suffix('.txt')))
        pipeline.get_by_name('zones').set_property('zones', json.dumps(analytics_zones(plan, 320, 240)))
        failures = []
        def capture(pad, info):
            try:
                with GST_PAD_PROBE_INFO_BUFFER(info) as buffer:
                    _filter_tracking_regions(buffer, pad.get_current_caps(), VideoFrame, None)
                    evidence.observe(buffer, pad.get_current_caps(), VideoFrame)
            except Exception as exc:
                failures.append(str(exc))
                return Gst.PadProbeReturn.DROP
            return Gst.PadProbeReturn.OK
        def clear_motion(pad, info):
            with GST_PAD_PROBE_INFO_BUFFER(info) as buffer:
                _filter_tracking_regions(buffer, pad.get_current_caps(), VideoFrame, frozenset())
            return Gst.PadProbeReturn.OK
        pipeline.get_by_name('motion').get_static_pad('src').add_probe(Gst.PadProbeType.BUFFER, clear_motion)
        detector.get_static_pad('src').add_probe(Gst.PadProbeType.BUFFER, capture)
        pipeline.set_state(Gst.State.PLAYING)
        results = []
        try:
            for index in range(12):
                sample = pipeline.get_by_name('out').emit('try-pull-sample', 20*Gst.SECOND)
                if sample is None:
                    error = pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
                    raise AssertionError(str(error.parse_error()) if error else failures)
                assert sample.get_caps().get_features(0).contains('memory:VAMemory')
                payload = _detection_metadata(sample, VideoFrame, inference_sequence=index+1, gst_second=Gst.SECOND,
                    clock_time_none=Gst.CLOCK_TIME_NONE, native_result=evidence.pop(sample.get_buffer().pts), spatial_plan=plan)
                obj = next(o for o in payload['objects'] if o['detection_provenance'] == 'native_fresh_detection')
                expected = .1 if index % 3 == 0 else .55
                assert abs(obj['box']['x1']/320 - expected) < .01, payload
                assert obj['native_zone_ids'] == ([] if index % 3 == 0 else ['0']), payload
                results.append(obj)
            assert not failures and evidence.invalid == 0
        finally:
            pipeline.set_state(Gst.State.NULL)
    return {"va_surface_sharing": "passed", "fresh_results": len(results)}


def check_verifier_crops(model):
    import numpy as np
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = root/'config.json'
        config.write_text(json.dumps({'model':str(model),'model_proc':'','nms':.45,'threshold':.1,
                                      'labels_path':str(model.with_suffix('.txt')),'labels':[]}))
        process = subprocess.Popen([sys.executable,'-m','survng.native_evidence_verify',str(config)],
                                   stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
        try:
            for width,height in ((193,191),(287,239),(640,480)):
                pixels, result = root/'frame.bgr', root/'result.json'
                result.unlink(missing_ok=True)
                pixels.write_bytes(np.zeros((height,width,3),np.uint8).tobytes())
                process.stdin.write(json.dumps({'width':width,'height':height,'pixels':str(pixels),'result':str(result)})+'\n')
                process.stdin.flush()
                deadline=time.monotonic()+40
                while not result.exists() and time.monotonic()<deadline and process.poll() is None:
                    time.sleep(.05)
                assert result.exists(), 'verifier did not return crop result'
                payload=json.loads(result.read_text())
                assert 'error' not in payload, payload
                assert len(payload['objects'])==1, payload
                obj=payload['objects'][0]
                assert obj['label']=='car' and abs(obj['box']['x1']/width-.1)<.01, payload
        finally:
            process.stdin.close()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
            diagnostics=process.stderr.read(); process.stderr.close()
        assert 'GStreamer-CRITICAL' not in diagnostics, diagnostics
    return {'odd_width_native_crops': 'passed', 'samples':3}


def geometry_samples():
    from survng.dlstreamer_live import _load_gstreamer, _normalize_gva_objects
    from survng.native_spatial import analytics_zones
    from gstgva import VideoFrame
    Gst = _load_gstreamer()
    zones = [{"name": "polygon", "points": [{"x": .203, "y": .107}, {"x": .893, "y": .297},
              {"x": .603, "y": .497}, {"x": .823, "y": .897}, {"x": .193, "y": .797}]}]
    records = []
    for width, height in ((100, 100), (101, 77), (640, 480)):
        caps = Gst.Caps.from_string(f"video/x-raw,format=BGR,width={width},height={height},framerate=5/1")
        pipeline = Gst.parse_launch('appsrc name=input format=time ! gvaanalytics name=zones evaluation-point=bottom-center draw-zones=false draw-tripwires=false ! gvametaconvert add-empty-results=true ! appsink name=out sync=false')
        pipeline.get_by_name('input').set_property('caps', caps)
        pipeline.get_by_name('zones').set_property('zones', json.dumps(analytics_zones({"zones": zones}, width, height)))
        pipeline.set_state(Gst.State.PLAYING)
        try:
            for i in range(80):
                x = (i * 37) % (width - 12)
                y = 5 + (i * 23) % (height - 10)
                buffer = Gst.Buffer.new_allocate(None, width*height*3, None)
                buffer.pts = i * Gst.SECOND // 5
                frame = VideoFrame(buffer, caps=caps)
                frame.add_region(x, y-5, 11, 5, "person", .9)
                pipeline.get_by_name('input').emit('push-buffer', buffer)
                sample = pipeline.get_by_name('out').emit('try-pull-sample', 3*Gst.SECOND)
                assert sample is not None
                payload = json.loads(VideoFrame(sample.get_buffer(), caps=sample.get_caps()).messages()[0])
                obj = _normalize_gva_objects(payload)[0]
                obj.setdefault('native_zone_ids', [])
                records.append({"width": width, "height": height, "object": obj})
        finally:
            pipeline.set_state(Gst.State.NULL)
    return {"zones": zones, "records": records}


def main():
    import numpy as np
    import openvino as ov
    from openvino import opset13 as ops
    with tempfile.TemporaryDirectory(prefix='survng-spatial-') as directory:
        model = Path(directory)/'fixture.xml'
        proc = model.with_suffix('.json')
        image = ops.parameter([1, 3, 64, 64], np.float32, name='image')
        zero = ops.reshape(ops.multiply(ops.reduce_mean(image, ops.constant([1, 2, 3]), False), ops.constant(0, np.float32)), [-1, 1, 1], False)
        boxes = np.zeros((1, 300, 6), dtype=np.float32)
        boxes[0, 0] = [6.4, 6.4, 38.4, 51.2, .9, 0]
        output = ops.add(ops.constant(boxes), zero, name='final_boxes')
        fixture = ov.Model([output], [image])
        fixture.set_rt_info("YOLO", ["model_info", "model_type"])
        ov.save_model(fixture, model, compress_to_fp16=False)
        model.with_suffix('.txt').write_text('car\n')
        (model.parent / "metadata.yaml").write_text("description: YOLO26 synthetic fixture\ntask: detect\n")
        proc = ""
        if '--verify-crops' in sys.argv:
            print(json.dumps(check_verifier_crops(model)))
            return
        if '--va' in sys.argv:
            print(json.dumps(check_va(model)))
            return
        if '--budget' in sys.argv:
            policy = dict(enabled=True, idle_fps=1, active_fps=5, cooldown_seconds=.4, approach_padding=.1,
                          motion_enabled=False, block_size=64, motion_threshold=1., min_persistence=2,
                          max_miss=1, iou_threshold=.3, smooth_alpha=.5, confirm_frames=1,
                          pixel_diff_threshold=255, min_rel_area=.0005)
            results = [check(model, proc, interval=3, threshold=1, budget=policy)]
            results.append(check(model, proc, budget=policy, idle_objects=True))
            policy['motion_enabled'] = True
            results.append(check(model, proc, threshold=1, budget=policy))
            policy.update(motion_threshold=.001, pixel_diff_threshold=1, min_persistence=1, min_rel_area=0)
            results.append(check(model, proc, threshold=1, budget=policy, motion_wake=True))
            print(json.dumps(results))
            return
        results = [check(model, proc), check(model, proc, interval=3), check(model, proc, batch=2), check(model, proc, threshold=1)]
        print(json.dumps(results))

if __name__ == '__main__':
    if '--geometry' in sys.argv:
        print(json.dumps(geometry_samples()))
    else:
        main()
