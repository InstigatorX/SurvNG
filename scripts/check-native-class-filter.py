"""Exercise class filtering through real gvatrack; run with /usr/bin/python3.

Synthetic frames only; no cameras, model inference, or service changes.
"""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survng.dlstreamer_live import _apply_dlstreamer_env, _filter_tracking_regions, _load_gstreamer

previous = os.environ.get("LD_LIBRARY_PATH", "")
_apply_dlstreamer_env()
if os.environ.get("LD_LIBRARY_PATH", "") != previous:
    os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve())])
Gst = _load_gstreamer()
import gi
gi.require_version("GstAnalytics", "1.0")
from gi.repository import GLib, GstAnalytics
from gstgva import VideoFrame
from gstgva.util import GST_PAD_PROBE_INFO_BUFFER


def check(selection):
    pipeline = Gst.parse_launch(
        "videotestsrc num-buffers=3 ! video/x-raw,format=BGR,width=320,height=240,framerate=5/1 "
        "! identity name=inject ! gvatrack tracking-type=short-term-imageless "
        "! gvametaconvert format=json add-empty-results=true ! appsink name=sink sync=false")
    errors = []
    expected = {"person": 2, "face": 4} if selection is None else {label: index for label, index in {"person": 2, "face": 4}.items() if label in selection}

    def probe(pad, info):
        try:
            with GST_PAD_PROBE_INFO_BUFFER(info) as buffer:
                caps = pad.get_current_caps()
                frame = VideoFrame(buffer, caps=caps)
                for label, box, index in [("person", (10, 20, 100, 150), 2), ("face", (20, 20, 30, 30), 4)]:
                    roi = frame.add_region(*box, label, .9)
                    relation = GstAnalytics.buffer_get_analytics_relation_meta(buffer)
                    labels, scores = [0] * 5, [0.0] * 5
                    labels[index], scores[index] = GLib.quark_from_string(label), .9
                    ok, cls = relation.add_cls_mtd(scores, labels)
                    assert ok and relation.set_relation(GstAnalytics.RelTypes.RELATE_TO, roi.meta().id, cls.id)
                memory = [hash(buffer.peek_memory(i)) for i in range(buffer.n_memory())]
                _filter_tracking_regions(buffer, caps, VideoFrame, selection)
                assert memory == [hash(buffer.peek_memory(i)) for i in range(buffer.n_memory())]
                assert {r.label(): r.label_id() for r in frame.regions()} == expected
        except Exception as exc:
            errors.append(repr(exc))
        return Gst.PadProbeReturn.OK

    pipeline.get_by_name("inject").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe)
    pipeline.set_state(Gst.State.PLAYING)
    identities = []
    try:
        for _ in range(3):
            sample = pipeline.get_by_name("sink").emit("try-pull-sample", 10 * Gst.SECOND)
            assert sample is not None, errors
            frame = VideoFrame(sample.get_buffer(), caps=sample.get_caps())
            regions = list(frame.regions())
            assert {r.label(): r.label_id() for r in regions} == expected, errors
            assert all(r.object_id() > 0 for r in regions)
            identities.append({r.label(): r.object_id() for r in regions})
            assert len(frame.messages()) == 1
        assert identities[0] == identities[1] == identities[2]
        assert not errors, errors
        print(f"Class filter passed: {selection}; class IDs and tracking IDs retained, pixel memory unchanged")
    finally:
        pipeline.set_state(Gst.State.NULL)


for selection in [None, frozenset(), frozenset(["person"]), frozenset(["face"]), frozenset(["person", "face"])]:
    check(selection)
tracker = Gst.ElementFactory.make("gvatrack")
print("Installed tracking modes:", [value.value_nick for value in tracker.find_property("tracking-type").enum_class.__enum_values__.values()])
