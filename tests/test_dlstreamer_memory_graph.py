"""Exercise graph construction without GI; real VA negotiation needs a GPU."""

from fractions import Fraction
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from survng import dlstreamer_live as live


@pytest.mark.parametrize("decoder,detect,role,test_source", [
    ("va", True, "live", False),
    ("auto", True, "live", False),
    ("va", False, "main", False),
    ("va", False, "live", False),
    ("va", True, "live", True),
])
def test_host_consumers_download_after_rate_limit_without_breaking_detection(
    monkeypatch, decoder, detect, role, test_source,
):
    elements, links = {}, []

    class Element:
        def __init__(self, factory, name):
            self.factory, self.name, self.properties = factory, name, {}
            elements[name] = self

        def set_property(self, name, value):
            self.properties[name] = value

        def connect(self, *args):
            pass

        def link(self, other):
            links.append((self.name, other.name))
            return True

        def get_name(self):
            return self.name

    pipeline = SimpleNamespace(
        add=lambda element: None, connect=lambda *args: None,
        get_bus=lambda: None, set_state=lambda state: 0,
        get_by_name=lambda name: elements.get(name),
    )
    gst = SimpleNamespace(
        Pipeline=SimpleNamespace(new=lambda name: pipeline),
        Caps=SimpleNamespace(from_string=lambda text: text),
        State=SimpleNamespace(PLAYING=1, NULL=0),
        StateChangeReturn=SimpleNamespace(FAILURE=-1),
    )
    monkeypatch.setitem(sys.modules, "gstgva", SimpleNamespace(VideoFrame=object))
    monkeypatch.setattr(live, "_element", lambda gst, factory, name: Element(factory, name))
    monkeypatch.setattr(live, "_factory_available", lambda gst, name: True)
    monkeypatch.setattr(live, "_make_live_source", lambda gst, **kwargs: (Element("source", "source"), "source"))
    monkeypatch.setattr(live, "_link_tee", lambda gst, tee, target: tee.link(target))
    stop = threading.Event()
    stop.set()  # Construct/link the real graph, but do not enter its native loop.
    args = live._parser().parse_args(["--decoder", decoder])
    live._pump_pipeline(
        gst, args, url="rtsp://fixture.invalid/video", stream_id="test",
        detect=detect, model_path=Path("fixture.xml"), instance_id="fixture",
        rate=Fraction(5), detect_rate=Fraction(5, 2), qualifier_width=320,
        jpeg_rate=Fraction(1), open_timeout=3, stdout=None, stdout_lock=None,
        stop_event=stop, encode_frame=None, encode_jpeg=None, encode_json=None,
        TYPE_DETECTIONS=2, TYPE_STATUS=3, install_signals=False,
        test_source=test_source, source_role=role,
    )
    va = decoder == "va" and detect and not test_source
    expected = "vapostproc" if va else "videoconvert"
    assert elements["qualifier-gray"].factory == "videoconvert"
    assert elements["jpeg-convert"].factory == expected
    assert ("drop-only-rate", "qualifier-download" if va else "qualifier-gray") in links
    assert ("jpeg-rate", "jpeg-convert") in links
    assert elements["drop-only-rate"].properties["drop-only"]
    assert elements["jpeg-rate"].properties["drop-only"]
    assert elements["frame-queue"].properties["max-size-buffers"] == 1
    assert elements["frame-caps"].properties["caps"].startswith("video/x-raw,format=")
    assert elements["jpeg-caps"].properties["caps"] == "video/x-raw,format=I420,framerate=1/1"
    if va:
        assert elements["qualifier-download"].factory == "vapostproc"
        assert elements["qualifier-host-caps"].properties["caps"] == "video/x-raw,format=NV12,width=320,pixel-aspect-ratio=1/1"
        assert ("qualifier-download", "qualifier-host-caps") in links
        assert ("qualifier-host-caps", "qualifier-gray") in links
        assert "qualifier-scale" not in elements  # resize before the host download
        assert ("qualifier-gray", "frame-caps") in links
        assert "width=320" in elements["frame-caps"].properties["caps"]
        assert elements["detect"].properties["pre-process-backend"] == "va-surface-sharing"
        assert elements["detect"].properties["pre-process-config"] == "VAAPI_THREAD_POOL_SIZE=1"
        assert elements["detect-rate-caps"].properties["caps"] == "video/x-raw(memory:VAMemory),framerate=5/2"
        assert ("detect-va-memory", "detect") in links
    else:
        assert elements["qualifier-scale"].factory == "videoscale"
    if not detect:
        assert "detect" not in elements
    else:
        assert elements["detect"].properties["ie-config"] == "PERFORMANCE_HINT=LATENCY,NUM_STREAMS=1"
        assert elements["detect"].properties["nireq"] == 1
        assert elements["detect"].properties["scheduling-policy"] == "throughput"


def test_negotiated_memory_uses_actual_pad_caps():
    caps = SimpleNamespace(get_size=lambda: 1,
                           get_features=lambda index: SimpleNamespace(to_string=lambda: "memory:VAMemory"))
    element = SimpleNamespace(get_static_pad=lambda name: SimpleNamespace(get_current_caps=lambda: caps))
    assert live._negotiated_memory(element) == "memory:VAMemory"
    assert live._negotiated_memory(None) is None
