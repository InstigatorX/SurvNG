"""GStreamer appsink worker for the experimental capture backend.

Runs on system Python, which has the GObject bindings. The venv does not.
The parent owns reconnects. This process writes one bounded BGR frame at a
time and exits on end-of-stream or a pipeline error.
"""

from __future__ import annotations

import queue
import sys
import threading

# In-band geometry. A side-channel caps line can race the next buffer.
FRAME_MAGIC = b"SVNG"


def _write_frames(stdout, pending: queue.Queue[bytes | None]) -> None:
    while True:
        packet = pending.get()
        if packet is None:
            return
        stdout.write(packet)
        stdout.flush()


def _publish(pending: queue.Queue[bytes | None], packet: bytes) -> None:
    """Keep at most one packed frame waiting while stdout is blocked."""
    try:
        pending.put_nowait(packet)
        return
    except queue.Full:
        pass
    try:
        pending.get_nowait()
    except queue.Empty:
        pass
    try:
        pending.put_nowait(packet)
    except queue.Full:
        pass


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: gstreamer_capture_worker.py LOCATION MAX_RATE TRANSPORT", file=sys.stderr)
        return 2
    location, rate_text, transport = argv[1], argv[2], argv[3]
    try:
        max_rate = max(1, min(10, int(rate_text)))
    except ValueError:
        print("max rate must be an integer", file=sys.stderr)
        return 2
    if transport not in {"tcp", "udp"}:
        print("transport must be tcp or udp", file=sys.stderr)
        return 2
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except Exception as error:
        print(f"gstreamer bindings are unavailable: {error}", file=sys.stderr)
        return 127

    Gst.init(None)
    pipeline = Gst.Pipeline.new("survng-gstreamer-capture")
    decode = Gst.ElementFactory.make("decodebin", "decode")
    rate = Gst.ElementFactory.make("videorate", "rate")
    convert = Gst.ElementFactory.make("videoconvert", "convert")
    capsfilter = Gst.ElementFactory.make("capsfilter", "caps")
    sink = Gst.ElementFactory.make("appsink", "sink")
    if location.startswith(("rtsp://", "rtsps://")):
        source = Gst.ElementFactory.make("rtspsrc", "source")
        # tcp is 0x4 and udp is 0x1 in GstRTSPLowerTrans. Set the integer so
        # this worker does not import the RTSP typelib.
        source.set_property("location", location)
        source.set_property("protocols", 4 if transport == "tcp" else 1)
        source.set_property("latency", 100)
        source.set_property("drop-on-latency", True)
    else:
        path = location[7:] if location.startswith("file://") else location
        source = Gst.ElementFactory.make("filesrc", "source")
        source.set_property("location", path)
    elements = (source, decode, rate, convert, capsfilter, sink)
    if any(element is None for element in elements):
        print("gstreamer capture elements are unavailable", file=sys.stderr)
        return 127
    for element in elements:
        pipeline.add(element)
    rate.set_property("drop-only", True)
    rate.set_property("max-rate", max_rate)
    capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGR"))
    sink.set_property("max-buffers", 1)
    sink.set_property("drop", True)
    sink.set_property("sync", False)
    sink.set_property("enable-last-sample", False)

    if location.startswith(("rtsp://", "rtsps://")):
        def _link_rtp(_element, pad) -> None:
            caps = pad.get_current_caps()
            if caps is not None and caps.get_size() > 0:
                media = caps.get_structure(0).get_string("media")
                if media not in {None, "video"}:
                    return
            sinkpad = decode.get_static_pad("sink")
            if sinkpad is not None and not sinkpad.is_linked():
                pad.link(sinkpad)

        source.connect("pad-added", _link_rtp)
    elif not source.link(decode):
        print("could not link the file source", file=sys.stderr)
        return 1

    def _link_video(_element, pad) -> None:
        caps = pad.get_current_caps()
        if caps is None or caps.get_size() == 0:
            return
        if not caps.get_structure(0).get_name().startswith("video/"):
            return
        sinkpad = rate.get_static_pad("sink")
        if sinkpad is not None and not sinkpad.is_linked():
            pad.link(sinkpad)

    decode.connect("pad-added", _link_video)
    if not rate.link(convert) or not convert.link(capsfilter) or not capsfilter.link(sink):
        print("could not link the bgr delivery chain", file=sys.stderr)
        return 1

    pending: queue.Queue[bytes | None] = queue.Queue(maxsize=1)
    writer = threading.Thread(
        target=_write_frames,
        args=(sys.stdout.buffer, pending),
        name="gstreamer-capture-writer",
        daemon=True,
    )
    writer.start()
    bus = pipeline.get_bus()
    pipeline.set_state(Gst.State.PLAYING)
    exit_code = 0
    try:
        while True:
            message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if message is not None:
                if message.type == Gst.MessageType.ERROR:
                    error, _debug = message.parse_error()
                    print(error.message, file=sys.stderr)
                    exit_code = 1
                break
            sample = sink.emit("try-pull-sample", 200 * Gst.MSECOND)
            if sample is None:
                continue
            buffer = sample.get_buffer()
            sample_caps = sample.get_caps()
            if buffer is None or sample_caps is None or sample_caps.get_size() == 0:
                continue
            structure = sample_caps.get_structure(0)
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))
            expected = width * height * 3
            if width < 1 or height < 1 or expected < 3:
                continue
            mapped, info = buffer.map(Gst.MapFlags.READ)
            if not mapped:
                continue
            try:
                if info.size < expected:
                    print("bgr buffer is shorter than its caps", file=sys.stderr)
                    exit_code = 1
                    break
                packet = FRAME_MAGIC + width.to_bytes(4, "little") + height.to_bytes(4, "little")
                packet += bytes(info.data[:expected])
            finally:
                buffer.unmap(info)
            _publish(pending, packet)
    finally:
        pipeline.set_state(Gst.State.NULL)
        pending.put(None)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
