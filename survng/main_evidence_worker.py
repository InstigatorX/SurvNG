"""Isolated encoded main-stream collector and bounded, on-demand GI decoder.

Camera URLs arrive on stdin, never in process arguments. The application owns
process admission; this helper never starts inference or an archive writer.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import socket
import sys
import threading
import time

from survng.app.main_evidence_ring import AccessUnit, EncodedRing, receive_control, send_control


def _gst():
    from survng.dlstreamer_live import _load_gstreamer
    return _load_gstreamer()


def _element(Gst, name, label):
    element = Gst.ElementFactory.make(name, label)
    if element is None:
        raise RuntimeError("main evidence plugin unavailable: " + name)
    return element


def _seal(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    fcntl.fcntl(fd, fcntl.F_ADD_SEALS,
                fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)


def _write(fd, data):
    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        view = view[count:]


class Collector:
    def __init__(self, Gst, config):
        self.Gst = Gst
        # Reserve thirds for current ring, a concurrently evicted pinned prefix,
        # and its immutable export. Copying cannot evade the encoded quota.
        self.ring = EncodedRing(max_bytes=int(config["max_bytes"]) // 3,
                                history_seconds=float(config["history_seconds"]))
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.caps = ""
        self.codec = ""
        self.width = self.height = 0
        self._ordinal = 0
        self._anchor = None
        self._monotonic_anchor = None
        self._parameter_sets = {}
        self.failure = ""
        self.warning_count = 0
        self.exported = None
        self.export_bytes = 0
        self.uncertainty = float(config["timestamp_uncertainty_seconds"])
        self.pipeline = Gst.Pipeline.new("survng-main-evidence")
        source = _element(Gst, "rtspsrc", "source")
        source.set_property("location", config["source_url"])
        source.set_property("protocols", 4)
        source.set_property("latency", 250)
        self.pipeline.add(source)
        self.sink = None

        def added(_source, pad):
            caps = pad.get_current_caps()
            if caps is None or caps.get_size() == 0 or self.sink is not None:
                return
            info = caps.get_structure(0)
            encoding = str(info.get_string("encoding-name") or "").upper()
            if encoding not in {"H264", "H265"}:
                return
            self.codec = encoding.lower()
            depay = _element(Gst, "rtp" + self.codec + "depay", "depay")
            parser = _element(Gst, self.codec + "parse", "parser")
            parser.set_property("config-interval", -1)
            cap = _element(Gst, "capsfilter", "access-units")
            cap.set_property("caps", Gst.Caps.from_string(
                f"video/x-{self.codec},stream-format=byte-stream,alignment=au"))
            sink = _element(Gst, "appsink", "encoded-sink")
            sink.set_property("sync", False)
            sink.set_property("max-buffers", 8)
            sink.set_property("drop", True)
            sink.set_property("enable-last-sample", False)
            chain = [depay, parser, cap, sink]
            for element in chain:
                self.pipeline.add(element)
            for left, right in zip(chain, chain[1:]):
                if not left.link(right):
                    raise RuntimeError("encoded collector link failed")

            def ordinal(_pad, probe):
                buffer = probe.get_buffer()
                if buffer is not None:
                    self._ordinal += 1
                    buffer.offset = self._ordinal
                return Gst.PadProbeReturn.OK

            # Ordinals precede appsink's lossy queue; a drop breaks the GOP.
            cap.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, ordinal)
            for element in chain:
                element.sync_state_with_parent()
            if pad.link(depay.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
                raise RuntimeError("encoded collector source link failed")
            self.sink = sink

        source.connect("pad-added", added)
        self.pipeline.set_state(Gst.State.PLAYING)
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self):
        Gst = self.Gst
        bus = self.pipeline.get_bus()
        try:
            while not self.stop.is_set():
                error = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS | Gst.MessageType.WARNING)
                if error is not None:
                    if error.type == Gst.MessageType.WARNING:
                        self.warning_count += 1
                        with self.lock:
                            self.ring.reset()
                        continue
                    # Raw GStreamer errors can contain the source URL.
                    self.failure = "source_error" if error.type == Gst.MessageType.ERROR else "source_eos"
                    return
                sink = self.sink
                if sink is None:
                    self.stop.wait(0.02)
                    continue
                sample = sink.emit("try-pull-sample", 100 * Gst.MSECOND)
                if sample is None:
                    continue
                buffer = sample.get_buffer()
                if (buffer.get_size() > self.ring.max_au_bytes or buffer.pts == Gst.CLOCK_TIME_NONE
                        or buffer.has_flags(Gst.BufferFlags.CORRUPTED)):
                    with self.lock:
                        self.ring.reset()
                    continue
                data = buffer.extract_dup(0, buffer.get_size())
                caps = sample.get_caps()
                structure = caps.get_structure(0)
                width = int(structure.get_value("width") or 0)
                height = int(structure.get_value("height") or 0)
                nals = [part for part in re.split(b"\x00\x00\x00?\x01", data) if part]
                types = set()
                for nal in nals:
                    kind = nal[0] & 31 if self.codec == "h264" else (nal[0] >> 1) & 63
                    types.add(kind)
                    if kind in ({7, 8} if self.codec == "h264" else {32, 33, 34}):
                        self._parameter_sets[kind] = nal
                required = {7, 8} if self.codec == "h264" else {32, 33, 34}
                idr = bool(types & ({5} if self.codec == "h264" else {19, 20})) and required <= self._parameter_sets.keys()
                if idr:
                    data = b"".join(b"\x00\x00\x00\x01" + self._parameter_sets[k]
                                    for k in sorted(required - types)) + data
                config = hashlib.sha256(caps.to_string().encode() + b"".join(
                    self._parameter_sets[k] for k in sorted(self._parameter_sets))).hexdigest()
                now = time.monotonic()
                pts = int(buffer.pts)
                discontinuity = buffer.has_flags(Gst.BufferFlags.DISCONT)
                if self._anchor is None or discontinuity:
                    self._anchor = time.time() - pts / Gst.SECOND
                    self._monotonic_anchor = now - pts / Gst.SECOND
                elif (abs(now - (self._monotonic_anchor + pts / Gst.SECOND)) > self.uncertainty
                      or abs((time.time() - now) - (self._anchor - self._monotonic_anchor)) > self.uncertainty):
                    # A stalled/rebased stream no longer meets its configured
                    # receive-time envelope. Do not silently reanchor old media.
                    with self.lock:
                        self.ring.reset(config)
                    self.failure = "timestamp_mapping_unreliable"
                    return
                unit = AccessUnit(int(buffer.offset), pts,
                    None if buffer.dts == Gst.CLOCK_TIME_NONE else int(buffer.dts),
                    None if buffer.duration == Gst.CLOCK_TIME_NONE else int(buffer.duration),
                    now, self._anchor + pts / Gst.SECOND, data, idr)
                with self.lock:
                    self.caps, self.width, self.height = caps.to_string(), width, height
                    self.ring.add(unit, config=config, discontinuity=discontinuity)
        except Exception:
            self.failure = "collector_error"

    def request(self, request, sock):
        operation = request.get("op")
        if operation == "release":
            self.exported = None
            self.export_bytes = 0
            send_control(sock, {"status": "released"})
            return
        with self.lock:
            status = self.ring.status()
            status.update({"failure": self.failure, "width": self.width, "height": self.height,
                           "export_bytes": self.export_bytes,
                           "source_warnings": self.warning_count,
                           "timestamp_uncertainty_seconds": self.uncertainty})
            status["timestamp_mapping"] = "uncalibrated_host_receive_estimate"
            if operation == "status":
                send_control(sock, status)
                return
            if operation != "window":
                raise ValueError("unsupported main evidence operation")
            if self.failure:
                send_control(sock, {"status": "miss", "reason": self.failure})
                return
            if self.exported is not None:
                send_control(sock, {"status": "capacity_denied"})
                return
            targets = [float(value) for value in request.get("targets", [])]
            if len(targets) > 32 or not targets:
                raise ValueError("invalid main evidence targets")
            state, units = self.ring.window(targets, max_bytes=self.ring.max_bytes)
            if state != "ready":
                send_control(sock, {"status": state})
                return
            # References are bounded and immutable while bytes are exported.
            caps, codec = self.caps, self.codec
            metadata = {"status": "ready", "session": self.ring.session,
                        "generation": self.ring.generation, "caps": caps, "codec": codec,
                        "width": self.width, "height": self.height,
                        "timestamp_uncertainty_seconds": self.uncertainty,
                        "timestamp_method": "host_receive_anchor", "units": []}
        fd = os.memfd_create("survng-encoded-window", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            offset = 0
            for unit in units:
                _write(fd, unit.data)
                metadata["units"].append({"offset": offset, "size": len(unit.data),
                    "ordinal": unit.ordinal, "pts_ns": unit.pts_ns, "dts_ns": unit.dts_ns,
                    "duration_ns": unit.duration_ns, "epoch": unit.epoch})
                offset += len(unit.data)
            _seal(fd)
            self.exported = time.monotonic()
            self.export_bytes = offset
            send_control(sock, metadata, fd)
        finally:
            os.close(fd)

    def close(self):
        self.stop.set()
        self.pipeline.set_state(self.Gst.State.NULL)
        self.thread.join(timeout=2)


class DecoderCleanupIncomplete(RuntimeError):
    """The helper must exit before another native decoder can be admitted."""


def decode(config, input_fd, output_fd):
    """Decode a fixed prefix. Missing reorder output returns pending, not empty detection."""
    Gst = _gst()
    codec = config["codec"]
    if codec not in {"h264", "h265"}:
        raise ValueError("unsupported codec")
    policy = config.get("decoder", "auto")
    choices = (["va" + codec + "dec"] if policy == "va" else
               ["avdec_" + codec] if policy == "cpu" else ["va" + codec + "dec", "avdec_" + codec])
    deadline = time.monotonic() + min(15.0, max(0.1, float(config["timeout_seconds"])))
    result = {"status": "unsupported", "reason": "decoder_unavailable"}
    for chosen in choices:
        if not Gst.ElementFactory.find(chosen):
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        # A failed decoder may already have written selected pixels. The next
        # backend must own a clean output under the same quota and deadline.
        os.ftruncate(output_fd, 0)
        os.lseek(output_fd, 0, os.SEEK_SET)
        try:
            result = _decode_with_decoder(Gst, {**config, "timeout_seconds": remaining},
                                          input_fd, output_fd, chosen)
        except DecoderCleanupIncomplete:
            return {"status": "miss", "reason": "decoder_cleanup_incomplete"}
        except Exception:
            result = {"status": "miss", "reason": "decode_error"}
        if result.get("reason") != "decode_error":
            return result
    return result


def _decode_with_decoder(Gst, config, input_fd, output_fd, chosen):
    deadline = time.monotonic() + max(0.0, float(config["timeout_seconds"]))
    codec = config["codec"]
    pipeline = Gst.parse_launch(
        f"appsrc name=input format=time block=true max-bytes=4194304 ! {codec}parse ! {chosen} ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink name=output sync=false max-buffers=2 drop=false enable-last-sample=false")
    stop = threading.Event()
    thread = None
    try:
        source = pipeline.get_by_name("input")
        source.set_property("caps", Gst.Caps.from_string(config["caps"]))
        sink = pipeline.get_by_name("output")
        units = config["units"]
        if not units or len(units) > 4096:
            raise ValueError("invalid encoded window")
        # Preserve both timestamp differences, including B-frame reordering.
        origin = min(value for unit in units for value in (unit["pts_ns"], unit.get("dts_ns")) if value is not None)
        by_pts = {}
        for unit in units:
            by_pts.setdefault(unit["pts_ns"] - origin, []).append(unit)
        targets = sorted(set(float(value) for value in config["targets"]))
        if not targets or len(targets) > 32:
            raise ValueError("invalid frame targets")
        failure = []

        def feed():
            try:
                for unit in units:
                    if stop.is_set():
                        break
                    data = os.pread(input_fd, unit["size"], unit["offset"])
                    if len(data) != unit["size"]:
                        raise ValueError("truncated encoded window")
                    buffer = Gst.Buffer.new_allocate(None, len(data), None)
                    buffer.fill(0, data)
                    buffer.pts = unit["pts_ns"] - origin
                    if unit.get("dts_ns") is not None:
                        buffer.dts = unit["dts_ns"] - origin
                    if unit.get("duration_ns") is not None:
                        buffer.duration = unit["duration_ns"]
                    if source.emit("push-buffer", buffer) != Gst.FlowReturn.OK:
                        break
                source.emit("end-of-stream")
            except Exception:
                failure.append("input_error")

        results = []
        written = 0
        pipeline.set_state(Gst.State.PLAYING)
        thread = threading.Thread(target=feed, daemon=True)
        thread.start()
        bus = pipeline.get_bus()
        while targets and time.monotonic() < deadline:
            sample = sink.emit("try-pull-sample", 100 * Gst.MSECOND)
            error = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.WARNING)
            if error is not None or failure:
                return {"status": "miss", "reason": "decode_error"}
            if sample is None:
                if sink.get_property("eos"):
                    break
                continue
            buffer = sample.get_buffer()
            matches = by_pts.get(int(buffer.pts), [])
            if len(matches) != 1 or buffer.has_flags(Gst.BufferFlags.CORRUPTED):
                continue
            unit = matches[0]
            chosen_targets = []
            while targets and unit["epoch"] >= targets[0]:
                target = targets.pop(0)
                if unit["epoch"] - target > float(config["maximum_frame_offset_seconds"]):
                    return {"status": "miss", "reason": "target_gap"}
                chosen_targets.append(target)
            if not chosen_targets:
                continue
            structure = sample.get_caps().get_structure(0)
            width, height = int(structure.get_value("width")), int(structure.get_value("height"))
            # GstVideo maps negotiated row stride; BGR rows may be padded.
            import gi
            gi.require_version("GstVideo", "1.0")
            from gi.repository import GstVideo
            info = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
            size = width * height * 3
            if width <= 0 or height <= 0 or written + size > int(config["maximum_output_bytes"]):
                return {"status": "capacity_denied", "reason": "output_limit"}
            data = buffer.extract_dup(0, buffer.get_size())
            for row in range(height):
                begin = info.offset[0] + row * info.stride[0]
                _write(output_fd, data[begin:begin + width * 3])
            results.append({"targets": chosen_targets, "offset": written, "size": size,
                            "width": width, "height": height, "unit": unit})
            written += size
        if targets:
            return {"status": "pending", "reason": "dependency_window_incomplete"}
        return {"status": "ready", "frames": results}
    finally:
        stop.set()
        try:
            if pipeline.set_state(Gst.State.NULL) == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("decoder teardown failed")
        except Exception as error:
            raise DecoderCleanupIncomplete("decoder pipeline did not stop") from error
        if thread is not None and thread.ident is not None:
            thread.join(timeout=2)
            if thread.is_alive():
                raise DecoderCleanupIncomplete("decoder feeder did not stop")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "decode"))
    parser.add_argument("--control-fd", type=int)
    parser.add_argument("--input-fd", type=int)
    parser.add_argument("--output-fd", type=int)
    args = parser.parse_args()
    config = json.loads(sys.stdin.buffer.readline(1024 * 1024))
    if args.mode == "decode":
        try:
            result = decode(config, args.input_fd, args.output_fd)
        except Exception:
            result = {"status": "miss", "reason": "decode_error"}
        print(json.dumps(result), flush=True)
        return
    sock = socket.socket(fileno=args.control_fd)
    collector = None
    try:
        collector = Collector(_gst(), config)
        while True:
            request, descriptor = receive_control(sock)
            if descriptor is not None:
                os.close(descriptor)
                raise ValueError("unexpected input descriptor")
            if request.get("op") == "stop":
                break
            collector.request(request, sock)
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    finally:
        if collector is not None:
            collector.close()
        sock.close()


if __name__ == "__main__":
    from survng.dlstreamer_live import _apply_dlstreamer_env
    previous = os.environ.get("LD_LIBRARY_PATH", "")
    _apply_dlstreamer_env()
    if previous != os.environ.get("LD_LIBRARY_PATH", ""):
        os.execv(sys.executable, [sys.executable, "-m", "survng.main_evidence_worker", *sys.argv[1:]])
    main()
