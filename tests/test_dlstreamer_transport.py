"""Process-boundary regressions: native diagnostics cannot enter the data pipe."""
from __future__ import annotations

import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from survng.app.dlstreamer_capture import _start_child
from survng.app.dlstreamer_protocol import (
    MessageReader, ProtocolWriter, TYPE_STATUS, decode_json_payload, encode_json,
    encode_frame_parts, encode_frame, decode_frame_payload, decode_stream_payload, TYPE_FRAME,
)

ROOT = Path(__file__).resolve().parents[1]


def test_writer_serializes_large_messages_and_completes_short_writes():
    class ShortWrites(io.BytesIO):
        def write(self, data):
            return super().write(data[:113])

    output = ShortWrites()
    writer = ProtocolWriter(output)
    def send(index):
        writer.send(encode_json(TYPE_STATUS, {"index": index, "data": "x" * 100000}))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(send, range(16)))
    reader = MessageReader()
    reader.feed(output.getvalue())
    indices = []
    while (message := reader.pop()) is not None:
        kind, payload = message
        assert kind == TYPE_STATUS
        decoded = decode_json_payload(payload)
        assert decoded["data"] == "x" * 100000
        indices.append(decoded["index"])
    assert sorted(indices) == list(range(16))


def test_writer_never_reuses_connection_after_partial_failure():
    class BrokenOutput:
        calls = 0
        def write(self, data):
            self.calls += 1
            if self.calls == 1:
                return 2
            raise BrokenPipeError("closed")
        def flush(self):
            pass

    output = BrokenOutput()
    writer = ProtocolWriter(output)
    with pytest.raises(BrokenPipeError):
        writer.send(b"first message")
    calls = output.calls
    with pytest.raises(BrokenPipeError):
        writer.send(b"next message")
    assert output.calls == calls


def test_frame_parts_share_pixels_and_remain_atomic_with_short_writes():
    class ShortWrites(io.BytesIO):
        def write(self, data):
            return super().write(data[:113])

    pixels = bytes(range(256)) * 300
    args = dict(width=320, height=240, sequence=1, pts=.2, pixels=pixels, stream_id='camera')
    parts = encode_frame_parts(**args)
    assert parts[1] is pixels
    assert b''.join(parts) == encode_frame(**args)
    output = ShortWrites()
    writer = ProtocolWriter(output)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: writer.send(parts), range(8)))
    reader = MessageReader()
    reader.feed(output.getvalue())
    count = 0
    while (message := reader.pop()) is not None:
        kind, payload = message
        assert kind == TYPE_FRAME
        stream, inner = decode_stream_payload(payload)
        assert stream == 'camera'
        assert decode_frame_payload(inner)[4] == pixels
        count += 1
    assert count == 8


def test_frame_parts_failure_after_header_poison_writer():
    class BrokenPixels:
        calls = 0
        def write(self, data):
            self.calls += 1
            if self.calls == 1:
                return len(data)
            raise BrokenPipeError('pixel write failed')
        def flush(self):
            pass

    output = BrokenPixels()
    writer = ProtocolWriter(output)
    parts = encode_frame_parts(width=1, height=1, sequence=1, pts=0, pixels=b'x')
    with pytest.raises(BrokenPipeError):
        writer.send(parts)
    with pytest.raises(BrokenPipeError):
        writer.send(parts)
    assert output.calls == 2


@pytest.mark.parametrize("fatal", [False, True])
def test_inherited_protocol_survives_exec_and_native_stdio_noise(fatal):
    # Exercise the real child entrypoint after exec, without cameras or GI.
    child = f'''
import os
from survng import dlstreamer_live as live
from survng.app.dlstreamer_protocol import TYPE_STATUS, encode_json

def run(argv=None, *, output):
    os.write(1, b"native stdout noise\\n")
    os.write(2, b"native stderr noise\\n")
    output.send(encode_json(TYPE_STATUS, {{"ok": True}}))
    if {fatal!r}:
        raise RuntimeError("synthetic failure")
    return 0
live.run = run
raise SystemExit(live.main(["--supervisor"]))
'''
    bootstrap = f'import os,sys; os.execv(sys.executable, [sys.executable,"-c",{child!r}])'
    process, data = _start_child([sys.executable, "-c", bootstrap], os.environ.copy(), str(ROOT))
    try:
        stdout, stderr = process.communicate(timeout=5)
        reader = MessageReader()
        reader.feed(data.read())
        first = reader.pop()
        assert first is not None
        assert decode_json_payload(first[1]) == {"ok": True}
        if fatal:
            from survng.app.dlstreamer_protocol import TYPE_FATAL
            kind, payload = reader.pop()
            assert kind == TYPE_FATAL
            assert decode_json_payload(payload)["error"] == "synthetic failure"
        assert reader.pop() is None
        assert stdout == b"native stdout noise\n"
        assert b"native stderr noise" in stderr
        assert process.returncode == int(fatal)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        data.close()


def test_spawn_failure_does_not_leak_pipe_descriptors():
    before = set(os.listdir("/proc/self/fd"))
    with pytest.raises(FileNotFoundError):
        _start_child(["/nonexistent-survng-test-child"], os.environ.copy(), str(ROOT))
    assert set(os.listdir("/proc/self/fd")) == before


def test_shutdown_failure_does_not_wait_for_blocked_protocol_writer():
    script = '''
import threading, time
from survng import dlstreamer_live as live
from survng.app.dlstreamer_protocol import TYPE_STATUS, encode_json

def run(argv=None, *, output):
    started = threading.Event()
    def blocked():
        started.set()
        output.send(encode_json(TYPE_STATUS, {"data": "x" * 2000000}))
    threading.Thread(target=blocked, daemon=True).start()
    assert started.wait(1)
    time.sleep(.05)
    raise live.StreamShutdownError("synthetic stuck worker")
live.run = run
raise SystemExit(live.main(["--supervisor"]))
'''
    process, data = _start_child([sys.executable, "-c", script], os.environ.copy(), str(ROOT))
    try:
        # Intentionally never drain data: shutdown must not acquire the writer
        # lock or flush the blocked pipe to emit its fatal message.
        process.wait(timeout=5)
        assert process.returncode == 1
        assert b"synthetic stuck worker" in process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        data.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


def test_native_diagnostics_are_redacted_before_channels_are_combined():
    from survng.app.dlstreamer_capture import _diagnostic_tail
    stderr = bytearray(b"stderr context\nrtsp://admin:unfinished-secret")
    stdout = bytearray(b"stdout context\n")
    result = _diagnostic_tail(stderr, stdout)
    assert "unfinished-secret" not in result
    assert "stdout context" in result
    stderr.extend(b"@camera/live\n")
    result = _diagnostic_tail(stderr, stdout)
    assert "unfinished-secret" not in result
    assert "rtsp://admin:***@camera/live" in result


def test_supervisor_signal_does_not_reenter_worker_registry_lock():
    import subprocess
    script = '''
import io, signal, threading
from fractions import Fraction
from types import SimpleNamespace
from survng import dlstreamer_live as live

locks = []
handlers = {}
def lock():
    value = threading.Lock()
    locks.append(value)
    return value
live.threading = SimpleNamespace(Lock=lock, Event=threading.Event, Thread=threading.Thread)
def install(signum, handler):
    handlers[signum] = handler
live.signal = SimpleNamespace(signal=install, SIGINT=signal.SIGINT, SIGTERM=signal.SIGTERM)
class Input:
    def readline(self):
        # Deliver shutdown while the command thread owns its registry lock.
        with locks[-1]:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        return b""
live.sys.stdin = SimpleNamespace(buffer=Input())
assert live._run_supervisor(
    object(), live._parser().parse_args([]), detect=False, model_path=None,
    instance_id="", rate=Fraction(1), detect_rate=Fraction(1),
    qualifier_width=320, jpeg_rate=None, open_timeout=3, stdout=io.BytesIO(),
) == 0
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr
