from __future__ import annotations

import hashlib
import logging
import socket
import struct
import threading
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from io import BytesIO

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from .config import CameraConfig

LOGGER = logging.getLogger(__name__)

MAGIC_HEADER = 0x0ABCDEF0
MSG_ID_LOGIN = 1
MSG_ID_VIDEO = 3
CLASS_LEGACY = 0x6514
CLASS_MODERN_NO_OFFSET = 0x6614
CLASS_MODERN = 0x6414
CLASS_MODERN_ZERO = 0x0000
LEGACY_UPGRADE_AES = 0xDC12
RX_TIMEOUT_SECONDS = 15.0

INFO_V1 = 0x31303031
INFO_V2 = 0x32303031
IFRAME_FIRST = 0x63643030
IFRAME_LAST = 0x63643039
PFRAME_FIRST = 0x63643130
PFRAME_LAST = 0x63643139
AAC = 0x62773530
ADPCM = 0x62773130


class EncryptionKind(Enum):
    UNENCRYPTED = "none"
    BC_ENCRYPT = "bc"
    AES = "aes"
    FULL_AES = "full_aes"


@dataclass
class BcMessage:
    msg_id: int
    channel_id: int
    stream_type: int
    msg_num: int
    response_code: int
    cls: int
    xml: ET.Element | None = None
    extension: ET.Element | None = None
    binary: bytes = b""


@dataclass
class VideoFrame:
    codec: str
    keyframe: bool
    microseconds: int
    data: bytes


class BaichuanError(RuntimeError):
    pass


class BaichuanNativeClient:
    """
    Minimal in-process Reolink Baichuan video client.

    Protocol details are ported from Neolink.NET, which is AGPL-3.0 licensed.
    Keep this module isolated so its provenance and license implications remain
    easy to audit if SurvNG is distributed.
    """

    def __init__(self, camera: CameraConfig, source: str = "live") -> None:
        bc = camera.baichuan
        self.host = bc.host or camera.onvif.host
        self.port = bc.port
        self.username = bc.username or camera.onvif.username
        self.password = bc.password or camera.onvif.password
        self.channel = bc.channel
        self.source = "main" if source == "main" else "live"
        self._sock: socket.socket | None = None
        self._msg_num = -1
        self._encryption = EncryptionKind.UNENCRYPTED
        self._aes_key: bytes | None = None
        self._binary_msg_nums: set[int] = set()

    def video_frames(self, stop: threading.Event | None = None) -> Iterator[VideoFrame]:
        if not self.host or not self.username:
            raise BaichuanError("baichuan host and username are required")
        stop = stop or threading.Event()
        with self:
            self.login()
            self.start_video()
            reader = MediaFrameReader(self._binary_chunks(stop))
            while not stop.is_set():
                frame = reader.read_frame()
                if isinstance(frame, VideoFrame):
                    yield frame

    def video_bytes(self, stop: threading.Event | None = None) -> Iterator[bytes]:
        have_keyframe = False
        for frame in self.video_frames(stop):
            if not have_keyframe:
                if not frame.keyframe:
                    continue
                have_keyframe = True
            yield frame.data

    def __enter__(self) -> BaichuanNativeClient:
        self._sock = socket.create_connection((self.host, self.port), timeout=10)
        self._sock.settimeout(RX_TIMEOUT_SECONDS)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def login(self) -> None:
        legacy = BcMessage(
            msg_id=MSG_ID_LOGIN,
            channel_id=self.channel,
            stream_type=0,
            msg_num=self._next_msg_num(),
            response_code=LEGACY_UPGRADE_AES,
            cls=CLASS_LEGACY,
        )
        self._send(legacy)
        reply = self._read()
        nonce = _child_text(reply.xml, "Encryption", "nonce")
        if not nonce:
            raise BaichuanError("login reply did not include a nonce")
        negotiated = reply.response_code & 0xFF

        body = ET.Element("body")
        login_user = ET.SubElement(body, "LoginUser", {"version": "1.1"})
        ET.SubElement(login_user, "userName").text = _md5_31(self.username + nonce)
        ET.SubElement(login_user, "password").text = _md5_31((self.password or "") + nonce)
        ET.SubElement(login_user, "userVer").text = "1"
        login_net = ET.SubElement(body, "LoginNet", {"version": "1.1"})
        ET.SubElement(login_net, "type").text = "LAN"
        ET.SubElement(login_net, "udpPort").text = "0"
        modern = BcMessage(
            msg_id=MSG_ID_LOGIN,
            channel_id=self.channel,
            stream_type=0,
            msg_num=self._next_msg_num(),
            response_code=0,
            cls=CLASS_MODERN,
            xml=body,
        )
        self._send(modern)
        modern_reply = self._read()
        if modern_reply.response_code not in (0, 200):
            raise BaichuanError(f"camera rejected login: response {modern_reply.response_code}")
        if modern_reply.response_code == 0 and modern_reply.xml is None and not modern_reply.binary:
            raise BaichuanError("camera rejected login")

        if negotiated not in (0x00, 0x01):
            self._encryption = EncryptionKind.AES if negotiated == 0x02 else EncryptionKind.FULL_AES
            self._aes_key = _make_aes_key(nonce, self.password or "")

    def start_video(self) -> None:
        stream_code = 0 if self.source == "main" else 1
        handle = 0 if self.source == "main" else 256
        stream_name = "mainStream" if self.source == "main" else "subStream"
        body = ET.Element("body")
        preview = ET.SubElement(body, "Preview", {"version": "1.1"})
        ET.SubElement(preview, "channelId").text = str(self.channel)
        ET.SubElement(preview, "handle").text = str(handle)
        ET.SubElement(preview, "streamType").text = stream_name
        self._send(
            BcMessage(
                msg_id=MSG_ID_VIDEO,
                channel_id=self.channel,
                stream_type=stream_code,
                msg_num=self._next_msg_num(),
                response_code=0,
                cls=CLASS_MODERN,
                xml=body,
            )
        )

    def _binary_chunks(self, stop: threading.Event) -> Iterator[bytes]:
        while not stop.is_set():
            msg = self._read()
            if msg.msg_id != MSG_ID_VIDEO:
                continue
            if msg.binary:
                yield msg.binary

    def _read(self) -> BcMessage:
        sock = self._require_socket()
        head = _recv_exact(sock, 20)
        magic, msg_id, body_len = struct.unpack_from("<III", head, 0)
        if magic != MAGIC_HEADER:
            raise BaichuanError(f"invalid Baichuan header 0x{magic:08x}")
        channel_id = head[12]
        stream_type = head[13]
        msg_num, response_code, cls = struct.unpack_from("<HHH", head, 14)
        payload_offset: int | None = None
        if _has_payload_offset(cls):
            payload_offset = struct.unpack("<I", _recv_exact(sock, 4))[0]
        if body_len > 64 * 1024 * 1024:
            raise BaichuanError(f"implausible Baichuan body length {body_len}")
        body = _recv_exact(sock, body_len) if body_len else b""
        msg = BcMessage(msg_id, channel_id, stream_type, msg_num, response_code, cls)

        if not _is_modern_class(cls):
            return msg

        if msg_id == MSG_ID_LOGIN and (response_code >> 8) == 0xDD:
            self._encryption = (
                EncryptionKind.UNENCRYPTED
                if (response_code & 0xFF) == 0
                else EncryptionKind.BC_ENCRYPT
            )

        ext_len = payload_offset or 0
        if ext_len > body_len:
            raise BaichuanError("Baichuan payload offset exceeds body length")
        if ext_len:
            ext_plain = self._decrypt(channel_id, body[:ext_len])
            msg.extension = _parse_xml(ext_plain, expected="Extension")
            if _element_int(msg.extension, "binaryData") == 1:
                self._binary_msg_nums.add(msg_num)

        payload = body[ext_len:]
        if payload:
            if msg_num in self._binary_msg_nums:
                encrypt_len = _element_int(msg.extension, "encryptLen")
                if self._encryption == EncryptionKind.FULL_AES and encrypt_len:
                    plain = self._decrypt(channel_id, payload)
                    msg.binary = plain[:encrypt_len]
                else:
                    msg.binary = payload
            else:
                plain = self._decrypt(channel_id, payload)
                msg.xml = _parse_xml(plain, expected="body")
                if msg.xml is None:
                    msg.binary = payload
        return msg

    def _send(self, msg: BcMessage) -> None:
        self._require_socket().sendall(self._serialize(msg))

    def _serialize(self, msg: BcMessage) -> bytes:
        payload_offset: int | None = None
        body = b""
        if _is_modern_class(msg.cls):
            if _has_payload_offset(msg.cls):
                payload_offset = 0
            if msg.xml is not None:
                body += self._encrypt(msg.channel_id, _serialize_xml(msg.xml))
        header = struct.pack(
            "<IIIBBHHH",
            MAGIC_HEADER,
            msg.msg_id,
            len(body),
            msg.channel_id,
            msg.stream_type,
            msg.msg_num,
            msg.response_code,
            msg.cls,
        )
        if payload_offset is not None:
            header += struct.pack("<I", payload_offset)
        return header + body

    def _encrypt(self, offset: int, data: bytes) -> bytes:
        if self._encryption == EncryptionKind.UNENCRYPTED:
            return data
        if self._encryption == EncryptionKind.BC_ENCRYPT or self._aes_key is None:
            return _bc_xor(offset, data)
        return _aes_cfb(data, self._aes_key, encrypting=True)

    def _decrypt(self, offset: int, data: bytes) -> bytes:
        if self._encryption == EncryptionKind.UNENCRYPTED:
            return data
        if self._encryption == EncryptionKind.BC_ENCRYPT or self._aes_key is None:
            return _bc_xor(offset, data)
        return _aes_cfb(data, self._aes_key, encrypting=False)

    def _next_msg_num(self) -> int:
        self._msg_num = (self._msg_num + 1) & 0xFFFF
        return self._msg_num

    def _require_socket(self) -> socket.socket:
        if self._sock is None:
            raise BaichuanError("not connected")
        return self._sock


class BaichuanFfmpegPipe:
    def __init__(self, camera: CameraConfig, source: str) -> None:
        self.camera = camera
        self.source = source
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self, stdin) -> None:
        self.thread = threading.Thread(target=self._run, args=(stdin,), daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
            self.thread = None

    def _run(self, stdin) -> None:
        try:
            client = BaichuanNativeClient(self.camera, self.source)
            for chunk in client.video_bytes(self.stop_event):
                if self.stop_event.is_set():
                    break
                stdin.write(chunk)
                stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        except Exception:
            LOGGER.exception("native Baichuan video pump failed for %s", self.camera.id)
        finally:
            try:
                stdin.close()
            except Exception:
                pass


class MediaFrameReader:
    def __init__(self, chunks: Iterator[bytes]) -> None:
        self.chunks = chunks
        self.buffer = bytearray()
        self.offset = 0

    def read_frame(self) -> object:
        while True:
            magic = struct.unpack("<I", self._peek(4))[0]
            known = (
                magic in (INFO_V1, INFO_V2, AAC, ADPCM)
                or IFRAME_FIRST <= magic <= IFRAME_LAST
                or PFRAME_FIRST <= magic <= PFRAME_LAST
            )
            if known:
                self._consume(4)
                break
            self._consume(1)

        if magic in (INFO_V1, INFO_V2):
            data = self._read(28)
            width = struct.unpack_from("<I", data, 4)[0]
            height = struct.unpack_from("<I", data, 8)[0]
            fps = data[13]
            return {"kind": "info", "width": width, "height": height, "fps": fps}
        if IFRAME_FIRST <= magic <= IFRAME_LAST:
            return self._read_video(True)
        if PFRAME_FIRST <= magic <= PFRAME_LAST:
            return self._read_video(False)
        if magic == AAC:
            return self._read_audio()
        return self._read_audio(adpcm=True)

    def _read_video(self, keyframe: bool) -> VideoFrame:
        head = self._read(16)
        codec = head[:4].decode("ascii", errors="replace")
        if codec not in ("H264", "H265"):
            raise BaichuanError(f"unsupported native Baichuan video codec {codec}")
        payload_size = struct.unpack_from("<I", head, 4)[0]
        additional_header_size = struct.unpack_from("<I", head, 8)[0]
        microseconds = struct.unpack_from("<I", head, 12)[0]
        self._read(4)
        if additional_header_size:
            self._read(additional_header_size)
        if payload_size > 32 * 1024 * 1024:
            raise BaichuanError(f"implausible video payload size {payload_size}")
        data = self._read(payload_size)
        self._skip_padding(payload_size)
        return VideoFrame(codec=codec, keyframe=keyframe, microseconds=microseconds, data=data)

    def _read_audio(self, adpcm: bool = False) -> dict[str, object]:
        payload_size = struct.unpack_from("<H", self._read(4), 0)[0]
        if adpcm:
            self._read(4)
            payload_size = max(payload_size - 4, 0)
        data = self._read(payload_size)
        self._skip_padding(payload_size)
        return {"kind": "audio", "data": data}

    def _skip_padding(self, payload_size: int) -> None:
        rem = payload_size % 8
        if rem:
            self._read(8 - rem)

    def _peek(self, count: int) -> bytes:
        while len(self.buffer) - self.offset < count:
            self.buffer.extend(next(self.chunks))
        return bytes(self.buffer[self.offset : self.offset + count])

    def _read(self, count: int) -> bytes:
        data = self._peek(count)
        self._consume(count)
        return data

    def _consume(self, count: int) -> None:
        self.offset += count
        if self.offset > 1024 * 1024:
            del self.buffer[: self.offset]
            self.offset = 0


def is_native_baichuan(camera: CameraConfig) -> bool:
    return camera.video_backend == "baichuan_native" and camera.baichuan.enabled


def ffmpeg_input_args(camera: CameraConfig, source: str) -> list[str]:
    if not is_native_baichuan(camera):
        return [
            "-fflags",
            "+genpts",
            "-dts_error_threshold",
            "10",
            "-rtsp_transport",
            "tcp",
            "-i",
            camera.source_url(source),
        ]
    return [
        "-use_wallclock_as_timestamps",
        "1",
        "-fflags",
        "+genpts",
        "-f",
        "h264",
        "-probesize",
        "512k",
        "-analyzeduration",
        "1000000",
        "-i",
        "pipe:0",
    ]


def ffmpeg_timestamp_repair_args(camera: CameraConfig) -> list[str]:
    if is_native_baichuan(camera):
        return []
    missing_pts = (
        "if(eq(PTS\\,NOPTS)\\,"
        "if(eq(DTS\\,NOPTS)\\,"
        "if(eq(PREV_OUTPTS\\,NOPTS)\\,0\\,PREV_OUTPTS+max(DURATION\\,1))\\,DTS)\\,PTS)"
    )
    missing_dts = (
        "if(eq(DTS\\,NOPTS)\\,"
        "if(eq(PTS\\,NOPTS)\\,"
        "if(eq(PREV_OUTDTS\\,NOPTS)\\,0\\,PREV_OUTDTS+max(DURATION\\,1))\\,PTS)\\,DTS)"
    )
    return ["-bsf:v", f"setts=pts={missing_pts}:dts={missing_dts}"]


def start_ffmpeg_pipe(camera: CameraConfig, source: str, process) -> BaichuanFfmpegPipe | None:
    if not is_native_baichuan(camera):
        return None
    if process.stdin is None:
        raise BaichuanError("ffmpeg stdin is unavailable for native Baichuan")
    pipe = BaichuanFfmpegPipe(camera, source)
    pipe.start(process.stdin)
    return pipe


def _has_payload_offset(cls: int) -> bool:
    return cls in (CLASS_MODERN, CLASS_MODERN_ZERO)


def _is_modern_class(cls: int) -> bool:
    return cls != CLASS_LEGACY


def _md5_31(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest().upper()[:31]


def _make_aes_key(nonce: str, password: str) -> bytes:
    value = hashlib.md5(f"{nonce}-{password}".encode("utf-8")).hexdigest().upper()
    return value[:16].encode("ascii")


def _bc_xor(offset: int, data: bytes) -> bytes:
    key = bytes([0x1F, 0x2D, 0x3C, 0x4B, 0x5A, 0x69, 0x78, 0xFF])
    return bytes(byte ^ key[(offset + index) % 8] ^ offset for index, byte in enumerate(data))


def _aes_cfb(data: bytes, key: bytes, encrypting: bool) -> bytes:
    cipher = Cipher(algorithms.AES(key), modes.CFB(b"0123456789abcdef"))
    ctx = cipher.encryptor() if encrypting else cipher.decryptor()
    return ctx.update(data) + ctx.finalize()


def _serialize_xml(root: ET.Element) -> bytes:
    return b'<?xml version="1.0" encoding="UTF-8" ?>\n' + ET.tostring(root, encoding="utf-8") + b"\n"


def _parse_xml(data: bytes, expected: str) -> ET.Element | None:
    try:
        text = data.split(b"\0", 1)[0].decode("utf-8", errors="ignore").strip()
        if not text:
            return None
        root = ET.fromstring(text)
    except ET.ParseError:
        return None
    return root if root.tag == expected else None


def _child_text(root: ET.Element | None, *path: str) -> str:
    if root is None:
        return ""
    current = root
    for part in path:
        found = current.find(part)
        if found is None:
            return ""
        current = found
    return (current.text or "").strip()


def _element_int(root: ET.Element | None, name: str) -> int | None:
    if root is None:
        return None
    found = root.find(name)
    if found is None or found.text is None:
        return None
    try:
        return int(found.text.strip())
    except ValueError:
        return None


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    out = BytesIO()
    while out.tell() < size:
        chunk = sock.recv(size - out.tell())
        if not chunk:
            raise BaichuanError("Baichuan connection closed")
        out.write(chunk)
    return out.getvalue()
