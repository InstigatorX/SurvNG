"""Bounded encoded evidence and private IPC; usable from system Python/GI.

No application configuration, NumPy, or detector imports belong in this module.
"""
from __future__ import annotations

import array
from collections import deque
from dataclasses import dataclass
import json
import math
import socket
import struct
import time
import uuid

MAX_CONTROL_BYTES = 1024 * 1024
MAX_ACCESS_UNITS = 4096


def send_control(sock: socket.socket, value: dict, fd: int | None = None) -> None:
    payload = json.dumps({**value, "protocol_version": 1}, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > MAX_CONTROL_BYTES:
        raise ValueError("main evidence control message too large")
    ancillary = [] if fd is None else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
    # SCM_RIGHTS belongs to the first byte. sendall handles any remaining bytes.
    packet = struct.pack("!I", len(payload)) + payload
    sent = sock.sendmsg([packet], ancillary)
    sock.sendall(packet[sent:])


def receive_control(sock: socket.socket) -> tuple[dict, int | None]:
    header, anc, flags, _ = sock.recvmsg(4, socket.CMSG_SPACE(array.array("i").itemsize))
    if not header:
        raise EOFError("main evidence peer closed")
    descriptor = None
    for level, kind, data in anc:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            values = array.array("i")
            values.frombytes(data[:len(data) - len(data) % values.itemsize])
            if values:
                descriptor = values[0]
    try:
        if flags & socket.MSG_CTRUNC:
            raise ValueError("truncated main evidence descriptor")
        while len(header) < 4:
            part = sock.recv(4 - len(header))
            if not part:
                raise EOFError("truncated main evidence header")
            header += part
        length = struct.unpack("!I", header)[0]
        if length > MAX_CONTROL_BYTES:
            raise ValueError("main evidence control message too large")
        payload = bytearray()
        while len(payload) < length:
            part = sock.recv(length - len(payload))
            if not part:
                raise EOFError("truncated main evidence payload")
            payload.extend(part)
        result = json.loads(payload)
        if not isinstance(result, dict):
            raise ValueError("invalid main evidence control message")
        if result.pop("protocol_version", None) != 1:
            raise ValueError("unsupported main evidence protocol version")
        return result, descriptor
    except BaseException:
        if descriptor is not None:
            import os
            os.close(descriptor)
        raise


@dataclass(frozen=True)
class AccessUnit:
    ordinal: int
    pts_ns: int
    dts_ns: int | None
    duration_ns: int | None
    received_monotonic: float
    epoch: float
    data: bytes
    random_access: bool


class EncodedRing:
    """One caller owns synchronization. Eviction never retains a broken GOP."""

    def __init__(self, *, max_bytes: int, history_seconds: float, max_au_bytes: int = 4 * 1024 * 1024):
        if max_bytes < 1 or not math.isfinite(history_seconds) or history_seconds <= 0:
            raise ValueError("invalid encoded ring limits")
        self.max_bytes = int(max_bytes)
        self.history_seconds = float(history_seconds)
        self.max_au_bytes = min(int(max_au_bytes), self.max_bytes)
        self.session = uuid.uuid4().hex
        self.generation = 0
        self.gops: deque[list[AccessUnit]] = deque()
        self.bytes = 0
        self.units = 0
        self.gaps = 0
        self.evictions = 0
        self._last_ordinal: int | None = None
        self._last_dts: int | None = None
        self._config = ""

    def reset(self, config: str = "") -> None:
        self.gops.clear()
        self.bytes = self.units = 0
        self.generation += 1
        self.gaps += 1
        self._last_ordinal = self._last_dts = None
        self._config = config

    def add(self, unit: AccessUnit, *, config: str, discontinuity: bool = False) -> bool:
        if (discontinuity or config != self._config
                or self._last_ordinal is not None and unit.ordinal != self._last_ordinal + 1
                or unit.dts_ns is not None and self._last_dts is not None and unit.dts_ns < self._last_dts):
            self.reset(config)
        self._last_ordinal = unit.ordinal
        self._last_dts = unit.dts_ns
        if len(unit.data) > self.max_au_bytes or unit.pts_ns < 0:
            self.reset(config)
            return False
        if unit.random_access:
            self.gops.append([])
        if not self.gops:
            return False
        self.gops[-1].append(unit)
        self.bytes += len(unit.data)
        self.units += 1
        now = unit.received_monotonic
        while self.gops and (self.bytes > self.max_bytes or self.units > MAX_ACCESS_UNITS
                or now - self.gops[0][0].received_monotonic > self.history_seconds):
            old = self.gops.popleft()
            self.bytes -= sum(len(item.data) for item in old)
            self.units -= len(old)
            self.evictions += 1
        return bool(self.gops)

    def window(self, targets: list[float], *, max_bytes: int) -> tuple[str, tuple[AccessUnit, ...]]:
        self.expire()
        if not targets or not all(math.isfinite(value) for value in targets):
            return "unsupported", ()
        groups = list(self.gops)
        if not groups:
            return "pending", ()
        earliest = min(targets)
        # Whole-prefix dependency windows allow the decoder to reorder B frames.
        start = next((i for i in range(len(groups) - 1, -1, -1)
                      if groups[i][0].epoch <= earliest), None)
        if start is None:
            return "miss", ()
        units = tuple(unit for group in groups[start:] for unit in group)
        if max(unit.epoch for unit in units) < max(targets):
            return "pending", ()
        if sum(len(unit.data) for unit in units) > max_bytes:
            return "capacity_denied", ()
        return "ready", units

    def status(self) -> dict:
        self.expire()
        units = [unit for group in self.gops for unit in group]
        return {"session": self.session, "generation": self.generation,
                "state": "ready" if units else "warming", "bytes": self.bytes,
                "access_units": self.units, "gops": len(self.gops),
                "gaps": self.gaps, "evictions": self.evictions,
                "covered_start": units[0].epoch if units else None,
                "covered_end": max((item.epoch for item in units), default=None)}

    def expire(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        while self.gops and now - self.gops[0][0].received_monotonic > self.history_seconds:
            old = self.gops.popleft()
            self.bytes -= sum(len(unit.data) for unit in old)
            self.units -= len(old)
            self.evictions += 1
