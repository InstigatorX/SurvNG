"""An HTTP session for ONVIF whose blocking calls a stop can interrupt.

Closing a ``requests`` session only drops idle pooled connections. A thread
blocked in ``connect()`` or waiting on a reply keeps waiting until its own
timeout, so an unreachable camera could hold an ONVIF worker for the full
timeout after it was told to stop. This session registers every socket it
creates before connecting it; ``shutdown(SHUT_RDWR)`` on a registered socket
wakes a thread blocked on it in connect, send or receive (Linux).

Two levels of interruption:
- ``abort_connecting()`` aborts connects still in progress and leaves
  established connections alone, so a healthy camera can still be sent
  Unsubscribe while the worker winds down.
- ``cancel()`` shuts every live socket and refuses new connections, for when
  the worker did not wind down in its grace period.
"""

from __future__ import annotations

import socket
import threading
import weakref
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import ConnectTimeoutError, NameResolutionError, NewConnectionError
from urllib3.util import connection as urllib3_connection


class OnvifTransportCancelled(OSError):
    """A connection attempt that a stop interrupted or refused."""


class SocketRegistry:
    """Sockets of one session, and whether that session was cancelled."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connecting: weakref.WeakSet[socket.socket] = weakref.WeakSet()
        self._open: weakref.WeakSet[socket.socket] = weakref.WeakSet()
        self._cancelled = False
        self._connect_aborts = 0

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def connect_aborts(self) -> int:
        with self._lock:
            return self._connect_aborts

    def begin(self, sock: socket.socket) -> int:
        """Register a socket about to connect; returns the abort count to compare."""
        with self._lock:
            if self._cancelled:
                raise OnvifTransportCancelled("ONVIF transport was cancelled")
            self._connecting.add(sock)
            self._open.add(sock)
            return self._connect_aborts

    def connected(self, sock: socket.socket) -> None:
        with self._lock:
            self._connecting.discard(sock)

    def discard(self, sock: socket.socket) -> None:
        with self._lock:
            self._connecting.discard(sock)
            self._open.discard(sock)

    def abort_connecting(self) -> None:
        with self._lock:
            self._connect_aborts += 1
            sockets = list(self._connecting)
        _shutdown(sockets)

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            self._connect_aborts += 1
            sockets = list(self._open)
        _shutdown(sockets)


def _shutdown(sockets: list[socket.socket]) -> None:
    for sock in sockets:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            # Already closed, or not connected yet on a platform that
            # refuses shutdown there; nothing else to wake.
            pass


def _connect(registry: SocketRegistry, connection: HTTPConnection) -> socket.socket:
    """urllib3's create_connection, registering each socket before connecting."""
    host, port = connection._dns_host, connection.port
    if host.startswith("["):
        host = host.strip("[]")
    family = urllib3_connection.allowed_gai_family()
    try:
        addresses = socket.getaddrinfo(host, port, family, socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise NameResolutionError(connection.host, connection, error) from error
    last_error: BaseException | None = None
    for af, socktype, proto, _name, address in addresses:
        sock = socket.socket(af, socktype, proto)
        try:
            aborts = registry.begin(sock)
        except OnvifTransportCancelled:
            sock.close()
            raise
        try:
            urllib3_connection._set_socket_options(sock, connection.socket_options)
            if connection.timeout is not None:
                sock.settimeout(connection.timeout)
            if connection.source_address:
                sock.bind(connection.source_address)
            sock.connect(address)
            if registry.connect_aborts() != aborts:
                raise OnvifTransportCancelled("ONVIF connect was aborted by a stop")
            registry.connected(sock)
            return sock
        except BaseException as error:
            registry.discard(sock)
            sock.close()
            if registry.connect_aborts() != aborts:
                # A shutdown during connect surfaces as a reset or a bad
                # descriptor; urllib3 mishandles both. Report it as what it is.
                raise OnvifTransportCancelled("ONVIF connect was aborted by a stop") from None
            if not isinstance(error, OSError):
                raise
            last_error = error
    if last_error is None:
        raise OSError("getaddrinfo returned no addresses")
    raise last_error


class _CancellableConnectionMixin:
    _survng_registry: SocketRegistry

    def _new_conn(self) -> socket.socket:  # type: ignore[override]
        try:
            return _connect(self._survng_registry, self)  # type: ignore[arg-type]
        except NameResolutionError:
            raise
        except OnvifTransportCancelled as error:
            raise NewConnectionError(self, f"Failed to establish a new connection: {error}") from None
        except socket.timeout as error:
            raise ConnectTimeoutError(
                self,
                f"Connection to {self.host} timed out. (connect timeout={self.timeout})",  # type: ignore[attr-defined]
            ) from error
        except OSError as error:
            raise NewConnectionError(self, f"Failed to establish a new connection: {error}") from error


def _pool_classes(registry: SocketRegistry) -> dict[str, type]:
    http_connection = type(
        "CancellableHTTPConnection",
        (_CancellableConnectionMixin, HTTPConnection),
        {"_survng_registry": registry},
    )
    https_connection = type(
        "CancellableHTTPSConnection",
        (_CancellableConnectionMixin, HTTPSConnection),
        {"_survng_registry": registry},
    )
    return {
        "http": type("CancellableHTTPConnectionPool", (HTTPConnectionPool,), {"ConnectionCls": http_connection}),
        "https": type("CancellableHTTPSConnectionPool", (HTTPSConnectionPool,), {"ConnectionCls": https_connection}),
    }


class _CancellableAdapter(HTTPAdapter):
    def __init__(self, registry: SocketRegistry, **kwargs: Any) -> None:
        self._survng_registry = registry
        super().__init__(**kwargs)

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = _pool_classes(self._survng_registry)
        self.poolmanager.key_fn_by_scheme = dict(self.poolmanager.key_fn_by_scheme)


class CancellableSession(requests.Session):
    """A requests session whose in-flight sockets a stop can interrupt."""

    def __init__(self) -> None:
        super().__init__()
        self.registry = SocketRegistry()
        adapter = _CancellableAdapter(self.registry)
        self.mount("http://", adapter)
        self.mount("https://", adapter)

    def abort_connecting(self) -> None:
        self.registry.abort_connecting()

    def cancel(self) -> None:
        """Interrupt every blocking call on this session and refuse new ones."""
        self.registry.cancel()
        self.close()
