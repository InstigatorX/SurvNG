"""Expected-volume identity for destructive media decisions.

A witness file on the location root is paired with a token stored off that
volume. Once an identity exists, a missing or mismatched witness preserves
every row. A mounted replacement disk is not enrolled by the delete path.
The first boot against a never-seen unmounted disk with require_mount false
still looks like a local directory; require_mount is the operator signal
for that case.
"""

from __future__ import annotations

import contextvars
import os
import secrets
import sqlite3
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from .media_storage import MediaStorageRegistry, path_presence


WITNESS_NAME = ".survng-volume"
VolumeVerdict = Literal["confirmed_absent", "present", "unavailable", "unknown"]

# Test hook. Runs after the segment stat and before the final witness read,
# while the location directory descriptor is still open.
before_final_witness: contextvars.ContextVar[Callable[[int], None] | None] = (
    contextvars.ContextVar("survng_before_final_witness", default=None)
)


class VolumeStoreUnavailable(OSError):
    """The off-volume identity database cannot be read. Preserve rows."""


@dataclass(frozen=True, slots=True)
class VolumeIdentity:
    root: str
    token: str
    external: bool


def identity_db_path(anchor: Path, media_roots: list[Path]) -> Path | None:
    """Return an identity-database path that is not inside a media root.

    ``anchor`` may be a directory or a database file. None means every
    candidate sits on a media volume, so callers must not invent an identity
    there during an outage.
    """
    resolved = anchor.expanduser().resolve(strict=False)
    if resolved.suffix:
        resolved = resolved.parent
    for root in media_roots:
        root_resolved = Path(root).expanduser().resolve(strict=False)
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            continue
        return None
    return resolved / "volume-identity.sqlite3"


def _abspath(path: Path | str) -> Path:
    return Path(os.path.abspath(path))


def _relative(path: Path, root: Path) -> str | None:
    try:
        relative = _abspath(path).relative_to(_abspath(root))
    except ValueError:
        return None
    text = relative.as_posix()
    if text.startswith("../") or text == ".." or text.startswith("/"):
        return None
    return text


def _open_dir(root: Path) -> int | None:
    try:
        return os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return None


def _read_witness(dir_fd: int) -> str | None:
    try:
        fd = os.open(WITNESS_NAME, os.O_RDONLY, dir_fd=dir_fd)
    except OSError:
        return None
    try:
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 128)
            if not block:
                break
            chunks.append(block)
            if sum(len(item) for item in chunks) > 128:
                return None
        text = b"".join(chunks).decode("ascii", errors="strict").strip()
    except (OSError, UnicodeError):
        return None
    finally:
        os.close(fd)
    return text or None


def _write_witness(dir_fd: int, token: str, *, overwrite: bool) -> bool:
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if overwrite else os.O_EXCL)
    try:
        fd = os.open(WITNESS_NAME, flags, 0o644, dir_fd=dir_fd)
    except OSError:
        return False
    try:
        os.write(fd, token.encode("ascii"))
    except OSError:
        return False
    finally:
        os.close(fd)
    return True


def _stat_fd(dir_fd: int, relative: str) -> Literal["present", "missing", "unknown"]:
    try:
        mode = os.stat(relative, dir_fd=dir_fd).st_mode
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "unknown"
    return "present" if stat.S_ISREG(mode) else "missing"


def _mounted(root: Path) -> bool:
    try:
        if os.path.ismount(root):
            return True
    except OSError:
        return False
    return MediaStorageRegistry._has_mounted_ancestor(root)


class VolumeIdentityStore:
    """Token store that must stay reachable when a media root is not."""

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def get(self, root: Path) -> VolumeIdentity | None:
        if self.path is None:
            return None
        key = str(_abspath(root))
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT root, token, external FROM volume_identity WHERE root = ?",
                    (key,),
                ).fetchone()
        except sqlite3.Error as error:
            raise VolumeStoreUnavailable(str(error)) from error
        if row is None:
            return None
        return VolumeIdentity(root=str(row[0]), token=str(row[1]), external=bool(row[2]))

    def note_mounted(self, root: Path) -> None:
        """Remember a volume the first time this process sees it mounted.

        An identity that already exists is left alone, including when the
        witness is missing or belongs to a different disk.
        """
        if self.path is None or not _mounted(root):
            return
        try:
            if self.get(root) is not None:
                return
        except VolumeStoreUnavailable:
            return
        dir_fd = _open_dir(root)
        if dir_fd is None:
            return
        try:
            current = _read_witness(dir_fd)
            token = current or secrets.token_hex(16)
            if current is None and not _write_witness(dir_fd, token, overwrite=False):
                return
        finally:
            os.close(dir_fd)
        try:
            self._insert(root, token, external=True)
        except VolumeStoreUnavailable:
            return

    def reconcile(self, root: Path, *, require_mount: bool) -> bool:
        """Operator action. Writes this location's token onto the current root.

        Refuses a require_mount location that is not mounted. Does not run
        from the delete path.
        """
        if self.path is None:
            return False
        if require_mount and not _mounted(root):
            return False
        try:
            existing = self.get(root)
        except VolumeStoreUnavailable:
            return False
        token = existing.token if existing is not None else secrets.token_hex(16)
        dir_fd = _open_dir(root)
        if dir_fd is None:
            return False
        try:
            if not _write_witness(dir_fd, token, overwrite=True):
                return False
        finally:
            os.close(dir_fd)
        try:
            self._upsert(root, token, external=bool(existing.external) if existing else _mounted(root))
        except VolumeStoreUnavailable:
            return False
        return True

    def _connect(self) -> sqlite3.Connection:
        if self.path is None:
            raise VolumeStoreUnavailable("volume identity database is not configured")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS volume_identity (
                root TEXT PRIMARY KEY,
                token TEXT NOT NULL,
                external INTEGER NOT NULL
            )
            """
        )
        connection.commit()
        return connection

    def _insert(self, root: Path, token: str, *, external: bool) -> None:
        key = str(_abspath(root))
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT OR IGNORE INTO volume_identity (root, token, external) VALUES (?, ?, ?)",
                    (key, token, int(external)),
                )
        except sqlite3.Error as error:
            raise VolumeStoreUnavailable(str(error)) from error

    def _upsert(self, root: Path, token: str, *, external: bool) -> None:
        key = str(_abspath(root))
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO volume_identity (root, token, external) VALUES (?, ?, ?)
                    ON CONFLICT(root) DO UPDATE SET token = excluded.token, external = excluded.external
                    """,
                    (key, token, int(external)),
                )
        except sqlite3.Error as error:
            raise VolumeStoreUnavailable(str(error)) from error


class VolumeGate:
    """Final same-mount check used immediately before a destructive decision."""

    def __init__(
        self,
        store: VolumeIdentityStore,
        *,
        media_storage: MediaStorageRegistry | None = None,
        fallback_root: Path | None = None,
    ) -> None:
        self.store = store
        self.media_storage = media_storage
        self.fallback_root = fallback_root

    def verdict(
        self,
        path: Path | str,
        *,
        presence: Callable[[Path | str], str] | None = None,
    ) -> VolumeVerdict:
        file_path = Path(path)
        root = self.root_for(file_path)
        if root is None:
            if self.media_storage is None:
                return self._local_verdict(file_path, presence or path_presence)
            return "unavailable"
        try:
            identity = self.store.get(root)
        except VolumeStoreUnavailable:
            return "unavailable"
        if self._require_mount(root) and not _mounted(root):
            return "unavailable"
        relative = _relative(file_path, root)
        checker = presence or path_presence
        if relative is None:
            if self.media_storage is None:
                return self._local_verdict(file_path, checker)
            return "unavailable"
        if identity is None:
            try:
                if not root.is_dir():
                    return "unavailable"
            except OSError:
                return "unavailable"
            return self._local_verdict(file_path, checker)
        dir_fd = _open_dir(root)
        if dir_fd is None:
            return "unavailable"
        try:
            return _fd_verdict(dir_fd, relative, identity.token)
        finally:
            os.close(dir_fd)

    def root_for(self, path: Path) -> Path | None:
        if self.media_storage is not None:
            for role in ("recordings", "snapshots", "motion_audits", "clips", "exports"):
                location_id = self.media_storage.location_id_for(path, role)
                if location_id is not None:
                    return self.media_storage._roots[location_id]
            location_id = self.media_storage.location_id_for(path)
            if location_id is not None:
                return self.media_storage._roots[location_id]
        if self.fallback_root is not None:
            try:
                _abspath(path).relative_to(_abspath(self.fallback_root))
            except ValueError:
                return None
            return self.fallback_root
        return None

    def _require_mount(self, root: Path) -> bool:
        if self.media_storage is None:
            return False
        root_abs = _abspath(root)
        for location in self.media_storage._locations.values():
            if _abspath(location.path) == root_abs:
                return bool(location.require_mount)
        return False

    @staticmethod
    def _local_verdict(path: Path, presence: Callable[[Path | str], str]) -> VolumeVerdict:
        presence_value = presence(path)
        if presence_value == "present":
            return "present"
        if presence_value == "missing":
            return "confirmed_absent"
        return "unknown"


def _fd_verdict(dir_fd: int, relative: str, token: str) -> VolumeVerdict:
    if _read_witness(dir_fd) != token:
        return "unavailable"
    presence = _stat_fd(dir_fd, relative)
    hook = before_final_witness.get()
    if hook is not None:
        hook(dir_fd)
    if _read_witness(dir_fd) != token:
        return "unavailable"
    if presence == "missing":
        return "confirmed_absent"
    if presence == "present":
        return "present"
    return "unknown"
