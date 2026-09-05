"""Safe SQLite reader/writer for Cursor's state.vscdb databases."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import warnings
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from . import dblock

_write_owners: set["CursorDB"] = set()

LEASE_KINDS = ("read", "pull", "ahead")
READ_MODE_ENV = "CURSAVES_READ_MODE"
SNAPSHOT_ROOT_ENV = "CURSAVES_SNAPSHOT_ROOT"


class _LiveUnavailable(Exception):
    """Live WAL snapshot is not available; fall back to Online Backup."""


class TempLease:
    """Directory under the snapshot root, owned by an exclusive ``.lease`` flock."""

    def __init__(self, kind: str, directory: Path, fd: int):
        if kind not in LEASE_KINDS:
            raise ValueError(f"unknown lease kind: {kind}")
        self.kind = kind
        self.path = directory
        self._fd: Optional[int] = fd
        self._released = False

    def release(self) -> None:
        """Remove the directory, then drop the flock. Idempotent."""
        if self._released:
            return
        self._released = True
        fd = self._fd
        self._fd = None
        try:
            _remove_lease_directory(self.path)
        finally:
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                try:
                    os.close(fd)
                except OSError:
                    pass


def snapshot_root() -> Path:
    """Root for leased temporary snapshots. Never glob ``/tmp/cursaves-*``."""
    override = os.environ.get(SNAPSHOT_ROOT_ENV)
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "cursaves-snapshots"


def _remove_lease_directory(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path)
    except OSError as exc:
        warnings.warn(
            f"could not remove temporary snapshot {path}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


_manager_thread_lock = threading.RLock()
_manager_fd: Optional[int] = None
_manager_depth = 0


def _manager_lock_path(root: Path) -> Path:
    return root / ".manager.lock"


def _acquire_manager_lock(root: Path) -> None:
    global _manager_fd, _manager_depth
    _manager_thread_lock.acquire()
    if _manager_depth > 0:
        _manager_depth += 1
        return
    fd = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(_manager_lock_path(root)), os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
    except BaseException:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        _manager_thread_lock.release()
        raise
    _manager_fd = fd
    _manager_depth = 1


def _release_manager_lock() -> None:
    global _manager_fd, _manager_depth
    if _manager_depth <= 0:
        return
    _manager_depth -= 1
    try:
        if _manager_depth > 0:
            return
        fd = _manager_fd
        _manager_fd = None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass
    finally:
        _manager_thread_lock.release()


class snapshot_manager_lock:
    """Exclusive lock for creating and reaping leased snapshot directories."""

    def __init__(self, root: Optional[Path] = None):
        self._root = root

    def __enter__(self) -> "snapshot_manager_lock":
        _acquire_manager_lock(self._root if self._root is not None else snapshot_root())
        return self

    def __exit__(self, *args) -> None:
        _release_manager_lock()


def _is_managed_kind_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name.startswith("."):
        return False
    return any(path.name.startswith(f"{kind}-") for kind in LEASE_KINDS)


def reap_orphaned_leases(root: Optional[Path] = None) -> list[Path]:
    """Remove leased directories whose owner process is gone.

    Only children of the snapshot root that match ``read-`` / ``pull-`` /
    ``ahead-`` are considered. A directory whose lease is still locked is
    left untouched. A managed directory without ``.lease`` is an incomplete
    create (crash between mkdtemp and lease) and is removed under the
    manager lock.
    """
    base = root if root is not None else snapshot_root()
    with snapshot_manager_lock(base):
        return _reap_orphaned_leases_locked(base)


def _reap_orphaned_leases_locked(base: Path) -> list[Path]:
    removed: list[Path] = []
    try:
        if not base.is_dir():
            return removed
        children = list(base.iterdir())
    except OSError:
        return removed
    for child in children:
        if not _is_managed_kind_dir(child):
            continue
        lease = child / ".lease"
        if not lease.is_file():
            _remove_lease_directory(child)
            removed.append(child)
            continue
        fd = None
        try:
            fd = os.open(str(lease), os.O_RDWR)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            continue
        try:
            os.close(fd)
        except OSError:
            pass
        _remove_lease_directory(child)
        removed.append(child)
    return removed


def acquire_lease(kind: str) -> TempLease:
    """Create a leased directory under the snapshot-root manager lock."""
    if kind not in LEASE_KINDS:
        raise ValueError(f"unknown lease kind: {kind}")
    root = snapshot_root()
    with snapshot_manager_lock(root):
        _reap_orphaned_leases_locked(root)
        tmp = Path(tempfile.mkdtemp(prefix=f"{kind}-", dir=str(root)))
        lease_path = tmp / ".lease"
        fd = None
        try:
            fd = os.open(str(lease_path), os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode("ascii"))
            os.fsync(fd)
        except BaseException:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            _remove_lease_directory(tmp)
            raise
        return TempLease(kind, tmp, fd)


def write_connections_open() -> bool:
    """True if any CursorDB still has an open write connection."""
    return bool(_write_owners)


def close_all_write_connections() -> None:
    """Close every tracked Cursor write connection."""
    for owner in list(_write_owners):
        owner.close_write()


def finish_cursor_writes() -> None:
    """Close write connections, then drop the process-wide sqlite write lock.

    Call this after a Cursor-write phase and before any ``repo_lock()``
    acquisition. Does not release while a write connection is still open.
    """
    close_all_write_connections()
    if write_connections_open():
        raise RuntimeError(
            "cannot release sqlite write lock while write connections are open"
        )
    dblock.release_write_lock()


def reset_write_tracking_for_tests() -> None:
    """Tests only."""
    global _manager_fd, _manager_depth
    close_all_write_connections()
    while _manager_depth > 0:
        _release_manager_lock()


def sqlite_file_uri(path: Path, *, mode: str = "ro") -> str:
    """Build a sqlite URI for *path* (absolute, query-safe)."""
    resolved = path.resolve()
    quoted = quote(resolved.as_posix(), safe="/")
    return f"file:{quoted}?mode={mode}"


def _db_scope(path: Path) -> str:
    from . import paths

    try:
        if path.resolve() == paths.get_global_db_path().resolve():
            return "global"
    except OSError:
        pass
    return "workspace"


def _note_snapshot(kind: str, scope: str) -> None:
    from . import syncstate

    if kind != "read":
        return
    if scope == "global":
        syncstate._counts.read_copy_global += 1
        syncstate._counts.sqlite_backups += 1
    else:
        syncstate._counts.read_copy_workspace += 1


def snapshot_live_db(
    src_path: Path,
    dst_path: Path,
    *,
    kind: str = "read",
    scope: str = "unknown",
) -> None:
    """Create a standalone consistent snapshot via the Online Backup API.

    The destination is a self-contained database: WAL/SHM sidecars do not
    need to be copied. SQLite coordinates with other connections (including
    Cursor) while pages are copied.

    *kind* is ``read`` (temporary consistent view) or ``safety`` (permanent
    pre-write backup). Counters for safety *imports* stay in the importer;
    this primitive counts the copy itself so tests can distinguish the two.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    src = None
    dst = None
    try:
        src = sqlite3.connect(sqlite_file_uri(src_path), uri=True, timeout=30.0)
        dst = sqlite3.connect(str(dst_path), timeout=30.0)
        src.backup(dst)
    except Exception:
        if dst is not None:
            try:
                dst.close()
            except sqlite3.Error:
                pass
            dst = None
        if dst_path.exists():
            dst_path.unlink(missing_ok=True)
        raise
    finally:
        if dst is not None:
            dst.close()
        if src is not None:
            src.close()
    if scope == "unknown":
        scope = _db_scope(src_path)
    _note_snapshot(kind, scope)


def preferred_read_mode() -> str:
    """``auto`` (default), ``live``, or ``backup``."""
    raw = os.environ.get(READ_MODE_ENV, "auto").strip().lower()
    if raw in ("auto", "live", "backup"):
        return raw
    return "auto"


class ReadEpoch:
    """One consistent read-only SQLite view for a single command phase.

    Not a process-global cache keyed on ``db_path``. Duration is the
    read-only phase: close before any Cursor write.
    """

    def __init__(self, db_path: Path, *, prefer_live: Optional[bool] = None):
        self.db_path = Path(db_path)
        self.mode: Optional[str] = None
        self._prefer_live = prefer_live
        self._conn: Optional[sqlite3.Connection] = None
        self._lease: Optional[TempLease] = None
        self._closed = False

    def __enter__(self) -> "ReadEpoch":
        reap_orphaned_leases()
        if self._closed:
            raise RuntimeError("ReadEpoch.close() has already been called")
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        try:
            if self._should_try_live():
                try:
                    self._open_live()
                    return self
                except _LiveUnavailable:
                    self._cleanup_partial()
            self._open_backup()
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *args) -> None:
        self.close()

    def _should_try_live(self) -> bool:
        if self._prefer_live is False:
            return False
        if self._prefer_live is True:
            return True
        return preferred_read_mode() != "backup"

    def _open_live(self) -> None:
        from . import syncstate

        uri = sqlite_file_uri(self.db_path, mode="ro")
        conn = None
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=30.0)
            conn.isolation_level = None
            conn.execute("PRAGMA query_only=ON")
            journal = conn.execute("PRAGMA journal_mode").fetchone()
            mode = str(journal[0]).lower() if journal else ""
            if mode != "wal":
                raise _LiveUnavailable(f"journal_mode={mode}")
            conn.execute("BEGIN")
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
        except _LiveUnavailable:
            if conn is not None:
                conn.close()
            raise
        except Exception as exc:
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            raise _LiveUnavailable(str(exc)) from exc
        self._conn = conn
        self.mode = "live"
        syncstate._counts.live_epochs += 1

    def _open_backup(self) -> None:
        from . import syncstate

        self._lease = acquire_lease("read")
        tmp_db = self._lease.path / "state.vscdb"
        with dblock.temporary_lock():
            snapshot_live_db(
                self.db_path,
                tmp_db,
                kind="read",
                scope=_db_scope(self.db_path),
            )
        self._conn = sqlite3.connect(str(tmp_db))
        self.mode = "backup"
        syncstate._counts.backup_epochs += 1

    def _cleanup_partial(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None
        if self._lease is not None:
            self._lease.release()
            self._lease = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ReadEpoch is not open")
        return self._conn

    @property
    def tmp_db_path(self) -> Optional[Path]:
        if self._lease is None:
            return None
        return self._lease.path / "state.vscdb"

    def close(self) -> None:
        """Close the connection and release any backup lease. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None
        if self._lease is not None:
            self._lease.release()
            self._lease = None


class CursorDB:
    """Safe interface to a Cursor state.vscdb database.

    Reads use a ``ReadEpoch`` (live WAL transaction or Online Backup copy).
    Pass a shared *read_epoch* to reuse one consistent view. Writes operate
    on the original file and require Cursor to be closed.
    """

    def __init__(self, db_path: Path, *, read_epoch: Optional[ReadEpoch] = None):
        self.db_path = db_path
        self.autocommit = True
        self._shared_epoch = read_epoch
        self._owned_epoch: Optional[ReadEpoch] = None
        self._tmp_path: Optional[Path] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._write_conn: Optional[sqlite3.Connection] = None

    def _ensure_read_copy(self) -> sqlite3.Connection:
        """Return the read-epoch connection, opening an owned epoch if needed."""
        if self._shared_epoch is not None:
            return self._shared_epoch.connection
        if self._owned_epoch is None:
            if not self.db_path.exists():
                raise FileNotFoundError(f"Database not found: {self.db_path}")
            epoch = ReadEpoch(self.db_path)
            epoch.__enter__()
            self._owned_epoch = epoch
            self._tmp_path = epoch.tmp_db_path
            self._conn = epoch.connection
        return self._owned_epoch.connection

    def close(self):
        """Close write connections and any owned read epoch."""
        self._conn = None
        self.close_write()
        if self._owned_epoch is not None:
            self._owned_epoch.close()
            self._owned_epoch = None
        self._tmp_path = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Read operations (on the read epoch) ─────────────────────────

    def _reader_conn(self) -> sqlite3.Connection:
        """Prefer the live write connection during a batched import."""
        if not self.autocommit and self._write_conn is not None:
            return self._write_conn
        return self._ensure_read_copy()

    def get_item(self, key: str, table: str = "ItemTable") -> Optional[str]:
        """Get a value from the key-value store."""
        conn = self._reader_conn()
        try:
            row = conn.execute(
                f"SELECT value FROM {table} WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            val = row[0]
            if isinstance(val, bytes):
                return val.decode("utf-8", errors="replace")
            return val
        except sqlite3.OperationalError:
            return None

    def get_item_binary(self, key: str, table: str = "ItemTable") -> Optional[bytes]:
        """Get a raw binary value from the key-value store."""
        conn = self._reader_conn()
        try:
            row = conn.execute(
                f"SELECT value FROM {table} WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            val = row[0]
            if isinstance(val, str):
                return val.encode("utf-8")
            return val
        except sqlite3.OperationalError:
            return None

    def get_disk_kv(self, key: str) -> Optional[str]:
        """Get a value from the cursorDiskKV table."""
        return self.get_item(key, table="cursorDiskKV")

    def list_keys(self, prefix: str = "", table: str = "cursorDiskKV") -> list[str]:
        """List all keys in a table, optionally filtered by prefix."""
        conn = self._reader_conn()
        try:
            if prefix:
                rows = conn.execute(
                    f"SELECT key FROM {table} WHERE key LIKE ?", (prefix + "%",)
                ).fetchall()
            else:
                rows = conn.execute(f"SELECT key FROM {table}").fetchall()
            return [r[0] for r in rows]
        except sqlite3.OperationalError:
            return []

    def count_keys_by_chat_prefix(
        self, key_type: str, table: str = "cursorDiskKV"
    ) -> dict[str, int]:
        """Count keys grouped by chat ID for a given key type prefix.

        For example, key_type="bubbleId" counts all keys like
        "bubbleId:<uuid>:..." and returns {<uuid>: count, ...}.

        Uses a single SQL query — efficient even on large databases.
        """
        conn = self._reader_conn()
        result: dict[str, int] = {}
        try:
            prefix = key_type + ":"
            rows = conn.execute(
                f"""SELECT SUBSTR(key, {len(prefix) + 1}, 36) AS cid, COUNT(*)
                    FROM {table}
                    WHERE key LIKE ?
                    GROUP BY cid""",
                (prefix + "%",),
            ).fetchall()
            for cid, count in rows:
                if cid and len(cid) == 36:
                    result[cid] = count
        except sqlite3.OperationalError:
            pass
        return result

    def get_json(self, key: str, table: str = "cursorDiskKV") -> Optional[Any]:
        """Get and parse a JSON value."""
        raw = self.get_item(key, table=table)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    # ── Write operations (on original file) ─────────────────────────

    def close_write(self) -> None:
        """Close the write connection on the original database, if any."""
        conn = self._write_conn
        if conn is not None:
            conn.close()
            self._write_conn = None
        _write_owners.discard(self)

    def _get_write_conn(self) -> sqlite3.Connection:
        """Get or create a connection for write operations on the ORIGINAL database."""
        dblock.acquire_write_lock()
        if self._write_conn is None:
            self._write_conn = sqlite3.connect(str(self.db_path))
            _write_owners.add(self)
            from . import syncstate

            syncstate._counts.cursor_write_connections += 1
            syncstate._counts.write_connections_opened += 1
        return self._write_conn

    def enable_batch_writes(self) -> None:
        """Disable per-call commits so a caller can use savepoints."""
        self.autocommit = False
        conn = self._get_write_conn()
        conn.isolation_level = None

    def begin(self) -> None:
        self._get_write_conn().execute("BEGIN")

    def commit_write(self) -> None:
        self._get_write_conn().execute("COMMIT")

    def rollback_write(self) -> None:
        self._get_write_conn().execute("ROLLBACK")

    def savepoint(self, name: str) -> None:
        self._get_write_conn().execute(f"SAVEPOINT {name}")

    def release_savepoint(self, name: str) -> None:
        self._get_write_conn().execute(f"RELEASE SAVEPOINT {name}")

    def rollback_to_savepoint(self, name: str) -> None:
        self._get_write_conn().execute(f"ROLLBACK TO SAVEPOINT {name}")

    def write_item(self, key: str, value: str, table: str = "ItemTable"):
        """Write a value to the key-value store on the ORIGINAL database.

        This operates directly on the original file, not the temp copy.
        Caller must ensure Cursor is not running.
        """
        conn = self._get_write_conn()
        conn.execute(
            f"INSERT OR REPLACE INTO {table} (key, value) VALUES (?, ?)",
            (key, value),
        )
        if self.autocommit:
            conn.commit()

    def write_disk_kv(self, key: str, value: str):
        """Write a value to cursorDiskKV on the ORIGINAL database."""
        self.write_item(key, value, table="cursorDiskKV")

    def write_json(self, key: str, data: Any, table: str = "cursorDiskKV"):
        """Write a JSON value to the ORIGINAL database."""
        self.write_item(key, json.dumps(data, separators=(",", ":")), table=table)

    def write_batch(self, items: list[tuple[str, str]], table: str = "cursorDiskKV"):
        """Write multiple key-value pairs in a single transaction.

        Much faster than calling write_item() in a loop -- uses one
        connection and one transaction for all items.
        """
        conn = self._get_write_conn()
        if self.autocommit:
            conn.execute("BEGIN")
            try:
                conn.executemany(
                    f"INSERT OR REPLACE INTO {table} (key, value) VALUES (?, ?)",
                    items,
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            return
        conn.executemany(
            f"INSERT OR REPLACE INTO {table} (key, value) VALUES (?, ?)",
            items,
        )

    def write_json_batch(self, items: list[tuple[str, Any]], table: str = "cursorDiskKV"):
        """Write multiple JSON key-value pairs in a single transaction."""
        serialized = [
            (key, json.dumps(data, separators=(",", ":")))
            for key, data in items
        ]
        self.write_batch(serialized, table=table)

    def delete_keys(self, keys: list[str], table: str = "cursorDiskKV") -> int:
        """Delete multiple keys in a single transaction on the ORIGINAL database.

        Returns the number of rows deleted.
        """
        if not keys:
            return 0
        conn = self._get_write_conn()
        if self.autocommit:
            conn.execute("BEGIN")
            try:
                total = 0
                for batch_start in range(0, len(keys), 500):
                    batch = keys[batch_start:batch_start + 500]
                    placeholders = ",".join("?" for _ in batch)
                    cur = conn.execute(
                        f"DELETE FROM {table} WHERE key IN ({placeholders})", batch
                    )
                    total += cur.rowcount
                conn.execute("COMMIT")
                return total
            except Exception:
                conn.execute("ROLLBACK")
                raise
        total = 0
        for batch_start in range(0, len(keys), 500):
            batch = keys[batch_start:batch_start + 500]
            placeholders = ",".join("?" for _ in batch)
            cur = conn.execute(
                f"DELETE FROM {table} WHERE key IN ({placeholders})", batch
            )
            total += cur.rowcount
        return total

    def delete_keys_by_prefix(self, prefix: str, table: str = "cursorDiskKV") -> int:
        """Delete all keys matching a prefix on the ORIGINAL database.

        Returns the number of rows deleted.
        """
        conn = self._get_write_conn()
        cur = conn.execute(
            f"DELETE FROM {table} WHERE key LIKE ?", (prefix + "%",)
        )
        if self.autocommit:
            conn.commit()
        return cur.rowcount


def backup_db(db_path: Path, keep: int = 2) -> Path:
    """Create a timestamped backup of a database file.

    Keeps only the most recent `keep` backups (default 2) and deletes
    older ones to prevent unbounded disk usage. The global DB can be
    multi-GB, so even a handful of stale backups can fill a disk.

    This is a permanent safety backup, not a temporary read snapshot.

    Returns the path to the new backup.
    """
    from datetime import datetime

    dblock.acquire_write_lock()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.stem}.backup_{timestamp}{db_path.suffix}"
    snapshot_live_db(db_path, backup_path, kind="safety", scope=_db_scope(db_path))

    # Clean up old backups, keeping only the newest `keep`
    pattern = f"{db_path.stem}.backup_*{db_path.suffix}"
    old_backups = sorted(
        db_path.parent.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in old_backups[keep:]:
        stale.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            sidecar = stale.parent / (stale.name + suffix)
            sidecar.unlink(missing_ok=True)

    return backup_path
