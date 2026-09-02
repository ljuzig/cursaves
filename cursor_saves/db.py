"""Safe SQLite reader/writer for Cursor's state.vscdb databases."""

import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from . import dblock

_write_owners: set["CursorDB"] = set()


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
    close_all_write_connections()


def sqlite_file_uri(path: Path, *, mode: str = "ro") -> str:
    """Build a sqlite URI for *path* (absolute, query-safe)."""
    resolved = path.resolve()
    quoted = quote(resolved.as_posix(), safe="/")
    return f"file:{quoted}?mode={mode}"


def snapshot_live_db(src_path: Path, dst_path: Path) -> None:
    """Create a standalone consistent snapshot via the Online Backup API.

    The destination is a self-contained database: WAL/SHM sidecars do not
    need to be copied. SQLite coordinates with other connections (including
    Cursor) while pages are copied.
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


class CursorDB:
    """Safe interface to a Cursor state.vscdb database.

    All reads are performed on a temporary copy of the database to avoid
    locking conflicts with a running Cursor instance. Writes operate on
    the original file and require Cursor to be closed.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._tmp_path: Optional[Path] = None
        self._conn: Optional[sqlite3.Connection] = None

    def _ensure_read_copy(self) -> sqlite3.Connection:
        """Copy the database to a temp file and open a read-only connection."""
        if self._conn is not None:
            return self._conn

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        # Consistent snapshot of the live DB. The cursaves flock only
        # serializes other cursaves writers; sqlite.backup() coordinates
        # with Cursor's own connections, including WAL.
        tmp_dir = tempfile.mkdtemp(prefix="cursaves-")
        tmp_db = Path(tmp_dir) / "state.vscdb"
        with dblock.temporary_lock():
            snapshot_live_db(self.db_path, tmp_db)

        self._tmp_path = tmp_db
        self._conn = sqlite3.connect(str(tmp_db))
        return self._conn

    def close(self):
        """Close connections and clean up temp files."""
        if self._conn:
            self._conn.close()
            self._conn = None
        self.close_write()
        if self._tmp_path:
            shutil.rmtree(self._tmp_path.parent, ignore_errors=True)
            self._tmp_path = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Read operations (on temp copy) ──────────────────────────────

    def get_item(self, key: str, table: str = "ItemTable") -> Optional[str]:
        """Get a value from the key-value store."""
        conn = self._ensure_read_copy()
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
        conn = self._ensure_read_copy()
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
        conn = self._ensure_read_copy()
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
        conn = self._ensure_read_copy()
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
        conn = getattr(self, "_write_conn", None)
        if conn is not None:
            conn.close()
            self._write_conn = None
        _write_owners.discard(self)

    def _get_write_conn(self) -> sqlite3.Connection:
        """Get or create a connection for write operations on the ORIGINAL database."""
        dblock.acquire_write_lock()
        if not hasattr(self, "_write_conn") or self._write_conn is None:
            self._write_conn = sqlite3.connect(str(self.db_path))
            _write_owners.add(self)
        return self._write_conn

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

    def delete_keys_by_prefix(self, prefix: str, table: str = "cursorDiskKV") -> int:
        """Delete all keys matching a prefix on the ORIGINAL database.

        Returns the number of rows deleted.
        """
        conn = self._get_write_conn()
        cur = conn.execute(
            f"DELETE FROM {table} WHERE key LIKE ?", (prefix + "%",)
        )
        conn.commit()
        return cur.rowcount



def backup_db(db_path: Path, keep: int = 2) -> Path:
    """Create a timestamped backup of a database file.

    Keeps only the most recent `keep` backups (default 2) and deletes
    older ones to prevent unbounded disk usage. The global DB can be
    multi-GB, so even a handful of stale backups can fill a disk.

    Returns the path to the new backup.
    """
    from datetime import datetime

    dblock.acquire_write_lock()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.stem}.backup_{timestamp}{db_path.suffix}"
    snapshot_live_db(db_path, backup_path)

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
