"""Advisory flock helpers for cursaves.

Two independent locks:

* sqlite write lock — Cursor ``state.vscdb`` writes (process-wide once
  acquired) and the brief window of a consistent live-db snapshot.
* repo lock — local snapshot-tree and backend operations that share
  ``~/.cursaves`` on this machine. Not a distributed lock for Git or S3.

LOCK ORDER:
When both resources are needed, always acquire repo lock before sqlite.
Never acquire repo lock while holding a sqlite write lock.

Canonical: ``repo.lock → sqlite lock``.
Forbidden: ``sqlite lock → repo.lock`` (raises ``LockOrderError``).

Command audit (no path may acquire repo.lock while the sqlite write
lock is held, unless repo.lock was already owned first):

* ``sync`` — stage ahead exports (repo, no sqlite write lock),
  Cursor writes during pull-behind, then ``finish_cursor_writes()``,
  then repo to promote staged snapshots and push.
* ``pull`` — repo (local snapshot-tree / backend pull) then Cursor
  writes; no later repo.
* ``push`` / ``watch`` / ``checkpoint`` — repo first, then temporary
  sqlite (reads). No process-wide write lock.
* ``copy`` / ``repair`` / ``doctor --recover`` / ``migrate`` /
  ``purge`` — Cursor writes only.
* ``delete`` / backend pull+push — repo only.

Both use ``fcntl.flock``. The kernel drops the lock when the process
exits or is killed. The lock *file* is never unlinked: a leftover file
is not a held lock, and unlinking it can split exclusive flocks across
two inodes.
"""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


_DEFAULT_TIMEOUT = 120.0
_RETRY_INTERVAL = 0.2

_sqlite_fd: Optional[int] = None
_write_held = False
_temp_depth = 0
_atexit_registered = False

_repo_fd: Optional[int] = None
_repo_depth = 0


class FileLockTimeout(RuntimeError):
    """Raised when a flock cannot be acquired before the timeout."""


class SqliteWriteLockTimeout(FileLockTimeout):
    """Raised when another process holds the sqlite write lock too long."""


class RepoLockTimeout(FileLockTimeout):
    """Raised when another local process holds the snapshot-repository lock."""


class LockOrderError(RuntimeError):
    """Raised when repo.lock is requested while the sqlite write lock is held."""


def lock_path() -> Path:
    """Return the path of the process-wide sqlite write lock file."""
    override = os.environ.get("CURSAVES_SQLITE_LOCK")
    if override:
        return Path(override)
    return Path.home() / ".config" / "cursaves" / "sqlite-write.lock"


def repo_lock_path() -> Path:
    """Return the path of the local snapshot-repository lock file.

    Kept outside ``~/.cursaves`` so it is never committed.
    """
    override = os.environ.get("CURSAVES_REPO_LOCK")
    if override:
        return Path(override)
    return Path.home() / ".config" / "cursaves" / "repo.lock"


def _timeout_seconds(env_name: str) -> float:
    raw = os.environ.get(env_name)
    if raw is None or raw == "":
        return _DEFAULT_TIMEOUT
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_TIMEOUT


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_meta(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_meta(fd: int) -> None:
    meta = {
        "pid": os.getpid(),
        "started": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "argv": [os.path.basename(sys.argv[0]), *sys.argv[1:6]],
    }
    payload = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, payload)
    os.fsync(fd)


def _holder_description(path: Path) -> str:
    meta = _read_meta(path)
    pid = meta.get("pid")
    argv = meta.get("argv") or []
    started = meta.get("started")
    parts = []
    if isinstance(pid, int):
        state = "alive" if _pid_alive(pid) else "dead"
        parts.append(f"pid {pid} ({state})")
    if argv:
        parts.append(" ".join(str(a) for a in argv))
    if started:
        parts.append(f"since {started}")
    return ", ".join(parts) if parts else "unknown holder"


def _open_lock_file(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)


def _format_seconds(timeout: float) -> str:
    if timeout >= 10:
        return f"{timeout:.0f}s"
    return f"{timeout:g}s"


def _wait_message(label: str, path: Path, timeout: float) -> str:
    return (
        f"Waiting for the {label} (held by {_holder_description(path)}); "
        f"retrying for {_format_seconds(timeout)}..."
    )


def _timeout_message(label: str, path: Path, timeout: float) -> str:
    return (
        f"timed out after {_format_seconds(timeout)} waiting for the {label} "
        f"(held by {_holder_description(path)})"
    )


def _flock_exclusive(
    path: Path,
    timeout: float,
    exc_class: type[FileLockTimeout],
    label: str,
) -> int:
    """Open *path* and take LOCK_EX. Never unlinks the file."""
    fd = _open_lock_file(path)
    deadline = time.monotonic() + timeout
    announced = False
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise exc_class(_timeout_message(label, path, timeout))
            if not announced:
                print(_wait_message(label, path, timeout), file=sys.stderr, flush=True)
                announced = True
            time.sleep(_RETRY_INTERVAL)
        except Exception:
            os.close(fd)
            raise


def _close_flock(fd: Optional[int]) -> None:
    if fd is None:
        return
    try:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.fsync(fd)
        except OSError:
            pass
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
    finally:
        os.close(fd)


def _acquire_sqlite_fd(timeout: Optional[float] = None) -> None:
    global _sqlite_fd
    if _sqlite_fd is not None:
        return
    wait = (
        _timeout_seconds("CURSAVES_SQLITE_LOCK_TIMEOUT")
        if timeout is None
        else timeout
    )
    fd = _flock_exclusive(
        lock_path(), wait, SqliteWriteLockTimeout, "sqlite write lock"
    )
    try:
        _write_meta(fd)
    except Exception:
        _close_flock(fd)
        raise
    _sqlite_fd = fd


def _release_sqlite_fd() -> None:
    global _sqlite_fd
    fd = _sqlite_fd
    _sqlite_fd = None
    _close_flock(fd)


def _ensure_atexit() -> None:
    global _atexit_registered
    if _atexit_registered:
        return
    atexit.register(_release_on_exit)
    _atexit_registered = True


def _release_on_exit() -> None:
    """Always drop flocks on interpreter shutdown. Does not unlink files."""
    global _write_held, _temp_depth, _repo_depth
    _write_held = False
    _temp_depth = 0
    _repo_depth = 0
    _release_sqlite_fd()
    _release_repo_fd()


def is_write_lock_held() -> bool:
    """True if this process holds the process-wide write lock."""
    return _write_held and _sqlite_fd is not None


def acquire_write_lock(timeout: Optional[float] = None) -> None:
    """Acquire the process-wide exclusive write lock (idempotent).

    Held until process exit, ``release_write_lock()``, or crash (kernel
    closes the fd and drops the flock). The lock file is never removed.
    """
    global _write_held
    _acquire_sqlite_fd(timeout)
    _write_held = True
    _ensure_atexit()


def release_write_lock() -> None:
    """Release the process-wide write lock if this process holds it.

    Temporary live-db guards still in progress keep the flock until they
    finish, so a write-lock release cannot unlock under an in-flight copy.
    """
    global _write_held
    if not _write_held:
        return
    _write_held = False
    if _temp_depth == 0:
        _release_sqlite_fd()


@contextmanager
def temporary_lock(timeout: Optional[float] = None) -> Iterator[None]:
    """Exclusive access for a short consistent live-db snapshot.

    No-op when this process already holds the write lock. Otherwise the
    flock is taken only for the with-block so long-lived readers (watch)
    do not block writers for the whole process lifetime.
    """
    global _temp_depth
    if _write_held:
        yield
        return
    _acquire_sqlite_fd(timeout)
    _temp_depth += 1
    try:
        yield
    finally:
        _temp_depth -= 1
        if _temp_depth == 0 and not _write_held:
            _release_sqlite_fd()


def _acquire_repo_fd(timeout: Optional[float] = None) -> None:
    global _repo_fd
    if _repo_fd is not None:
        return
    wait = (
        _timeout_seconds("CURSAVES_REPO_LOCK_TIMEOUT")
        if timeout is None
        else timeout
    )
    fd = _flock_exclusive(
        repo_lock_path(), wait, RepoLockTimeout, "snapshot-repository lock"
    )
    try:
        _write_meta(fd)
    except Exception:
        _close_flock(fd)
        raise
    _repo_fd = fd
    _ensure_atexit()


def _release_repo_fd() -> None:
    global _repo_fd
    fd = _repo_fd
    _repo_fd = None
    _close_flock(fd)


def is_repo_lock_held() -> bool:
    """True if this process currently holds the snapshot-repository lock."""
    return _repo_depth > 0 and _repo_fd is not None


@contextmanager
def repo_lock(timeout: Optional[float] = None) -> Iterator[None]:
    """Serialize local operations that share the ``~/.cursaves`` snapshot tree.

    Reentrant. Never unlinks. This is not a distributed lock for Git or S3.

    Refuses a new acquisition while this process already holds the sqlite
    write lock (or a temporary exclusive sqlite flock). Reentrant
    ``repo_lock()`` after ``repo → sqlite`` is allowed.
    """
    global _repo_depth
    if _repo_depth == 0 and (is_write_lock_held() or _temp_depth > 0):
        raise LockOrderError(
            "lock order violation: repo lock requested while "
            "sqlite write lock is already held"
        )
    if _repo_depth == 0:
        _acquire_repo_fd(timeout)
    _repo_depth += 1
    try:
        yield
    finally:
        _repo_depth -= 1
        if _repo_depth == 0:
            _release_repo_fd()


def lock_status() -> dict[str, Any]:
    """Inspect the sqlite lock without taking the process-wide write lock."""
    path = lock_path()
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "held": False,
        "pid": None,
        "alive": False,
        "argv": [],
        "started": None,
        "holder": None,
    }
    if not path.exists():
        return result

    meta = _read_meta(path)
    pid = meta.get("pid")
    result["pid"] = pid if isinstance(pid, int) else None
    result["argv"] = meta.get("argv") or []
    result["started"] = meta.get("started")
    result["holder"] = _holder_description(path)
    if isinstance(pid, int):
        result["alive"] = _pid_alive(pid)

    fd = os.open(str(path), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        result["held"] = False
    except BlockingIOError:
        result["held"] = True
    finally:
        os.close(fd)
    return result


def reset_for_tests() -> None:
    """Drop in-process lock state. Tests only. Does not unlink lock files."""
    global _write_held, _temp_depth, _repo_depth
    _write_held = False
    _temp_depth = 0
    _repo_depth = 0
    _release_sqlite_fd()
    _release_repo_fd()
