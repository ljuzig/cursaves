"""Process-wide sqlite write lock, online backup, and repo lock."""

from __future__ import annotations

import fcntl
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from cursor_saves import cli, db, dblock, importer


@pytest.fixture
def sqlite_lock(tmp_path, monkeypatch):
    lock = tmp_path / "sqlite-write.lock"
    repo = tmp_path / "repo.lock"
    monkeypatch.setenv("CURSAVES_SQLITE_LOCK", str(lock))
    monkeypatch.setenv("CURSAVES_REPO_LOCK", str(repo))
    monkeypatch.setenv("CURSAVES_SQLITE_LOCK_TIMEOUT", "1")
    monkeypatch.setenv("CURSAVES_REPO_LOCK_TIMEOUT", "1")
    dblock.reset_for_tests()
    db.reset_write_tracking_for_tests()
    yield lock
    db.reset_write_tracking_for_tests()
    dblock.reset_for_tests()


def _make_state_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE ItemTable (key TEXT UNIQUE, value BLOB)")
    conn.execute("CREATE TABLE cursorDiskKV (key TEXT UNIQUE, value BLOB)")
    conn.execute("INSERT INTO ItemTable VALUES ('hello', 'world')")
    conn.commit()
    conn.close()
    return path


def _foreign_flock(lock_path: Path):
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def test_write_lock_is_reentrant(sqlite_lock):
    dblock.acquire_write_lock()
    dblock.acquire_write_lock()
    assert dblock.is_write_lock_held()
    dblock.release_write_lock()
    assert not dblock.is_write_lock_held()


def test_release_does_not_unlink_lock_file(sqlite_lock):
    dblock.acquire_write_lock()
    dblock.release_write_lock()
    assert sqlite_lock.exists()
    assert not hasattr(dblock, "unlock")


def test_lock_files_are_created_private(sqlite_lock):
    dblock.acquire_write_lock()
    dblock.release_write_lock()
    with dblock.repo_lock():
        pass
    assert (sqlite_lock.stat().st_mode & 0o777) == 0o600
    assert (dblock.repo_lock_path().stat().st_mode & 0o777) == 0o600


def test_write_meta_failure_releases_sqlite_flock(sqlite_lock, monkeypatch):
    def boom(_fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(dblock, "_write_meta", boom)
    with pytest.raises(OSError, match="simulated fsync failure"):
        dblock.acquire_write_lock()
    assert not dblock.is_write_lock_held()
    other = os.open(str(sqlite_lock), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(other, fcntl.LOCK_UN)
    finally:
        os.close(other)


def test_write_meta_failure_releases_repo_flock(sqlite_lock, monkeypatch):
    def boom(_fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(dblock, "_write_meta", boom)
    with pytest.raises(OSError, match="simulated fsync failure"):
        with dblock.repo_lock():
            pass
    assert not dblock.is_repo_lock_held()
    other = os.open(str(dblock.repo_lock_path()), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(other, fcntl.LOCK_UN)
    finally:
        os.close(other)


def test_write_lock_blocks_other_flock(sqlite_lock):
    dblock.acquire_write_lock()
    other = os.open(str(sqlite_lock), os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(other)
    dblock.release_write_lock()
    other = os.open(str(sqlite_lock), os.O_RDWR)
    try:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(other, fcntl.LOCK_UN)
    finally:
        os.close(other)


def test_temporary_lock_does_not_hold_after_copy(sqlite_lock):
    with dblock.temporary_lock():
        other = os.open(str(sqlite_lock), os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(other)
    other = os.open(str(sqlite_lock), os.O_RDWR)
    try:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(other, fcntl.LOCK_UN)
    finally:
        os.close(other)
    assert not dblock.is_write_lock_held()


def test_temporary_lock_is_noop_when_write_lock_held(sqlite_lock):
    dblock.acquire_write_lock()
    with dblock.temporary_lock():
        assert dblock.is_write_lock_held()
    assert dblock.is_write_lock_held()
    dblock.release_write_lock()


def test_write_lock_times_out_when_foreign_holder(sqlite_lock):
    fd = _foreign_flock(sqlite_lock)
    try:
        with pytest.raises(dblock.SqliteWriteLockTimeout, match="sqlite write lock"):
            dblock.acquire_write_lock(timeout=0.4)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_timeout_message_does_not_suggest_unlock(sqlite_lock):
    fd = _foreign_flock(sqlite_lock)
    try:
        with pytest.raises(dblock.SqliteWriteLockTimeout, match="(?s).*") as exc:
            dblock.acquire_write_lock(timeout=0.3)
        assert "unlock" not in str(exc.value).lower()
        assert "timed out" in str(exc.value).lower()
        assert "held by" in str(exc.value)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_wait_message_printed_while_blocked(sqlite_lock, capsys):
    fd = _foreign_flock(sqlite_lock)
    try:
        with pytest.raises(dblock.SqliteWriteLockTimeout):
            dblock.acquire_write_lock(timeout=0.4)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    err = capsys.readouterr().err
    assert "Waiting for the sqlite write lock" in err
    assert "held by" in err
    assert err.count("Waiting for the sqlite write lock") == 1


def test_no_wait_message_when_lock_is_free(sqlite_lock, capsys):
    dblock.acquire_write_lock()
    dblock.release_write_lock()
    err = capsys.readouterr().err
    assert "Waiting" not in err


def test_repo_wait_message_printed_while_blocked(sqlite_lock, capsys):
    fd = _foreign_flock(dblock.repo_lock_path())
    try:
        with pytest.raises(dblock.RepoLockTimeout):
            with dblock.repo_lock(timeout=0.4):
                pass
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    err = capsys.readouterr().err
    assert "Waiting for the snapshot-repository lock" in err
    assert "held by" in err


def test_write_lock_blocks_other_process(sqlite_lock):
    env = os.environ.copy()
    env["CURSAVES_SQLITE_LOCK"] = str(sqlite_lock)
    env["CURSAVES_SQLITE_LOCK_TIMEOUT"] = "1"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time\n"
            "from cursor_saves import dblock\n"
            "dblock.acquire_write_lock()\n"
            "print('held', flush=True)\n"
            "time.sleep(30)\n",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = holder.stdout.readline()
        assert "held" in line
        with pytest.raises(dblock.SqliteWriteLockTimeout):
            dblock.acquire_write_lock(timeout=0.6)
    finally:
        holder.terminate()
        try:
            holder.wait(timeout=5)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.wait(timeout=5)


def test_write_item_acquires_process_lock(sqlite_lock, tmp_path):
    db_path = _make_state_db(tmp_path / "state.vscdb")
    with db.CursorDB(db_path) as cdb:
        cdb.write_item("hello", "there")
        assert dblock.is_write_lock_held()
    assert dblock.is_write_lock_held()
    dblock.release_write_lock()


def test_backup_db_acquires_process_lock(sqlite_lock, tmp_path):
    db_path = _make_state_db(tmp_path / "state.vscdb")
    backup = db.backup_db(db_path, keep=1)
    assert backup.exists()
    assert dblock.is_write_lock_held()
    dblock.release_write_lock()


def test_read_copy_releases_lock_afterward(sqlite_lock, tmp_path):
    db_path = _make_state_db(tmp_path / "state.vscdb")
    with db.CursorDB(db_path) as cdb:
        assert cdb.get_item("hello") == "world"
        assert not dblock.is_write_lock_held()
    other = os.open(str(sqlite_lock), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(other, fcntl.LOCK_UN)
    finally:
        os.close(other)


def test_readonly_snapshot_does_not_leave_process_lock(sqlite_lock, tmp_path):
    db_path = _make_state_db(tmp_path / "state.vscdb")
    with db.CursorDB(db_path) as cdb:
        cdb.get_item("hello")
    assert not dblock.is_write_lock_held()
    status = dblock.lock_status()
    assert status["held"] is False


def test_init_workspace_db_acquires_process_lock(sqlite_lock, tmp_path):
    path = tmp_path / "ws" / "state.vscdb"
    path.parent.mkdir(parents=True)
    importer._init_workspace_db(path)
    assert path.exists()
    assert dblock.is_write_lock_held()
    dblock.release_write_lock()


def test_online_backup_from_live_wal(sqlite_lock, tmp_path):
    db_path = tmp_path / "state.vscdb"
    writer = sqlite3.connect(str(db_path))
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE ItemTable (key TEXT UNIQUE, value BLOB)")
    writer.execute("CREATE TABLE cursorDiskKV (key TEXT UNIQUE, value BLOB)")
    writer.execute("INSERT INTO ItemTable VALUES ('hello', 'v1')")
    writer.commit()

    reader = sqlite3.connect(str(db_path))
    writer.execute("INSERT OR REPLACE INTO ItemTable VALUES ('hello', 'v2')")
    writer.execute("INSERT INTO ItemTable VALUES ('extra', 'yes')")
    writer.commit()

    dest = tmp_path / "snap.vscdb"
    db.snapshot_live_db(db_path, dest)

    check = sqlite3.connect(str(dest))
    try:
        assert check.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert (
            check.execute("SELECT value FROM ItemTable WHERE key='hello'").fetchone()[0]
            == "v2"
        )
        assert (
            check.execute("SELECT value FROM ItemTable WHERE key='extra'").fetchone()[0]
            == "yes"
        )
    finally:
        check.close()
        reader.close()
        writer.close()

    assert not Path(str(dest) + "-wal").exists()
    assert not Path(str(dest) + "-shm").exists()

    with db.CursorDB(db_path) as cdb:
        assert cdb.get_item("hello") == "v2"
        assert cdb.get_item("extra") == "yes"
    assert not dblock.is_write_lock_held()


def test_repo_lock_is_reentrant_and_released(sqlite_lock):
    with dblock.repo_lock():
        assert dblock.is_repo_lock_held()
        with dblock.repo_lock():
            assert dblock.is_repo_lock_held()
    assert not dblock.is_repo_lock_held()
    assert dblock.repo_lock_path().exists()


def test_repo_lock_blocks_other_process(sqlite_lock, tmp_path):
    repo = tmp_path / "repo.lock"
    env = os.environ.copy()
    env["CURSAVES_REPO_LOCK"] = str(repo)
    env["CURSAVES_REPO_LOCK_TIMEOUT"] = "1"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time\n"
            "from cursor_saves import dblock\n"
            "lock = dblock.repo_lock()\n"
            "lock.__enter__()\n"
            "print('held', flush=True)\n"
            "time.sleep(30)\n",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = holder.stdout.readline()
        assert "held" in line
        with pytest.raises(dblock.RepoLockTimeout):
            with dblock.repo_lock(timeout=0.6):
                pass
    finally:
        holder.terminate()
        try:
            holder.wait(timeout=5)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.wait(timeout=5)


def test_repo_then_temporary_sqlite_lock_allowed(sqlite_lock):
    with dblock.repo_lock():
        with dblock.temporary_lock():
            assert dblock.is_repo_lock_held()
            assert not dblock.is_write_lock_held()


def test_repo_then_sqlite_write_lock_allowed(sqlite_lock):
    with dblock.repo_lock():
        dblock.acquire_write_lock()
        assert dblock.is_write_lock_held()
        with dblock.repo_lock():
            assert dblock.is_repo_lock_held()
        dblock.release_write_lock()


def test_sqlite_write_then_new_repo_lock_is_order_error(sqlite_lock):
    dblock.acquire_write_lock()
    try:
        with pytest.raises(dblock.LockOrderError, match="lock order violation"):
            with dblock.repo_lock():
                pass
    finally:
        dblock.release_write_lock()


def test_subprocess_deadlock_regression(sqlite_lock, tmp_path):
    """Process A takes repo then sqlite; B holds sqlite and must not wait on repo."""
    env = os.environ.copy()
    env["CURSAVES_SQLITE_LOCK"] = str(sqlite_lock)
    env["CURSAVES_REPO_LOCK"] = str(tmp_path / "repo.lock")
    env["CURSAVES_SQLITE_LOCK_TIMEOUT"] = "5"
    env["CURSAVES_REPO_LOCK_TIMEOUT"] = "5"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from cursor_saves import dblock\n"
            "lock = dblock.repo_lock()\n"
            "lock.__enter__()\n"
            "print('held_repo', flush=True)\n"
            "input()\n"
            "dblock.acquire_write_lock()\n"
            "print('held_sqlite', flush=True)\n",
        ],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert "held_repo" in child.stdout.readline()
        dblock.acquire_write_lock()
        with pytest.raises(dblock.LockOrderError):
            with dblock.repo_lock(timeout=0.4):
                pass
        dblock.release_write_lock()
        child.stdin.write("\n")
        child.stdin.flush()
        assert "held_sqlite" in child.stdout.readline()
        child.wait(timeout=5)
        assert child.returncode == 0
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


def test_sync_releases_sqlite_before_repo(sqlite_lock, tmp_path, monkeypatch):
    db_path = _make_state_db(tmp_path / "state.vscdb")
    order: list[str] = []
    real_repo_lock = dblock.repo_lock

    def guarded_repo_lock(*args, **kwargs):
        if dblock.is_write_lock_held():
            raise AssertionError(
                "repo lock requested while sqlite write lock is still held"
            )
        order.append("repo")
        return real_repo_lock(*args, **kwargs)

    monkeypatch.setattr(dblock, "repo_lock", guarded_repo_lock)

    def fake_pull_behind(sync_dir, plan=None):
        with db.CursorDB(db_path) as cdb:
            cdb.write_item("hello", "behind")
            assert db.write_connections_open()
            assert dblock.is_write_lock_held()
        assert not db.write_connections_open()
        assert dblock.is_write_lock_held()
        order.append("cursor_write")
        return 1

    def fake_push_ahead(sync_dir, auto=False, backend=None, plan=None, session=None):
        assert not db.write_connections_open()
        assert not dblock.is_write_lock_held()
        with dblock.repo_lock():
            order.append("ahead_push")
        return 1

    real_finish = db.finish_cursor_writes

    def tracking_finish():
        assert not db.write_connections_open()
        real_finish()
        assert not dblock.is_write_lock_held()
        order.append("released")

    monkeypatch.setattr(db, "finish_cursor_writes", tracking_finish)
    monkeypatch.setattr(cli, "_pull_behind", fake_pull_behind)
    monkeypatch.setattr(cli, "_finish_sync_push", fake_push_ahead)
    monkeypatch.setattr(cli, "_require_sync_repo", lambda: tmp_path)
    monkeypatch.setattr(cli.paths, "get_snapshots_dir", lambda: tmp_path / "snapshots")
    monkeypatch.setattr(cli.paths, "get_sync_dir", lambda: tmp_path)
    monkeypatch.setattr(
        cli,
        "get_backend",
        lambda: type(
            "B",
            (),
            {
                "has_remote": staticmethod(lambda: False),
                "pull": staticmethod(lambda d: True),
                "push": staticmethod(lambda d: True),
            },
        )(),
    )

    cli.cmd_sync(type("Args", (), {})())
    assert order == ["cursor_write", "released", "repo", "ahead_push"]


def test_sync_source_releases_before_push():
    import inspect

    src = inspect.getsource(cli.cmd_sync)
    assert src.index("finish_cursor_writes") < src.index("_finish_sync_push")


def test_command_lock_audit_documented():
    doc = dblock.__doc__ or ""
    for name in (
        "sync",
        "pull",
        "copy",
        "repair",
        "doctor --recover",
        "migrate",
        "purge",
        "watch",
    ):
        assert name in doc


def test_cli_reports_lock_timeout_without_traceback(capsys, monkeypatch):
    def boom(_args):
        raise dblock.SqliteWriteLockTimeout(
            "timed out after 1s waiting for the sqlite write lock "
            "(held by pid 9 (alive))"
        )

    monkeypatch.setattr(cli, "cmd_status", boom)
    monkeypatch.setattr(sys, "argv", ["cursaves", "status"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Error: timed out after 1s waiting for the sqlite write lock" in err
    assert "Traceback" not in err
    assert "unlock" not in err.lower()


def test_cli_reports_lock_order_error_without_traceback(capsys, monkeypatch):
    def boom(_args):
        raise dblock.LockOrderError(
            "lock order violation: repo lock requested while "
            "sqlite write lock is already held"
        )

    monkeypatch.setattr(cli, "cmd_status", boom)
    monkeypatch.setattr(sys, "argv", ["cursaves", "status"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Error: lock order violation" in err
    assert "Traceback" not in err
