"""Stable SQLite read views and leased temporary snapshots (v0.9.11)."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from cursor_saves import cli, db, export, importer, paths, pull, syncstate
from tests.test_syncstate import (
    CID_A,
    CID_B,
    CID_C,
    PROJECT_PATH,
    WS_HASH,
    _backend,
    _commit_env,
    _conversation,
    _init_db,
    _msg,
    _write_local,
    _write_sharded_snapshot,
    _write_snapshot_file,
    _write_workspace,
)
from tests.test_sync_workspace import _args, _use_workspaces, _ws

pytest_plugins = ["tests.test_syncstate"]


HASH_A = "aaaabbbb111111111111111111111111"
HASH_B = "aaaabbbb222222222222222222222222"
HASH_C = "ccccdddd333333333333333333333333"


def _managed_lease_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [
        p
        for p in root.iterdir()
        if p.name.startswith(("read-", "pull-", "ahead-"))
    ]


def _enable_wal(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    conn.commit()
    conn.close()
    assert str(mode).lower() == "wal"


def _write_ws_json(ws_dir: Path, folder: str, host: str | None = None) -> None:
    ws_dir.mkdir(parents=True, exist_ok=True)
    if host:
        payload = {"folder": f"vscode-remote://ssh-remote+{host}{folder}"}
    else:
        payload = {"folder": f"file://{folder}"}
    (ws_dir / "workspace.json").write_text(json.dumps(payload))


def _count_global_epochs(monkeypatch, global_db: Path):
    n = {"n": 0}
    real = db.ReadEpoch.__enter__

    def wrapped(self):
        try:
            if Path(self.db_path).resolve() == Path(global_db).resolve():
                n["n"] += 1
        except OSError:
            pass
        return real(self)

    monkeypatch.setattr(db.ReadEpoch, "__enter__", wrapped)
    return n


def _count_snapshots(monkeypatch):
    calls: list[tuple[str, str]] = []
    real = db.snapshot_live_db

    def wrapped(src, dst, *, kind="read", scope="unknown"):
        calls.append((kind, scope))
        return real(src, dst, kind=kind, scope=scope)

    monkeypatch.setattr(db, "snapshot_live_db", wrapped)
    return calls


def test_digest_and_cache_versions_unchanged():
    assert syncstate.SEMANTIC_DIGEST_VERSION == 4
    assert syncstate._CACHE_VERSION == 5


def test_hash_selector_full_and_unique_prefix(tmp_path, monkeypatch):
    storage = tmp_path / "workspaceStorage"
    monkeypatch.setattr(paths, "get_workspace_storage_dir", lambda: storage)
    a = storage / HASH_A
    c = storage / HASH_C
    _write_ws_json(a, "/home/user/alpha")
    _write_ws_json(c, "/home/user/gamma")

    full = paths.resolve_workspace(HASH_A)
    assert full is not None
    assert full["workspace_dir"] == a
    prefix = paths.resolve_workspace(HASH_C[:8])
    assert prefix is not None
    assert prefix["workspace_dir"] == c


def test_ambiguous_hash_prefix_is_rejected(tmp_path, monkeypatch):
    storage = tmp_path / "workspaceStorage"
    monkeypatch.setattr(paths, "get_workspace_storage_dir", lambda: storage)
    _write_ws_json(storage / HASH_A, "/home/user/alpha")
    _write_ws_json(storage / HASH_B, "/home/user/beta")
    with pytest.raises(paths.AmbiguousWorkspaceError) as exc:
        paths.resolve_workspace("aaaabbbb")
    assert len(exc.value.matches) == 2


def test_numeric_and_path_selectors_keep_conversation_order(tmp_path, monkeypatch):
    storage = tmp_path / "workspaceStorage"
    monkeypatch.setattr(paths, "get_workspace_storage_dir", lambda: storage)
    older = storage / HASH_C
    newer = storage / HASH_A
    _write_ws_json(older, "/home/user/gamma")
    _write_ws_json(newer, "/home/user/alpha")
    snap = _conversation([_msg(1, "hi")], composer_id=CID_A, name="A")
    _write_workspace(older, [snap])
    _write_workspace(newer, [snap])
    os.utime(older / "state.vscdb", (1, 1))
    os.utime(newer / "state.vscdb", (2, 2))

    listed = paths.list_workspaces_with_conversations()
    assert [ws["workspace_dir"] for ws in listed] == [newer, older]
    assert paths.resolve_workspace("1")["workspace_dir"] == newer
    assert paths.resolve_workspace("2")["workspace_dir"] == older
    assert paths.resolve_workspace("alpha")["workspace_dir"] == newer


def test_hash_resolve_does_not_enumerate_conversations(tmp_path, monkeypatch):
    storage = tmp_path / "workspaceStorage"
    monkeypatch.setattr(paths, "get_workspace_storage_dir", lambda: storage)
    ws = storage / HASH_A
    _write_ws_json(ws, "/home/user/alpha")
    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("must not enumerate conversations")

    monkeypatch.setattr(paths, "list_workspaces_with_conversations", boom)
    found = paths.resolve_workspace(HASH_A)
    assert found is not None
    assert found["workspace_dir"] == ws
    assert called["n"] == 0


def test_ssh_workspace_json_identity_unchanged(tmp_path, monkeypatch):
    storage = tmp_path / "workspaceStorage"
    monkeypatch.setattr(paths, "get_workspace_storage_dir", lambda: storage)
    ws = storage / HASH_A
    _write_ws_json(ws, "/home/lju/nixos", host="MindLoop1")
    found = paths.resolve_workspace(HASH_A)
    assert found is not None
    assert found["type"] == "ssh"
    assert found["host"] == "MindLoop1"
    assert found["path"] == "/home/lju/nixos"


def test_status_uses_one_global_epoch(sync_env, monkeypatch):
    _commit_env(
        sync_env,
        [_conversation([_msg(1, "A")], composer_id=CID_A, name="A")],
        [_conversation([_msg(1, "A")], composer_id=CID_A, name="A")],
    )
    monkeypatch.setattr(cli, "_ensure_synced", lambda: None)
    _backend(monkeypatch)
    epochs = _count_global_epochs(monkeypatch, sync_env["global_db"])
    copies = _count_snapshots(monkeypatch)
    syncstate.reset_op_counts()
    cli.cmd_status(_args(workspace="1"))
    assert epochs["n"] == 1
    assert sum(1 for kind, scope in copies if kind == "read" and scope == "global") == 1
    assert syncstate.op_counts().sqlite_backups == 1


def test_list_and_status_skip_global_copy_on_wal(sync_env, monkeypatch):
    snap = _conversation([_msg(1, "A")], composer_id=CID_A, name="A")
    _commit_env(sync_env, [snap], [snap])
    _enable_wal(sync_env["global_db"])
    monkeypatch.setattr(cli, "_ensure_synced", lambda: None)
    _backend(monkeypatch)
    copies = _count_snapshots(monkeypatch)
    syncstate.reset_op_counts()
    cli.cmd_list(
        type("Args", (), {"workspace": "1", "project": None, "json": False})()
    )
    assert syncstate.op_counts().read_copy_global == 0
    assert syncstate.op_counts().live_epochs == 1
    assert all(kind != "read" or scope != "global" for kind, scope in copies)

    copies.clear()
    syncstate.reset_op_counts()
    cli.cmd_status(_args(workspace="1"))
    assert syncstate.op_counts().read_copy_global == 0
    assert syncstate.op_counts().sqlite_backups == 0
    assert syncstate.op_counts().live_epochs == 1


def test_live_view_stable_across_concurrent_commit(tmp_path):
    path = tmp_path / "state.vscdb"
    writer = sqlite3.connect(str(path))
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE ItemTable (key TEXT UNIQUE, value BLOB)")
    writer.execute("CREATE TABLE cursorDiskKV (key TEXT UNIQUE, value BLOB)")
    writer.execute("INSERT INTO ItemTable VALUES ('hello', 'v1')")
    writer.commit()

    with db.ReadEpoch(path) as epoch:
        assert epoch.mode == "live"
        assert (
            epoch.connection.execute(
                "SELECT value FROM ItemTable WHERE key='hello'"
            ).fetchone()[0]
            == "v1"
        )
        writer.execute("INSERT OR REPLACE INTO ItemTable VALUES ('hello', 'v2')")
        writer.commit()
        assert (
            epoch.connection.execute(
                "SELECT value FROM ItemTable WHERE key='hello'"
            ).fetchone()[0]
            == "v1"
        )
    writer.close()


def test_wal_writer_not_blocked_during_live_read(tmp_path):
    path = tmp_path / "state.vscdb"
    setup = sqlite3.connect(str(path))
    setup.execute("PRAGMA journal_mode=WAL")
    setup.execute("CREATE TABLE ItemTable (key TEXT UNIQUE, value BLOB)")
    setup.execute("CREATE TABLE cursorDiskKV (key TEXT UNIQUE, value BLOB)")
    setup.execute("INSERT INTO ItemTable VALUES ('hello', 'v1')")
    setup.commit()
    setup.close()

    with db.ReadEpoch(path) as epoch:
        assert epoch.mode == "live"
        writer = sqlite3.connect(str(path), timeout=1.0)
        try:
            writer.execute("INSERT OR REPLACE INTO ItemTable VALUES ('hello', 'v2')")
            writer.commit()
        finally:
            writer.close()


def test_non_wal_falls_back_to_backup(tmp_path):
    path = tmp_path / "state.vscdb"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE ItemTable (key TEXT UNIQUE, value BLOB)")
    conn.execute("CREATE TABLE cursorDiskKV (key TEXT UNIQUE, value BLOB)")
    conn.execute("INSERT INTO ItemTable VALUES ('hello', 'v1')")
    conn.commit()
    conn.close()
    with db.ReadEpoch(path) as epoch:
        assert epoch.mode == "backup"
        assert epoch.tmp_db_path is not None
        copied = epoch.tmp_db_path
        assert copied.exists()
    assert not copied.exists()


def test_read_mode_backup_opt_out_on_wal(tmp_path, monkeypatch):
    path = tmp_path / "state.vscdb"
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE ItemTable (key TEXT UNIQUE, value BLOB)")
    conn.execute("CREATE TABLE cursorDiskKV (key TEXT UNIQUE, value BLOB)")
    conn.commit()
    conn.close()
    monkeypatch.setenv("CURSAVES_READ_MODE", "backup")
    with db.ReadEpoch(path) as epoch:
        assert epoch.mode == "backup"


def test_read_error_is_unknown_not_empty(sync_env):
    snap = _conversation([_msg(1, "A")], composer_id=CID_A, name="A")
    _commit_env(sync_env, [snap], [snap])
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()

        def boom(_cid):
            raise syncstate.ClassifyError("unreadable")

        session.composer_cell = boom  # type: ignore[method-assign]
        rel = syncstate.classify_conversation(
            session, index, CID_A, project_identifier="project"
        )
    assert rel == syncstate.SyncRelation.UNKNOWN


def test_targeted_inventory_matches_full_scan(sync_env):
    snaps = [
        _conversation([_msg(1, f"t{i}")], composer_id=cid, name=f"C{i}")
        for i, cid in enumerate((CID_A, CID_B, CID_C))
    ]
    _commit_env(sync_env, snaps, snaps)
    with syncstate.SyncReadSession() as full:
        full.prepare_inventory(None)
        full_fp = {cid: full.raw_fingerprint(cid) for cid in (CID_A, CID_B, CID_C)}
        assert full._inventory_complete
    with syncstate.SyncReadSession() as targeted:
        targeted.prepare_inventory({CID_A, CID_C})
        assert targeted.raw_fingerprint(CID_A) == full_fp[CID_A]
        assert targeted.raw_fingerprint(CID_C) == full_fp[CID_C]
        assert not targeted._inventory_complete
        assert CID_B not in targeted._row_fp


def test_unsafe_sync_does_not_stage_or_write(sync_env, monkeypatch):
    remote = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="D")
    local = _conversation([_msg(1, "A"), _msg(2, "X")], composer_id=CID_A, name="D")
    _commit_env(sync_env, [local], [remote], digest=False)
    leases = {"ahead": 0, "pull": 0}
    real_lease = db.acquire_lease

    def wrapped(kind):
        if kind in leases:
            leases[kind] += 1
        return real_lease(kind)

    monkeypatch.setattr(db, "acquire_lease", wrapped)
    monkeypatch.setattr(
        db, "backup_db", lambda *a, **k: (_ for _ in ()).throw(AssertionError("backup"))
    )
    monkeypatch.setattr(
        cli,
        "import_snapshot",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("import")),
    )
    _backend(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_sync(type("Args", (), {"force": False})())
    assert exc.value.code == 1
    assert leases["ahead"] == 0
    assert leases["pull"] == 0


def test_ahead_export_uses_preflight_view(sync_env, monkeypatch):
    remote = _conversation([_msg(1, "A")], composer_id=CID_A, name="Ahead")
    local = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="Ahead")
    _commit_env(sync_env, [local], [remote], digest=False)
    _enable_wal(sync_env["global_db"])
    seen = []

    def capture(snapshot, dest):
        seen.append(snapshot)
        return export._save_snapshot_unlocked(snapshot, dest)

    monkeypatch.setattr(export, "save_snapshot", capture)
    _backend(monkeypatch)
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
        gconn = sqlite3.connect(str(sync_env["global_db"]))
        changed = _conversation(
            [_msg(1, "A"), _msg(2, "CHANGED")], composer_id=CID_A, name="Ahead"
        )
        _write_local(gconn, changed)
        gconn.commit()
        gconn.close()
        staged = syncstate.stage_ahead_exports(plan, session)
    assert staged is not None
    assert seen
    payload = json.dumps(seen[0].get("bubbleEntries") or seen[0])
    assert "CHANGED" not in payload
    assert "B" in payload
    staged.discard()


def test_local_guard_detects_change_between_preflight_and_import(sync_env):
    remote = _conversation(
        [_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="Behind"
    )
    local = _conversation([_msg(1, "A")], composer_id=CID_A, name="Behind")
    _commit_env(sync_env, [local], [remote], digest=False)
    with syncstate.SyncReadSession() as session:
        index = pull.scoped_snapshot_index(PROJECT_PATH)
        target = {
            "path": PROJECT_PATH,
            "workspace_dir": sync_env["ws_dir"],
            "host": None,
            "type": "local",
        }
        plan = syncstate.build_pull_plan(session, index, target)
        item = plan.import_candidates[0]
        assert item.local_guard is not None
        gconn = sqlite3.connect(str(sync_env["global_db"]))
        _write_local(
            gconn,
            _conversation([_msg(1, "CHANGED")], composer_id=CID_A, name="Behind"),
        )
        gconn.commit()
        gconn.close()
        with db.CursorDB(sync_env["global_db"]) as gdb:
            with db.CursorDB(sync_env["ws_dir"] / "state.vscdb") as wdb:
                assert not syncstate.local_guard_still_matches(item, gdb, wdb)


def test_lease_cleanup_on_success_and_interrupt(tmp_path, monkeypatch):
    path = tmp_path / "state.vscdb"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE ItemTable (key TEXT UNIQUE, value BLOB)")
    conn.execute("CREATE TABLE cursorDiskKV (key TEXT UNIQUE, value BLOB)")
    conn.commit()
    conn.close()
    root = Path(os.environ["CURSAVES_SNAPSHOT_ROOT"])
    monkeypatch.setenv("CURSAVES_READ_MODE", "backup")
    with db.ReadEpoch(path) as epoch:
        assert epoch.mode == "backup"
        leased = _managed_lease_dirs(root)
        assert leased
    assert _managed_lease_dirs(root) == []

    real = db.snapshot_live_db

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(db, "snapshot_live_db", boom)
    with pytest.raises(KeyboardInterrupt):
        with db.ReadEpoch(path):
            pass
    assert _managed_lease_dirs(root) == []
    monkeypatch.setattr(db, "snapshot_live_db", real)

    def fail(*a, **k):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(db, "snapshot_live_db", fail)
    with pytest.raises(RuntimeError):
        with db.ReadEpoch(path):
            pass
    assert _managed_lease_dirs(root) == []


def test_close_is_idempotent(tmp_path):
    path = tmp_path / "state.vscdb"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE ItemTable (key TEXT UNIQUE, value BLOB)")
    conn.execute("CREATE TABLE cursorDiskKV (key TEXT UNIQUE, value BLOB)")
    conn.commit()
    conn.close()
    epoch = db.ReadEpoch(path)
    epoch.__enter__()
    epoch.close()
    epoch.close()


def test_reaper_recovers_orphan_and_spares_active_lease(tmp_path, monkeypatch):
    root = tmp_path / "cursaves-snapshots"
    monkeypatch.setenv("CURSAVES_SNAPSHOT_ROOT", str(root))
    orphan = root / "read-orphan"
    orphan.mkdir(parents=True)
    (orphan / ".lease").write_text("dead\n")
    (orphan / "state.vscdb").write_text("x")
    live = db.acquire_lease("read")
    decoy_sib = tmp_path / "cursaves-profile"
    decoy_sib.mkdir()
    (decoy_sib / "keep").write_text("safe")
    decoy_child = root / "cursaves-profile"
    decoy_child.mkdir()
    (decoy_child / ".lease").write_text("nope")
    removed = db.reap_orphaned_leases()
    assert orphan in removed or not orphan.exists()
    assert live.path.exists()
    assert (decoy_sib / "keep").exists()
    assert decoy_child.exists()
    live.release()
    assert not live.path.exists()


def test_sigterm_then_next_run_reaps(tmp_path, monkeypatch):
    root = tmp_path / "cursaves-snapshots"
    monkeypatch.setenv("CURSAVES_SNAPSHOT_ROOT", str(root))
    env = os.environ.copy()
    env["CURSAVES_SNAPSHOT_ROOT"] = str(root)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            "from cursor_saves import db\n"
            "lease = db.acquire_lease('read')\n"
            "print(lease.path, flush=True)\n"
            "time.sleep(60)\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    leased = Path(proc.stdout.readline().strip())
    assert leased.exists()
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)
    removed = db.reap_orphaned_leases()
    assert leased in removed or not leased.exists()


def test_sigkill_then_next_run_reaps(tmp_path, monkeypatch):
    root = tmp_path / "cursaves-snapshots"
    monkeypatch.setenv("CURSAVES_SNAPSHOT_ROOT", str(root))
    env = os.environ.copy()
    env["CURSAVES_SNAPSHOT_ROOT"] = str(root)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            "from cursor_saves import db\n"
            "lease = db.acquire_lease('pull')\n"
            "print(lease.path, flush=True)\n"
            "time.sleep(60)\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    leased = Path(proc.stdout.readline().strip())
    assert leased.exists()
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=5)
    removed = db.reap_orphaned_leases()
    assert leased in removed or not leased.exists()


def test_noop_sync_on_wal_has_no_global_copy(sync_env, monkeypatch):
    snaps = [
        _conversation([_msg(1, "A")], composer_id=CID_A, name="A"),
        _conversation([_msg(1, "B")], composer_id=CID_B, name="B"),
    ]
    _commit_env(sync_env, snaps, snaps, digest=True)
    _enable_wal(sync_env["global_db"])
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    _backend(monkeypatch)
    monkeypatch.setattr(
        cli, "import_snapshot", lambda *a, **k: (_ for _ in ()).throw(AssertionError())
    )
    copies = _count_snapshots(monkeypatch)
    syncstate.reset_op_counts()
    cli.cmd_sync(_args(workspace=WS_HASH[:8]))
    assert syncstate.op_counts().read_copy_global == 0
    assert syncstate.op_counts().sqlite_backups == 0
    assert all(kind != "read" or scope != "global" for kind, scope in copies)
    assert syncstate.op_counts().safety_global_backups == 0


def test_scoped_status_index_keeps_legacy_ssh_bucket(sync_env, monkeypatch, capsys):
    ssh_path = "/home/lju/nixos"
    host = "MindLoop1"
    remote = _conversation(
        [_msg(1, "A")],
        composer_id=CID_A,
        name="SSH",
        sourceHost=host,
        sourceProjectPath=ssh_path,
        projectIdentifier="nixos",
    )
    local = _conversation(
        [_msg(1, "A")],
        composer_id=CID_A,
        name="SSH",
        sourceHost=host,
        sourceProjectPath=ssh_path,
        projectIdentifier="nixos",
    )
    gconn = _init_db(sync_env["global_db"])
    _write_local(gconn, local)
    gconn.commit()
    gconn.close()
    ws_dir = sync_env["ws_dir"]
    _write_workspace(ws_dir, [local])
    _write_snapshot_file(sync_env["snaps"] / "nixos", remote, with_digest=True)
    monkeypatch.setattr(cli, "_ensure_synced", lambda: None)
    _backend(monkeypatch)
    monkeypatch.setattr(
        cli,
        "_resolve_project_and_workspace",
        lambda args, session=None: (ssh_path, ws_dir, host),
    )
    cli.cmd_status(type("Args", (), {"workspace": "1", "project": None})())
    out = capsys.readouterr().out
    assert "In both:                 1" in out
    assert "Snapshot files:          1" in out


def test_sync_local_change_after_preflight_aborts_before_write_and_push(
    sync_env, monkeypatch
):
    remote = _conversation(
        [_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="Behind"
    )
    local = _conversation([_msg(1, "A")], composer_id=CID_A, name="Behind")
    _commit_env(sync_env, [local], [remote], digest=False)
    real_build = syncstate.build_sync_plan

    def wrapped(*a, **k):
        plan = real_build(*a, **k)
        assert plan.behind
        assert plan.behind[0].local_guard is not None
        gconn = sqlite3.connect(str(sync_env["global_db"]))
        _write_local(
            gconn,
            _conversation([_msg(1, "CHANGED")], composer_id=CID_A, name="Behind"),
        )
        gconn.commit()
        gconn.close()
        return plan

    monkeypatch.setattr(syncstate, "build_sync_plan", wrapped)
    imports = {"n": 0}
    monkeypatch.setattr(
        cli,
        "import_snapshot",
        lambda *a, **k: imports.__setitem__("n", imports["n"] + 1) or True,
    )
    monkeypatch.setattr(
        db,
        "backup_db",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("backup")),
    )
    backend = _backend(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_sync(type("Args", (), {"force": False})())
    assert exc.value.code == 1
    assert imports["n"] == 0
    assert backend.pushes == 0


def test_sync_behind_snapshot_change_after_preflight_is_not_imported(
    sync_env, monkeypatch
):
    remote = _conversation(
        [_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="Behind"
    )
    local = _conversation([_msg(1, "A")], composer_id=CID_A, name="Behind")
    _commit_env(sync_env, [local], [remote], digest=False)
    real_build = syncstate.build_sync_plan

    def wrapped(*a, **k):
        plan = real_build(*a, **k)
        assert plan.behind
        changed = _conversation(
            [_msg(1, "A"), _msg(2, "B"), _msg(3, "NEW")],
            composer_id=CID_A,
            name="Behind",
        )
        _write_snapshot_file(sync_env["project_dir"], changed, gzip_body=True)
        return plan

    monkeypatch.setattr(syncstate, "build_sync_plan", wrapped)
    imported: list[bytes] = []

    def spy_import(path, *a, **k):
        imported.append(Path(path).read_bytes())
        return True

    monkeypatch.setattr(cli, "import_snapshot", spy_import)
    monkeypatch.setattr(
        db,
        "backup_db",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("backup")),
    )
    backend = _backend(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_sync(type("Args", (), {"force": False})())
    assert exc.value.code == 1
    assert imported == []
    assert backend.pushes == 0


def test_ahead_stage_rejects_destination_changed_since_preflight(
    sync_env, monkeypatch
):
    remote = _conversation([_msg(1, "A")], composer_id=CID_A, name="Ahead")
    local = _conversation(
        [_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="Ahead"
    )
    _commit_env(sync_env, [local], [remote], digest=False)
    monkeypatch.setattr(
        export,
        "save_snapshot",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("stage")),
    )
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
        assert plan.ahead
        pinned = plan.ahead[0].classified_identity
        assert plan.ahead[0].dest_expected_present is True
        assert pinned
        changed = _conversation(
            [_msg(1, "A"), _msg(2, "FROM-B")], composer_id=CID_A, name="Other"
        )
        _write_snapshot_file(sync_env["project_dir"], changed, gzip_body=True)
        with pytest.raises(syncstate.SyncPreflightStale):
            syncstate.stage_ahead_exports(plan, session)
        assert plan.ahead[0].classified_identity == pinned

    never = _conversation([_msg(1, "new")], composer_id=CID_B, name="Never")
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _write_local(gconn, never)
    gconn.commit()
    gconn.close()
    _write_workspace(sync_env["ws_dir"], [never])
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
        never_items = [
            i for i in plan.items if i.relation == syncstate.SyncRelation.NEVER_PUSHED
        ]
        assert never_items
        assert never_items[0].dest_expected_present is False
        appeared = _conversation([_msg(1, "late")], composer_id=CID_B, name="Late")
        _write_snapshot_file(sync_env["project_dir"], appeared, gzip_body=True)
        with pytest.raises(syncstate.SyncPreflightStale):
            syncstate.stage_ahead_exports(plan, session)


def test_ahead_promote_rejects_destination_changed_since_preflight(sync_env):
    remote = _conversation([_msg(1, "A")], composer_id=CID_A, name="Ahead")
    local = _conversation(
        [_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="Ahead"
    )
    _commit_env(sync_env, [local], [remote], digest=False)
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
        staged = syncstate.stage_ahead_exports(plan, session)
    assert staged is not None
    dest = sync_env["project_dir"]
    changed = _conversation(
        [_msg(1, "A"), _msg(2, "FROM-B")], composer_id=CID_A, name="Other"
    )
    _write_snapshot_file(dest, changed, gzip_body=True)
    with pytest.raises(syncstate.SyncPreflightStale):
        syncstate.promote_staged_ahead(staged)
    assert staged.promoted is False
    body = importer.read_snapshot_file(
        dest / f"{CID_A}.json.gz",
        json.loads((dest / f"{CID_A}.meta.json").read_text()),
    )
    texts = [b.get("text") for b in (body.get("bubbleEntries") or {}).values()]
    assert "FROM-B" in texts
    staged.discard()


def test_ahead_promote_unsharded_to_sharded_replaces_old_components(sync_env):
    old = _conversation([_msg(1, "OLD")], composer_id=CID_A, name="A")
    new = _conversation([_msg(1, "NEW"), _msg(2, "MORE")], composer_id=CID_A, name="A")
    dest = sync_env["project_dir"]
    _write_snapshot_file(dest, old, gzip_body=True)
    dest_main = dest / f"{CID_A}.json.gz"
    dest_meta = json.loads((dest / f"{CID_A}.meta.json").read_text())
    present, identity = syncstate.destination_snapshot_identity(dest_main, dest_meta)
    lease = db.acquire_lease("ahead")
    staged_proj = lease.path / "snapshots" / "project"
    _write_sharded_snapshot(staged_proj, new, parts=2, with_digest=True)
    staged = syncstate.StagedAhead(
        lease=lease,
        count=1,
        expectations=[
            syncstate.AheadExpectation(
                composer_id=CID_A,
                project_identifier="project",
                dest_main=dest_main,
                dest_meta=dest_meta,
                expected_present=present,
                expected_identity=identity,
                staged_project=staged_proj,
            )
        ],
    )
    assert syncstate.promote_staged_ahead(staged) == 1
    assert not dest_main.exists()
    assert (dest / f"{CID_A}.json.gz.00").exists()
    assert (dest / f"{CID_A}.json.gz.01").exists()
    meta = json.loads((dest / f"{CID_A}.meta.json").read_text())
    body = importer.read_snapshot_file(dest_main, meta)
    dumped = json.dumps(body)
    assert "NEW" in dumped
    assert "OLD" not in dumped


def test_ahead_promote_sharded_to_unsharded_replaces_old_components(sync_env):
    old = _conversation([_msg(1, "OLD")], composer_id=CID_A, name="A")
    new = _conversation([_msg(1, "NEW")], composer_id=CID_A, name="A")
    dest = sync_env["project_dir"]
    _write_sharded_snapshot(dest, old, parts=2, with_digest=True)
    dest_main = dest / f"{CID_A}.json.gz"
    dest_meta = json.loads((dest / f"{CID_A}.meta.json").read_text())
    present, identity = syncstate.destination_snapshot_identity(dest_main, dest_meta)
    lease = db.acquire_lease("ahead")
    staged_proj = lease.path / "snapshots" / "project"
    _write_snapshot_file(staged_proj, new, gzip_body=True)
    staged = syncstate.StagedAhead(
        lease=lease,
        count=1,
        expectations=[
            syncstate.AheadExpectation(
                composer_id=CID_A,
                project_identifier="project",
                dest_main=dest_main,
                dest_meta=dest_meta,
                expected_present=present,
                expected_identity=identity,
                staged_project=staged_proj,
            )
        ],
    )
    assert syncstate.promote_staged_ahead(staged) == 1
    assert dest_main.exists()
    assert not (dest / f"{CID_A}.json.gz.00").exists()
    assert not (dest / f"{CID_A}.json.gz.01").exists()
    meta = json.loads((dest / f"{CID_A}.meta.json").read_text())
    body = importer.read_snapshot_file(dest_main, meta)
    dumped = json.dumps(body)
    assert "NEW" in dumped
    assert "OLD" not in dumped


def test_reaper_cannot_delete_lease_during_acquire(tmp_path, monkeypatch):
    root = tmp_path / "cursaves-snapshots"
    monkeypatch.setenv("CURSAVES_SNAPSHOT_ROOT", str(root))
    env = os.environ.copy()
    env["CURSAVES_SNAPSHOT_ROOT"] = str(root)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, os, tempfile, time\n"
            "from pathlib import Path\n"
            "from cursor_saves import db\n"
            "root = db.snapshot_root()\n"
            "fd = None\n"
            "with db.snapshot_manager_lock():\n"
            "    tmp = Path(tempfile.mkdtemp(prefix='read-', dir=str(root)))\n"
            "    fd = os.open(str(tmp / '.lease'), os.O_CREAT | os.O_RDWR, 0o644)\n"
            "    print(tmp, flush=True)\n"
            "    time.sleep(1.5)\n"
            "    fcntl.flock(fd, fcntl.LOCK_EX)\n"
            "time.sleep(30)\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    leased = Path(proc.stdout.readline().strip())
    assert leased.exists()
    removed: list[Path] = []

    def reap():
        removed.extend(db.reap_orphaned_leases())

    worker = threading.Thread(target=reap)
    worker.start()
    time.sleep(0.2)
    assert leased.exists()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert leased.exists()
    assert leased not in removed
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


def test_reaper_cannot_delete_lease_during_acquire_same_process(
    tmp_path, monkeypatch
):
    root = tmp_path / "cursaves-snapshots"
    monkeypatch.setenv("CURSAVES_SNAPSHOT_ROOT", str(root))
    started = threading.Event()
    finish = threading.Event()
    held: dict[str, object] = {}

    def creator():
        with db.snapshot_manager_lock(root):
            tmp = Path(tempfile.mkdtemp(prefix="read-", dir=str(root)))
            held["path"] = tmp
            started.set()
            assert finish.wait(timeout=5)
            fd = os.open(str(tmp / ".lease"), os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(fd, fcntl.LOCK_EX)
            held["fd"] = fd

    worker = threading.Thread(target=creator)
    worker.start()
    assert started.wait(timeout=2)
    removed: list[Path] = []

    def reap():
        removed.extend(db.reap_orphaned_leases(root))

    reaper = threading.Thread(target=reap)
    reaper.start()
    time.sleep(0.2)
    path = Path(held["path"])
    assert path.exists()
    assert reaper.is_alive()
    finish.set()
    reaper.join(timeout=5)
    worker.join(timeout=5)
    assert path.exists()
    assert path not in removed
    fd = held.get("fd")
    if isinstance(fd, int):
        try:
            os.close(fd)
        except OSError:
            pass


def test_reaper_recovers_managed_dir_killed_before_lease_creation(
    tmp_path, monkeypatch
):
    root = tmp_path / "cursaves-snapshots"
    monkeypatch.setenv("CURSAVES_SNAPSHOT_ROOT", str(root))
    incomplete = root / "read-killed-before-lease"
    incomplete.mkdir(parents=True)
    (incomplete / "partial").write_text("orphaned")
    removed = db.reap_orphaned_leases()
    assert incomplete in removed or not incomplete.exists()
    assert not incomplete.exists()


def test_live_connect_failure_falls_back_to_backup(tmp_path, monkeypatch):
    path = tmp_path / "state.vscdb"
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE ItemTable (key TEXT UNIQUE, value BLOB)")
    conn.execute("CREATE TABLE cursorDiskKV (key TEXT UNIQUE, value BLOB)")
    conn.execute("INSERT INTO ItemTable VALUES ('hello', 'v1')")
    conn.commit()
    conn.close()
    real = sqlite3.connect
    ro_calls = {"n": 0}

    def wrapped(*args, **kwargs):
        target = args[0] if args else ""
        if kwargs.get("uri") and isinstance(target, str) and "mode=ro" in target:
            ro_calls["n"] += 1
            if ro_calls["n"] == 1:
                raise sqlite3.OperationalError("live connect failed")
        return real(*args, **kwargs)

    monkeypatch.setattr(db.sqlite3, "connect", wrapped)
    with db.ReadEpoch(path) as epoch:
        assert epoch.mode == "backup"
        assert (
            epoch.connection.execute(
                "SELECT value FROM ItemTable WHERE key='hello'"
            ).fetchone()[0]
            == "v1"
        )
    assert ro_calls["n"] >= 2


def test_full_numeric_workspace_hash_still_resolves(tmp_path, monkeypatch):
    storage = tmp_path / "workspaceStorage"
    monkeypatch.setattr(paths, "get_workspace_storage_dir", lambda: storage)
    numeric = "1" * 32
    num_dir = storage / numeric
    other = storage / HASH_A
    _write_ws_json(num_dir, "/home/user/numeric")
    _write_ws_json(other, "/home/user/alpha")
    snap = _conversation([_msg(1, "hi")], composer_id=CID_A, name="A")
    _write_workspace(num_dir, [snap])
    _write_workspace(other, [snap])
    os.utime(num_dir / "state.vscdb", (1, 1))
    os.utime(other / "state.vscdb", (2, 2))

    found = paths.resolve_workspace(numeric)
    assert found is not None
    assert found["workspace_dir"] == num_dir
    assert paths.resolve_workspace("1")["workspace_dir"] == other
