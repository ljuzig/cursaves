"""Semantic sync classification and preflight write safety (v0.9.5)."""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
from pathlib import Path

import pytest

from cursor_saves import cli, db, dblock, export, paths, syncstate


PROJECT_PATH = "/home/user/project"
HOST_A = "host-a"
HOST_B = "host-b"
CID_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
CID_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
CID_C = "cccccccc-cccc-cccc-cccc-cccccccccccc"
CID_D = "dddddddd-dddd-dddd-dddd-dddddddddddd"
CID_E = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
CID_F = "ffffffff-ffff-ffff-ffff-ffffffffffff"
WS_HASH = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


# ── Snapshot / DB helpers ─────────────────────────────────────────────


def _msg(i: int, text: str, **extra) -> dict:
    return {"id": f"bubble-{i}", "type": extra.pop("type", 1), "text": text, **extra}


def _conversation(messages, *, composer_id: str, name: str = "Chat", **top) -> dict:
    headers = []
    bubbles = {}
    blobs = {}
    for i, msg in enumerate(messages):
        bid = msg.get("id", f"bubble-{i}")
        headers.append({"bubbleId": bid, "type": msg.get("type", 1)})
        bubble = {
            "bubbleId": bid,
            "type": msg.get("type", 1),
            "text": msg.get("text", ""),
        }
        for key in ("toolFormerData", "richText", "toolResults", "codeBlocks"):
            if key in msg:
                bubble[key] = msg[key]
        if "blob_id" in msg:
            bubble["contentHash"] = msg["blob_id"]
            blobs[msg["blob_id"]] = msg.get("blob_data", "blob")
        bubbles[bid] = bubble
    return {
        "version": top.get("version", 3),
        "exportedAt": top.get("exportedAt", "2026-01-01T00:00:00Z"),
        "sourceMachine": top.get("sourceMachine", "test-machine"),
        "sourceHost": top.get("sourceHost"),
        "sourceProjectPath": top.get("sourceProjectPath", PROJECT_PATH),
        "projectIdentifier": top.get("projectIdentifier", "project"),
        "composerId": composer_id,
        "composerData": {
            "composerId": composer_id,
            "name": name,
            "fullConversationHeadersOnly": headers,
        },
        "bubbleEntries": bubbles,
        "contentBlobs": blobs,
    }


def _hashes(snapshot: dict) -> list[str]:
    return syncstate.snapshot_unit_hashes(snapshot)


def _relation(local: dict, remote: dict) -> syncstate.SyncRelation:
    return syncstate.compare_unit_hashes(_hashes(local), _hashes(remote))


def _init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE IF NOT EXISTS ItemTable (key TEXT UNIQUE, value BLOB)")
    conn.execute("CREATE TABLE IF NOT EXISTS cursorDiskKV (key TEXT UNIQUE, value BLOB)")
    return conn


def _put_json(conn: sqlite3.Connection, key: str, value, table: str = "cursorDiskKV") -> None:
    conn.execute(
        f"INSERT OR REPLACE INTO {table} (key, value) VALUES (?, ?)",
        (key, json.dumps(value)),
    )


def _write_local(conn: sqlite3.Connection, snapshot: dict) -> None:
    cid = snapshot["composerId"]
    _put_json(conn, f"composerData:{cid}", snapshot["composerData"])
    cmap = (snapshot.get("composerData") or {}).get("conversationMap") or {}
    entries = snapshot.get("bubbleEntries") or cmap
    for bid, bubble in entries.items():
        _put_json(conn, f"bubbleId:{cid}:{bid}", bubble)
    for hid, blob in (snapshot.get("contentBlobs") or {}).items():
        value = blob if isinstance(blob, str) else json.dumps(blob)
        conn.execute(
            "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
            (f"composer.content.{hid}", value),
        )


def _write_workspace(ws_dir: Path, snapshots: list[dict]) -> None:
    conn = _init_db(ws_dir / "state.vscdb")
    _put_json(
        conn,
        "composer.composerData",
        {
            "allComposers": [
                {
                    "composerId": s["composerId"],
                    "name": (s.get("composerData") or {}).get("name", "Untitled"),
                }
                for s in snapshots
            ]
        },
        table="ItemTable",
    )
    conn.commit()
    conn.close()


def _write_snapshot_file(
    project_dir: Path,
    snapshot: dict,
    *,
    with_digest: bool = True,
    gzip_body: bool = True,
    meta_extra: dict | None = None,
) -> Path:
    cid = snapshot["composerId"]
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{cid}.json.gz"
    if gzip_body:
        raw = json.dumps(snapshot, ensure_ascii=False).encode()
        path.write_bytes(gzip.compress(raw, compresslevel=1))
    else:
        # Indexed by filename + sidecar. Digest hits never decompress this.
        path.write_bytes(b"\x1f\x8b")
    cd = snapshot.get("composerData") or {}
    meta = {
        "composerId": cid,
        "name": cd.get("name"),
        "messageCount": len(cd.get("fullConversationHeadersOnly") or []),
        "exportedAt": snapshot.get("exportedAt"),
        "sourceMachine": snapshot.get("sourceMachine"),
        "sourceHost": snapshot.get("sourceHost"),
        "sourceProjectPath": snapshot.get("sourceProjectPath"),
        "projectIdentifier": snapshot.get("projectIdentifier"),
        "version": snapshot.get("version", 3),
    }
    if meta_extra:
        meta.update(meta_extra)
    if with_digest:
        meta["semanticDigest"] = syncstate.snapshot_semantic_digest(snapshot)
        meta["semanticDigestVersion"] = syncstate.SEMANTIC_DIGEST_VERSION
        meta["snapshotContentDigest"] = syncstate.snapshot_content_digest(path, meta)
    else:
        meta.pop("semanticDigest", None)
        meta.pop("semanticDigestVersion", None)
        meta.pop("snapshotContentDigest", None)
    (project_dir / f"{cid}.meta.json").write_text(json.dumps(meta))
    return path


def _write_sharded_snapshot(
    project_dir: Path,
    snapshot: dict,
    *,
    parts: int = 2,
    with_digest: bool = False,
) -> Path:
    cid = snapshot["composerId"]
    project_dir.mkdir(parents=True, exist_ok=True)
    raw = gzip.compress(json.dumps(snapshot, ensure_ascii=False).encode(), compresslevel=1)
    chunk = max(1, (len(raw) + parts - 1) // parts)
    for i in range(parts):
        (project_dir / f"{cid}.json.gz.{i:02d}").write_bytes(raw[i * chunk : (i + 1) * chunk])
    cd = snapshot.get("composerData") or {}
    meta = {
        "composerId": cid,
        "name": cd.get("name"),
        "messageCount": len(cd.get("fullConversationHeadersOnly") or []),
        "exportedAt": snapshot.get("exportedAt"),
        "sourceMachine": snapshot.get("sourceMachine"),
        "sourceHost": snapshot.get("sourceHost"),
        "sourceProjectPath": snapshot.get("sourceProjectPath"),
        "projectIdentifier": snapshot.get("projectIdentifier"),
        "version": snapshot.get("version", 3),
        "shardCount": parts,
    }
    if with_digest:
        meta["semanticDigest"] = syncstate.snapshot_semantic_digest(snapshot)
        meta["semanticDigestVersion"] = syncstate.SEMANTIC_DIGEST_VERSION
        meta["snapshotContentDigest"] = syncstate.snapshot_content_digest(
            project_dir / f"{cid}.json.gz", meta
        )
    (project_dir / f"{cid}.meta.json").write_text(json.dumps(meta))
    return project_dir / f"{cid}.json.gz"


def _commit_env(
    env: dict,
    locals_: list[dict],
    remotes: list[dict],
    *,
    digest: bool = True,
    gzip_cids: set[str] | None = None,
) -> None:
    gconn = _init_db(env["global_db"])
    for snap in locals_:
        _write_local(gconn, snap)
    gconn.commit()
    gconn.close()
    _write_workspace(env["ws_dir"], locals_)
    for snap in remotes:
        cid = snap["composerId"]
        write_gz = True if gzip_cids is None and not digest else bool(gzip_cids and cid in gzip_cids)
        _write_snapshot_file(
            env["project_dir"],
            snap,
            with_digest=digest,
            gzip_body=write_gz,
        )


@pytest.fixture
def sync_env(tmp_path, monkeypatch):
    lock = tmp_path / "sqlite-write.lock"
    repo = tmp_path / "repo.lock"
    monkeypatch.setenv("CURSAVES_SQLITE_LOCK", str(lock))
    monkeypatch.setenv("CURSAVES_REPO_LOCK", str(repo))
    monkeypatch.setenv("CURSAVES_SQLITE_LOCK_TIMEOUT", "1")
    monkeypatch.setenv("CURSAVES_REPO_LOCK_TIMEOUT", "1")
    dblock.reset_for_tests()
    db.reset_write_tracking_for_tests()
    paths.invalidate_headers_cache()
    syncstate.reset_op_counts()

    global_db = tmp_path / "cursor" / "globalStorage" / "state.vscdb"
    ws_dir = tmp_path / "cursor" / "workspaceStorage" / WS_HASH
    snaps = tmp_path / "snapshots"
    snaps.mkdir(parents=True)
    sync_dir = tmp_path / "cursaves"
    (sync_dir / ".git").mkdir(parents=True)

    def list_ws():
        return [
            {
                "type": "local",
                "host": None,
                "path": PROJECT_PATH,
                "workspace_dir": ws_dir,
                "conversations": 0,
            }
        ]

    monkeypatch.setattr(paths, "get_cursor_user_dir", lambda: tmp_path / "cursor")
    monkeypatch.setattr(paths, "get_global_db_path", lambda: global_db)
    monkeypatch.setattr(paths, "get_workspace_storage_dir", lambda: tmp_path / "cursor" / "workspaceStorage")
    monkeypatch.setattr(paths, "get_snapshots_dir", lambda: snaps)
    monkeypatch.setattr(paths, "get_sync_dir", lambda: sync_dir)
    monkeypatch.setattr(paths, "is_sync_repo_initialized", lambda: True)
    monkeypatch.setattr(paths, "_build_global_headers_map", lambda *a, **k: {})
    monkeypatch.setattr(paths, "list_workspaces_with_conversations", lambda *a, **k: list_ws())
    monkeypatch.setattr(paths, "list_all_workspaces", lambda: list_ws())
    monkeypatch.setattr(paths, "_get_git_remote_url", lambda _path: None)
    monkeypatch.setattr(paths, "get_machine_id", lambda: "test-machine")
    monkeypatch.setattr(paths, "get_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(cli, "_require_sync_repo", lambda: sync_dir)
    monkeypatch.setattr(
        cli,
        "resolve_sync_import_targets",
        lambda meta: [
            {
                "workspace_dir": ws_dir,
                "path": PROJECT_PATH,
                "type": "local",
                "host": None,
            }
        ],
    )

    yield {
        "tmp": tmp_path,
        "global_db": global_db,
        "ws_dir": ws_dir,
        "snaps": snaps,
        "sync_dir": sync_dir,
        "project_dir": snaps / "project",
    }
    db.reset_write_tracking_for_tests()
    dblock.reset_for_tests()
    paths.invalidate_headers_cache()
    syncstate.reset_op_counts()


def _backend(monkeypatch, *, has_remote: bool = True):
    class Backend:
        def __init__(self):
            self.pulls = 0
            self.pushes = 0

        def has_remote(self):
            return has_remote

        def pull(self, _d):
            self.pulls += 1
            return True

        def push(self, _d):
            self.pushes += 1
            return True

    backend = Backend()
    monkeypatch.setattr(cli, "get_backend", lambda: backend)
    return backend


def _count_backups(monkeypatch):
    n = {"n": 0}
    real = db.snapshot_live_db

    def wrapped(*args, **kwargs):
        n["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(db, "snapshot_live_db", wrapped)
    return n


def test_isolation_never_uses_real_cursor_or_cursaves_dirs():
    isolated_db = paths.get_global_db_path().resolve()
    isolated_snaps = paths.get_snapshots_dir().resolve()
    homes = []
    try:
        import pwd

        homes.append(Path(pwd.getpwuid(os.getuid()).pw_dir))
    except (ImportError, KeyError, OSError):
        pass
    for home in homes:
        cursor = (home / ".config" / "Cursor").resolve()
        snaps = (home / ".cursaves").resolve()
        assert cursor not in isolated_db.parents
        assert snaps not in isolated_snaps.parents


# ── Prefix relations ──────────────────────────────────────────────────


def test_equal_is_up_to_date():
    snap = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A)
    assert _relation(snap, snap) == syncstate.SyncRelation.UP_TO_DATE


def test_local_prefix_extension_is_ahead():
    remote = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A)
    local = _conversation([_msg(1, "A"), _msg(2, "B"), _msg(3, "C")], composer_id=CID_A)
    assert _relation(local, remote) == syncstate.SyncRelation.LOCAL_AHEAD


def test_remote_prefix_extension_is_behind():
    remote = _conversation([_msg(1, "A"), _msg(2, "B"), _msg(3, "C")], composer_id=CID_A)
    local = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A)
    assert _relation(local, remote) == syncstate.SyncRelation.BEHIND


def test_same_message_count_different_content_is_diverged():
    remote = _conversation([_msg(1, "A"), _msg(2, "B"), _msg(3, "C")], composer_id=CID_A)
    local = _conversation([_msg(1, "A"), _msg(2, "B"), _msg(3, "X")], composer_id=CID_A)
    assert len(remote["composerData"]["fullConversationHeadersOnly"]) == 3
    assert len(local["composerData"]["fullConversationHeadersOnly"]) == 3
    assert _relation(local, remote) == syncstate.SyncRelation.DIVERGED


def test_false_prefix_local_longer_is_diverged():
    remote = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A)
    local = _conversation([_msg(1, "A"), _msg(2, "X"), _msg(3, "C")], composer_id=CID_A)
    assert _relation(local, remote) == syncstate.SyncRelation.DIVERGED


def test_false_prefix_remote_longer_is_diverged():
    remote = _conversation([_msg(1, "A"), _msg(2, "X"), _msg(3, "C")], composer_id=CID_A)
    local = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A)
    assert _relation(local, remote) == syncstate.SyncRelation.DIVERGED


def test_same_id_modified_content_is_diverged():
    remote = _conversation([_msg(1, "hello")], composer_id=CID_A)
    local = _conversation([_msg(1, "changed")], composer_id=CID_A)
    assert remote["bubbleEntries"]["bubble-1"]["bubbleId"] == local["bubbleEntries"]["bubble-1"]["bubbleId"]
    assert _relation(local, remote) == syncstate.SyncRelation.DIVERGED


def test_same_text_different_tool_data_is_diverged():
    remote = _conversation(
        [_msg(1, "hello", type=2, toolFormerData={"tool": "shell", "args": "ls"})],
        composer_id=CID_A,
    )
    local = _conversation(
        [_msg(1, "hello", type=2, toolFormerData={"tool": "shell", "args": "pwd"})],
        composer_id=CID_A,
    )
    assert _relation(local, remote) == syncstate.SyncRelation.DIVERGED


def test_referenced_blob_change_is_diverged():
    remote = _conversation(
        [_msg(1, "see", blob_id="blob-aaa", blob_data="one")],
        composer_id=CID_A,
    )
    local = _conversation(
        [_msg(1, "see", blob_id="blob-aaa", blob_data="two")],
        composer_id=CID_A,
    )
    assert _relation(local, remote) == syncstate.SyncRelation.DIVERGED


def test_blob_only_on_new_local_message_is_ahead():
    remote = _conversation([_msg(1, "A")], composer_id=CID_A)
    local = _conversation(
        [_msg(1, "A"), _msg(2, "B", blob_id="blob-new", blob_data="extra")],
        composer_id=CID_A,
    )
    assert _relation(local, remote) == syncstate.SyncRelation.LOCAL_AHEAD


def test_cursaves_metadata_only_is_up_to_date():
    remote = _conversation(
        [_msg(1, "A"), _msg(2, "B")],
        composer_id=CID_A,
        exportedAt="2020-01-01T00:00:00Z",
        sourceMachine="machine-a",
        projectIdentifier="old-id",
    )
    local = _conversation(
        [_msg(1, "A"), _msg(2, "B")],
        composer_id=CID_A,
        exportedAt="2026-09-02T00:00:00Z",
        sourceMachine="machine-b",
        projectIdentifier="new-id",
    )
    assert _relation(local, remote) == syncstate.SyncRelation.UP_TO_DATE


def test_legacy_conversation_map_classifies_without_rewrite(tmp_path):
    headers = [{"bubbleId": "bubble-1", "type": 1}]
    bubble = {"bubbleId": "bubble-1", "type": 1, "text": "legacy"}
    snapshot = {
        "version": 1,
        "exportedAt": "2024-01-01T00:00:00Z",
        "sourceMachine": "old-machine",
        "sourceProjectPath": PROJECT_PATH,
        "projectIdentifier": "project",
        "composerId": CID_A,
        "composerData": {
            "composerId": CID_A,
            "name": "Legacy",
            "fullConversationHeadersOnly": headers,
            "conversationMap": {"bubble-1": bubble},
        },
    }
    path = tmp_path / f"{CID_A}.json.gz"
    raw = gzip.compress(json.dumps(snapshot).encode())
    path.write_bytes(raw)
    before = path.read_bytes()
    parsed = syncstate.importer.read_snapshot_file(path)
    hashes = syncstate.snapshot_unit_hashes(parsed)
    assert len(hashes) == 1
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns


def test_legacy_and_current_same_content_are_equal():
    current = _conversation([_msg(1, "legacy")], composer_id=CID_A, name="Legacy")
    legacy = {
        "version": 1,
        "composerId": CID_A,
        "composerData": {
            "composerId": CID_A,
            "name": "Legacy",
            "fullConversationHeadersOnly": current["composerData"]["fullConversationHeadersOnly"],
            "conversationMap": current["bubbleEntries"],
        },
    }
    assert _hashes(current) == _hashes(legacy)


# ── Index / session classification ────────────────────────────────────


def test_digest_fast_path_skips_deep_read(sync_env):
    snap = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="Synced")
    _commit_env(sync_env, [snap], [snap], digest=True)
    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        rel = syncstate.classify_conversation(
            session, index, CID_A, project_identifier="project"
        )
    assert rel == syncstate.SyncRelation.UP_TO_DATE
    assert syncstate.op_counts().deep_snapshot_reads == 0
    assert syncstate.op_counts().snapshot_directory_scans == 1
    assert syncstate.op_counts().full_local_exports == 0
    assert not db.write_connections_open()


def test_plan_reuses_parsed_snapshot_cache(sync_env):
    snap = _conversation([_msg(1, "A")], composer_id=CID_A)
    _commit_env(sync_env, [snap], [snap], digest=False)
    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
        again = syncstate.classify_conversation(
            session, index, CID_A, project_identifier="project"
        )
    assert plan.items[0].relation == syncstate.SyncRelation.UP_TO_DATE
    assert again == syncstate.SyncRelation.UP_TO_DATE
    assert syncstate.op_counts().deep_snapshot_reads == 1
    assert syncstate.op_counts().snapshot_directory_scans == 1


# ── Sync preflight ────────────────────────────────────────────────────


def test_diverged_blocks_all_writes(sync_env, monkeypatch):
    behind_remote = _conversation(
        [_msg(1, "A"), _msg(2, "B"), _msg(3, "C")], composer_id=CID_B, name="Behind"
    )
    behind_local = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_B, name="Behind")
    ahead_remote = _conversation([_msg(1, "A")], composer_id=CID_C, name="Ahead")
    ahead_local = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_C, name="Ahead")
    diverged_remote = _conversation(
        [_msg(1, "A"), _msg(2, "B"), _msg(3, "C")], composer_id=CID_D, name="Diverged"
    )
    diverged_local = _conversation(
        [_msg(1, "A"), _msg(2, "B"), _msg(3, "X")], composer_id=CID_D, name="Diverged"
    )
    _commit_env(
        sync_env,
        [behind_local, ahead_local, diverged_local],
        [behind_remote, ahead_remote, diverged_remote],
        digest=False,
    )

    backend = _backend(monkeypatch)
    imports = {"n": 0}
    saves = {"n": 0}
    monkeypatch.setattr(cli, "import_snapshot", lambda *a, **k: imports.__setitem__("n", imports["n"] + 1) or True)
    monkeypatch.setattr(export, "save_snapshot", lambda *a, **k: saves.__setitem__("n", saves["n"] + 1) or Path("x"))

    with pytest.raises(SystemExit) as exc:
        cli.cmd_sync(type("Args", (), {"force": False})())
    assert exc.value.code == 1
    assert imports["n"] == 0
    assert saves["n"] == 0
    assert backend.pushes == 0
    assert backend.pulls == 1


def test_force_does_not_override_divergence(sync_env, monkeypatch, capsys):
    remote = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="Diverged")
    local = _conversation([_msg(1, "A"), _msg(2, "X")], composer_id=CID_A, name="Diverged")
    _commit_env(sync_env, [local], [remote], digest=False)
    backend = _backend(monkeypatch)
    imports = {"n": 0}
    monkeypatch.setattr(cli, "import_snapshot", lambda *a, **k: imports.__setitem__("n", imports["n"] + 1) or True)

    with pytest.raises(SystemExit) as exc:
        cli.cmd_sync(type("Args", (), {"force": True})())
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Sync aborted: divergent conversations detected." in err
    assert "Sync stopped before importing into Cursor or creating/pushing snapshots." in err
    assert imports["n"] == 0
    assert backend.pushes == 0


def test_unknown_snapshot_blocks_writes(sync_env, monkeypatch, capsys):
    local = _conversation([_msg(1, "A")], composer_id=CID_F, name="Broken")
    _commit_env(sync_env, [local], [], digest=False)
    project_dir = sync_env["project_dir"]
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / f"{CID_F}.json.gz").write_bytes(b"not-a-gzip-snapshot")
    (project_dir / f"{CID_F}.meta.json").write_text(
        json.dumps({"composerId": CID_F, "name": "Broken", "messageCount": 1})
    )

    backend = _backend(monkeypatch)
    imports = {"n": 0}
    saves = {"n": 0}
    monkeypatch.setattr(cli, "import_snapshot", lambda *a, **k: imports.__setitem__("n", imports["n"] + 1) or True)
    monkeypatch.setattr(export, "save_snapshot", lambda *a, **k: saves.__setitem__("n", saves["n"] + 1) or Path("x"))

    with pytest.raises(SystemExit) as exc:
        cli.cmd_sync(type("Args", (), {"force": False})())
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "could not be classified" in err
    assert imports["n"] == 0
    assert saves["n"] == 0
    assert backend.pushes == 0


def test_linear_sync_imports_then_releases_then_pushes(sync_env, monkeypatch):
    synced = [
        _conversation([_msg(1, f"s{i}")], composer_id=f"00000000-0000-0000-0000-{i:012d}", name=f"S{i}")
        for i in range(10)
    ]
    behind_remote = _conversation(
        [_msg(1, "A"), _msg(2, "B"), _msg(3, "C")], composer_id=CID_B, name="Behind"
    )
    behind_local = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_B, name="Behind")
    ahead_remote = _conversation([_msg(1, "A")], composer_id=CID_C, name="Ahead")
    ahead_local = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_C, name="Ahead")
    _commit_env(
        sync_env,
        synced + [behind_local, ahead_local],
        synced + [behind_remote, ahead_remote],
        digest=True,
        gzip_cids={CID_B, CID_C},
    )

    backend = _backend(monkeypatch)
    order = []
    monkeypatch.setattr(
        cli,
        "import_snapshot",
        lambda *a, **k: order.append("import") or True,
    )
    monkeypatch.setattr(
        export,
        "save_snapshot",
        lambda snap, d: order.append("save") or (d / f"{snap['composerId']}.json.gz"),
    )
    monkeypatch.setattr(
        export,
        "checkpoint_project",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not re-read live Cursor")),
    )
    real_finish = db.finish_cursor_writes

    def tracking_finish():
        order.append("finish")
        real_finish()

    monkeypatch.setattr(db, "finish_cursor_writes", tracking_finish)
    real_push = backend.push

    def tracking_push(d):
        order.append("push")
        return real_push(d)

    backend.push = tracking_push

    cli.cmd_sync(type("Args", (), {"force": False})())
    assert order[0] == "save"
    assert order.index("save") < order.index("import")
    assert order.index("import") < order.index("finish")
    assert order.index("finish") < order.index("push")
    assert backend.pulls == 1
    assert backend.pushes == 1


def test_op_counts_digest_majority_synced(sync_env, monkeypatch):
    synced = [
        _conversation([_msg(1, f"s{i}")], composer_id=f"10000000-0000-0000-0000-{i:012d}", name=f"S{i}")
        for i in range(96)
    ]
    ahead_remote = _conversation([_msg(1, "A")], composer_id=CID_C, name="Ahead")
    ahead_local = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_C, name="Ahead")
    never = _conversation([_msg(1, "new")], composer_id=CID_E, name="Never")
    _commit_env(
        sync_env,
        synced + [ahead_local, never],
        synced + [ahead_remote],
        digest=True,
        gzip_cids={CID_C},
    )
    backups = _count_backups(monkeypatch)
    _backend(monkeypatch)
    monkeypatch.setattr(cli, "import_snapshot", lambda *a, **k: True)
    monkeypatch.setattr(
        export,
        "save_snapshot",
        lambda snap, d: d / f"{snap['composerId']}.json.gz",
    )

    syncstate.reset_op_counts()
    cli.cmd_sync(type("Args", (), {"force": False})())
    counts = syncstate.op_counts()
    assert counts.snapshot_directory_scans == 1
    assert counts.deep_snapshot_reads == 1
    assert counts.full_local_exports == 1
    assert backups["n"] <= 4
    assert counts.cursor_write_connections == 0
    assert not db.write_connections_open()


def test_op_counts_all_synced_no_deep_reads(sync_env, monkeypatch):
    synced = [
        _conversation([_msg(1, f"s{i}")], composer_id=f"20000000-0000-0000-0000-{i:012d}", name=f"S{i}")
        for i in range(98)
    ]
    _commit_env(sync_env, synced, synced, digest=True)
    _backend(monkeypatch)
    monkeypatch.setattr(cli, "import_snapshot", lambda *a, **k: (_ for _ in ()).throw(AssertionError("import")))
    monkeypatch.setattr(export, "save_snapshot", lambda *a, **k: (_ for _ in ()).throw(AssertionError("save")))

    syncstate.reset_op_counts()
    cli.cmd_sync(type("Args", (), {"force": False})())
    counts = syncstate.op_counts()
    assert counts.deep_snapshot_reads == 0
    assert counts.full_local_exports == 0
    assert counts.snapshot_directory_scans == 1
    assert not db.write_connections_open()


def test_legacy_snapshots_decompressed_once(sync_env):
    snaps = [
        _conversation([_msg(1, f"l{i}")], composer_id=f"30000000-0000-0000-0000-{i:012d}", name=f"L{i}")
        for i in range(100)
    ]
    _commit_env(sync_env, snaps, snaps, digest=False)
    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
    assert len(plan.items) == 100
    assert all(i.relation == syncstate.SyncRelation.UP_TO_DATE for i in plan.items)
    assert syncstate.op_counts().deep_snapshot_reads == 100
    assert syncstate.op_counts().snapshot_directory_scans == 1
    assert syncstate.op_counts().sqlite_backups == 1


def test_workspaces_summary_adds_diverged_only_when_needed(sync_env, capsys):
    synced = _conversation([_msg(1, "A")], composer_id=CID_A, name="Synced")
    diverged_remote = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_D, name="Diverged")
    diverged_local = _conversation([_msg(1, "A"), _msg(2, "X")], composer_id=CID_D, name="Diverged")
    _commit_env(sync_env, [synced, diverged_local], [synced, diverged_remote], digest=False)

    cli.cmd_workspaces(type("Args", (), {})())
    out = capsys.readouterr().out
    assert "1 synced" in out
    assert "1 diverged" in out


def test_workspaces_summary_unchanged_when_all_synced(sync_env, capsys):
    snap = _conversation([_msg(1, "A")], composer_id=CID_A, name="Synced")
    _commit_env(sync_env, [snap], [snap], digest=True)
    cli.cmd_workspaces(type("Args", (), {})())
    out = capsys.readouterr().out
    assert "1 synced" in out
    assert "diverged" not in out
    assert "unknown" not in out


def test_new_snapshot_meta_includes_optional_digest(tmp_path, monkeypatch):
    monkeypatch.setenv("CURSAVES_REPO_LOCK", str(tmp_path / "repo.lock"))
    monkeypatch.setenv("CURSAVES_REPO_LOCK_TIMEOUT", "1")
    dblock.reset_for_tests()
    monkeypatch.setattr(paths, "get_machine_id", lambda: "test-machine")
    monkeypatch.setattr(paths, "get_project_identifier", lambda path, source_host=None: "project")
    snap = _conversation([_msg(1, "A")], composer_id=CID_A)
    export.save_snapshot(snap, tmp_path)
    meta = json.loads((tmp_path / "project" / f"{CID_A}.meta.json").read_text())
    assert meta["semanticDigest"].startswith("sha256:")
    assert meta["semanticDigestVersion"] == syncstate.SEMANTIC_DIGEST_VERSION
    assert meta["snapshotContentDigest"].startswith("sha256:")
    gz = tmp_path / "project" / f"{CID_A}.json.gz"
    parsed = syncstate.importer.read_snapshot_file(gz)
    assert "semanticDigest" not in parsed
    assert "semanticDigestVersion" not in parsed
    assert "snapshotContentDigest" not in parsed
    assert meta["snapshotContentDigest"] == syncstate.snapshot_content_digest(gz, meta)


def test_different_tool_path_is_diverged():
    remote = _conversation(
        [_msg(1, "hi", type=2, toolFormerData={"command": "read", "path": "/a/file.txt"})],
        composer_id=CID_A,
    )
    local = _conversation(
        [_msg(1, "hi", type=2, toolFormerData={"command": "read", "path": "/b/file.txt"})],
        composer_id=CID_A,
    )
    assert _relation(local, remote) == syncstate.SyncRelation.DIVERGED


def test_different_uri_is_diverged():
    remote = _conversation(
        [_msg(1, "open", type=2, toolFormerData={"uri": {"scheme": "file", "path": "/a"}})],
        composer_id=CID_A,
    )
    local = _conversation(
        [_msg(1, "open", type=2, toolFormerData={"uri": {"scheme": "file", "path": "/b"}})],
        composer_id=CID_A,
    )
    assert _relation(local, remote) == syncstate.SyncRelation.DIVERGED


def test_unknown_bubble_field_is_diverged():
    remote = _conversation([_msg(1, "A")], composer_id=CID_A)
    local = _conversation([_msg(1, "A")], composer_id=CID_A)
    local["bubbleEntries"]["bubble-1"]["newCursorField"] = "novel"
    assert _relation(local, remote) == syncstate.SyncRelation.DIVERGED


def test_semantic_header_field_is_diverged():
    remote = _conversation([_msg(1, "A")], composer_id=CID_A)
    local = _conversation([_msg(1, "A")], composer_id=CID_A)
    local["composerData"]["fullConversationHeadersOnly"][0]["serverBubbleId"] = "keep-transport"
    assert _relation(local, remote) == syncstate.SyncRelation.UP_TO_DATE
    local["composerData"]["fullConversationHeadersOnly"][0]["capabilityType"] = "extra"
    assert _relation(local, remote) == syncstate.SyncRelation.DIVERGED


def test_envelope_named_bubble_field_is_semantic():
    remote = _conversation([_msg(1, "A")], composer_id=CID_A)
    local = _conversation([_msg(1, "A")], composer_id=CID_A)
    local["bubbleEntries"]["bubble-1"]["sourceHost"] = "other-host"
    assert _relation(local, remote) == syncstate.SyncRelation.DIVERGED


def test_all_thinking_blocks_are_semantic():
    remote = _conversation([_msg(1, "A")], composer_id=CID_A)
    local = _conversation([_msg(1, "A")], composer_id=CID_A)
    local["bubbleEntries"]["bubble-1"]["allThinkingBlocks"] = [{"text": "thought"}]
    assert _relation(local, remote) == syncstate.SyncRelation.DIVERGED


def test_missing_referenced_bubble_is_a_tombstone():
    snap = _conversation([_msg(1, "A")], composer_id=CID_A)
    del snap["bubbleEntries"]["bubble-1"]
    hashes = syncstate.snapshot_unit_hashes(snap)
    assert len(hashes) == 1
    present = _conversation([_msg(1, "A")], composer_id=CID_A)
    assert hashes != syncstate.snapshot_unit_hashes(present)


def test_same_cid_other_project_is_never_pushed(sync_env):
    local = _conversation([_msg(1, "A")], composer_id=CID_A, projectIdentifier="project")
    other = _conversation([_msg(1, "stolen")], composer_id=CID_A, projectIdentifier="other-project")
    gconn = _init_db(sync_env["global_db"])
    _write_local(gconn, local)
    gconn.commit()
    gconn.close()
    _write_workspace(sync_env["ws_dir"], [local])
    _write_snapshot_file(
        sync_env["snaps"] / "other-project", other, with_digest=True, gzip_body=True
    )

    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        assert index.get(CID_A) is None
        assert index.get(CID_A, "project") is None
        other_rec = index.get(CID_A, "other-project")
        assert other_rec is not None
        assert other_rec.project_identifier == "other-project"
        assert (
            syncstate.classify_conversation(
                session, index, CID_A, project_identifier="project"
            )
            == syncstate.SyncRelation.NEVER_PUSHED
        )
        assert (
            syncstate.classify_conversation(
                session, index, CID_A, project_identifier="other-project"
            )
            == syncstate.SyncRelation.DIVERGED
        )
        plan = syncstate.build_sync_plan(session, index)
        by_origin = {i.project_identifier: i for i in plan.items}
        assert by_origin["project"].relation == syncstate.SyncRelation.NEVER_PUSHED
        assert by_origin["other-project"].relation == syncstate.SyncRelation.BEHIND


def test_same_cid_ssh_hosts_do_not_cross_match(sync_env):
    pid_a = paths.get_project_identifier(PROJECT_PATH, source_host=HOST_A)
    pid_b = paths.get_project_identifier(PROJECT_PATH, source_host=HOST_B)
    snap_a = _conversation(
        [_msg(1, "from-a")],
        composer_id=CID_A,
        projectIdentifier=pid_a,
        sourceHost=HOST_A,
        sourceProjectPath=PROJECT_PATH,
    )
    snap_b = _conversation(
        [_msg(1, "from-b")],
        composer_id=CID_A,
        projectIdentifier=pid_b,
        sourceHost=HOST_B,
        sourceProjectPath=PROJECT_PATH,
    )
    local = _conversation(
        [_msg(1, "from-a")],
        composer_id=CID_A,
        projectIdentifier=pid_a,
        sourceHost=HOST_A,
    )
    gconn = _init_db(sync_env["global_db"])
    _write_local(gconn, local)
    gconn.commit()
    gconn.close()
    _write_workspace(sync_env["ws_dir"], [local])
    _write_snapshot_file(sync_env["snaps"] / pid_a, snap_a, with_digest=True, gzip_body=True)
    _write_snapshot_file(sync_env["snaps"] / pid_b, snap_b, with_digest=True, gzip_body=True)

    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        assert index.get(CID_A, pid_a).meta["sourceHost"] == HOST_A
        assert index.get(CID_A, pid_b).meta["sourceHost"] == HOST_B
        assert (
            syncstate.classify_conversation(
                session,
                index,
                CID_A,
                project_identifier=pid_a,
                source_host=HOST_A,
                source_path=PROJECT_PATH,
            )
            == syncstate.SyncRelation.UP_TO_DATE
        )
        assert (
            syncstate.classify_conversation(
                session,
                index,
                CID_A,
                project_identifier=pid_b,
                source_host=HOST_B,
                source_path=PROJECT_PATH,
            )
            == syncstate.SyncRelation.DIVERGED
        )


def test_ssh_identity_uses_posix_path_normalization():
    assert paths.normalize_origin_path(
        "/home/user/foo/../project", source_host=HOST_A
    ) == "/home/user/project"
    assert paths.get_project_identifier(
        "/home/user/foo/../project", source_host=HOST_A
    ) == paths.get_project_identifier(PROJECT_PATH, source_host=HOST_A)
    assert paths.get_project_identifier(
        PROJECT_PATH, source_host=HOST_A
    ) != paths.get_project_identifier(PROJECT_PATH, source_host=HOST_B)


def test_nine_hundred_warm_cache_skips_deep_work(sync_env):
    n = 900
    snaps = [
        _conversation([_msg(1, f"x{i}")], composer_id=f"90000000-0000-0000-0000-{i:012d}", name=f"X{i}")
        for i in range(n)
    ]
    _commit_env(sync_env, snaps, snaps, digest=False)

    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
    assert len(plan.items) == n
    assert all(i.relation == syncstate.SyncRelation.UP_TO_DATE for i in plan.items)
    assert syncstate.op_counts().legacy_snapshot_decompressions == n
    assert syncstate.op_counts().local_semantic_rehashes == n
    assert syncstate.op_counts().local_inventory_json_parses == 0
    assert syncstate.op_counts().local_composer_json_parses == n

    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
    assert len(plan.items) == n
    assert syncstate.op_counts().legacy_snapshot_decompressions == 0
    assert syncstate.op_counts().local_semantic_rehashes == 0
    assert syncstate.op_counts().local_inventory_json_parses == 0
    assert syncstate.op_counts().local_composer_json_parses == 0
    assert syncstate.op_counts().sqlite_backups == 1
    assert syncstate.op_counts().snapshot_directory_scans == 1
    assert syncstate.op_counts().full_local_exports == 0
    assert syncstate.op_counts().deep_snapshot_reads == 0

    changed = snaps[:12]
    gconn = _init_db(sync_env["global_db"])
    for snap in changed:
        snap["bubbleEntries"]["bubble-1"]["text"] = "changed"
        _write_local(gconn, snap)
    gconn.commit()
    gconn.close()

    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
    assert len(plan.items) == n
    assert sum(1 for i in plan.items if i.relation == syncstate.SyncRelation.DIVERGED) == 12
    assert syncstate.op_counts().local_semantic_rehashes == 12
    assert syncstate.op_counts().legacy_snapshot_decompressions == 12
    assert syncstate.op_counts().deep_snapshot_reads == 12


def test_planner_does_not_cross_compare_snapshot_only_origin(sync_env):
    """Same CID in workspace A must not classify a snapshot-only project B."""
    local_a = _conversation([_msg(1, "from-a")], composer_id=CID_A, name="A")
    snap_b = _conversation(
        [_msg(1, "from-b")],
        composer_id=CID_A,
        name="B",
        projectIdentifier="other-project",
    )
    gconn = _init_db(sync_env["global_db"])
    _write_local(gconn, local_a)
    gconn.commit()
    gconn.close()
    _write_workspace(sync_env["ws_dir"], [local_a])
    _write_snapshot_file(
        sync_env["snaps"] / "other-project",
        snap_b,
        with_digest=True,
        gzip_body=True,
    )

    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
    by_origin = {i.project_identifier: i for i in plan.items}
    assert by_origin["project"].relation == syncstate.SyncRelation.NEVER_PUSHED
    assert by_origin["other-project"].relation == syncstate.SyncRelation.BEHIND
    assert by_origin["other-project"].relation != syncstate.SyncRelation.DIVERGED


def test_incomplete_inventory_does_not_reuse_local_cache(sync_env, monkeypatch):
    snap = _conversation([_msg(1, "A")], composer_id=CID_A, name="Cached")
    _commit_env(sync_env, [snap], [snap], digest=True)
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        syncstate.build_sync_plan(session, index)

    def fail_inventory(self):
        self._inventory_complete = False
        self._row_fp = {}

    monkeypatch.setattr(syncstate.SyncReadSession, "_load_inventory", fail_inventory)
    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
    assert plan.items[0].relation == syncstate.SyncRelation.UP_TO_DATE
    assert syncstate.op_counts().local_semantic_rehashes == 1


def test_stale_semantic_digest_version_forces_deep_compare(sync_env):
    snap = _conversation([_msg(1, "A")], composer_id=CID_A)
    _commit_env(sync_env, [snap], [snap], digest=True, gzip_cids={CID_A})
    meta_path = sync_env["project_dir"] / f"{CID_A}.meta.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("semanticDigestVersion", None)
    meta_path.write_text(json.dumps(meta))

    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        rel = syncstate.classify_conversation(
            session, index, CID_A, project_identifier="project"
        )
    assert rel == syncstate.SyncRelation.UP_TO_DATE
    assert syncstate.op_counts().legacy_snapshot_decompressions == 1
    assert syncstate.op_counts().deep_snapshot_reads == 1


def test_persistent_cache_stores_digest_not_unit_hashes(sync_env):
    snap = _conversation([_msg(1, "A")], composer_id=CID_A)
    _commit_env(sync_env, [snap], [snap], digest=False)
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        syncstate.build_sync_plan(session, index)

    cache_path = sync_env["tmp"] / "cache" / "sync-semantics.json"
    payload = json.loads(cache_path.read_text())
    assert payload["snapshots"]
    assert list(payload["snapshots"]) == [f"project|{CID_A}"]
    for rec in payload["snapshots"].values():
        assert "unitHashes" not in rec
        assert rec["semanticDigest"].startswith("sha256:")
        assert rec["semanticDigestVersion"] == syncstate.SEMANTIC_DIGEST_VERSION
        assert rec["sourceIdentity"]
        assert "localPayloadVersion" not in rec
    assert payload["local"]
    for rec in payload["local"].values():
        assert "unitHashes" not in rec
        assert rec["semanticDigestVersion"] == syncstate.SEMANTIC_DIGEST_VERSION
        assert rec["localPayloadVersion"] == syncstate.LOCAL_PAYLOAD_VERSION
        assert rec["rowFingerprint"].startswith("sha256:")
        assert "blobRefs" in rec
        assert rec["blobFingerprint"].startswith("sha256:")


def test_cache_flush_uses_unique_tempfile(sync_env):
    cache = syncstate.SemanticsCache()
    cache.put_local(CID_A, "sha256:fp", [], "sha256:blobs", "sha256:digest")
    cache.flush()
    cache_dir = sync_env["tmp"] / "cache"
    assert (cache_dir / "sync-semantics.json").exists()
    assert not (cache_dir / "sync-semantics.tmp").exists()
    assert list(cache_dir.glob("sync-semantics-*.tmp")) == []


def test_explicit_blob_change_invalidates_local_cache(sync_env):
    snap = _conversation(
        [_msg(1, "A", blob_id="blob-aaa", blob_data="v1")],
        composer_id=CID_A,
        name="Blob",
    )
    _commit_env(sync_env, [snap], [snap], digest=True, gzip_cids={CID_A})
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        syncstate.build_sync_plan(session, index)
        first_fp = session.raw_fingerprint(CID_A)

    snap["contentBlobs"]["blob-aaa"] = "v2"
    gconn = _init_db(sync_env["global_db"])
    _write_local(gconn, snap)
    gconn.commit()
    gconn.close()

    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
        assert session.raw_fingerprint(CID_A) == first_fp
    assert plan.items[0].relation == syncstate.SyncRelation.DIVERGED
    assert syncstate.op_counts().local_semantic_rehashes == 1


def test_blob_ref_json_key_order_is_up_to_date():
    remote = _conversation([_msg(1, "hi")], composer_id=CID_A)
    remote["bubbleEntries"]["bubble-1"]["extra"] = {"p": "blob-A", "q": "blob-B"}
    remote["contentBlobs"] = {"blob-A": "A", "blob-B": "B"}
    local = _conversation([_msg(1, "hi")], composer_id=CID_A)
    local["bubbleEntries"]["bubble-1"]["extra"] = {"q": "blob-B", "p": "blob-A"}
    local["contentBlobs"] = {"blob-B": "B", "blob-A": "A"}
    assert _relation(local, remote) == syncstate.SyncRelation.UP_TO_DATE


def test_sync_without_snapshots_skips_db_and_does_not_push_never_pushed(sync_env, monkeypatch):
    never = _conversation([_msg(1, "new")], composer_id=CID_E, name="Never")
    _commit_env(sync_env, [never], [], digest=False)
    saves = {"n": 0}
    monkeypatch.setattr(
        export,
        "save_snapshot",
        lambda *a, **k: saves.__setitem__("n", saves["n"] + 1) or Path("x"),
    )
    monkeypatch.setattr(
        cli,
        "import_snapshot",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("import")),
    )
    _backend(monkeypatch)
    syncstate.reset_op_counts()
    cli.cmd_sync(type("Args", (), {"force": False})())
    assert saves["n"] == 0
    assert syncstate.op_counts().sqlite_backups == 0
    assert syncstate.op_counts().full_local_exports == 0


def test_sharded_snapshot_shard_change_invalidates_cache(sync_env):
    snap = _conversation([_msg(1, "A")], composer_id=CID_A, name="Shard")
    gconn = _init_db(sync_env["global_db"])
    _write_local(gconn, snap)
    gconn.commit()
    gconn.close()
    _write_workspace(sync_env["ws_dir"], [snap])
    _write_sharded_snapshot(sync_env["project_dir"], snap, parts=2)

    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
    assert plan.items[0].relation == syncstate.SyncRelation.UP_TO_DATE

    shard = sync_env["project_dir"] / f"{CID_A}.json.gz.01"
    shard.write_bytes(shard.read_bytes() + b"x")

    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
    assert plan.items[0].relation != syncstate.SyncRelation.UP_TO_DATE
    assert plan.items[0].relation == syncstate.SyncRelation.UNKNOWN
    assert syncstate.op_counts().deep_snapshot_reads >= 1


def test_directory_is_authoritative_over_meta_project_identifier(sync_env):
    snap = _conversation(
        [_msg(1, "A")],
        composer_id=CID_A,
        name="Misfiled",
        projectIdentifier="project-B",
    )
    _write_snapshot_file(
        sync_env["snaps"] / "project-A", snap, with_digest=True, gzip_body=True
    )

    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        assert index.get(CID_A, "project-B") is None
        rec = index.get(CID_A, "project-A")
        assert rec is not None
        assert rec.invalid_origin
        assert rec.project_identifier == "project-A"
        assert (
            syncstate.classify_conversation(
                session, index, CID_A, project_identifier="project-B"
            )
            == syncstate.SyncRelation.UNKNOWN
        )
        assert (
            syncstate.classify_conversation(
                session, index, CID_A, project_identifier="project-A"
            )
            == syncstate.SyncRelation.UNKNOWN
        )
        plan = syncstate.build_sync_plan(session, index)
    items_a = [i for i in plan.items if i.project_identifier == "project-A"]
    assert len(items_a) == 1
    assert items_a[0].relation == syncstate.SyncRelation.UNKNOWN


def test_snapshot_only_corrupt_gzip_is_unknown_before_writes(sync_env, monkeypatch, capsys):
    _write_workspace(sync_env["ws_dir"], [])
    project_dir = sync_env["project_dir"]
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / f"{CID_F}.json.gz").write_bytes(b"not-a-gzip-snapshot")
    (project_dir / f"{CID_F}.meta.json").write_text(
        json.dumps(
            {
                "composerId": CID_F,
                "name": "Broken",
                "projectIdentifier": "project",
            }
        )
    )

    backend = _backend(monkeypatch)
    imports = {"n": 0}
    saves = {"n": 0}
    monkeypatch.setattr(
        cli,
        "import_snapshot",
        lambda *a, **k: imports.__setitem__("n", imports["n"] + 1) or True,
    )
    monkeypatch.setattr(
        export,
        "save_snapshot",
        lambda *a, **k: saves.__setitem__("n", saves["n"] + 1) or Path("x"),
    )

    with pytest.raises(SystemExit) as exc:
        cli.cmd_sync(type("Args", (), {"force": False})())
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "could not be classified" in err
    assert imports["n"] == 0
    assert saves["n"] == 0
    assert backend.pushes == 0


def test_sharded_sidecar_digest_is_not_trusted_after_shard_change(sync_env):
    snap = _conversation([_msg(1, "A")], composer_id=CID_A, name="Shard")
    gconn = _init_db(sync_env["global_db"])
    _write_local(gconn, snap)
    gconn.commit()
    gconn.close()
    _write_workspace(sync_env["ws_dir"], [snap])
    _write_sharded_snapshot(sync_env["project_dir"], snap, parts=2, with_digest=True)

    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
    assert plan.items[0].relation == syncstate.SyncRelation.UP_TO_DATE

    shard = sync_env["project_dir"] / f"{CID_A}.json.gz.01"
    shard.write_bytes(shard.read_bytes() + b"x")

    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
    assert plan.items[0].relation != syncstate.SyncRelation.UP_TO_DATE
    assert plan.items[0].relation == syncstate.SyncRelation.UNKNOWN
    assert syncstate.op_counts().deep_snapshot_reads >= 1


def test_sidecar_digest_without_content_binding_is_not_trusted(sync_env):
    snap = _conversation([_msg(1, "A")], composer_id=CID_A)
    _commit_env(sync_env, [snap], [snap], digest=True, gzip_cids={CID_A})
    meta_path = sync_env["project_dir"] / f"{CID_A}.meta.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("snapshotContentDigest", None)
    meta_path.write_text(json.dumps(meta))

    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        rel = syncstate.classify_conversation(
            session, index, CID_A, project_identifier="project"
        )
    assert rel == syncstate.SyncRelation.UP_TO_DATE
    assert syncstate.op_counts().legacy_snapshot_decompressions == 1
    assert syncstate.op_counts().deep_snapshot_reads == 1


def test_snapshot_cache_keeps_one_entry_per_conversation(sync_env):
    first = _conversation([_msg(1, "A")], composer_id=CID_A)
    _commit_env(sync_env, [first], [first], digest=False)
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        syncstate.build_sync_plan(session, index)

    second = _conversation([_msg(1, "B")], composer_id=CID_A)
    _write_snapshot_file(sync_env["project_dir"], second, with_digest=False, gzip_body=True)
    gconn = _init_db(sync_env["global_db"])
    _write_local(gconn, second)
    gconn.commit()
    gconn.close()

    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        syncstate.build_sync_plan(session, index)

    payload = json.loads((sync_env["tmp"] / "cache" / "sync-semantics.json").read_text())
    assert list(payload["snapshots"]) == [f"project|{CID_A}"]
    assert payload["snapshots"][f"project|{CID_A}"]["sourceIdentity"]

