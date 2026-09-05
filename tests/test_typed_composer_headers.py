"""Typed composerHeaders discovery and registration (v0.9.13)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from cursor_saves import cli, db, export, importer, paths, pull, syncstate, typed_headers
from tests.test_syncstate import (
    CID_A,
    CID_B,
    CID_C,
    CID_D,
    CID_E,
    PROJECT_PATH,
    WS_HASH,
    _backend,
    _commit_env,
    _conversation,
    _init_db,
    _msg,
    _put_json,
    _write_local,
    _write_snapshot_file,
    _write_workspace,
)
from tests.test_sync_workspace import _args, _use_workspaces, _ws

pytest_plugins = ["tests.test_syncstate"]

_REAL_BUILD_HEADERS = paths._build_global_headers_map

WS_HASH_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
CREATED_AT = 1_700_000_000_000
LAST_UPDATED = 1_700_000_100_000

IPPO_CHATS = (
    (CID_A, "Side Chat"),
    (CID_B, "MDC creation based on project"),
    (CID_C, "Database structure review"),
    (CID_D, "Betflag horse racing data"),
)


def _create_typed_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS composerHeaders (
            composerId TEXT PRIMARY KEY,
            workspaceId TEXT,
            createdAt INTEGER,
            lastUpdatedAt INTEGER,
            isArchived INTEGER,
            isSubagent INTEGER,
            recency INTEGER,
            checkpointAt INTEGER,
            value TEXT
        )
        """
    )


def _workspace_json(ws_dir: Path, path: str = PROJECT_PATH) -> None:
    ws_dir.mkdir(parents=True, exist_ok=True)
    (ws_dir / "workspace.json").write_text(json.dumps({"folder": f"file://{path}"}))


def _legacy_header(cid: str, name: str, ws_hash: str = WS_HASH, **extra) -> dict:
    header = {
        "type": "head",
        "composerId": cid,
        "name": name,
        "createdAt": CREATED_AT,
        "lastUpdatedAt": LAST_UPDATED,
        "unifiedMode": "agent",
        "forceMode": extra.pop("forceMode", "edit"),
        "workspaceIdentifier": {"id": ws_hash},
    }
    header.update(extra)
    return header


def _write_legacy_json(conn: sqlite3.Connection, headers: list[dict]) -> None:
    _put_json(
        conn,
        "composer.composerHeaders",
        {"allComposers": headers},
        table="ItemTable",
    )


def _insert_typed(
    conn: sqlite3.Connection,
    cid: str,
    ws_hash: str,
    value: dict,
    *,
    created_at: int = CREATED_AT,
    last_updated: int = LAST_UPDATED,
    recency: int | None = None,
    is_archived: int | None = None,
    is_subagent: int | None = None,
) -> None:
    archived = (
        is_archived
        if is_archived is not None
        else (1 if value.get("isArchived") else 0)
    )
    subagent = (
        is_subagent
        if is_subagent is not None
        else (
            1
            if value.get("isSubagent") or value.get("isBestOfNSubcomposer")
            else 0
        )
    )
    conn.execute(
        "INSERT INTO composerHeaders ("
        "composerId, workspaceId, createdAt, lastUpdatedAt, "
        "isArchived, isSubagent, recency, checkpointAt, value"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)",
        (
            cid,
            ws_hash,
            created_at,
            last_updated,
            archived,
            subagent,
            last_updated if recency is None else recency,
            json.dumps(value),
        ),
    )


def _typed_row(global_db: Path, cid: str):
    conn = sqlite3.connect(str(global_db))
    try:
        return typed_headers.get_typed_row(conn, cid)
    finally:
        conn.close()


def _typed_ids(global_db: Path) -> set[str]:
    conn = sqlite3.connect(str(global_db))
    try:
        if not typed_headers.has_typed_headers_table(conn):
            return set()
        rows = conn.execute("SELECT composerId FROM composerHeaders").fetchall()
        return {r[0] for r in rows}
    finally:
        conn.close()


def _content_blob(global_db: Path, cid: str) -> tuple[str | None, dict[str, str]]:
    with db.CursorDB(global_db) as cdb:
        cd = cdb.get_item(f"composerData:{cid}")
        bubbles = {}
        for key in cdb.list_keys(f"bubbleId:{cid}:"):
            bubbles[key] = cdb.get_item(key)
        return cd, bubbles


def _active(cid: str, name: str, text: str | None = None) -> dict:
    snap = _conversation(
        [_msg(1, text or f"m-{cid[:4]}")],
        composer_id=cid,
        name=name,
    )
    snap["composerData"]["createdAt"] = CREATED_AT
    snap["composerData"]["lastUpdatedAt"] = LAST_UPDATED
    snap["composerData"]["unifiedMode"] = "agent"
    snap["composerData"]["forceMode"] = "chat"
    return snap


def _ippotrack_env(sync_env, *, with_typed_table: bool = True, typed_rows: bool = False):
    snaps = [_active(cid, name) for cid, name in IPPO_CHATS]
    _commit_env(sync_env, snaps, snaps)
    _workspace_json(sync_env["ws_dir"])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    if with_typed_table:
        _create_typed_table(gconn)
    _write_legacy_json(
        gconn,
        [_legacy_header(cid, name) for cid, name in IPPO_CHATS],
    )
    if typed_rows:
        for cid, name in IPPO_CHATS:
            _insert_typed(
                gconn,
                cid,
                WS_HASH,
                _legacy_header(cid, name),
            )
    gconn.commit()
    gconn.close()
    return snaps


def _restore_real_headers(monkeypatch) -> None:
    monkeypatch.setattr(paths, "_build_global_headers_map", _REAL_BUILD_HEADERS)


def _plan(sync_env, monkeypatch=None):
    if monkeypatch is not None:
        _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        return syncstate.build_sync_plan(
            session,
            index,
            target_workspace={
                "path": PROJECT_PATH,
                "workspace_dir": sync_env["ws_dir"],
                "host": None,
                "type": "local",
            },
        )


def test_no_typed_table_keeps_legacy_union(sync_env):
    snap = _active(CID_A, "Legacy")
    _commit_env(sync_env, [snap], [snap])
    ids = paths.get_workspace_composer_ids(sync_env["ws_dir"] / "state.vscdb")
    assert CID_A in ids
    conn = sqlite3.connect(str(sync_env["global_db"]))
    assert not typed_headers.has_typed_headers_table(conn)
    conn.close()


def test_typed_table_discovers_typed_row(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    snap = _active(CID_A, "Typed")
    _commit_env(sync_env, [snap], [snap])
    _write_workspace(sync_env["ws_dir"], [])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    _insert_typed(gconn, CID_A, WS_HASH, _legacy_header(CID_A, "Typed"))
    gconn.commit()
    gconn.close()
    with syncstate.SyncReadSession() as session:
        ids = paths.get_workspace_composer_ids(
            sync_env["ws_dir"] / "state.vscdb", session=session
        )
    assert ids == [CID_A]


def test_typed_row_wins_legacy_workspace_conflict(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    snap = _active(CID_A, "Conflict")
    ws_b = sync_env["tmp"] / "cursor" / "workspaceStorage" / WS_HASH_B
    _commit_env(sync_env, [snap], [snap])
    _write_workspace(sync_env["ws_dir"], [snap])
    _write_workspace(ws_b, [])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    _write_legacy_json(gconn, [_legacy_header(CID_A, "Conflict", WS_HASH)])
    _insert_typed(gconn, CID_A, WS_HASH_B, _legacy_header(CID_A, "Conflict", WS_HASH_B))
    gconn.commit()
    gconn.close()
    with syncstate.SyncReadSession() as session:
        ids_a = paths.get_workspace_composer_ids(
            sync_env["ws_dir"] / "state.vscdb", session=session
        )
        ids_b = paths.get_workspace_composer_ids(
            ws_b / "state.vscdb", session=session
        )
    assert CID_A not in ids_a
    assert ids_b == [CID_A]


def test_legacy_only_active_stays_discoverable(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env)
    plan = _plan(sync_env, monkeypatch)
    assert {i.composer_id for i in plan.items} == {CID_A, CID_B, CID_C, CID_D}
    assert {i.relation for i in plan.items} == {syncstate.SyncRelation.UP_TO_DATE}
    assert {i.registration for i in plan.items} == {
        typed_headers.RegistrationHealth.LEGACY_ONLY
    }


def test_missing_typed_row_is_repairable(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    snap = _active(CID_A, "Missing")
    _commit_env(sync_env, [snap], [snap])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    plan = _plan(sync_env, monkeypatch)
    assert plan.items[0].registration == typed_headers.RegistrationHealth.LEGACY_ONLY
    assert importer.repair_typed_registrations(plan) == 1
    assert _typed_ids(sync_env["global_db"]) == {CID_A}


def test_repair_does_not_modify_composer_or_bubbles(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env)
    before = {cid: _content_blob(sync_env["global_db"], cid) for cid, _ in IPPO_CHATS}
    plan = _plan(sync_env, monkeypatch)
    assert importer.repair_typed_registrations(plan) == 4
    after = {cid: _content_blob(sync_env["global_db"], cid) for cid, _ in IPPO_CHATS}
    assert before == after


def test_repair_preserves_created_at_and_recency(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env)
    now_ms = int(time.time() * 1000)
    plan = _plan(sync_env, monkeypatch)
    importer.repair_typed_registrations(plan)
    row = _typed_row(sync_env["global_db"], CID_A)
    assert row is not None
    assert row.created_at == CREATED_AT
    assert row.recency == LAST_UPDATED
    assert row.recency < now_ms - 1000
    header = row.header
    assert header["forceMode"] == "edit"
    assert "subtitle" not in header
    assert "contextUsagePercent" not in header
    assert header["totalLinesAdded"] == 0


def test_existing_rich_typed_header_is_not_clobbered(sync_env):
    snap = _active(CID_A, "Rich")
    _commit_env(sync_env, [snap], [snap])
    rich = _legacy_header(CID_A, "Rich")
    rich["subtitle"] = "keep-me"
    rich["contextUsagePercent"] = 42
    rich["totalLinesAdded"] = 99
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    _insert_typed(gconn, CID_A, WS_HASH, rich)
    gconn.commit()
    gconn.close()
    with db.CursorDB(sync_env["global_db"]) as cdb:
        result = typed_headers.register_current(
            cdb,
            CID_A,
            WS_HASH,
            snap["composerData"],
            workspace_identifier={"id": WS_HASH},
            update_stable=True,
        )
    assert result == "preserved"
    row = _typed_row(sync_env["global_db"], CID_A)
    assert row.header["subtitle"] == "keep-me"
    assert row.header["contextUsagePercent"] == 42
    assert row.header["totalLinesAdded"] == 99


def test_typed_row_other_workspace_is_fail_closed(sync_env):
    snap = _active(CID_A, "Other")
    _commit_env(sync_env, [snap], [snap])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    _insert_typed(gconn, CID_A, WS_HASH_B, _legacy_header(CID_A, "Other", WS_HASH_B))
    gconn.commit()
    gconn.close()
    with db.CursorDB(sync_env["global_db"]) as cdb:
        with pytest.raises(typed_headers.MisregisteredHeaderError):
            typed_headers.register_current(
                cdb,
                CID_A,
                WS_HASH,
                snap["composerData"],
                workspace_identifier={"id": WS_HASH},
            )
    row = _typed_row(sync_env["global_db"], CID_A)
    assert row.workspace_id == WS_HASH_B


def test_behind_import_creates_typed_row(sync_env, monkeypatch):
    remote = _conversation(
        [_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="Behind"
    )
    local = _conversation([_msg(1, "A")], composer_id=CID_A, name="Behind")
    _commit_env(sync_env, [local], [remote], digest=False)
    _workspace_json(sync_env["ws_dir"])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    _backend(monkeypatch)
    cli.cmd_sync(_args(workspace=WS_HASH[:8]))
    assert CID_A in _typed_ids(sync_env["global_db"])


def test_verify_imported_fails_without_typed_row(sync_env):
    snap = _active(CID_A, "Verify")
    _commit_env(sync_env, [snap], [snap])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    with db.CursorDB(sync_env["global_db"]) as cdb:
        assert not importer._verify_imported(
            cdb,
            CID_A,
            snap["bubbleEntries"],
            workspace_dir=sync_env["ws_dir"],
        )


def test_healthy_noop_sync_does_not_write(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env, typed_rows=True)
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    _backend(monkeypatch)
    monkeypatch.setattr(
        cli, "import_snapshot", lambda *a, **k: (_ for _ in ()).throw(AssertionError())
    )
    writes_before = syncstate.op_counts().cursor_write_connections
    backups_before = syncstate.op_counts().safety_global_backups
    syncstate.reset_op_counts()
    cli.cmd_sync(type("Args", (), {"force": False})())
    counts = syncstate.op_counts()
    assert counts.safety_global_backups == 0
    assert counts.cursor_write_connections == 0
    assert writes_before == 0
    assert backups_before == 0


def test_up_to_date_missing_registration_writes_only_header(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env)
    before = {cid: _content_blob(sync_env["global_db"], cid) for cid, _ in IPPO_CHATS}
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    _backend(monkeypatch)
    monkeypatch.setattr(
        cli, "import_snapshot", lambda *a, **k: (_ for _ in ()).throw(AssertionError())
    )
    cli.cmd_sync(type("Args", (), {"force": False})())
    assert _typed_ids(sync_env["global_db"]) == {CID_A, CID_B, CID_C, CID_D}
    after = {cid: _content_blob(sync_env["global_db"], cid) for cid, _ in IPPO_CHATS}
    assert before == after
    assert syncstate.op_counts().safety_global_backups == 1


def test_purge_removes_typed_row(sync_env):
    snap = _active(CID_A, "Purge")
    _commit_env(sync_env, [snap], [snap])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    _insert_typed(gconn, CID_A, WS_HASH, _legacy_header(CID_A, "Purge"))
    gconn.commit()
    gconn.close()
    deleted, _keys = importer.purge_chats([CID_A], force=True)
    assert deleted == 1
    assert CID_A not in _typed_ids(sync_env["global_db"])


def test_copy_creates_typed_row(sync_env):
    snap = _active(CID_A, "Copy")
    _commit_env(sync_env, [snap], [snap])
    target = sync_env["tmp"] / "cursor" / "workspaceStorage" / WS_HASH_B
    _write_workspace(target, [])
    _workspace_json(target, "/home/user/other")
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    ok, fail = importer.copy_between_workspaces(
        [CID_A],
        sync_env["ws_dir"],
        target,
        PROJECT_PATH,
        "/home/user/other",
        force=True,
    )
    assert ok == 1
    assert fail == 0
    copied = _typed_ids(sync_env["global_db"])
    assert CID_A not in copied
    assert len(copied) == 1
    new_id = next(iter(copied))
    row = _typed_row(sync_env["global_db"], new_id)
    assert row.workspace_id == WS_HASH_B


def test_migrate_dry_run_detects_legacy_only(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env)
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    migrated, already = importer.migrate_to_global_headers(dry_run=True, force=True)
    assert migrated == 4
    assert already == 0
    assert _typed_ids(sync_env["global_db"]) == set()


def test_migrate_ignores_typed_present(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env, typed_rows=True)
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    migrated, already = importer.migrate_to_global_headers(dry_run=True, force=True)
    assert migrated == 0
    assert already == 4


def test_typed_catalog_reuses_epoch_connection(sync_env, monkeypatch):
    gconn = _init_db(sync_env["global_db"])
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    global_path = str(sync_env["global_db"].resolve())
    with syncstate.SyncReadSession() as session:
        session.headers_map()
        real = sqlite3.connect

        def guarded(*args, **kwargs):
            target = str(args[0]) if args else ""
            if global_path in target:
                raise AssertionError(f"opened a second global connection: {target}")
            return real(*args, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", guarded)
        catalog = session.typed_catalog()
        assert catalog == {}
        assert session.typed_table_exists() is True


def test_ippotrack_repair_via_sync(sync_env, monkeypatch, capsys):
    _restore_real_headers(monkeypatch)
    snaps = _ippotrack_env(sync_env)
    before = {cid: _content_blob(sync_env["global_db"], cid) for cid, _ in IPPO_CHATS}
    before_digests = {
        cid: syncstate.snapshot_semantic_digest(snap) for (cid, _), snap in zip(IPPO_CHATS, snaps)
    }
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    _backend(monkeypatch)
    monkeypatch.setattr(cli, "_ensure_synced", lambda: None)
    cli.cmd_status(_args(workspace="1"))
    status = capsys.readouterr().out
    assert "Registered:              0" in status
    assert "Repair needed:           4" in status

    cli.cmd_sync(type("Args", (), {"force": False})())
    out = capsys.readouterr().out
    assert "Repaired Cursor registration for 4" in out
    assert _typed_ids(sync_env["global_db"]) == {CID_A, CID_B, CID_C, CID_D}
    after = {cid: _content_blob(sync_env["global_db"], cid) for cid, _ in IPPO_CHATS}
    assert before == after
    for (cid, _), snap in zip(IPPO_CHATS, snaps):
        assert syncstate.snapshot_semantic_digest(snap) == before_digests[cid]

    cli.cmd_status(_args(workspace="1"))
    status = capsys.readouterr().out
    assert "Registered:              4" in status
    assert "Repair needed:           0" in status


def _typed_fingerprint(global_db: Path):
    conn = sqlite3.connect(str(global_db))
    try:
        return conn.execute(
            "SELECT composerId, workspaceId, createdAt, lastUpdatedAt, recency, value "
            "FROM composerHeaders ORDER BY composerId"
        ).fetchall()
    finally:
        conn.close()


def _clear_typed_rows(global_db: Path) -> None:
    conn = sqlite3.connect(str(global_db))
    try:
        conn.execute("DELETE FROM composerHeaders")
        conn.commit()
    finally:
        conn.close()


def test_pull_up_to_date_missing_typed_creates_header_only(sync_env, monkeypatch):
    _ippotrack_env(sync_env)
    before = {cid: _content_blob(sync_env["global_db"], cid) for cid, _ in IPPO_CHATS}
    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    assert result.imported == 0
    assert result.repaired == 4
    assert _typed_ids(sync_env["global_db"]) == {CID_A, CID_B, CID_C, CID_D}
    after = {cid: _content_blob(sync_env["global_db"], cid) for cid, _ in IPPO_CHATS}
    assert before == after


def test_pull_up_to_date_healthy_typed_is_true_noop(sync_env, monkeypatch, capsys):
    _ippotrack_env(sync_env, typed_rows=True)
    syncstate.reset_op_counts()
    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=False,
    )
    out = capsys.readouterr().out
    counts = syncstate.op_counts()
    assert result.imported == 0
    assert result.repaired == 0
    assert counts.safety_global_backups == 0
    assert counts.cursor_write_connections == 0
    assert counts.cursor_running_checks == 0
    assert "Nothing to import." in out


def test_pull_all_content_present_does_not_early_return(sync_env, monkeypatch, capsys):
    _ippotrack_env(sync_env)
    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    out = capsys.readouterr().out
    assert result.repaired == 4
    assert "Nothing to import." not in out
    assert "Repaired Cursor registration for 4" in out
    assert syncstate.op_counts().safety_global_backups == 1


def test_pull_registration_repair_leaves_payload_unchanged(sync_env, monkeypatch):
    _ippotrack_env(sync_env)
    before = {cid: _content_blob(sync_env["global_db"], cid) for cid, _ in IPPO_CHATS}
    pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    after = {cid: _content_blob(sync_env["global_db"], cid) for cid, _ in IPPO_CHATS}
    assert before == after


def test_sync_and_pull_same_registration_for_up_to_date(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env)
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    _backend(monkeypatch)
    pull_result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    assert pull_result.repaired == 4
    after_pull = _typed_fingerprint(sync_env["global_db"])
    _clear_typed_rows(sync_env["global_db"])
    paths.invalidate_headers_cache()
    cli.cmd_sync(type("Args", (), {"force": False})())
    after_sync = _typed_fingerprint(sync_env["global_db"])
    assert after_pull == after_sync
    assert {row[0] for row in after_sync} == {CID_A, CID_B, CID_C, CID_D}


def _ws_registration_state(ws_dir: Path) -> tuple[dict, dict]:
    with db.CursorDB(ws_dir / "state.vscdb") as cdb:
        data = cdb.get_json("composer.composerData", table="ItemTable") or {}
        panes = {}
        for key in cdb.list_keys(
            "workbench.panel.composerChatViewPane.", table="ItemTable"
        ):
            panes[key] = cdb.get_json(key, table="ItemTable")
        return data, panes


def test_same_uri_different_id_is_stale_not_foreign():
    ident_a = {
        "id": WS_HASH,
        "uri": {
            "scheme": "file",
            "fsPath": PROJECT_PATH,
            "external": f"file://{PROJECT_PATH}",
        },
    }
    ident_b = {
        "id": WS_HASH_B,
        "uri": {
            "scheme": "file",
            "fsPath": PROJECT_PATH,
            "external": f"file://{PROJECT_PATH}",
        },
    }
    assert typed_headers.classify_workspace_binding(
        typed_workspace_id=WS_HASH,
        target_workspace_id=WS_HASH_B,
        typed_identifier=ident_a,
        target_identifier=ident_b,
    ) == typed_headers.WorkspaceBinding.STALE_WORKSPACE_ID_SAME_URI
    assert typed_headers.classify_registration(
        typed_table_exists=True,
        typed_workspace_id=WS_HASH,
        target_workspace_id=WS_HASH_B,
        in_legacy_sources=False,
        typed_identifier=ident_a,
        target_identifier=ident_b,
    ) == typed_headers.RegistrationHealth.STALE_WORKSPACE
    foreign = {
        "id": WS_HASH_B,
        "uri": {
            "scheme": "file",
            "fsPath": "/home/user/other",
            "external": "file:///home/user/other",
        },
    }
    assert typed_headers.classify_workspace_binding(
        typed_workspace_id=WS_HASH,
        target_workspace_id=WS_HASH_B,
        typed_identifier=ident_a,
        target_identifier=foreign,
    ) == typed_headers.WorkspaceBinding.FOREIGN_WORKSPACE


def test_registration_repair_does_not_change_selected_focused_pane(
    sync_env, monkeypatch
):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env)
    pane_key = "workbench.panel.composerChatViewPane.main"
    pane_val = {f"agent.view.{CID_A}": {"pinned": True}}
    with db.CursorDB(sync_env["ws_dir"] / "state.vscdb") as cdb:
        data = cdb.get_json("composer.composerData", table="ItemTable") or {}
        data["selectedComposerIds"] = [CID_B]
        data["lastFocusedComposerIds"] = [CID_C]
        cdb.write_json("composer.composerData", data, table="ItemTable")
        cdb.write_json(pane_key, pane_val, table="ItemTable")
    before = _ws_registration_state(sync_env["ws_dir"])
    plan = _plan(sync_env, monkeypatch)
    assert importer.repair_typed_registrations(plan) == 4
    assert _ws_registration_state(sync_env["ws_dir"]) == before
    assert _typed_ids(sync_env["global_db"]) == {CID_A, CID_B, CID_C, CID_D}


def test_archived_legacy_only_stays_archived_after_typed_migration(
    sync_env, monkeypatch
):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env)
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    headers = [_legacy_header(cid, name) for cid, name in IPPO_CHATS]
    headers[0]["isArchived"] = True
    _write_legacy_json(gconn, headers)
    gconn.commit()
    gconn.close()
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    migrated, _already = importer.migrate_to_global_headers(dry_run=False, force=True)
    assert migrated == 4
    row = _typed_row(sync_env["global_db"], CID_A)
    assert row is not None
    assert row.is_archived == 1
    assert row.header.get("isArchived") is True
    assert _typed_row(sync_env["global_db"], CID_B).is_archived == 0


def test_subagent_is_not_migrated_as_top_level(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env)
    extra = _active(CID_E, "Sub task")
    extra["composerData"]["isSubagent"] = True
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _write_local(gconn, extra)
    headers = [_legacy_header(cid, name) for cid, name in IPPO_CHATS]
    headers.append(_legacy_header(CID_E, "Sub task", isSubagent=True))
    _write_legacy_json(gconn, headers)
    gconn.commit()
    gconn.close()
    _write_workspace(
        sync_env["ws_dir"],
        [_active(cid, name) for cid, name in IPPO_CHATS] + [extra],
    )
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    migrated, _already = importer.migrate_to_global_headers(dry_run=True, force=True)
    assert migrated == 4
    assert CID_E not in _typed_ids(sync_env["global_db"])

    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _insert_typed(
        gconn,
        CID_E,
        WS_HASH,
        _legacy_header(CID_E, "Sub task", isSubagent=True),
    )
    gconn.commit()
    gconn.close()
    paths.invalidate_headers_cache()
    with syncstate.SyncReadSession() as session:
        ids = paths.get_workspace_composer_ids(
            sync_env["ws_dir"] / "state.vscdb", session=session
        )
    assert CID_E not in ids
    convos = export.get_workspace_conversations(
        PROJECT_PATH, workspace_dir=sync_env["ws_dir"], session=None
    )
    assert CID_E not in {c.get("composerId") for c in convos}


def test_typed_row_appears_between_preflight_and_write_no_clobber(
    sync_env, monkeypatch
):
    snap = _active(CID_A, "Race")
    _commit_env(sync_env, [snap], [snap])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    original = typed_headers._insert_synthetic

    def raced(conn, composer_id, workspace_id, composer_data, **kwargs):
        rich = _legacy_header(composer_id, "Race")
        rich["subtitle"] = "cursor-won"
        other = sqlite3.connect(str(sync_env["global_db"]))
        try:
            _insert_typed(other, composer_id, workspace_id, rich)
            other.commit()
        finally:
            other.close()
        return original(conn, composer_id, workspace_id, composer_data, **kwargs)

    monkeypatch.setattr(typed_headers, "_insert_synthetic", raced)
    with db.CursorDB(sync_env["global_db"]) as cdb:
        result = typed_headers.register_current(
            cdb,
            CID_A,
            WS_HASH,
            snap["composerData"],
            workspace_identifier={"id": WS_HASH},
        )
    assert result == "preserved"
    row = _typed_row(sync_env["global_db"], CID_A)
    assert row.header["subtitle"] == "cursor-won"


def test_typed_row_appears_in_foreign_workspace_between_preflight_and_write_aborts(
    sync_env, monkeypatch
):
    snap = _active(CID_A, "Foreign race")
    _commit_env(sync_env, [snap], [snap])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    original = typed_headers._insert_synthetic

    def raced(conn, composer_id, workspace_id, composer_data, **kwargs):
        other = sqlite3.connect(str(sync_env["global_db"]))
        try:
            _insert_typed(
                other,
                composer_id,
                WS_HASH_B,
                _legacy_header(composer_id, "Foreign race", WS_HASH_B),
            )
            other.commit()
        finally:
            other.close()
        return original(conn, composer_id, workspace_id, composer_data, **kwargs)

    monkeypatch.setattr(typed_headers, "_insert_synthetic", raced)
    with db.CursorDB(sync_env["global_db"]) as cdb:
        with pytest.raises(typed_headers.MisregisteredHeaderError) as exc:
            typed_headers.register_current(
                cdb,
                CID_A,
                WS_HASH,
                snap["composerData"],
                workspace_identifier={"id": WS_HASH},
            )
    assert exc.value.binding == typed_headers.WorkspaceBinding.FOREIGN_WORKSPACE
    row = _typed_row(sync_env["global_db"], CID_A)
    assert row.workspace_id == WS_HASH_B


def test_payload_disappears_before_repair_no_orphan_header(sync_env):
    snap = _active(CID_A, "Gone")
    _commit_env(sync_env, [snap], [snap])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    with db.CursorDB(sync_env["global_db"]) as cdb:
        cdb.delete_keys([f"composerData:{CID_A}"])
        result = typed_headers.register_current(
            cdb,
            CID_A,
            WS_HASH,
            snap["composerData"],
            workspace_identifier={"id": WS_HASH},
        )
    assert result == "skipped_inactive"
    assert CID_A not in _typed_ids(sync_env["global_db"])


def test_incompatible_composer_headers_schema_no_write(sync_env, monkeypatch):
    snap = _active(CID_A, "Schema")
    _commit_env(sync_env, [snap], [snap])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    gconn.execute("CREATE TABLE composerHeaders (composerId TEXT PRIMARY KEY)")
    gconn.commit()
    gconn.close()
    with db.CursorDB(sync_env["global_db"]) as cdb:
        with pytest.raises(typed_headers.TypedSchemaError):
            typed_headers.register_current(
                cdb,
                CID_A,
                WS_HASH,
                snap["composerData"],
                workspace_identifier={"id": WS_HASH},
            )
    conn = sqlite3.connect(str(sync_env["global_db"]))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(composerHeaders)")}
        rows = conn.execute("SELECT composerId FROM composerHeaders").fetchall()
    finally:
        conn.close()
    assert "workspaceId" not in cols
    assert rows == []
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    with pytest.raises(typed_headers.TypedSchemaError):
        importer.migrate_to_global_headers(dry_run=False, force=True)


def test_import_exposes_header_only_after_payload_written(sync_env, monkeypatch):
    snap = _active(CID_E, "Imported")
    path = _write_snapshot_file(sync_env["project_dir"], snap)
    _write_workspace(sync_env["ws_dir"], [])
    _workspace_json(sync_env["ws_dir"])
    gconn = _init_db(sync_env["global_db"])
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    seen: list[str] = []
    real = typed_headers.register_current

    def wrapped(cdb, composer_id, *args, **kwargs):
        assert cdb.get_json(f"composerData:{composer_id}") is not None
        seen.append(composer_id)
        return real(cdb, composer_id, *args, **kwargs)

    monkeypatch.setattr(typed_headers, "register_current", wrapped)
    assert importer.import_snapshot(
        path,
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        skip_backup=True,
    )
    assert seen == [CID_E]
    assert CID_E in _typed_ids(sync_env["global_db"])


def test_migrate_ignores_empty_selected_focused_draft(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env)
    empty_id = "4a7aaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    with db.CursorDB(sync_env["global_db"]) as cdb:
        cdb.write_json(
            f"composerData:{empty_id}",
            {
                "composerId": empty_id,
                "name": "",
                "fullConversationHeadersOnly": [],
            },
        )
    with db.CursorDB(sync_env["ws_dir"] / "state.vscdb") as cdb:
        data = cdb.get_json("composer.composerData", table="ItemTable") or {}
        data["selectedComposerIds"] = [empty_id]
        data["lastFocusedComposerIds"] = [empty_id]
        cdb.write_json("composer.composerData", data, table="ItemTable")
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    migrated, already = importer.migrate_to_global_headers(dry_run=True, force=True)
    assert migrated == 4
    assert already == 0
    assert empty_id not in _typed_ids(sync_env["global_db"])


def test_migrate_skips_ambiguous_cid_claimed_by_multiple_workspaces(
    sync_env, monkeypatch
):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env)
    ws_b = sync_env["tmp"] / "cursor" / "workspaceStorage" / WS_HASH_B
    _write_workspace(ws_b, [_active(CID_A, "Side Chat")])
    _workspace_json(ws_b, "/home/user/other")
    _use_workspaces(
        monkeypatch,
        [
            _ws(sync_env["ws_dir"], PROJECT_PATH),
            _ws(ws_b, "/home/user/other"),
        ],
    )
    migrated, _already = importer.migrate_to_global_headers(dry_run=True, force=True)
    assert migrated == 3


def test_repair_command_ensures_typed_registration(sync_env, monkeypatch):
    import base64

    blob_id = "ab" * 16
    snap = _active(CID_A, "Blobs")
    snap["agentBlobs"] = {blob_id: base64.b64encode(b"kv-bytes").decode()}
    _commit_env(sync_env, [snap], [snap], digest=False)
    _workspace_json(sync_env["ws_dir"])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    monkeypatch.setattr(export, "_extract_agent_blob_ids", lambda _cd: {blob_id})
    cli.cmd_repair(type("Args", (), {})())
    assert CID_A in _typed_ids(sync_env["global_db"])
    with db.CursorDB(sync_env["global_db"]) as cdb:
        assert (
            cdb.get_item_binary(f"agentKv:blob:{blob_id}", table="cursorDiskKV")
            == b"kv-bytes"
        )


def test_global_and_workspace_sync_same_registration_invariant(
    sync_env, monkeypatch
):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env)
    extra = _active(CID_E, "Other workspace")
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _write_local(gconn, extra)
    gconn.commit()
    gconn.close()
    ws_b = sync_env["tmp"] / "cursor" / "workspaceStorage" / WS_HASH_B
    _write_workspace(ws_b, [extra])
    _workspace_json(ws_b, "/home/user/other")
    _use_workspaces(
        monkeypatch,
        [
            _ws(sync_env["ws_dir"], PROJECT_PATH),
            _ws(ws_b, "/home/user/other"),
        ],
    )
    _backend(monkeypatch)
    cli.cmd_sync(_args(workspace=WS_HASH[:8]))
    scoped = _typed_fingerprint(sync_env["global_db"])
    assert {row[0] for row in scoped} == {CID_A, CID_B, CID_C, CID_D}
    _clear_typed_rows(sync_env["global_db"])
    paths.invalidate_headers_cache()
    cli.cmd_sync(type("Args", (), {"force": False})())
    worldwide = _typed_fingerprint(sync_env["global_db"])
    assert {row[0] for row in worldwide} == {CID_A, CID_B, CID_C, CID_D, CID_E}
    assert [row for row in worldwide if row[0] != CID_E] == scoped


def _make_incompatible_typed(global_db: Path) -> None:
    conn = sqlite3.connect(str(global_db))
    conn.execute("DROP TABLE IF EXISTS composerHeaders")
    conn.execute("CREATE TABLE composerHeaders (composerId TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()
    paths.invalidate_headers_cache()


def _ippotrack_local_only(sync_env, *, typed_rows: bool = False):
    snaps = [_active(cid, name) for cid, name in IPPO_CHATS]
    _commit_env(sync_env, snaps, [])
    _workspace_json(sync_env["ws_dir"])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    _write_legacy_json(
        gconn,
        [_legacy_header(cid, name) for cid, name in IPPO_CHATS],
    )
    if typed_rows:
        for cid, name in IPPO_CHATS:
            _insert_typed(gconn, cid, WS_HASH, _legacy_header(cid, name))
    gconn.commit()
    gconn.close()
    return snaps


def _inject_foreign_typed_row(sync_env, monkeypatch, cid: str):
    original = typed_headers._insert_synthetic

    def raced(conn, composer_id, workspace_id, composer_data, **kwargs):
        if composer_id == cid:
            other = sqlite3.connect(str(sync_env["global_db"]))
            try:
                _insert_typed(
                    other,
                    composer_id,
                    WS_HASH_B,
                    _legacy_header(composer_id, "race", WS_HASH_B),
                )
                other.commit()
            finally:
                other.close()
        return original(conn, composer_id, workspace_id, composer_data, **kwargs)

    monkeypatch.setattr(typed_headers, "_insert_synthetic", raced)


def _plant_foreign_typed(sync_env, cid: str, name: str = "Foreign") -> None:
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    _insert_typed(
        gconn,
        cid,
        WS_HASH_B,
        _legacy_header(cid, name, WS_HASH_B),
    )
    gconn.commit()
    gconn.close()
    paths.invalidate_headers_cache()


def _stale_header(cid: str, name: str) -> dict:
    header = _legacy_header(cid, name, WS_HASH_B)
    header["workspaceIdentifier"] = {
        "id": WS_HASH_B,
        "uri": {
            "scheme": "file",
            "fsPath": PROJECT_PATH,
            "external": f"file://{PROJECT_PATH}",
        },
    }
    return header


def test_import_file_typed_schema_error_leaves_state_unchanged(sync_env, monkeypatch):
    snap = _active(CID_E, "Incoming")
    path = _write_snapshot_file(sync_env["project_dir"], snap)
    _write_workspace(sync_env["ws_dir"], [])
    _workspace_json(sync_env["ws_dir"])
    gconn = _init_db(sync_env["global_db"])
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    _make_incompatible_typed(sync_env["global_db"])
    before_ws = _ws_registration_state(sync_env["ws_dir"])
    before_blob = _content_blob(sync_env["global_db"], CID_E)
    with pytest.raises(typed_headers.TypedSchemaError):
        importer.import_snapshot(
            path,
            PROJECT_PATH,
            target_workspace_dir=sync_env["ws_dir"],
            skip_backup=True,
        )
    assert _content_blob(sync_env["global_db"], CID_E) == before_blob
    assert _ws_registration_state(sync_env["ws_dir"]) == before_ws
    assert CID_E not in _typed_ids(sync_env["global_db"])


def test_import_file_foreign_race_leaves_state_unchanged(sync_env, monkeypatch):
    snap = _active(CID_E, "Incoming")
    path = _write_snapshot_file(sync_env["project_dir"], snap)
    _write_workspace(sync_env["ws_dir"], [])
    _workspace_json(sync_env["ws_dir"])
    gconn = _init_db(sync_env["global_db"])
    _create_typed_table(gconn)
    _insert_typed(gconn, CID_E, WS_HASH_B, _legacy_header(CID_E, "Incoming", WS_HASH_B))
    gconn.commit()
    gconn.close()
    before_ws = _ws_registration_state(sync_env["ws_dir"])
    with pytest.raises(typed_headers.MisregisteredHeaderError):
        importer.import_snapshot(
            path,
            PROJECT_PATH,
            target_workspace_dir=sync_env["ws_dir"],
            skip_backup=True,
        )
    assert _content_blob(sync_env["global_db"], CID_E)[0] is None
    assert _ws_registration_state(sync_env["ws_dir"]) == before_ws
    assert _typed_row(sync_env["global_db"], CID_E).workspace_id == WS_HASH_B


def test_sync_behind_typed_schema_error_leaves_payload_unchanged(
    sync_env, monkeypatch
):
    remote = _conversation(
        [_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="Behind"
    )
    local = _conversation([_msg(1, "A")], composer_id=CID_A, name="Behind")
    _commit_env(sync_env, [local], [remote], digest=False)
    _workspace_json(sync_env["ws_dir"])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    _make_incompatible_typed(sync_env["global_db"])
    before = _content_blob(sync_env["global_db"], CID_A)
    before_ws = _ws_registration_state(sync_env["ws_dir"])
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    _backend(monkeypatch)
    with pytest.raises(SystemExit):
        cli.cmd_sync(_args(workspace=WS_HASH[:8], force=True))
    assert _content_blob(sync_env["global_db"], CID_A) == before
    assert _ws_registration_state(sync_env["ws_dir"]) == before_ws


def test_sync_behind_foreign_race_leaves_payload_unchanged(sync_env, monkeypatch):
    remote = _conversation(
        [_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="Behind"
    )
    local = _conversation([_msg(1, "A")], composer_id=CID_A, name="Behind")
    _commit_env(sync_env, [local], [remote], digest=False)
    _workspace_json(sync_env["ws_dir"])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    before = _content_blob(sync_env["global_db"], CID_A)
    before_ws = _ws_registration_state(sync_env["ws_dir"])
    real_get = typed_headers.get_typed_row

    def live_foreign(conn, composer_id):
        if composer_id == CID_A:
            return typed_headers.TypedHeaderRow(
                composer_id=CID_A,
                workspace_id=WS_HASH_B,
                created_at=CREATED_AT,
                last_updated_at=LAST_UPDATED,
                is_archived=0,
                is_subagent=0,
                recency=LAST_UPDATED,
                checkpoint_at=None,
                value=json.dumps(_legacy_header(CID_A, "Behind", WS_HASH_B)),
            )
        return real_get(conn, composer_id)

    monkeypatch.setattr(typed_headers, "get_typed_row", live_foreign)
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    _backend(monkeypatch)
    with pytest.raises(SystemExit):
        cli.cmd_sync(_args(workspace=WS_HASH[:8], force=True))
    assert _content_blob(sync_env["global_db"], CID_A) == before
    assert _ws_registration_state(sync_env["ws_dir"]) == before_ws
    assert CID_A not in _typed_ids(sync_env["global_db"])


def test_copy_typed_schema_error_leaves_state_unchanged(sync_env):
    snap = _active(CID_A, "Copy")
    _commit_env(sync_env, [snap], [snap])
    target = sync_env["tmp"] / "cursor" / "workspaceStorage" / WS_HASH_B
    _write_workspace(target, [])
    _workspace_json(target, "/home/user/other")
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    _make_incompatible_typed(sync_env["global_db"])
    before_src = _content_blob(sync_env["global_db"], CID_A)
    before_src_ws = _ws_registration_state(sync_env["ws_dir"])
    before_tgt_ws = _ws_registration_state(target)
    before_ids = _typed_ids(sync_env["global_db"])
    ok, fail = importer.copy_between_workspaces(
        [CID_A],
        sync_env["ws_dir"],
        target,
        PROJECT_PATH,
        "/home/user/other",
        force=True,
    )
    assert ok == 0
    assert fail == 1
    assert _content_blob(sync_env["global_db"], CID_A) == before_src
    assert _ws_registration_state(sync_env["ws_dir"]) == before_src_ws
    assert _ws_registration_state(target) == before_tgt_ws
    assert _typed_ids(sync_env["global_db"]) == before_ids


def test_copy_foreign_race_leaves_state_unchanged(sync_env, monkeypatch):
    snap = _active(CID_A, "Copy")
    _commit_env(sync_env, [snap], [snap])
    target = sync_env["tmp"] / "cursor" / "workspaceStorage" / WS_HASH_B
    _write_workspace(target, [])
    _workspace_json(target, "/home/user/other")
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    before_src = _content_blob(sync_env["global_db"], CID_A)
    before_tgt_ws = _ws_registration_state(target)
    original = typed_headers._insert_synthetic

    def raced(conn, composer_id, workspace_id, composer_data, **kwargs):
        other = sqlite3.connect(str(sync_env["global_db"]))
        try:
            _insert_typed(
                other,
                composer_id,
                WS_HASH,
                _legacy_header(composer_id, "race", WS_HASH),
            )
            other.commit()
        finally:
            other.close()
        return original(conn, composer_id, workspace_id, composer_data, **kwargs)

    monkeypatch.setattr(typed_headers, "_insert_synthetic", raced)
    ok, fail = importer.copy_between_workspaces(
        [CID_A],
        sync_env["ws_dir"],
        target,
        PROJECT_PATH,
        "/home/user/other",
        force=True,
    )
    assert ok == 0
    assert fail == 1
    assert _content_blob(sync_env["global_db"], CID_A) == before_src
    assert _ws_registration_state(target) == before_tgt_ws
    copied = _typed_ids(sync_env["global_db"])
    assert CID_A not in copied


def test_repair_foreign_race_aborts_sync(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env)
    _inject_foreign_typed_row(sync_env, monkeypatch, CID_A)
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    _backend(monkeypatch)
    with pytest.raises(SystemExit):
        cli.cmd_sync(type("Args", (), {"force": False})())
    row = _typed_row(sync_env["global_db"], CID_A)
    assert row is None or row.workspace_id == WS_HASH_B


def test_repair_foreign_race_aborts_pull(sync_env, monkeypatch):
    _ippotrack_env(sync_env)
    _inject_foreign_typed_row(sync_env, monkeypatch, CID_A)
    with pytest.raises(typed_headers.MisregisteredHeaderError):
        pull.run_workspace_pull(
            PROJECT_PATH,
            target_workspace_dir=sync_env["ws_dir"],
            force=True,
        )
    row = _typed_row(sync_env["global_db"], CID_A)
    assert row is None or row.workspace_id == WS_HASH_B


def test_empty_snapshot_index_sync_workspace_repairs_legacy_only(
    sync_env, monkeypatch
):
    _restore_real_headers(monkeypatch)
    _ippotrack_local_only(sync_env)
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    _backend(monkeypatch)
    cli.cmd_sync(_args(workspace=WS_HASH[:8], force=True))
    assert _typed_ids(sync_env["global_db"]) == {CID_A, CID_B, CID_C, CID_D}


def test_empty_snapshot_index_global_sync_repairs_legacy_only(
    sync_env, monkeypatch
):
    _restore_real_headers(monkeypatch)
    _ippotrack_local_only(sync_env)
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    _backend(monkeypatch)
    cli.cmd_sync(type("Args", (), {"force": False})())
    assert _typed_ids(sync_env["global_db"]) == {CID_A, CID_B, CID_C, CID_D}


def test_empty_snapshot_index_registered_is_true_noop(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    _ippotrack_local_only(sync_env, typed_rows=True)
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    _backend(monkeypatch)
    syncstate.reset_op_counts()
    cli.cmd_sync(type("Args", (), {"force": False})())
    counts = syncstate.op_counts()
    assert counts.safety_global_backups == 0
    assert counts.cursor_write_connections == 0


def test_sync_skips_registration_write_when_cursor_running(
    sync_env, monkeypatch
):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env)
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    _backend(monkeypatch)
    monkeypatch.setattr(pull, "is_cursor_running", lambda: True)
    syncstate.reset_op_counts()
    cli.cmd_sync(type("Args", (), {"force": False})())
    assert _typed_ids(sync_env["global_db"]) == set()
    assert syncstate.op_counts().safety_global_backups == 0


def test_multi_target_pull_repair_uses_one_safety_backup(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env)
    extra = _active(CID_E, "Other workspace")
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _write_local(gconn, extra)
    gconn.commit()
    gconn.close()
    ws_b = sync_env["tmp"] / "cursor" / "workspaceStorage" / WS_HASH_B
    _write_workspace(ws_b, [extra])
    _workspace_json(ws_b, "/home/user/other")
    syncstate.reset_op_counts()
    result = pull.run_multi_target_pull(
        [
            (
                {
                    "path": PROJECT_PATH,
                    "workspace_dir": sync_env["ws_dir"],
                    "host": None,
                    "type": "local",
                },
                None,
            ),
            (
                {
                    "path": "/home/user/other",
                    "workspace_dir": ws_b,
                    "host": None,
                    "type": "local",
                },
                None,
            ),
        ],
        force=True,
    )
    assert result.repaired >= 4
    assert syncstate.op_counts().safety_global_backups == 1


def test_migrate_skips_task_like_cid_without_top_level_evidence(
    sync_env, monkeypatch
):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env)
    task_id = "task-call_aaaaaaaaaaaaaaaaaaaaaaaa"
    extra = _active(task_id, "Hidden task")
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _write_local(gconn, extra)
    headers = [_legacy_header(cid, name) for cid, name in IPPO_CHATS]
    headers.append(_legacy_header(task_id, "Hidden task"))
    _write_legacy_json(gconn, headers)
    gconn.commit()
    gconn.close()
    _write_workspace(
        sync_env["ws_dir"],
        [_active(cid, name) for cid, name in IPPO_CHATS] + [extra],
    )
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    migrated, _already = importer.migrate_to_global_headers(dry_run=True, force=True)
    assert migrated == 4
    assert task_id not in _typed_ids(sync_env["global_db"])


def test_task_like_selected_pane_only_is_not_typed(sync_env, monkeypatch):
    """task-* in selected/pane only, with no composerId/isSubagent in payload."""
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env)
    task_id = "task-call_bbbbbbbbbbbbbbbbbbbbbbbb"
    extra = _active(task_id, "Pane task")
    extra["composerData"].pop("composerId", None)
    extra["composerData"].pop("isSubagent", None)
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _write_local(gconn, extra)
    gconn.commit()
    gconn.close()
    _write_workspace(
        sync_env["ws_dir"],
        [_active(cid, name) for cid, name in IPPO_CHATS],
    )
    pane_key = "workbench.panel.composerChatViewPane.main"
    with db.CursorDB(sync_env["ws_dir"] / "state.vscdb") as cdb:
        data = cdb.get_json("composer.composerData", table="ItemTable") or {}
        data["selectedComposerIds"] = [task_id]
        data["lastFocusedComposerIds"] = [task_id]
        cdb.write_json("composer.composerData", data, table="ItemTable")
        cdb.write_json(pane_key, {f"agent.view.{task_id}": {}}, table="ItemTable")
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    migrated, _already = importer.migrate_to_global_headers(dry_run=True, force=True)
    assert migrated == 4
    _backend(monkeypatch)
    cli.cmd_sync(type("Args", (), {"force": False})())
    assert _typed_ids(sync_env["global_db"]) == {CID_A, CID_B, CID_C, CID_D}
    assert task_id not in _typed_ids(sync_env["global_db"])


def test_purge_list_hides_typed_subagents(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    _ippotrack_env(sync_env, typed_rows=True)
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _insert_typed(
        gconn,
        CID_E,
        WS_HASH,
        _legacy_header(CID_E, "Sub", isSubagent=True),
    )
    extra = _active(CID_E, "Sub")
    extra["composerData"]["isSubagent"] = True
    _write_local(gconn, extra)
    gconn.commit()
    gconn.close()
    chats = importer.list_all_chats_with_sizes()
    ids = {c["composerId"] for c in chats}
    assert CID_E not in ids
    assert {CID_A, CID_B, CID_C, CID_D} <= ids


def test_sync_up_to_date_preexisting_foreign_typed_aborts(sync_env, monkeypatch, capsys):
    _restore_real_headers(monkeypatch)
    snap = _active(CID_A, "Foreign uptodate")
    _commit_env(sync_env, [snap], [snap])
    _workspace_json(sync_env["ws_dir"])
    _plant_foreign_typed(sync_env, CID_A, "Foreign uptodate")
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    plan = _plan(sync_env, monkeypatch)
    assert plan.registration_conflicts
    assert plan.registration_conflicts[0].composer_id == CID_A
    assert plan.registration_conflicts[0].registration == (
        typed_headers.RegistrationHealth.MISREGISTERED
    )
    assert plan.registration_conflicts[0].relation == syncstate.SyncRelation.UP_TO_DATE
    before = _content_blob(sync_env["global_db"], CID_A)
    saved: list[str] = []
    monkeypatch.setattr(
        export, "save_snapshot", lambda *a, **k: saved.append("saved")
    )
    _backend(monkeypatch)
    with pytest.raises(SystemExit):
        cli.cmd_sync(_args(workspace=WS_HASH[:8], force=True))
    out = capsys.readouterr()
    assert "Already in sync" not in out.out
    assert "typed-registered in another workspace" in out.err
    assert _content_blob(sync_env["global_db"], CID_A) == before
    assert _typed_row(sync_env["global_db"], CID_A).workspace_id == WS_HASH_B
    assert saved == []


def test_sync_local_ahead_preexisting_foreign_typed_does_not_push(
    sync_env, monkeypatch
):
    _restore_real_headers(monkeypatch)
    remote = _active(CID_A, "Foreign ahead")
    local = _conversation(
        [_msg(1, f"m-{CID_A[:4]}"), _msg(2, "extra")],
        composer_id=CID_A,
        name="Foreign ahead",
    )
    local["composerData"]["createdAt"] = CREATED_AT
    local["composerData"]["lastUpdatedAt"] = LAST_UPDATED
    _commit_env(sync_env, [local], [remote], digest=False)
    _workspace_json(sync_env["ws_dir"])
    _plant_foreign_typed(sync_env, CID_A, "Foreign ahead")
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    plan = _plan(sync_env, monkeypatch)
    assert plan.registration_conflicts
    assert plan.registration_conflicts[0].relation == syncstate.SyncRelation.LOCAL_AHEAD
    before = _content_blob(sync_env["global_db"], CID_A)
    saved: list[str] = []
    monkeypatch.setattr(
        export, "save_snapshot", lambda *a, **k: saved.append("saved")
    )
    backend = _backend(monkeypatch)
    with pytest.raises(SystemExit):
        cli.cmd_sync(_args(workspace=WS_HASH[:8], force=True))
    assert saved == []
    assert backend.pushes == 0
    assert _content_blob(sync_env["global_db"], CID_A) == before
    assert _typed_row(sync_env["global_db"], CID_A).workspace_id == WS_HASH_B


def test_pull_up_to_date_preexisting_foreign_typed_aborts(
    sync_env, monkeypatch, capsys
):
    snap = _active(CID_A, "Foreign pull")
    _commit_env(sync_env, [snap], [snap])
    _workspace_json(sync_env["ws_dir"])
    _plant_foreign_typed(sync_env, CID_A, "Foreign pull")
    before = _content_blob(sync_env["global_db"], CID_A)
    with pytest.raises(typed_headers.RegistrationConflictError):
        pull.run_workspace_pull(
            PROJECT_PATH,
            target_workspace_dir=sync_env["ws_dir"],
            force=True,
        )
    out = capsys.readouterr()
    assert "Nothing to import." not in out.out
    assert "typed-registered in another workspace" in out.err
    assert _content_blob(sync_env["global_db"], CID_A) == before
    assert _typed_row(sync_env["global_db"], CID_A).workspace_id == WS_HASH_B


def test_sync_stale_workspace_id_same_uri_aborts(sync_env, monkeypatch, capsys):
    _restore_real_headers(monkeypatch)
    snap = _active(CID_A, "Stale")
    _commit_env(sync_env, [snap], [snap])
    _workspace_json(sync_env["ws_dir"])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    _insert_typed(gconn, CID_A, WS_HASH_B, _stale_header(CID_A, "Stale"))
    gconn.commit()
    gconn.close()
    paths.invalidate_headers_cache()
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    plan = _plan(sync_env, monkeypatch)
    assert plan.registration_conflicts
    assert plan.registration_conflicts[0].registration == (
        typed_headers.RegistrationHealth.STALE_WORKSPACE
    )
    before = _content_blob(sync_env["global_db"], CID_A)
    _backend(monkeypatch)
    with pytest.raises(SystemExit):
        cli.cmd_sync(_args(workspace=WS_HASH[:8], force=True))
    out = capsys.readouterr()
    assert "Already in sync" not in out.out
    assert "stale_workspace" in out.err
    assert _content_blob(sync_env["global_db"], CID_A) == before
    assert _typed_row(sync_env["global_db"], CID_A).workspace_id == WS_HASH_B


def test_import_file_identical_missing_typed_repairs_header_only(sync_env):
    snap = _active(CID_A, "Same")
    _commit_env(sync_env, [snap], [snap], digest=False)
    _workspace_json(sync_env["ws_dir"])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    path = sync_env["project_dir"] / f"{CID_A}.json.gz"
    before = _content_blob(sync_env["global_db"], CID_A)
    before_ws = _ws_registration_state(sync_env["ws_dir"])
    assert importer.import_snapshot(
        path,
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        skip_backup=True,
    )
    assert CID_A in _typed_ids(sync_env["global_db"])
    assert _typed_row(sync_env["global_db"], CID_A).workspace_id == WS_HASH
    assert _content_blob(sync_env["global_db"], CID_A) == before
    assert _ws_registration_state(sync_env["ws_dir"]) == before_ws


def test_import_file_local_ahead_missing_typed_repairs_header_only(sync_env):
    remote = _active(CID_A, "Ahead")
    local = _conversation(
        [_msg(1, f"m-{CID_A[:4]}"), _msg(2, "extra")],
        composer_id=CID_A,
        name="Ahead",
    )
    local["composerData"]["createdAt"] = CREATED_AT
    local["composerData"]["lastUpdatedAt"] = LAST_UPDATED
    _commit_env(sync_env, [local], [remote], digest=False)
    _workspace_json(sync_env["ws_dir"])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    path = sync_env["project_dir"] / f"{CID_A}.json.gz"
    before = _content_blob(sync_env["global_db"], CID_A)
    before_ws = _ws_registration_state(sync_env["ws_dir"])
    assert importer.import_snapshot(
        path,
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        skip_backup=True,
    )
    assert CID_A in _typed_ids(sync_env["global_db"])
    assert _content_blob(sync_env["global_db"], CID_A) == before
    assert _ws_registration_state(sync_env["ws_dir"]) == before_ws
    with db.CursorDB(sync_env["global_db"]) as cdb:
        cd = cdb.get_json(f"composerData:{CID_A}")
    assert len(cd["fullConversationHeadersOnly"]) == 2


def test_import_all_already_present_missing_typed_repairs(sync_env):
    snap = _active(CID_A, "All")
    _commit_env(sync_env, [snap], [snap], digest=False)
    _workspace_json(sync_env["ws_dir"])
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    gconn.commit()
    gconn.close()
    before = _content_blob(sync_env["global_db"], CID_A)
    before_ws = _ws_registration_state(sync_env["ws_dir"])
    success, failure = importer.import_all_snapshots(
        PROJECT_PATH,
        force=True,
        target_workspace_dir=sync_env["ws_dir"],
    )
    assert failure == 0
    assert success >= 1
    assert CID_A in _typed_ids(sync_env["global_db"])
    assert _content_blob(sync_env["global_db"], CID_A) == before
    assert _ws_registration_state(sync_env["ws_dir"]) == before_ws
