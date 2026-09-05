"""Ignore stale JSON-null Composer pane tombstones (v0.9.15)."""

from __future__ import annotations

import json
import sqlite3

from cursor_saves import cli, export, importer, paths, syncstate
from tests.test_empty_registrations import (
    _active,
    _empty,
    _plan,
    _status_out,
    _target,
)
from tests.test_syncstate import (
    CID_A,
    CID_B,
    CID_C,
    CID_D,
    CID_E,
    CID_F,
    PROJECT_PATH,
    WS_HASH,
    _commit_env,
    _conversation,
    _init_db,
    _msg,
    _put_json,
    _write_local,
    _write_snapshot_file,
    _write_workspace,
)
from tests.test_sync_workspace import _backend
from tests.test_typed_composer_headers import (
    _create_typed_table,
    _insert_typed,
    _legacy_header,
    _restore_real_headers,
    _typed_ids,
)

pytest_plugins = ["tests.test_syncstate"]

TOMBSTONES = [
    f"{i:08x}-0000-4000-8000-000000000000" for i in range(8)
]


def _put_null_composer(conn: sqlite3.Connection, cid: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
        (f"composerData:{cid}", "null"),
    )


def _write_pane_refs(
    conn: sqlite3.Connection,
    cids: list[str],
    key: str = "workbench.panel.composerChatViewPane.main",
) -> None:
    row = conn.execute("SELECT value FROM ItemTable WHERE key = ?", (key,)).fetchone()
    pane = json.loads(row[0]) if row else {}
    for cid in cids:
        pane[f"agent.view.{cid}"] = {}
    _put_json(conn, key, pane, table="ItemTable")


def _install_pane_tombstones(
    sync_env,
    cids: list[str],
    *,
    also_global_hidden: bool = False,
) -> None:
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    for cid in cids:
        _put_null_composer(gconn, cid)
    if also_global_hidden:
        _write_pane_refs(
            gconn,
            cids,
            key="workbench.panel.composerChatViewPane.main.hidden",
        )
    gconn.commit()
    gconn.close()
    wconn = sqlite3.connect(str(sync_env["ws_dir"] / "state.vscdb"))
    _write_pane_refs(wconn, cids)
    wconn.commit()
    wconn.close()


def test_pane_null_tombstone_absent_from_plan_and_not_unknown(sync_env):
    active = _active(CID_A)
    _commit_env(sync_env, [active], [active])
    _install_pane_tombstones(sync_env, [CID_B])

    with syncstate.SyncReadSession() as session:
        assert (
            syncstate.classify_local_conversation(session, CID_B)
            == syncstate.LocalPresence.NULL_TOMBSTONE
        )
        ids = paths.get_workspace_composer_ids(
            sync_env["ws_dir"] / "state.vscdb", session=session
        )
        assert CID_B not in ids
        assert CID_A in ids
        plan = _plan(sync_env)
    assert CID_B not in {i.composer_id for i in plan.items}
    assert plan.unknown == []
    assert not plan.unsafe


def test_global_hidden_and_workspace_pane_ignored_once(sync_env):
    active = _active(CID_A)
    _commit_env(sync_env, [active], [active])
    _install_pane_tombstones(sync_env, [CID_B], also_global_hidden=True)

    plan = _plan(sync_env)
    ids = [i.composer_id for i in plan.items]
    assert ids.count(CID_B) == 0
    assert CID_A in ids


def test_null_tombstone_with_snapshot_is_importable(sync_env):
    remote = _conversation([_msg(1, "backed-up")], composer_id=CID_B, name="Remote")
    _write_workspace(sync_env["ws_dir"], [])
    gconn = _init_db(sync_env["global_db"])
    _put_null_composer(gconn, CID_B)
    gconn.commit()
    gconn.close()
    _install_pane_tombstones(sync_env, [CID_B])
    _write_snapshot_file(sync_env["project_dir"], remote, gzip_body=True)

    with syncstate.SyncReadSession() as session:
        assert (
            syncstate.classify_local_conversation(session, CID_B)
            == syncstate.LocalPresence.NULL_TOMBSTONE
        )
        index = syncstate.SnapshotIndex.build()
        sync_plan = syncstate.build_sync_plan(
            session, index, target_workspace=_target(sync_env)
        )
        pull_index = syncstate.SnapshotIndex.build_for_project("project")
        pull_plan = syncstate.build_pull_plan(
            session, pull_index, _target(sync_env)
        )
    by_id = {i.composer_id: i.relation for i in sync_plan.items}
    assert by_id[CID_B] == syncstate.SyncRelation.BEHIND
    assert CID_B not in {i.composer_id for i in sync_plan.unknown}
    assert len(pull_plan.import_candidates) == 1
    assert pull_plan.import_candidates[0].composer_id == CID_B
    assert pull_plan.import_candidates[0].relation == (
        syncstate.PullRelation.MISSING_LOCAL
    )
    assert pull_plan.unknown == []


def test_null_tombstone_with_typed_row_is_diagnosed(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    _write_workspace(sync_env["ws_dir"], [])
    gconn = _init_db(sync_env["global_db"])
    _create_typed_table(gconn)
    _put_null_composer(gconn, CID_B)
    _insert_typed(gconn, CID_B, WS_HASH, _legacy_header(CID_B, "Broken"))
    gconn.commit()
    gconn.close()
    _install_pane_tombstones(sync_env, [CID_B])

    with syncstate.SyncReadSession() as session:
        assert (
            syncstate.classify_local_conversation(session, CID_B)
            == syncstate.LocalPresence.NULL_TOMBSTONE
        )
        assert session.typed_row(CID_B) is not None
        ids = paths.get_workspace_composer_ids(
            sync_env["ws_dir"] / "state.vscdb", session=session
        )
        assert CID_B in ids
        plan = syncstate.build_sync_plan(
            session,
            syncstate.SnapshotIndex.build(),
            target_workspace=_target(sync_env),
        )
    assert {i.composer_id for i in plan.unknown} == {CID_B}
    assert plan.unsafe


def test_malformed_payload_still_invalid_unknown(sync_env):
    active = _active(CID_A)
    _commit_env(sync_env, [active], [active])
    conn = sqlite3.connect(str(sync_env["global_db"]))
    conn.execute(
        "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
        (f"composerData:{CID_B}", "{not-json"),
    )
    conn.execute(
        "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
        (f"composerData:{CID_C}", json.dumps(["not", "a", "dict"])),
    )
    conn.commit()
    conn.close()
    _write_workspace(
        sync_env["ws_dir"],
        [
            active,
            {"composerId": CID_B, "composerData": {"name": "bad-json"}},
            {"composerId": CID_C, "composerData": {"name": "bad-list"}},
        ],
    )

    with syncstate.SyncReadSession() as session:
        assert (
            syncstate.classify_local_conversation(session, CID_B)
            == syncstate.LocalPresence.INVALID
        )
        assert (
            syncstate.classify_local_conversation(session, CID_C)
            == syncstate.LocalPresence.INVALID
        )
        assert syncstate.classify_local_payload(None) == syncstate.LocalPresence.INVALID
    plan = _plan(sync_env)
    assert {CID_B, CID_C} <= {i.composer_id for i in plan.unknown}
    assert plan.unsafe


def test_modern_empty_headers_policy_unchanged(sync_env, monkeypatch, capsys):
    empty = _empty()
    active = _active(CID_A)
    _commit_env(sync_env, [active, empty], [active])
    with syncstate.SyncReadSession() as session:
        assert (
            syncstate.classify_local_conversation(session, CID_E)
            == syncstate.LocalPresence.EMPTY
        )
    listed = export.list_conversations(PROJECT_PATH, workspace_dir=sync_env["ws_dir"])
    assert {c["id"] for c in listed} == {CID_A}
    plan = _plan(sync_env)
    assert CID_E not in {i.composer_id for i in plan.items}
    out = _status_out(monkeypatch, capsys)
    assert "Local conversations:     1" in out
    assert "Unknown:" not in out


def test_task_pane_null_tombstone_never_typed(sync_env, monkeypatch):
    _restore_real_headers(monkeypatch)
    active = _active(CID_A)
    _commit_env(sync_env, [active], [active])
    task_id = "task-call_cccccccccccccccccccccccc"
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _create_typed_table(gconn)
    _put_null_composer(gconn, task_id)
    gconn.commit()
    gconn.close()
    _install_pane_tombstones(sync_env, [task_id])

    importer.migrate_to_global_headers(dry_run=False, force=True)
    assert task_id not in _typed_ids(sync_env["global_db"])
    listed = export.list_conversations(PROJECT_PATH, workspace_dir=sync_env["ws_dir"])
    assert task_id not in {c["id"] for c in listed}
    plan = _plan(sync_env)
    assert task_id not in {i.composer_id for i in plan.items}


def test_active_plus_pane_tombstones_status_and_sync(
    sync_env, monkeypatch, capsys
):
    actives = [_active(cid, f"m-{cid[:4]}") for cid in (CID_A, CID_B, CID_C, CID_D, CID_E, CID_F)]
    _commit_env(sync_env, actives, actives)
    _install_pane_tombstones(sync_env, TOMBSTONES, also_global_hidden=True)

    listed = export.list_conversations(PROJECT_PATH, workspace_dir=sync_env["ws_dir"])
    assert {c["id"] for c in listed} == {s["composerId"] for s in actives}
    assert not ({c["id"] for c in listed} & set(TOMBSTONES))

    plan = _plan(sync_env)
    assert {i.composer_id for i in plan.items} == {s["composerId"] for s in actives}
    assert plan.unknown == []
    assert not plan.unsafe
    assert all(i.relation == syncstate.SyncRelation.UP_TO_DATE for i in plan.items)

    out = _status_out(monkeypatch, capsys)
    assert "Local conversations:     6" in out
    assert "Unknown:" not in out

    monkeypatch.setattr(cli, "_ensure_synced", lambda: None)
    _backend(monkeypatch)
    cli.cmd_sync(type("Args", (), {"workspace": None, "force": False})())
    sync_out = capsys.readouterr().out
    assert "unknown" not in sync_out.lower()
