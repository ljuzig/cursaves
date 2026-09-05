"""Empty and dangling workspace CID registrations are not local chats (v0.9.10)."""

from __future__ import annotations

import hashlib
import json
import sqlite3

from cursor_saves import cli, export, paths, syncstate
from tests.test_syncstate import (
    CID_A,
    CID_B,
    CID_C,
    CID_D,
    CID_E,
    CID_F,
    HOST_A,
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
from tests.test_sync_workspace import _args, _spy_writes, _use_workspaces, _ws

pytest_plugins = ["tests.test_syncstate"]


def _empty(composer_id: str = CID_E, name=None) -> dict:
    return _conversation([], composer_id=composer_id, name=name)


def _active(composer_id: str, text: str = "hello") -> dict:
    return _conversation([_msg(1, text)], composer_id=composer_id, name=f"Chat-{composer_id[:4]}")


def _ippotrack(sync_env) -> list[dict]:
    actives = [_active(cid, f"m-{cid[:4]}") for cid in (CID_A, CID_B, CID_C, CID_D)]
    _commit_env(sync_env, actives + [_empty()], actives)
    return actives


def _status_out(monkeypatch, capsys, workspace: str = "1") -> str:
    monkeypatch.setattr(cli, "_ensure_synced", lambda: None)
    _backend(monkeypatch)
    cli.cmd_status(_args(workspace=workspace))
    return capsys.readouterr().out


def _target(sync_env) -> dict:
    return {
        "path": PROJECT_PATH,
        "workspace_dir": sync_env["ws_dir"],
        "host": None,
        "type": "local",
    }


def _plan(sync_env, target=None):
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        return syncstate.build_sync_plan(
            session, index, target_workspace=target or _target(sync_env)
        )


def test_classify_local_payload_matrix():
    assert syncstate.classify_local_payload(None) == syncstate.LocalPresence.INVALID
    assert (
        syncstate.classify_local_payload({"fullConversationHeadersOnly": []})
        == syncstate.LocalPresence.EMPTY
    )
    assert (
        syncstate.classify_local_payload(
            {"fullConversationHeadersOnly": [{"bubbleId": "b1"}]}
        )
        == syncstate.LocalPresence.ACTIVE
    )
    assert syncstate.classify_local_payload([1, 2]) == syncstate.LocalPresence.INVALID
    assert syncstate.classify_local_payload({"name": "x"}) == syncstate.LocalPresence.INVALID
    assert (
        syncstate.classify_local_payload({"fullConversationHeadersOnly": None})
        == syncstate.LocalPresence.INVALID
    )
    assert (
        syncstate.classify_local_payload({"fullConversationHeadersOnly": {}})
        == syncstate.LocalPresence.INVALID
    )


def test_ippotrack_status_counts_active_only(sync_env, monkeypatch, capsys):
    _ippotrack(sync_env)
    out = _status_out(monkeypatch, capsys)
    assert "Local conversations:     4" in out
    assert "Snapshot files:          4" in out
    assert "In both:                 4" in out
    assert "Local only (unexported): 0" in out
    assert "Snapshot only (not imported): 0" in out
    assert CID_E[:12] not in out


def test_empty_does_not_appear_in_list(sync_env):
    _ippotrack(sync_env)
    listed = export.list_conversations(PROJECT_PATH, workspace_dir=sync_env["ws_dir"])
    ids = {c["id"] for c in listed}
    assert ids == {CID_A, CID_B, CID_C, CID_D}
    assert CID_E not in ids


def test_list_does_not_scan_global_bubble_inventory(sync_env):
    _ippotrack(sync_env)
    with syncstate.SyncReadSession() as session:
        export.list_conversations(
            PROJECT_PATH, workspace_dir=sync_env["ws_dir"], session=session
        )
        assert session._inventory_attempted is False
        assert session._inventory_complete is False


def test_dangling_without_snapshot_is_not_local_or_unexported(
    sync_env, monkeypatch, capsys
):
    active = _active(CID_A)
    dangling = {"composerId": CID_F, "composerData": {"name": None}}
    gconn = _init_db(sync_env["global_db"])
    _write_local(gconn, active)
    gconn.commit()
    gconn.close()
    _write_workspace(sync_env["ws_dir"], [active, dangling])
    _write_snapshot_file(sync_env["project_dir"], active)

    with syncstate.SyncReadSession() as session:
        assert (
            syncstate.classify_local_conversation(session, CID_F)
            == syncstate.LocalPresence.DANGLING
        )
    listed = export.list_conversations(PROJECT_PATH, workspace_dir=sync_env["ws_dir"])
    assert {c["id"] for c in listed} == {CID_A}

    out = _status_out(monkeypatch, capsys)
    assert "Local conversations:     1" in out
    assert "Local only (unexported): 0" in out
    assert CID_F[:12] not in out

    plan = _plan(sync_env)
    assert CID_F not in {i.composer_id for i in plan.items}


def test_push_and_checkpoint_do_not_export_empty_or_dangling(sync_env):
    active = _active(CID_A)
    empty = _empty()
    dangling = {"composerId": CID_F, "composerData": {"name": None}}
    gconn = _init_db(sync_env["global_db"])
    _write_local(gconn, active)
    _write_local(gconn, empty)
    gconn.commit()
    gconn.close()
    _write_workspace(sync_env["ws_dir"], [active, empty, dangling])

    saved = export.checkpoint_project(
        PROJECT_PATH, workspace_dir=sync_env["ws_dir"]
    )
    written = {p.name.split(".")[0] for p in saved}
    assert written == {CID_A}
    assert not (sync_env["project_dir"] / f"{CID_E}.json.gz").exists()
    assert not (sync_env["project_dir"] / f"{CID_F}.json.gz").exists()

    assert export.export_conversation(PROJECT_PATH, CID_E) is None
    assert export.export_conversation(PROJECT_PATH, CID_F) is None
    assert export.export_conversation(PROJECT_PATH, CID_A) is not None


def test_targeted_sync_all_synced_plus_empty_is_not_local_ahead(
    sync_env, monkeypatch
):
    actives = _ippotrack(sync_env)
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    imported, saved = _spy_writes(monkeypatch)
    _backend(monkeypatch)

    plan = _plan(sync_env)
    assert {i.composer_id for i in plan.items} == {s["composerId"] for s in actives}
    assert plan.ahead == []
    assert plan.by_relation(syncstate.SyncRelation.NEVER_PUSHED) == []
    assert all(i.relation == syncstate.SyncRelation.UP_TO_DATE for i in plan.items)

    cli.cmd_sync(_args(workspace=WS_HASH[:8]))
    assert imported == []
    assert saved == []


def test_empty_becomes_active_after_first_header(sync_env):
    empty = _empty()
    _commit_env(sync_env, [empty], [])
    assert export.checkpoint_project(
        PROJECT_PATH, workspace_dir=sync_env["ws_dir"]
    ) == []

    grown = _conversation([_msg(1, "first")], composer_id=CID_E, name=None)
    gconn = sqlite3.connect(str(sync_env["global_db"]))
    _write_local(gconn, grown)
    gconn.commit()
    gconn.close()

    with syncstate.SyncReadSession() as session:
        assert (
            syncstate.classify_local_conversation(session, CID_E)
            == syncstate.LocalPresence.ACTIVE
        )
    saved = export.checkpoint_project(
        PROJECT_PATH, workspace_dir=sync_env["ws_dir"]
    )
    assert {p.name.split(".")[0] for p in saved} == {CID_E}


def test_headers_with_tombstoned_bubbles_remain_active(sync_env):
    snap = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A)
    _commit_env(sync_env, [snap], [])
    conn = sqlite3.connect(str(sync_env["global_db"]))
    conn.execute("DELETE FROM cursorDiskKV WHERE key = ?", (f"bubbleId:{CID_A}:bubble-2",))
    conn.commit()
    conn.close()

    with syncstate.SyncReadSession() as session:
        assert (
            syncstate.classify_local_conversation(session, CID_A)
            == syncstate.LocalPresence.ACTIVE
        )
    listed = export.list_conversations(PROJECT_PATH, workspace_dir=sync_env["ws_dir"])
    assert {c["id"] for c in listed} == {CID_A}


def test_malformed_composer_data_is_invalid_not_ignored(
    sync_env, monkeypatch, capsys
):
    active = _active(CID_A)
    _commit_env(sync_env, [active], [active])
    conn = sqlite3.connect(str(sync_env["global_db"]))
    conn.execute(
        "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
        (f"composerData:{CID_B}", json.dumps(["not", "a", "dict"])),
    )
    conn.execute(
        "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
        (
            f"composerData:{CID_C}",
            json.dumps({"fullConversationHeadersOnly": "nope"}),
        ),
    )
    conn.execute(
        "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
        (f"composerData:{CID_D}", "{not-json"),
    )
    conn.commit()
    conn.close()
    _write_workspace(
        sync_env["ws_dir"],
        [
            active,
            {"composerId": CID_B, "composerData": {"name": "bad-list"}},
            {"composerId": CID_C, "composerData": {"name": "bad-headers"}},
            {"composerId": CID_D, "composerData": {"name": "bad-json"}},
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
        assert (
            syncstate.classify_local_conversation(session, CID_D)
            == syncstate.LocalPresence.INVALID
        )

    plan = _plan(sync_env)
    unknown_ids = {i.composer_id for i in plan.unknown}
    assert {CID_B, CID_C, CID_D} <= unknown_ids
    assert plan.unsafe

    out = _status_out(monkeypatch, capsys)
    assert "Unknown:" in out
    assert "Local only (unexported): 0" in out


def test_empty_plus_snapshot_keeps_restore_classification(sync_env):
    empty = _empty()
    remote = _conversation([_msg(1, "backed-up")], composer_id=CID_E, name="Remote")
    _commit_env(sync_env, [empty], [remote], digest=False)
    plan = _plan(sync_env)
    by_id = {i.composer_id: i.relation for i in plan.items}
    assert by_id[CID_E] == syncstate.SyncRelation.BEHIND
    assert CID_E not in {
        i.composer_id
        for i in plan.by_relation(syncstate.SyncRelation.NEVER_PUSHED)
    }
    assert CID_E not in {i.composer_id for i in plan.ahead}


def test_pull_never_pushed_ignores_empty_shell(sync_env):
    _ippotrack(sync_env)
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build_for_project("project")
        plan = syncstate.build_pull_plan(session, index, _target(sync_env))
    assert plan.never_pushed == 0


def test_empty_registration_does_not_bypass_global_collision(sync_env):
    snap_a = _conversation([_msg(1, "from-snap")], composer_id=CID_A, name="X")
    other = _conversation([_msg(1, "from-b")], composer_id=CID_A, name="B-local")
    empty = _empty()
    other_ws = sync_env["tmp"] / "cursor" / "workspaceStorage" / ("b" * 32)
    gconn = _init_db(sync_env["global_db"])
    _write_local(gconn, other)
    _write_local(gconn, empty)
    gconn.commit()
    gconn.close()
    _write_workspace(sync_env["ws_dir"], [empty])
    _write_workspace(other_ws, [other])
    _write_snapshot_file(sync_env["project_dir"], snap_a, gzip_body=True)

    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build_for_project("project")
        plan = syncstate.build_pull_plan(session, index, _target(sync_env))
    assert len(plan.collisions) == 1
    assert plan.collisions[0].composer_id == CID_A
    assert plan.collisions[0].action == syncstate.PullAction.SKIP
    assert plan.import_candidates == []


def test_ssh_empty_shell_is_not_local_ahead(sync_env):
    pid_a = paths.get_project_identifier(PROJECT_PATH, source_host=HOST_A)
    local = _conversation(
        [_msg(1, "A")],
        composer_id=CID_A,
        sourceHost=HOST_A,
        sourceProjectPath=PROJECT_PATH,
        projectIdentifier=pid_a,
    )
    empty = _conversation(
        [],
        composer_id=CID_E,
        sourceHost=HOST_A,
        sourceProjectPath=PROJECT_PATH,
        projectIdentifier=pid_a,
        name=None,
    )
    gconn = _init_db(sync_env["global_db"])
    _write_local(gconn, local)
    _write_local(gconn, empty)
    gconn.commit()
    gconn.close()
    _write_workspace(sync_env["ws_dir"], [local, empty])
    _write_snapshot_file(sync_env["snaps"] / pid_a, local, with_digest=True, gzip_body=True)

    ws = {
        "type": "ssh",
        "host": HOST_A,
        "path": PROJECT_PATH,
        "workspace_dir": sync_env["ws_dir"],
    }
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index, target_workspace=ws)
        assert (
            syncstate.classify_local_conversation(session, CID_E)
            == syncstate.LocalPresence.EMPTY
        )
    assert {i.composer_id for i in plan.items} == {CID_A}
    assert plan.ahead == []
    assert plan.by_relation(syncstate.SyncRelation.NEVER_PUSHED) == []


def test_workspaces_summary_omits_empty_not_pushed(sync_env, capsys):
    _ippotrack(sync_env)
    cli.cmd_workspaces(type("Args", (), {})())
    out = capsys.readouterr().out
    assert "Refs" in out
    assert "4 synced" in out
    assert "not pushed" not in out
    assert "ahead" not in out


def test_json_null_composer_data_is_invalid(sync_env):
    active = _active(CID_A)
    _commit_env(sync_env, [active], [active])
    conn = sqlite3.connect(str(sync_env["global_db"]))
    conn.execute(
        "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
        (f"composerData:{CID_B}", "null"),
    )
    conn.commit()
    conn.close()
    _write_workspace(
        sync_env["ws_dir"],
        [active, {"composerId": CID_B, "composerData": {"name": "null-row"}}],
    )

    with syncstate.SyncReadSession() as session:
        assert (
            syncstate.classify_local_conversation(session, CID_B)
            == syncstate.LocalPresence.INVALID
        )
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index, target_workspace=_target(sync_env))
    by_id = {i.composer_id: i.relation for i in plan.items}
    assert by_id[CID_B] == syncstate.SyncRelation.UNKNOWN
    assert plan.unsafe


def test_invalid_cid_in_two_workspaces_counted_once(sync_env, monkeypatch, capsys):
    other_ws = sync_env["tmp"] / "cursor" / "workspaceStorage" / ("c" * 32)
    gconn = _init_db(sync_env["global_db"])
    gconn.execute(
        "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
        (f"composerData:{CID_B}", json.dumps(["not", "a", "dict"])),
    )
    gconn.commit()
    gconn.close()
    stub = {"composerId": CID_B, "composerData": {"name": "bad"}}
    _write_workspace(sync_env["ws_dir"], [stub])
    _write_workspace(other_ws, [stub])
    for ws_dir in (sync_env["ws_dir"], other_ws):
        ws_dir.mkdir(parents=True, exist_ok=True)
        (ws_dir / "workspace.json").write_text(
            json.dumps({"folder": f"file://{PROJECT_PATH}"})
        )

    monkeypatch.setattr(cli, "_ensure_synced", lambda: None)
    _backend(monkeypatch)
    cli.cmd_status(type("Args", (), {"workspace": None, "project": PROJECT_PATH})())
    out = capsys.readouterr().out
    assert "Unknown:                  1" in out


def test_status_uses_one_global_read_session(sync_env, monkeypatch):
    _ippotrack(sync_env)
    monkeypatch.setattr(cli, "_ensure_synced", lambda: None)
    _backend(monkeypatch)
    syncstate.reset_op_counts()
    cli.cmd_status(_args(workspace="1"))
    assert syncstate.op_counts().sqlite_backups == 1


def test_checkpoint_uses_one_global_read_session(sync_env):
    _ippotrack(sync_env)
    syncstate.reset_op_counts()
    export.checkpoint_project(PROJECT_PATH, workspace_dir=sync_env["ws_dir"])
    assert syncstate.op_counts().sqlite_backups == 1


def test_warm_synced_plan_does_not_reparse_composer_json(sync_env):
    snaps = [_active(cid) for cid in (CID_A, CID_B, CID_C, CID_D)]
    _commit_env(sync_env, snaps, snaps, digest=False)
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        syncstate.build_sync_plan(session, index, target_workspace=_target(sync_env))

    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index, target_workspace=_target(sync_env))
    assert len(plan.items) == 4
    assert all(i.relation == syncstate.SyncRelation.UP_TO_DATE for i in plan.items)
    assert syncstate.op_counts().local_composer_json_parses == 0
    assert syncstate.op_counts().local_semantic_rehashes == 0


def test_dangling_plus_snapshot_is_behind_in_sync_not_restored_by_pull(sync_env):
    remote = _conversation([_msg(1, "backed-up")], composer_id=CID_F, name="Remote")
    dangling = {"composerId": CID_F, "composerData": {"name": None}}
    _write_workspace(sync_env["ws_dir"], [dangling])
    gconn = _init_db(sync_env["global_db"])
    gconn.commit()
    gconn.close()
    _write_snapshot_file(sync_env["project_dir"], remote, gzip_body=True)

    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        sync_plan = syncstate.build_sync_plan(
            session, index, target_workspace=_target(sync_env)
        )
        pull_index = syncstate.SnapshotIndex.build_for_project("project")
        pull_plan = syncstate.build_pull_plan(session, pull_index, _target(sync_env))
    by_id = {i.composer_id: i.relation for i in sync_plan.items}
    assert by_id[CID_F] == syncstate.SyncRelation.BEHIND
    assert len(pull_plan.import_candidates) == 0
    assert pull_plan.items[0].relation == syncstate.PullRelation.UNKNOWN
    assert pull_plan.items[0].action == syncstate.PullAction.SKIP


def test_legacy_v5_local_cache_does_not_hide_invalid(sync_env):
    payload = {"name": "broken"}
    empty = _conversation([], composer_id=CID_B)
    gconn = _init_db(sync_env["global_db"])
    _put_json(gconn, f"composerData:{CID_B}", payload)
    gconn.commit()
    gconn.close()
    _write_workspace(
        sync_env["ws_dir"],
        [{"composerId": CID_B, "composerData": payload}],
    )
    _write_snapshot_file(sync_env["project_dir"], empty, with_digest=True, gzip_body=True)

    raw = json.dumps(payload).encode("utf-8")
    hasher = hashlib.sha256()
    syncstate._hash_field(hasher, f"composerData:{CID_B}".encode("utf-8"), raw)
    row_fp = "sha256:" + hasher.hexdigest()
    cache_path = sync_env["tmp"] / "cache" / "sync-semantics.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "version": syncstate._CACHE_VERSION,
                "snapshots": {},
                "local": {
                    CID_B: {
                        "rowFingerprint": row_fp,
                        "blobRefs": [],
                        "blobFingerprint": "sha256:" + hashlib.sha256().hexdigest(),
                        "semanticDigest": syncstate.conversation_digest([]),
                        "semanticDigestVersion": syncstate.SEMANTIC_DIGEST_VERSION,
                    }
                },
            }
        )
    )

    with syncstate.SyncReadSession() as session:
        assert (
            syncstate.classify_local_conversation(session, CID_B)
            == syncstate.LocalPresence.INVALID
        )
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index, target_workspace=_target(sync_env))
    assert [(i.composer_id, i.relation.value) for i in plan.items] == [
        (CID_B, "unknown")
    ]
    assert plan.unsafe


def test_classify_conversation_invalid_without_snapshot_is_unknown(sync_env):
    gconn = _init_db(sync_env["global_db"])
    _put_json(gconn, f"composerData:{CID_B}", {"name": "broken"})
    gconn.commit()
    gconn.close()
    _write_workspace(
        sync_env["ws_dir"],
        [{"composerId": CID_B, "composerData": {"name": "broken"}}],
    )
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        rel = syncstate.classify_conversation(
            session, index, CID_B, project_identifier="project"
        )
    assert rel == syncstate.SyncRelation.UNKNOWN


def test_corrupt_global_row_outside_target_is_collision(sync_env):
    snap_a = _conversation([_msg(1, "from-snap")], composer_id=CID_A, name="X")
    other_ws = sync_env["tmp"] / "cursor" / "workspaceStorage" / ("b" * 32)
    gconn = _init_db(sync_env["global_db"])
    _put_json(gconn, f"composerData:{CID_A}", ["not", "a", "dict"])
    gconn.commit()
    gconn.close()
    _write_workspace(sync_env["ws_dir"], [])
    _write_workspace(
        other_ws, [{"composerId": CID_A, "composerData": {"name": "B-local"}}]
    )
    _write_snapshot_file(sync_env["project_dir"], snap_a, gzip_body=True)

    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build_for_project("project")
        plan = syncstate.build_pull_plan(session, index, _target(sync_env))
    assert len(plan.collisions) == 1
    assert plan.collisions[0].composer_id == CID_A
    assert plan.collisions[0].relation == syncstate.PullRelation.GLOBAL_COLLISION
    assert plan.collisions[0].action == syncstate.PullAction.SKIP
    assert plan.import_candidates == []


def test_sync_plan_releases_composer_cells(sync_env):
    _ippotrack(sync_env)
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        syncstate.build_sync_plan(session, index, target_workspace=_target(sync_env))
        assert session._composer_cells == {}
        assert session._presence


def test_pull_plan_releases_composer_cells(sync_env):
    _ippotrack(sync_env)
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build_for_project("project")
        syncstate.build_pull_plan(session, index, _target(sync_env))
        assert session._composer_cells == {}


def test_checkpoint_does_not_preload_all_composer_cells(sync_env, monkeypatch):
    _ippotrack(sync_env)
    seen: list[int] = []
    orig = syncstate.SyncReadSession.export_conversation

    def wrapped(self, *args, **kwargs):
        seen.append(len(self._composer_cells))
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(syncstate.SyncReadSession, "export_conversation", wrapped)
    saved = export.checkpoint_project(
        PROJECT_PATH, workspace_dir=sync_env["ws_dir"]
    )
    assert {p.name.split(".")[0] for p in saved} == {CID_A, CID_B, CID_C, CID_D}
    assert seen
    assert seen[0] <= 1
    assert max(seen) <= 1
