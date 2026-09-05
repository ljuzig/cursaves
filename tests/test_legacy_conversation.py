"""Legacy monolithic composerData.conversation semantics (v0.9.12)."""

from __future__ import annotations

import json

import pytest

from cursor_saves import cli, db, export, importer, syncstate
from tests.test_syncstate import (
    CID_A,
    PROJECT_PATH,
    _backend,
    _commit_env,
    _conversation,
    _init_db,
    _msg,
    _write_local,
    _write_snapshot_file,
    _write_workspace,
)

pytest_plugins = ["tests.test_syncstate"]


def _legacy_msg(i: int, text: str, **extra) -> dict:
    msg = {
        "bubbleId": extra.pop("bubbleId", f"bubble-{i}"),
        "type": extra.pop("type", 1),
        "text": text,
    }
    msg.update(extra)
    return msg


def _legacy_conversation(messages, *, composer_id: str, name: str = "Legacy", **top) -> dict:
    blobs = {}
    items = []
    for i, msg in enumerate(messages):
        item = dict(msg)
        if "blob_id" in item:
            hid = item.pop("blob_id")
            blobs[hid] = item.pop("blob_data", "blob")
            item["contentHash"] = hid
        items.append(item)
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
            "conversation": items,
        },
        "bubbleEntries": {},
        "contentBlobs": blobs,
    }


def _classify(sync_env, local, remote):
    _commit_env(sync_env, [local], [remote], digest=False, gzip_cids={local["composerId"]})
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        return syncstate.classify_conversation(
            session, index, local["composerId"], project_identifier="project"
        )


def test_legacy_nonempty_is_active():
    data = {"conversation": [_legacy_msg(1, "hello")]}
    assert syncstate.classify_local_payload(data) == syncstate.LocalPresence.ACTIVE
    assert syncstate.semantic_message_count(data) == 1


def test_legacy_empty_list_is_empty():
    data = {"conversation": []}
    assert syncstate.classify_local_payload(data) == syncstate.LocalPresence.EMPTY
    assert syncstate.semantic_message_count(data) == 0


def test_missing_headers_malformed_conversation_is_invalid():
    assert syncstate.classify_local_payload({"conversation": None}) == (
        syncstate.LocalPresence.INVALID
    )
    assert syncstate.classify_local_payload({"conversation": {}}) == (
        syncstate.LocalPresence.INVALID
    )
    assert syncstate.classify_local_payload({"conversation": "nope"}) == (
        syncstate.LocalPresence.INVALID
    )


def test_name_only_stays_invalid():
    assert syncstate.classify_local_payload({"name": "x"}) == syncstate.LocalPresence.INVALID


def test_headers_key_wins_over_legacy_conversation():
    data = {
        "fullConversationHeadersOnly": [{"bubbleId": "b1"}],
        "conversation": [_legacy_msg(1, "ignored")],
    }
    assert syncstate.classify_local_payload(data) == syncstate.LocalPresence.ACTIVE
    assert syncstate.semantic_message_count(data) == 1
    malformed = {
        "fullConversationHeadersOnly": None,
        "conversation": [_legacy_msg(1, "still-invalid")],
    }
    assert syncstate.classify_local_payload(malformed) == syncstate.LocalPresence.INVALID
    assert syncstate.semantic_message_count(malformed) == 0


def test_legacy_unit_does_not_collide_with_modern_unit():
    modern = _conversation([_msg(1, "same")], composer_id=CID_A)
    legacy = _legacy_conversation([_legacy_msg(1, "same")], composer_id=CID_A)
    modern_hashes = syncstate.snapshot_unit_hashes(modern)
    legacy_hashes = syncstate.snapshot_unit_hashes(legacy)
    assert modern_hashes
    assert legacy_hashes
    assert modern_hashes != legacy_hashes


def test_legacy_local_equals_snapshot_is_up_to_date(sync_env):
    snap = _legacy_conversation(
        [_legacy_msg(1, "A"), _legacy_msg(2, "B")], composer_id=CID_A
    )
    assert _classify(sync_env, snap, snap) == syncstate.SyncRelation.UP_TO_DATE


def test_legacy_local_prefix_longer_is_ahead(sync_env):
    remote = _legacy_conversation([_legacy_msg(1, "A")], composer_id=CID_A)
    local = _legacy_conversation(
        [_legacy_msg(1, "A"), _legacy_msg(2, "B")], composer_id=CID_A
    )
    assert _classify(sync_env, local, remote) == syncstate.SyncRelation.LOCAL_AHEAD


def test_legacy_snapshot_prefix_longer_is_behind(sync_env):
    local = _legacy_conversation([_legacy_msg(1, "A")], composer_id=CID_A)
    remote = _legacy_conversation(
        [_legacy_msg(1, "A"), _legacy_msg(2, "B")], composer_id=CID_A
    )
    assert _classify(sync_env, local, remote) == syncstate.SyncRelation.BEHIND


def test_legacy_changed_message_is_diverged(sync_env):
    remote = _legacy_conversation(
        [_legacy_msg(1, "A"), _legacy_msg(2, "B")], composer_id=CID_A
    )
    local = _legacy_conversation(
        [_legacy_msg(1, "A"), _legacy_msg(2, "X")], composer_id=CID_A
    )
    assert _classify(sync_env, local, remote) == syncstate.SyncRelation.DIVERGED


def test_legacy_changed_referenced_blob_is_diverged(sync_env):
    remote = _legacy_conversation(
        [_legacy_msg(1, "A", blob_id="blob-aaa", blob_data="v1")],
        composer_id=CID_A,
    )
    local = _legacy_conversation(
        [_legacy_msg(1, "A", blob_id="blob-aaa", blob_data="v2")],
        composer_id=CID_A,
    )
    assert _classify(sync_env, local, remote) == syncstate.SyncRelation.DIVERGED


def test_legacy_missing_required_blob_is_unknown(sync_env):
    remote = _legacy_conversation(
        [_legacy_msg(1, "A", blob_id="blob-aaa", blob_data="v1")],
        composer_id=CID_A,
    )
    local = _legacy_conversation(
        [_legacy_msg(1, "A", blob_id="blob-aaa", blob_data="v1")],
        composer_id=CID_A,
    )
    local["contentBlobs"] = {}
    assert _classify(sync_env, local, remote) == syncstate.SyncRelation.UNKNOWN


def test_legacy_non_dict_item_is_unknown(sync_env):
    snap = _legacy_conversation([_legacy_msg(1, "A")], composer_id=CID_A)
    snap["composerData"]["conversation"] = ["not-an-object"]
    assert _classify(sync_env, snap, snap) == syncstate.SyncRelation.UNKNOWN


def test_v3_legacy_sidecar_is_not_trusted_after_digest_bump(sync_env):
    snap = _legacy_conversation(
        [_legacy_msg(1, "A"), _legacy_msg(2, "B")], composer_id=CID_A
    )
    _commit_env(sync_env, [snap], [snap], digest=True, gzip_cids={CID_A})
    meta_path = sync_env["project_dir"] / f"{CID_A}.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["semanticDigestVersion"] = 3
    meta["semanticDigest"] = syncstate.conversation_digest([])
    meta_path.write_text(json.dumps(meta))

    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        rel = syncstate.classify_conversation(
            session, index, CID_A, project_identifier="project"
        )
        rec = index.get(CID_A, "project")
        assert rec is not None
        assert not index._sidecar_bound_to_body(rec)
        index.cache.flush()
    assert rel == syncstate.SyncRelation.UP_TO_DATE
    assert syncstate.op_counts().deep_snapshot_reads == 1
    cache = json.loads((sync_env["tmp"] / "cache" / "sync-semantics.json").read_text())
    stored = next(iter(cache["snapshots"].values()))
    assert stored["semanticDigestVersion"] == 4
    assert stored["semanticDigest"] != syncstate.conversation_digest([])


def test_legacy_semantic_cache_is_reused_after_first_parse(sync_env):
    snap = _legacy_conversation(
        [_legacy_msg(1, "A"), _legacy_msg(2, "B")], composer_id=CID_A
    )
    _commit_env(sync_env, [snap], [snap], digest=False, gzip_cids={CID_A})
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        syncstate.build_sync_plan(session, index)

    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
    assert [i.relation for i in plan.items] == [syncstate.SyncRelation.UP_TO_DATE]
    assert syncstate.op_counts().deep_snapshot_reads == 0
    assert syncstate.op_counts().local_semantic_rehashes == 0


def test_list_shows_legacy_and_uses_conversation_length(sync_env):
    snap = _legacy_conversation(
        [_legacy_msg(1, "A"), _legacy_msg(2, "B")], composer_id=CID_A, name="Old"
    )
    _commit_env(sync_env, [snap], [snap], digest=False)
    listed = export.list_conversations(
        PROJECT_PATH, workspace_dir=sync_env["ws_dir"]
    )
    assert len(listed) == 1
    assert listed[0]["id"] == CID_A
    assert listed[0]["messageCount"] == 2


def test_snapshot_sidecar_uses_conversation_length(tmp_path, monkeypatch):
    monkeypatch.setenv("CURSAVES_REPO_LOCK", str(tmp_path / "repo.lock"))
    monkeypatch.setenv("CURSAVES_REPO_LOCK_TIMEOUT", "1")
    from cursor_saves import dblock, paths

    dblock.reset_for_tests()
    monkeypatch.setattr(paths, "get_machine_id", lambda: "test-machine")
    monkeypatch.setattr(
        paths, "get_project_identifier", lambda path, source_host=None: "project"
    )
    snap = _legacy_conversation(
        [_legacy_msg(1, "A"), _legacy_msg(2, "B"), _legacy_msg(3, "C")],
        composer_id=CID_A,
    )
    export.save_snapshot(snap, tmp_path)
    meta = json.loads((tmp_path / "project" / f"{CID_A}.meta.json").read_text())
    assert meta["messageCount"] == 3
    assert meta["semanticDigestVersion"] == 4


def test_sync_with_identical_legacy_is_not_unsafe(sync_env, monkeypatch):
    snap = _legacy_conversation(
        [_legacy_msg(1, "A"), _legacy_msg(2, "B")], composer_id=CID_A
    )
    _commit_env(sync_env, [snap], [snap], digest=False, gzip_cids={CID_A})
    backend = _backend(monkeypatch)
    monkeypatch.setattr(
        cli,
        "import_snapshot",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("import")),
    )
    cli.cmd_sync(type("Args", (), {"force": False})())
    assert backend.pushes == 0


def test_local_guard_works_for_legacy(sync_env):
    local = _legacy_conversation([_legacy_msg(1, "A")], composer_id=CID_A)
    remote = _legacy_conversation(
        [_legacy_msg(1, "A"), _legacy_msg(2, "B")], composer_id=CID_A
    )
    _commit_env(sync_env, [local], [remote], digest=False, gzip_cids={CID_A})
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
        behind = plan.behind
        assert behind
        guard = behind[0].local_guard
        assert guard is not None
        assert guard.expect_present
        assert guard.row_fingerprint
        with db.CursorDB(sync_env["global_db"]) as gdb:
            with db.CursorDB(sync_env["ws_dir"] / "state.vscdb") as wdb:
                assert syncstate.local_guard_still_matches(behind[0], gdb, wdb)
        gconn = _init_db(sync_env["global_db"])
        _write_local(
            gconn,
            _legacy_conversation([_legacy_msg(1, "CHANGED")], composer_id=CID_A),
        )
        gconn.commit()
        gconn.close()
        with db.CursorDB(sync_env["global_db"]) as gdb:
            with db.CursorDB(sync_env["ws_dir"] / "state.vscdb") as wdb:
                assert not syncstate.local_guard_still_matches(behind[0], gdb, wdb)


def _unnamed_legacy(messages, *, composer_id: str = CID_A) -> dict:
    snap = _legacy_conversation(messages, composer_id=composer_id, name="")
    snap["composerData"].pop("name", None)
    return snap


def test_snapshot_malformed_headers_with_legacy_fallback_is_unknown(sync_env):
    remote = _legacy_conversation([_legacy_msg(1, "A")], composer_id=CID_A)
    remote["composerData"]["fullConversationHeadersOnly"] = None
    local = _legacy_conversation([_legacy_msg(1, "A")], composer_id=CID_A)
    with pytest.raises(syncstate.ClassifyError, match="fullConversationHeadersOnly"):
        syncstate.snapshot_unit_hashes(remote)
    assert _classify(sync_env, local, remote) == syncstate.SyncRelation.UNKNOWN


def test_snapshot_only_malformed_headers_is_not_import_candidate(sync_env):
    remote = _legacy_conversation([_legacy_msg(1, "A")], composer_id=CID_A)
    remote["composerData"]["fullConversationHeadersOnly"] = None
    _write_workspace(sync_env["ws_dir"], [])
    _init_db(sync_env["global_db"]).close()
    _write_snapshot_file(
        sync_env["project_dir"], remote, with_digest=False, gzip_body=True
    )
    target = {
        "path": PROJECT_PATH,
        "workspace_dir": sync_env["ws_dir"],
        "host": None,
        "type": "local",
    }
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        sync_plan = syncstate.build_sync_plan(session, index, target_workspace=target)
        pull_index = syncstate.SnapshotIndex.build_for_project("project")
        pull_plan = syncstate.build_pull_plan(session, pull_index, target)
    assert pull_plan.import_candidates == []
    assert sync_plan.items
    assert pull_plan.items
    assert all(
        item.relation == syncstate.SyncRelation.UNKNOWN for item in sync_plan.items
    )
    assert all(
        item.relation == syncstate.PullRelation.UNKNOWN for item in pull_plan.items
    )


def test_legacy_behind_without_name_is_actually_imported(sync_env):
    local = _unnamed_legacy([_legacy_msg(1, "A")])
    remote = _unnamed_legacy([_legacy_msg(1, "A"), _legacy_msg(2, "B")])
    _commit_env(sync_env, [local], [remote], digest=False, gzip_cids={CID_A})
    path = sync_env["project_dir"] / f"{CID_A}.json.gz"
    ok = importer.import_snapshot(
        path,
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        skip_backup=True,
        skip_conflict=True,
        quiet=True,
    )
    assert ok is True
    with db.CursorDB(sync_env["global_db"]) as cdb:
        written = cdb.get_json(f"composerData:{CID_A}")
    assert written is not None
    conversation = written.get("conversation") or []
    assert len(conversation) == 2
    assert conversation[1]["text"] == "B"


def test_direct_import_legacy_does_not_overwrite_existing_active_legacy(sync_env):
    local = _legacy_conversation(
        [_legacy_msg(1, "A"), _legacy_msg(2, "X")], composer_id=CID_A
    )
    remote = _legacy_conversation(
        [_legacy_msg(1, "A"), _legacy_msg(2, "B")], composer_id=CID_A
    )
    _commit_env(sync_env, [local], [remote], digest=False, gzip_cids={CID_A})
    path = sync_env["project_dir"] / f"{CID_A}.json.gz"
    ok = importer.import_snapshot(
        path,
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        skip_backup=True,
        quiet=True,
    )
    assert ok is False
    with db.CursorDB(sync_env["global_db"]) as cdb:
        written = cdb.get_json(f"composerData:{CID_A}")
    texts = [m.get("text") for m in (written or {}).get("conversation") or []]
    assert texts == ["A", "X"]


def test_direct_import_legacy_into_missing_local_still_imports(sync_env):
    remote = _legacy_conversation(
        [_legacy_msg(1, "A"), _legacy_msg(2, "B")], composer_id=CID_A
    )
    _write_workspace(sync_env["ws_dir"], [])
    _init_db(sync_env["global_db"]).close()
    path = _write_snapshot_file(
        sync_env["project_dir"], remote, with_digest=False, gzip_body=True
    )
    ok = importer.import_snapshot(
        path,
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        skip_backup=True,
        quiet=True,
    )
    assert ok is True
    with db.CursorDB(sync_env["global_db"]) as cdb:
        written = cdb.get_json(f"composerData:{CID_A}")
    conversation = (written or {}).get("conversation") or []
    assert [m.get("text") for m in conversation] == ["A", "B"]
