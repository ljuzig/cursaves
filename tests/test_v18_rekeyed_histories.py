"""v17↔v18 compatibility lineage stays append-only (v0.9.16).

Cursaves never rewrites Cursor identifiers. These fixtures model an
index reconstruction in which leftover snapshot bodies remain as
physical rows. Content-only retries without leftover bodies stay
DIVERGED.
"""

from __future__ import annotations

import copy
import json

from cursor_saves import cli, db, lineage, syncstate
from tests.test_syncstate import (
    CID_A,
    PROJECT_PATH,
    WS_HASH,
    _active_texts,
    _commit_env,
    _composer_ids,
    _conversation,
    _fork_clone_id,
    _hashes,
    _init_db,
    _msg,
    _snapshot_cids,
    _workspace_cids,
    _write_local,
    _write_snapshot_file,
)
from tests.test_sync_workspace import _backend

pytest_plugins = ["tests.test_syncstate"]


def _set_v(snap: dict, version: int) -> dict:
    out = copy.deepcopy(snap)
    out["composerData"]["_v"] = version
    return out


def _rewrite_active_ids(snap: dict, *, suffix: str = "v18") -> dict:
    """Simulate a new active header list. Does not delete leftover bodies."""
    out = copy.deepcopy(snap)
    new_headers = []
    new_bubbles = {}
    for header in out["composerData"]["fullConversationHeadersOnly"]:
        old = header["bubbleId"]
        new = f"{old}-{suffix}"
        rewritten = dict(header)
        rewritten["bubbleId"] = new
        new_headers.append(rewritten)
        bubble = (out.get("bubbleEntries") or {}).get(old)
        if bubble is not None:
            body = dict(bubble)
            body["bubbleId"] = new
            new_bubbles[new] = body
    out["composerData"]["fullConversationHeadersOnly"] = new_headers
    out["bubbleEntries"] = new_bubbles
    return out


def _append(snap: dict, *messages: dict) -> dict:
    extra = _conversation(list(messages), composer_id=snap["composerId"])
    out = copy.deepcopy(snap)
    out["composerData"]["fullConversationHeadersOnly"].extend(
        extra["composerData"]["fullConversationHeadersOnly"]
    )
    out.setdefault("bubbleEntries", {}).update(extra["bubbleEntries"])
    return out


def _insert_after_first(snap: dict, message: dict) -> dict:
    extra = _conversation([message], composer_id=snap["composerId"])
    out = copy.deepcopy(snap)
    headers = out["composerData"]["fullConversationHeadersOnly"]
    out["composerData"]["fullConversationHeadersOnly"] = (
        headers[:1]
        + extra["composerData"]["fullConversationHeadersOnly"]
        + headers[1:]
    )
    out.setdefault("bubbleEntries", {}).update(extra["bubbleEntries"])
    return out


def _swap_first_two(snap: dict) -> dict:
    out = copy.deepcopy(snap)
    headers = out["composerData"]["fullConversationHeadersOnly"]
    headers[0], headers[1] = headers[1], headers[0]
    return out


def _merge_bubbles(target: dict, source: dict) -> dict:
    out = copy.deepcopy(target)
    out.setdefault("bubbleEntries", {}).update(
        copy.deepcopy(source.get("bubbleEntries") or {})
    )
    return out


def _store_plain(sync_env, local: dict, remote: dict) -> None:
    """No leftover bodies — a retry or replacement, not reconstruction."""
    _commit_env(sync_env, [local], [remote], digest=False)


def _store_reconstructed_local(sync_env, local: dict, remote: dict) -> None:
    """Local active index changed; snapshot bodies remain as extra local rows."""
    _commit_env(sync_env, [_merge_bubbles(local, remote)], [remote], digest=False)


def _store_reconstructed_snapshot(sync_env, local: dict, remote: dict) -> None:
    """Snapshot export kept leftover local bodies in the document."""
    _commit_env(sync_env, [local], [_merge_bubbles(remote, local)], digest=False)


def _classify(sync_env) -> syncstate.SyncRelation:
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        return syncstate.classify_conversation(
            session, index, CID_A, project_identifier="project"
        )


def _v17_base(*messages: dict) -> dict:
    return _set_v(_conversation(list(messages), composer_id=CID_A, name="Chat"), 17)


def _set_tool_call(snap: dict, bubble_id: str, tool_call_id: str) -> dict:
    out = copy.deepcopy(snap)
    out["bubbleEntries"][bubble_id]["toolCallId"] = tool_call_id
    return out


def test_digest_and_cache_versions_unchanged():
    assert syncstate.SEMANTIC_DIGEST_VERSION == 5
    assert syncstate._CACHE_VERSION == 6


def test_canonical_hashes_still_diverge_after_id_rewrite():
    v17 = _v17_base(_msg(1, "hello"), _msg(2, "world"))
    v18 = _set_v(_rewrite_active_ids(v17), 18)
    assert _hashes(v17) != _hashes(v18)
    assert syncstate.compare_unit_hashes(_hashes(v18), _hashes(v17)) == (
        syncstate.SyncRelation.DIVERGED
    )


def test_logical_fingerprint_ignores_ids_and_header_layout():
    header_a = {
        "bubbleId": "old",
        "type": 1,
        "createdAt": 1,
        "grouping": {"textPreview": "x"},
        "contentHeightHint": 10,
    }
    header_b = {
        "bubbleId": "new",
        "type": 1,
        "createdAt": 99,
        "grouping": {"textPreview": "yyyy"},
        "contentHeightHint": 99,
    }
    bubble_a = {"bubbleId": "old", "type": 1, "text": "hello", "serverBubbleId": "s1"}
    bubble_b = {"bubbleId": "new", "type": 1, "text": "hello", "serverBubbleId": "s2"}
    assert lineage.logical_fingerprint(header_a, bubble_a, {}) == (
        lineage.logical_fingerprint(header_b, bubble_b, {})
    )


def test_logical_fingerprint_keeps_bubble_root_layout_fields():
    header = {"bubbleId": "a", "type": 1}
    grouping_a = {"bubbleId": "a", "type": 1, "text": "hello", "grouping": {"x": 1}}
    grouping_b = {"bubbleId": "a", "type": 1, "text": "hello", "grouping": {"x": 2}}
    height_a = {"bubbleId": "a", "type": 1, "text": "hello", "contentHeightHint": 10}
    height_b = {"bubbleId": "a", "type": 1, "text": "hello", "contentHeightHint": 99}
    assert lineage.logical_fingerprint(header, grouping_a, {}) != (
        lineage.logical_fingerprint(header, grouping_b, {})
    )
    assert lineage.logical_fingerprint(header, height_a, {}) != (
        lineage.logical_fingerprint(header, height_b, {})
    )


def test_logical_fingerprint_keeps_text_and_tool_payloads():
    header = {"bubbleId": "a", "type": 2}
    text_a = {"bubbleId": "a", "type": 2, "text": "one"}
    text_b = {"bubbleId": "a", "type": 2, "text": "two"}
    tool_a = {"bubbleId": "a", "type": 2, "toolFormerData": {"name": "read", "a": 1}}
    tool_b = {"bubbleId": "a", "type": 2, "toolFormerData": {"name": "read", "a": 2}}
    call_a = {"bubbleId": "a", "type": 2, "text": "x", "toolCallId": "t1"}
    call_b = {"bubbleId": "a", "type": 2, "text": "x", "toolCallId": "t2"}
    assert lineage.logical_fingerprint(header, text_a, {}) != (
        lineage.logical_fingerprint(header, text_b, {})
    )
    assert lineage.logical_fingerprint(header, tool_a, {}) != (
        lineage.logical_fingerprint(header, tool_b, {})
    )
    assert lineage.logical_fingerprint(header, call_a, {}) != (
        lineage.logical_fingerprint(header, call_b, {})
    )


def test_reopened_without_new_content_is_up_to_date(sync_env):
    remote = _v17_base(_msg(1, "hello"), _msg(2, "world"))
    local = _set_v(_rewrite_active_ids(remote), 18)
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE
    assert syncstate.op_counts().compatibility_lineage_checks >= 1


def test_reconstruction_plus_continuation_is_local_ahead(sync_env):
    remote = _v17_base(_msg(1, "hello"), _msg(2, "world"))
    local = _append(_set_v(_rewrite_active_ids(remote), 18), _msg(10, "continued locally"))
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.LOCAL_AHEAD


def test_symmetric_remote_reconstruction_plus_continuation_is_behind(sync_env):
    local = _v17_base(_msg(1, "hello"), _msg(2, "world"))
    remote = _append(_set_v(_rewrite_active_ids(local), 18), _msg(10, "from other machine"))
    _store_reconstructed_snapshot(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.BEHIND


def test_reconstructed_identical_content_is_up_to_date(sync_env):
    remote = _v17_base(_msg(1, "only"))
    local = _set_v(_rewrite_active_ids(remote), 18)
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def test_identical_retry_without_leftover_bodies_is_up_to_date(sync_env):
    remote = _v17_base(
        _msg(1, "run tests"),
        _msg(2, "ok", toolFormerData={"name": "shell", "command": "pytest", "output": "X"}),
    )
    local = _set_v(_rewrite_active_ids(remote), 18)
    _store_plain(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def test_identical_retry_appended_after_reconstruction_is_local_ahead(sync_env):
    remote = _v17_base(_msg(1, "run tests"), _msg(2, "ok"))
    rebuilt = _set_v(_rewrite_active_ids(remote), 18)
    local = _append(rebuilt, _msg(10, "run tests"), _msg(11, "ok"))
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.LOCAL_AHEAD


def test_same_tool_payload_different_tool_call_id_is_diverged(sync_env):
    remote = _set_tool_call(
        _v17_base(
            _msg(1, "run", toolFormerData={"name": "shell", "command": "ls", "output": "x"})
        ),
        "bubble-1",
        "call-old",
    )
    local = _set_tool_call(
        _set_v(_rewrite_active_ids(remote), 18),
        "bubble-1-v18",
        "call-new",
    )
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_identical_acks_are_not_paired_arbitrarily(sync_env):
    remote = _v17_base(_msg(1, "yes"))
    local = _set_v(
        _rewrite_active_ids(
            _conversation(
                [_msg(1, "continue"), _msg(2, "yes"), _msg(3, "ok"), _msg(4, "yes")],
                composer_id=CID_A,
            )
        ),
        18,
    )
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.LOCAL_AHEAD


def test_leftover_bodies_without_unique_lineage_is_diverged(sync_env):
    remote = _v17_base(_msg(1, "hello"))
    local = _set_v(_conversation([_msg(9, "totally different")], composer_id=CID_A), 18)
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_leftover_snapshot_body_modified_is_diverged(sync_env):
    remote = _v17_base(_msg(1, "hello"))
    local = _set_v(_rewrite_active_ids(remote), 18)
    merged = _merge_bubbles(local, remote)
    merged["bubbleEntries"]["bubble-1"]["text"] = "HELLO"
    _store_plain(sync_env, merged, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_leftover_bubble_grouping_changed_is_diverged(sync_env):
    remote = _v17_base(_msg(1, "hello"))
    local = _set_v(_rewrite_active_ids(remote), 18)
    merged = _merge_bubbles(local, remote)
    merged["bubbleEntries"]["bubble-1"]["grouping"] = {"textPreview": "changed"}
    _store_plain(sync_env, merged, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_leftover_bubble_content_height_hint_changed_is_diverged(sync_env):
    remote = _v17_base(_msg(1, "hello"))
    local = _set_v(_rewrite_active_ids(remote), 18)
    merged = _merge_bubbles(local, remote)
    merged["bubbleEntries"]["bubble-1"]["contentHeightHint"] = 99
    _store_plain(sync_env, merged, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_header_layout_metadata_still_compatible(sync_env):
    remote = _v17_base(_msg(1, "hello"), _msg(2, "world"))
    local = _set_v(_rewrite_active_ids(remote), 18)
    for header in local["composerData"]["fullConversationHeadersOnly"]:
        header["grouping"] = {"textPreview": "ui"}
        header["contentHeightHint"] = 42
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def test_old_user_text_changed_is_diverged(sync_env):
    remote = _v17_base(_msg(1, "hello"), _msg(2, "world"))
    local = _set_v(_rewrite_active_ids(remote), 18)
    local["bubbleEntries"][local["composerData"]["fullConversationHeadersOnly"][0]["bubbleId"]][
        "text"
    ] = "HELLO"
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_old_tool_payload_changed_is_diverged(sync_env):
    remote = _v17_base(
        _msg(1, "run", toolFormerData={"name": "read", "path": "a"}),
        _msg(2, "ok"),
    )
    local = _set_v(_rewrite_active_ids(remote), 18)
    first = local["composerData"]["fullConversationHeadersOnly"][0]["bubbleId"]
    local["bubbleEntries"][first]["toolFormerData"] = {"name": "read", "path": "b"}
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_old_logical_turn_retired_with_physical_body_intact_is_allowed(sync_env):
    remote = _v17_base(_msg(1, "hello"), _msg(2, "world"), _msg(3, "again"))
    local = _set_v(_rewrite_active_ids(remote), 18)
    local["composerData"]["fullConversationHeadersOnly"].pop(1)
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def test_old_logical_turn_retired_with_physical_body_missing_is_behind(sync_env):
    remote = _v17_base(_msg(1, "hello"), _msg(2, "world"), _msg(3, "again"))
    local = _set_v(_rewrite_active_ids(remote), 18)
    local["composerData"]["fullConversationHeadersOnly"].pop(1)
    merged = _merge_bubbles(local, remote)
    merged["bubbleEntries"].pop("bubble-2", None)
    _store_plain(sync_env, merged, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.BEHIND


def test_source_tip_retired_is_up_to_date(sync_env):
    remote = _v17_base(_msg(1, "hello"), _msg(2, "world"), _msg(3, "again"))
    local = _set_v(_rewrite_active_ids(remote), 18)
    local["composerData"]["fullConversationHeadersOnly"].pop(2)
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def test_source_tip_retired_plus_continuation_is_local_ahead(sync_env, monkeypatch, capsys):
    remote = _v17_base(_msg(1, "hello"), _msg(2, "world"), _msg(3, "again"))
    local = _append(
        _set_v(_rewrite_active_ids(remote), 18),
        _msg(10, "continued after retirement"),
    )
    local["composerData"]["fullConversationHeadersOnly"] = [
        header
        for header in local["composerData"]["fullConversationHeadersOnly"]
        if header["bubbleId"] != "bubble-3-v18"
    ]
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.LOCAL_AHEAD
    _backend(monkeypatch)
    cli.cmd_sync(type("Args", (), {"force": False})())
    out = capsys.readouterr().out
    assert "Preserving both branches" not in out
    assert "aborted" not in out.lower()
    assert _composer_ids(sync_env) == {CID_A}
    assert CID_A in _snapshot_cids(sync_env["project_dir"])


def test_old_logical_turns_reordered_is_diverged(sync_env):
    remote = _v17_base(_msg(1, "hello"), _msg(2, "world"))
    local = _swap_first_two(_set_v(_rewrite_active_ids(remote), 18))
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_ambiguous_duplicate_logical_fingerprints_is_diverged(sync_env):
    remote = _v17_base(_msg(1, "same"))
    local = _set_v(
        _rewrite_active_ids(
            _conversation(
                [_msg(1, "same"), _msg(2, "mid"), _msg(3, "same")],
                composer_id=CID_A,
            )
        ),
        18,
    )
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.LOCAL_AHEAD


def test_extra_real_message_inside_history_is_local_ahead(sync_env):
    remote = _v17_base(_msg(1, "hello"), _msg(2, "world"))
    local = _insert_after_first(_set_v(_rewrite_active_ids(remote), 18), _msg(9, "inserted"))
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.LOCAL_AHEAD


def test_v17_to_v17_id_rewrite_does_not_enter_compatibility(sync_env):
    remote = _v17_base(_msg(1, "hello"))
    local = _set_v(_rewrite_active_ids(remote), 17)
    _store_reconstructed_local(sync_env, local, remote)
    syncstate.reset_op_counts()
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED
    assert syncstate.op_counts().compatibility_lineage_checks == 0


def test_v18_id_rewrite_without_physical_proof_is_up_to_date(sync_env):
    remote = _set_v(_conversation([_msg(1, "hello")], composer_id=CID_A), 18)
    local = _set_v(_rewrite_active_ids(remote), 18)
    _store_plain(sync_env, local, remote)
    syncstate.reset_op_counts()
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE
    assert syncstate.op_counts().compatibility_lineage_checks >= 1


def test_v18_to_v18_reconstruction_plus_continuation_is_local_ahead(sync_env):
    remote = _set_v(_conversation([_msg(1, "hello"), _msg(2, "world")], composer_id=CID_A), 18)
    local = _append(_set_v(_rewrite_active_ids(remote), 18), _msg(10, "continued locally"))
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.LOCAL_AHEAD
    assert syncstate.op_counts().compatibility_lineage_checks >= 1


def test_export_then_peer_v18_converges(sync_env):
    remote_v17 = _v17_base(_msg(1, "hello"), _msg(2, "world"))
    ahead = _append(
        _set_v(_rewrite_active_ids(remote_v17), 18),
        _msg(10, "continued locally"),
    )
    _store_reconstructed_local(sync_env, ahead, remote_v17)
    assert _classify(sync_env) == syncstate.SyncRelation.LOCAL_AHEAD

    with syncstate.SyncReadSession() as session:
        exported = session.export_conversation(PROJECT_PATH, CID_A)
    assert exported is not None
    assert exported["composerData"]["_v"] == 18
    leftover_ids = set((remote_v17.get("bubbleEntries") or {}))
    exported_ids = set((exported.get("bubbleEntries") or {}))
    assert leftover_ids <= exported_ids

    _write_snapshot_file(
        sync_env["project_dir"], exported, with_digest=False, gzip_body=True
    )
    peer = _set_v(remote_v17, 18)
    gconn = _init_db(sync_env["global_db"])
    gconn.execute("DELETE FROM cursorDiskKV")
    _write_local(gconn, peer)
    gconn.commit()
    gconn.close()
    assert _classify(sync_env) == syncstate.SyncRelation.BEHIND

    staging = sync_env["tmp"] / "behind-stage"
    staging.mkdir()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
    behind_ids = [item.composer_id for item in plan.behind]
    assert behind_ids == [CID_A]
    assert not plan.unsafe
    assert syncstate.stage_behind_snapshots(plan, staging)
    imported = cli._pull_behind(sync_env["sync_dir"], plan=plan)
    assert imported == 1
    with db.CursorDB(sync_env["global_db"]) as cdb:
        written = cdb.get_json(f"composerData:{CID_A}")
        extras = [
            key for key in cdb.list_keys("composerData:")
            if key != f"composerData:{CID_A}"
        ]
    assert written is not None
    assert written["_v"] == 18
    assert extras == []
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE
    with syncstate.SyncReadSession() as session:
        rebuilt = syncstate.build_sync_plan(session, syncstate.SnapshotIndex.build())
    item = next(row for row in rebuilt.items if row.composer_id == CID_A)
    assert item.relation == syncstate.SyncRelation.UP_TO_DATE


def test_unique_subsequence_rejects_ambiguous_and_missing():
    mapping, ambiguous = lineage.unique_subsequence_map(["a"], ["a", "b", "a"])
    assert mapping is None
    assert ambiguous
    mapping, ambiguous = lineage.unique_subsequence_map(["a", "c"], ["a", "b", "c"])
    assert mapping == [0, 2]
    assert ambiguous == []
    mapping, ambiguous = lineage.unique_subsequence_map(["x"], ["a"])
    assert mapping is None
    assert ambiguous == []


def test_content_map_without_physical_proof_is_not_enough():
    remote = snapshot_compat = [
        lineage.make_compat_unit(0, {"bubbleId": "old", "type": 1}, {"text": "run tests"}, {}),
        lineage.make_compat_unit(
            1,
            {"bubbleId": "old-tool", "type": 2},
            {"text": "ok", "toolFormerData": {"name": "shell", "output": "X"}},
            {},
        ),
    ]
    local = [
        lineage.make_compat_unit(0, {"bubbleId": "new", "type": 1}, {"text": "run tests"}, {}),
        lineage.make_compat_unit(
            1,
            {"bubbleId": "new-tool", "type": 2},
            {"text": "ok", "toolFormerData": {"name": "shell", "output": "X"}},
            {},
        ),
    ]
    assert lineage.classify_compat_lineage(local, snapshot_compat) == (
        syncstate.SyncRelation.UP_TO_DATE
    )
    leftover = {
        unit.bubble_id: lineage._payload_body_digest(unit.payload)
        for unit in snapshot_compat
    }
    dest_digests = {
        **leftover,
        **{
            unit.bubble_id: lineage._payload_body_digest(unit.payload)
            for unit in local
        },
    }
    assert lineage.classify_compat_lineage(
        local,
        snapshot_compat,
        local_digests=dest_digests,
        snapshot_digests=leftover,
        local_schema_v=18,
        snapshot_schema_v=17,
    ) == syncstate.SyncRelation.UP_TO_DATE


def test_lineage_report_fields_for_local_ahead(sync_env, monkeypatch, capsys):
    remote = _v17_base(_msg(1, "hello"), _msg(2, "world"))
    local = _append(_set_v(_rewrite_active_ids(remote), 18), _msg(10, "continued locally"))
    _store_reconstructed_local(sync_env, local, remote)
    monkeypatch.setattr(cli, "_ensure_synced", lambda: None)
    _backend(monkeypatch)
    cli.cmd_lineage(
        type(
            "Args",
            (),
            {
                "workspace": WS_HASH[:8],
                "project": None,
                "composer": CID_A,
                "json": True,
            },
        )()
    )
    report = json.loads(capsys.readouterr().out)
    assert report["snapshot_active_units"] == 2
    assert report["local_active_units"] == 3
    assert report["exact_logical_matches"] == 2
    assert report["source_missing_units"] == 0
    assert report["destination_extra_units"] == 1
    assert report["extras_before"] == 0
    assert report["extras_inside"] == 0
    assert report["extras_after"] == 1
    assert report["ambiguous_duplicate_fingerprints"] == []
    assert report["compatibility_candidate"] is True
    assert report["snapshot_bodies_preserved_locally"] is True
    assert report["mapping_direction"] == "snapshot_in_local"
    assert report["lineage_relation"] == "local_ahead"
    assert report["common_bubble_ids"] == 0
    assert report["source_missing_units"] == 0
    assert report["body_candidates_zero"] == 0
    assert report["body_candidates_unique"] == 0
    assert report["physical_snapshot_active_body_ids"] == 2
    assert report["physical_same_id_unchanged"] == 2
    assert report["physical_unaccounted"] == 0
    assert report["forensic_unexplained_units"] == 0


def test_behind_lineage_report_uses_local_in_snapshot_mapping(sync_env, monkeypatch, capsys):
    local = _v17_base(_msg(1, "hello"), _msg(2, "world"))
    remote = _append(_set_v(_rewrite_active_ids(local), 18), _msg(10, "from other machine"))
    _store_reconstructed_snapshot(sync_env, local, remote)
    monkeypatch.setattr(cli, "_ensure_synced", lambda: None)
    _backend(monkeypatch)
    cli.cmd_lineage(
        type(
            "Args",
            (),
            {
                "workspace": WS_HASH[:8],
                "project": None,
                "composer": CID_A,
                "json": True,
            },
        )()
    )
    report = json.loads(capsys.readouterr().out)
    assert report["compatibility_candidate"] is True
    assert report["mapping_direction"] == "local_in_snapshot"
    assert report["lineage_relation"] == "behind"
    assert report["exact_logical_matches"] == 2
    assert report["source_missing_units"] == 0
    assert report["destination_extra_units"] == 1
    assert report["extras_before"] == 0
    assert report["extras_inside"] == 0
    assert report["extras_after"] == 1


def test_lineage_flags_shared_bubble_id_with_changed_text():
    remote = _v17_base(_msg(1, "hello"))
    local = _set_v(copy.deepcopy(remote), 18)
    local["bubbleEntries"]["bubble-1"]["text"] = "HELLO"
    local_units = lineage.snapshot_compat_units(local)
    remote_units = lineage.snapshot_compat_units(remote)
    report = lineage.diagnose_lineage(
        local_units,
        remote_units,
        composer_id=CID_A,
        local_data=local["composerData"],
        remote_data=remote["composerData"],
    )
    assert report.common_bubble_ids == 1
    assert report.common_bubble_ids_different_canonical == 1
    assert report.common_bubble_ids_different_logical == 1
    assert report.lineage_relation is None


def test_diagnose_without_gate_does_not_report_a_relation():
    remote = _set_v(_conversation([_msg(1, "hello")], composer_id=CID_A), 16)
    local = _set_v(_rewrite_active_ids(remote), 16)
    report = lineage.diagnose_lineage(
        lineage.snapshot_compat_units(local),
        lineage.snapshot_compat_units(remote),
        composer_id=CID_A,
        local_data=local["composerData"],
        remote_data=remote["composerData"],
    )
    assert report.compatibility_candidate is False
    assert report.lineage_relation is None


def test_forensic_counts_header_only_replacements_without_classifying():
    remote = _v17_base(_msg(1, "hello"), _msg(2, "world"))
    for header in remote["composerData"]["fullConversationHeadersOnly"]:
        header["capabilityType"] = 7
    local = _set_v(_rewrite_active_ids(remote), 18)
    for header in local["composerData"]["fullConversationHeadersOnly"]:
        header.pop("capabilityType", None)
    local_units = lineage.snapshot_compat_units(local)
    remote_units = lineage.snapshot_compat_units(remote)
    report = lineage.diagnose_lineage(
        local_units,
        remote_units,
        composer_id=CID_A,
        local_data=local["composerData"],
        remote_data=remote["composerData"],
        snapshot=remote,
    )
    assert report.source_missing_units == 2
    assert report.body_candidates_zero == 0
    assert report.body_candidates_unique == 2
    assert report.body_candidates_ambiguous == 0
    assert report.unique_replacements_header_only == 2
    assert report.unique_replacements_body_diff == 0
    assert report.structural_categories == {"user": 2}
    assert report.header_path_histogram.get("header.capabilityType") == 2
    assert report.physical_same_body_other_id == 2
    assert report.physical_unaccounted == 0
    assert report.forensic_unexplained_units == 0
    assert report.lineage_relation is None
    assert lineage.classify_compat_lineage(
        local_units, remote_units, snapshot_bodies_local=True
    ) is None


def test_forensic_zero_candidates_when_text_changes():
    remote = _v17_base(_msg(1, "hello"))
    local = _set_v(_rewrite_active_ids(remote), 18)
    local["bubbleEntries"]["bubble-1-v18"]["text"] = "HELLO"
    report = lineage.diagnose_lineage(
        lineage.snapshot_compat_units(local),
        lineage.snapshot_compat_units(remote),
        composer_id=CID_A,
        local_data=local["composerData"],
        remote_data=remote["composerData"],
        snapshot=remote,
    )
    assert report.source_missing_units == 1
    assert report.body_candidates_zero == 1
    assert report.body_candidates_unique == 0
    assert report.physical_unaccounted == 1
    assert report.forensic_unexplained_units == 1
    assert report.lineage_relation is None


def test_forensic_ambiguous_when_two_dest_share_diagnostic_body():
    remote = _v17_base(_msg(1, "same"))
    remote["composerData"]["fullConversationHeadersOnly"][0]["capabilityType"] = 7
    local = _set_v(
        _conversation(
            [_msg(2, "same"), _msg(3, "same")],
            composer_id=CID_A,
        ),
        18,
    )
    report = lineage.diagnose_lineage(
        lineage.snapshot_compat_units(local),
        lineage.snapshot_compat_units(remote),
        composer_id=CID_A,
        local_data=local["composerData"],
        remote_data=remote["composerData"],
        snapshot=remote,
    )
    assert report.source_missing_units == 1
    assert report.body_candidates_ambiguous == 1
    assert report.body_candidates_unique == 0
    assert report.forensic_unexplained_units == 0
    assert report.lineage_relation is None


def test_forensic_categorizes_tool_and_simulated_units():
    remote = _v17_base(
        _msg(1, "run", type=2, toolFormerData={"name": "shell", "command": "ls"}),
        _msg(2, "done", type=2),
    )
    remote["bubbleEntries"]["bubble-2"]["isSimulatedMsg"] = True
    local = _set_v(_rewrite_active_ids(remote), 18)
    local["bubbleEntries"]["bubble-1-v18"]["toolFormerData"] = {
        "name": "shell",
        "command": "ls",
    }
    local["bubbleEntries"]["bubble-2-v18"]["isSimulatedMsg"] = True
    for header in remote["composerData"]["fullConversationHeadersOnly"]:
        header["capabilityType"] = 1
    for header in local["composerData"]["fullConversationHeadersOnly"]:
        header.pop("capabilityType", None)
    report = lineage.diagnose_lineage(
        lineage.snapshot_compat_units(local),
        lineage.snapshot_compat_units(remote),
        composer_id=CID_A,
        local_data=local["composerData"],
        remote_data=remote["composerData"],
        snapshot=remote,
    )
    assert report.body_candidates_unique == 2
    assert report.structural_categories == {
        "simulated": 1,
        "tool_call": 1,
    }


def test_forensic_probes_changed_physical_body_paths_and_samples():
    remote = _v17_base(
        _msg(1, "hello"),
        _msg(2, "run tool", toolFormerData={"name": "test", "status": "running"}),
    )
    local = _set_v(copy.deepcopy(remote), 18)
    local["bubbleEntries"]["bubble-2"]["toolFormerData"]["status"] = "finished"
    local["bubbleEntries"]["bubble-2"]["newField"] = 123
    report = lineage.diagnose_lineage(
        lineage.snapshot_compat_units(local),
        lineage.snapshot_compat_units(remote),
        composer_id=CID_A,
        local_data=local["composerData"],
        remote_data=remote["composerData"],
        snapshot=remote,
    )
    assert report.physical_same_id_changed == 1
    assert report.changed_physical_body_paths == {
        "newField": 1,
        "toolFormerData.status": 1,
    }
    sample_classes = {s["path_class"]: s for s in report.changed_physical_body_samples}
    assert "toolFormerData.status" in sample_classes
    assert sample_classes["toolFormerData.status"]["snapshot"] == "running"
    assert sample_classes["toolFormerData.status"]["local"] == "finished"
    assert "newField" in sample_classes
    assert sample_classes["newField"]["snapshot"] == "<missing>"
    assert sample_classes["newField"]["local"] == 123
    formatted = lineage.format_lineage_report(report)
    assert "changed physical bodies: 1" in formatted
    assert "toolFormerData.status" in formatted


def test_history_retirement_accounting_with_tip():
    remote = _v17_base(_msg(1, "A"), _msg(2, "B"), _msg(3, "C"))
    local = _set_v(_v17_base(_msg(1, "A"), _msg(3, "C"), _msg(4, "D")), 18)
    local["bubbleEntries"]["bubble-2"] = copy.deepcopy(remote["bubbleEntries"]["bubble-2"])
    report = lineage.diagnose_lineage(
        lineage.snapshot_compat_units(local),
        lineage.snapshot_compat_units(remote),
        composer_id=CID_A,
        local_data=local,
        remote_data=remote["composerData"],
        snapshot=remote,
    )
    assert report.active_represented == 2
    assert report.retired_preserved == 1
    assert report.body_rewritten == 0
    assert report.unaccounted_missing == 0
    assert report.logical_source_tip_preserved is True
    assert report.extras_before_anchor == 0
    assert report.extras_inside_anchor == 0
    assert report.extras_after_anchor == 1
    formatted = lineage.format_lineage_report(report)
    assert "retired preserved (physical intact):  1" in formatted
    assert "logical source tip preserved:         yes" in formatted
    assert "destination extras after anchor:      1" in formatted


def test_forensic_samples_every_path_class_with_array_indices():
    remote = _v17_base(
        _msg(1, "hello"),
        _msg(2, "context msg"),
    )
    remote["bubbleEntries"]["bubble-2"]["context"] = {
        "selectedImages": [
            {
                "path": "/images/11111111-1111-1111-1111-111111111111.png",
                "uuid": "old-uuid",
                "loadedAt": 100,
            }
        ],
        "fileSelections": [{"uri": "file:///a/b/c"}],
    }
    local = _set_v(copy.deepcopy(remote), 18)
    local["bubbleEntries"]["bubble-2"]["context"] = {
        "selectedImages": [
            {
                "path": "/images/22222222-2222-2222-2222-222222222222.png",
                "uuid": "new-uuid",
                "loadedAt": 200,
            }
        ],
        "fileSelections": [{"uri": "file:///a/b/d"}],
    }
    local["bubbleEntries"]["bubble-2"]["isPlanExecution"] = True
    report = lineage.diagnose_lineage(
        lineage.snapshot_compat_units(local),
        lineage.snapshot_compat_units(remote),
        composer_id=CID_A,
        local_data=local,
        remote_data=remote["composerData"],
        snapshot=remote,
    )
    assert report.physical_same_id_changed == 1
    sample_classes = {s["path_class"]: s for s in report.changed_physical_body_samples}
    assert "context.selectedImages[].path" in sample_classes
    assert "context.fileSelections[].uri" in sample_classes
    assert "isPlanExecution" in sample_classes
    # loadedAt and uuid are normalized away as migration/materialization metadata
    assert "context.selectedImages[].uuid" not in sample_classes
    assert "context.selectedImages[].loadedAt" not in sample_classes
    formatted = lineage.format_lineage_report(report)
    assert "[context.selectedImages[].path]" in formatted
    assert "[context.fileSelections[].uri]" in formatted
    assert "[isPlanExecution]" in formatted


def test_context_selection_text_changed_is_diverged(sync_env):
    remote = _v17_base(_msg(1, "look at code"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "selections": [{"uri": "file:///home/user/project/main.py", "text": "def old_func(): pass"}]
    }
    local = _set_v(_rewrite_active_ids(remote), 18)
    first_id = local["composerData"]["fullConversationHeadersOnly"][0]["bubbleId"]
    local["bubbleEntries"][first_id]["context"] = {
        "selections": [{"uri": "file:///home/user/project/main.py", "text": "def new_func(): pass"}]
    }
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_external_url_changed_is_diverged(sync_env):
    remote = _v17_base(_msg(1, "see docs"))
    remote["bubbleEntries"]["bubble-1"]["externalLinks"] = [{"url": "https://example.com/docs-v1"}]
    local = _set_v(_rewrite_active_ids(remote), 18)
    first_id = local["composerData"]["fullConversationHeadersOnly"][0]["bubbleId"]
    local["bubbleEntries"][first_id]["externalLinks"] = [{"url": "https://example.com/docs-v2"}]
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_selected_image_stable_identity_changed_is_diverged(sync_env):
    remote = _v17_base(_msg(1, "screenshot"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "selectedImages": [
            {"path": "/images/11111111-1111-1111-1111-111111111111-aaaa.png"}
        ]
    }
    local = _set_v(_rewrite_active_ids(remote), 18)
    first_id = local["composerData"]["fullConversationHeadersOnly"][0]["bubbleId"]
    local["bubbleEntries"][first_id]["context"] = {
        "selectedImages": [
            {"path": "/images/22222222-2222-2222-2222-222222222222-aaaa.png"}
        ]
    }
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_selected_image_only_materialization_metadata_changed_is_allowed(sync_env):
    remote = _v17_base(_msg(1, "screenshot"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "selectedImages": [
            {
                "path": "/images/11111111-1111-1111-1111-111111111111-aaaa1111-1111-1111-1111-111111111111.png",
                "loadedAt": 1000,
                "uuid": "old-mat-uuid",
            }
        ]
    }
    local = _set_v(_rewrite_active_ids(remote), 18)
    first_id = local["composerData"]["fullConversationHeadersOnly"][0]["bubbleId"]
    local["bubbleEntries"][first_id]["context"] = {
        "selectedImages": [
            {
                "path": "/images/11111111-1111-1111-1111-111111111111-bbbb2222-2222-2222-2222-222222222222.png",
                "loadedAt": 2000,
                "uuid": "new-mat-uuid",
            }
        ]
    }
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def test_v17_context_link_relocated_to_v18_root_external_links_is_allowed(sync_env):
    remote = _v17_base(_msg(1, "link check"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "externalLinks": [{"url": "https://example.com/ref", "addedWithoutMention": False}],
        "mentions": {
            "externalLinks": {
                "https://example.com/ref": {"url": "https://example.com/ref"}
            }
        },
    }
    local = _set_v(_rewrite_active_ids(remote), 18)
    first_id = local["composerData"]["fullConversationHeadersOnly"][0]["bubbleId"]
    local["bubbleEntries"][first_id]["externalLinks"] = [{"url": "https://example.com/ref", "uuid": "91"}]
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def test_runtime_metadata_rewritten_is_allowed(sync_env):
    remote = _v17_base(_msg(1, "hello"))
    remote["bubbleEntries"]["bubble-1"].update({
        "conversationState": "~stateTokenOld123",
        "modelInfo": {"modelName": "model-old"},
        "requestId": "old-req-id-1234",
        "contextWindowStatusAtCreation": {"tokenLimit": 200000, "tokensUsed": 50000},
    })
    local = _set_v(_rewrite_active_ids(remote), 18)
    first_id = local["composerData"]["fullConversationHeadersOnly"][0]["bubbleId"]
    local["bubbleEntries"][first_id].update({
        "conversationState": "~",
        "requestId": "",
        "conversationTurnIndex": 15,
        "isPlanExecution": False,
    })
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def test_source_tip_migration_equivalent_and_active_allows_continuation(sync_env):
    remote = _v17_base(_msg(1, "first"), _msg(2, "tip"))
    remote["bubbleEntries"]["bubble-2"].update({
        "conversationState": "~oldToken",
        "modelInfo": {"modelName": "fast"},
        "contextWindowStatusAtCreation": {"percentageRemaining": 90},
    })
    local = _set_v(_rewrite_active_ids(remote), 18)
    second_id = local["composerData"]["fullConversationHeadersOnly"][1]["bubbleId"]
    local["bubbleEntries"][second_id].update({
        "conversationState": "~",
        "conversationTurnIndex": 2,
    })
    local = _append(local, _msg(10, "appended after tip"))
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.LOCAL_AHEAD


def test_heterogeneous_selections_types_do_not_crash_and_preserve_semantics(sync_env):
    """Reproduces the exact TypeError: '<' not supported between instances of 'dict' and 'str'."""
    sels = [
        {"uri": "file:///path/a.py", "range": {"line": 1}, "text": "plain string text"},
        {"uri": "file:///path/b.py", "range": {"nested": {"start": 2, "end": 5}}, "text": {"structured": "dict", "tokens": [1, 2]}},
        {"uri": {"scheme": "custom", "target": "/c.py"}, "range": 42, "text": "mixed uri dict"},
    ]
    remote = _v17_base(_msg(1, "code review"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {"selections": copy.deepcopy(sels)}
    local = _set_v(_rewrite_active_ids(remote), 18)
    first_id = local["composerData"]["fullConversationHeadersOnly"][0]["bubbleId"]
    local["bubbleEntries"][first_id]["context"] = {"selections": copy.deepcopy(sels)}
    _store_reconstructed_local(sync_env, local, remote)
    # Must not raise TypeError, must classify as UP_TO_DATE
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE

    # Now change the dict text in one selection: must fail closed to DIVERGED
    local_changed = copy.deepcopy(local)
    local_changed["bubbleEntries"][first_id]["context"]["selections"][1]["text"]["tokens"] = [9, 9]
    _store_reconstructed_local(sync_env, local_changed, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_heterogeneous_links_files_images_do_not_crash(sync_env):
    """Verify links, files, and images with dict/non-string values do not crash sorting/dedupe."""
    remote = _v17_base(_msg(1, "resources"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "externalLinks": [
            "https://example.com/simple",
            {"url": {"custom_url_struct": "https://example.com/complex"}},
        ],
        "fileSelections": [
            "/simple/path.txt",
            {"uri": {"nested_uri": "file:///nested/path.txt"}},
        ],
        "selectedImages": [
            {"path": "/images/11111111-1111-1111-1111-111111111111.png"},
            {"unusual_field": {"not_a_normal_image": True}},
        ],
    }
    local = _set_v(_rewrite_active_ids(remote), 18)
    first_id = local["composerData"]["fullConversationHeadersOnly"][0]["bubbleId"]
    local["bubbleEntries"][first_id]["context"] = copy.deepcopy(
        remote["bubbleEntries"]["bubble-1"]["context"]
    )
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def test_compatibility_fallback_never_raises_on_hostile_shapes(sync_env):
    """Invariant: compatibility lineage fallback must never crash status or classify_conversation."""
    remote = _v17_base(_msg(1, "hostile"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "selections": [None, 12345, ["invalid", "inner", "list"]],
        "externalLinks": False,
        "mentions": {"selections": {"corrupt_json": "not valid json {"}},
    }
    local = _set_v(_rewrite_active_ids(remote), 18)
    first_id = local["composerData"]["fullConversationHeadersOnly"][0]["bubbleId"]
    local["bubbleEntries"][first_id]["context"] = {
        "selections": "totally unexpected string instead of list",
    }
    _store_reconstructed_local(sync_env, local, remote)

    # Calling classify_conversation must return DIVERGED without raising
    relation = _classify(sync_env)
    assert relation == syncstate.SyncRelation.DIVERGED

    # Unreadable session/rec must fail closed as UNKNOWN, not as a keep-both fork
    with syncstate.SyncReadSession() as session:
        bad_rec = type("Rec", (), {"composer_id": "non-existent-cid", "path": None})()
        assert (
            lineage.classify_after_canonical_diverged(session, bad_rec)
            == syncstate.SyncRelation.UNKNOWN
        )


def test_compatibility_internal_exception_is_unknown(sync_env, monkeypatch):
    remote = _set_v(_v17_base(_msg(1, "A"), _msg(2, "C")), 18)
    local = _set_v(_v17_base(_msg(1, "A"), _msg(2, "X")), 18)
    _store_plain(sync_env, local, remote)

    def boom(_bubble):
        raise TypeError("normalizer exploded")

    monkeypatch.setattr(lineage, "migration_normalize_body", boom)
    assert _classify(sync_env) == syncstate.SyncRelation.UNKNOWN
    with syncstate.SyncReadSession() as session:
        plan = syncstate.build_sync_plan(session, syncstate.SnapshotIndex.build())
    assert plan.unsafe


def test_unknown_nonempty_context_fields_are_not_silently_discarded(sync_env):
    """Fail-closed: unknown non-empty fields in context must not be stripped; differing ones must DIVERGE."""
    remote = _v17_base(_msg(1, "prompt"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "customTelemetryFlag": "telemetry-value-123",
        "browserSelections": [],  # empty boilerplate
    }
    local = _set_v(_rewrite_active_ids(remote), 18)
    first_id = local["composerData"]["fullConversationHeadersOnly"][0]["bubbleId"]
    local["bubbleEntries"][first_id]["context"] = {
        "browserSelections": [],  # missing customTelemetryFlag
    }
    _store_reconstructed_local(sync_env, local, remote)
    # Missing customTelemetryFlag on local must DIVERGE (not UP_TO_DATE)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED

    # If both have the same customTelemetryFlag, they match!
    local["bubbleEntries"][first_id]["context"]["customTelemetryFlag"] = "telemetry-value-123"
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE

    # Empty boilerplate collections on one side only are allowed
    local_empty_boilerplate = copy.deepcopy(local)
    local_empty_boilerplate["bubbleEntries"][first_id]["context"] = {
        "customTelemetryFlag": "telemetry-value-123",
        "browserSelections": [],
        "composers": [],
        "cursorCommands": [],
    }
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "customTelemetryFlag": "telemetry-value-123",
    }
    _store_reconstructed_local(sync_env, local_empty_boilerplate, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def test_real_shaped_mentions_keys_project_correctly_to_v18_root(sync_env):
    """Test v17 mentions where keys contain the semantic data and values are metadata."""
    sel_json = json.dumps({
        "uri": "file:///home/user/project/schema.sql",
        "range": {"selectionStartLineNumber": 1, "selectionStartColumn": 1, "positionLineNumber": 5, "positionColumn": 20},
        "text": "CREATE TABLE events (id INT);",
    })
    remote = _v17_base(_msg(1, "query database"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "mentions": {
            "externalLinks": {
                "https://example.com/api/v1": {"addedWithoutMention": False},
            },
            "fileSelections": {
                "file:///home/user/project/schema.sql": {"addedWithoutMention": False},
            },
            "selections": {
                sel_json: {"addedWithoutMention": False},
            },
        }
    }

    local = _set_v(_rewrite_active_ids(remote), 18)
    first_id = local["composerData"]["fullConversationHeadersOnly"][0]["bubbleId"]
    local["bubbleEntries"][first_id]["externalLinks"] = [
        {"url": "https://example.com/api/v1", "uuid": "link-uuid-1"},
    ]
    local["bubbleEntries"][first_id]["fileSelections"] = [
        {"uri": "file:///home/user/project/schema.sql"},
    ]
    local["bubbleEntries"][first_id]["selections"] = [
        {
            "uri": "file:///home/user/project/schema.sql",
            "range": {"selectionStartLineNumber": 1, "selectionStartColumn": 1, "positionLineNumber": 5, "positionColumn": 20},
            "text": "CREATE TABLE events (id INT);",
        }
    ]
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def test_real_shaped_mentions_semantic_value_changed_is_diverged(sync_env):
    """If semantic content in mentions keys changes, it must DIVERGE."""
    remote = _v17_base(_msg(1, "query"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "mentions": {
            "externalLinks": {"https://example.com/original": {}},
            "fileSelections": {"file:///home/user/project/original.py": {}},
        }
    }
    local = _set_v(_rewrite_active_ids(remote), 18)
    first_id = local["composerData"]["fullConversationHeadersOnly"][0]["bubbleId"]
    local["bubbleEntries"][first_id]["externalLinks"] = [{"url": "https://example.com/modified"}]
    local["bubbleEntries"][first_id]["fileSelections"] = [{"uri": "file:///home/user/project/original.py"}]
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_source_with_middle_retirements_allowed(sync_env):
    """S = [1, 2, 3, 4, 5], D = [1, 3, 5] with physical bodies 2 and 4 intact -> UP_TO_DATE."""
    remote = _v17_base(
        _msg(1, "one"),
        _msg(2, "two"),
        _msg(3, "three"),
        _msg(4, "four"),
        _msg(5, "five"),
    )
    local = _set_v(_rewrite_active_ids(remote), 18)
    headers = local["composerData"]["fullConversationHeadersOnly"]
    # Retires index 3 ('four') and index 1 ('two')
    local["composerData"]["fullConversationHeadersOnly"] = [headers[0], headers[2], headers[4]]
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def test_retirement_plus_continuation_is_local_ahead(sync_env):
    """S = [1, 2, 3], D = [1, 3, 4, 5] with physical body 2 intact -> LOCAL_AHEAD."""
    remote = _v17_base(
        _msg(1, "one"),
        _msg(2, "two"),
        _msg(3, "three"),
    )
    local = _set_v(_rewrite_active_ids(remote), 18)
    headers = local["composerData"]["fullConversationHeadersOnly"]
    local["composerData"]["fullConversationHeadersOnly"] = [headers[0], headers[2]]
    local = _append(local, _msg(10, "four"), _msg(11, "five"))
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.LOCAL_AHEAD


def test_destination_historical_unit_inserted_before_tip_is_diverged(sync_env):
    """If an extra historical unit is inserted before the source tip in D -> DIVERGED."""
    remote = _v17_base(
        _msg(1, "one"),
        _msg(2, "two"),
        _msg(3, "three"),
    )
    local = _set_v(_rewrite_active_ids(remote), 18)
    headers = local["composerData"]["fullConversationHeadersOnly"]
    extra = _conversation([_msg(99, "inserted between one and two")], composer_id=CID_A)
    extra_h = extra["composerData"]["fullConversationHeadersOnly"][0]
    local["composerData"]["fullConversationHeadersOnly"] = [headers[0], extra_h, headers[1], headers[2]]
    local.setdefault("bubbleEntries", {}).update(extra["bubbleEntries"])
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.LOCAL_AHEAD


def test_same_id_anchors_crossing_is_diverged(sync_env):
    """If active units with common bubbleIds cross order in destination -> DIVERGED."""
    remote = _v17_base(
        _msg(1, "first"),
        _msg(2, "second"),
        _msg(3, "third"),
    )
    # Local keeps IDs of msg 1 and msg 2 intact, but swaps them
    local = _set_v(copy.deepcopy(remote), 18)
    headers = local["composerData"]["fullConversationHeadersOnly"]
    headers[0], headers[1] = headers[1], headers[0]
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def _office_shaped_snapshots(
    *,
    n_exact: int = 14061,
    n_prune: int = 31,
    n_retired: int = 3049,
    n_append: int = 5465,
) -> tuple[dict, dict]:
    """v17 snapshot + v18 local with same-ID exact, same-ID context prune, retirements, appends.

    Source order: exact[0..n_exact-2], retired, prune, exact tip.
    Dest keeps exact + pruned (subset context) + tip, then appends. Retired
    bodies remain as leftover rows.
    """
    remote_headers: list[dict] = []
    remote_bubbles: dict[str, dict] = {}
    local_headers: list[dict] = []
    local_bubbles: dict[str, dict] = {}

    def add_exact(bid: str, text: str, *, dest_active: bool) -> None:
        header = {"bubbleId": bid, "type": 1}
        body = {"bubbleId": bid, "type": 1, "text": text}
        remote_headers.append(header)
        remote_bubbles[bid] = body
        if dest_active:
            local_headers.append(dict(header))
        local_bubbles[bid] = dict(body)

    for i in range(n_exact - 1):
        add_exact(f"exact-{i}", f"exact-turn-{i}", dest_active=True)

    for i in range(n_retired):
        add_exact(f"retired-{i}", f"retired-turn-{i}", dest_active=False)

    for i in range(n_prune):
        bid = f"prune-{i}"
        header = {"bubbleId": bid, "type": 1}
        links = [
            {"url": f"https://example.com/{i}/a"},
            {"url": f"https://example.com/{i}/b"},
            {"url": f"https://example.com/{i}/c"},
        ]
        remote_headers.append(header)
        remote_bubbles[bid] = {
            "bubbleId": bid,
            "type": 1,
            "text": f"prune-turn-{i}",
            "context": {"externalLinks": links},
        }
        local_headers.append(dict(header))
        local_bubbles[bid] = {
            "bubbleId": bid,
            "type": 1,
            "text": f"prune-turn-{i}",
            "context": {"externalLinks": [links[0], links[2]]},
        }

    add_exact(f"exact-{n_exact - 1}", f"exact-turn-{n_exact - 1}", dest_active=True)

    for i in range(n_append):
        bid = f"append-{i}"
        header = {"bubbleId": bid, "type": 1}
        body = {"bubbleId": bid, "type": 1, "text": f"append-turn-{i}"}
        local_headers.append(header)
        local_bubbles[bid] = body

    remote = {
        "composerId": CID_A,
        "composerData": {
            "_v": 17,
            "name": "Chat",
            "fullConversationHeadersOnly": remote_headers,
        },
        "bubbleEntries": remote_bubbles,
    }
    local = {
        "composerId": CID_A,
        "composerData": {
            "_v": 18,
            "name": "Chat",
            "fullConversationHeadersOnly": local_headers,
        },
        "bubbleEntries": local_bubbles,
    }
    return remote, local


def test_full_office_shaped_accounting_fixture():
    """Real Office shape: 14061 exact, 31 context-pruned, 3049 retired, 5465 append."""
    remote, local = _office_shaped_snapshots()
    local_units = lineage.snapshot_compat_units(local)
    remote_units = lineage.snapshot_compat_units(remote)
    local_digests = {
        bid: lineage._compat_body_digest(body)
        for bid, body in local["bubbleEntries"].items()
    }
    snap_digests = {
        bid: lineage._compat_body_digest(body)
        for bid, body in remote["bubbleEntries"].items()
    }
    assert lineage.classify_compat_lineage(
        local_units,
        remote_units,
        local_digests=local_digests,
        snapshot_digests=snap_digests,
        local_schema_v=18,
        snapshot_schema_v=17,
    ) == syncstate.SyncRelation.LOCAL_AHEAD

    report = lineage.diagnose_lineage(
        local_units,
        remote_units,
        composer_id=CID_A,
        local_data=local,
        remote_data=remote["composerData"],
        snapshot=remote,
    )
    assert report.compatibility_candidate is True
    assert report.lineage_relation == "local_ahead"
    assert report.active_represented == 14092
    assert report.migration_equivalent_active_rewrites == 31
    assert report.retired_preserved == 3049
    assert report.semantically_changed_active == 0
    assert report.unaccounted_missing == 0
    assert report.logical_source_tip_preserved is True
    assert report.extras_before_anchor == 0
    assert report.extras_inside_anchor == 0
    assert report.extras_after_anchor == 5465
    formatted = lineage.format_lineage_report(report)
    assert "migration-equivalent active rewrites: 31" in formatted
    assert "logical source tip preserved:         yes" in formatted
    assert report.proof_trace["common_id_context_pruned"] == 31
    assert report.proof_trace["common_id_rejected"] == 0
    assert report.proof_trace["tip_found"] is True
    assert report.proof_trace["final_reject_reason"] == ""


def test_same_id_v17_to_v18_external_links_pruned_is_allowed(sync_env):
    remote = _v17_base(_msg(1, "see docs"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "externalLinks": [
            {"url": "https://example.com/a"},
            {"url": "https://example.com/b"},
            {"url": "https://example.com/c"},
        ]
    }
    local = _set_v(copy.deepcopy(remote), 18)
    local["bubbleEntries"]["bubble-1"]["context"] = {
        "externalLinks": [
            {"url": "https://example.com/a"},
            {"url": "https://example.com/c"},
        ]
    }
    _store_plain(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def test_same_id_file_selections_pruned_is_allowed(sync_env):
    remote = _v17_base(_msg(1, "open files"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "fileSelections": [
            {"uri": "file:///home/user/a.py"},
            {"uri": "file:///home/user/b.py"},
            {"uri": "file:///home/user/c.py"},
        ]
    }
    local = _set_v(copy.deepcopy(remote), 18)
    local["bubbleEntries"]["bubble-1"]["context"] = {
        "fileSelections": [
            {"uri": "file:///home/user/a.py"},
            {"uri": "file:///home/user/c.py"},
        ]
    }
    _store_plain(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def test_same_id_selections_pruned_is_allowed(sync_env):
    remote = _v17_base(_msg(1, "look at code"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "selections": [
            {"uri": "file:///home/user/a.py", "text": "foo"},
            {"uri": "file:///home/user/b.py", "text": "bar"},
            {"uri": "file:///home/user/c.py", "text": "baz"},
        ]
    }
    local = _set_v(copy.deepcopy(remote), 18)
    local["bubbleEntries"]["bubble-1"]["context"] = {
        "selections": [
            {"uri": "file:///home/user/a.py", "text": "foo"},
            {"uri": "file:///home/user/c.py", "text": "baz"},
        ]
    }
    _store_plain(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def test_same_id_url_changed_is_diverged(sync_env):
    remote = _v17_base(_msg(1, "see docs"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "externalLinks": [{"url": "https://example.com/a"}]
    }
    local = _set_v(copy.deepcopy(remote), 18)
    local["bubbleEntries"]["bubble-1"]["context"] = {
        "externalLinks": [{"url": "https://example.com/b"}]
    }
    _store_plain(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_same_id_selection_text_changed_is_diverged(sync_env):
    remote = _v17_base(_msg(1, "look at code"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "selections": [{"uri": "file:///home/user/main.py", "text": "foo"}]
    }
    local = _set_v(copy.deepcopy(remote), 18)
    local["bubbleEntries"]["bubble-1"]["context"] = {
        "selections": [{"uri": "file:///home/user/main.py", "text": "bar"}]
    }
    _store_plain(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_same_id_text_changed_is_diverged(sync_env):
    remote = _v17_base(_msg(1, "hello"))
    local = _set_v(copy.deepcopy(remote), 18)
    local["bubbleEntries"]["bubble-1"]["text"] = "HELLO"
    _store_plain(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_same_id_tool_payload_changed_is_diverged(sync_env):
    remote = _v17_base(
        _msg(1, "run", toolFormerData={"name": "read", "path": "a"})
    )
    local = _set_v(copy.deepcopy(remote), 18)
    local["bubbleEntries"]["bubble-1"]["toolFormerData"] = {
        "name": "read",
        "path": "b",
    }
    _store_plain(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_same_id_blob_changed_is_diverged(sync_env):
    remote = _v17_base(_msg(1, "file", blob_id="blob-a", blob_data="AAAA"))
    local = _set_v(copy.deepcopy(remote), 18)
    local["contentBlobs"]["blob-a"] = "BBBB"
    _store_plain(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_same_id_residual_context_removed_is_diverged(sync_env):
    remote = _v17_base(_msg(1, "prompt"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "customTelemetryFlag": "telemetry-value-123",
        "externalLinks": [
            {"url": "https://example.com/a"},
            {"url": "https://example.com/b"},
        ],
    }
    local = _set_v(copy.deepcopy(remote), 18)
    local["bubbleEntries"]["bubble-1"]["context"] = {
        "externalLinks": [{"url": "https://example.com/a"}],
    }
    _store_plain(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_same_id_context_added_is_diverged(sync_env):
    remote = _v17_base(_msg(1, "see docs"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "externalLinks": [{"url": "https://example.com/a"}]
    }
    local = _set_v(copy.deepcopy(remote), 18)
    local["bubbleEntries"]["bubble-1"]["context"] = {
        "externalLinks": [
            {"url": "https://example.com/a"},
            {"url": "https://example.com/b"},
        ]
    }
    _store_plain(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.DIVERGED


def test_same_id_prunes_retirements_and_append_is_local_ahead(sync_env):
    remote = _v17_base(
        _msg(1, "keep-a"),
        _msg(2, "retire-me"),
        _msg(3, "prune-me"),
        _msg(4, "tip"),
    )
    remote["bubbleEntries"]["bubble-3"]["context"] = {
        "externalLinks": [
            {"url": "https://example.com/a"},
            {"url": "https://example.com/b"},
            {"url": "https://example.com/c"},
        ]
    }
    local = _set_v(copy.deepcopy(remote), 18)
    local["composerData"]["fullConversationHeadersOnly"] = [
        header
        for header in local["composerData"]["fullConversationHeadersOnly"]
        if header["bubbleId"] != "bubble-2"
    ]
    local["bubbleEntries"]["bubble-3"]["context"] = {
        "externalLinks": [
            {"url": "https://example.com/a"},
            {"url": "https://example.com/c"},
        ]
    }
    local = _append(local, _msg(10, "continued"))
    _store_reconstructed_local(sync_env, local, remote)
    assert _classify(sync_env) == syncstate.SyncRelation.LOCAL_AHEAD


def test_independent_v18_peers_shared_leftovers_is_up_to_date(sync_env):
    v17 = _v17_base(_msg(1, "hello"), _msg(2, "world"))
    office = _set_v(_rewrite_active_ids(v17, suffix="office"), 18)
    zeus = _set_v(_rewrite_active_ids(v17, suffix="zeus"), 18)
    _store_plain(
        sync_env,
        _merge_bubbles(zeus, v17),
        _merge_bubbles(office, v17),
    )
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def test_independent_v18_peers_shared_leftovers_continuation_is_behind(sync_env):
    v17 = _v17_base(_msg(1, "hello"), _msg(2, "world"))
    office = _append(
        _set_v(_rewrite_active_ids(v17, suffix="office"), 18),
        _msg(10, "from office"),
    )
    zeus = _set_v(_rewrite_active_ids(v17, suffix="zeus"), 18)
    _store_plain(
        sync_env,
        _merge_bubbles(zeus, v17),
        _merge_bubbles(office, v17),
    )
    assert _classify(sync_env) == syncstate.SyncRelation.BEHIND


def test_independent_v18_peers_shared_leftovers_pull_converges(sync_env):
    v17 = _v17_base(_msg(1, "hello"), _msg(2, "world"))
    office = _append(
        _set_v(_rewrite_active_ids(v17, suffix="office"), 18),
        _msg(10, "from office"),
    )
    zeus = _set_v(_rewrite_active_ids(v17, suffix="zeus"), 18)
    _store_plain(
        sync_env,
        _merge_bubbles(zeus, v17),
        _merge_bubbles(office, v17),
    )
    assert _classify(sync_env) == syncstate.SyncRelation.BEHIND

    staging = sync_env["tmp"] / "v18-peer-stage"
    staging.mkdir()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
    assert [item.composer_id for item in plan.behind] == [CID_A]
    assert not plan.unsafe
    assert syncstate.stage_behind_snapshots(plan, staging)
    imported = cli._pull_behind(sync_env["sync_dir"], plan=plan)
    assert imported == 1
    assert _composer_ids(sync_env) == {CID_A}
    assert _workspace_cids(sync_env["ws_dir"]) == {CID_A}
    with db.CursorDB(sync_env["global_db"]) as cdb:
        extras = [
            key for key in cdb.list_keys("composerData:")
            if key != f"composerData:{CID_A}"
        ]
    assert extras == []
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def test_independent_v18_same_logical_without_shared_ancestor_is_up_to_date(sync_env):
    v17 = _v17_base(_msg(1, "hello"), _msg(2, "world"))
    office = _set_v(_rewrite_active_ids(v17, suffix="office"), 18)
    zeus = _set_v(_rewrite_active_ids(v17, suffix="zeus"), 18)
    _store_plain(sync_env, zeus, office)
    assert _classify(sync_env) == syncstate.SyncRelation.UP_TO_DATE


def _diagnose_pair(local: dict, remote: dict) -> lineage.LineageReport:
    return lineage.diagnose_lineage(
        lineage.snapshot_compat_units(local),
        lineage.snapshot_compat_units(remote),
        composer_id=CID_A,
        local_data=local,
        remote_data=remote["composerData"],
        snapshot=remote,
    )


def test_failed_proof_still_reports_why_same_id_url_changed():
    """A failed proof must still name the gate; it must not look like prune=0/tip=no."""
    remote = _v17_base(_msg(1, "see docs"), _msg(2, "tip"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "externalLinks": [{"url": "https://example.com/a"}]
    }
    local = _set_v(copy.deepcopy(remote), 18)
    local["bubbleEntries"]["bubble-1"]["context"] = {
        "externalLinks": [{"url": "https://example.com/b"}]
    }
    report = _diagnose_pair(local, remote)
    assert report.lineage_relation is None
    assert report.proof_trace["common_id_rejected"] == 1
    assert report.proof_trace["rejected_context_not_subset"] == 1
    assert report.proof_trace["final_reject_reason"] == "same-ID context not subset"
    formatted = lineage.format_lineage_report(report)
    assert "Compatibility proof trace:" in formatted
    assert "final reject reason:      same-ID context not subset" in formatted
    assert lineage.classify_compat_lineage(
        lineage.snapshot_compat_units(local),
        lineage.snapshot_compat_units(remote),
        local_schema_v=18,
        snapshot_schema_v=17,
    ) is None


def test_failed_proof_reports_header_mismatch_not_as_missing_tip():
    remote = _v17_base(_msg(1, "hello"), _msg(2, "tip"))
    remote["composerData"]["fullConversationHeadersOnly"][0]["capabilityType"] = 7
    local = _set_v(copy.deepcopy(remote), 18)
    local["composerData"]["fullConversationHeadersOnly"][0].pop("capabilityType", None)
    report = _diagnose_pair(local, remote)
    assert report.lineage_relation is None
    assert report.proof_trace["rejected_header_mismatch"] == 1
    assert report.proof_trace["final_reject_reason"] == "same-ID header mismatch"
    assert report.proof_trace["tip_found"] is True


def test_failed_proof_reports_other_body_mismatch_for_text_change():
    remote = _v17_base(_msg(1, "hello"), _msg(2, "tip"))
    local = _set_v(copy.deepcopy(remote), 18)
    local["bubbleEntries"]["bubble-1"]["text"] = "HELLO"
    report = _diagnose_pair(local, remote)
    assert report.lineage_relation is None
    assert report.proof_trace["rejected_other_body_mismatch"] == 1
    assert report.proof_trace["final_reject_reason"] == "same-ID other body mismatch"


def test_failed_proof_reports_blob_mismatch():
    remote = _v17_base(
        _msg(1, "file", blob_id="blob-a", blob_data="AAAA"),
        _msg(2, "tip"),
    )
    local = _set_v(copy.deepcopy(remote), 18)
    local["contentBlobs"]["blob-a"] = "BBBB"
    report = _diagnose_pair(local, remote)
    assert report.lineage_relation is None
    assert report.proof_trace["rejected_blobs_mismatch"] == 1
    assert report.proof_trace["final_reject_reason"] == "same-ID blobs mismatch"


def test_failed_proof_keeps_accepted_prunes_when_tip_is_missing():
    """Prune can succeed and the later gate still has to be named."""
    remote = _v17_base(_msg(1, "keep"), _msg(2, "tip"))
    remote["bubbleEntries"]["bubble-1"]["context"] = {
        "externalLinks": [
            {"url": "https://example.com/a"},
            {"url": "https://example.com/b"},
        ]
    }
    local = _set_v(copy.deepcopy(remote), 18)
    local["bubbleEntries"]["bubble-1"]["context"] = {
        "externalLinks": [{"url": "https://example.com/a"}]
    }
    local["composerData"]["fullConversationHeadersOnly"].pop(1)
    report = _diagnose_pair(local, remote)
    assert report.lineage_relation == "up_to_date"
    assert report.proof_trace["common_id_context_pruned"] == 1
    assert report.proof_trace["common_id_rejected"] == 0
    assert report.proof_trace["tip_found"] is False
    assert report.proof_trace["final_reject_reason"] == "source tip not found"
    formatted = lineage.format_lineage_report(report)
    assert "context-pruned:         1" in formatted
    assert "found:                  no" in formatted
    assert "final reject reason:      source tip not found" in formatted
    assert "sync action: same-CID" in formatted


def test_failed_proof_reports_missing_retired_body():
    remote = _v17_base(_msg(1, "hello"), _msg(2, "world"), _msg(3, "again"))
    local = _set_v(_rewrite_active_ids(remote), 18)
    local["composerData"]["fullConversationHeadersOnly"].pop(1)
    merged = _merge_bubbles(local, remote)
    merged["bubbleEntries"].pop("bubble-2", None)
    report = _diagnose_pair(merged, remote)
    assert report.lineage_relation == "behind"
    assert report.proof_trace["retirement_missing_body"] >= 1
    assert report.proof_trace["final_reject_reason"] == "retired body missing"
