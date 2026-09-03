"""Missing-bubble tombstones are first-class semantic units (v0.9.8)."""

from __future__ import annotations

import copy
import json
import sqlite3

import pytest

from cursor_saves import syncstate
from tests.test_syncstate import (
    CID_A,
    _commit_env,
    _conversation,
    _msg,
    _relation,
)

pytest_plugins = ["tests.test_syncstate"]


def _drop_bubbles(snap: dict, *bids: str) -> dict:
    out = copy.deepcopy(snap)
    for bid in bids:
        out.get("bubbleEntries", {}).pop(bid, None)
        cmap = (out.get("composerData") or {}).get("conversationMap")
        if isinstance(cmap, dict):
            cmap.pop(bid, None)
    return out


def test_same_missing_bubble_both_sides_is_up_to_date():
    full = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A)
    remote = _drop_bubbles(full, "bubble-2")
    local = _drop_bubbles(full, "bubble-2")
    assert _relation(local, remote) == syncstate.SyncRelation.UP_TO_DATE


def test_same_missing_bubble_plus_remote_append_is_behind():
    base = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A)
    remote = _conversation(
        [_msg(1, "A"), _msg(2, "B"), _msg(3, "C")], composer_id=CID_A
    )
    local = _drop_bubbles(base, "bubble-2")
    remote = _drop_bubbles(remote, "bubble-2")
    assert _relation(local, remote) == syncstate.SyncRelation.BEHIND


def test_same_missing_bubble_plus_local_append_is_ahead():
    base = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A)
    local = _conversation(
        [_msg(1, "A"), _msg(2, "B"), _msg(3, "C")], composer_id=CID_A
    )
    remote = _drop_bubbles(base, "bubble-2")
    local = _drop_bubbles(local, "bubble-2")
    assert _relation(local, remote) == syncstate.SyncRelation.LOCAL_AHEAD


def test_missing_local_present_remote_is_diverged():
    full = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A)
    local = _drop_bubbles(full, "bubble-2")
    assert _relation(local, full) == syncstate.SyncRelation.DIVERGED


def test_present_local_missing_remote_is_diverged():
    full = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A)
    remote = _drop_bubbles(full, "bubble-2")
    assert _relation(full, remote) == syncstate.SyncRelation.DIVERGED


def test_same_missing_bubble_different_header_is_diverged():
    remote = _drop_bubbles(
        _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A), "bubble-2"
    )
    local = _drop_bubbles(
        _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A), "bubble-2"
    )
    local["composerData"]["fullConversationHeadersOnly"][1]["type"] = 2
    assert _relation(local, remote) == syncstate.SyncRelation.DIVERGED


def test_null_primary_bubble_falls_back_to_conversation_map():
    snap = _conversation([_msg(1, "A")], composer_id=CID_A)
    body = snap["bubbleEntries"]["bubble-1"]
    snap["bubbleEntries"]["bubble-1"] = None
    snap["composerData"]["conversationMap"] = {"bubble-1": body}
    hashes = syncstate.snapshot_unit_hashes(snap)
    full = _conversation([_msg(1, "A")], composer_id=CID_A)
    assert hashes == syncstate.snapshot_unit_hashes(full)


def test_null_local_bubble_falls_back_to_conversation_map(sync_env):
    snap = _conversation([_msg(1, "A")], composer_id=CID_A)
    _commit_env(sync_env, [snap], [snap], digest=False)
    body = snap["bubbleEntries"]["bubble-1"]
    conn = sqlite3.connect(str(sync_env["global_db"]))
    conn.execute(
        "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
        (f"bubbleId:{CID_A}:bubble-1", "null"),
    )
    row = conn.execute(
        "SELECT value FROM cursorDiskKV WHERE key = ?",
        (f"composerData:{CID_A}",),
    ).fetchone()
    data = json.loads(row[0] if not isinstance(row[0], bytes) else row[0].decode())
    data["conversationMap"] = {"bubble-1": body}
    conn.execute(
        "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
        (f"composerData:{CID_A}", json.dumps(data)),
    )
    conn.commit()
    conn.close()

    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        rel = syncstate.classify_conversation(
            session, index, CID_A, project_identifier="project"
        )
    assert rel == syncstate.SyncRelation.UP_TO_DATE


def test_corrupt_bubble_json_is_unknown():
    snap = _conversation([_msg(1, "A")], composer_id=CID_A)
    snap["bubbleEntries"]["bubble-1"] = "not-an-object"
    with pytest.raises(syncstate.ClassifyError, match="not an object"):
        syncstate.snapshot_unit_hashes(snap)


def test_corrupt_bubble_entries_container_is_unknown():
    snap = _conversation([_msg(1, "A")], composer_id=CID_A)
    snap["bubbleEntries"] = ["not", "an", "object"]
    with pytest.raises(syncstate.ClassifyError, match="bubbleEntries"):
        syncstate.snapshot_unit_hashes(snap)


def test_corrupt_local_conversation_map_is_unknown(sync_env):
    snap = _drop_bubbles(
        _conversation([_msg(1, "A")], composer_id=CID_A), "bubble-1"
    )
    _commit_env(sync_env, [snap], [snap], digest=False)
    conn = sqlite3.connect(str(sync_env["global_db"]))
    row = conn.execute(
        "SELECT value FROM cursorDiskKV WHERE key = ?",
        (f"composerData:{CID_A}",),
    ).fetchone()
    data = json.loads(row[0] if not isinstance(row[0], bytes) else row[0].decode())
    data["conversationMap"] = "not-an-object"
    conn.execute(
        "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
        (f"composerData:{CID_A}", json.dumps(data)),
    )
    conn.commit()
    conn.close()

    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        rel = syncstate.classify_conversation(
            session, index, CID_A, project_identifier="project"
        )
    assert rel == syncstate.SyncRelation.UNKNOWN


def test_missing_required_blob_is_unknown():
    snap = _conversation(
        [_msg(1, "A", blob_id="blob-aaa", blob_data="v1")],
        composer_id=CID_A,
    )
    del snap["contentBlobs"]["blob-aaa"]
    with pytest.raises(syncstate.ClassifyError, match="referenced blob missing"):
        syncstate.snapshot_unit_hashes(snap)


def test_missing_bubble_still_requires_header_blobs():
    snap = _conversation(
        [_msg(1, "A", blob_id="blob-aaa", blob_data="v1")],
        composer_id=CID_A,
    )
    snap["composerData"]["fullConversationHeadersOnly"][0]["contentHash"] = "blob-aaa"
    snap = _drop_bubbles(snap, "bubble-1")
    del snap["contentBlobs"]["blob-aaa"]
    with pytest.raises(syncstate.ClassifyError, match="referenced blob missing"):
        syncstate.snapshot_unit_hashes(snap)


def test_sparse_tombstones_match_when_positions_agree():
    msgs = [_msg(i, f"m{i}") for i in range(1, 21)]
    full = _conversation(msgs, composer_id=CID_A, name="Sparse")
    missing = [f"bubble-{i}" for i in (2, 5, 9, 14, 18)]
    remote = _drop_bubbles(full, *missing)
    local = _drop_bubbles(full, *missing)
    assert len(syncstate.snapshot_unit_hashes(remote)) == 20
    assert _relation(local, remote) == syncstate.SyncRelation.UP_TO_DATE


def test_task_style_almost_all_bubbles_missing_is_up_to_date():
    msgs = [_msg(i, f"t{i}") for i in range(1, 36)]
    full = _conversation(msgs, composer_id=CID_A, name="task-foo")
    missing = [f"bubble-{i}" for i in range(2, 36)]
    remote = _drop_bubbles(full, *missing)
    local = _drop_bubbles(full, *missing)
    hashes = syncstate.snapshot_unit_hashes(remote)
    assert len(hashes) == 35
    assert _relation(local, remote) == syncstate.SyncRelation.UP_TO_DATE
    assert hashes != syncstate.snapshot_unit_hashes(full)


def test_sql_error_reading_bubble_is_unknown(sync_env, monkeypatch):
    class BoomConn:
        def execute(self, *_a, **_k):
            raise sqlite3.OperationalError("disk I/O error")

    class BoomDB:
        def _reader_conn(self):
            return BoomConn()

    with pytest.raises(syncstate.ClassifyError, match="failed to read"):
        syncstate._strict_select_json(BoomDB(), f"bubbleId:{CID_A}:bubble-1")

    snap = _conversation([_msg(1, "A")], composer_id=CID_A)
    _commit_env(sync_env, [snap], [snap], digest=True, gzip_cids={CID_A})

    def boom(*_a, **_k):
        raise syncstate.ClassifyError("failed to read bubbleId: disk I/O error")

    monkeypatch.setattr(syncstate, "_strict_select_json", boom)
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        rel = syncstate.classify_conversation(
            session, index, CID_A, project_identifier="project"
        )
    assert rel == syncstate.SyncRelation.UNKNOWN


def test_corrupt_local_bubble_json_is_unknown(sync_env):
    snap = _conversation([_msg(1, "A")], composer_id=CID_A)
    _commit_env(sync_env, [snap], [snap], digest=True, gzip_cids={CID_A})
    conn = sqlite3.connect(str(sync_env["global_db"]))
    conn.execute(
        "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
        (f"bubbleId:{CID_A}:bubble-1", "not-json{"),
    )
    conn.commit()
    conn.close()

    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        rel = syncstate.classify_conversation(
            session, index, CID_A, project_identifier="project"
        )
    assert rel == syncstate.SyncRelation.UNKNOWN


def test_v1_sidecar_digest_never_trusted_under_v2(sync_env):
    snap = _conversation([_msg(1, "A")], composer_id=CID_A)
    _commit_env(sync_env, [snap], [snap], digest=True, gzip_cids={CID_A})
    meta_path = sync_env["project_dir"] / f"{CID_A}.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["semanticDigestVersion"] = 1
    meta["semanticDigest"] = "sha256:not-a-real-v2-digest"
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
    assert rel == syncstate.SyncRelation.UP_TO_DATE
    assert syncstate.op_counts().deep_snapshot_reads == 1
    assert syncstate.op_counts().legacy_snapshot_decompressions == 1

    cache_path = sync_env["tmp"] / "cache" / "sync-semantics.json"
    payload = json.loads(cache_path.read_text())
    assert payload["version"] == syncstate._CACHE_VERSION
    for rec in payload["snapshots"].values():
        assert rec["semanticDigestVersion"] == 2
        assert rec["semanticDigest"] != "sha256:not-a-real-v2-digest"

    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        rel = syncstate.classify_conversation(
            session, index, CID_A, project_identifier="project"
        )
    assert rel == syncstate.SyncRelation.UP_TO_DATE
    assert syncstate.op_counts().deep_snapshot_reads == 0


def test_nine_hundred_warm_v2_preserves_fast_path(sync_env):
    n = 900
    snaps = [
        _conversation(
            [_msg(1, f"x{i}")],
            composer_id=f"80000000-0000-0000-0000-{i:012d}",
            name=f"V2{i}",
        )
        for i in range(n)
    ]
    _commit_env(sync_env, snaps, snaps, digest=False)

    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        syncstate.build_sync_plan(session, index)

    cache_path = sync_env["tmp"] / "cache" / "sync-semantics.json"
    payload = json.loads(cache_path.read_text())
    assert payload["version"] == 4
    assert all(
        rec["semanticDigestVersion"] == 2 for rec in payload["snapshots"].values()
    )

    syncstate.reset_op_counts()
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        plan = syncstate.build_sync_plan(session, index)
    assert len(plan.items) == n
    assert all(i.relation == syncstate.SyncRelation.UP_TO_DATE for i in plan.items)
    assert syncstate.op_counts().deep_snapshot_reads == 0
    assert syncstate.op_counts().legacy_snapshot_decompressions == 0
    assert syncstate.op_counts().local_semantic_rehashes == 0


def test_local_and_snapshot_tombstones_classify_up_to_date(sync_env):
    full = _conversation(
        [_msg(i, f"m{i}") for i in range(1, 8)],
        composer_id=CID_A,
        name="Tombstones",
    )
    tomb = _drop_bubbles(full, "bubble-2", "bubble-4", "bubble-7")
    _commit_env(sync_env, [tomb], [tomb], digest=False)

    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        rel = syncstate.classify_conversation(
            session, index, CID_A, project_identifier="project"
        )
    assert rel == syncstate.SyncRelation.UP_TO_DATE
