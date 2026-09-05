"""Cursor UI layout metadata is not part of the semantic digest (v0.9.14)."""

from __future__ import annotations

import json

from cursor_saves import syncstate
from tests.test_syncstate import (
    CID_A,
    _commit_env,
    _conversation,
    _msg,
    _relation,
)

pytest_plugins = ["tests.test_syncstate"]

_BUBBLE = {"bubbleId": "bubble-1", "type": 1, "text": "A"}
_GROUPING_OLD = {
    "capabilityType": 15,
    "textPreview": "short",
}
_GROUPING_NEW = {
    "capabilityType": 15,
    "textPreview": "a much longer preview after opening the chat",
}


def _header(*, hint=None, grouping=None, extra=None) -> dict:
    header = {"bubbleId": "bubble-1", "type": 1}
    if hint is not None:
        header["contentHeightHint"] = hint
    if grouping is not None:
        header["grouping"] = grouping
    if extra:
        header.update(extra)
    return header


def _unit(header: dict) -> str:
    return syncstate.unit_hash(header, _BUBBLE, {})


def test_content_height_hint_value_changed_same_unit_hash():
    assert _unit(_header(hint=1453)) == _unit(_header(hint=1563))


def test_content_height_hint_present_or_missing_same_unit_hash():
    assert _unit(_header(hint=1453)) == _unit(_header())


def test_grouping_changed_same_unit_hash():
    assert _unit(_header(grouping=_GROUPING_OLD)) == _unit(
        _header(grouping=_GROUPING_NEW)
    )


def test_ordinary_semantic_header_changed_different_hash():
    assert _unit(_header(extra={"capabilityType": "extra"})) != _unit(_header())


def _v17_snapshot() -> dict:
    snap = _conversation(
        [_msg(1, "hello"), _msg(2, "world")],
        composer_id=CID_A,
        name="Ippotrack chat",
    )
    snap["composerData"]["_v"] = 17
    snap["composerData"]["activeCanvas"] = {"id": "canvas-1"}
    headers = snap["composerData"]["fullConversationHeadersOnly"]
    headers[0]["contentHeightHint"] = 1453
    headers[1]["contentHeightHint"] = 42
    headers[0]["grouping"] = dict(_GROUPING_OLD)
    return snap


def _v18_opened_local() -> dict:
    snap = _conversation(
        [_msg(1, "hello"), _msg(2, "world")],
        composer_id=CID_A,
        name="Ippotrack chat",
    )
    snap["composerData"]["_v"] = 18
    snap["composerData"]["committedCustomMode"] = None
    headers = snap["composerData"]["fullConversationHeadersOnly"]
    headers[0]["contentHeightHint"] = 1563
    headers[1]["contentHeightHint"] = 86
    headers[0]["grouping"] = dict(_GROUPING_NEW)
    return snap


def test_v17_snapshot_vs_v18_opened_local_is_up_to_date():
    remote = _v17_snapshot()
    local = _v18_opened_local()
    assert remote["bubbleEntries"] == local["bubbleEntries"]
    assert _relation(local, remote) == syncstate.SyncRelation.UP_TO_DATE


def test_v4_sidecar_and_v5_cache_are_recomputed_as_v5(sync_env):
    snap = _conversation([_msg(1, "A")], composer_id=CID_A)
    _commit_env(sync_env, [snap], [snap], digest=True, gzip_cids={CID_A})
    meta_path = sync_env["project_dir"] / f"{CID_A}.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["semanticDigestVersion"] = 4
    meta["semanticDigest"] = "sha256:not-a-real-current-digest"
    meta_path.write_text(json.dumps(meta))

    cache_path = sync_env["tmp"] / "cache" / "sync-semantics.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "version": 5,
                "snapshots": {
                    f"project|{CID_A}": {
                        "semanticDigest": "sha256:stale-v4-cache",
                        "semanticDigestVersion": 4,
                    }
                },
                "local": {},
            }
        )
    )

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

    payload = json.loads(cache_path.read_text())
    assert payload["version"] == syncstate._CACHE_VERSION
    assert payload["version"] == 6
    for rec in payload["snapshots"].values():
        assert rec["semanticDigestVersion"] == syncstate.SEMANTIC_DIGEST_VERSION
        assert rec["semanticDigestVersion"] == 5
        assert rec["semanticDigest"] != "sha256:not-a-real-current-digest"
        assert rec["semanticDigest"] != "sha256:stale-v4-cache"
