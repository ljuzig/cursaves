"""Derived header.grouping is not part of the semantic digest (v0.9.9)."""

from __future__ import annotations

import copy
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

_REAL_GROUPING = {
    "capabilityType": 15,
    "isRenderable": True,
    "isToolGroupable": True,
    "toolCallCase": "grepToolCall",
    "toolCallId": "tool-call-aaaaaaaa",
    "toolDisplayComputed": True,
    "toolDisplayPath": "/home/lju/architect/frontend",
}


def _with_header_grouping(snap: dict, grouping) -> dict:
    out = copy.deepcopy(snap)
    out["composerData"]["fullConversationHeadersOnly"][0]["grouping"] = grouping
    return out


def test_grouping_only_remote_is_up_to_date():
    remote = _with_header_grouping(
        _conversation([_msg(1, "A")], composer_id=CID_A), _REAL_GROUPING
    )
    local = _conversation([_msg(1, "A")], composer_id=CID_A)
    assert _relation(local, remote) == syncstate.SyncRelation.UP_TO_DATE


def test_grouping_only_local_is_up_to_date():
    remote = _conversation([_msg(1, "A")], composer_id=CID_A)
    local = _with_header_grouping(
        _conversation([_msg(1, "A")], composer_id=CID_A), _REAL_GROUPING
    )
    assert _relation(local, remote) == syncstate.SyncRelation.UP_TO_DATE


def test_different_grouping_contents_are_up_to_date():
    remote = _with_header_grouping(
        _conversation([_msg(1, "A")], composer_id=CID_A), _REAL_GROUPING
    )
    local = _with_header_grouping(
        _conversation([_msg(1, "A")], composer_id=CID_A),
        {**_REAL_GROUPING, "toolDisplayPath": "/tmp/other"},
    )
    assert _relation(local, remote) == syncstate.SyncRelation.UP_TO_DATE


def test_grouping_nested_arbitrary_fields_are_up_to_date():
    remote = _with_header_grouping(
        _conversation([_msg(1, "A")], composer_id=CID_A),
        {"nested": {"a": 1, "b": [True, {"c": "x"}]}},
    )
    local = _conversation([_msg(1, "A")], composer_id=CID_A)
    assert _relation(local, remote) == syncstate.SyncRelation.UP_TO_DATE


def test_same_grouping_bubble_change_is_diverged():
    remote = _with_header_grouping(
        _conversation([_msg(1, "A")], composer_id=CID_A), _REAL_GROUPING
    )
    local = _with_header_grouping(
        _conversation([_msg(1, "B")], composer_id=CID_A), _REAL_GROUPING
    )
    assert _relation(local, remote) == syncstate.SyncRelation.DIVERGED


def test_same_grouping_semantic_header_change_is_diverged():
    remote = _with_header_grouping(
        _conversation([_msg(1, "A")], composer_id=CID_A), _REAL_GROUPING
    )
    local = _with_header_grouping(
        _conversation([_msg(1, "A")], composer_id=CID_A), _REAL_GROUPING
    )
    local["composerData"]["fullConversationHeadersOnly"][0]["capabilityType"] = "extra"
    assert _relation(local, remote) == syncstate.SyncRelation.DIVERGED


def test_real_workspace_grouping_only_on_snapshot_is_up_to_date():
    remote = _with_header_grouping(
        _conversation([_msg(1, "A")], composer_id=CID_A), _REAL_GROUPING
    )
    local = _conversation([_msg(1, "A")], composer_id=CID_A)
    assert _relation(local, remote) == syncstate.SyncRelation.UP_TO_DATE


def test_v2_sidecar_digest_never_trusted_under_current(sync_env):
    snap = _conversation([_msg(1, "A")], composer_id=CID_A)
    _commit_env(sync_env, [snap], [snap], digest=True, gzip_cids={CID_A})
    meta_path = sync_env["project_dir"] / f"{CID_A}.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["semanticDigestVersion"] = 2
    meta["semanticDigest"] = "sha256:not-a-real-current-digest"
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

    cache_path = sync_env["tmp"] / "cache" / "sync-semantics.json"
    payload = json.loads(cache_path.read_text())
    assert payload["version"] == syncstate._CACHE_VERSION
    for rec in payload["snapshots"].values():
        assert rec["semanticDigestVersion"] == syncstate.SEMANTIC_DIGEST_VERSION
        assert rec["semanticDigest"] != "sha256:not-a-real-current-digest"
