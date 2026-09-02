"""Incremental pull + batched imports (v0.9.6)."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest

from cursor_saves import cli, db, dblock, importer, paths, pull, syncstate
from tests.test_syncstate import (
    CID_A,
    CID_B,
    CID_C,
    HOST_A,
    HOST_B,
    PROJECT_PATH,
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

pytest_plugins = ["tests.test_syncstate"]


def _cid(n: int) -> str:
    return f"{n:08x}-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def _ws(sync_env, *, host=None, path=PROJECT_PATH, ws_dir=None) -> dict:
    return {
        "path": path,
        "workspace_dir": ws_dir or sync_env["ws_dir"],
        "host": host,
        "type": "ssh" if host else "local",
    }


def _count_write_locks(monkeypatch):
    n = {"n": 0}
    real = dblock.acquire_write_lock

    def wrapped(*args, **kwargs):
        n["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(dblock, "acquire_write_lock", wrapped)
    return n


def _local_composer(global_db: Path, composer_id: str) -> dict | None:
    conn = sqlite3.connect(str(global_db))
    row = conn.execute(
        "SELECT value FROM cursorDiskKV WHERE key = ?",
        (f"composerData:{composer_id}",),
    ).fetchone()
    conn.close()
    if not row:
        return None
    val = row[0]
    if isinstance(val, bytes):
        val = val.decode()
    return json.loads(val)


def _commit_real(env: dict, locals_: list[dict], remotes: list[dict]) -> None:
    """Like ``_commit_env``, but every remote is a real gzip (needed to import)."""
    gconn = _init_db(env["global_db"])
    for snap in locals_:
        _write_local(gconn, snap)
    gconn.commit()
    gconn.close()
    _write_workspace(env["ws_dir"], locals_)
    for snap in remotes:
        _write_snapshot_file(env["project_dir"], snap, with_digest=True, gzip_body=True)


def _write_ws_json(ws_dir: Path, path: str = PROJECT_PATH, host: str | None = None) -> None:
    ws_dir.mkdir(parents=True, exist_ok=True)
    if host:
        uri = f"vscode-remote://ssh-remote+{host}{path}"
        (ws_dir / "workspace.json").write_text(json.dumps({"folder": uri}))
    else:
        (ws_dir / "workspace.json").write_text(json.dumps({"folder": f"file://{path}"}))


def test_synced_ahead_local_only_is_noop(sync_env, monkeypatch, capsys):
    synced = [
        _conversation([_msg(1, f"s{i}")], composer_id=_cid(i), name=f"S{i}")
        for i in range(96)
    ]
    ahead_remote = _conversation([_msg(1, "a")], composer_id=CID_A, name="Ahead")
    ahead_local = _conversation(
        [_msg(1, "a"), _msg(2, "b")], composer_id=CID_A, name="Ahead"
    )
    local_only = _conversation([_msg(1, "only")], composer_id=CID_B, name="Local")
    _commit_real(sync_env, synced + [ahead_local, local_only], synced + [ahead_remote])
    locks = _count_write_locks(monkeypatch)
    monkeypatch.setattr(pull, "is_cursor_running", lambda: True)
    syncstate.reset_op_counts()

    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=False,
    )
    out = capsys.readouterr().out
    counts = syncstate.op_counts()
    assert result.imported == 0
    assert len(result.plan.import_candidates) == 0
    assert counts.imports_attempted == 0
    assert counts.safety_global_backups == 0
    assert counts.safety_workspace_backups == 0
    assert counts.write_connections_opened == 0
    assert counts.cursor_running_checks == 0
    assert locks["n"] == 0
    assert "96 already synced" in out
    assert "1 local ahead" in out
    assert "1 local only" in out
    assert "Nothing to import." in out


def test_one_behind_stages_and_imports_once(sync_env):
    synced = [
        _conversation([_msg(1, f"s{i}")], composer_id=_cid(i), name=f"S{i}")
        for i in range(96)
    ]
    local_b = _conversation([_msg(1, "a")], composer_id=CID_A, name="Behind")
    remote_b = _conversation(
        [_msg(1, "a"), _msg(2, "b")], composer_id=CID_A, name="Behind"
    )
    _commit_real(sync_env, synced + [local_b], synced + [remote_b])
    syncstate.reset_op_counts()

    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    counts = syncstate.op_counts()
    assert result.imported == 1
    assert counts.staged_snapshots == 1
    assert counts.imports_attempted == 1
    assert counts.imports_completed == 1
    written = _local_composer(sync_env["global_db"], CID_A)
    assert len(written["fullConversationHeadersOnly"]) == 2


def test_local_ahead_is_skipped(sync_env):
    remote = _conversation([_msg(1, "a")], composer_id=CID_A, name="Ahead")
    local = _conversation(
        [_msg(1, "a"), _msg(2, "b")], composer_id=CID_A, name="Ahead"
    )
    _commit_real(sync_env, [local], [remote])
    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    assert result.imported == 0
    assert result.plan.ahead[0].action == syncstate.PullAction.SKIP
    assert syncstate.op_counts().imports_attempted == 0


def test_global_cid_collision_does_not_overwrite(sync_env, capsys):
    """CID present globally for workspace B must not be imported into A."""
    snap_a = _conversation([_msg(1, "from-snap")], composer_id=CID_A, name="X")
    other = _conversation([_msg(1, "from-b")], composer_id=CID_A, name="B-local")
    other_ws = sync_env["tmp"] / "cursor" / "workspaceStorage" / ("b" * 32)
    gconn = _init_db(sync_env["global_db"])
    _write_local(gconn, other)
    gconn.commit()
    gconn.close()
    _write_workspace(sync_env["ws_dir"], [])
    _write_workspace(other_ws, [other])
    _write_ws_json(other_ws, "/home/user/other")
    _write_snapshot_file(sync_env["project_dir"], snap_a, gzip_body=True)

    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build_for_project("project")
        plan = syncstate.build_pull_plan(session, index, _ws(sync_env))
    assert len(plan.collisions) == 1
    assert plan.collisions[0].action == syncstate.PullAction.SKIP
    assert plan.import_candidates == []

    syncstate.reset_op_counts()
    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    out = capsys.readouterr().out
    assert result.imported == 0
    assert syncstate.op_counts().imports_attempted == 0
    assert "1 existing elsewhere" in out
    written = _local_composer(sync_env["global_db"], CID_A)
    assert written["name"] == "B-local"
    assert written["fullConversationHeadersOnly"][0]["bubbleId"] == "bubble-1"
    conn = sqlite3.connect(str(other_ws / "state.vscdb"))
    ws_data = json.loads(
        conn.execute(
            "SELECT value FROM ItemTable WHERE key = ?",
            ("composer.composerData",),
        ).fetchone()[0]
    )
    conn.close()
    assert ws_data["allComposers"][0]["composerId"] == CID_A
    assert ws_data["allComposers"][0]["name"] == "B-local"


def test_absent_from_target_and_global_is_missing_local(sync_env):
    snap = _conversation([_msg(1, "new")], composer_id=CID_A, name="New")
    _write_workspace(sync_env["ws_dir"], [])
    _write_snapshot_file(sync_env["project_dir"], snap, gzip_body=True)
    _init_db(sync_env["global_db"]).close()
    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    assert result.imported == 1
    assert result.plan.missing_local[0].composer_id == CID_A
    assert _local_composer(sync_env["global_db"], CID_A)["name"] == "New"


def test_interactive_selected_snapshot_can_restore_to_different_local_path(sync_env):
    chosen = _conversation(
        [_msg(1, "keep")],
        composer_id=CID_A,
        name="Chosen",
        sourceProjectPath="/old/foo",
        projectIdentifier="original-project",
    )
    other = _conversation(
        [_msg(1, "skip-me")],
        composer_id=CID_B,
        name="Other",
        sourceProjectPath="/old/foo",
        projectIdentifier="original-project",
    )
    bucket = sync_env["snaps"] / "original-project"
    chosen_path = _write_snapshot_file(bucket, chosen, gzip_body=True)
    _write_snapshot_file(bucket, other, gzip_body=True)
    _init_db(sync_env["global_db"]).close()

    restore_path = "/restore/completely-different-name"
    restore_ws = sync_env["tmp"] / "cursor" / "workspaceStorage" / ("r" * 32)
    _write_workspace(restore_ws, [])
    _write_ws_json(restore_ws, restore_path)
    assert paths.get_project_identifier(restore_path) != "original-project"
    assert paths.get_project_identifier(restore_path) != "foo"

    result = pull.run_multi_target_pull(
        [
            (
                {
                    "path": restore_path,
                    "workspace_dir": restore_ws,
                    "host": None,
                    "type": "local",
                },
                [chosen_path],
            )
        ],
        force=True,
    )
    assert result.imported == 1
    assert [i.composer_id for i in result.plan.items] == [CID_A]
    assert _local_composer(sync_env["global_db"], CID_A)["name"] == "Chosen"
    assert _local_composer(sync_env["global_db"], CID_B) is None


def test_same_count_same_semantics_is_up_to_date(sync_env):
    snap = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A)
    _commit_env(sync_env, [snap], [snap])
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build_for_project("project")
        plan = syncstate.build_pull_plan(session, index, _ws(sync_env))
    assert plan.items[0].relation == syncstate.PullRelation.UP_TO_DATE
    assert plan.items[0].action == syncstate.PullAction.SKIP


def test_same_count_different_content_is_diverged_skip(sync_env):
    remote = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A)
    local = _conversation([_msg(1, "A"), _msg(2, "X")], composer_id=CID_A)
    _commit_real(sync_env, [local], [remote])
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build_for_project("project")
        plan = syncstate.build_pull_plan(session, index, _ws(sync_env))
    assert plan.items[0].relation == syncstate.PullRelation.DIVERGED
    assert plan.items[0].action == syncstate.PullAction.SKIP


def test_local_longer_but_not_prefix_is_diverged(sync_env):
    remote = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A)
    local = _conversation(
        [_msg(1, "A"), _msg(2, "X"), _msg(3, "C")], composer_id=CID_A
    )
    _commit_real(sync_env, [local], [remote])
    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build_for_project("project")
        plan = syncstate.build_pull_plan(session, index, _ws(sync_env))
    assert plan.items[0].relation == syncstate.PullRelation.DIVERGED
    assert plan.import_candidates == []


def test_twenty_behind_one_workspace_reuses_backups(sync_env):
    locals_ = [
        _conversation([_msg(1, f"a{i}")], composer_id=_cid(i), name=f"C{i}")
        for i in range(20)
    ]
    remotes = [
        _conversation(
            [_msg(1, f"a{i}"), _msg(2, f"b{i}")], composer_id=_cid(i), name=f"C{i}"
        )
        for i in range(20)
    ]
    _commit_real(sync_env, locals_, remotes)
    syncstate.reset_op_counts()
    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    counts = syncstate.op_counts()
    assert result.imported == 20
    assert counts.safety_global_backups == 1
    assert counts.safety_workspace_backups == 1
    assert counts.write_connections_opened == 2
    assert counts.imports_completed == 20


def test_twenty_behind_three_workspaces_one_global_backup(sync_env):
    targets = []
    for i, name in enumerate(("alpha", "beta", "gamma")):
        ws_hash = f"{name[0] * 32}"
        ws_dir = sync_env["tmp"] / "cursor" / "workspaceStorage" / ws_hash
        project_path = f"/home/user/{name}"
        n = 7 if i < 2 else 6
        locals_ = [
            _conversation(
                [_msg(1, f"{name}{j}")],
                composer_id=_cid(100 + i * 10 + j),
                name=f"{name}{j}",
                sourceProjectPath=project_path,
                projectIdentifier=name,
            )
            for j in range(n)
        ]
        remotes = [
            _conversation(
                [_msg(1, f"{name}{j}"), _msg(2, "more")],
                composer_id=_cid(100 + i * 10 + j),
                name=f"{name}{j}",
                sourceProjectPath=project_path,
                projectIdentifier=name,
            )
            for j in range(n)
        ]
        gconn = _init_db(sync_env["global_db"])
        for snap in locals_:
            _write_local(gconn, snap)
        gconn.commit()
        gconn.close()
        _write_workspace(ws_dir, locals_)
        _write_ws_json(ws_dir, project_path)
        for snap in remotes:
            _write_snapshot_file(sync_env["snaps"] / name, snap)
        targets.append((
            {
                "path": project_path,
                "workspace_dir": ws_dir,
                "host": None,
                "type": "local",
            },
            None,
        ))

    syncstate.reset_op_counts()
    result = pull.run_multi_target_pull(targets, force=True)
    counts = syncstate.op_counts()
    assert result.imported == 20
    assert counts.safety_global_backups == 1
    assert counts.safety_workspace_backups == 3
    assert counts.write_connections_opened == 4


def test_ssh_other_hosts_never_enter_pull_plan(sync_env):
    ssh_path = "/home/lju/nixos"
    mine = _conversation(
        [_msg(1, "ml1")],
        composer_id=CID_A,
        name="ML1",
        sourceHost="MindLoop1",
        sourceProjectPath=ssh_path,
        projectIdentifier="ssh-MindLoop1-home-lju-nixos",
    )
    other_b = _conversation(
        [_msg(1, "ml2")],
        composer_id=CID_A,
        name="ML2",
        sourceHost="MindLoop2",
        sourceProjectPath=ssh_path,
        projectIdentifier="ssh-MindLoop2-home-lju-nixos",
    )
    other_c = _conversation(
        [_msg(1, "p9")],
        composer_id=CID_A,
        name="P9",
        sourceHost="P9Em",
        sourceProjectPath=ssh_path,
        projectIdentifier="ssh-P9Em-home-lju-nixos",
    )
    _write_workspace(sync_env["ws_dir"], [])
    _write_snapshot_file(sync_env["snaps"] / "ssh-MindLoop1-home-lju-nixos", mine)
    _write_snapshot_file(sync_env["snaps"] / "ssh-MindLoop2-home-lju-nixos", other_b)
    _write_snapshot_file(sync_env["snaps"] / "ssh-P9Em-home-lju-nixos", other_c)
    _init_db(sync_env["global_db"]).close()

    index = syncstate.SnapshotIndex.build_for_project("ssh-MindLoop1-home-lju-nixos")
    assert list(index.by_key) == [("ssh-MindLoop1-home-lju-nixos", CID_A)]
    workspace = {
        "path": ssh_path,
        "workspace_dir": sync_env["ws_dir"],
        "host": "MindLoop1",
        "type": "ssh",
    }
    with syncstate.SyncReadSession() as session:
        plan = syncstate.build_pull_plan(session, index, workspace)
    assert [i.composer_id for i in plan.items] == [CID_A]
    assert plan.items[0].meta.get("sourceHost") == "MindLoop1"


def test_remote_code_workspace_identity(sync_env):
    ws_file = "/home/user/app/app.code-workspace"
    mine = _conversation(
        [_msg(1, "ws")],
        composer_id=CID_A,
        name="WS",
        sourceHost=HOST_A,
        sourceProjectPath=ws_file,
        projectIdentifier="ssh-host-a-home-user-app-app.code-workspace",
    )
    other = _conversation(
        [_msg(1, "other")],
        composer_id=CID_B,
        name="Other",
        sourceHost=HOST_B,
        sourceProjectPath=ws_file,
        projectIdentifier="ssh-host-b-home-user-app-app.code-workspace",
    )
    _write_workspace(sync_env["ws_dir"], [])
    _write_snapshot_file(
        sync_env["snaps"] / "ssh-host-a-home-user-app-app.code-workspace", mine
    )
    _write_snapshot_file(
        sync_env["snaps"] / "ssh-host-b-home-user-app-app.code-workspace", other
    )
    _init_db(sync_env["global_db"]).close()
    workspace = {
        "path": ws_file,
        "workspace_dir": sync_env["ws_dir"],
        "host": HOST_A,
        "type": "ssh",
    }
    index = pull.scoped_snapshot_index(ws_file, HOST_A)
    with syncstate.SyncReadSession() as session:
        plan = syncstate.build_pull_plan(session, index, workspace)
    assert [i.composer_id for i in plan.items] == [CID_A]
    assert plan.items[0].meta.get("sourceProjectPath") == ws_file


def test_zero_candidates_cursor_running_no_warning(sync_env, monkeypatch, capsys):
    snap = _conversation([_msg(1, "A")], composer_id=CID_A)
    _commit_env(sync_env, [snap], [snap])
    monkeypatch.setattr(pull, "is_cursor_running", lambda: True)
    locks = _count_write_locks(monkeypatch)
    syncstate.reset_op_counts()
    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=False,
    )
    out = capsys.readouterr().out
    assert result.imported == 0
    assert "WARNING" not in out
    assert syncstate.op_counts().cursor_running_checks == 0
    assert locks["n"] == 0


def test_diverged_does_not_block_safe_behind(sync_env, capsys):
    behind = [
        _conversation([_msg(1, f"b{i}")], composer_id=_cid(i), name=f"B{i}")
        for i in range(3)
    ]
    behind_r = [
        _conversation(
            [_msg(1, f"b{i}"), _msg(2, "x")], composer_id=_cid(i), name=f"B{i}"
        )
        for i in range(3)
    ]
    div_l = _conversation([_msg(1, "L")], composer_id=CID_A, name="Div")
    div_r = _conversation([_msg(1, "R")], composer_id=CID_A, name="Div")
    _commit_real(sync_env, behind + [div_l], behind_r + [div_r])
    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    out = capsys.readouterr().out
    assert result.imported == 3
    assert len(result.plan.diverged) == 1
    assert "1 diverged — skipped" in out
    assert _local_composer(sync_env["global_db"], CID_A)["fullConversationHeadersOnly"][0][
        "bubbleId"
    ]


def test_unknown_does_not_block_safe_candidates(sync_env):
    good_l = _conversation([_msg(1, "g")], composer_id=CID_A, name="Good")
    good_r = _conversation(
        [_msg(1, "g"), _msg(2, "more")], composer_id=CID_A, name="Good"
    )
    bad = _conversation([_msg(1, "bad")], composer_id=CID_B, name="Bad")
    _commit_real(sync_env, [good_l], [good_r, bad])
    bad_path = sync_env["project_dir"] / f"{CID_B}.json.gz"
    bad_path.write_bytes(b"not-gzip")
    meta = json.loads((sync_env["project_dir"] / f"{CID_B}.meta.json").read_text())
    meta.pop("semanticDigest", None)
    meta.pop("snapshotContentDigest", None)
    (sync_env["project_dir"] / f"{CID_B}.meta.json").write_text(json.dumps(meta))

    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    assert result.imported == 1
    assert len(result.plan.unknown) == 1
    assert result.plan.unknown[0].action == syncstate.PullAction.SKIP


def test_corrupt_candidate_unknown_before_write(sync_env, monkeypatch):
    bad = _conversation([_msg(1, "bad")], composer_id=CID_A, name="Bad")
    _write_workspace(sync_env["ws_dir"], [])
    path = _write_snapshot_file(sync_env["project_dir"], bad, with_digest=False)
    path.write_bytes(b"\x1f\x8bnot-valid")
    _init_db(sync_env["global_db"]).close()
    locks = _count_write_locks(monkeypatch)
    syncstate.reset_op_counts()
    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    assert result.imported == 0
    assert result.plan.items[0].relation == syncstate.PullRelation.UNKNOWN
    assert syncstate.op_counts().safety_global_backups == 0
    assert locks["n"] == 0


def test_toctou_imports_staged_not_mutated_original(sync_env):
    local = _conversation([_msg(1, "old")], composer_id=CID_A, name="T")
    remote = _conversation(
        [_msg(1, "old"), _msg(2, "classified")], composer_id=CID_A, name="T"
    )
    mutated = _conversation(
        [_msg(1, "old"), _msg(2, "MUTATED")], composer_id=CID_A, name="T"
    )
    _commit_real(sync_env, [local], [remote])
    staging = sync_env["tmp"] / "stage"
    staging.mkdir()
    with dblock.repo_lock():
        index = syncstate.SnapshotIndex.build_for_project("project")
        with syncstate.SyncReadSession() as session:
            plan = syncstate.build_pull_plan(session, index, _ws(sync_env))
            syncstate.stage_import_candidates(plan, staging)
    assert plan.import_candidates[0].staged_path is not None
    _write_snapshot_file(sync_env["project_dir"], mutated)
    read_paths: list[Path] = []
    orig = importer.read_snapshot_file

    def wrapped(path, meta=None):
        read_paths.append(Path(path))
        return orig(path, meta)

    importer.read_snapshot_file = wrapped
    try:
        with importer.ImportSession() as batch:
            assert batch.import_snapshot(plan.import_candidates[0])
    finally:
        importer.read_snapshot_file = orig
        db.finish_cursor_writes()

    conn = sqlite3.connect(str(sync_env["global_db"]))
    bubble = json.loads(
        conn.execute(
            "SELECT value FROM cursorDiskKV WHERE key = ?",
            (f"bubbleId:{CID_A}:bubble-2",),
        ).fetchone()[0]
    )
    conn.close()
    assert bubble["text"] == "classified"
    staged = plan.import_candidates[0].staged_path
    assert staged in read_paths
    assert sync_env["project_dir"] / f"{CID_A}.json.gz" not in read_paths


def test_sharded_candidate_stages_all_parts(sync_env):
    local = _conversation([_msg(1, "a")], composer_id=CID_A, name="Sharded")
    remote = _conversation(
        [_msg(1, "a"), _msg(2, "b"), _msg(3, "c")], composer_id=CID_A, name="Sharded"
    )
    gconn = _init_db(sync_env["global_db"])
    _write_local(gconn, local)
    gconn.commit()
    gconn.close()
    _write_workspace(sync_env["ws_dir"], [local])
    _write_sharded_snapshot(sync_env["project_dir"], remote, parts=3, with_digest=True)

    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    assert result.imported == 1
    assert syncstate.op_counts().staged_snapshots == 1
    written = _local_composer(sync_env["global_db"], CID_A)
    assert len(written["fullConversationHeadersOnly"]) == 3


def test_nine_hundred_synced_warm_cache(sync_env):
    snaps = [
        _conversation([_msg(1, f"m{i}")], composer_id=_cid(i), name=f"C{i}")
        for i in range(900)
    ]
    _commit_env(sync_env, snaps, snaps)
    pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    syncstate.reset_op_counts()
    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    counts = syncstate.op_counts()
    assert result.imported == 0
    assert counts.imports_attempted == 0
    assert counts.safety_global_backups == 0
    assert counts.safety_workspace_backups == 0
    assert counts.write_connections_opened == 0
    assert counts.deep_snapshot_reads == 0
    assert counts.local_semantic_rehashes == 0
    assert counts.snapshot_content_hashes == 0
    assert counts.pull_target_scans == 1
    assert counts.snapshot_directory_scans == 1
    assert counts.sqlite_backups == 1


def test_nine_hundred_synced_plus_twelve_behind(sync_env):
    snaps = [
        _conversation([_msg(1, f"m{i}")], composer_id=_cid(i), name=f"C{i}")
        for i in range(900)
    ]
    _commit_env(sync_env, snaps, snaps)
    pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    for i in range(12):
        longer = _conversation(
            [_msg(1, f"m{i}"), _msg(2, "extra")],
            composer_id=_cid(i),
            name=f"C{i}",
        )
        _write_snapshot_file(sync_env["project_dir"], longer)
    syncstate.reset_op_counts()
    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    counts = syncstate.op_counts()
    assert result.imported == 12
    assert counts.imports_completed == 12
    assert counts.staged_snapshots == 12
    assert counts.deep_snapshot_reads == 12
    assert counts.local_semantic_rehashes == 12
    assert 12 <= counts.snapshot_content_hashes <= 36


def test_restore_all_imports_synced_and_ahead_exact_origin(sync_env):
    synced = _conversation([_msg(1, "s")], composer_id=CID_A, name="S")
    ahead_r = _conversation([_msg(1, "a")], composer_id=CID_B, name="A")
    ahead_l = _conversation(
        [_msg(1, "a"), _msg(2, "local")], composer_id=CID_B, name="A"
    )
    other_host = _conversation(
        [_msg(1, "ssh")],
        composer_id=CID_C,
        name="SSH",
        sourceHost=HOST_A,
        sourceProjectPath=PROJECT_PATH,
        projectIdentifier="ssh-host-a-home-user-project",
    )
    _commit_real(sync_env, [synced, ahead_l], [synced, ahead_r])
    _write_snapshot_file(sync_env["snaps"] / "ssh-host-a-home-user-project", other_host)

    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
        restore_all=True,
    )
    assert result.imported == 2
    assert {i.composer_id for i in result.plan.items} == {CID_A, CID_B}
    assert CID_C not in {i.composer_id for i in result.plan.items}


def test_restore_all_batch_performance(sync_env):
    snaps = [
        _conversation([_msg(1, f"r{i}")], composer_id=_cid(i), name=f"R{i}")
        for i in range(100)
    ]
    _commit_real(sync_env, snaps, snaps)
    syncstate.reset_op_counts()
    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
        restore_all=True,
    )
    counts = syncstate.op_counts()
    assert result.imported == 100
    assert counts.safety_global_backups == 1
    assert counts.safety_workspace_backups == 1
    assert counts.imports_completed == 100


def test_cli_pull_uses_incremental_entry(sync_env, monkeypatch):
    snap = _conversation([_msg(1, "A")], composer_id=CID_A)
    _commit_env(sync_env, [snap], [snap])
    _write_ws_json(sync_env["ws_dir"])
    _backend(monkeypatch, has_remote=True)
    monkeypatch.setattr(cli, "_require_sync_repo", lambda: sync_env["sync_dir"])
    captured = {}

    def fake_run(project_path, target_workspace_dir=None, source_host=None, **kwargs):
        captured["project_path"] = project_path
        captured["target_workspace_dir"] = target_workspace_dir
        captured["source_host"] = source_host
        captured["restore_all"] = kwargs.get("restore_all")
        return pull.PullResult(imported=0)

    monkeypatch.setattr(cli.pull, "run_workspace_pull", fake_run)
    cli.cmd_pull(
        argparse.Namespace(
            workspace=None,
            project=PROJECT_PATH,
            select=False,
            force=False,
            restore_all=False,
        )
    )
    assert captured["project_path"] == PROJECT_PATH
    assert captured["restore_all"] is False


def test_scoped_index_does_not_scan_other_projects(sync_env):
    a = _conversation([_msg(1, "a")], composer_id=CID_A, projectIdentifier="project")
    b = _conversation(
        [_msg(1, "b")],
        composer_id=CID_B,
        projectIdentifier="other",
    )
    _write_snapshot_file(sync_env["project_dir"], a)
    _write_snapshot_file(sync_env["snaps"] / "other", b)
    syncstate.reset_op_counts()
    index = syncstate.SnapshotIndex.build_for_project("project")
    assert ("project", CID_A) in index.by_key
    assert ("other", CID_B) not in index.by_key
    assert syncstate.op_counts().pull_target_scans == 1
    assert syncstate.op_counts().snapshot_directory_scans == 1


def test_import_setup_failure_releases_write_lock(sync_env, monkeypatch):
    local = _conversation([_msg(1, "a")], composer_id=CID_A, name="Behind")
    remote = _conversation(
        [_msg(1, "a"), _msg(2, "b")], composer_id=CID_A, name="Behind"
    )
    _commit_real(sync_env, [local], [remote])

    def boom(self, ws_dir):
        raise RuntimeError("workspace setup failed")

    monkeypatch.setattr(importer.ImportSession, "_ensure_workspace", boom)
    result = pull.run_workspace_pull(
        PROJECT_PATH,
        target_workspace_dir=sync_env["ws_dir"],
        force=True,
    )
    assert result.imported == 0
    assert result.failed == 1
    assert not db.write_connections_open()
    assert dblock.is_write_lock_held() is False


def _plan_and_stage(sync_env, staging: Path) -> syncstate.PullPlan:
    staging.mkdir(parents=True, exist_ok=True)
    with dblock.repo_lock():
        index = syncstate.SnapshotIndex.build_for_project("project")
        with syncstate.SyncReadSession() as session:
            plan = syncstate.build_pull_plan(session, index, _ws(sync_env))
            syncstate.stage_import_candidates(plan, staging)
    return plan


def test_local_change_after_staging_is_not_overwritten(sync_env):
    local = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="T")
    remote = _conversation(
        [_msg(1, "A"), _msg(2, "B"), _msg(3, "C")], composer_id=CID_A, name="T"
    )
    changed = _conversation(
        [_msg(1, "A"), _msg(2, "B"), _msg(3, "X")], composer_id=CID_A, name="T"
    )
    _commit_real(sync_env, [local], [remote])
    plan = _plan_and_stage(sync_env, sync_env["tmp"] / "stage-local")
    assert plan.import_candidates
    assert plan.import_candidates[0].relation == syncstate.PullRelation.BEHIND

    gconn = _init_db(sync_env["global_db"])
    _write_local(gconn, changed)
    gconn.commit()
    gconn.close()

    try:
        with importer.ImportSession() as batch:
            assert batch.import_snapshot(plan.import_candidates[0]) is None
    finally:
        db.finish_cursor_writes()

    assert syncstate.op_counts().local_guard_skips == 1
    assert syncstate.op_counts().imports_completed == 0
    written = _local_composer(sync_env["global_db"], CID_A)
    assert len(written["fullConversationHeadersOnly"]) == 3
    conn = sqlite3.connect(str(sync_env["global_db"]))
    bubble = json.loads(
        conn.execute(
            "SELECT value FROM cursorDiskKV WHERE key = ?",
            (f"bubbleId:{CID_A}:bubble-3",),
        ).fetchone()[0]
    )
    conn.close()
    assert bubble["text"] == "X"


def test_missing_local_becomes_collision_before_write(sync_env):
    snap = _conversation([_msg(1, "from-snap")], composer_id=CID_A, name="Snap")
    inserted = _conversation([_msg(1, "from-other")], composer_id=CID_A, name="Other")
    _write_workspace(sync_env["ws_dir"], [])
    _write_snapshot_file(sync_env["project_dir"], snap, gzip_body=True)
    _init_db(sync_env["global_db"]).close()

    plan = _plan_and_stage(sync_env, sync_env["tmp"] / "stage-missing")
    assert plan.import_candidates
    assert plan.import_candidates[0].relation == syncstate.PullRelation.MISSING_LOCAL

    gconn = _init_db(sync_env["global_db"])
    _write_local(gconn, inserted)
    gconn.commit()
    gconn.close()

    try:
        with importer.ImportSession() as batch:
            assert batch.import_snapshot(plan.import_candidates[0]) is None
    finally:
        db.finish_cursor_writes()

    assert syncstate.op_counts().local_guard_skips == 1
    written = _local_composer(sync_env["global_db"], CID_A)
    assert written["name"] == "Other"


def test_target_membership_change_after_preflight_skips_import(sync_env):
    local = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="T")
    remote = _conversation(
        [_msg(1, "A"), _msg(2, "B"), _msg(3, "C")], composer_id=CID_A, name="T"
    )
    _commit_real(sync_env, [local], [remote])
    plan = _plan_and_stage(sync_env, sync_env["tmp"] / "stage-member")
    assert plan.import_candidates[0].local_guard.expect_in_target is True

    _write_workspace(sync_env["ws_dir"], [])

    try:
        with importer.ImportSession() as batch:
            assert batch.import_snapshot(plan.import_candidates[0]) is None
    finally:
        db.finish_cursor_writes()

    assert syncstate.op_counts().local_guard_skips == 1
    written = _local_composer(sync_env["global_db"], CID_A)
    assert len(written["fullConversationHeadersOnly"]) == 2


class _BoomConn:
    def execute(self, *_a, **_k):
        raise sqlite3.OperationalError("disk I/O error")


class _BoomDB:
    def _reader_conn(self):
        return _BoomConn()


def test_live_guard_helpers_propagate_read_errors():
    boom = _BoomDB()
    with pytest.raises(sqlite3.OperationalError):
        syncstate._live_row_fingerprint(boom, CID_A)
    with pytest.raises(sqlite3.OperationalError):
        syncstate._live_blob_fingerprint(boom, ["blob"])
    with pytest.raises(sqlite3.OperationalError):
        syncstate._live_target_has_composer(boom, boom, CID_A, Path("/tmp/ws"))


def test_missing_local_live_read_error_skips_import(sync_env, monkeypatch):
    snap = _conversation([_msg(1, "new")], composer_id=CID_A, name="New")
    _write_workspace(sync_env["ws_dir"], [])
    _write_snapshot_file(sync_env["project_dir"], snap, gzip_body=True)
    _init_db(sync_env["global_db"]).close()
    plan = _plan_and_stage(sync_env, sync_env["tmp"] / "stage-read-err")
    assert plan.import_candidates[0].relation == syncstate.PullRelation.MISSING_LOCAL

    def boom_fp(*_a, **_k):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(syncstate, "_live_row_fingerprint", boom_fp)
    try:
        with importer.ImportSession() as batch:
            assert batch.import_snapshot(plan.import_candidates[0]) is None
    finally:
        db.finish_cursor_writes()

    assert syncstate.op_counts().local_guard_skips == 1
    assert syncstate.op_counts().imports_completed == 0
    assert _local_composer(sync_env["global_db"], CID_A) is None


def test_target_membership_read_error_skips_import(sync_env, monkeypatch):
    local = _conversation([_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="T")
    remote = _conversation(
        [_msg(1, "A"), _msg(2, "B"), _msg(3, "C")], composer_id=CID_A, name="T"
    )
    _commit_real(sync_env, [local], [remote])
    plan = _plan_and_stage(sync_env, sync_env["tmp"] / "stage-member-err")
    assert plan.import_candidates

    def boom_mem(*_a, **_k):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(syncstate, "_live_target_has_composer", boom_mem)
    try:
        with importer.ImportSession() as batch:
            assert batch.import_snapshot(plan.import_candidates[0]) is None
    finally:
        db.finish_cursor_writes()

    assert syncstate.op_counts().local_guard_skips == 1
    assert syncstate.op_counts().imports_completed == 0
    written = _local_composer(sync_env["global_db"], CID_A)
    assert len(written["fullConversationHeadersOnly"]) == 2
