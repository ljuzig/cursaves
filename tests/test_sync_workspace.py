"""Targeted ``sync -w`` (v0.9.7). Global ``sync`` without ``-w`` is unchanged."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cursor_saves import cli, db, export, paths, pull, syncstate
from tests.test_syncstate import (
    CID_A,
    CID_B,
    CID_C,
    CID_D,
    PROJECT_PATH,
    WS_HASH,
    _backend,
    _conversation,
    _init_db,
    _msg,
    _write_local,
    _write_snapshot_file,
    _write_workspace,
)

pytest_plugins = ["tests.test_syncstate"]

PATH_B = "/home/user/other"
WS_HASH_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
SSH_PATH = "/home/lju/nixos"


def _args(*, workspace=None, force=False):
    return type("Args", (), {"workspace": workspace, "force": force})()


def _ws(ws_dir: Path, path: str, host: str | None = None) -> dict:
    return {
        "type": "ssh" if host else "local",
        "host": host,
        "path": path,
        "workspace_dir": ws_dir,
        "conversations": 0,
    }


def _use_workspaces(monkeypatch, workspaces: list[dict]) -> list[dict]:
    listed = list(workspaces)
    monkeypatch.setattr(paths, "list_workspaces_with_conversations", lambda: listed)
    return listed


def _add_locals(global_db: Path, snapshots: list[dict]) -> None:
    conn = _init_db(global_db)
    for snap in snapshots:
        _write_local(conn, snap)
    conn.commit()
    conn.close()


def _target(ws: dict) -> dict:
    return {
        "path": ws["path"],
        "workspace_dir": ws["workspace_dir"],
        "host": ws.get("host"),
        "type": ws.get("type") or ("ssh" if ws.get("host") else "local"),
    }


def _spy_writes(monkeypatch):
    imported: list[dict] = []
    saved: list[str] = []

    def fake_import(path, project_path, target_workspace_dir=None, **kwargs):
        imported.append(
            {
                "cid": Path(path).name.split(".")[0],
                "project_path": project_path,
                "workspace_dir": target_workspace_dir,
            }
        )
        return True

    def fake_save(snap, dest):
        saved.append(snap["composerId"])
        return Path(dest) / f"{snap['composerId']}.json.gz"

    monkeypatch.setattr(cli, "import_snapshot", fake_import)
    monkeypatch.setattr(export, "save_snapshot", fake_save)
    return imported, saved


def test_sync_parser_accepts_number_and_hash(monkeypatch):
    seen: list[str | None] = []
    monkeypatch.setattr(cli, "cmd_sync", lambda args: seen.append(args.workspace))
    monkeypatch.setattr(sys, "argv", ["cursaves", "sync", "-w", "3"])
    cli.main()
    monkeypatch.setattr(sys, "argv", ["cursaves", "sync", "-w", "dc0adfcf"])
    cli.main()
    monkeypatch.setattr(sys, "argv", ["cursaves", "sync"])
    cli.main()
    assert seen == ["3", "dc0adfcf", None]


def test_targeted_plan_classifies_only_selected_workspace(sync_env, monkeypatch):
    behind_remote = _conversation(
        [_msg(1, "A"), _msg(2, "B"), _msg(3, "C")],
        composer_id=CID_A,
        name="A-behind",
    )
    behind_local = _conversation(
        [_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="A-behind"
    )
    other_remote = _conversation(
        [_msg(1, "X"), _msg(2, "Y")],
        composer_id=CID_B,
        name="B-chat",
        sourceProjectPath=PATH_B,
        projectIdentifier="other",
    )
    other_local = _conversation(
        [_msg(1, "X")],
        composer_id=CID_B,
        name="B-chat",
        sourceProjectPath=PATH_B,
        projectIdentifier="other",
    )
    ws_b = sync_env["tmp"] / "cursor" / "workspaceStorage" / WS_HASH_B
    _add_locals(sync_env["global_db"], [behind_local, other_local])
    _write_workspace(sync_env["ws_dir"], [behind_local])
    _write_workspace(ws_b, [other_local])
    _write_snapshot_file(sync_env["project_dir"], behind_remote)
    _write_snapshot_file(sync_env["snaps"] / "other", other_remote)

    ws_a = _ws(sync_env["ws_dir"], PROJECT_PATH)
    _use_workspaces(monkeypatch, [ws_a, _ws(ws_b, PATH_B)])

    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build()
        targeted = syncstate.build_sync_plan(
            session, index, target_workspace=_target(ws_a)
        )
        global_plan = syncstate.build_sync_plan(session, index)

    assert {i.composer_id for i in targeted.items} == {CID_A}
    assert targeted.items[0].relation == syncstate.SyncRelation.BEHIND
    assert targeted.target_workspace["workspace_dir"] == sync_env["ws_dir"]
    assert {i.composer_id for i in global_plan.items} == {CID_A, CID_B}


def test_targeted_sync_imports_only_selected_behind(sync_env, monkeypatch):
    behind_remote = _conversation(
        [_msg(1, "A"), _msg(2, "B"), _msg(3, "C")],
        composer_id=CID_A,
        name="A-behind",
    )
    behind_local = _conversation(
        [_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="A-behind"
    )
    other_remote = _conversation(
        [_msg(1, "X"), _msg(2, "Y")],
        composer_id=CID_B,
        name="B-behind",
        sourceProjectPath=PATH_B,
        projectIdentifier="other",
    )
    other_local = _conversation(
        [_msg(1, "X")],
        composer_id=CID_B,
        name="B-behind",
        sourceProjectPath=PATH_B,
        projectIdentifier="other",
    )
    ws_b = sync_env["tmp"] / "cursor" / "workspaceStorage" / WS_HASH_B
    _add_locals(sync_env["global_db"], [behind_local, other_local])
    _write_workspace(sync_env["ws_dir"], [behind_local])
    _write_workspace(ws_b, [other_local])
    _write_snapshot_file(sync_env["project_dir"], behind_remote, gzip_body=True)
    _write_snapshot_file(sync_env["snaps"] / "other", other_remote, gzip_body=True)
    _use_workspaces(
        monkeypatch,
        [_ws(sync_env["ws_dir"], PROJECT_PATH), _ws(ws_b, PATH_B)],
    )
    imported, saved = _spy_writes(monkeypatch)
    _backend(monkeypatch)

    cli.cmd_sync(_args(workspace=WS_HASH[:8]))

    assert [row["cid"] for row in imported] == [CID_A]
    assert [row["workspace_dir"] for row in imported] == [sync_env["ws_dir"]]
    assert saved == []


def test_targeted_sync_pushes_only_selected_ahead(sync_env, monkeypatch):
    ahead_remote = _conversation([_msg(1, "A")], composer_id=CID_A, name="A-ahead")
    ahead_local = _conversation(
        [_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="A-ahead"
    )
    other_remote = _conversation(
        [_msg(1, "X")],
        composer_id=CID_B,
        name="B-ahead",
        sourceProjectPath=PATH_B,
        projectIdentifier="other",
    )
    other_local = _conversation(
        [_msg(1, "X"), _msg(2, "Y")],
        composer_id=CID_B,
        name="B-ahead",
        sourceProjectPath=PATH_B,
        projectIdentifier="other",
    )
    ws_b = sync_env["tmp"] / "cursor" / "workspaceStorage" / WS_HASH_B
    _add_locals(sync_env["global_db"], [ahead_local, other_local])
    _write_workspace(sync_env["ws_dir"], [ahead_local])
    _write_workspace(ws_b, [other_local])
    _write_snapshot_file(sync_env["project_dir"], ahead_remote, gzip_body=True)
    _write_snapshot_file(sync_env["snaps"] / "other", other_remote, gzip_body=True)
    _use_workspaces(
        monkeypatch,
        [_ws(sync_env["ws_dir"], PROJECT_PATH), _ws(ws_b, PATH_B)],
    )
    imported, saved = _spy_writes(monkeypatch)
    backend = _backend(monkeypatch)

    cli.cmd_sync(_args(workspace="1"))

    assert imported == []
    assert saved == [CID_A]
    assert backend.pushes == 1


def test_same_cid_in_other_workspace_untouched(sync_env, monkeypatch):
    """A snapshot CID that already lives in workspace B must not enter A."""
    remote_a = _conversation(
        [_msg(1, "from-a")], composer_id=CID_A, name="Shared", projectIdentifier="project"
    )
    owner_b = _conversation(
        [_msg(1, "owned-by-b")],
        composer_id=CID_A,
        name="Shared",
        sourceProjectPath=PATH_B,
        projectIdentifier="other",
    )
    ws_b = sync_env["tmp"] / "cursor" / "workspaceStorage" / WS_HASH_B
    _add_locals(sync_env["global_db"], [owner_b])
    _write_workspace(sync_env["ws_dir"], [])
    _write_workspace(ws_b, [owner_b])
    _write_snapshot_file(sync_env["project_dir"], remote_a, gzip_body=True)
    _use_workspaces(
        monkeypatch,
        [_ws(sync_env["ws_dir"], PROJECT_PATH), _ws(ws_b, PATH_B)],
    )
    imported, saved = _spy_writes(monkeypatch)
    monkeypatch.setattr(
        cli,
        "resolve_sync_import_targets",
        lambda meta: (_ for _ in ()).throw(
            AssertionError("targeted sync must not resolve another workspace")
        ),
    )
    _backend(monkeypatch)

    cli.cmd_sync(_args(workspace=WS_HASH))

    assert imported == []
    assert saved == []
    conn = _init_db(sync_env["global_db"])
    bubble = conn.execute(
        "SELECT value FROM cursorDiskKV WHERE key LIKE ?",
        (f"bubbleId:{CID_A}:%",),
    ).fetchone()
    conn.close()
    assert bubble is not None
    text = bubble[0].decode() if isinstance(bubble[0], bytes) else bubble[0]
    assert "owned-by-b" in text


def test_ssh_same_path_other_host_untouched(sync_env, monkeypatch):
    id_a = paths.get_project_identifier(SSH_PATH, source_host="MindLoop1")
    id_b = paths.get_project_identifier(SSH_PATH, source_host="MindLoop2")
    mine = _conversation(
        [_msg(1, "ml1"), _msg(2, "more")],
        composer_id=CID_A,
        name="ML1",
        sourceHost="MindLoop1",
        sourceProjectPath=SSH_PATH,
        projectIdentifier=id_a,
    )
    mine_local = _conversation(
        [_msg(1, "ml1")],
        composer_id=CID_A,
        name="ML1",
        sourceHost="MindLoop1",
        sourceProjectPath=SSH_PATH,
        projectIdentifier=id_a,
    )
    other = _conversation(
        [_msg(1, "ml2"), _msg(2, "div")],
        composer_id=CID_B,
        name="ML2",
        sourceHost="MindLoop2",
        sourceProjectPath=SSH_PATH,
        projectIdentifier=id_b,
    )
    other_local = _conversation(
        [_msg(1, "ml2"), _msg(2, "other")],
        composer_id=CID_B,
        name="ML2",
        sourceHost="MindLoop2",
        sourceProjectPath=SSH_PATH,
        projectIdentifier=id_b,
    )
    ws_b = sync_env["tmp"] / "cursor" / "workspaceStorage" / WS_HASH_B
    _add_locals(sync_env["global_db"], [mine_local, other_local])
    _write_workspace(sync_env["ws_dir"], [mine_local])
    _write_workspace(ws_b, [other_local])
    _write_snapshot_file(sync_env["snaps"] / id_a, mine, gzip_body=True)
    _write_snapshot_file(sync_env["snaps"] / id_b, other, gzip_body=True)
    ws_a = _ws(sync_env["ws_dir"], SSH_PATH, host="MindLoop1")
    _use_workspaces(monkeypatch, [ws_a, _ws(ws_b, SSH_PATH, host="MindLoop2")])
    imported, saved = _spy_writes(monkeypatch)
    _backend(monkeypatch)

    with syncstate.SyncReadSession() as session:
        index = pull.scoped_snapshot_index(SSH_PATH, "MindLoop1")
        plan = syncstate.build_sync_plan(
            session, index, target_workspace=_target(ws_a)
        )
    assert [i.composer_id for i in plan.items] == [CID_A]
    assert all(i.source_host == "MindLoop1" for i in plan.items)
    assert plan.unsafe is False

    cli.cmd_sync(_args(workspace=WS_HASH[:8]))
    assert [row["cid"] for row in imported] == [CID_A]
    assert [row["workspace_dir"] for row in imported] == [sync_env["ws_dir"]]
    assert saved == []


def test_targeted_sync_accepts_exact_legacy_ssh_bucket(sync_env, monkeypatch):
    """Legacy snapshots/nixos/ stays readable when host + path are exact."""
    canonical = paths.get_project_identifier(SSH_PATH, source_host="MindLoop1")
    assert canonical == "ssh-MindLoop1-home-lju-nixos"
    remote = _conversation(
        [_msg(1, "ml1"), _msg(2, "more")],
        composer_id=CID_A,
        name="Legacy-ML1",
        sourceHost="MindLoop1",
        sourceProjectPath=SSH_PATH,
        projectIdentifier="nixos",
    )
    local = _conversation(
        [_msg(1, "ml1")],
        composer_id=CID_A,
        name="Legacy-ML1",
        sourceHost="MindLoop1",
        sourceProjectPath=SSH_PATH,
        projectIdentifier="nixos",
    )
    other_host = _conversation(
        [_msg(1, "ml2"), _msg(2, "div")],
        composer_id=CID_B,
        name="Legacy-ML2",
        sourceHost="MindLoop2",
        sourceProjectPath=SSH_PATH,
        projectIdentifier="nixos",
    )
    _add_locals(sync_env["global_db"], [local])
    _write_workspace(sync_env["ws_dir"], [local])
    _write_snapshot_file(sync_env["snaps"] / "nixos", remote, gzip_body=True)
    _write_snapshot_file(sync_env["snaps"] / "nixos", other_host, gzip_body=True)
    assert not (sync_env["snaps"] / canonical).exists()
    ws_a = _ws(sync_env["ws_dir"], SSH_PATH, host="MindLoop1")
    _use_workspaces(monkeypatch, [ws_a])
    imported, saved = _spy_writes(monkeypatch)
    _backend(monkeypatch)

    index = pull.scoped_snapshot_index(SSH_PATH, "MindLoop1")
    assert index.scoped_project_identifier == "nixos"
    with syncstate.SyncReadSession() as session:
        plan = syncstate.build_sync_plan(
            session, index, target_workspace=_target(ws_a)
        )
    assert [i.composer_id for i in plan.items] == [CID_A]
    assert plan.items[0].relation == syncstate.SyncRelation.BEHIND
    assert plan.items[0].snapshot_path is not None
    assert plan.items[0].snapshot_path.parent.name == "nixos"
    assert CID_B not in {i.composer_id for i in plan.items}

    cli.cmd_sync(_args(workspace=WS_HASH[:8]))
    assert [row["cid"] for row in imported] == [CID_A]
    assert [row["workspace_dir"] for row in imported] == [sync_env["ws_dir"]]
    assert saved == []


def test_targeted_sync_ignores_other_host_in_legacy_ssh_bucket(sync_env, monkeypatch):
    """Same basename bucket, other SSH host: never classified or imported."""
    other = _conversation(
        [_msg(1, "ml2"), _msg(2, "div")],
        composer_id=CID_B,
        name="Legacy-ML2",
        sourceHost="MindLoop2",
        sourceProjectPath=SSH_PATH,
        projectIdentifier="nixos",
    )
    mine_local = _conversation(
        [_msg(1, "ml1")],
        composer_id=CID_A,
        name="ML1-local",
        sourceHost="MindLoop1",
        sourceProjectPath=SSH_PATH,
        projectIdentifier="nixos",
    )
    _add_locals(sync_env["global_db"], [mine_local])
    _write_workspace(sync_env["ws_dir"], [mine_local])
    _write_snapshot_file(sync_env["snaps"] / "nixos", other, gzip_body=True)
    ws_a = _ws(sync_env["ws_dir"], SSH_PATH, host="MindLoop1")
    _use_workspaces(monkeypatch, [ws_a])
    imported, saved = _spy_writes(monkeypatch)
    _backend(monkeypatch)

    index = pull.scoped_snapshot_index(SSH_PATH, "MindLoop1")
    # MindLoop2-only legacy dir is not this origin; scoped index is empty
    # or, if a matcher returned the dir, the host filter still drops it.
    with syncstate.SyncReadSession() as session:
        plan = syncstate.build_sync_plan(
            session, index, target_workspace=_target(ws_a)
        )
    assert CID_B not in {i.composer_id for i in plan.items}
    assert all(
        (i.meta.get("sourceHost") or i.source_host) != "MindLoop2"
        for i in plan.items
    )

    cli.cmd_sync(_args(workspace=WS_HASH[:8]))
    assert imported == []
    assert saved == []


def test_selected_divergence_aborts_targeted_sync(sync_env, monkeypatch):
    remote = _conversation(
        [_msg(1, "A"), _msg(2, "B")], composer_id=CID_A, name="Diverged"
    )
    local = _conversation(
        [_msg(1, "A"), _msg(2, "X")], composer_id=CID_A, name="Diverged"
    )
    _add_locals(sync_env["global_db"], [local])
    _write_workspace(sync_env["ws_dir"], [local])
    _write_snapshot_file(sync_env["project_dir"], remote, gzip_body=True)
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    imported, saved = _spy_writes(monkeypatch)
    backend = _backend(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        cli.cmd_sync(_args(workspace=WS_HASH[:8]))
    assert exc.value.code == 1
    assert imported == []
    assert saved == []
    assert backend.pushes == 0


def test_unselected_divergence_does_not_abort_targeted_sync(sync_env, monkeypatch):
    synced = _conversation([_msg(1, "ok")], composer_id=CID_A, name="A-synced")
    diverged_remote = _conversation(
        [_msg(1, "A"), _msg(2, "B")],
        composer_id=CID_D,
        name="B-div",
        sourceProjectPath=PATH_B,
        projectIdentifier="other",
    )
    diverged_local = _conversation(
        [_msg(1, "A"), _msg(2, "X")],
        composer_id=CID_D,
        name="B-div",
        sourceProjectPath=PATH_B,
        projectIdentifier="other",
    )
    ws_b = sync_env["tmp"] / "cursor" / "workspaceStorage" / WS_HASH_B
    _add_locals(sync_env["global_db"], [synced, diverged_local])
    _write_workspace(sync_env["ws_dir"], [synced])
    _write_workspace(ws_b, [diverged_local])
    _write_snapshot_file(sync_env["project_dir"], synced, gzip_body=True)
    _write_snapshot_file(sync_env["snaps"] / "other", diverged_remote, gzip_body=True)
    _use_workspaces(
        monkeypatch,
        [_ws(sync_env["ws_dir"], PROJECT_PATH), _ws(ws_b, PATH_B)],
    )
    imported, saved = _spy_writes(monkeypatch)
    backend = _backend(monkeypatch)

    cli.cmd_sync(_args(workspace=WS_HASH[:8]))
    assert imported == []
    assert saved == []
    assert backend.pushes == 0


def test_global_sync_still_aborts_on_unselected_divergence(sync_env, monkeypatch):
    synced = _conversation([_msg(1, "ok")], composer_id=CID_A, name="A-synced")
    diverged_remote = _conversation(
        [_msg(1, "A"), _msg(2, "B")],
        composer_id=CID_D,
        name="B-div",
        sourceProjectPath=PATH_B,
        projectIdentifier="other",
    )
    diverged_local = _conversation(
        [_msg(1, "A"), _msg(2, "X")],
        composer_id=CID_D,
        name="B-div",
        sourceProjectPath=PATH_B,
        projectIdentifier="other",
    )
    ws_b = sync_env["tmp"] / "cursor" / "workspaceStorage" / WS_HASH_B
    _add_locals(sync_env["global_db"], [synced, diverged_local])
    _write_workspace(sync_env["ws_dir"], [synced])
    _write_workspace(ws_b, [diverged_local])
    _write_snapshot_file(sync_env["project_dir"], synced, gzip_body=True)
    _write_snapshot_file(sync_env["snaps"] / "other", diverged_remote, gzip_body=True)
    _use_workspaces(
        monkeypatch,
        [_ws(sync_env["ws_dir"], PROJECT_PATH), _ws(ws_b, PATH_B)],
    )
    imported, saved = _spy_writes(monkeypatch)
    backend = _backend(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        cli.cmd_sync(_args())
    assert exc.value.code == 1
    assert imported == []
    assert saved == []
    assert backend.pushes == 0


def test_unknown_workspace_selector_exits(sync_env, monkeypatch, capsys):
    _use_workspaces(monkeypatch, [_ws(sync_env["ws_dir"], PROJECT_PATH)])
    _backend(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_sync(_args(workspace="no-such-workspace"))
    assert exc.value.code == 1
    assert "No workspace matching" in capsys.readouterr().err


def test_targeted_remote_only_binds_to_selected_workspace(sync_env, monkeypatch):
    remote = _conversation(
        [_msg(1, "new")], composer_id=CID_C, name="Remote-only"
    )
    _init_db(sync_env["global_db"]).close()
    _write_workspace(sync_env["ws_dir"], [])
    _write_snapshot_file(sync_env["project_dir"], remote, gzip_body=True)
    ws_a = _ws(sync_env["ws_dir"], PROJECT_PATH)
    _use_workspaces(monkeypatch, [ws_a])
    imported, _saved = _spy_writes(monkeypatch)
    monkeypatch.setattr(
        cli,
        "resolve_sync_import_targets",
        lambda meta: (_ for _ in ()).throw(
            AssertionError("targeted sync must not resolve another workspace")
        ),
    )
    _backend(monkeypatch)

    with syncstate.SyncReadSession() as session:
        index = syncstate.SnapshotIndex.build_for_project("project")
        plan = syncstate.build_sync_plan(
            session, index, target_workspace=_target(ws_a)
        )
    assert len(plan.behind) == 1
    assert plan.behind[0].workspace_dir == sync_env["ws_dir"]

    cli.cmd_sync(_args(workspace=WS_HASH[:8]))
    assert [row["cid"] for row in imported] == [CID_C]
    assert imported[0]["workspace_dir"] == sync_env["ws_dir"]
