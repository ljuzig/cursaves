"""SSH Remote workspace identity and host-safe matching.

The project path of a Remote SSH workspace exists on the remote host,
not necessarily on the machine running cursaves. Therefore git -C
cannot reliably discover remote.origin.url locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cursor_saves import cli, export, importer, paths, watch


PROJECT_PATH = "/home/user/project"
OLD_PROJECT_PATH = "/srv/old/project"
SERVICE_PATH = "/opt/service"
WORKSPACE_FILE = "/home/user/app/app.code-workspace"
LOCAL_WORKSPACE_FILE = "/home/user/local/project.code-workspace"
HOST_A = "host-a"
HOST_B = "host-b"
HOST_C = "host-c"
UNKNOWN_HOST = "unknown-host"
REMOTE_HOST = "remote-host"
SSH_ID_A = "ssh-host-a-home-user-project"
SSH_ID_B = "ssh-host-b-home-user-project"
SSH_ID_C = "ssh-host-c-home-user-project"
SSH_ID_WORKSPACE = "ssh-host-a-home-user-app-app.code-workspace"
WS_HASH = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
COMPOSER_ID = "11111111-1111-1111-1111-111111111111"


FAKE_WORKSPACES = [
    {
        "type": "ssh",
        "host": HOST_A,
        "path": PROJECT_PATH,
        "workspace_dir": Path("/tmp/fake-ws/host-a"),
        "mtime": 3,
    },
    {
        "type": "ssh",
        "host": HOST_B,
        "path": PROJECT_PATH,
        "workspace_dir": Path("/tmp/fake-ws/host-b"),
        "mtime": 2,
    },
    {
        "type": "ssh",
        "host": HOST_C,
        "path": PROJECT_PATH,
        "workspace_dir": Path("/tmp/fake-ws/host-c"),
        "mtime": 1,
    },
    {
        "type": "local",
        "host": None,
        "path": PROJECT_PATH,
        "workspace_dir": Path("/tmp/fake-ws/local"),
        "mtime": 0,
    },
]


def _write_sidecar(
    directory: Path,
    composer_id: str,
    *,
    source_host,
    source_path=PROJECT_PATH,
    project_identifier="project",
):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{composer_id}.json.gz").write_bytes(b"\x1f\x8b")
    meta = {
        "composerId": composer_id,
        "name": composer_id,
        "messageCount": 1,
        "sourceHost": source_host,
        "sourceProjectPath": source_path,
        "projectIdentifier": project_identifier,
        "sourceMachine": "test-machine",
        "version": 3,
    }
    (directory / f"{composer_id}.meta.json").write_text(json.dumps(meta))
    return directory / f"{composer_id}.json.gz"


def _ssh_remote_uri(host: str, path: str, *, hex_host: bool = True) -> str:
    if hex_host:
        encoded = json.dumps({"hostName": host}, separators=(",", ":")).encode().hex()
        authority = f"ssh-remote%2B{encoded}"
    else:
        authority = f"ssh-remote+{host}"
    return f"vscode-remote://{authority}{path}"


def _write_workspace_json(storage: Path, ws_hash: str, payload: dict) -> Path:
    ws_dir = storage / ws_hash
    ws_dir.mkdir(parents=True)
    (ws_dir / "workspace.json").write_text(json.dumps(payload))
    return ws_dir


# ── Identity ──────────────────────────────────────────────────────────


def test_ssh_identity_is_host_and_path():
    assert paths.get_project_identifier(PROJECT_PATH, source_host=HOST_A) == SSH_ID_A
    assert paths.get_project_identifier(PROJECT_PATH, source_host=HOST_B) == SSH_ID_B
    assert paths.get_project_identifier(PROJECT_PATH, source_host=HOST_C) == SSH_ID_C
    assert paths.get_project_identifier(
        PROJECT_PATH, source_host=HOST_A
    ) != paths.get_project_identifier(PROJECT_PATH, source_host=HOST_B)


def test_workspace_helper_uses_host():
    assert paths.get_workspace_project_identifier(FAKE_WORKSPACES[0]) == SSH_ID_A
    assert paths.get_workspace_project_identifier(FAKE_WORKSPACES[2]) == SSH_ID_C


def test_ssh_identity_normalizes_equivalent_paths():
    assert (
        paths.get_project_identifier("/home/user/project/", source_host=HOST_A)
        == SSH_ID_A
    )
    assert (
        paths.get_project_identifier("/home/user/foo/../project", source_host=HOST_A)
        == SSH_ID_A
    )


def test_local_git_identity_unchanged(monkeypatch):
    monkeypatch.setattr(
        paths, "_get_git_remote_url", lambda _path: "git@github.com:user/repo.git"
    )
    assert paths.get_project_identifier("/any/local/path") == "github.com-user-repo"
    monkeypatch.setattr(
        paths,
        "_get_git_remote_url",
        lambda _path: "https://github.com/user/repo.git",
    )
    assert paths.get_project_identifier("/any/local/path") == "github.com-user-repo"


def test_local_non_git_uses_basename(monkeypatch):
    monkeypatch.setattr(paths, "_get_git_remote_url", lambda _path: None)
    assert paths.get_project_identifier("/home/foo/myproject") == "myproject"


# ── Host-aware matching ───────────────────────────────────────────────


def test_matching_with_source_host_is_exclusive(monkeypatch):
    monkeypatch.setattr(paths, "list_all_workspaces", lambda: list(FAKE_WORKSPACES))
    matches = paths.find_all_matching_workspaces(PROJECT_PATH, source_host=HOST_A)
    assert [ws["host"] for ws in matches] == [HOST_A]
    assert all(ws["type"] == "ssh" for ws in matches)


def test_matching_unknown_host_has_no_cross_host_fallback(monkeypatch):
    monkeypatch.setattr(paths, "list_all_workspaces", lambda: list(FAKE_WORKSPACES))
    matches = paths.find_all_matching_workspaces(
        PROJECT_PATH, source_host=UNKNOWN_HOST
    )
    assert matches == []


def test_matching_without_host_keeps_path_basename_behavior(monkeypatch):
    monkeypatch.setattr(paths, "list_all_workspaces", lambda: list(FAKE_WORKSPACES))
    matches = paths.find_all_matching_workspaces(PROJECT_PATH)
    hosts = {ws["host"] for ws in matches}
    assert hosts == {HOST_A, HOST_B, HOST_C, None}
    assert len(matches) == 4

    basename_matches = paths.find_all_matching_workspaces("/elsewhere/project")
    assert {ws["host"] for ws in basename_matches} == hosts


def test_ssh_matching_does_not_fallback_to_same_basename_on_same_host(monkeypatch):
    workspaces = list(FAKE_WORKSPACES) + [
        {
            "type": "ssh",
            "host": HOST_A,
            "path": OLD_PROJECT_PATH,
            "workspace_dir": Path("/tmp/fake-ws/host-a-old"),
            "mtime": 4,
        }
    ]
    monkeypatch.setattr(paths, "list_all_workspaces", lambda: workspaces)
    matches = paths.find_all_matching_workspaces(PROJECT_PATH, source_host=HOST_A)
    assert [ws["path"] for ws in matches] == [PROJECT_PATH]
    assert paths.find_all_matching_workspaces(
        "/elsewhere/project", source_host=HOST_A
    ) == []


def test_ssh_matching_normalizes_equivalent_paths(monkeypatch):
    ws = {
        "type": "ssh",
        "host": HOST_A,
        "path": "/home/user/project/",
        "workspace_dir": Path("/tmp/ws"),
        "mtime": 1,
    }
    monkeypatch.setattr(paths, "list_all_workspaces", lambda: [ws])

    assert paths.find_all_matching_workspaces(
        "/home/user/foo/../project",
        source_host=HOST_A,
    ) == [ws]


def test_group_snapshots_by_origin_normalizes_equivalent_paths(tmp_path):
    snapshots = tmp_path / "snapshots" / "project"
    trailing = _write_sidecar(
        snapshots, "trailing", source_host=HOST_A, source_path="/home/user/project/"
    )
    dotted = _write_sidecar(
        snapshots, "dotted", source_host=HOST_A, source_path="/home/user/foo/../project"
    )

    groups = importer.group_snapshots_by_origin([trailing, dotted])
    assert set(groups) == {(HOST_A, PROJECT_PATH)}
    assert groups[(HOST_A, PROJECT_PATH)] == [trailing, dotted]


def test_ssh_exact_duplicate_workspaces_newest_first(monkeypatch):
    older = {
        "type": "ssh",
        "host": HOST_A,
        "path": PROJECT_PATH,
        "workspace_dir": Path("/tmp/fake-ws/host-a-old"),
        "mtime": 10,
    }
    newer = {
        "type": "ssh",
        "host": HOST_A,
        "path": PROJECT_PATH,
        "workspace_dir": Path("/tmp/fake-ws/host-a-new"),
        "mtime": 99,
    }
    monkeypatch.setattr(paths, "list_all_workspaces", lambda: [older, newer])
    matches = paths.find_all_matching_workspaces(PROJECT_PATH, source_host=HOST_A)
    assert [ws["workspace_dir"] for ws in matches] == [
        newer["workspace_dir"],
        older["workspace_dir"],
    ]
    targets = importer.resolve_sync_import_targets(
        {
            "sourceHost": HOST_A,
            "sourceProjectPath": PROJECT_PATH,
            "composerId": "missing-locally",
        },
        registered_composer_ids={},
    )
    assert [ws["workspace_dir"] for ws in targets] == [newer["workspace_dir"]]


# ── Export ────────────────────────────────────────────────────────────


class _FakeCdb:
    def get_json(self, key, table=None):
        if key.startswith("composerData:"):
            return {
                "name": "test chat",
                "fullConversationHeadersOnly": [],
            }
        return None

    def list_keys(self, prefix, table=None):
        return []

    def get_disk_kv(self, key):
        return None

    def get_item_binary(self, key, table=None):
        return None

    def close(self):
        pass


def test_export_conversation_uses_ssh_project_identifier(monkeypatch):
    monkeypatch.setattr(paths, "get_global_db_path", lambda: Path("/tmp/no-cursor-db"))
    monkeypatch.setattr(paths, "get_machine_id", lambda: "test-machine")
    monkeypatch.setattr(export, "get_transcript", lambda *_args, **_kwargs: None)

    snapshot = export.export_conversation(
        PROJECT_PATH,
        COMPOSER_ID,
        _cdb=_FakeCdb(),
        source_host=HOST_A,
    )
    assert snapshot is not None
    assert snapshot["sourceHost"] == HOST_A
    assert snapshot["sourceProjectPath"] == PROJECT_PATH
    assert snapshot["projectIdentifier"] == SSH_ID_A


# ── Snapshot lookup ───────────────────────────────────────────────────


def test_pull_w_prefers_ssh_identity_directory(tmp_path):
    snapshots = tmp_path / "snapshots"
    ssh_dir = snapshots / SSH_ID_A
    legacy = snapshots / "project"
    _write_sidecar(
        ssh_dir,
        "new-id",
        source_host=HOST_A,
        project_identifier=SSH_ID_A,
    )
    _write_sidecar(legacy, "legacy-id", source_host=HOST_A)

    found = importer.find_snapshot_dir_for_project(
        PROJECT_PATH,
        snapshots_dir=snapshots,
        source_host=HOST_A,
    )
    assert found == ssh_dir


def test_pull_w_legacy_bucket_is_not_accepted_indiscriminately(tmp_path):
    snapshots = tmp_path / "snapshots"
    legacy = snapshots / "project"
    from_a = _write_sidecar(legacy, "from-a", source_host=HOST_A)
    _write_sidecar(legacy, "from-b", source_host=HOST_B)

    found = importer.find_snapshot_dir_for_project(
        PROJECT_PATH,
        snapshots_dir=snapshots,
        source_host=HOST_A,
    )
    assert found == legacy

    filtered = importer.filter_snapshots_by_origin(
        importer.list_snapshot_files(found),
        HOST_A,
        source_project_path=PROJECT_PATH,
    )
    assert [p.name for p in filtered] == [from_a.name]

    rejected = importer.find_snapshot_dir_for_project(
        PROJECT_PATH,
        snapshots_dir=snapshots,
        source_host=UNKNOWN_HOST,
    )
    assert rejected is None


def test_legacy_bucket_filters_by_host_and_path(tmp_path):
    snapshots = tmp_path / "snapshots"
    legacy = snapshots / "project"
    current = _write_sidecar(
        legacy, "current", source_host=HOST_A, source_path=PROJECT_PATH
    )
    _write_sidecar(
        legacy, "old", source_host=HOST_A, source_path=OLD_PROJECT_PATH
    )

    found = importer.find_snapshot_dir_for_project(
        PROJECT_PATH,
        snapshots_dir=snapshots,
        source_host=HOST_A,
    )
    assert found == legacy

    filtered = importer.filter_snapshots_by_origin(
        importer.list_snapshot_files(found),
        HOST_A,
        source_project_path=PROJECT_PATH,
    )
    assert [p.name for p in filtered] == [current.name]


def test_pull_w_scans_oddly_named_bucket_by_origin(tmp_path):
    snapshots = tmp_path / "snapshots"
    odd = snapshots / "github.com-someone-unrelated"
    _write_sidecar(
        odd,
        "relocated",
        source_host=HOST_A,
        source_path=PROJECT_PATH,
        project_identifier="github.com-someone-unrelated",
    )
    _write_sidecar(
        snapshots / "other-host-same-path",
        "other-host",
        source_host=HOST_B,
        source_path=PROJECT_PATH,
        project_identifier="other-host-same-path",
    )
    _write_sidecar(
        snapshots / "same-host-other-path",
        "other-path",
        source_host=HOST_A,
        source_path=OLD_PROJECT_PATH,
        project_identifier="same-host-other-path",
    )

    found = importer.find_snapshot_dir_for_project(
        PROJECT_PATH,
        snapshots_dir=snapshots,
        source_host=HOST_A,
    )
    assert found == odd

    filtered = importer.filter_snapshots_by_origin(
        importer.list_snapshot_files(found),
        HOST_A,
        source_project_path=PROJECT_PATH,
    )
    assert [p.name for p in filtered] == ["relocated.json.gz"]


def test_path_only_pull_never_matches_ssh_snapshot_bucket(tmp_path):
    snapshots = tmp_path / "snapshots"
    ssh_dir = snapshots / "ssh-remote-host-opt-service"
    _write_sidecar(
        ssh_dir,
        "remote-chat",
        source_host=REMOTE_HOST,
        source_path=SERVICE_PATH,
        project_identifier="ssh-remote-host-opt-service",
    )

    found = importer.find_snapshot_dir_for_project(
        SERVICE_PATH,
        snapshots_dir=snapshots,
        source_host=None,
    )
    assert found is None


def test_path_only_pull_rejects_legacy_ssh_basename_bucket(tmp_path):
    snapshots = tmp_path / "snapshots"
    legacy = snapshots / "service"
    _write_sidecar(
        legacy,
        "remote-chat",
        source_host=REMOTE_HOST,
        source_path=SERVICE_PATH,
        project_identifier="service",
    )

    assert importer.find_snapshot_dir_for_project(
        SERVICE_PATH,
        snapshots_dir=snapshots,
        source_host=None,
    ) is None


def test_path_only_pull_rejects_mixed_local_and_ssh_bucket(tmp_path):
    snapshots = tmp_path / "snapshots"
    legacy = snapshots / "service"
    _write_sidecar(
        legacy,
        "local-chat",
        source_host=None,
        source_path=SERVICE_PATH,
        project_identifier="service",
    )
    _write_sidecar(
        legacy,
        "remote-chat",
        source_host=REMOTE_HOST,
        source_path=SERVICE_PATH,
        project_identifier="service",
    )

    assert importer.find_snapshot_dir_for_project(
        SERVICE_PATH,
        snapshots_dir=snapshots,
        source_host=None,
    ) is None


def test_path_only_pull_accepts_local_only_basename_bucket(tmp_path):
    snapshots = tmp_path / "snapshots"
    local_dir = snapshots / "service"
    _write_sidecar(
        local_dir,
        "local-chat",
        source_host=None,
        source_path=SERVICE_PATH,
        project_identifier="service",
    )

    found = importer.find_snapshot_dir_for_project(
        SERVICE_PATH,
        snapshots_dir=snapshots,
        source_host=None,
    )
    assert found == local_dir


# ── pull -s / sync safety ─────────────────────────────────────────────


def test_host_a_snapshot_never_matches_other_hosts(monkeypatch):
    monkeypatch.setattr(paths, "list_all_workspaces", lambda: list(FAKE_WORKSPACES))
    meta = {
        "sourceHost": HOST_A,
        "sourceProjectPath": PROJECT_PATH,
        "composerId": "abc",
    }
    matches = importer.workspaces_for_snapshot_meta(meta)
    assert [ws["host"] for ws in matches] == [HOST_A]

    targets = importer.resolve_sync_import_targets(
        meta, registered_composer_ids={}
    )
    assert len(targets) == 1
    assert targets[0]["host"] == HOST_A


def test_pull_s_groups_selected_chats_by_origin(tmp_path):
    snapshots = tmp_path / "snapshots" / "project"
    chat_a = _write_sidecar(snapshots, "chat-a", source_host=HOST_A)
    chat_b = _write_sidecar(snapshots, "chat-b", source_host=HOST_B)

    groups = importer.group_snapshots_by_origin([chat_a, chat_b])
    assert set(groups) == {
        (HOST_A, PROJECT_PATH),
        (HOST_B, PROJECT_PATH),
    }
    assert groups[(HOST_A, PROJECT_PATH)] == [chat_a]
    assert groups[(HOST_B, PROJECT_PATH)] == [chat_b]


def test_pull_s_ssh_without_workspace_is_fail_closed(monkeypatch):
    monkeypatch.setattr(paths, "list_all_workspaces", lambda: list(FAKE_WORKSPACES))

    no_match = paths.find_all_matching_workspaces(
        PROJECT_PATH, source_host=UNKNOWN_HOST
    )
    assert no_match == []
    action, targets = importer.pull_select_import_plan(UNKNOWN_HOST, no_match)
    assert action == "skip"
    assert targets is None

    action, targets = importer.pull_select_import_plan(HOST_A, [])
    assert action == "skip"
    assert targets is None

    action, targets = importer.pull_select_import_plan(None, [])
    assert action == "fallback"
    assert targets is None

    matches = paths.find_all_matching_workspaces(PROJECT_PATH, source_host=HOST_A)
    action, targets = importer.pull_select_import_plan(HOST_A, matches)
    assert action == "import"
    assert [ws["host"] for ws in targets] == [HOST_A]


def test_pull_s_hostless_never_targets_ssh():
    targets = [
        {"type": "ssh", "host": HOST_A},
        {"type": "local", "host": None},
        {"type": "ssh", "host": HOST_B},
    ]
    action, selected = importer.pull_select_import_plan(None, targets)
    assert action == "import"
    assert selected == [{"type": "local", "host": None}]


def test_pull_s_hostless_with_only_ssh_matches_falls_back():
    action, selected = importer.pull_select_import_plan(
        None,
        [{"type": "ssh", "host": HOST_A}],
    )
    assert action == "fallback"
    assert selected is None


def test_sync_does_not_pick_first_workspace_of_another_host(monkeypatch):
    monkeypatch.setattr(paths, "list_all_workspaces", lambda: list(FAKE_WORKSPACES))
    meta = {
        "sourceHost": HOST_C,
        "sourceProjectPath": PROJECT_PATH,
        "composerId": "missing-locally",
    }
    targets = importer.resolve_sync_import_targets(
        meta, registered_composer_ids={}
    )
    assert [ws["host"] for ws in targets] == [HOST_C]


def test_sync_hostless_snapshot_never_targets_ssh(monkeypatch):
    monkeypatch.setattr(paths, "list_all_workspaces", lambda: list(FAKE_WORKSPACES))
    meta = {
        "sourceHost": None,
        "sourceProjectPath": PROJECT_PATH,
        "composerId": "hostless",
    }
    assert importer.workspaces_for_snapshot_meta(meta)
    targets = importer.resolve_sync_import_targets(
        meta, registered_composer_ids={}
    )
    assert targets
    assert all(ws.get("type") != "ssh" for ws in targets)
    assert [ws["type"] for ws in targets] == ["local"]


# ── watch / import --file ─────────────────────────────────────────────


def test_watch_workspace_preserves_host_and_workspace_dir(monkeypatch):
    ws_dir = Path("/tmp/fake-ws/host-a")
    captured = {}

    def fake_checkpoint(project_path, composer_ids=None, workspace_dir=None, source_host=None):
        captured["project_path"] = project_path
        captured["workspace_dir"] = workspace_dir
        captured["source_host"] = source_host
        return []

    monkeypatch.setattr(export, "checkpoint_project", fake_checkpoint)
    watch.checkpoint_watched_project(
        PROJECT_PATH,
        workspace_dir=ws_dir,
        source_host=HOST_A,
    )
    assert captured == {
        "project_path": PROJECT_PATH,
        "workspace_dir": ws_dir,
        "source_host": HOST_A,
    }

    loop_args = {}

    def fake_watch_loop(**kwargs):
        loop_args.update(kwargs)

    monkeypatch.setattr(cli.paths, "resolve_workspace", lambda _sel: {
        "path": PROJECT_PATH,
        "workspace_dir": ws_dir,
        "host": HOST_A,
    })
    monkeypatch.setattr(cli, "watch_loop", fake_watch_loop)
    cli.cmd_watch(argparse.Namespace(
        workspace=HOST_A,
        project=None,
        interval=60,
        no_git=True,
        verbose=False,
    ))
    assert loop_args["project_path"] == PROJECT_PATH
    assert loop_args["workspace_dir"] == ws_dir
    assert loop_args["source_host"] == HOST_A


def test_watch_fingerprint_uses_explicit_workspace_dir(tmp_path, monkeypatch):
    chosen = tmp_path / "host-a"
    other = tmp_path / "host-b"
    chosen.mkdir()
    other.mkdir()
    (chosen / "state.vscdb").write_bytes(b"host-a")
    (other / "state.vscdb").write_bytes(b"host-b")

    monkeypatch.setattr(paths, "get_global_db_path", lambda: tmp_path / "missing-global.vscdb")

    def boom(_project_path):
        raise AssertionError("path-only workspace lookup must not be used when -w is set")

    monkeypatch.setattr(paths, "find_workspace_dirs_for_project", boom)
    fingerprint = watch._get_db_fingerprint(PROJECT_PATH, workspace_dir=chosen)
    assert fingerprint is not None


def test_push_w_and_pull_w_agree_on_ssh_identity(tmp_path, monkeypatch):
    ws_dir = Path("/tmp/fake-ws/host-a")
    captured_push = {}
    captured_pull = {}

    class _NoRemote:
        def has_remote(self):
            return False

        def pull(self, _snapshots_dir):
            return True

        def push(self, _snapshots_dir):
            return True

    def fake_checkpoint(project_path, composer_ids=None, workspace_dir=None, source_host=None):
        captured_push["project_path"] = project_path
        captured_push["workspace_dir"] = workspace_dir
        captured_push["source_host"] = source_host
        captured_push["project_id"] = paths.get_project_identifier(
            project_path, source_host=source_host
        )
        return [tmp_path / "fake.json.gz"]

    def fake_run_pull(
        project_path,
        target_workspace_dir=None,
        source_host=None,
        **_kwargs,
    ):
        captured_pull["project_path"] = project_path
        captured_pull["target_workspace_dir"] = target_workspace_dir
        captured_pull["source_host"] = source_host
        captured_pull["project_id"] = paths.get_project_identifier(
            project_path, source_host=source_host
        )
        from cursor_saves.pull import PullResult
        return PullResult(imported=1)

    monkeypatch.setattr(cli, "_require_sync_repo", lambda: tmp_path)
    monkeypatch.setattr(cli, "get_backend", lambda: _NoRemote())
    monkeypatch.setattr(cli.paths, "get_snapshots_dir", lambda: tmp_path / "snapshots")
    monkeypatch.setattr(cli.paths, "resolve_workspace", lambda _sel: {
        "path": PROJECT_PATH,
        "workspace_dir": ws_dir,
        "host": HOST_A,
        "type": "ssh",
    })
    monkeypatch.setattr(export, "checkpoint_project", fake_checkpoint)
    monkeypatch.setattr(cli.pull, "run_workspace_pull", fake_run_pull)
    monkeypatch.setattr(cli, "_maybe_reload", lambda _args: None)

    cli.cmd_push(argparse.Namespace(
        workspace=HOST_A,
        project=None,
        select=False,
        all_chats=True,
        ahead=False,
    ))
    cli.cmd_pull(argparse.Namespace(
        workspace=HOST_A,
        project=None,
        select=False,
        force=True,
        restore_all=False,
    ))

    assert captured_push == {
        "project_path": PROJECT_PATH,
        "workspace_dir": ws_dir,
        "source_host": HOST_A,
        "project_id": SSH_ID_A,
    }
    assert captured_pull == {
        "project_path": PROJECT_PATH,
        "target_workspace_dir": ws_dir,
        "source_host": HOST_A,
        "project_id": SSH_ID_A,
    }
    assert captured_push["project_id"] == captured_pull["project_id"]


def test_import_file_workspace_preserves_exact_workspace(tmp_path, monkeypatch):
    ws_dir = Path("/tmp/fake-ws/host-a")
    snapshot = tmp_path / "chat.json.gz"
    snapshot.write_bytes(b"\x1f\x8b")
    (tmp_path / "chat.meta.json").write_text(json.dumps({
        "composerId": "abc",
        "sourceHost": HOST_A,
        "sourceProjectPath": PROJECT_PATH,
    }))

    captured = {}

    def fake_import(snapshot_path, project_path, target_workspace_dir=None, skip_backup=False):
        captured["project_path"] = project_path
        captured["target_workspace_dir"] = target_workspace_dir
        return True

    monkeypatch.setattr(cli.paths, "resolve_workspace", lambda _sel: {
        "path": PROJECT_PATH,
        "workspace_dir": ws_dir,
        "host": HOST_A,
        "type": "ssh",
    })
    monkeypatch.setattr(cli, "import_snapshot", fake_import)
    monkeypatch.setattr(cli, "_maybe_reload", lambda _args: None)
    cli.cmd_import(argparse.Namespace(
        workspace=HOST_A,
        project=None,
        all=False,
        file=str(snapshot),
        force=False,
    ))
    assert captured["project_path"] == PROJECT_PATH
    assert captured["target_workspace_dir"] == ws_dir


def test_import_file_rejects_cross_host_snapshot():
    assert importer.reject_cross_origin_import(HOST_A, HOST_B) == (
        f"ERROR: target workspace belongs to {HOST_A}, "
        f"but snapshot belongs to {HOST_B}"
    )
    assert importer.reject_cross_origin_import(
        HOST_A,
        HOST_A,
        target_project_path=PROJECT_PATH,
        snapshot_project_path=PROJECT_PATH,
    ) is None
    assert importer.reject_cross_origin_import(HOST_A, None) == (
        f"ERROR: target workspace belongs to {HOST_A}, "
        "but snapshot belongs to unknown host"
    )
    assert importer.reject_cross_origin_import(None, HOST_B) == (
        f"ERROR: snapshot belongs to SSH host {HOST_B}, "
        "but target is not an explicitly selected SSH workspace; "
        "use -w to select the target workspace"
    )
    assert importer.reject_cross_origin_import(None, None) is None
    assert importer.reject_cross_origin_import(
        HOST_A,
        HOST_A,
        target_project_path=PROJECT_PATH,
        snapshot_project_path=OLD_PROJECT_PATH,
    ) == (
        f"ERROR: target workspace is {HOST_A}:{PROJECT_PATH}, "
        f"but snapshot belongs to {HOST_A}:{OLD_PROJECT_PATH}"
    )


# ── delete -w ─────────────────────────────────────────────────────────


def test_delete_workspace_uses_ssh_identity(tmp_path, monkeypatch):
    snapshots = tmp_path / "snapshots"
    ssh_dir = snapshots / SSH_ID_A
    legacy = snapshots / "project"
    _write_sidecar(ssh_dir, "ssh-chat", source_host=HOST_A, project_identifier=SSH_ID_A)
    _write_sidecar(legacy, "legacy-chat", source_host=None)

    class _NoRemote:
        def has_remote(self):
            return False

        def pull(self, _snapshots_dir):
            return True

        def push(self, _snapshots_dir):
            return True

    monkeypatch.setattr(cli.paths, "get_snapshots_dir", lambda: snapshots)
    monkeypatch.setattr(cli.paths, "get_sync_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "get_backend", lambda: _NoRemote())
    monkeypatch.setattr(cli.paths, "resolve_workspace", lambda _sel: {
        "path": PROJECT_PATH,
        "workspace_dir": Path("/tmp/fake-ws/host-a"),
        "host": HOST_A,
        "type": "ssh",
    })

    cli.cmd_delete(argparse.Namespace(
        workspace=HOST_A,
        project=None,
        all=True,
        id=None,
        select=False,
        all_projects=False,
        yes=True,
    ))

    assert not ssh_dir.exists() or not importer.list_snapshot_files(ssh_dir)
    assert importer.list_snapshot_files(legacy)


# ── Remote .code-workspace ────────────────────────────────────────────


def test_list_all_workspaces_enumerates_remote_ssh_code_workspace(tmp_path, monkeypatch):
    storage = tmp_path / "workspaceStorage"
    ws_dir = _write_workspace_json(
        storage,
        WS_HASH,
        {"workspace": _ssh_remote_uri(HOST_A, WORKSPACE_FILE)},
    )
    _write_workspace_json(
        storage,
        "local-code-workspace",
        {"workspace": f"file://{LOCAL_WORKSPACE_FILE}"},
    )
    monkeypatch.setattr(paths, "get_workspace_storage_dir", lambda: storage)

    workspaces = paths.list_all_workspaces()
    by_dir = {ws["workspace_dir"]: ws for ws in workspaces}

    remote = by_dir[ws_dir]
    assert remote["type"] == "ssh"
    assert remote["host"] == HOST_A
    assert remote["path"] == WORKSPACE_FILE
    assert remote["workspace_dir"].name == WS_HASH
    assert paths.get_workspace_project_identifier(remote) == SSH_ID_WORKSPACE

    local = next(ws for ws in workspaces if ws["workspace_dir"].name == "local-code-workspace")
    assert local["type"] == "workspace"
    assert local["host"] is None
    assert local["path"] == LOCAL_WORKSPACE_FILE


def test_remote_ssh_code_workspace_matching_is_host_exact(tmp_path, monkeypatch):
    storage = tmp_path / "workspaceStorage"
    host_a_dir = _write_workspace_json(
        storage,
        WS_HASH,
        {"workspace": _ssh_remote_uri(HOST_A, WORKSPACE_FILE)},
    )
    _write_workspace_json(
        storage,
        "host-b-same-path",
        {"workspace": _ssh_remote_uri(HOST_B, WORKSPACE_FILE)},
    )
    monkeypatch.setattr(paths, "get_workspace_storage_dir", lambda: storage)

    matches = paths.find_all_matching_workspaces(
        WORKSPACE_FILE,
        source_host=HOST_A,
    )
    assert [ws["host"] for ws in matches] == [HOST_A]
    assert [ws["workspace_dir"] for ws in matches] == [host_a_dir]

    assert (
        paths.find_all_matching_workspaces(
            WORKSPACE_FILE,
            source_host=HOST_B,
        )[0]["host"]
        == HOST_B
    )
    assert paths.find_all_matching_workspaces(
        WORKSPACE_FILE,
        source_host=HOST_A,
    )[0]["host"] != HOST_B


def test_find_workspace_dirs_includes_remote_code_workspace(tmp_path, monkeypatch):
    storage = tmp_path / "workspaceStorage"
    ws_dir = _write_workspace_json(
        storage,
        WS_HASH,
        {"workspace": _ssh_remote_uri(HOST_A, WORKSPACE_FILE)},
    )
    monkeypatch.setattr(paths, "get_workspace_storage_dir", lambda: storage)
    assert paths.find_workspace_dirs_for_project(WORKSPACE_FILE) == [ws_dir]


def test_parse_ssh_remote_uri_accepts_only_ssh_remote():
    host, path = paths._parse_ssh_remote_uri(_ssh_remote_uri(HOST_A, WORKSPACE_FILE))
    assert host == HOST_A
    assert path == WORKSPACE_FILE
    assert paths._parse_ssh_remote_uri(
        "vscode-remote://dev-container+foo/workspace"
    ) is None
    host, path = paths._parse_ssh_remote_uri(
        f"vscode-remote://ssh-remote+{HOST_A}/home/user/my%20proj"
    )
    assert host == HOST_A
    assert path == "/home/user/my proj"
