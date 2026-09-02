"""Cross-platform Cursor desktop process detection."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from cursor_saves import importer, process


def _write_proc(
    proc_root: Path,
    pid: int,
    *,
    comm: str,
    exe: str | None,
    argv: list[str],
) -> Path:
    pid_dir = proc_root / str(pid)
    pid_dir.mkdir(parents=True, exist_ok=True)
    (pid_dir / "comm").write_text(comm + "\n")
    if exe:
        try:
            (pid_dir / "exe").symlink_to(exe)
        except FileExistsError:
            pass
    cmdline = b"\x00".join(a.encode() for a in argv)
    if argv:
        cmdline += b"\x00"
    (pid_dir / "cmdline").write_bytes(cmdline)
    return pid_dir


@pytest.fixture
def proc_root(tmp_path, monkeypatch):
    root = tmp_path / "proc"
    root.mkdir()
    monkeypatch.setenv("CURSAVES_PROC_ROOT", str(root))
    monkeypatch.setattr(process.platform, "system", lambda: "Linux")
    return root


def test_linux_desktop_cursor_capital_c(proc_root):
    _write_proc(
        proc_root,
        200,
        comm="Cursor",
        exe="/opt/Cursor/cursor",
        argv=["/opt/Cursor/cursor"],
    )
    assert process.is_cursor_running_linux(proc_root) is True
    assert process.is_cursor_running() is True


def test_linux_desktop_cursor_lowercase(proc_root):
    _write_proc(
        proc_root,
        201,
        comm="cursor",
        exe="/usr/bin/cursor",
        argv=["/usr/bin/cursor"],
    )
    assert process.is_cursor_running_linux(proc_root) is True


def test_linux_nixos_wrapper(proc_root, tmp_path):
    wrapped = tmp_path / "nix" / "store" / "hash-cursor-1.7.0" / "bin" / ".cursor-wrapped"
    wrapped.parent.mkdir(parents=True)
    wrapped.write_text("")
    _write_proc(
        proc_root,
        202,
        comm=".cursor-wrapped",
        exe=str(wrapped),
        argv=[str(wrapped)],
    )
    bin_cursor = tmp_path / "nix" / "store" / "hash-cursor-1.7.0" / "bin" / "cursor-bin"
    bin_cursor.write_text("")
    _write_proc(
        proc_root,
        203,
        comm="cursor-bin",
        exe=str(bin_cursor),
        argv=[str(bin_cursor)],
    )
    assert process.is_cursor_running_linux(proc_root) is True


def test_linux_appimage_is_desktop(proc_root):
    _write_proc(
        proc_root,
        204,
        comm="Cursor-1.7.0.AppImage",
        exe="/opt/Cursor-1.7.0.AppImage",
        argv=["/opt/Cursor-1.7.0.AppImage"],
    )
    assert process.is_cursor_running_linux(proc_root) is True


def test_linux_appimage_path_with_spaces():
    assert process.linux_proc_is_desktop_cursor(
        "something",
        "/home/user/My Apps/Cursor-1.7.0.AppImage",
        ["/home/user/My Apps/Cursor-1.7.0.AppImage"],
    )


def test_linux_cursor_under_cursaves_tools_dir_is_desktop():
    assert process.linux_proc_is_desktop_cursor(
        "cursor",
        "/home/user/cursaves-tools/bin/cursor",
        ["/home/user/cursaves-tools/bin/cursor"],
    )


def test_linux_shell_command_mentioning_cursor_is_not_desktop():
    assert not process.linux_proc_is_desktop_cursor(
        "bash",
        "/run/current-system/sw/bin/bash",
        ["bash", "-c", "/usr/bin/cursor --version"],
    )


def test_linux_cursaves_is_not_cursor(proc_root):
    _write_proc(
        proc_root,
        300,
        comm="cursaves",
        exe="/nix/store/hash-cursaves-0.9.1/bin/cursaves",
        argv=["cursaves", "pull"],
    )
    assert process.linux_proc_is_desktop_cursor(
        "cursaves", "/nix/store/hash-cursaves-0.9.1/bin/cursaves", ["cursaves"]
    ) is False
    assert process.is_cursor_running_linux(proc_root) is False


def test_linux_cursor_server_is_not_desktop(proc_root):
    _write_proc(
        proc_root,
        301,
        comm="cursor-server",
        exe="/home/user/.cursor-server/bin/cursor-server",
        argv=["cursor-server"],
    )
    assert process.is_cursor_running_linux(proc_root) is False
    assert not process.linux_proc_is_desktop_cursor(
        "cursor",
        "/home/user/.cursor-server/bin/cursor",
        ["/home/user/.cursor-server/bin/cursor"],
    )


def test_linux_cursor_agent_is_not_desktop(proc_root):
    _write_proc(
        proc_root,
        302,
        comm="cursor-agent",
        exe="/usr/bin/cursor-agent",
        argv=["cursor-agent"],
    )
    assert process.is_cursor_running_linux(proc_root) is False


def test_linux_vanished_pid_does_not_raise(proc_root):
    vanished = proc_root / "404"
    vanished.mkdir()
    _write_proc(proc_root, 1, comm="bash", exe="/bin/bash", argv=["bash"])
    assert process.is_cursor_running_linux(proc_root) is False


def test_linux_exe_permissionerror_does_not_raise(proc_root, monkeypatch):
    _write_proc(proc_root, 9, comm="bash", exe=None, argv=["bash"])
    (proc_root / "9" / "exe").write_text("not-a-symlink")

    def boom(path):
        raise PermissionError("denied")

    monkeypatch.setattr(os, "readlink", boom)
    assert process.is_cursor_running_linux(proc_root) is False


def test_linux_self_pid_is_ignored(proc_root):
    _write_proc(
        proc_root,
        os.getpid(),
        comm="cursor",
        exe="/usr/bin/cursor",
        argv=["/usr/bin/cursor"],
    )
    assert process.is_cursor_running_linux(proc_root, self_pid=os.getpid()) is False


def test_linux_no_cursor(proc_root):
    _write_proc(proc_root, 10, comm="bash", exe="/bin/bash", argv=["bash"])
    assert process.is_cursor_running() is False


def test_macos_main_app_detected(monkeypatch):
    monkeypatch.setattr(process.platform, "system", lambda: "Darwin")

    def fake_run(cmd, **kwargs):
        assert cmd == ["ps", "-axo", "args"]
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="/Applications/Cursor.app/Contents/MacOS/Cursor\n",
            stderr="",
        )

    monkeypatch.setattr(process.subprocess, "run", fake_run)
    assert process.is_cursor_running() is True


def test_macos_helper_ignored(monkeypatch):
    monkeypatch.setattr(process.platform, "system", lambda: "Darwin")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="/Applications/Cursor.app/Contents/MacOS/Cursor Helper\n",
            stderr="",
        )

    monkeypatch.setattr(process.subprocess, "run", fake_run)
    assert process.is_cursor_running() is False


def test_macos_no_cursor(monkeypatch):
    monkeypatch.setattr(process.platform, "system", lambda: "Darwin")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="cursaves pull\n/usr/bin/python\n", stderr=""
        )

    monkeypatch.setattr(process.subprocess, "run", fake_run)
    assert process.is_cursor_running() is False


def test_macos_ps_missing(monkeypatch):
    monkeypatch.setattr(process.platform, "system", lambda: "Darwin")

    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("ps")

    monkeypatch.setattr(process.subprocess, "run", fake_run)
    assert process.is_cursor_running() is False


def test_other_platform_is_conservative(monkeypatch):
    monkeypatch.setattr(process.platform, "system", lambda: "Windows")
    assert process.is_cursor_running() is False


def test_macos_line_helper_matches_historical_rule():
    assert process.macos_ps_line_is_cursor(
        "/Applications/Cursor.app/Contents/MacOS/Cursor"
    )
    assert not process.macos_ps_line_is_cursor(
        "/Applications/Cursor.app/Contents/MacOS/Cursor Helper"
    )
    assert not process.macos_ps_line_is_cursor(
        "/Applications/Cursor.app/Contents/Frameworks/Cursor Helper"
    )


def test_import_warns_when_linux_cursor_running(tmp_path, proc_root, capsys):
    _write_proc(
        proc_root,
        200,
        comm="cursor",
        exe="/usr/bin/cursor",
        argv=["/usr/bin/cursor"],
    )
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    imported, failed = importer.import_from_snapshot_dir(snaps, "/tmp/proj")
    assert (imported, failed) == (0, 0)
    err = capsys.readouterr().err
    assert "WARNING: Cursor is running" in err
    assert "--force" in err


def test_import_force_skips_running_check(tmp_path, proc_root, capsys):
    _write_proc(
        proc_root,
        200,
        comm="cursor",
        exe="/usr/bin/cursor",
        argv=["/usr/bin/cursor"],
    )
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    imported, failed = importer.import_from_snapshot_dir(
        snaps, "/tmp/proj", force=True
    )
    assert (imported, failed) == (0, 0)
    assert "WARNING: Cursor is running" not in capsys.readouterr().err


def test_importer_reexports_process_helper():
    assert importer.is_cursor_running is process.is_cursor_running
