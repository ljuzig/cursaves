"""Cross-platform detection of the Cursor desktop app."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Iterable, Optional


# Desktop Cursor executable / wrapper basenames (compared case-insensitively).
_LINUX_DESKTOP_BASENAMES = frozenset({
    "cursor",
    "cursor-bin",
    ".cursor-wrapped",
})

# Helpers and other cursaves tooling — not the desktop app.
_LINUX_EXCLUDED_BASENAMES = frozenset({
    "cursaves",
    "cursor-server",
    "cursor-agent",
})

_LINUX_EXCLUDED_PATH_COMPONENTS = frozenset({
    ".cursor-server",
    "cursor-server",
    ".cursor-agent",
    "cursor-agent",
})


def _proc_root() -> Path:
    override = os.environ.get("CURSAVES_PROC_ROOT")
    if override:
        return Path(override)
    return Path("/proc")


def _basename(path_or_name: str) -> str:
    if not path_or_name:
        return ""
    return Path(path_or_name.rstrip("/")).name.lower()


def _has_excluded_linux_component(path: str) -> bool:
    """True if *path* contains a known non-desktop Cursor component."""
    if not path:
        return False
    components = {part.lower() for part in Path(path).parts}
    return bool(components & _LINUX_EXCLUDED_PATH_COMPONENTS)


def macos_ps_line_is_cursor(line: str) -> bool:
    """Return True if a ``ps -axo args`` line is the macOS Cursor app.

    Matches the historical check: the main ``Cursor.app`` executable,
    excluding Helper / Frameworks (and therefore CursorUIViewService).
    """
    return (
        "Cursor.app/Contents/MacOS/Cursor" in line
        and "Helper" not in line
        and "Frameworks" not in line
    )


def _is_cursor_running_macos() -> bool:
    try:
        result = subprocess.run(
            ["ps", "-axo", "args"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False
        return any(macos_ps_line_is_cursor(line) for line in result.stdout.splitlines())
    except FileNotFoundError:
        return False


def linux_proc_is_desktop_cursor(
    comm: str,
    exe: str = "",
    argv: Optional[list[str]] = None,
) -> bool:
    """Pure check: is this Linux process the Cursor desktop app?"""
    argv = argv or []
    argv0 = argv[0] if argv else ""
    names = {
        comm.strip().lower(),
        _basename(exe),
        _basename(argv0),
    }
    names.discard("")

    if names & _LINUX_EXCLUDED_BASENAMES:
        return False

    if _has_excluded_linux_component(exe) or _has_excluded_linux_component(argv0):
        return False

    if names & _LINUX_DESKTOP_BASENAMES:
        return True

    for raw in (exe, argv0):
        base = _basename(raw)
        if base.endswith(".appimage") and base.startswith("cursor"):
            return True
    return False


def _read_proc_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None


def _read_proc_exe(pid_dir: Path) -> str:
    try:
        return os.readlink(str(pid_dir / "exe"))
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return ""


def _read_proc_argv(pid_dir: Path) -> list[str]:
    try:
        raw = (pid_dir / "cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return []
    if not raw:
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\x00") if part]


def iter_linux_processes(
    proc_root: Optional[Path] = None,
    self_pid: Optional[int] = None,
) -> Iterable[tuple[int, str, str, list[str]]]:
    """Yield ``(pid, comm, exe, argv)`` from a /proc tree. Never raises."""
    root = proc_root if proc_root is not None else _proc_root()
    skip = os.getpid() if self_pid is None else self_pid
    try:
        entries = list(root.iterdir())
    except (FileNotFoundError, PermissionError, OSError):
        return
    for entry in entries:
        name = entry.name
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == skip:
            continue
        comm_raw = _read_proc_text(entry / "comm")
        if comm_raw is None:
            continue
        comm = comm_raw.splitlines()[0] if comm_raw else ""
        exe = _read_proc_exe(entry)
        argv = _read_proc_argv(entry)
        yield pid, comm, exe, argv


def is_cursor_running_linux(
    proc_root: Optional[Path] = None,
    self_pid: Optional[int] = None,
) -> bool:
    """True if a Cursor desktop process is visible under /proc."""
    for _pid, comm, exe, argv in iter_linux_processes(proc_root, self_pid):
        if linux_proc_is_desktop_cursor(comm, exe, argv):
            return True
    return False


def is_cursor_running() -> bool:
    """Check if the main Cursor desktop app is running.

    * macOS: historical ``ps -axo args`` + ``Cursor.app/Contents/MacOS/Cursor``.
    * Linux: /proc comm + exe + argv, excluding cursaves / cursor-server /
      cursor-agent.
    * other platforms: conservative False (do not block writes).
    """
    system = platform.system()
    if system == "Darwin":
        return _is_cursor_running_macos()
    if system == "Linux":
        return is_cursor_running_linux()
    return False
