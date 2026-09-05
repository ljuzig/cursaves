"""Keep every test off the real Cursor home and ~/.cursaves tree."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cursor_saves import backends, db, dblock, paths, syncstate

# Captured before any fixture rewrites HOME. Optional passwd home covers
# nix-shell, where $HOME is already a temp directory.
_REAL_HOME = Path.home()
_HOMES = {_REAL_HOME}
try:
    import pwd

    _HOMES.add(Path(pwd.getpwuid(os.getuid()).pw_dir))
except (ImportError, KeyError, OSError):
    pass

_FORBIDDEN_PREFIXES = tuple(
    home / rel
    for home in _HOMES
    for rel in (
        Path(".config") / "Cursor",
        Path(".cursaves"),
        Path("Library") / "Application Support" / "Cursor",
        Path(".cursor"),
        Path(".config") / "cursaves",
    )
)


def _is_forbidden(path: Path) -> bool:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path.expanduser()
    for banned in _FORBIDDEN_PREFIXES:
        try:
            banned_r = banned.resolve()
        except OSError:
            banned_r = banned
        if resolved == banned_r or banned_r in resolved.parents:
            return True
    return False


def _guard(path: Path) -> Path:
    if _is_forbidden(path):
        raise RuntimeError(
            f"test attempted to use real user data path: {path}"
        )
    return path


@pytest.fixture(autouse=True)
def isolate_from_real_user_data(tmp_path_factory, monkeypatch):
    """Redirect HOME, Cursor dirs, sync dirs, locks, and import-time config."""
    root = tmp_path_factory.mktemp("isolated-home")
    home = root / "home"
    xdg = home / ".config"
    cursor_user = xdg / "Cursor" / "User"
    (cursor_user / "globalStorage").mkdir(parents=True)
    (cursor_user / "workspaceStorage").mkdir(parents=True)
    sync_dir = home / ".cursaves"
    (sync_dir / "snapshots").mkdir(parents=True)
    config_dir = xdg / "cursaves"
    config_dir.mkdir(parents=True)
    (home / ".cursor" / "projects").mkdir(parents=True)
    cache_dir = home / ".cache" / "cursaves"
    cache_dir.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.setenv("CURSAVES_SQLITE_LOCK", str(root / "sqlite-write.lock"))
    monkeypatch.setenv("CURSAVES_REPO_LOCK", str(root / "repo.lock"))
    monkeypatch.setenv("CURSAVES_SQLITE_LOCK_TIMEOUT", "1")
    monkeypatch.setenv("CURSAVES_REPO_LOCK_TIMEOUT", "1")
    snap_root = root / "cursaves-snapshots"
    snap_root.mkdir()
    monkeypatch.setenv("CURSAVES_SNAPSHOT_ROOT", str(snap_root))

    monkeypatch.setattr(backends, "_CONFIG_PATH", config_dir / "config.json")

    def fake_cursor_user_dir() -> Path:
        return _guard(cursor_user)

    def fake_sync_dir() -> Path:
        return _guard(sync_dir)

    def fake_snapshots_dir() -> Path:
        snaps = sync_dir / "snapshots"
        snaps.mkdir(parents=True, exist_ok=True)
        return _guard(snaps)

    def fake_projects_dir() -> Path:
        return _guard(home / ".cursor" / "projects")

    def fake_global_db() -> Path:
        return _guard(cursor_user / "globalStorage" / "state.vscdb")

    def fake_workspace_storage() -> Path:
        return _guard(cursor_user / "workspaceStorage")

    def fake_cache_dir() -> Path:
        return _guard(cache_dir)

    monkeypatch.setattr(paths, "get_cursor_user_dir", fake_cursor_user_dir)
    monkeypatch.setattr(paths, "get_global_db_path", fake_global_db)
    monkeypatch.setattr(paths, "get_workspace_storage_dir", fake_workspace_storage)
    monkeypatch.setattr(paths, "get_sync_dir", fake_sync_dir)
    monkeypatch.setattr(paths, "get_snapshots_dir", fake_snapshots_dir)
    monkeypatch.setattr(paths, "get_cursor_projects_dir", fake_projects_dir)
    monkeypatch.setattr(paths, "get_cache_dir", fake_cache_dir)

    paths.invalidate_headers_cache()
    dblock.reset_for_tests()
    db.reset_write_tracking_for_tests()
    syncstate.reset_op_counts()

    yield

    paths.invalidate_headers_cache()
    db.reset_write_tracking_for_tests()
    dblock.reset_for_tests()
    syncstate.reset_op_counts()
