"""Platform detection and Cursor storage path resolution."""

import json
import os
import platform
import posixpath
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlsplit


class AmbiguousWorkspaceError(ValueError):
    """More than one workspace matches an 8-character hash prefix."""

    def __init__(self, selector: str, matches: list[dict]):
        self.selector = selector
        self.matches = matches
        names = ", ".join(ws["workspace_dir"].name[:8] for ws in matches[:5])
        extra = "" if len(matches) <= 5 else f" (+{len(matches) - 5} more)"
        super().__init__(
            f"ambiguous workspace prefix {selector!r}: {names}{extra}"
        )


def is_workspace_hash_selector(selector: str) -> bool:
    """True for a full workspace hash or an 8-character hex prefix.

    Short digit-only strings stay numeric (``-w 1``). A full-length
    all-digit hash (16+ chars) is still a hash.
    """
    if not selector:
        return False
    if not all(c in "0123456789abcdefABCDEF" for c in selector):
        return False
    if len(selector) >= 16:
        return True
    if selector.isdigit():
        return False
    return len(selector) == 8


def get_cursor_user_dir() -> Path:
    """Return the Cursor User data directory for the current platform.

    macOS:  ~/Library/Application Support/Cursor/User
    Linux:  ~/.config/Cursor/User
    """
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "Cursor" / "User"
    elif system == "Linux":
        base = Path.home() / ".config" / "Cursor" / "User"
    else:
        print(
            f"Error: Unsupported platform '{system}'.\n"
            f"cursaves supports macOS and Linux.\n"
            f"On macOS, Cursor data is at ~/Library/Application Support/Cursor/User/\n"
            f"On Linux, Cursor data is at ~/.config/Cursor/User/",
            file=sys.stderr,
        )
        sys.exit(1)

    if not base.exists():
        print(
            f"Error: Cursor data directory not found at:\n"
            f"  {base}\n\n"
            f"This usually means:\n"
            f"  - Cursor is not installed on this machine, or\n"
            f"  - Cursor has never been opened (no data created yet), or\n"
            f"  - Cursor stores data at a non-standard location\n\n"
            f"Expected path for {system}: {base}",
            file=sys.stderr,
        )
        sys.exit(1)

    return base


def get_global_db_path() -> Path:
    """Return the path to Cursor's global state.vscdb."""
    return get_cursor_user_dir() / "globalStorage" / "state.vscdb"


def get_workspace_storage_dir() -> Path:
    """Return the path to Cursor's workspace storage directory."""
    return get_cursor_user_dir() / "workspaceStorage"


def get_cursor_projects_dir() -> Path:
    """Return the path to ~/.cursor/projects/ (agent transcripts, etc.)."""
    return Path.home() / ".cursor" / "projects"


def sanitize_project_path(project_path: str) -> str:
    """Convert a project path to Cursor's sanitized directory name format.

    /Users/callum/Desktop/Projects/myrepo -> Users-callum-Desktop-Projects-myrepo
    """
    # Strip leading slash and replace / with -
    return project_path.strip("/").replace("/", "-")


def _decode_ssh_host(host: str) -> str:
    """Decode an SSH host identifier.

    Cursor encodes SSH hosts as hex-encoded JSON, e.g.:
    7b22686f73744e616d65223a22636f7265227d -> {"hostName":"core"} -> core
    """
    try:
        # Try to decode as hex
        decoded = bytes.fromhex(host).decode("utf-8")
        # Try to parse as JSON
        data = json.loads(decoded)
        if isinstance(data, dict) and "hostName" in data:
            return data["hostName"]
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        pass
    return host


def _parse_ssh_remote_uri(uri: str) -> Optional[tuple[str, str]]:
    """Parse a vscode-remote:// SSH URI into (host, path).

    Accepts both folder and .code-workspace URIs. Only ssh-remote
    authorities are treated as SSH; other vscode-remote schemes
    (e.g. dev-container) are ignored.
    """
    parsed = urlsplit(uri)
    if parsed.scheme != "vscode-remote":
        return None
    authority = unquote(parsed.netloc)
    if authority.startswith("ssh-remote+"):
        host = authority[len("ssh-remote+") :]
    else:
        return None
    if not host:
        return None
    host = _decode_ssh_host(host)
    path = unquote(parsed.path or "")
    if not path or path == "/":
        return None
    return host, path


def find_workspace_dirs_for_project(project_path: str) -> list[Path]:
    """Find all workspace directories that map to a given project path.

    Scans workspace.json files in workspaceStorage/ to find matches.
    Returns list of workspace directory paths, newest first.
    """
    ws_storage = get_workspace_storage_dir()
    if not ws_storage.exists():
        return []

    # Normalise the target path for comparison
    target = os.path.normpath(os.path.expanduser(project_path))

    matches = []
    for ws_dir in ws_storage.iterdir():
        if not ws_dir.is_dir():
            continue
        ws_json = ws_dir / "workspace.json"
        if not ws_json.exists():
            continue
        try:
            data = json.loads(ws_json.read_text())
            folder_uri = data.get("folder") or ""
            if not folder_uri:
                folder_uri = data.get("workspace") or ""
            if folder_uri.startswith("file://"):
                folder_path = folder_uri[len("file://") :]
                folder_path = folder_path.replace("%20", " ")
            else:
                parsed = _parse_ssh_remote_uri(folder_uri)
                if parsed is None:
                    continue
                _host, folder_path = parsed

            if os.path.normpath(folder_path) == target:
                matches.append(ws_dir)
        except (json.JSONDecodeError, OSError):
            continue

    # Sort by modification time, newest first
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches


def find_transcript_dir(project_path: str) -> Optional[Path]:
    """Find the agent-transcripts directory for a project."""
    projects_dir = get_cursor_projects_dir()
    if not projects_dir.exists():
        return None

    sanitized = sanitize_project_path(project_path)
    transcript_dir = projects_dir / sanitized / "agent-transcripts"
    if transcript_dir.exists():
        return transcript_dir

    return None


def get_project_path() -> str:
    """Get the current project path (current working directory)."""
    return os.getcwd()


def list_all_workspaces() -> list[dict]:
    """List all Cursor workspaces with metadata.

    Returns a list of dicts with:
      - folder_uri: raw URI from workspace.json
      - path: extracted filesystem path (for workspace, path to the .code-workspace file)
      - type: 'local', 'ssh', or 'workspace'
      - host: SSH hostname (for ssh type, None otherwise)
      - workspace_dir: Path to the workspace directory
      - mtime: modification time of the workspace DB
    """
    ws_storage = get_workspace_storage_dir()
    if not ws_storage.exists():
        return []

    workspaces = []
    for ws_dir in ws_storage.iterdir():
        if not ws_dir.is_dir():
            continue
        ws_json = ws_dir / "workspace.json"
        if not ws_json.exists():
            continue
        try:
            data = json.loads(ws_json.read_text())

            ws_type = "local"
            host = None
            folder_path = ""
            folder_uri = ""

            # .code-workspace files use "workspace" instead of "folder".
            # Remote SSH multi-root workspaces use the same vscode-remote URI.
            if "workspace" in data and not data.get("folder"):
                ws_uri = data["workspace"]
                if ws_uri.startswith("file://"):
                    folder_uri = ws_uri
                    folder_path = ws_uri[len("file://") :]
                    folder_path = folder_path.replace("%20", " ")
                    ws_type = "workspace"
                else:
                    parsed = _parse_ssh_remote_uri(ws_uri)
                    if parsed is None:
                        continue
                    host, folder_path = parsed
                    folder_uri = ws_uri
                    ws_type = "ssh"
            else:
                folder_uri = data.get("folder", "")
                if not folder_uri:
                    continue

                if folder_uri.startswith("file://"):
                    folder_path = folder_uri[len("file://") :]
                    folder_path = folder_path.replace("%20", " ")
                else:
                    parsed = _parse_ssh_remote_uri(folder_uri)
                    if parsed is None:
                        continue
                    host, folder_path = parsed
                    ws_type = "ssh"

            # Get DB modification time
            db_path = ws_dir / "state.vscdb"
            mtime = db_path.stat().st_mtime if db_path.exists() else 0

            workspaces.append({
                "folder_uri": folder_uri,
                "path": os.path.normpath(folder_path),
                "type": ws_type,
                "host": host,
                "workspace_dir": ws_dir,
                "mtime": mtime,
            })
        except (json.JSONDecodeError, OSError):
            continue

    # Sort by modification time, newest first
    workspaces.sort(key=lambda w: w["mtime"], reverse=True)
    return workspaces


def get_global_composer_headers(session=None) -> list[dict]:
    """Read the central composer.composerHeaders from the global DB.

    Returns the allComposers list from composer.composerHeaders in the
    global DB's ItemTable. This is the legacy JSON index. When the
    typed ``composerHeaders`` table exists, that table is current.

    Pass an open SyncReadSession to reuse its read epoch. Returns an
    empty list if not present (pre-3.0 Cursor).
    """
    if session is not None:
        return session.composer_headers()

    from . import db

    global_db = get_global_db_path()
    if not global_db.exists():
        return []
    try:
        with db.CursorDB(global_db) as cdb:
            headers = cdb.get_json("composer.composerHeaders", table="ItemTable")
            if headers and isinstance(headers, dict):
                return headers.get("allComposers", [])
    except Exception:
        pass
    return []


_global_headers_cache: Optional[dict[str, list[dict]]] = None
_global_typed_cache: Optional[tuple[bool, dict]] = None


def _headers_map_from_entries(entries) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for entry in entries:
        wi = entry.get("workspaceIdentifier", {})
        ws_id = wi.get("id", "")
        if ws_id:
            result.setdefault(ws_id, []).append(entry)
    return result


def _load_global_index_unlocked() -> None:
    """Read JSON headers and the typed catalog from one CursorDB."""
    global _global_headers_cache, _global_typed_cache
    from . import db
    from . import typed_headers

    global_db = get_global_db_path()
    if not global_db.exists():
        _global_headers_cache = {}
        _global_typed_cache = (False, {})
        return
    with db.CursorDB(global_db) as cdb:
        headers = cdb.get_json("composer.composerHeaders", table="ItemTable")
        entries = []
        if headers and isinstance(headers, dict):
            entries = headers.get("allComposers", []) or []
        _global_headers_cache = _headers_map_from_entries(entries)
        conn = cdb._ensure_read_copy()
        exists = typed_headers.typed_table_usable(conn)
        catalog = typed_headers.load_typed_catalog(conn) if exists else {}
        _global_typed_cache = (exists, catalog)


def _build_global_headers_map(session=None) -> dict[str, list[dict]]:
    """Build a workspace-hash → [composer header entries] map.

    When *session* is given, the map is taken from that read epoch and
    is not stored as a process-global cache. Without a session the
    process cache is a convenience for write-adjacent callers, not a
    read epoch: it is invalidated after Cursor writes.
    """
    if session is not None:
        return session.headers_map()

    global _global_headers_cache
    if _global_headers_cache is None:
        _load_global_index_unlocked()
    return _global_headers_cache


def typed_index_state(session=None) -> tuple[bool, dict]:
    """Return ``(typed_table_exists, catalog)`` from the command epoch.

    Without *session*, shares the one-connection process cache used by
    ``_build_global_headers_map``.
    """
    if session is not None:
        return session.typed_table_exists(), session.typed_catalog()

    global _global_typed_cache
    if _global_typed_cache is None:
        _load_global_index_unlocked()
    return _global_typed_cache


def invalidate_headers_cache():
    """Clear the cached global headers map (call after writing to the global DB)."""
    global _global_headers_cache, _global_typed_cache
    _global_headers_cache = None
    _global_typed_cache = None


def _legacy_workspace_composer_ids(
    ws_db_path: Path,
    ws_hash: str,
    headers_map: dict[str, list[dict]],
) -> set[str]:
    """JSON headers + workspace selected/focused/pane/allComposers."""
    from . import db

    ids: set[str] = set()
    for entry in headers_map.get(ws_hash, []):
        cid = entry.get("composerId")
        if cid:
            ids.add(cid)
    try:
        with db.CursorDB(ws_db_path) as cdb:
            data = cdb.get_json("composer.composerData", table="ItemTable")
            if not data:
                return ids
            for c in data.get("allComposers", []):
                cid = c.get("composerId")
                if cid:
                    ids.add(cid)
            for cid in data.get("selectedComposerIds", []):
                if cid:
                    ids.add(cid)
            for cid in data.get("lastFocusedComposerIds", []):
                if cid:
                    ids.add(cid)
            for key in cdb.list_keys(
                "workbench.panel.composerChatViewPane.", table="ItemTable"
            ):
                pane = cdb.get_json(key, table="ItemTable")
                if isinstance(pane, dict):
                    for view_key in pane:
                        if ".view." in view_key:
                            cid = view_key.rsplit(".", 1)[-1]
                            if cid:
                                ids.add(cid)
    except Exception:
        pass
    return ids


def get_workspace_composer_ids(
    ws_db_path: Path,
    session=None,
    headers_map: Optional[dict[str, list[dict]]] = None,
) -> list[str]:
    """Extract all composer IDs associated with a workspace.

    Two generations, not a naive union:

    * CURRENT: physical ``composerHeaders`` table. A CID with a typed
      row is bound exclusively to that row's ``workspaceId``.
    * LEGACY: ``composer.composerHeaders`` JSON plus workspace
      selected/focused/pane/allComposers, used only for CIDs that have
      no typed row.

    When the typed table is absent (older Cursor), behaviour matches
    the previous JSON + workspace union.

    Pass *session* or *headers_map* to reuse the command's global read
    epoch instead of opening another copy.
    """
    ws_hash = ws_db_path.parent.name
    if headers_map is None:
        headers_map = _build_global_headers_map(session)
    typed_exists, typed_catalog = typed_index_state(session)
    legacy_ids = _legacy_workspace_composer_ids(ws_db_path, ws_hash, headers_map)
    if not typed_exists:
        return list(legacy_ids)

    typed_ids = {
        cid
        for cid, row in typed_catalog.items()
        if getattr(row, "workspace_id", None) == ws_hash
        and not getattr(row, "is_subagent", 0)
    }
    typed_all = set(typed_catalog.keys())
    return list(typed_ids | (legacy_ids - typed_all))


def list_workspaces_with_conversations(session=None) -> list[dict]:
    """List workspaces that have at least one conversation.

    Returns the same dicts as list_all_workspaces(), plus a
    'conversations' key with the count. Pass *session* so membership
    reads share one global headers map.
    """
    headers_map = _build_global_headers_map(session)
    result = []
    for ws in list_all_workspaces():
        db_path = ws["workspace_dir"] / "state.vscdb"
        if not db_path.exists():
            continue
        composer_ids = get_workspace_composer_ids(
            db_path, session=session, headers_map=headers_map
        )
        if composer_ids:
            ws["conversations"] = len(composer_ids)
            result.append(ws)
    return result


def _resolve_workspace_by_hash(selector: str) -> Optional[dict]:
    """Resolve a full hash or unique 8-character prefix via workspace.json only."""
    matches: list[dict] = []
    for ws in list_all_workspaces():
        name = ws["workspace_dir"].name
        if len(selector) == 8:
            if name.startswith(selector):
                matches.append(ws)
        elif name == selector:
            matches.append(ws)
    if len(matches) > 1:
        raise AmbiguousWorkspaceError(selector, matches)
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_workspace(selector: str, session=None) -> Optional[dict]:
    """Resolve a workspace selector to a workspace dict.

    The selector can be:
      - A number (1-based index from list_workspaces_with_conversations)
      - A workspace hash (directory name under workspaceStorage/)
      - A path substring (matched against workspace paths)

    Full hashes and 8-character prefixes use ``list_all_workspaces()``
    and do not enumerate conversations. Ambiguous prefixes raise
    ``AmbiguousWorkspaceError``. Numeric and substring selectors keep
    the conversation-filtered order.
    """
    if is_workspace_hash_selector(selector):
        return _resolve_workspace_by_hash(selector)

    workspaces = list_workspaces_with_conversations(session=session)

    if selector.isdigit():
        idx = int(selector)
        if workspaces and 1 <= idx <= len(workspaces):
            return workspaces[idx - 1]
        if len(selector) == 8 or len(selector) >= 16:
            return _resolve_workspace_by_hash(selector)
        return None

    for ws in workspaces:
        if selector in ws["path"]:
            return ws

    return None


def get_sync_dir() -> Path:
    """Return the cursaves sync directory (~/.cursaves/).

    This is the git repo that holds snapshots and is synced between machines.
    """
    return Path.home() / ".cursaves"


def get_snapshots_dir() -> Path:
    """Return the snapshots directory (~/.cursaves/snapshots/)."""
    snapshots = get_sync_dir() / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    return snapshots


def is_sync_repo_initialized() -> bool:
    """Check if a sync backend has been configured (git repo or cloud)."""
    sync_dir = get_sync_dir()
    if (sync_dir / ".git").exists():
        return True
    # Check for non-git backend config
    config_path = Path.home() / ".config" / "cursaves" / "config.json"
    if config_path.exists():
        try:
            import json
            cfg = json.loads(config_path.read_text())
            return cfg.get("backend") in ("s3", "azure")
        except Exception:
            pass
    return False


def get_machine_id() -> str:
    """Return a human-readable machine identifier."""
    import socket

    return socket.gethostname()


# ── Workspace matching for imports ─────────────────────────────────────


def find_all_matching_workspaces(
    source_path: str,
    source_host: Optional[str] = None,
) -> list[dict]:
    """Find all workspaces that could receive imports from source_path.

    Matches by:
    1. Exact path match (for SSH workspaces with same remote path)
    2. Same basename (fallback for different directory structures)

    When source_host is provided, matching is restricted to SSH workspaces
    with that exact host AND the exact remote path. There is no fallback to
    a different host, and no basename guess on the same host — better to
    match nothing than import a chat into the wrong workspace.

    Returns list of workspace dicts with type, host, path, workspace_dir,
    sorted by match quality (exact matches first) then by mtime.
    """
    all_ws = list_all_workspaces()
    source_normalized = os.path.normpath(source_path)
    source_basename = os.path.basename(source_normalized)

    if source_host:
        source_normalized = normalize_origin_path(source_path, source_host=source_host)
        matches = [
            ws
            for ws in all_ws
            if ws.get("type") == "ssh"
            and ws.get("host") == source_host
            and normalize_origin_path(ws["path"], source_host=source_host) == source_normalized
        ]
        matches.sort(key=lambda w: w.get("mtime", 0), reverse=True)
        return matches

    exact_matches = []
    basename_matches = []

    for ws in all_ws:
        ws_path = os.path.normpath(ws["path"])
        ws_basename = os.path.basename(ws_path)

        if ws_path == source_normalized:
            exact_matches.append(ws)
        elif ws_basename == source_basename:
            basename_matches.append(ws)

    # Return exact matches first, then basename matches
    return exact_matches + basename_matches


def format_workspace_display(ws: dict, include_path: bool = True) -> str:
    """Format a workspace dict for display.

    Returns a string like "ssh core /mnt/home/.../project", "(local) /home/.../project",
    or "(workspace) /home/.../my-proj.code-workspace"
    """
    if ws["type"] == "ssh":
        host = ws.get("host") or "unknown"
        if include_path:
            path = ws["path"]
            if len(path) > 40:
                path = "..." + path[-37:]
            return f"ssh {host} {path}"
        return f"ssh {host}"
    elif ws["type"] == "workspace":
        if include_path:
            path = ws["path"]
            if len(path) > 45:
                path = "..." + path[-42:]
            return f"(workspace) {path}"
        return "(workspace)"
    else:
        if include_path:
            path = ws["path"]
            if len(path) > 45:
                path = "..." + path[-42:]
            return f"(local) {path}"
        return "(local)"


# ── Project identification ────────────────────────────────────────────


def normalize_origin_path(project_path: str, *, source_host: Optional[str] = None) -> str:
    """Normalize a project path for origin identity.

    SSH remote paths belong to the remote host and are POSIX even when
    cursaves runs on another platform. Local paths use local semantics.
    """
    if source_host:
        return posixpath.normpath(project_path)
    return os.path.normpath(project_path)


def get_cache_dir() -> Path:
    """Regenerable local cache (semantic fingerprints). Not in the git repo.

    Linux: ``$XDG_CACHE_HOME/cursaves`` or ``~/.cache/cursaves``.
    macOS: ``~/Library/Caches/cursaves``.
    """
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Caches" / "cursaves"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "cursaves"
    return Path.home() / ".cache" / "cursaves"


def get_project_identifier(
    project_path: str,
    source_host: Optional[str] = None,
) -> str:
    """Get a stable identifier for a project, used as the snapshot subdirectory.

    Uses the git remote origin URL if available (normalized to a filesystem-safe
    string).  Falls back to the directory basename for non-git projects.

    This means:
      - Same repo under different local names (bob/ vs alice/) → same identifier
      - Different repos that happen to share a name → different identifiers

    The project path of a Remote SSH workspace exists on the remote host,
    not necessarily on the machine running cursaves. Therefore git -C
    cannot reliably discover remote.origin.url locally. When source_host is
    set, identity is derived from host + normalized remote path instead.
    """
    if source_host:
        normalized_path = normalize_origin_path(project_path, source_host=source_host)
        return _sanitize_identifier(f"ssh/{source_host}/{normalized_path}")

    remote_url = _get_git_remote_url(project_path)
    if remote_url:
        return _normalize_remote_url(remote_url)
    return os.path.basename(os.path.normpath(project_path))


def get_workspace_project_identifier(ws: dict) -> str:
    """Project identifier for a workspace dict, including SSH host when present."""
    return get_project_identifier(ws["path"], source_host=ws.get("host"))


def _get_git_remote_url(project_path: str) -> Optional[str]:
    """Get the git remote origin URL for a project, if any."""
    try:
        result = subprocess.run(
            ["git", "-C", project_path, "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _normalize_remote_url(url: str) -> str:
    """Normalize a git remote URL to a stable, filesystem-safe directory name.

    git@github.com:user/repo.git     → github.com-user-repo
    https://github.com/user/repo.git → github.com-user-repo
    ssh://git@github.com/user/repo   → github.com-user-repo
    """
    # Strip trailing .git
    url = re.sub(r"\.git$", "", url)

    # SSH shorthand: git@host:user/repo
    m = re.match(r"^[\w.-]+@([\w.-]+):(.*)", url)
    if m:
        host, path = m.group(1), m.group(2)
        return _sanitize_identifier(f"{host}/{path}")

    # HTTPS / SSH URI: https://host/path or ssh://git@host/path
    m = re.match(r"^(?:https?|ssh)://(?:[\w.-]+@)?([\w.-]+)/(.*)", url)
    if m:
        host, path = m.group(1), m.group(2)
        return _sanitize_identifier(f"{host}/{path}")

    # Unknown format -- sanitize whatever we got
    return _sanitize_identifier(url)


def _sanitize_identifier(s: str) -> str:
    """Turn an arbitrary string into a safe directory name.

    Replaces slashes, colons, @, etc. with '-' and collapses runs of dashes.
    """
    s = re.sub(r"[/:@\\]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")
