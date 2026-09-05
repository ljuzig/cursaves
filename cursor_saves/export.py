"""Export and list operations -- read-only, safe to run while Cursor is open."""

import gzip
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import db, dblock, paths


def get_workspace_conversations(
    project_path: str,
    workspace_dir: Optional[Path] = None,
    session=None,
) -> list[dict]:
    """Get the list of conversations for a project.

    If workspace_dir is provided, only reads from that specific workspace
    (avoids cross-host contamination for SSH workspaces with the same path).

    Combines global headers index (Cursor 3.0+), workspace DB allComposers
    (Cursor 2.x), and per-workspace pane/selection entries to build a
    complete list. Metadata for IDs only found via pane entries is fetched
    from the global DB's composerData. Pass *session* to reuse the
    command's global read epoch.
    """
    if workspace_dir is not None:
        ws_dirs = [workspace_dir]
    else:
        ws_dirs = paths.find_workspace_dirs_for_project(project_path)
    if not ws_dirs:
        return []

    all_conversations = []
    seen_ids: set[str] = set()
    ids_needing_metadata: list[tuple[str, str]] = []  # (composerId, ws_dir_str)
    headers_map = paths._build_global_headers_map(session)

    for ws_dir in ws_dirs:
        ws_hash = ws_dir.name
        ws_dir_str = str(ws_dir)

        # Source 1: global headers (has full metadata inline)
        for entry in headers_map.get(ws_hash, []):
            cid = entry.get("composerId")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                entry_copy = dict(entry)
                entry_copy["_workspaceDir"] = ws_dir_str
                all_conversations.append(entry_copy)

        # Source 2: workspace DB
        db_path = ws_dir / "state.vscdb"
        if not db_path.exists():
            continue

        with db.CursorDB(db_path) as cdb:
            data = cdb.get_json("composer.composerData", table="ItemTable")
            if not data:
                continue

            # allComposers (Cursor 2.x — has full metadata)
            for c in data.get("allComposers", []):
                cid = c.get("composerId")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    c["_workspaceDir"] = ws_dir_str
                    all_conversations.append(c)

            # selectedComposerIds + pane entries (need metadata lookup)
            extra_ids: set[str] = set()
            for cid in data.get("selectedComposerIds", []):
                if cid and cid not in seen_ids:
                    extra_ids.add(cid)
            for cid in data.get("lastFocusedComposerIds", []):
                if cid and cid not in seen_ids:
                    extra_ids.add(cid)
            for key in cdb.list_keys(
                "workbench.panel.composerChatViewPane.", table="ItemTable"
            ):
                pane = cdb.get_json(key, table="ItemTable")
                if isinstance(pane, dict):
                    for view_key in pane:
                        if ".view." in view_key:
                            cid = view_key.rsplit(".", 1)[-1]
                            if cid and cid not in seen_ids:
                                extra_ids.add(cid)

            for cid in extra_ids:
                seen_ids.add(cid)
                ids_needing_metadata.append((cid, ws_dir_str))

    # Fetch metadata for IDs found only via pane/selection entries
    if ids_needing_metadata:
        cdb = session.cdb if session is not None else None
        owned = None
        if cdb is None:
            global_db = paths.get_global_db_path()
            if global_db.exists():
                owned = db.CursorDB(global_db)
                cdb = owned
        try:
            if cdb is not None:
                for cid, ws_dir_str in ids_needing_metadata:
                    cd = cdb.get_json(f"composerData:{cid}")
                    if cd:
                        all_conversations.append({
                            "composerId": cid,
                            "name": cd.get("name", ""),
                            "createdAt": cd.get("createdAt", 0),
                            "lastUpdatedAt": cd.get("lastUpdatedAt", 0),
                            "unifiedMode": cd.get("unifiedMode", "agent"),
                            "forceMode": cd.get("forceMode", ""),
                            "_workspaceDir": ws_dir_str,
                        })
        finally:
            if owned is not None:
                owned.close()

    all_conversations.sort(
        key=lambda c: c.get("createdAt", 0), reverse=True
    )
    return all_conversations


def get_conversation_data(composer_id: str) -> Optional[dict]:
    """Fetch the full conversation data from the global DB."""
    global_db = paths.get_global_db_path()
    if not global_db.exists():
        return None

    try:
        with db.CursorDB(global_db) as cdb:
            return cdb.get_json(f"composerData:{composer_id}")
    except (OSError, FileNotFoundError) as e:
        print(f"Warning: Could not read global DB: {e}", file=sys.stderr)
        return None


def get_content_blobs(composer_id: str) -> dict[str, str]:
    """Fetch all content blobs referenced by a conversation.

    Scans the conversation data for content hash references and
    retrieves them from the global DB.
    """
    global_db = paths.get_global_db_path()
    if not global_db.exists():
        return {}

    conv_data = get_conversation_data(composer_id)
    if not conv_data:
        return {}

    # Serialise once for searching
    conv_json = json.dumps(conv_data)

    # Collect all content hashes referenced in the conversation
    # They appear in fullConversationHeadersOnly as bubbleId references
    # and the actual content is stored under composer.content.{hash}
    blobs = {}
    try:
        with db.CursorDB(global_db) as cdb:
            content_keys = cdb.list_keys("composer.content.")
            for key in content_keys:
                content_hash = key[len("composer.content."):]
                if content_hash in conv_json:
                    val = cdb.get_disk_kv(key)
                    if val:
                        blobs[content_hash] = val
    except (OSError, FileNotFoundError):
        pass  # Non-fatal: content blobs are supplementary

    return blobs


def get_message_contexts(composer_id: str) -> dict[str, Any]:
    """Fetch messageRequestContext entries for a conversation."""
    global_db = paths.get_global_db_path()
    if not global_db.exists():
        return {}

    contexts = {}
    with db.CursorDB(global_db) as cdb:
        keys = cdb.list_keys(f"messageRequestContext:{composer_id}:")
        for key in keys:
            val = cdb.get_json(key)
            if val:
                # Store with a short key (just the message part)
                short_key = key[len(f"messageRequestContext:{composer_id}:"):]
                contexts[short_key] = val

    return contexts


def get_bubble_entries(composer_id: str) -> dict[str, Any]:
    """Fetch individual message bubble entries for a conversation.

    Cursor stores message content under bubbleId:{composerId}:{bubbleId} keys.
    This is the new storage format (as of 2026) where conversationMap is empty
    and messages are stored individually.
    """
    global_db = paths.get_global_db_path()
    if not global_db.exists():
        return {}

    bubbles = {}
    with db.CursorDB(global_db) as cdb:
        keys = cdb.list_keys(f"bubbleId:{composer_id}:")
        for key in keys:
            val = cdb.get_json(key)
            if val:
                # Store with just the bubble ID as key
                bubble_id = key[len(f"bubbleId:{composer_id}:"):]
                bubbles[bubble_id] = val

    return bubbles


def get_transcript(project_path: str, composer_id: str) -> Optional[str]:
    """Get the agent transcript for a conversation, if it exists."""
    transcript_dir = paths.find_transcript_dir(project_path)
    if not transcript_dir:
        return None

    transcript_file = transcript_dir / f"{composer_id}.txt"
    if transcript_file.exists():
        try:
            return transcript_file.read_text()
        except OSError:
            return None

    return None


def format_timestamp(ts_ms: int) -> str:
    """Format a millisecond timestamp to a readable string."""
    if not ts_ms:
        return "unknown"
    try:
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, OSError):
        return "unknown"


def list_conversations(
    project_path: str,
    workspace_dir: Optional[Path] = None,
    session=None,
) -> list[dict]:
    """List all conversations for a project with display-friendly info.

    Returns list of dicts with: id, name, date, mode, messageCount.
    Pass an open SyncReadSession to reuse its SQLite copy and payload cache.
    """
    conversations = get_workspace_conversations(
        project_path, workspace_dir=workspace_dir, session=session
    )
    if not conversations:
        return []

    from . import syncstate

    def _collect(sess: "syncstate.SyncReadSession") -> list[dict]:
        results = []
        for c in conversations:
            composer_id = c.get("composerId", "unknown")
            if sess.local_presence(composer_id) != syncstate.LocalPresence.ACTIVE:
                continue
            conv_data = sess.composer_data(composer_id) or {}
            results.append({
                "id": composer_id,
                "name": c.get("name", "Untitled"),
                "date": format_timestamp(c.get("createdAt", 0)),
                "lastUpdated": format_timestamp(c.get("lastUpdatedAt", c.get("createdAt", 0))),
                "mode": c.get("unifiedMode", c.get("forceMode", "unknown")),
                "messageCount": syncstate.semantic_message_count(conv_data),
            })
        return results

    if session is not None:
        return _collect(session)
    with syncstate.SyncReadSession() as owned:
        return _collect(owned)


MAX_COMPRESSED_SIZE_MB = 95  # Stay under GitHub's 100MB limit
SHARD_SIZE_BYTES = 90 * 1024 * 1024  # 90MB per shard (GitHub rejects files > 100MB)
MAX_RECENT_CONTEXTS = 20     # Always keep this many recent message contexts


def _trim_message_contexts(contexts: dict[str, Any], max_size_bytes: int) -> dict[str, Any]:
    """Trim older message contexts to stay under size limit.
    
    Keeps the most recent contexts (by key, which includes message ID).
    """
    if not contexts:
        return contexts
    
    # Sort by key (message IDs are typically chronological or we keep all if small)
    sorted_keys = sorted(contexts.keys())
    
    # Always keep the last N contexts
    recent_keys = set(sorted_keys[-MAX_RECENT_CONTEXTS:])
    
    # Calculate current size
    current_size = sum(len(json.dumps(v)) for v in contexts.values())
    
    if current_size <= max_size_bytes:
        return contexts
    
    # Remove oldest contexts until we're under the limit
    trimmed = {}
    kept_size = 0
    
    # First, always include recent contexts
    for key in sorted_keys[-MAX_RECENT_CONTEXTS:]:
        trimmed[key] = contexts[key]
        kept_size += len(json.dumps(contexts[key]))
    
    # Then add older ones if there's room
    for key in sorted_keys[:-MAX_RECENT_CONTEXTS]:
        entry_size = len(json.dumps(contexts[key]))
        if kept_size + entry_size <= max_size_bytes:
            trimmed[key] = contexts[key]
            kept_size += entry_size
    
    return trimmed


def _extract_agent_blob_ids(conv_data: dict) -> set[str]:
    """Extract agentKv blob IDs referenced by a conversation.

    The composerData.conversationState field is a base64-encoded protobuf
    prefixed with '~'. It contains 32-byte blob IDs at multiple protobuf
    field numbers (1, 3, 8, 13, etc.) with wire type 2 (length-delimited).
    These reference agentKv:blob:{hex} entries in cursorDiskKV.

    Uses proper protobuf wire format parsing (varint tags + length
    prefixes) rather than naive byte scanning to avoid phantom matches.
    """
    import base64

    cs = conv_data.get("conversationState", "")
    if not cs or not isinstance(cs, str) or not cs.startswith("~") or len(cs) < 10:
        return set()

    try:
        raw = base64.b64decode(cs[1:])
    except Exception:
        return set()

    def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
        result = 0
        shift = 0
        while offset < len(data):
            b = data[offset]
            result |= (b & 0x7F) << shift
            offset += 1
            if (b & 0x80) == 0:
                return result, offset
            shift += 7
        return result, offset

    blob_ids: set[str] = set()
    i = 0
    end = len(raw)
    while i < end:
        tag, next_i = _read_varint(raw, i)
        wire_type = tag & 0x07

        if wire_type == 2 and next_i < end:
            length, data_start = _read_varint(raw, next_i)
            if length == 32 and data_start + 32 <= end:
                blob_ids.add(raw[data_start : data_start + 32].hex())
                i = data_start + 32
            elif length > 0 and data_start + length <= end:
                i = data_start + length
            else:
                i = next_i
        elif wire_type == 0:
            _, i = _read_varint(raw, next_i)
        elif wire_type == 5:
            i = next_i + 4
        elif wire_type == 1:
            i = next_i + 8
        else:
            i = next_i if next_i > i else i + 1

    return blob_ids


def _extract_agent_blobs(
    conv_data: dict,
    cdb: "db.CursorDB",
) -> dict[str, str]:
    """Fetch agentKv blob entries referenced by a conversation.

    Returns a dict mapping hex blob IDs to their base64-encoded values.
    Values are stored as binary in the DB; we base64-encode them for JSON
    serialization in the snapshot.
    """
    import base64

    blob_ids = _extract_agent_blob_ids(conv_data)
    if not blob_ids:
        return {}

    blobs: dict[str, str] = {}
    for bid in blob_ids:
        key = f"agentKv:blob:{bid}"
        val = cdb.get_item_binary(key, table="cursorDiskKV")
        if val is not None:
            blobs[bid] = base64.b64encode(val).decode("ascii")
    return blobs


def export_conversation(
    project_path: str,
    composer_id: str,
    _cdb: Optional[db.CursorDB] = None,
    source_host: Optional[str] = None,
    composer_data: Optional[dict] = None,
) -> Optional[dict]:
    """Export a single conversation to a self-contained snapshot dict.

    Includes messageContexts (file contents, git diffs) for seamless continuation.
    Size trimming happens in save_snapshot after checking compressed size.

    Pass an open CursorDB via _cdb to avoid re-copying the global DB.
    Pass source_host for SSH workspaces (e.g. "core-3").
    Pass composer_data to reuse a payload already read by SyncReadSession.
    """
    global_db = paths.get_global_db_path()
    own_cdb = _cdb is None
    if own_cdb:
        _cdb = db.CursorDB(global_db)

    try:
        conv_data = (
            composer_data
            if composer_data is not None
            else _cdb.get_json(f"composerData:{composer_id}")
        )
        from . import syncstate
        if syncstate.classify_local_payload(conv_data) != syncstate.LocalPresence.ACTIVE:
            return None

        # Bubble entries (individual message content)
        bubbles = {}
        for key in _cdb.list_keys(f"bubbleId:{composer_id}:"):
            val = _cdb.get_json(key)
            if val:
                bubble_id = key[len(f"bubbleId:{composer_id}:"):]
                bubbles[bubble_id] = val

        # Content blobs referenced by this conversation
        conv_json = json.dumps(conv_data)
        blobs = {}
        for key in _cdb.list_keys("composer.content."):
            content_hash = key[len("composer.content."):]
            if content_hash in conv_json:
                val = _cdb.get_disk_kv(key)
                if val:
                    blobs[content_hash] = val

        # Message request contexts
        contexts = {}
        for key in _cdb.list_keys(f"messageRequestContext:{composer_id}:"):
            val = _cdb.get_json(key)
            if val:
                short_key = key[len(f"messageRequestContext:{composer_id}:"):]
                contexts[short_key] = val

        # Checkpoint data (workspace state snapshots at each agent turn)
        checkpoints = {}
        for key in _cdb.list_keys(f"checkpointId:{composer_id}:"):
            val = _cdb.get_json(key)
            if val:
                cp_id = key[len(f"checkpointId:{composer_id}:"):]
                checkpoints[cp_id] = val

        # Agent state blobs (encrypted agent context needed for continuation).
        # The conversationState field in composerData is a protobuf containing
        # references to agentKv:blob:{hex} entries. Without these, Cursor's
        # agent loop fails with "Blob not found" when continuing the chat.
        agent_blobs = _extract_agent_blobs(conv_data, _cdb)

        snapshot = {
            "version": 3,
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "sourceMachine": paths.get_machine_id(),
            "sourceHost": source_host,
            "sourceProjectPath": paths.normalize_origin_path(
                project_path, source_host=source_host
            ),
            "projectIdentifier": paths.get_project_identifier(
                project_path, source_host=source_host
            ),
            "composerId": composer_id,
            "composerData": conv_data,
            "contentBlobs": blobs,
            "bubbleEntries": bubbles,
            "checkpoints": checkpoints,
            "agentBlobs": agent_blobs,
            "transcript": get_transcript(project_path, composer_id),
            "messageContexts": contexts,
        }

        return snapshot
    finally:
        if own_cdb:
            _cdb.close()


def _compress_snapshot(snapshot: dict) -> bytes:
    """Compress a snapshot dict to gzip bytes."""
    json_bytes = json.dumps(snapshot, ensure_ascii=False).encode("utf-8")
    import io
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as f:
        f.write(json_bytes)
    return buf.getvalue()


def clear_snapshot_components(project_dir: Path, composer_id: str) -> None:
    """Remove every on-disk representation of *composer_id* in *project_dir*.

    Covers uncompressed JSON, a single ``.json.gz``, shards, and the
    sidecar. Used by ``save_snapshot`` and ahead promote so a CID never
    keeps a mix of old and new layouts.
    """
    uncompressed = project_dir / f"{composer_id}.json"
    if uncompressed.exists():
        uncompressed.unlink()
    for old in project_dir.glob(f"{composer_id}.json.gz*"):
        old.unlink()
    meta = project_dir / f"{composer_id}.meta.json"
    if meta.exists():
        meta.unlink()


def install_staged_snapshot(
    dest_project: Path,
    composer_id: str,
    staged_project: Path,
) -> None:
    """Replace dest files for *composer_id* with exactly the staged set."""
    dest_project.mkdir(parents=True, exist_ok=True)
    clear_snapshot_components(dest_project, composer_id)
    if not staged_project.is_dir():
        return
    for src in staged_project.iterdir():
        if src.name.startswith(composer_id):
            shutil.copy2(src, dest_project / src.name)


def save_snapshot(snapshot: dict, snapshots_dir: Path) -> Path:
    """Save a snapshot dict to a compressed JSON file.
    
    If compressed size exceeds the limit, trims older messageContexts
    while keeping recent ones for seamless continuation.

    Returns the path to the saved file.
    """
    with dblock.repo_lock():
        return _save_snapshot_unlocked(snapshot, snapshots_dir)


def _save_snapshot_unlocked(snapshot: dict, snapshots_dir: Path) -> Path:
    # Organise by project identifier (git remote URL or directory name)
    project_id = snapshot.get("projectIdentifier")
    if not project_id:
        # Fallback for v1 snapshots without projectIdentifier
        project_id = os.path.basename(snapshot.get("sourceProjectPath", "unknown"))
    project_dir = snapshots_dir / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    composer_id = snapshot["composerId"]
    
    # Compress and check size
    max_size = MAX_COMPRESSED_SIZE_MB * 1024 * 1024
    compressed = _compress_snapshot(snapshot)
    
    # If too large, trim messageContexts and retry
    if len(compressed) > max_size and snapshot.get("messageContexts"):
        contexts = snapshot["messageContexts"]
        # Binary search for acceptable context size
        # Start by keeping only recent contexts
        trimmed_contexts = _trim_message_contexts(contexts, max_size * 5)  # Rough estimate
        snapshot["messageContexts"] = trimmed_contexts
        compressed = _compress_snapshot(snapshot)
        
        # If still too large, keep removing contexts
        while len(compressed) > max_size and len(snapshot.get("messageContexts", {})) > MAX_RECENT_CONTEXTS:
            # Remove half of the remaining non-recent contexts
            sorted_keys = sorted(snapshot["messageContexts"].keys())
            keep_keys = sorted_keys[len(sorted_keys)//2:]  # Keep newer half
            snapshot["messageContexts"] = {k: snapshot["messageContexts"][k] for k in keep_keys}
            compressed = _compress_snapshot(snapshot)
        
        # If still too large even with only recent contexts, remove contexts entirely
        if len(compressed) > max_size:
            snapshot["messageContexts"] = {}
            compressed = _compress_snapshot(snapshot)

    clear_snapshot_components(project_dir, composer_id)

    # Save snapshot (shard if too large for GitHub)
    snapshot_file = project_dir / f"{composer_id}.json.gz"
    if len(compressed) > SHARD_SIZE_BYTES:
        num_shards = 0
        for i in range(0, len(compressed), SHARD_SIZE_BYTES):
            shard_path = project_dir / f"{composer_id}.json.gz.{i // SHARD_SIZE_BYTES:02d}"
            shard_path.write_bytes(compressed[i:i + SHARD_SIZE_BYTES])
            num_shards += 1
        print(f"  Sharded into {num_shards} parts ({len(compressed) / 1024 / 1024:.1f} MB total)")
    else:
        snapshot_file.write_bytes(compressed)

    # Write lightweight metadata sidecar (avoids decompressing for listings)
    from . import syncstate

    cd = snapshot.get("composerData", {})
    num_shards = 0
    if len(compressed) > SHARD_SIZE_BYTES:
        num_shards = (len(compressed) + SHARD_SIZE_BYTES - 1) // SHARD_SIZE_BYTES
    meta = {
        "composerId": composer_id,
        "name": cd.get("name"),
        "messageCount": syncstate.semantic_message_count(cd),
        "exportedAt": snapshot.get("exportedAt"),
        "sourceMachine": snapshot.get("sourceMachine"),
        "sourceHost": snapshot.get("sourceHost"),
        "sourceProjectPath": snapshot.get("sourceProjectPath"),
        "projectIdentifier": snapshot.get("projectIdentifier"),
        "version": snapshot.get("version"),
        "shardCount": num_shards if num_shards else None,
    }
    try:
        meta["semanticDigest"] = syncstate.snapshot_semantic_digest(snapshot)
        meta["semanticDigestVersion"] = syncstate.SEMANTIC_DIGEST_VERSION
        meta["snapshotContentDigest"] = syncstate.snapshot_content_digest(
            snapshot_file, meta
        )
    except Exception:
        pass
    meta_file = project_dir / f"{composer_id}.meta.json"
    meta_file.write_text(json.dumps(meta, indent=2))

    return snapshot_file


def checkpoint_project(
    project_path: str,
    composer_ids: Optional[list[str]] = None,
    workspace_dir: Optional[Path] = None,
    source_host: Optional[str] = None,
) -> list[Path]:
    """Export conversations for a project to snapshots/.

    If composer_ids is given, only export those conversations.
    If workspace_dir is given, only reads from that specific workspace.
    Otherwise, export all conversations from all matching workspaces.

    Returns list of saved snapshot file paths.
    """
    snapshots_dir = paths.get_snapshots_dir()

    with dblock.repo_lock():
        return _checkpoint_project_unlocked(
            project_path,
            snapshots_dir,
            composer_ids=composer_ids,
            workspace_dir=workspace_dir,
            source_host=source_host,
        )


def _checkpoint_project_unlocked(
    project_path: str,
    snapshots_dir: Path,
    composer_ids: Optional[list[str]] = None,
    workspace_dir: Optional[Path] = None,
    source_host: Optional[str] = None,
) -> list[Path]:
    t0 = time.time()
    print("  Fetching workspace conversations...", file=sys.stderr, flush=True)
    from . import syncstate

    last_log_time = t0
    saved = []
    with syncstate.SyncReadSession() as session:
        conversations = get_workspace_conversations(
            project_path, workspace_dir=workspace_dir, session=session
        )
        print(
            f"  Found {len(conversations)} conversation(s) in workspace(s)",
            file=sys.stderr,
            flush=True,
        )

        to_process: list[tuple[dict, str]] = []
        for c in conversations:
            composer_id: str | None = c.get("composerId")
            if not composer_id:
                continue
            if composer_ids is not None and composer_id not in composer_ids:
                continue
            to_process.append((c, composer_id))

        print(f"  Processing {len(to_process)} conversation(s)...", file=sys.stderr, flush=True)
        for i, (c, composer_id) in enumerate(to_process, 1):
            try:
                if session.local_presence(composer_id) != syncstate.LocalPresence.ACTIVE:
                    continue
                snapshot = session.export_conversation(
                    project_path, composer_id, source_host=source_host
                )
                if snapshot:
                    path = save_snapshot(snapshot, snapshots_dir)
                    saved.append(path)
            finally:
                session.release_composer_cell(composer_id)

            if i % 10 == 0 or (time.time() - last_log_time) >= 10:
                print(f"  [{i}/{len(to_process)}] {composer_id}", file=sys.stderr, flush=True)
                last_log_time = time.time()

    total = time.time() - t0
    print(f"  Completed in {total:.1f}s", file=sys.stderr, flush=True)
    return saved
