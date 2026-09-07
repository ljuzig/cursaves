"""Preserve both branches when same-CID histories are not safe to merge.

Remote snapshot keeps the original composer ID. The local branch is
cloned under a new CID, exported, then the snapshot is restored onto
the original ID. Next sync compares X↔X and Y↔Y with no leftover
ambiguity.

This module never decides lineage. Callers pass already-classified
``DIVERGED`` items.
"""

from __future__ import annotations

import copy
import gzip
import json
import uuid
from pathlib import Path
from typing import Optional

from . import db, importer, paths, syncstate


LOCAL_FORK_SUFFIX = "local"


def rekey_snapshot(snapshot: dict, new_id: str, *, name_suffix: str = LOCAL_FORK_SUFFIX) -> dict:
    """Copy a snapshot onto a new composer ID. Bubble bodies keep their IDs."""
    out = copy.deepcopy(snapshot)
    out["composerId"] = new_id
    data = out.get("composerData")
    if isinstance(data, dict):
        data["composerId"] = new_id
        name = data.get("name") or "Untitled"
        tag = f"({name_suffix})"
        if tag not in name:
            data["name"] = f"{name} {tag}"
    return out


def write_temp_snapshot(snapshot: dict, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{snapshot['composerId']}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(snapshot, handle, separators=(",", ":"))
    return path


_COMPOSER_OWNED_PREFIXES = (
    "bubbleId:{cid}:",
    "checkpointId:{cid}:",
    "messageRequestContext:{cid}:",
)


def _import_on_connections(
    snapshot_path: Path,
    item: "syncstate.PlannedItem",
    global_cdb: "db.CursorDB",
    workspace_cdb: "db.CursorDB",
) -> bool:
    return importer.import_snapshot(
        snapshot_path,
        item.project_path,
        target_workspace_dir=item.workspace_dir,
        skip_backup=True,
        skip_conflict=True,
        quiet=True,
        global_cdb=global_cdb,
        workspace_cdb=workspace_cdb,
    )


def _purge_composer_owned_rows(global_cdb: "db.CursorDB", composer_id: str) -> None:
    """Delete CID-owned physical rows so a restore is a replacement, not an overlay.

    Leaves ``composerData:{cid}`` for the following import to overwrite.
    Does not touch ``composer.content.*`` or ``agentKv:blob:*``.
    """
    for template in _COMPOSER_OWNED_PREFIXES:
        global_cdb.delete_keys_by_prefix(template.format(cid=composer_id))


def reconcile_fork(
    item: "syncstate.PlannedItem",
    local_snapshot: dict,
    remote_snapshot_path: Path,
    staging_dir: Path,
) -> Optional[tuple[str, dict]]:
    """Clone local branch to a new CID, restore snapshot onto the original CID.

    Both Cursor writes share one transaction. The caller writes the new
    snapshot only after releasing the sqlite write lock. Returns
    ``(new_cid, cloned_snapshot)``, or None on failure.
    """
    if item.workspace_dir is None or item.snapshot_path is None:
        return None
    new_id = str(uuid.uuid4())
    cloned = rekey_snapshot(local_snapshot, new_id)
    clone_path = write_temp_snapshot(cloned, staging_dir)

    global_path = paths.get_global_db_path()
    ws_db = Path(item.workspace_dir) / "state.vscdb"
    if not global_path.exists() or not ws_db.exists():
        return None

    global_cdb = db.CursorDB(global_path)
    workspace_cdb = db.CursorDB(ws_db)
    global_cdb.enable_batch_writes()
    workspace_cdb.enable_batch_writes()
    try:
        if not syncstate.local_guard_still_matches(item, global_cdb, workspace_cdb):
            syncstate._counts.local_guard_skips += 1
            return None
        global_cdb.begin()
        workspace_cdb.begin()
        if not _import_on_connections(clone_path, item, global_cdb, workspace_cdb):
            importer._rollback_write_pair(global_cdb, workspace_cdb)
            return None
        _purge_composer_owned_rows(global_cdb, item.composer_id)
        if not _import_on_connections(remote_snapshot_path, item, global_cdb, workspace_cdb):
            importer._rollback_write_pair(global_cdb, workspace_cdb)
            return None
        global_cdb.commit_write()
        workspace_cdb.commit_write()
    except Exception:
        importer._rollback_write_pair(global_cdb, workspace_cdb)
        return None
    finally:
        global_cdb.close()
        workspace_cdb.close()

    return new_id, cloned
