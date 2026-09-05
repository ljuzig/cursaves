"""Current-generation ``composerHeaders`` table (typed SQLite index).

Cursor now stores the chat–workspace index in a physical table. The
legacy ``ItemTable['composer.composerHeaders']`` JSON blob is a fallback
only for CIDs that have no typed row. This module never CREATE-s or
ALTERs the table: if it is absent, callers keep the JSON writer; if it
exists with an unexpected schema, writers fail closed.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from urllib.parse import unquote, urlsplit

TABLE_NAME = "composerHeaders"

REQUIRED_COLUMNS = frozenset({
    "composerId",
    "workspaceId",
    "createdAt",
    "lastUpdatedAt",
    "isArchived",
    "isSubagent",
    "recency",
    "checkpointAt",
    "value",
})

_OPTIONAL_HEADER_KEYS = (
    "name",
    "lastUpdatedAt",
    "hasPendingPlan",
    "conversationCheckpointLastUpdatedAt",
    "isArchived",
    "isSubagent",
    "isBestOfNSubcomposer",
)


class TypedSchemaStatus(str, Enum):
    ABSENT = "absent"
    USABLE = "usable"
    INCOMPATIBLE = "incompatible"


class WorkspaceBinding(str, Enum):
    SAME = "same"
    FOREIGN_WORKSPACE = "foreign_workspace"
    STALE_WORKSPACE_ID_SAME_URI = "stale_workspace_id_same_uri"


class RegistrationHealth(str, Enum):
    """Visibility of a chat in the current Cursor index.

    Independent of ``SyncRelation``, which describes content.
    """

    REGISTERED = "registered"
    LEGACY_ONLY = "legacy_only"
    MISSING = "missing"
    MISREGISTERED = "misregistered"
    STALE_WORKSPACE = "stale_workspace"


REGISTRATION_CONFLICTS = frozenset({
    RegistrationHealth.MISREGISTERED,
    RegistrationHealth.STALE_WORKSPACE,
})


class TypedSchemaError(Exception):
    """``composerHeaders`` exists but does not match the expected columns."""


class RegistrationConflictError(Exception):
    """Preflight found a typed row bound to an incompatible workspace."""

    def __init__(self, items):
        self.items = list(items)
        n = len(self.items)
        super().__init__(
            f"{n} conversation(s) have a typed composerHeaders row "
            f"in an incompatible workspace; will not auto-rebind"
        )


class MisregisteredHeaderError(Exception):
    """Typed row exists for a different workspace; refuse a silent rebind."""

    def __init__(
        self,
        composer_id: str,
        typed_workspace_id: str,
        target_workspace_id: str,
        *,
        binding: WorkspaceBinding = WorkspaceBinding.FOREIGN_WORKSPACE,
    ):
        self.composer_id = composer_id
        self.typed_workspace_id = typed_workspace_id
        self.target_workspace_id = target_workspace_id
        self.binding = binding
        if binding == WorkspaceBinding.STALE_WORKSPACE_ID_SAME_URI:
            detail = (
                f"composer {composer_id} has a stale workspaceId "
                f"{typed_workspace_id} for the same project URI as "
                f"{target_workspace_id}; will not auto-rebind"
            )
        else:
            detail = (
                f"composer {composer_id} is registered in workspace "
                f"{typed_workspace_id}, not {target_workspace_id}"
            )
        super().__init__(detail)


@dataclass(frozen=True)
class TypedHeaderRow:
    composer_id: str
    workspace_id: str
    created_at: Optional[int]
    last_updated_at: Optional[int]
    is_archived: int
    is_subagent: int
    recency: Optional[int]
    checkpoint_at: Optional[int]
    value: str

    @property
    def header(self) -> dict:
        try:
            parsed = json.loads(self.value) if self.value else {}
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


def has_typed_headers_table(conn: sqlite3.Connection) -> bool:
    return typed_schema_status(conn) != TypedSchemaStatus.ABSENT


def typed_schema_status(conn: sqlite3.Connection) -> TypedSchemaStatus:
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (TABLE_NAME,),
        ).fetchone()
    except sqlite3.Error:
        return TypedSchemaStatus.ABSENT
    if row is None:
        return TypedSchemaStatus.ABSENT
    try:
        cols = conn.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()
    except sqlite3.Error:
        return TypedSchemaStatus.INCOMPATIBLE
    names = {r[1] for r in cols}
    if REQUIRED_COLUMNS <= names:
        return TypedSchemaStatus.USABLE
    return TypedSchemaStatus.INCOMPATIBLE


def typed_table_usable(conn: sqlite3.Connection) -> bool:
    return typed_schema_status(conn) == TypedSchemaStatus.USABLE


def assert_typed_schema_writable(conn: sqlite3.Connection) -> TypedSchemaStatus:
    """ABSENT (use legacy writer) or USABLE. INCOMPATIBLE raises."""
    status = typed_schema_status(conn)
    if status == TypedSchemaStatus.INCOMPATIBLE:
        try:
            names = {
                r[1] for r in conn.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()
            }
        except sqlite3.Error:
            names = set()
        missing = ", ".join(sorted(REQUIRED_COLUMNS - names)) or "unknown"
        raise TypedSchemaError(
            "Cursor composerHeaders table exists but does not match the "
            f"expected columns (missing: {missing}). cursaves will not "
            "write this schema."
        )
    return status


def load_typed_catalog(conn: sqlite3.Connection) -> dict[str, TypedHeaderRow]:
    """Read the typed index from an already-open connection.

    Callers must pass the command's ReadEpoch / write connection — never
    open a second global SQLite connection just for this table.
    """
    if typed_schema_status(conn) != TypedSchemaStatus.USABLE:
        return {}
    try:
        rows = conn.execute(
            "SELECT composerId, workspaceId, createdAt, lastUpdatedAt, "
            "isArchived, isSubagent, recency, checkpointAt, value "
            f"FROM {TABLE_NAME}"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    catalog: dict[str, TypedHeaderRow] = {}
    for row in rows:
        cid = row[0]
        if not cid:
            continue
        catalog[cid] = _row_from_tuple(row)
    return catalog


def get_typed_row(conn: sqlite3.Connection, composer_id: str) -> Optional[TypedHeaderRow]:
    if typed_schema_status(conn) != TypedSchemaStatus.USABLE:
        return None
    try:
        row = conn.execute(
            "SELECT composerId, workspaceId, createdAt, lastUpdatedAt, "
            "isArchived, isSubagent, recency, checkpointAt, value "
            f"FROM {TABLE_NAME} WHERE composerId = ?",
            (composer_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    return _row_from_tuple(row)


def canonical_workspace_uri(identifier: Optional[dict]) -> Optional[str]:
    """Stable URI key for comparing workspaceIdentity across reincarnations."""
    if not isinstance(identifier, dict):
        return None
    uri = identifier.get("uri")
    raw = ""
    if isinstance(uri, dict):
        raw = uri.get("external") or ""
        if not raw:
            scheme = uri.get("scheme") or "file"
            path = uri.get("fsPath") or uri.get("path") or ""
            if path and scheme == "file":
                raw = "file://" + path
            elif path:
                raw = path
    elif isinstance(uri, str):
        raw = uri
    if not raw:
        return None
    parsed = urlsplit(raw)
    path = unquote(parsed.path or "").replace("%20", " ")
    if parsed.scheme == "file":
        return "file://" + path.rstrip("/") or "file://"
    if parsed.scheme:
        return f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/")
    return raw.rstrip("/")


def classify_workspace_binding(
    *,
    typed_workspace_id: str,
    target_workspace_id: str,
    typed_identifier: Optional[dict] = None,
    target_identifier: Optional[dict] = None,
) -> WorkspaceBinding:
    if typed_workspace_id == target_workspace_id:
        return WorkspaceBinding.SAME
    typed_uri = canonical_workspace_uri(typed_identifier)
    target_uri = canonical_workspace_uri(target_identifier)
    if typed_uri and target_uri and typed_uri == target_uri:
        return WorkspaceBinding.STALE_WORKSPACE_ID_SAME_URI
    return WorkspaceBinding.FOREIGN_WORKSPACE


def classify_registration(
    *,
    typed_table_exists: bool,
    typed_workspace_id: Optional[str],
    target_workspace_id: str,
    in_legacy_sources: bool,
    typed_identifier: Optional[dict] = None,
    target_identifier: Optional[dict] = None,
) -> RegistrationHealth:
    if not typed_table_exists:
        return RegistrationHealth.REGISTERED
    if typed_workspace_id is None:
        return (
            RegistrationHealth.LEGACY_ONLY
            if in_legacy_sources
            else RegistrationHealth.MISSING
        )
    binding = classify_workspace_binding(
        typed_workspace_id=typed_workspace_id,
        target_workspace_id=target_workspace_id,
        typed_identifier=typed_identifier,
        target_identifier=target_identifier,
    )
    if binding == WorkspaceBinding.SAME:
        return RegistrationHealth.REGISTERED
    if binding == WorkspaceBinding.STALE_WORKSPACE_ID_SAME_URI:
        return RegistrationHealth.STALE_WORKSPACE
    return RegistrationHealth.MISREGISTERED


def is_registration_conflict(health: Optional[RegistrationHealth]) -> bool:
    return health in REGISTRATION_CONFLICTS


def format_registration_conflict_abort(items, command: str) -> str:
    """Human-readable abort for pre-existing foreign/stale typed rows."""
    lines = [
        f"{command} aborted: conversations are typed-registered "
        "in another workspace.",
        "",
    ]
    for item in items:
        health = getattr(item.registration, "value", item.registration) or "conflict"
        name = getattr(item, "name", "") or "Untitled"
        cid = getattr(item, "composer_id", "")
        lines.append(f"  {cid[:12]}  {name}  ({health})")
    lines.append("")
    lines.append(
        f"Will not auto-rebind. {command} stopped before writing "
        "Cursor or snapshots."
    )
    return "\n".join(lines)


def _pick(key: str, *sources: Optional[dict], default: Any = None) -> Any:
    for src in sources:
        if isinstance(src, dict) and key in src and src[key] is not None:
            return src[key]
    return default


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_flag(value: Any) -> bool:
    if value in (True, 1, "1", "true", "True"):
        return True
    return False


def _task_like_composer_id(composer_id: Optional[str]) -> bool:
    return isinstance(composer_id, str) and composer_id.startswith("task-")


def _explicitly_top_level(
    *sources: Optional[dict], typed_row: Optional[TypedHeaderRow] = None
) -> bool:
    if typed_row is not None:
        return not typed_row.is_subagent
    for src in sources:
        if not isinstance(src, dict) or "isSubagent" not in src:
            continue
        if src["isSubagent"] is None:
            continue
        if not _as_flag(src["isSubagent"]) and not _as_flag(
            src.get("isBestOfNSubcomposer")
        ):
            return True
    return False


def looks_like_subagent(
    *sources: Optional[dict],
    typed_row: Optional[TypedHeaderRow] = None,
    composer_id: Optional[str] = None,
) -> bool:
    if typed_row is not None and typed_row.is_subagent:
        return True
    for src in sources:
        if not isinstance(src, dict):
            continue
        if _as_flag(src.get("isSubagent")):
            return True
        if _as_flag(src.get("isBestOfNSubcomposer")):
            return True
    cid = composer_id
    if cid is None:
        for src in sources:
            if isinstance(src, dict) and src.get("composerId"):
                cid = src["composerId"]
                break
    if _task_like_composer_id(cid) and not _explicitly_top_level(
        *sources, typed_row=typed_row
    ):
        return True
    return False


def archived_flag(*sources: Optional[dict], typed_row: Optional[TypedHeaderRow] = None) -> bool:
    if typed_row is not None and typed_row.is_archived:
        return True
    for src in sources:
        if not isinstance(src, dict):
            continue
        if "isArchived" in src and src["isArchived"] is not None:
            return _as_flag(src["isArchived"])
    return False


def synthesize_header_value(
    composer_id: str,
    *,
    workspace_identifier: dict,
    composer_data: Optional[dict] = None,
    typed_header: Optional[dict] = None,
    legacy_header: Optional[dict] = None,
) -> dict:
    """Minimum viable typed ``value``. Prefer typed, then legacy, then composerData.

    Does not invent UI-derived fields (subtitle, line counts, tracked repos).
    ``forceMode`` is not assumed to be a 1:1 copy of composerData.
    """
    cd = composer_data if isinstance(composer_data, dict) else {}
    created_at = _as_int(
        _pick("createdAt", typed_header, legacy_header, cd, default=0),
        0,
    )
    is_archived = archived_flag(typed_header, legacy_header, cd)
    is_subagent = looks_like_subagent(
        typed_header, legacy_header, cd, composer_id=composer_id
    )
    header = {
        "type": "head",
        "composerId": composer_id,
        "createdAt": created_at,
        "unifiedMode": _pick(
            "unifiedMode", typed_header, legacy_header, cd, default="agent"
        ),
        "forceMode": _pick("forceMode", typed_header, legacy_header, cd, default=""),
        "hasUnreadMessages": False,
        "totalLinesAdded": 0,
        "totalLinesRemoved": 0,
        "hasBlockingPendingActions": False,
        "isDraft": False,
        "isWorktree": False,
        "worktreeStartedReadOnly": False,
        "isSpec": False,
        "isProject": False,
        "isBestOfNSubcomposer": _as_flag(
            _pick("isBestOfNSubcomposer", typed_header, legacy_header, cd, default=False)
        ),
        "isArchived": is_archived,
        "isSubagent": is_subagent,
        "numSubComposers": 0,
        "referencedPlans": [],
        "trackedGitRepos": [],
        "workspaceIdentifier": workspace_identifier,
    }
    for key in _OPTIONAL_HEADER_KEYS:
        if key in header:
            continue
        value = _pick(key, typed_header, legacy_header, cd)
        if value is not None:
            header[key] = value
    return header


def register_current(
    cdb: Any,
    composer_id: str,
    workspace_id: str,
    composer_data: dict,
    *,
    workspace_identifier: dict,
    legacy_header: Optional[dict] = None,
    update_stable: bool = False,
    known_checkpoint_time: Optional[int] = None,
    require_active: bool = True,
) -> str:
    """Insert or preserve a typed row. Never ``INSERT OR REPLACE``.

    Re-reads the live typed row and composerData before writing.
    Returns ``inserted``, ``preserved``, or ``skipped_inactive``.
    Raises ``MisregisteredHeaderError`` / ``TypedSchemaError``.
    """
    conn = cdb._get_write_conn()
    if assert_typed_schema_writable(conn) != TypedSchemaStatus.USABLE:
        raise TypedSchemaError("composerHeaders table is not writable")
    if require_active:
        live = cdb.get_json(f"composerData:{composer_id}")
        from . import syncstate

        if syncstate.classify_local_payload(live) != syncstate.LocalPresence.ACTIVE:
            return "skipped_inactive"
        composer_data = live if isinstance(live, dict) else composer_data

    existing = get_typed_row(conn, composer_id)
    if existing is None:
        try:
            _insert_synthetic(
                conn,
                composer_id,
                workspace_id,
                composer_data,
                workspace_identifier=workspace_identifier,
                legacy_header=legacy_header,
                known_checkpoint_time=known_checkpoint_time,
            )
        except sqlite3.IntegrityError:
            if cdb.autocommit:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
            existing = get_typed_row(conn, composer_id)
            if existing is None:
                raise
            return _resolve_existing(
                existing,
                composer_id,
                workspace_id,
                composer_data,
                workspace_identifier=workspace_identifier,
                update_stable=update_stable,
                known_checkpoint_time=known_checkpoint_time,
                conn=conn,
                cdb=cdb,
            )
        if cdb.autocommit:
            conn.commit()
        return "inserted"
    return _resolve_existing(
        existing,
        composer_id,
        workspace_id,
        composer_data,
        workspace_identifier=workspace_identifier,
        update_stable=update_stable,
        known_checkpoint_time=known_checkpoint_time,
        conn=conn,
        cdb=cdb,
    )


def delete_typed_headers(cdb: Any, composer_ids: list[str]) -> int:
    """Delete typed rows. Returns the number of rows removed."""
    if not composer_ids:
        return 0
    conn = cdb._get_write_conn()
    status = assert_typed_schema_writable(conn)
    if status != TypedSchemaStatus.USABLE:
        return 0
    deleted = 0
    for cid in composer_ids:
        cur = conn.execute(
            f"DELETE FROM {TABLE_NAME} WHERE composerId = ?", (cid,)
        )
        deleted += cur.rowcount
    if cdb.autocommit:
        conn.commit()
    return deleted


def _resolve_existing(
    existing: TypedHeaderRow,
    composer_id: str,
    workspace_id: str,
    composer_data: dict,
    *,
    workspace_identifier: dict,
    update_stable: bool,
    known_checkpoint_time: Optional[int],
    conn: sqlite3.Connection,
    cdb: Any,
) -> str:
    binding = classify_workspace_binding(
        typed_workspace_id=existing.workspace_id,
        target_workspace_id=workspace_id,
        typed_identifier=existing.header.get("workspaceIdentifier"),
        target_identifier=workspace_identifier,
    )
    if binding != WorkspaceBinding.SAME:
        raise MisregisteredHeaderError(
            composer_id,
            existing.workspace_id,
            workspace_id,
            binding=binding,
        )
    if update_stable:
        _update_stable_fields(
            conn,
            existing,
            composer_data,
            known_checkpoint_time=known_checkpoint_time,
        )
        if cdb.autocommit:
            conn.commit()
    return "preserved"


def _row_from_tuple(row: tuple) -> TypedHeaderRow:
    value = row[8]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    elif value is None:
        value = ""
    return TypedHeaderRow(
        composer_id=row[0],
        workspace_id=row[1] or "",
        created_at=_as_int(row[2]),
        last_updated_at=_as_int(row[3]),
        is_archived=_as_int(row[4], 0) or 0,
        is_subagent=_as_int(row[5], 0) or 0,
        recency=_as_int(row[6]),
        checkpoint_at=_as_int(row[7]),
        value=value,
    )


def _insert_synthetic(
    conn: sqlite3.Connection,
    composer_id: str,
    workspace_id: str,
    composer_data: dict,
    *,
    workspace_identifier: dict,
    legacy_header: Optional[dict],
    known_checkpoint_time: Optional[int],
) -> None:
    header = synthesize_header_value(
        composer_id,
        workspace_identifier=workspace_identifier,
        composer_data=composer_data,
        legacy_header=legacy_header,
    )
    created_at = _as_int(header.get("createdAt"), 0) or 0
    last_updated = _as_int(
        _pick("lastUpdatedAt", legacy_header, composer_data)
    )
    recency = last_updated if last_updated is not None else created_at
    checkpoint_at = _as_int(known_checkpoint_time)
    if checkpoint_at is None:
        checkpoint_at = _as_int(
            _pick(
                "conversationCheckpointLastUpdatedAt",
                legacy_header,
                composer_data,
            )
        )
    is_archived = 1 if archived_flag(legacy_header, composer_data) else 0
    is_subagent = 1 if looks_like_subagent(
        legacy_header, composer_data, composer_id=composer_id
    ) else 0
    conn.execute(
        f"INSERT INTO {TABLE_NAME} ("
        "composerId, workspaceId, createdAt, lastUpdatedAt, "
        "isArchived, isSubagent, recency, checkpointAt, value"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            composer_id,
            workspace_id,
            created_at,
            last_updated,
            is_archived,
            is_subagent,
            recency,
            checkpoint_at,
            json.dumps(header, separators=(",", ":")),
        ),
    )


def _update_stable_fields(
    conn: sqlite3.Connection,
    existing: TypedHeaderRow,
    composer_data: dict,
    *,
    known_checkpoint_time: Optional[int],
) -> None:
    last_updated = _as_int(composer_data.get("lastUpdatedAt"), existing.last_updated_at)
    recency = last_updated if last_updated is not None else existing.created_at
    checkpoint_at = _as_int(known_checkpoint_time, existing.checkpoint_at)
    conn.execute(
        f"UPDATE {TABLE_NAME} SET lastUpdatedAt = ?, recency = ?, checkpointAt = ? "
        "WHERE composerId = ?",
        (last_updated, recency, checkpoint_at, existing.composer_id),
    )
