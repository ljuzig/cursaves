"""Semantic sync classification: prefix compare, not message counts.

Authoritative relation is a prefix comparison of per-message semantic
unit hashes. ``messageCount`` is never used to decide a write.

Lookup is origin-aware: ``(project_identifier, composer_id)``, with the
Commit 1 SSH rules (exact host + POSIX-normalized remote path). There is
no cross-project or cross-host fallback.

A regenerable on-disk cache stores snapshot and local fingerprints so
legacy gzip and local semantic parses are not repeated every command.
The cache is never authoritative when missing or stale. Snapshot files
are never rewritten.

A header with no bubble body is a first-class tombstone, not an error.
``SEMANTIC_DIGEST_VERSION`` 4 keeps that bubble-state model and adds
legacy monolithic ``composerData.conversation`` units, namespaced so
they cannot collide with modern header/bubble hashes. Older sidecar
digests are never trusted; the snapshot is deep-read and the
regenerable cache stores the current version.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Collection, Iterator, Optional

from . import db, dblock, export, importer, paths, typed_headers


class SyncRelation(str, Enum):
    UP_TO_DATE = "up_to_date"
    LOCAL_AHEAD = "local_ahead"
    BEHIND = "behind"
    DIVERGED = "diverged"
    NEVER_PUSHED = "never_pushed"
    UNKNOWN = "unknown"


class LocalPresence(str, Enum):
    """Workspace registration vs an exportable local conversation."""

    ACTIVE = "active"
    EMPTY = "empty"
    DANGLING = "dangling"
    INVALID = "invalid"


NOT_LOCAL = "not_local"

# Top-level cursaves envelope — never part of conversation meaning.
_NON_SEMANTIC_SNAPSHOT_KEYS = frozenset({
    "exportedAt",
    "sourceMachine",
    "sourceHost",
    "sourceProjectPath",
    "projectIdentifier",
})

# Proven transport-only fields, stripped only at header/bubble roots.
_TOP_LEVEL_TRANSPORT_KEYS = frozenset({
    "createdAt",
    "lastUpdatedAt",
    "exportedAt",
    "checkpointId",
    "serverBubbleId",
})

_EXPLICIT_BLOB_KEYS = frozenset({"contentHash", "blobId", "contentHashes"})

_CACHE_VERSION = 5
LOCAL_PAYLOAD_VERSION = 1
SEMANTIC_DIGEST_VERSION = 4
_LEGACY_CONVERSATION_SCHEMA = "legacy-conversation"
_HASH_CHUNK = 1024 * 1024


class ClassifyError(Exception):
    """Conversation cannot be classified safely."""


class SyncPreflightStale(Exception):
    """Cursor or snapshot state changed after the read-only preflight."""


@dataclass
class OpCounts:
    sqlite_backups: int = 0
    read_copy_global: int = 0
    read_copy_workspace: int = 0
    live_epochs: int = 0
    backup_epochs: int = 0
    snapshot_directory_scans: int = 0
    deep_snapshot_reads: int = 0
    legacy_snapshot_decompressions: int = 0
    full_local_exports: int = 0
    cursor_write_connections: int = 0
    local_semantic_rehashes: int = 0
    local_inventory_json_parses: int = 0
    local_composer_json_parses: int = 0
    pull_target_scans: int = 0
    pull_candidates: int = 0
    staged_snapshots: int = 0
    safety_global_backups: int = 0
    safety_workspace_backups: int = 0
    write_connections_opened: int = 0
    imports_attempted: int = 0
    imports_completed: int = 0
    cursor_running_checks: int = 0
    snapshot_content_hashes: int = 0
    local_guard_checks: int = 0
    local_guard_skips: int = 0


_counts = OpCounts()


def reset_op_counts() -> None:
    global _counts
    _counts = OpCounts()


def op_counts() -> OpCounts:
    return _counts


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _as_bytes(val: Any) -> bytes:
    if val is None:
        return b""
    if isinstance(val, bytes):
        return val
    if isinstance(val, str):
        return val.encode("utf-8")
    return _canonical_json(val).encode("utf-8")


def _hash_field(hasher: Any, key: bytes, value: bytes) -> None:
    hasher.update(len(key).to_bytes(8, "big"))
    hasher.update(key)
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def _hash_file_field(hasher: Any, key: bytes, path: Path) -> None:
    hasher.update(len(key).to_bytes(8, "big"))
    hasher.update(key)
    hasher.update(path.stat().st_size.to_bytes(8, "big"))
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK)
            if not chunk:
                break
            hasher.update(chunk)


def _normalize_unit_object(obj: Any, *, top: bool = False) -> Any:
    """Include-by-default. Drop only proven root-level transport keys."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if top and key in _TOP_LEVEL_TRANSPORT_KEYS:
                continue
            out[key] = _normalize_unit_object(value, top=False)
        return out
    if isinstance(obj, (list, tuple)):
        return [_normalize_unit_object(v, top=False) for v in obj]
    return obj


def _walk_strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _walk_strings(value)


def _blob_bytes(value: Any) -> bytes:
    return _as_bytes(value)


def _explicit_blob_refs(obj: Any) -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _EXPLICIT_BLOB_KEYS:
                if isinstance(value, str):
                    found.append(value)
                elif isinstance(value, list):
                    found.extend(v for v in value if isinstance(v, str))
            else:
                found.extend(_explicit_blob_refs(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found.extend(_explicit_blob_refs(value))
    return found


def _referenced_blob_ids(payload: Any, available: set[str]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for text in list(_walk_strings(payload)) + _explicit_blob_refs(payload):
        if text in available and text not in seen:
            seen.add(text)
            found.append(text)
    return found


def _required_blob_ids(payload: Any, available: set[str]) -> list[str]:
    required: list[str] = []
    seen: set[str] = set()
    for text in _explicit_blob_refs(payload):
        if text not in seen:
            seen.add(text)
            required.append(text)
    for text in _walk_strings(payload):
        if text in available and text not in seen:
            seen.add(text)
            required.append(text)
    return required


def _semantic_header(header: dict) -> dict:
    """Header fields that participate in the digest.

    ``grouping`` is derived UI/tool-render metadata. Only this key is
    dropped — other header fields stay semantic.
    """
    cleaned = dict(header)
    cleaned.pop("grouping", None)
    return cleaned


def _unit_payload(header: dict, bubble: Optional[dict], blobs: dict[str, Any]) -> dict[str, Any]:
    """Hash header, explicit bubble state, and blobs.

    A missing bubble is a tombstone keyed by ``bubbleId``, not a generic
    empty object — so missing↔missing compares and missing↔present diverges.
    Header fields and header-referenced blobs stay in the unit either way.
    """
    header = _semantic_header(header)
    bid = header.get("bubbleId")
    if not bid:
        raise ClassifyError("header is missing bubbleId")
    if bubble is None:
        bubble_part: dict[str, Any] = {"state": "missing", "bubbleId": bid}
        ref_payload: Any = header
    else:
        bubble_part = {
            "state": "present",
            "value": _normalize_unit_object(bubble, top=True),
        }
        ref_payload = (header, bubble)
    payload: dict[str, Any] = {
        "header": _normalize_unit_object(header, top=True),
        "bubble": bubble_part,
    }
    refs = _referenced_blob_ids(ref_payload, set(blobs))
    if refs:
        payload["blobs"] = {
            ref: _sha256_bytes(_blob_bytes(blobs[ref])) for ref in sorted(refs)
        }
    return payload


def unit_hash(header: dict, bubble: Optional[dict], blobs: dict[str, Any]) -> str:
    return _sha256_text(_canonical_json(_unit_payload(header, bubble, blobs)))


def _strict_select_row(
    cdb: "db.CursorDB", key: str, table: str = "cursorDiskKV"
) -> tuple[bool, Any]:
    """Return ``(row_present, parsed)``. JSON null is ``(True, None)``."""
    try:
        conn = cdb._reader_conn()
        row = conn.execute(
            f"SELECT value FROM {table} WHERE key = ?",
            (key,),
        ).fetchone()
    except Exception as exc:
        raise ClassifyError(f"failed to read {key}: {exc}") from exc
    if row is None:
        return False, None
    raw = row[0]
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ClassifyError(f"corrupt JSON at {key}: {exc}") from exc
    return True, parsed


def _strict_select_json(cdb: "db.CursorDB", key: str, table: str = "cursorDiskKV") -> Optional[Any]:
    """Parse a cell. None means the key is absent or contains JSON null."""
    present, parsed = _strict_select_row(cdb, key, table)
    return parsed if present else None


def classify_local_payload(data: Any) -> LocalPresence:
    """Classify a present ``composerData`` cell.

    ``None`` is JSON null, not an absent row: that is INVALID. Callers
    that see a missing SQL row must return DANGLING themselves.

    Two message lists are recognized: modern
    ``fullConversationHeadersOnly`` and, only when that key is absent,
    legacy monolithic ``conversation``. Ambiguous or malformed payloads
    stay INVALID.
    """
    if data is None or not isinstance(data, dict):
        return LocalPresence.INVALID
    if "fullConversationHeadersOnly" in data:
        headers = data["fullConversationHeadersOnly"]
        if not isinstance(headers, list):
            return LocalPresence.INVALID
        return LocalPresence.EMPTY if not headers else LocalPresence.ACTIVE
    if "conversation" in data:
        conversation = data["conversation"]
        if not isinstance(conversation, list):
            return LocalPresence.INVALID
        return LocalPresence.EMPTY if not conversation else LocalPresence.ACTIVE
    return LocalPresence.INVALID


def semantic_message_count(data: Any) -> int:
    """Display count using the same schema precedence as classification."""
    if not isinstance(data, dict):
        return 0
    if "fullConversationHeadersOnly" in data:
        headers = data["fullConversationHeadersOnly"]
        return len(headers) if isinstance(headers, list) else 0
    if "conversation" in data:
        conversation = data["conversation"]
        return len(conversation) if isinstance(conversation, list) else 0
    return 0


def classify_local_conversation(
    session: "SyncReadSession", composer_id: str
) -> LocalPresence:
    """Workspace CID → ACTIVE / EMPTY / DANGLING / INVALID.

    Headers or legacy ``conversation`` items present with missing bubble
    bodies stay ACTIVE. Only a true empty recognized list is EMPTY.
    Read/JSON errors and JSON null are INVALID.
    """
    return session.local_presence(composer_id)


def is_inactive_registration(presence: LocalPresence, has_snapshot: bool) -> bool:
    """EMPTY/DANGLING with no snapshot: not a local conversation to export."""
    return presence in (LocalPresence.EMPTY, LocalPresence.DANGLING) and not has_snapshot


def conversation_digest(unit_hashes: list[str]) -> str:
    return "sha256:" + _sha256_text("\n".join(unit_hashes))


def compare_unit_hashes(local: list[str], remote: list[str]) -> SyncRelation:
    if local == remote:
        return SyncRelation.UP_TO_DATE
    if len(local) > len(remote) and local[: len(remote)] == remote:
        return SyncRelation.LOCAL_AHEAD
    if len(remote) > len(local) and remote[: len(local)] == local:
        return SyncRelation.BEHIND
    return SyncRelation.DIVERGED


def _headers(composer_data: Optional[dict]) -> list[dict]:
    if not composer_data:
        return []
    headers = composer_data.get("fullConversationHeadersOnly") or []
    return [h for h in headers if isinstance(h, dict)]


def _legacy_messages(conversation: Any) -> list[dict]:
    """Ordered legacy ``conversation`` items. Fail closed on bad entries."""
    if not isinstance(conversation, list):
        raise ClassifyError("legacy conversation is not a list")
    messages: list[dict] = []
    for index, item in enumerate(conversation):
        if not isinstance(item, dict):
            raise ClassifyError(
                f"legacy conversation[{index}] is not an object"
            )
        if not item.get("bubbleId"):
            raise ClassifyError(
                f"legacy conversation[{index}] is missing bubbleId"
            )
        messages.append(item)
    return messages


def _legacy_unit_hash(message: dict, blobs: dict[str, Any]) -> str:
    """One namespaced unit hash for a legacy monolithic message."""
    normalized = _normalize_unit_object(message, top=True)
    refs = _required_blob_ids(normalized, set(blobs))
    for ref in refs:
        if ref not in blobs:
            raise ClassifyError(f"referenced blob missing: {ref}")
    payload: dict[str, Any] = {
        "schema": _LEGACY_CONVERSATION_SCHEMA,
        "message": normalized,
    }
    hashed_refs = _referenced_blob_ids(normalized, set(blobs))
    if hashed_refs:
        payload["blobs"] = {
            ref: _sha256_bytes(_blob_bytes(blobs[ref]))
            for ref in sorted(hashed_refs)
        }
    return _sha256_text(_canonical_json(payload))


def _legacy_unit_hashes(conversation: Any, blobs: dict[str, Any]) -> list[str]:
    return [_legacy_unit_hash(message, blobs) for message in _legacy_messages(conversation)]


def _optional_mapping(value: Any, name: str) -> Optional[dict]:
    """Absent/null is OK. A present non-object is unreadable, not a tombstone."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ClassifyError(f"{name} is not an object")
    return value


def _coerce_bubble(bubble: Any, bubble_id: str) -> Optional[dict]:
    """Present dict, None tombstone, or ClassifyError for corrupt bodies."""
    if bubble is None:
        return None
    if not isinstance(bubble, dict):
        raise ClassifyError(f"referenced bubble is not an object: {bubble_id}")
    return bubble


def _bubble_from_snapshot(snapshot: dict, bubble_id: str) -> Optional[dict]:
    """Return the bubble body, or None when the header is a tombstone.

    A missing key or JSON null on the primary map means “no body here”:
    try ``conversationMap`` next. A present non-object is unreadable
    and must not be hashed as a tombstone. Corrupt containers are
    UNKNOWN, not empty maps.
    """
    entries = _optional_mapping(snapshot.get("bubbleEntries"), "bubbleEntries")
    if entries is not None and bubble_id in entries:
        bubble = _coerce_bubble(entries[bubble_id], bubble_id)
        if bubble is not None:
            return bubble
    composer = snapshot["composerData"]
    cmap = _optional_mapping(composer.get("conversationMap"), "conversationMap")
    if cmap is not None and bubble_id in cmap:
        return _coerce_bubble(cmap[bubble_id], bubble_id)
    return None


def snapshot_unit_hashes(snapshot: dict) -> list[str]:
    """Build unit hashes from a parsed snapshot dict (legacy-safe)."""
    if not isinstance(snapshot, dict):
        raise ClassifyError("snapshot is not an object")
    composer = snapshot.get("composerData")
    if composer is None:
        raise ClassifyError("snapshot is missing composerData")
    if not isinstance(composer, dict):
        raise ClassifyError("composerData is not an object")

    blobs = snapshot.get("contentBlobs") or {}
    if not isinstance(blobs, dict):
        blobs = {}
    if "fullConversationHeadersOnly" in composer:
        raw_headers = composer["fullConversationHeadersOnly"]
        if not isinstance(raw_headers, list):
            raise ClassifyError("fullConversationHeadersOnly is not a list")
        headers = [h for h in raw_headers if isinstance(h, dict)]
        hashes: list[str] = []
        for header in headers:
            bid = header.get("bubbleId")
            if not bid:
                raise ClassifyError("header is missing bubbleId")
            bubble = _bubble_from_snapshot(snapshot, bid)
            sem_header = _semantic_header(header)
            ref_payload: Any = sem_header if bubble is None else (sem_header, bubble)
            refs = _required_blob_ids(ref_payload, set(blobs))
            for ref in refs:
                if ref not in blobs:
                    raise ClassifyError(f"referenced blob missing: {ref}")
            hashes.append(unit_hash(header, bubble, blobs))
        return hashes
    if "conversation" in composer:
        return _legacy_unit_hashes(composer.get("conversation"), blobs)
    raise ClassifyError("composerData has no recognized message list")


def snapshot_semantic_digest(snapshot: dict) -> str:
    return conversation_digest(snapshot_unit_hashes(snapshot))


def _file_identity(path: Path) -> tuple[int, int, int]:
    st = path.stat()
    ctime_ns = getattr(st, "st_ctime_ns", 0)
    return (st.st_size, st.st_mtime_ns, ctime_ns)


def snapshot_source_identity(
    snapshot_path: Path, meta: Optional[dict] = None
) -> tuple[tuple[int, int, int], ...]:
    """Identity of every file ``read_snapshot_file`` would consult."""
    parts: list[tuple[int, int, int]] = []
    for path in importer.snapshot_component_files(snapshot_path, meta):
        try:
            parts.append(_file_identity(path))
        except OSError:
            continue
    return tuple(parts) if parts else ((0, 0, 0),)


def snapshot_content_digest(
    snapshot_path: Path, meta: Optional[dict] = None
) -> str:
    """Hash on-disk compressed bytes of main + shards. No decompress or JSON."""
    _counts.snapshot_content_hashes += 1
    hasher = hashlib.sha256()
    for path in importer.snapshot_component_files(snapshot_path, meta):
        _hash_file_field(hasher, path.name.encode("utf-8"), path)
    return "sha256:" + hasher.hexdigest()


def _identity_as_lists(
    identity: tuple[tuple[int, int, int], ...],
) -> list[list[int]]:
    return [list(part) for part in identity]


def _identity_from_stored(stored: Any) -> Optional[tuple[tuple[int, int, int], ...]]:
    if not isinstance(stored, list) or not stored:
        return None
    parts: list[tuple[int, int, int]] = []
    for item in stored:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return None
        try:
            parts.append((int(item[0]), int(item[1]), int(item[2])))
        except (TypeError, ValueError):
            return None
    return tuple(parts)


def _cache_file() -> Path:
    return paths.get_cache_dir() / "sync-semantics.json"


def _empty_cache() -> dict:
    return {"version": _CACHE_VERSION, "snapshots": {}, "local": {}}


class SemanticsCache:
    """Optional regenerable cache. Invalid/missing entries are recomputed."""

    def __init__(self):
        self._data = _empty_cache()
        self._dirty = False
        path = _cache_file()
        if path.exists():
            try:
                loaded = json.loads(path.read_text())
                if loaded.get("version") == _CACHE_VERSION:
                    loaded.setdefault("snapshots", {})
                    loaded.setdefault("local", {})
                    self._data = loaded
            except (json.JSONDecodeError, OSError):
                self._data = _empty_cache()

    @staticmethod
    def snapshot_key(project_identifier: str, composer_id: str) -> str:
        return f"{project_identifier}|{composer_id}"

    def get_snapshot(
        self,
        key: str,
        identity: tuple[tuple[int, int, int], ...],
    ) -> Optional[dict]:
        rec = self._data["snapshots"].get(key)
        if not isinstance(rec, dict) or not rec.get("semanticDigest"):
            return None
        if rec.get("semanticDigestVersion") != SEMANTIC_DIGEST_VERSION:
            return None
        if _identity_from_stored(rec.get("sourceIdentity")) != identity:
            return None
        return rec

    def put_snapshot(
        self,
        key: str,
        identity: tuple[tuple[int, int, int], ...],
        digest: str,
    ) -> None:
        self._data["snapshots"][key] = {
            "sourceIdentity": _identity_as_lists(identity),
            "semanticDigest": digest,
            "semanticDigestVersion": SEMANTIC_DIGEST_VERSION,
        }
        self._dirty = True

    def get_local(self, composer_id: str) -> Optional[dict]:
        rec = self._data["local"].get(composer_id)
        if not isinstance(rec, dict):
            return None
        if not rec.get("rowFingerprint") or not rec.get("semanticDigest"):
            return None
        if rec.get("semanticDigestVersion") != SEMANTIC_DIGEST_VERSION:
            return None
        if rec.get("localPayloadVersion") != LOCAL_PAYLOAD_VERSION:
            return None
        if "blobRefs" not in rec or not rec.get("blobFingerprint"):
            return None
        return rec

    def put_local(
        self,
        composer_id: str,
        row_fp: str,
        blob_refs: list[str],
        blob_fp: str,
        digest: str,
    ) -> None:
        self._data["local"][composer_id] = {
            "rowFingerprint": row_fp,
            "blobRefs": list(blob_refs),
            "blobFingerprint": blob_fp,
            "semanticDigest": digest,
            "semanticDigestVersion": SEMANTIC_DIGEST_VERSION,
            "localPayloadVersion": LOCAL_PAYLOAD_VERSION,
        }
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        path = _cache_file()
        tmp_name = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix="sync-semantics-",
                suffix=".tmp",
                dir=str(path.parent),
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._data, handle, separators=(",", ":"))
            os.replace(tmp_name, path)
            tmp_name = None
            self._dirty = False
        except (OSError, TypeError, ValueError):
            self._dirty = False
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass


@dataclass
class SnapshotRecord:
    composer_id: str
    path: Path
    meta: dict
    project_dir: Path
    project_identifier: str
    identity: tuple[tuple[int, int, int], ...] = ((0, 0, 0),)
    invalid_origin: bool = False


@dataclass
class SnapshotIndex:
    """One-pass origin-aware index. Parsed JSON is not retained."""

    by_key: dict[tuple[str, str], SnapshotRecord] = field(default_factory=dict)
    _remote_units: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    _remote_digest: dict[tuple[str, str], str] = field(default_factory=dict)
    cache: SemanticsCache = field(default_factory=SemanticsCache)
    scoped_project_identifier: Optional[str] = None

    @classmethod
    def build(
        cls,
        snapshots_dir: Optional[Path] = None,
        project_identifier: Optional[str] = None,
    ) -> "SnapshotIndex":
        _counts.snapshot_directory_scans += 1
        if project_identifier:
            _counts.pull_target_scans += 1
        root = snapshots_dir if snapshots_dir is not None else paths.get_snapshots_dir()
        index = cls(scoped_project_identifier=project_identifier)
        if not root.exists():
            return index
        if project_identifier:
            project_dir = root / project_identifier
            dirs = [project_dir] if project_dir.is_dir() else []
        else:
            dirs = sorted(p for p in root.iterdir() if p.is_dir())
        for project_dir in dirs:
            index._index_project_dir(project_dir)
        return index

    @classmethod
    def build_for_project(
        cls,
        project_identifier: str,
        snapshots_dir: Optional[Path] = None,
    ) -> "SnapshotIndex":
        """Index only ``snapshots/<project_identifier>/``.

        Records the bucket name on ``scoped_project_identifier`` so a
        targeted planner can look up snapshots in a legacy directory
        (e.g. ``nixos/``) instead of the canonical SSH identity.
        """
        return cls.build(
            snapshots_dir=snapshots_dir,
            project_identifier=project_identifier,
        )

    def _index_project_dir(self, project_dir: Path) -> None:
        for snap_path in importer.list_snapshot_files(project_dir):
            stem = snap_path.name
            if stem.endswith(".json.gz"):
                stem = stem[: -len(".json.gz")]
            elif stem.endswith(".json"):
                stem = stem[: -len(".json")]
            meta_path = snap_path.parent / f"{stem}.meta.json"
            meta: dict = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except (json.JSONDecodeError, OSError):
                    meta = {}
            cid = meta.get("composerId") or stem
            if not cid:
                continue
            project_id = project_dir.name
            meta_project_id = meta.get("projectIdentifier")
            invalid_origin = bool(meta_project_id and meta_project_id != project_id)
            identity = snapshot_source_identity(snap_path, meta)
            rec = SnapshotRecord(
                composer_id=cid,
                path=snap_path,
                meta=meta,
                project_dir=project_dir,
                project_identifier=project_id,
                identity=identity,
                invalid_origin=invalid_origin,
            )
            self.by_key[(project_id, cid)] = rec

    def get(
        self,
        composer_id: str,
        project_identifier: Optional[str] = None,
        *,
        source_host: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> Optional[SnapshotRecord]:
        if not project_identifier:
            return None
        rec = self.by_key.get((project_identifier, composer_id))
        if rec is None:
            return None
        if source_host:
            if rec.meta.get("sourceHost") != source_host:
                return None
            if source_path:
                got = paths.normalize_origin_path(
                    rec.meta.get("sourceProjectPath") or "", source_host=source_host
                )
                want = paths.normalize_origin_path(source_path, source_host=source_host)
                if got != want:
                    return None
        return rec

    def remote_semantics(self, rec: SnapshotRecord) -> tuple[Optional[list[str]], Optional[str]]:
        """Return (unit_hashes or None, digest or None). JSON is discarded."""
        if rec.invalid_origin:
            raise ClassifyError("snapshot projectIdentifier does not match directory")
        key = (rec.project_identifier, rec.composer_id)
        if key in self._remote_digest:
            return self._remote_units.get(key), self._remote_digest[key]

        cache_key = SemanticsCache.snapshot_key(
            rec.project_identifier, rec.composer_id
        )
        cached = self.cache.get_snapshot(cache_key, rec.identity)
        if cached:
            digest = cached["semanticDigest"]
            self._remote_digest[key] = digest
            return None, digest

        if self._sidecar_bound_to_body(rec):
            digest = rec.meta["semanticDigest"]
            self._remote_digest[key] = digest
            self.cache.put_snapshot(cache_key, rec.identity, digest)
            return None, digest

        return self._load_remote_from_file(rec)

    @staticmethod
    def _sidecar_bound_to_body(rec: SnapshotRecord) -> bool:
        """Trust sidecar semanticDigest only when on-disk bytes still match."""
        meta_digest = rec.meta.get("semanticDigest")
        meta_ver = rec.meta.get("semanticDigestVersion")
        content = rec.meta.get("snapshotContentDigest")
        if not (meta_digest and content and meta_ver == SEMANTIC_DIGEST_VERSION):
            return False
        try:
            return snapshot_content_digest(rec.path, rec.meta) == content
        except OSError:
            return False

    def _load_remote_from_file(self, rec: SnapshotRecord) -> tuple[list[str], str]:
        key = (rec.project_identifier, rec.composer_id)
        _counts.deep_snapshot_reads += 1
        _counts.legacy_snapshot_decompressions += 1
        try:
            data = importer.read_snapshot_file(rec.path, rec.meta)
        except Exception as exc:
            raise ClassifyError(f"unreadable snapshot: {exc}") from exc
        if not isinstance(data, dict):
            raise ClassifyError("snapshot is not an object")
        hashes = snapshot_unit_hashes(data)
        digest = conversation_digest(hashes)
        del data
        self._remote_units[key] = hashes
        self._remote_digest[key] = digest
        cache_key = SemanticsCache.snapshot_key(
            rec.project_identifier, rec.composer_id
        )
        self.cache.put_snapshot(cache_key, rec.identity, digest)
        return hashes, digest

    def remote_unit_hashes(self, rec: SnapshotRecord) -> list[str]:
        units, digest = self.remote_semantics(rec)
        if units is not None:
            return units
        units, _ = self._load_remote_from_file(rec)
        return units

    def ensure_remote_readable(self, rec: SnapshotRecord) -> None:
        """Validate a snapshot-only candidate before treating it as behind.

        A prior successful read (cache hit on source identity) is enough.
        Meta digest alone is not: the file may have become unreadable.
        """
        if rec.invalid_origin:
            raise ClassifyError("snapshot projectIdentifier does not match directory")
        cache_key = SemanticsCache.snapshot_key(
            rec.project_identifier, rec.composer_id
        )
        cached = self.cache.get_snapshot(cache_key, rec.identity)
        if cached:
            self._remote_digest[(rec.project_identifier, rec.composer_id)] = cached[
                "semanticDigest"
            ]
            return
        if self._sidecar_bound_to_body(rec):
            self._remote_digest[(rec.project_identifier, rec.composer_id)] = rec.meta[
                "semanticDigest"
            ]
            self.cache.put_snapshot(cache_key, rec.identity, rec.meta["semanticDigest"])
            return
        self._load_remote_from_file(rec)


_COMPOSER_ABSENT = object()
_COMPOSER_UNREADABLE = object()


class SyncReadSession:
    """One consistent read-only SQLite view of the global Cursor DB.

    Owns a single ``ReadEpoch`` for the command's read-only phase, plus
    the header map and per-CID presence/payload derived from that view.
    Not a process-global cache keyed on ``db_path``.
    """

    def __init__(self, global_db: Optional[Path] = None):
        self._global_path = global_db if global_db is not None else paths.get_global_db_path()
        self._epoch: Optional[db.ReadEpoch] = None
        self._cdb: Optional[db.CursorDB] = None
        self._local_hashes: dict[str, list[str]] = {}
        self._local_digest: dict[str, str] = {}
        self._row_fp: dict[str, str] = {}
        self._blob_ids: Optional[set[str]] = None
        self._inventory_complete = False
        self._inventory_attempted = False
        self._inventory_scope: Optional[set[str]] = None
        self._targeted_absent: set[str] = set()
        self._composer_cells: dict[str, Any] = {}
        self._presence: dict[str, LocalPresence] = {}
        self._composer_headers: Optional[list[dict]] = None
        self._headers_map: Optional[dict[str, list[dict]]] = None
        self._typed_loaded = False
        self._typed_table_exists = False
        self._typed_schema = typed_headers.TypedSchemaStatus.ABSENT
        self._typed_catalog: dict[str, typed_headers.TypedHeaderRow] = {}
        self.cache = SemanticsCache()

    def __enter__(self) -> "SyncReadSession":
        if self._global_path.exists():
            self._epoch = db.ReadEpoch(self._global_path)
            self._epoch.__enter__()
            self._cdb = db.CursorDB(self._global_path, read_epoch=self._epoch)
        return self

    def prepare_inventory(self, composer_ids: Optional[Collection[str]]) -> None:
        """Choose full scan (``None``) or a targeted CID set. Caller decides."""
        self._inventory_scope = None if composer_ids is None else set(composer_ids)
        self._ensure_inventory()

    def _ensure_inventory(self, composer_ids: Optional[Collection[str]] = None) -> None:
        """Scan composer/bubble bytes only when a digest fast-path needs it."""
        if self._inventory_complete or self._cdb is None:
            return
        ids = set(composer_ids) if composer_ids is not None else self._inventory_scope
        if ids is not None:
            self._load_inventory_targeted(ids)
            return
        if self._inventory_attempted:
            return
        self._inventory_attempted = True
        self._load_inventory()

    def __exit__(self, *args) -> None:
        self.cache.flush()
        if self._cdb is not None:
            self._cdb.close()
            self._cdb = None
        if self._epoch is not None:
            self._epoch.close()
            self._epoch = None

    @property
    def cdb(self) -> Optional[db.CursorDB]:
        return self._cdb

    @property
    def epoch(self) -> Optional[db.ReadEpoch]:
        return self._epoch

    def composer_headers(self) -> list[dict]:
        if self._composer_headers is None:
            self._composer_headers = []
            if self._cdb is not None:
                headers = self._cdb.get_json(
                    "composer.composerHeaders", table="ItemTable"
                )
                if headers and isinstance(headers, dict):
                    self._composer_headers = headers.get("allComposers", []) or []
        return self._composer_headers

    def headers_map(self) -> dict[str, list[dict]]:
        if self._headers_map is None:
            self._headers_map = paths._headers_map_from_entries(self.composer_headers())
        return self._headers_map

    def _ensure_typed_index(self) -> None:
        """Load the typed catalog from this session's epoch connection."""
        if self._typed_loaded:
            return
        self._typed_loaded = True
        self._typed_table_exists = False
        self._typed_schema = typed_headers.TypedSchemaStatus.ABSENT
        self._typed_catalog = {}
        if self._cdb is None:
            return
        conn = self._cdb._ensure_read_copy()
        self._typed_schema = typed_headers.typed_schema_status(conn)
        self._typed_table_exists = (
            self._typed_schema == typed_headers.TypedSchemaStatus.USABLE
        )
        if self._typed_table_exists:
            self._typed_catalog = typed_headers.load_typed_catalog(conn)

    def typed_table_exists(self) -> bool:
        self._ensure_typed_index()
        return self._typed_table_exists

    def typed_schema(self) -> typed_headers.TypedSchemaStatus:
        self._ensure_typed_index()
        return self._typed_schema

    def typed_catalog(self) -> dict[str, typed_headers.TypedHeaderRow]:
        self._ensure_typed_index()
        return self._typed_catalog

    def typed_row(self, composer_id: str) -> Optional[typed_headers.TypedHeaderRow]:
        return self.typed_catalog().get(composer_id)

    def _ingest_composer_row(
        self, cid: str, composer_raw: bytes, bubbles: list[tuple[str, bytes]]
    ) -> str:
        hasher = hashlib.sha256()
        _hash_field(hasher, f"composerData:{cid}".encode("utf-8"), composer_raw)
        for bid, raw in bubbles:
            _hash_field(hasher, f"bubbleId:{cid}:{bid}".encode("utf-8"), raw)
        return "sha256:" + hasher.hexdigest()

    def _load_inventory(self) -> None:
        """Stream raw composer/bubble bytes into per-CID row fingerprints.

        Does not deserialize JSON. Blob refs are discovered on the deep path
        and reused from cache on subsequent warm runs.
        """
        if self._inventory_complete or self._cdb is None:
            return
        self._inventory_complete = False
        self._row_fp = {}
        hashers: dict[str, Any] = {}
        try:
            conn = self._cdb._ensure_read_copy()
            for key, val in conn.execute(
                "SELECT key, value FROM cursorDiskKV WHERE key LIKE 'composerData:%'"
            ):
                cid = key.split(":", 1)[1]
                raw = _as_bytes(val)
                hasher = hashlib.sha256()
                _hash_field(hasher, f"composerData:{cid}".encode("utf-8"), raw)
                hashers[cid] = hasher

            for key, val in conn.execute(
                "SELECT key, value FROM cursorDiskKV "
                "WHERE key LIKE 'bubbleId:%' ORDER BY key"
            ):
                parts = key.split(":", 2)
                if len(parts) < 3:
                    continue
                cid, bid = parts[1], parts[2]
                hasher = hashers.get(cid)
                if hasher is None:
                    continue
                raw = _as_bytes(val)
                _hash_field(hasher, f"bubbleId:{cid}:{bid}".encode("utf-8"), raw)

            self._row_fp = {
                cid: "sha256:" + hasher.hexdigest()
                for cid, hasher in hashers.items()
            }
            self._inventory_complete = True
        except Exception:
            self._row_fp = {}
            self._inventory_complete = False

    def _load_inventory_targeted(self, composer_ids: Collection[str]) -> None:
        """Fingerprint only *composer_ids* with the same hash as a full scan."""
        if self._cdb is None:
            return
        try:
            conn = self._cdb._ensure_read_copy()
        except Exception:
            return
        for cid in composer_ids:
            if cid in self._row_fp or cid in self._targeted_absent:
                continue
            try:
                row = conn.execute(
                    "SELECT value FROM cursorDiskKV WHERE key = ?",
                    (f"composerData:{cid}",),
                ).fetchone()
            except Exception:
                continue
            if row is None:
                self._targeted_absent.add(cid)
                continue
            bubbles: list[tuple[str, bytes]] = []
            try:
                for key, val in conn.execute(
                    "SELECT key, value FROM cursorDiskKV "
                    "WHERE key LIKE ? ORDER BY key",
                    (f"bubbleId:{cid}:%",),
                ):
                    parts = key.split(":", 2)
                    if len(parts) < 3:
                        continue
                    bubbles.append((parts[2], _as_bytes(val)))
            except Exception:
                continue
            self._row_fp[cid] = self._ingest_composer_row(
                cid, _as_bytes(row[0]), bubbles
            )

    def raw_fingerprint(self, composer_id: str) -> Optional[str]:
        if composer_id in self._row_fp:
            return self._row_fp[composer_id]
        if composer_id in self._targeted_absent:
            return None
        if self._inventory_complete:
            return None
        self._ensure_inventory(
            [composer_id] if self._inventory_scope is not None else None
        )
        if composer_id in self._row_fp:
            return self._row_fp[composer_id]
        if composer_id in self._targeted_absent or self._inventory_complete:
            return None
        return None

    def composer_cell(self, composer_id: str) -> tuple[bool, Any]:
        """Return ``(row_present, parsed)``. JSON null is ``(True, None)``."""
        if composer_id in self._composer_cells:
            val = self._composer_cells[composer_id]
            if val is _COMPOSER_UNREADABLE:
                raise ClassifyError("local composerData is unreadable")
            if val is _COMPOSER_ABSENT:
                return False, None
            return True, val
        if (
            self._inventory_complete or composer_id in self._targeted_absent
        ) and composer_id not in self._row_fp:
            self._composer_cells[composer_id] = _COMPOSER_ABSENT
            return False, None
        if self._cdb is None:
            self._composer_cells[composer_id] = _COMPOSER_ABSENT
            return False, None
        try:
            present, parsed = _strict_select_row(self._cdb, f"composerData:{composer_id}")
        except ClassifyError:
            self._composer_cells[composer_id] = _COMPOSER_UNREADABLE
            raise
        if not present:
            self._composer_cells[composer_id] = _COMPOSER_ABSENT
            return False, None
        _counts.local_composer_json_parses += 1
        self._composer_cells[composer_id] = parsed
        return True, parsed

    def local_presence(self, composer_id: str) -> LocalPresence:
        cached = self._presence.get(composer_id)
        if cached is not None:
            return cached
        if self._cdb is None:
            presence = LocalPresence.DANGLING
        else:
            try:
                present, data = self.composer_cell(composer_id)
            except ClassifyError:
                presence = LocalPresence.INVALID
            else:
                presence = (
                    LocalPresence.DANGLING
                    if not present
                    else classify_local_payload(data)
                )
        self._presence[composer_id] = presence
        return presence

    def composer_data(self, composer_id: str) -> Optional[dict]:
        try:
            present, data = self.composer_cell(composer_id)
        except ClassifyError:
            return None
        if not present or not isinstance(data, dict):
            return None
        return data

    def release_composer_cell(self, composer_id: str) -> None:
        """Drop a cached composerData payload. Presence stays."""
        self._composer_cells.pop(composer_id, None)

    def _available_blob_ids(self) -> set[str]:
        if self._blob_ids is not None:
            return self._blob_ids
        if self._cdb is None:
            self._blob_ids = set()
            return self._blob_ids
        prefix = "composer.content."
        try:
            self._blob_ids = {
                key[len(prefix):] for key in self._cdb.list_keys(prefix)
            }
        except Exception:
            self._blob_ids = set()
        return self._blob_ids

    def _hash_blob_refs(self, refs: list[str]) -> str:
        hasher = hashlib.sha256()
        for ref in refs:
            val = None
            if self._cdb is not None:
                val = self._cdb.get_item_binary(
                    f"composer.content.{ref}", table="cursorDiskKV"
                )
            _hash_field(
                hasher,
                ref.encode("utf-8"),
                _as_bytes(val) if val is not None else b"\0missing",
            )
        return "sha256:" + hasher.hexdigest()

    def cached_or_compute_digest(self, composer_id: str) -> Optional[str]:
        """Reuse cached semantic digest when row + blob fingerprints match."""
        if composer_id in self._local_digest:
            return self._local_digest[composer_id]
        self._ensure_inventory()
        row_fp = self.raw_fingerprint(composer_id)
        known_absent = (
            self._inventory_complete or composer_id in self._targeted_absent
        )
        if row_fp is None and known_absent:
            return None
        if row_fp:
            cached = self.cache.get_local(composer_id)
            if cached and cached["rowFingerprint"] == row_fp:
                blob_fp = self._hash_blob_refs(list(cached.get("blobRefs") or []))
                if blob_fp == cached.get("blobFingerprint"):
                    self._local_digest[composer_id] = cached["semanticDigest"]
                    return cached["semanticDigest"]
        hashes = self.local_unit_hashes(composer_id)
        if hashes is None:
            return None
        return self._local_digest.get(composer_id)

    def _load_local_blobs(
        self,
        payload: Any,
        available: set[str],
        discovered: set[str],
    ) -> dict[str, Any]:
        blobs: dict[str, Any] = {}
        for ref in _required_blob_ids(payload, available):
            val = self._cdb.get_disk_kv(f"composer.content.{ref}") if self._cdb else None
            if val is None:
                raise ClassifyError(f"referenced blob missing locally: {ref}")
            blobs[ref] = val
            discovered.add(ref)
        return blobs

    def _modern_local_unit_hashes(
        self,
        composer_id: str,
        data: dict,
        available: set[str],
        discovered: set[str],
    ) -> list[str]:
        hashes: list[str] = []
        for header in _headers(data):
            bid = header.get("bubbleId")
            if not bid:
                raise ClassifyError("local header is missing bubbleId")
            bubble = (
                _strict_select_json(self._cdb, f"bubbleId:{composer_id}:{bid}")
                if self._cdb is not None
                else None
            )
            if bubble is None:
                cmap = _optional_mapping(
                    data.get("conversationMap"),
                    "local conversationMap",
                )
                if cmap is not None and bid in cmap:
                    bubble = _coerce_bubble(cmap[bid], bid)
            elif not isinstance(bubble, dict):
                raise ClassifyError(f"local bubble {bid} is not an object")
            sem_header = _semantic_header(header)
            blob_payload: Any = sem_header if bubble is None else (sem_header, bubble)
            blobs = self._load_local_blobs(blob_payload, available, discovered)
            hashes.append(unit_hash(header, bubble, blobs))
        return hashes

    def _legacy_local_unit_hashes(
        self,
        data: dict,
        available: set[str],
        discovered: set[str],
    ) -> list[str]:
        hashes: list[str] = []
        for message in _legacy_messages(data.get("conversation")):
            normalized = _normalize_unit_object(message, top=True)
            blobs = self._load_local_blobs(normalized, available, discovered)
            hashes.append(_legacy_unit_hash(message, blobs))
        return hashes

    def local_unit_hashes(self, composer_id: str) -> Optional[list[str]]:
        if composer_id in self._local_hashes:
            return self._local_hashes[composer_id]
        present, data = self.composer_cell(composer_id)
        if not present:
            return None
        if classify_local_payload(data) == LocalPresence.INVALID:
            raise ClassifyError("local composerData is unreadable")
        _counts.local_semantic_rehashes += 1
        available = self._available_blob_ids()
        discovered: set[str] = set()
        if "fullConversationHeadersOnly" in data:
            hashes = self._modern_local_unit_hashes(
                composer_id, data, available, discovered
            )
        else:
            hashes = self._legacy_local_unit_hashes(data, available, discovered)
        self._local_hashes[composer_id] = hashes
        digest = conversation_digest(hashes)
        self._local_digest[composer_id] = digest
        row_fp = self._row_fp.get(composer_id)
        if row_fp:
            blob_refs = sorted(discovered)
            self.cache.put_local(
                composer_id,
                row_fp,
                blob_refs,
                self._hash_blob_refs(blob_refs),
                digest,
            )
        return hashes

    def local_digest(self, composer_id: str) -> Optional[str]:
        return self.cached_or_compute_digest(composer_id)

    def export_conversation(
        self,
        project_path: str,
        composer_id: str,
        source_host: Optional[str] = None,
    ) -> Optional[dict]:
        if self._cdb is None:
            return None
        _counts.full_local_exports += 1
        cached = self.composer_data(composer_id)
        snapshot = export.export_conversation(
            project_path,
            composer_id,
            _cdb=self._cdb,
            source_host=source_host,
            composer_data=cached,
        )
        self.release_composer_cell(composer_id)
        return snapshot


@dataclass
class PlannedItem:
    composer_id: str
    relation: SyncRelation
    name: str = ""
    snapshot_path: Optional[Path] = None
    staged_path: Optional[Path] = None
    meta: dict = field(default_factory=dict)
    workspace_dir: Optional[Path] = None
    project_path: str = ""
    source_host: Optional[str] = None
    project_identifier: str = ""
    classified_identity: tuple = ()
    classified_content_digest: str = ""
    dest_expected_present: Optional[bool] = None
    local_guard: Optional["LocalGuard"] = None
    registration: Optional[typed_headers.RegistrationHealth] = None


@dataclass
class SyncPlan:
    items: list[PlannedItem] = field(default_factory=list)
    target_workspace: Optional[dict] = None
    typed_table_exists: bool = False

    def by_relation(self, relation: SyncRelation) -> list[PlannedItem]:
        return [i for i in self.items if i.relation == relation]

    @property
    def diverged(self) -> list[PlannedItem]:
        return self.by_relation(SyncRelation.DIVERGED)

    @property
    def unknown(self) -> list[PlannedItem]:
        return self.by_relation(SyncRelation.UNKNOWN)

    @property
    def behind(self) -> list[PlannedItem]:
        return self.by_relation(SyncRelation.BEHIND)

    @property
    def ahead(self) -> list[PlannedItem]:
        return self.by_relation(SyncRelation.LOCAL_AHEAD)

    @property
    def unsafe(self) -> bool:
        return bool(self.diverged or self.unknown)

    @property
    def registration_conflicts(self) -> list[PlannedItem]:
        """Typed row already bound to an incompatible workspace."""
        return [
            item
            for item in self.items
            if typed_headers.is_registration_conflict(item.registration)
        ]


def _registration_for_listed(
    session: SyncReadSession,
    composer_id: str,
    workspace_id: str,
    target_identifier: Optional[dict] = None,
) -> typed_headers.RegistrationHealth:
    """CID came from workspace membership. No typed row → legacy-only."""
    if not session.typed_table_exists():
        return typed_headers.RegistrationHealth.REGISTERED
    row = session.typed_row(composer_id)
    typed_ident = None
    if row is not None:
        typed_ident = row.header.get("workspaceIdentifier")
    return typed_headers.classify_registration(
        typed_table_exists=True,
        typed_workspace_id=row.workspace_id if row is not None else None,
        target_workspace_id=workspace_id,
        in_legacy_sources=row is None,
        typed_identifier=typed_ident,
        target_identifier=target_identifier,
    )


def _registration_for_snapshot_only(
    session: SyncReadSession,
    composer_id: str,
    workspace_id: str,
    target_identifier: Optional[dict] = None,
) -> typed_headers.RegistrationHealth:
    """Snapshot-only CID: missing typed row is MISSING, not legacy-only."""
    if not session.typed_table_exists():
        return typed_headers.RegistrationHealth.REGISTERED
    row = session.typed_row(composer_id)
    typed_ident = None
    if row is not None:
        typed_ident = row.header.get("workspaceIdentifier")
    return typed_headers.classify_registration(
        typed_table_exists=True,
        typed_workspace_id=row.workspace_id if row is not None else None,
        target_workspace_id=workspace_id,
        in_legacy_sources=False,
        typed_identifier=typed_ident,
        target_identifier=target_identifier,
    )


def _planned_name(
    session: SyncReadSession,
    rec: Optional[SnapshotRecord],
    composer_id: str,
    relation: SyncRelation,
) -> str:
    if rec is not None:
        meta_name = rec.meta.get("name")
        if meta_name:
            return meta_name
    if relation in (SyncRelation.DIVERGED, SyncRelation.UNKNOWN):
        data = session.composer_data(composer_id)
        if isinstance(data, dict) and data.get("name"):
            return data["name"]
        return "Untitled"
    return "Untitled"


def classify_pair(
    local_hashes: Optional[list[str]],
    remote_hashes: Optional[list[str]],
    *,
    has_snapshot: bool,
) -> SyncRelation:
    if not has_snapshot:
        return SyncRelation.NEVER_PUSHED if local_hashes is not None else SyncRelation.UNKNOWN
    if local_hashes is None:
        return SyncRelation.BEHIND
    if remote_hashes is None:
        return SyncRelation.UNKNOWN
    return compare_unit_hashes(local_hashes, remote_hashes)


def classify_conversation(
    session: SyncReadSession,
    index: SnapshotIndex,
    composer_id: str,
    *,
    project_identifier: Optional[str] = None,
    source_host: Optional[str] = None,
    source_path: Optional[str] = None,
    workspace: Optional[dict] = None,
) -> SyncRelation:
    if workspace is not None:
        if project_identifier is None:
            project_identifier = paths.get_workspace_project_identifier(workspace)
        if source_host is None:
            source_host = workspace.get("host")
        if source_path is None:
            source_path = workspace.get("path")

    rec = index.get(
        composer_id,
        project_identifier,
        source_host=source_host,
        source_path=source_path,
    )
    if rec is None:
        presence = classify_local_conversation(session, composer_id)
        if presence == LocalPresence.ACTIVE:
            return SyncRelation.NEVER_PUSHED
        return SyncRelation.UNKNOWN
    if rec.invalid_origin:
        return SyncRelation.UNKNOWN

    session.cache = index.cache
    try:
        local_digest = session.cached_or_compute_digest(composer_id)
    except ClassifyError:
        return SyncRelation.UNKNOWN

    try:
        remote_units, remote_digest = index.remote_semantics(rec)
    except ClassifyError:
        return SyncRelation.UNKNOWN

    if local_digest and remote_digest and local_digest == remote_digest:
        return SyncRelation.UP_TO_DATE

    try:
        local_hashes = session.local_unit_hashes(composer_id)
        if remote_units is None:
            remote_units = index.remote_unit_hashes(rec)
    except ClassifyError:
        return SyncRelation.UNKNOWN
    relation = classify_pair(local_hashes, remote_units, has_snapshot=True)
    index._remote_units.pop((rec.project_identifier, rec.composer_id), None)
    return relation


def classify_snapshot_vs_local(
    session: SyncReadSession,
    index: SnapshotIndex,
    composer_id: str,
    *,
    project_identifier: Optional[str] = None,
) -> str:
    rec = index.get(composer_id, project_identifier) if project_identifier else None
    if rec is None and project_identifier:
        return NOT_LOCAL
    if rec is None:
        return NOT_LOCAL
    try:
        local_digest = session.cached_or_compute_digest(composer_id)
    except ClassifyError:
        return SyncRelation.UNKNOWN.value
    if local_digest is None and session.composer_data(composer_id) is None:
        return NOT_LOCAL
    relation = classify_conversation(
        session, index, composer_id, project_identifier=rec.project_identifier
    )
    return relation.value


def build_sync_plan(
    session: SyncReadSession,
    index: SnapshotIndex,
    workspaces: Optional[list[dict]] = None,
    target_workspace: Optional[dict] = None,
) -> SyncPlan:
    """Classify local conversations and snapshots. Read-only.

    With *target_workspace*, only that workspace is classified. Other
    hosts and workspaces never enter the plan — including a diverged
    chat on an unselected workspace. Without it, every workspace is
    classified (the historical global ``sync``).
    """
    plan = SyncPlan(target_workspace=target_workspace)
    plan.typed_table_exists = session.typed_table_exists()
    seen: set[tuple[str, str]] = set()
    session.cache = index.cache

    if target_workspace is not None:
        workspaces = [target_workspace]
    elif workspaces is None:
        workspaces = paths.list_workspaces_with_conversations(session=session)

    target_ids: Optional[set[str]] = None
    snapshot_project_id: Optional[str] = None
    if target_workspace is not None:
        target_ids = _target_workspace_composer_ids(
            target_workspace, session=session
        )
        session.prepare_inventory(target_ids)
        # Identity of the already-resolved snapshot bucket, not the
        # canonical workspace ID. A scoped index of snapshots/nixos/
        # must still match SSH chats whose workspace ID is
        # ssh-MindLoop1-home-lju-nixos. Host + path stay exact via
        # index.get / _record_matches_target.
        snapshot_project_id = (
            index.scoped_project_identifier
            or paths.get_workspace_project_identifier(target_workspace)
        )
    else:
        session.prepare_inventory(None)

    for ws in workspaces:
        ws_dir = ws["workspace_dir"]
        db_path = ws_dir / "state.vscdb"
        if not db_path.exists():
            continue
        project_id = paths.get_workspace_project_identifier(ws)
        lookup_id = snapshot_project_id or project_id
        target_identifier = importer._build_workspace_identifier(ws_dir)
        composer_ids = paths.get_workspace_composer_ids(db_path, session=session)
        for cid in composer_ids:
            key = (lookup_id, cid)
            if key in seen:
                continue
            seen.add(key)
            try:
                rec = index.get(
                    cid,
                    lookup_id,
                    source_host=ws.get("host"),
                    source_path=ws.get("path"),
                )
                # Presence is only needed when there is no snapshot. A warm
                # digest hit must not re-parse composerData just to learn
                # ACTIVE vs EMPTY. EMPTY/DANGLING with a snapshot stay in
                # the plan so restore classification can still run.
                if rec is None:
                    presence = classify_local_conversation(session, cid)
                    if is_inactive_registration(presence, False):
                        continue
                    if presence == LocalPresence.INVALID:
                        relation = SyncRelation.UNKNOWN
                    else:
                        relation = SyncRelation.NEVER_PUSHED
                else:
                    try:
                        relation = classify_conversation(
                            session, index, cid, workspace=ws, project_identifier=lookup_id
                        )
                    except ClassifyError:
                        relation = SyncRelation.UNKNOWN
                item = PlannedItem(
                    composer_id=cid,
                    relation=relation,
                    name=_planned_name(session, rec, cid, relation),
                    snapshot_path=rec.path if rec else None,
                    meta=rec.meta if rec else {},
                    workspace_dir=ws_dir,
                    project_path=ws.get("path") or "",
                    source_host=ws.get("host"),
                    project_identifier=lookup_id,
                    registration=_registration_for_listed(
                        session, cid, ws_dir.name, target_identifier
                    ),
                )
                if relation == SyncRelation.BEHIND:
                    present = session.raw_fingerprint(cid) is not None
                    if rec is None or not _pin_behind_item(
                        item,
                        session,
                        rec,
                        expect_present=present,
                        expect_in_target=True,
                    ):
                        item.relation = SyncRelation.UNKNOWN
                elif relation == SyncRelation.LOCAL_AHEAD:
                    if rec is None or not _pin_ahead_dest(item, rec):
                        item.relation = SyncRelation.UNKNOWN
                elif relation == SyncRelation.NEVER_PUSHED:
                    _pin_ahead_dest(item, rec)
                plan.items.append(item)
            finally:
                session.release_composer_cell(cid)

    for rec in index.by_key.values():
        key = (rec.project_identifier, rec.composer_id)
        if key in seen:
            continue
        if target_workspace is not None:
            if not _record_matches_target(rec, target_workspace):
                continue
            if rec.project_identifier != snapshot_project_id:
                continue
        try:
            # CID present globally but not in this workspace belongs to
            # someone else. Do not treat it as this target's behind.
            if target_workspace is not None and rec.composer_id not in (
                target_ids or set()
            ) and _global_has_composer(session, rec.composer_id):
                # Typed-wins hides this CID from the target listing. A
                # foreign/stale typed row is still this target's conflict:
                # do not skip it as "already in sync".
                remote_ws_dir = target_workspace.get("workspace_dir")
                health = _registration_for_snapshot_only(
                    session,
                    rec.composer_id,
                    Path(remote_ws_dir).name if remote_ws_dir is not None else "",
                    importer._build_workspace_identifier(remote_ws_dir)
                    if remote_ws_dir is not None
                    else None,
                )
                if typed_headers.is_registration_conflict(health):
                    seen.add(key)
                    try:
                        relation = classify_conversation(
                            session,
                            index,
                            rec.composer_id,
                            workspace=target_workspace,
                            project_identifier=snapshot_project_id,
                        )
                    except ClassifyError:
                        relation = SyncRelation.UNKNOWN
                    plan.items.append(
                        PlannedItem(
                            composer_id=rec.composer_id,
                            relation=relation,
                            name=rec.meta.get("name") or "Untitled",
                            snapshot_path=rec.path,
                            meta=rec.meta,
                            workspace_dir=remote_ws_dir,
                            project_path=target_workspace.get("path") or "",
                            source_host=target_workspace.get("host"),
                            project_identifier=rec.project_identifier,
                            registration=health,
                        )
                    )
                continue
            seen.add(key)
            # No local workspace registered this CID for this origin. Do not
            # consult the global composer: it may belong to another project.
            if rec.invalid_origin:
                relation = SyncRelation.UNKNOWN
            else:
                try:
                    index.ensure_remote_readable(rec)
                except ClassifyError:
                    relation = SyncRelation.UNKNOWN
                else:
                    relation = SyncRelation.BEHIND
            if target_workspace is not None:
                remote_ws_dir = target_workspace.get("workspace_dir")
                remote_path = target_workspace.get("path") or ""
                remote_host = target_workspace.get("host")
            else:
                remote_ws_dir = None
                remote_path = rec.meta.get("sourceProjectPath") or ""
                remote_host = rec.meta.get("sourceHost")
            item = PlannedItem(
                composer_id=rec.composer_id,
                relation=relation,
                name=rec.meta.get("name") or "Untitled",
                snapshot_path=rec.path,
                meta=rec.meta,
                workspace_dir=remote_ws_dir,
                project_path=remote_path,
                source_host=remote_host,
                project_identifier=rec.project_identifier,
                registration=_registration_for_snapshot_only(
                    session,
                    rec.composer_id,
                    Path(remote_ws_dir).name if remote_ws_dir is not None else "",
                    importer._build_workspace_identifier(remote_ws_dir)
                    if remote_ws_dir is not None
                    else None,
                ),
            )
            if relation == SyncRelation.BEHIND:
                present = session.raw_fingerprint(rec.composer_id) is not None
                if not _pin_behind_item(
                    item,
                    session,
                    rec,
                    expect_present=present,
                    expect_in_target=False,
                ):
                    item.relation = SyncRelation.UNKNOWN
            plan.items.append(item)
        finally:
            session.release_composer_cell(rec.composer_id)
    index.cache.flush()
    session.cache.flush()
    return plan


class PullRelation(str, Enum):
    """Target-scoped pull classification. Not used by ``sync``."""

    MISSING_LOCAL = "missing_local"
    GLOBAL_COLLISION = "global_collision"
    UP_TO_DATE = "up_to_date"
    BEHIND = "behind"
    LOCAL_AHEAD = "local_ahead"
    DIVERGED = "diverged"
    UNKNOWN = "unknown"
    NEVER_PUSHED = "never_pushed"


class PullAction(str, Enum):
    IMPORT = "import"
    SKIP = "skip"


@dataclass
class LocalGuard:
    """Preflight snapshot of local Cursor state for one import candidate."""

    expect_present: bool
    expect_in_target: bool
    row_fingerprint: str = ""
    blob_refs: list[str] = field(default_factory=list)
    blob_fingerprint: str = ""


_SYNC_TO_PULL = {
    SyncRelation.UP_TO_DATE: PullRelation.UP_TO_DATE,
    SyncRelation.BEHIND: PullRelation.BEHIND,
    SyncRelation.LOCAL_AHEAD: PullRelation.LOCAL_AHEAD,
    SyncRelation.DIVERGED: PullRelation.DIVERGED,
    SyncRelation.UNKNOWN: PullRelation.UNKNOWN,
    SyncRelation.NEVER_PUSHED: PullRelation.NEVER_PUSHED,
}


@dataclass
class PullItem:
    composer_id: str
    relation: PullRelation
    action: PullAction = PullAction.SKIP
    name: str = ""
    snapshot_path: Optional[Path] = None
    staged_path: Optional[Path] = None
    meta: dict = field(default_factory=dict)
    workspace_dir: Optional[Path] = None
    project_path: str = ""
    source_host: Optional[str] = None
    project_identifier: str = ""
    classified_identity: tuple = ()
    classified_content_digest: str = ""
    local_guard: Optional[LocalGuard] = None
    registration: Optional[typed_headers.RegistrationHealth] = None


@dataclass
class PullPlan:
    items: list[PullItem] = field(default_factory=list)
    never_pushed: int = 0
    restore_all: bool = False
    typed_table_exists: bool = False

    def by_relation(self, relation: PullRelation) -> list[PullItem]:
        return [i for i in self.items if i.relation == relation]

    @property
    def import_candidates(self) -> list[PullItem]:
        return [i for i in self.items if i.action == PullAction.IMPORT]

    @property
    def synced(self) -> list[PullItem]:
        return self.by_relation(PullRelation.UP_TO_DATE)

    @property
    def behind(self) -> list[PullItem]:
        return self.by_relation(PullRelation.BEHIND)

    @property
    def ahead(self) -> list[PullItem]:
        return self.by_relation(PullRelation.LOCAL_AHEAD)

    @property
    def missing_local(self) -> list[PullItem]:
        return self.by_relation(PullRelation.MISSING_LOCAL)

    @property
    def diverged(self) -> list[PullItem]:
        return self.by_relation(PullRelation.DIVERGED)

    @property
    def unknown(self) -> list[PullItem]:
        return self.by_relation(PullRelation.UNKNOWN)

    @property
    def collisions(self) -> list[PullItem]:
        return self.by_relation(PullRelation.GLOBAL_COLLISION)

    @property
    def registration_conflicts(self) -> list[PullItem]:
        """Typed row already bound to an incompatible workspace."""
        return [
            item
            for item in self.items
            if typed_headers.is_registration_conflict(item.registration)
        ]


def _pull_action(relation: PullRelation, restore_all: bool) -> PullAction:
    if relation in (
        PullRelation.DIVERGED,
        PullRelation.UNKNOWN,
        PullRelation.NEVER_PUSHED,
        PullRelation.GLOBAL_COLLISION,
    ):
        return PullAction.SKIP
    if restore_all and relation in (
        PullRelation.MISSING_LOCAL,
        PullRelation.BEHIND,
        PullRelation.UP_TO_DATE,
        PullRelation.LOCAL_AHEAD,
    ):
        return PullAction.IMPORT
    if relation in (PullRelation.MISSING_LOCAL, PullRelation.BEHIND):
        return PullAction.IMPORT
    return PullAction.SKIP


def _record_matches_target(rec: SnapshotRecord, workspace: dict) -> bool:
    """Origin filter: host + normalized path before any semantic compare."""
    host = workspace.get("host")
    path = workspace.get("path") or ""
    if rec.invalid_origin:
        return True
    if host:
        if rec.meta.get("sourceHost") != host:
            return False
        snap_path = rec.meta.get("sourceProjectPath") or ""
        if path and paths.normalize_origin_path(
            snap_path, source_host=host
        ) != paths.normalize_origin_path(path, source_host=host):
            return False
        return True
    if rec.meta.get("sourceHost"):
        return False
    return True


def _target_workspace_composer_ids(workspace: dict, session=None) -> set[str]:
    ws_dir = workspace.get("workspace_dir")
    if not ws_dir:
        return set()
    ws_db = Path(ws_dir) / "state.vscdb"
    if not ws_db.exists():
        return set()
    return set(paths.get_workspace_composer_ids(ws_db, session=session))


def _global_has_composer(session: SyncReadSession, composer_id: str) -> bool:
    """True if the global Cursor DB already has this composer.

    Presence only — never used to classify semantics against another
    workspace. Fail closed if the inventory is incomplete.
    """
    if session.cdb is None:
        return False
    if session._inventory_complete:
        return session.raw_fingerprint(composer_id) is not None
    try:
        present, _ = session.composer_cell(composer_id)
    except ClassifyError:
        return True
    return present


def _candidate_content_digest(rec: SnapshotRecord) -> str:
    """Identity of bytes we are about to stage. Hash only when needed."""
    sidecar = rec.meta.get("snapshotContentDigest")
    if isinstance(sidecar, str) and sidecar:
        return sidecar
    return snapshot_content_digest(rec.path, rec.meta)


def _pin_ahead_dest(item: PlannedItem, rec: Optional[SnapshotRecord]) -> bool:
    """Pin the destination identity classified at preflight. Absence is False."""
    if rec is None:
        item.dest_expected_present = False
        item.classified_identity = ()
        return True
    item.dest_expected_present = True
    item.classified_identity = rec.identity
    return True


def _item_destination_main(item: PlannedItem) -> Path:
    if item.snapshot_path is not None:
        return item.snapshot_path
    return (
        paths.get_snapshots_dir()
        / item.project_identifier
        / f"{item.composer_id}.json.gz"
    )


def _dest_matches_preflight(item: PlannedItem) -> bool:
    if item.dest_expected_present is None:
        return False
    present, identity = destination_snapshot_identity(
        _item_destination_main(item), item.meta
    )
    if present != item.dest_expected_present:
        return False
    if present and identity != item.classified_identity:
        return False
    return True


def verify_ahead_destinations(plan: SyncPlan) -> None:
    """Abort if any AHEAD/NEVER_PUSHED destination changed since preflight."""
    for item in plan.items:
        if item.relation not in (
            SyncRelation.LOCAL_AHEAD,
            SyncRelation.NEVER_PUSHED,
        ):
            continue
        if not _dest_matches_preflight(item):
            raise SyncPreflightStale(
                f"destination changed for {item.composer_id}"
            )


def _pin_behind_item(
    item: PlannedItem,
    session: SyncReadSession,
    rec: SnapshotRecord,
    *,
    expect_present: bool,
    expect_in_target: bool,
) -> bool:
    """Attach LocalGuard + classified snapshot identity. False = fail closed."""
    item.classified_identity = rec.identity
    try:
        item.classified_content_digest = _candidate_content_digest(rec)
    except OSError:
        return False
    guard = capture_local_guard(
        session,
        item.composer_id,
        expect_present=expect_present,
        expect_in_target=expect_in_target,
    )
    if guard is None:
        return False
    item.local_guard = guard
    return True


def capture_local_guard(
    session: SyncReadSession,
    composer_id: str,
    *,
    expect_present: bool,
    expect_in_target: bool,
) -> Optional[LocalGuard]:
    """Record the local source identity seen during preflight.

    For a present conversation this reuses Commit 4 fingerprints. Returns
    None if a present chat cannot be fingerprinted (fail closed).
    """
    if not expect_present:
        return LocalGuard(expect_present=False, expect_in_target=expect_in_target)
    # DANGLING (registration, no composerData row) + snapshot is BEHIND
    # in the sync planner, but there is no local row to fingerprint.
    # Fail closed here: UNKNOWN/SKIP. Restoring that case is separate
    # from treating empty shells as not-pushed.
    row_fp = session.raw_fingerprint(composer_id)
    if not row_fp:
        return None
    cached = session.cache.get_local(composer_id)
    if cached and cached.get("rowFingerprint") == row_fp:
        return LocalGuard(
            expect_present=True,
            expect_in_target=expect_in_target,
            row_fingerprint=row_fp,
            blob_refs=list(cached.get("blobRefs") or []),
            blob_fingerprint=cached.get("blobFingerprint") or "",
        )
    try:
        session.local_unit_hashes(composer_id)
    except ClassifyError:
        return None
    cached = session.cache.get_local(composer_id)
    if not cached or cached.get("rowFingerprint") != row_fp:
        return None
    return LocalGuard(
        expect_present=True,
        expect_in_target=expect_in_target,
        row_fingerprint=row_fp,
        blob_refs=list(cached.get("blobRefs") or []),
        blob_fingerprint=cached.get("blobFingerprint") or "",
    )


def _live_select_value(cdb: "db.CursorDB", key: str, table: str) -> Optional[Any]:
    """Return the live cell, or None if the key is absent. SQL errors propagate."""
    conn = cdb._reader_conn()
    row = conn.execute(
        f"SELECT value FROM {table} WHERE key = ?",
        (key,),
    ).fetchone()
    if row is None:
        return None
    return row[0]


def _live_select_json(cdb: "db.CursorDB", key: str, table: str) -> Optional[Any]:
    """Parse a live JSON cell. Absent key → None. Corrupt/SQL errors propagate."""
    raw = _live_select_value(cdb, key, table)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _live_row_fingerprint(cdb: "db.CursorDB", composer_id: str) -> Optional[str]:
    """Hash live composer + bubble rows. None only if the composer row is absent."""
    conn = cdb._reader_conn()
    row = conn.execute(
        "SELECT value FROM cursorDiskKV WHERE key = ?",
        (f"composerData:{composer_id}",),
    ).fetchone()
    if row is None:
        return None
    hasher = hashlib.sha256()
    _hash_field(
        hasher,
        f"composerData:{composer_id}".encode("utf-8"),
        _as_bytes(row[0]),
    )
    bubbles = conn.execute(
        "SELECT key, value FROM cursorDiskKV "
        "WHERE key LIKE ? ORDER BY key",
        (f"bubbleId:{composer_id}:%",),
    )
    for key, val in bubbles:
        _hash_field(hasher, key.encode("utf-8") if isinstance(key, str) else key, _as_bytes(val))
    return "sha256:" + hasher.hexdigest()


def _live_blob_fingerprint(cdb: "db.CursorDB", refs: list[str]) -> str:
    hasher = hashlib.sha256()
    for ref in refs:
        val = _live_select_value(cdb, f"composer.content.{ref}", "cursorDiskKV")
        _hash_field(
            hasher,
            ref.encode("utf-8"),
            _as_bytes(val) if val is not None else b"\0missing",
        )
    return "sha256:" + hasher.hexdigest()


def _live_target_has_composer(
    workspace_cdb: "db.CursorDB",
    global_cdb: "db.CursorDB",
    composer_id: str,
    workspace_dir: Optional[Path],
) -> bool:
    data = _live_select_json(workspace_cdb, "composer.composerData", "ItemTable")
    if isinstance(data, dict):
        for entry in data.get("allComposers") or []:
            if isinstance(entry, dict) and entry.get("composerId") == composer_id:
                return True
        for field in ("selectedComposerIds", "lastFocusedComposerIds"):
            ids = data.get(field) or []
            if composer_id in ids:
                return True
    if workspace_dir is None:
        return False
    ws_hash = Path(workspace_dir).name
    live_conn = global_cdb._reader_conn()
    if typed_headers.typed_table_usable(live_conn):
        typed_row = typed_headers.get_typed_row(live_conn, composer_id)
        if typed_row is not None:
            return typed_row.workspace_id == ws_hash
    headers = _live_select_json(global_cdb, "composer.composerHeaders", "ItemTable")
    if not isinstance(headers, dict):
        return False
    for entry in headers.get("allComposers") or []:
        if not isinstance(entry, dict) or entry.get("composerId") != composer_id:
            continue
        wi = entry.get("workspaceIdentifier") or {}
        if wi.get("id") == ws_hash:
            return True
    return False


def local_guard_still_matches(
    item: Any,
    global_cdb: "db.CursorDB",
    workspace_cdb: "db.CursorDB",
) -> bool:
    """True if live Cursor state still matches the preflight guard.

    A read error is a mismatch, never an inferred absence.
    """
    _counts.local_guard_checks += 1
    guard = item.local_guard
    if guard is None:
        return False
    try:
        live_fp = _live_row_fingerprint(global_cdb, item.composer_id)
        present = live_fp is not None
        if present != guard.expect_present:
            return False
        in_target = _live_target_has_composer(
            workspace_cdb, global_cdb, item.composer_id, item.workspace_dir
        )
        if in_target != guard.expect_in_target:
            return False
        if not guard.expect_present:
            return True
        if live_fp != guard.row_fingerprint:
            return False
        if _live_blob_fingerprint(global_cdb, guard.blob_refs) != guard.blob_fingerprint:
            return False
        return True
    except Exception:
        return False


def build_pull_plan(
    session: SyncReadSession,
    index: SnapshotIndex,
    target_workspace: dict,
    *,
    restore_all: bool = False,
    selected_paths: Optional[list[Path]] = None,
) -> PullPlan:
    """Classify snapshots for one pull target. Does not walk other workspaces.

    Local presence is the target workspace's composer IDs only. A CID that
    exists in the global Cursor DB but not in the target is a collision:
    importing it would overwrite the other workspace's conversation.
    """
    plan = PullPlan(restore_all=restore_all)
    plan.typed_table_exists = session.typed_table_exists()
    session.cache = index.cache
    selected = {p.resolve() for p in selected_paths} if selected_paths else None
    target_ids = _target_workspace_composer_ids(target_workspace, session=session)
    session.prepare_inventory(target_ids)
    snapshot_cids: set[str] = set()
    ws_dir = target_workspace.get("workspace_dir")
    project_path = target_workspace.get("path") or ""
    source_host = target_workspace.get("host")

    for rec in index.by_key.values():
        if selected is not None and rec.path.resolve() not in selected:
            continue
        if not _record_matches_target(rec, target_workspace):
            continue
        snapshot_cids.add(rec.composer_id)
        try:
            if rec.invalid_origin:
                relation = PullRelation.UNKNOWN
            elif rec.composer_id not in target_ids:
                if _global_has_composer(session, rec.composer_id):
                    relation = PullRelation.GLOBAL_COLLISION
                else:
                    try:
                        index.ensure_remote_readable(rec)
                    except ClassifyError:
                        relation = PullRelation.UNKNOWN
                    else:
                        relation = PullRelation.MISSING_LOCAL
            else:
                try:
                    sync_rel = classify_conversation(
                        session,
                        index,
                        rec.composer_id,
                        project_identifier=rec.project_identifier,
                        source_host=source_host,
                        source_path=project_path,
                        workspace=target_workspace,
                    )
                except ClassifyError:
                    sync_rel = SyncRelation.UNKNOWN
                relation = _SYNC_TO_PULL.get(sync_rel, PullRelation.UNKNOWN)

            action = _pull_action(relation, restore_all)
            content_digest = ""
            local_guard = None
            if action == PullAction.IMPORT:
                try:
                    content_digest = _candidate_content_digest(rec)
                except OSError:
                    relation = PullRelation.UNKNOWN
                    action = PullAction.SKIP
            if action == PullAction.IMPORT:
                present = relation != PullRelation.MISSING_LOCAL
                local_guard = capture_local_guard(
                    session,
                    rec.composer_id,
                    expect_present=present,
                    expect_in_target=present,
                )
                if local_guard is None:
                    relation = PullRelation.UNKNOWN
                    action = PullAction.SKIP

            ws_hash = Path(ws_dir).name if ws_dir is not None else ""
            target_identifier = (
                importer._build_workspace_identifier(ws_dir)
                if ws_dir is not None
                else None
            )
            if rec.composer_id in target_ids and ws_hash:
                registration = _registration_for_listed(
                    session, rec.composer_id, ws_hash, target_identifier
                )
            else:
                registration = _registration_for_snapshot_only(
                    session, rec.composer_id, ws_hash, target_identifier
                )
            item = PullItem(
                composer_id=rec.composer_id,
                relation=relation,
                action=action,
                name=rec.meta.get("name") or "Untitled",
                snapshot_path=rec.path,
                meta=rec.meta,
                workspace_dir=ws_dir,
                project_path=project_path,
                source_host=source_host,
                project_identifier=rec.project_identifier,
                classified_identity=rec.identity,
                classified_content_digest=content_digest,
                local_guard=local_guard,
                registration=registration,
            )
            plan.items.append(item)
        finally:
            session.release_composer_cell(rec.composer_id)

    never_pushed = 0
    ws_hash = Path(ws_dir).name if ws_dir is not None else ""
    for cid in target_ids - snapshot_cids:
        try:
            if classify_local_conversation(session, cid) == LocalPresence.ACTIVE:
                never_pushed += 1
                data = session.composer_data(cid) or {}
                plan.items.append(
                    PullItem(
                        composer_id=cid,
                        relation=PullRelation.NEVER_PUSHED,
                        action=PullAction.SKIP,
                        name=data.get("name") or "Untitled",
                        workspace_dir=ws_dir,
                        project_path=project_path,
                        source_host=source_host,
                        registration=_registration_for_listed(
                            session,
                            cid,
                            ws_hash,
                            importer._build_workspace_identifier(ws_dir)
                            if ws_dir is not None
                            else None,
                        )
                        if ws_hash
                        else None,
                    )
                )
        finally:
            session.release_composer_cell(cid)
    plan.never_pushed = never_pushed
    _counts.pull_candidates += len(plan.import_candidates)
    index.cache.flush()
    session.cache.flush()
    return plan


def _stage_snapshot_item(item: Any, staging_dir: Path) -> bool:
    """Freeze one classified snapshot into *staging_dir*. False if stale/unreadable."""
    if item.snapshot_path is None:
        return False
    try:
        current_identity = snapshot_source_identity(item.snapshot_path, item.meta)
    except OSError:
        return False
    if item.classified_identity and current_identity != item.classified_identity:
        return False

    dest_dir = staging_dir / item.project_identifier / item.composer_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    components = importer.snapshot_component_files(item.snapshot_path, item.meta)
    if not components:
        return False
    try:
        for src in components:
            shutil.copy2(src, dest_dir / src.name)
        sidecar = importer.snapshot_sidecar_path(item.snapshot_path)
        if sidecar.exists():
            shutil.copy2(sidecar, dest_dir / sidecar.name)
    except OSError:
        return False

    staged_main = dest_dir / item.snapshot_path.name
    try:
        staged_digest = snapshot_content_digest(staged_main, item.meta)
    except OSError:
        return False
    if item.classified_content_digest and staged_digest != item.classified_content_digest:
        return False
    try:
        if snapshot_source_identity(item.snapshot_path, item.meta) != current_identity:
            return False
    except OSError:
        return False

    item.staged_path = staged_main
    _counts.staged_snapshots += 1
    return True


def stage_import_candidates(plan: PullPlan, staging_dir: Path) -> list[PullItem]:
    """Copy only import candidates (main + shards + sidecar) under *staging_dir*.

    Verifies the source identity is unchanged and the staged bytes match
    the classified ``snapshotContentDigest``. A mismatch becomes UNKNOWN.
    """
    staged: list[PullItem] = []
    for item in list(plan.import_candidates):
        if not _stage_snapshot_item(item, staging_dir):
            item.action = PullAction.SKIP
            item.relation = PullRelation.UNKNOWN
            continue
        staged.append(item)
    return staged


def stage_behind_snapshots(plan: SyncPlan, staging_dir: Path) -> bool:
    """Pin every BEHIND snapshot. False if any classified file changed."""
    if plan.unsafe or plan.registration_conflicts:
        return False
    for item in plan.behind:
        if not _stage_snapshot_item(item, staging_dir):
            return False
    return True


def verify_behind_guards(items: list[Any]) -> bool:
    """True if every item's LocalGuard still matches live Cursor state."""
    if not items:
        return True
    global_path = paths.get_global_db_path()
    if not global_path.exists():
        return all(
            item.local_guard is not None and not item.local_guard.expect_present
            for item in items
        )
    with db.CursorDB(global_path) as gdb:
        opened: dict[Path, db.CursorDB] = {}
        try:
            for item in items:
                if item.local_guard is None:
                    return False
                ws_dir = item.workspace_dir
                if ws_dir is None:
                    live_fp = _live_row_fingerprint(gdb, item.composer_id)
                    present = live_fp is not None
                    if present != item.local_guard.expect_present:
                        return False
                    if item.local_guard.expect_in_target:
                        return False
                    if not item.local_guard.expect_present:
                        continue
                    if live_fp != item.local_guard.row_fingerprint:
                        return False
                    if (
                        _live_blob_fingerprint(gdb, item.local_guard.blob_refs)
                        != item.local_guard.blob_fingerprint
                    ):
                        return False
                    continue
                ws_dir = Path(ws_dir)
                if ws_dir not in opened:
                    ws_db = ws_dir / "state.vscdb"
                    if not ws_db.exists():
                        return False
                    opened[ws_dir] = db.CursorDB(ws_db)
                if not local_guard_still_matches(item, gdb, opened[ws_dir]):
                    return False
            return True
        except Exception:
            return False
        finally:
            for cdb in opened.values():
                cdb.close()


def destination_snapshot_identity(
    dest_main: Path, meta: Optional[dict] = None
) -> tuple[bool, tuple]:
    """Return ``(present, identity)`` for a destination snapshot path."""
    components = importer.snapshot_component_files(dest_main, meta)
    if not components:
        return False, ()
    try:
        return True, snapshot_source_identity(dest_main, meta)
    except OSError:
        return False, ()


@dataclass
class AheadExpectation:
    """Destination identity observed when the LOCAL_AHEAD plan was built."""

    composer_id: str
    project_identifier: str
    dest_main: Path
    dest_meta: dict
    expected_present: bool
    expected_identity: tuple
    staged_project: Path


@dataclass
class StagedAhead:
    """LOCAL_AHEAD snapshots materialized from the preflight read view."""

    lease: Any
    count: int
    expectations: list[AheadExpectation] = field(default_factory=list)
    promoted: bool = False

    def discard(self) -> None:
        if self.promoted or self.lease is None:
            return
        self.lease.release()
        self.lease = None


def stage_ahead_exports(
    plan: SyncPlan,
    session: SyncReadSession,
) -> Optional[StagedAhead]:
    """Export LOCAL_AHEAD chats from the open preflight view into a lease.

    Must be called before the read epoch is closed. Does nothing when the
    plan is unsafe or has no ahead items — no lease is created.
    """
    if plan.unsafe or plan.registration_conflicts:
        return None
    verify_ahead_destinations(plan)
    if not plan.ahead:
        return None
    lease = db.acquire_lease("ahead")
    snapshots_root = lease.path / "snapshots"
    expectations: list[AheadExpectation] = []
    saved = 0
    try:
        target_dir = None
        if plan.target_workspace is not None:
            target_dir = plan.target_workspace.get("workspace_dir")
        for item in plan.ahead:
            if target_dir is not None and item.workspace_dir != target_dir:
                continue
            snapshot = session.export_conversation(
                item.project_path,
                item.composer_id,
                source_host=item.source_host,
            )
            if not snapshot:
                continue
            dest_main = _item_destination_main(item)
            export.save_snapshot(snapshot, snapshots_root)
            expectations.append(
                AheadExpectation(
                    composer_id=item.composer_id,
                    project_identifier=item.project_identifier,
                    dest_main=dest_main,
                    dest_meta=dict(item.meta),
                    expected_present=bool(item.dest_expected_present),
                    expected_identity=item.classified_identity,
                    staged_project=snapshots_root / item.project_identifier,
                )
            )
            saved += 1
    except BaseException:
        lease.release()
        raise
    if saved == 0:
        lease.release()
        return None
    return StagedAhead(lease=lease, count=saved, expectations=expectations)


def promote_staged_ahead(staged: StagedAhead) -> int:
    """CAS-install staged ahead snapshots into the real snapshots tree.

    Rechecks every expected destination identity under ``repo_lock``
    before the first mutation. A stale destination aborts the whole
    promote. Replacement uses the same component-clearing semantics as
    ``save_snapshot``.
    """
    with dblock.repo_lock():
        for exp in staged.expectations:
            present, identity = destination_snapshot_identity(
                exp.dest_main, exp.dest_meta
            )
            if present != exp.expected_present:
                raise SyncPreflightStale(
                    f"destination changed for {exp.composer_id}"
                )
            if present and identity != exp.expected_identity:
                raise SyncPreflightStale(
                    f"destination changed for {exp.composer_id}"
                )
        for exp in staged.expectations:
            dest_proj = exp.dest_main.parent
            export.install_staged_snapshot(
                dest_proj, exp.composer_id, exp.staged_project
            )
    staged.promoted = True
    staged.lease.release()
    staged.lease = None
    return staged.count
