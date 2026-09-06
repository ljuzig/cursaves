"""Read-only v17↔v18 compatibility lineage check (v0.9.16).

This module never writes Cursor identifiers, snapshots, or SQLite rows.
It is a comparator used only after the prefix comparator returns
DIVERGED, and only for an observed modern ``_v`` 17↔18 crossing or the
transitive ``18↔18`` handoff produced by that crossing.

Direction preserves the original permissive directionality, refined
by novel-content accounting. Semantic analysis is a veto for real
forks, not a proof that every migration detail stayed linearly
equivalent:

* ancestor + novel content only locally  -> LOCAL_AHEAD
* ancestor + novel content only remotely -> BEHIND
* ancestor + no novel content            -> UP_TO_DATE
* ancestor + novel content on both sides -> FORK
* direct semantic contradiction          -> FORK

Retirement, v18 rekeying, context pruning, and a source tip that is no
longer active are not forks by themselves. Same-count histories that
are not semantically equivalent still fork. No percentage match.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from . import importer, syncstate


# Compatibility only drops proven identity keys. Header layout fields
# (grouping, contentHeightHint) are already removed by _semantic_header.
# createdAt / serverBubbleId are already removed by _normalize_unit_object.
# Bubble-root grouping and contentHeightHint stay semantic: they were
# never proven non-semantic on bodies (v0.9.14 header-only invariant).
_COMPAT_HEADER_IDENTITY_KEYS = frozenset({"bubbleId"})
_COMPAT_BUBBLE_IDENTITY_KEYS = frozenset({"bubbleId"})

_COMPAT_FROM = 17
_COMPAT_TO = 18
_FIRST_CHANGED = 8


@dataclass(frozen=True)
class CompatUnit:
    """One active-history unit. Cursor IDs are kept only for diagnostics."""

    index: int
    bubble_id: str
    fingerprint: str
    canonical_hash: str
    has_body: bool
    payload: dict[str, Any] = field(default_factory=dict)
    diagnostic_fingerprint: str = ""


@dataclass
class LineageReport:
    """Read-only comparison of local vs snapshot logical units."""

    composer_id: str = ""
    local_schema_v: Optional[int] = None
    snapshot_schema_v: Optional[int] = None
    compatibility_candidate: bool = False
    snapshot_active_units: int = 0
    local_active_units: int = 0
    exact_logical_matches: int = 0
    source_missing_units: int = 0
    destination_extra_units: int = 0
    ambiguous_duplicate_fingerprints: list[str] = field(default_factory=list)
    first_changed_logical_units: list[dict] = field(default_factory=list)
    extras_before: int = 0
    extras_inside: int = 0
    extras_after: int = 0
    common_bubble_ids: int = 0
    common_bubble_ids_different_canonical: int = 0
    common_bubble_ids_different_logical: int = 0
    common_bubble_semantic_diffs: list[dict] = field(default_factory=list)
    snapshot_bodies_preserved_locally: bool = False
    local_bodies_preserved_in_snapshot: bool = False
    mapping_direction: Optional[str] = None
    lineage_relation: Optional[str] = None
    mapping_unique: bool = False
    body_candidates_zero: int = 0
    body_candidates_unique: int = 0
    body_candidates_ambiguous: int = 0
    unique_replacements_header_only: int = 0
    unique_replacements_body_diff: int = 0
    structural_categories: dict[str, int] = field(default_factory=dict)
    header_path_histogram: dict[str, int] = field(default_factory=dict)
    physical_snapshot_active_body_ids: int = 0
    physical_same_id_present: int = 0
    physical_same_id_unchanged: int = 0
    physical_same_id_changed: int = 0
    physical_same_body_other_id: int = 0
    physical_unaccounted: int = 0
    forensic_unexplained_units: int = 0
    retired_physically_preserved: int = 0
    active_exact_preserved: int = 0
    source_tip_preserved: bool = False
    destination_extras_after_source_tip: int = 0
    active_represented: int = 0
    retired_preserved: int = 0
    body_rewritten: int = 0
    unaccounted_missing: int = 0
    logical_source_tip_preserved: bool = False
    extras_before_anchor: int = 0
    extras_inside_anchor: int = 0
    extras_after_anchor: int = 0
    migration_equivalent_active_rewrites: int = 0
    semantically_changed_active: int = 0
    changed_physical_body_paths: dict[str, int] = field(default_factory=dict)
    changed_physical_body_samples: list[dict] = field(default_factory=list)
    proof_trace: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProofTrace:
    """Read-only walk of every compatibility gate. Never used to classify."""

    common_id_exact: int = 0
    common_id_context_pruned: int = 0
    common_id_rejected: int = 0
    rejected_header_mismatch: int = 0
    rejected_blobs_mismatch: int = 0
    rejected_context_not_subset: int = 0
    rejected_other_body_mismatch: int = 0
    rejected_duplicate_id: int = 0
    anchors_monotone: Optional[bool] = None
    tip_exact_candidates: int = 0
    tip_prune_candidates: int = 0
    tip_found: bool = False
    embedding_status: str = "not-attempted"
    failed_segment: str = ""
    retirement_candidates: int = 0
    retirement_preserved: int = 0
    retirement_missing_body: int = 0
    retirement_changed_body: int = 0
    retirement_still_active: int = 0
    remap_candidates: int = 0
    remap_physical_ancestor_ok: int = 0
    remap_failed: int = 0
    final_reject_reason: str = ""
    blockers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def schema_version(composer_data: Any) -> Optional[int]:
    if not isinstance(composer_data, dict):
        return None
    if "composerData" in composer_data and isinstance(composer_data["composerData"], dict):
        composer_data = composer_data["composerData"]
    value = composer_data.get("_v")
    return value if isinstance(value, int) else None


def is_compatibility_lineage_candidate(local_data: Any, remote_data: Any) -> bool:
    """True for the observed v17↔v18 crossing or its v18↔v18 handoff.

    ``18↔18`` is allowed so the second device can pull after the first
    push promoted a v18 snapshot. ``17↔17`` and any other pair stay
    on the canonical comparator.
    """
    if not isinstance(local_data, dict) or not isinstance(remote_data, dict):
        return False
    local_v = schema_version(local_data)
    remote_v = schema_version(remote_data)
    allowed = (
        {local_v, remote_v} == {_COMPAT_FROM, _COMPAT_TO}
        or (local_v == _COMPAT_TO and remote_v == _COMPAT_TO)
    )
    if not allowed:
        return False
    for raw in (local_data, remote_data):
        data = raw.get("composerData", raw) if isinstance(raw, dict) else raw
        if not isinstance(data, dict):
            return False
        headers = data.get("fullConversationHeadersOnly")
        if not isinstance(headers, list) or not headers:
            return False
        if "conversation" in data and "fullConversationHeadersOnly" not in data:
            return False
    return True


def _strip_compat_keys(obj: Any, keys: frozenset[str]) -> Any:
    if not isinstance(obj, dict):
        return obj
    return {key: value for key, value in obj.items() if key not in keys}


_MIGRATION_RUNTIME_ROOT_KEYS = frozenset({
    "conversationState",
    "conversationTurnIndex",
    "contextWindowStatusAtCreation",
    "modelInfo",
    "requestId",
})


def _stable_json_key(value: Any) -> str:
    try:
        return syncstate._canonical_json(value)
    except Exception:
        return str(value)


def _normalize_image_path(path: Any) -> Any:
    if not isinstance(path, str):
        return path
    uuid_pattern = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    return re.sub(rf"({uuid_pattern})-({uuid_pattern})", r"\1", path)


def _normalize_image_entry(img: Any) -> Any:
    if not isinstance(img, dict):
        return img
    out = dict(img)
    out.pop("loadedAt", None)
    out.pop("uuid", None)
    if "path" in out:
        out["path"] = _normalize_image_path(out["path"])
    return out


def _project_logical_context(body: dict) -> dict[str, Any]:
    """Project v17/v18 context into a stable canonical logical structure.

    Preserves all semantic payloads:
      - external link URLs
      - file selection URIs/paths
      - code/text selection URIs, ranges, and text
      - selected image stable paths/types
      - non-empty unrecognized / residual context and mentions fields
    Drops UI mention duplicates, empty collection templates, and materialization UUIDs.
    """
    ctx = body.get("context")
    if not isinstance(ctx, dict):
        ctx = {}
    mentions = body.get("mentions")
    if not isinstance(mentions, dict):
        mentions = ctx.get("mentions") if isinstance(ctx.get("mentions"), dict) else {}

    proj: dict[str, Any] = {}

    # 1. External Links
    raw_links: list[Any] = []
    if isinstance(body.get("externalLinks"), list):
        raw_links.extend(body["externalLinks"])
    if isinstance(ctx.get("externalLinks"), list):
        raw_links.extend(ctx["externalLinks"])

    all_mentions: list[dict] = []
    if isinstance(body.get("mentions"), dict):
        all_mentions.append(body["mentions"])
    if isinstance(ctx.get("mentions"), dict):
        all_mentions.append(ctx["mentions"])

    for m in all_mentions:
        ext_m = m.get("externalLinks")
        if isinstance(ext_m, dict):
            for k, v in ext_m.items():
                if isinstance(k, str) and k:
                    raw_links.append({"url": k})
                if isinstance(v, dict) and "url" in v and v["url"]:
                    raw_links.append({"url": v["url"]})
                elif isinstance(v, str) and v:
                    raw_links.append({"url": v})
        elif isinstance(ext_m, list):
            raw_links.extend(ext_m)

    links: list[dict[str, Any]] = []
    seen_links: set[str] = set()
    for item in raw_links:
        if isinstance(item, dict):
            url = item.get("url")
            entry = {"url": url if url is not None else item}
        elif item is not None:
            entry = {"url": item}
        else:
            continue
        key = _stable_json_key(entry)
        if key not in seen_links:
            seen_links.add(key)
            links.append(entry)
    if links:
        links.sort(key=_stable_json_key)
        proj["externalLinks"] = links

    # 2. File Selections
    raw_files: list[Any] = []
    if isinstance(body.get("fileSelections"), list):
        raw_files.extend(body["fileSelections"])
    if isinstance(ctx.get("fileSelections"), list):
        raw_files.extend(ctx["fileSelections"])

    for m in all_mentions:
        fs_m = m.get("fileSelections")
        if isinstance(fs_m, dict):
            for k, v in fs_m.items():
                if isinstance(k, str) and k:
                    raw_files.append({"uri": k})
                if isinstance(v, dict):
                    uri_v = v.get("uri") or v.get("path") or v.get("fsPath")
                    if uri_v:
                        raw_files.append({"uri": uri_v})
                elif isinstance(v, str) and v:
                    raw_files.append({"uri": v})
        elif isinstance(fs_m, list):
            raw_files.extend(fs_m)

    files: list[dict[str, Any]] = []
    seen_files: set[str] = set()
    for item in raw_files:
        if isinstance(item, dict):
            uri = item.get("uri")
            if uri is None:
                uri = item.get("path")
            if uri is None:
                uri = item.get("fsPath")
            entry = {"uri": uri if uri is not None else item}
        elif item is not None:
            entry = {"uri": item}
        else:
            continue
        key = _stable_json_key(entry)
        if key not in seen_files:
            seen_files.add(key)
            files.append(entry)
    if files:
        files.sort(key=_stable_json_key)
        proj["fileSelections"] = files

    # 3. Selections (Code/text selections)
    raw_sels: list[Any] = []
    if isinstance(body.get("selections"), list):
        raw_sels.extend(body["selections"])
    if isinstance(ctx.get("selections"), list):
        raw_sels.extend(ctx["selections"])

    for m in all_mentions:
        sel_m = m.get("selections")
        if isinstance(sel_m, dict):
            for k, v in sel_m.items():
                if isinstance(k, str):
                    parsed = None
                    if k.startswith("{") and k.endswith("}"):
                        try:
                            parsed = json.loads(k)
                        except Exception:
                            parsed = None
                    if isinstance(parsed, dict):
                        raw_sels.append(parsed)
                    elif k:
                        raw_sels.append({"text": k})
                if isinstance(v, dict):
                    if any(f in v for f in ("uri", "path", "range", "text")):
                        raw_sels.append(v)
        elif isinstance(sel_m, list):
            raw_sels.extend(sel_m)

    sels: list[dict[str, Any]] = []
    seen_sels: set[str] = set()
    for item in raw_sels:
        entry: dict[str, Any] = {}
        if isinstance(item, dict):
            if "uri" in item and item["uri"] is not None:
                entry["uri"] = item["uri"]
            elif "path" in item and item["path"] is not None:
                entry["uri"] = item["path"]
            if "range" in item and item["range"] is not None:
                entry["range"] = item["range"]
            if "text" in item and item["text"] is not None:
                entry["text"] = item["text"]
            for k, val in item.items():
                if k not in ("uri", "path", "range", "text") and val is not None:
                    entry[k] = val
        elif isinstance(item, str):
            entry = {"text": item}
        elif item is not None:
            entry = {"value": item}
        if entry:
            key = _stable_json_key(entry)
            if key not in seen_sels:
                seen_sels.add(key)
                sels.append(entry)
    if sels:
        sels.sort(key=_stable_json_key)
        proj["selections"] = sels

    # 4. Selected Images
    raw_imgs: list[Any] = []
    if isinstance(body.get("selectedImages"), list):
        raw_imgs.extend(body["selectedImages"])
    if isinstance(ctx.get("selectedImages"), list):
        raw_imgs.extend(ctx["selectedImages"])

    imgs: list[dict[str, Any]] = []
    seen_imgs: set[str] = set()
    for item in raw_imgs:
        if isinstance(item, dict):
            entry = _normalize_image_entry(item)
        elif item is not None:
            entry = {"value": item}
        else:
            continue
        key = _stable_json_key(entry)
        if key not in seen_imgs:
            seen_imgs.add(key)
            imgs.append(entry)
    if imgs:
        imgs.sort(key=_stable_json_key)
        proj["selectedImages"] = imgs

    # 5. Residual / unknown non-empty fields in context and mentions
    handled_ctx = {
        "externalLinks",
        "fileSelections",
        "selections",
        "selectedImages",
        "mentions",
    }
    residual: dict[str, Any] = {}
    for k, v in ctx.items():
        if k not in handled_ctx and v is not None and v != [] and v != {} and v != "":
            residual[k] = v

    handled_mentions = {
        "externalLinks",
        "fileSelections",
        "selections",
    }
    for k, v in mentions.items():
        if k not in handled_mentions and v is not None and v != [] and v != {} and v != "":
            residual[f"mentions.{k}"] = v

    if residual:
        proj["residual"] = {k: residual[k] for k in sorted(residual)}

    return proj


def migration_normalize_body(bubble: Any) -> Any:
    """Normalize a bubble body for v17/v18 compatibility comparison.

    Never used for canonical v5 digests. Strips proven runtime keys,
    normalizes default values (isPlanExecution, isSteer), and projects
    context / mentions into a canonical logical representation.
    """
    if not isinstance(bubble, dict):
        return bubble
    norm = syncstate._normalize_unit_object(bubble, top=True)
    if not isinstance(norm, dict):
        return norm
    body = dict(norm)
    body.pop("bubbleId", None)

    for k in _MIGRATION_RUNTIME_ROOT_KEYS:
        body.pop(k, None)

    if body.get("isPlanExecution") is False:
        body.pop("isPlanExecution", None)

    body.pop("isSteer", None)

    canonical_ctx = _project_logical_context(body)
    body.pop("context", None)
    body.pop("mentions", None)
    body.pop("externalLinks", None)
    body.pop("fileSelections", None)
    body.pop("selections", None)
    body.pop("selectedImages", None)
    if canonical_ctx:
        body["context"] = canonical_ctx

    return body


_PRUNABLE_CONTEXT_KEYS = frozenset({
    "externalLinks",
    "fileSelections",
    "selections",
})


def _payload_minus_prunable(payload: dict[str, Any]) -> dict[str, Any]:
    """Logical payload with the three migration-prunable collections removed."""
    bubble = payload.get("bubble")
    if not isinstance(bubble, dict):
        return {"header": payload.get("header"), "blobs": payload.get("blobs"), "bubble": bubble}
    value = bubble.get("value")
    if not isinstance(value, dict):
        return {
            "header": payload.get("header"),
            "blobs": payload.get("blobs"),
            "bubble": bubble,
        }
    ctx = value.get("context")
    if isinstance(ctx, dict):
        rest_ctx = {key: val for key, val in ctx.items() if key not in _PRUNABLE_CONTEXT_KEYS}
        rest_value = {key: val for key, val in value.items() if key != "context"}
        if rest_ctx:
            rest_value = {**rest_value, "context": rest_ctx}
    else:
        rest_value = value
    return {
        "header": payload.get("header"),
        "blobs": payload.get("blobs"),
        "bubble": {**bubble, "value": rest_value},
    }


def _prunable_collections(payload: dict[str, Any]) -> Optional[dict[str, list[Any]]]:
    bubble = payload.get("bubble")
    if not isinstance(bubble, dict) or bubble.get("state") != "present":
        return None
    value = bubble.get("value")
    if not isinstance(value, dict):
        return None
    ctx = value.get("context")
    if not isinstance(ctx, dict):
        ctx = {}
    out: dict[str, list[Any]] = {}
    for key in _PRUNABLE_CONTEXT_KEYS:
        items = ctx.get(key)
        if items is None:
            out[key] = []
        elif isinstance(items, list):
            out[key] = items
        else:
            return None
    return out


def _collection_is_subset(dest_items: list[Any], source_items: list[Any]) -> bool:
    dest_keys = {_stable_json_key(item) for item in dest_items}
    source_keys = {_stable_json_key(item) for item in source_items}
    return dest_keys <= source_keys


def _same_id_reject_category(source: CompatUnit, dest: CompatUnit) -> str:
    """Why a same-ID pair is not exact and not ``MIGRATION_CONTEXT_PRUNED``."""
    src_core = _payload_minus_prunable(source.payload)
    dst_core = _payload_minus_prunable(dest.payload)
    if src_core.get("header") != dst_core.get("header"):
        return "header mismatch"
    if src_core.get("blobs") != dst_core.get("blobs"):
        return "blobs mismatch"
    src_cols = _prunable_collections(source.payload)
    dst_cols = _prunable_collections(dest.payload)
    subset_ok = (
        src_cols is not None
        and dst_cols is not None
        and all(
            _collection_is_subset(dst_cols[key], src_cols[key])
            for key in _PRUNABLE_CONTEXT_KEYS
        )
    )
    if src_core != dst_core:
        return "other body mismatch"
    if not subset_ok:
        return "context not subset"
    return "other body mismatch"


def is_migration_context_pruned(source: CompatUnit, dest: CompatUnit) -> bool:
    """True for a same-ID v17→v18 monotone prune of the three context collections.

    Everything outside ``externalLinks`` / ``fileSelections`` / ``selections``
    must be identical, including header, text/type/role, tool I/O,
    ``toolCallId``, blob refs, ``selectedImages``, and residual context.
    Destination collections must be subsets of the source collections.
    """
    if source.bubble_id != dest.bubble_id:
        return False
    if source.fingerprint == dest.fingerprint:
        return True
    if _payload_minus_prunable(source.payload) != _payload_minus_prunable(dest.payload):
        return False
    src_cols = _prunable_collections(source.payload)
    dst_cols = _prunable_collections(dest.payload)
    if src_cols is None or dst_cols is None:
        return False
    for key in _PRUNABLE_CONTEXT_KEYS:
        if not _collection_is_subset(dst_cols[key], src_cols[key]):
            return False
    return True


def _has_shared_physical_ancestor(
    source_active_ids: set[str],
    dest_active_ids: set[str],
    source_digests: Optional[dict[str, str]],
    dest_digests: Optional[dict[str, str]],
) -> bool:
    """A leftover body present on both sides with the same compat digest.

    The identifier must be physical on both sides and non-active on at
    least one side. This is the v18↔v18 ancestry proof that a
    content-equivalent retry without leftovers cannot satisfy.
    """
    if not source_digests or not dest_digests:
        return False
    for bid, digest in source_digests.items():
        other = dest_digests.get(bid)
        if other is None or other != digest:
            continue
        if bid not in source_active_ids or bid not in dest_active_ids:
            return True
    return False


_UI_METADATA_KEYS = frozenset({"grouping", "contentHeightHint"})


def _strip_ui_metadata(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    return {key: value for key, value in obj.items() if key not in _UI_METADATA_KEYS}


def _compat_body_digest(bubble: dict, *, conversational: bool = False) -> str:
    body = migration_normalize_body(bubble)
    del conversational
    return syncstate._sha256_text(syncstate._canonical_json(body))


def logical_unit_payload(
    header: dict,
    bubble: Optional[dict],
    blobs: dict[str, Any],
) -> dict[str, Any]:
    """Canonical unit payload minus proven identity keys only."""
    header = _strip_compat_keys(
        syncstate._normalize_unit_object(syncstate._semantic_header(header), top=True),
        _COMPAT_HEADER_IDENTITY_KEYS,
    )
    if bubble is None:
        bubble_part: dict[str, Any] = {"state": "missing"}
        ref_payload: Any = header
    else:
        bubble_part = {
            "state": "present",
            "value": migration_normalize_body(bubble),
        }
        ref_payload = (header, bubble)
    payload: dict[str, Any] = {"header": header, "bubble": bubble_part}
    refs = syncstate._referenced_blob_ids(ref_payload, set(blobs))
    if refs:
        payload["blobs"] = {
            ref: syncstate._sha256_bytes(syncstate._blob_bytes(blobs[ref]))
            for ref in sorted(refs)
        }
    return payload


def logical_fingerprint(
    header: dict,
    bubble: Optional[dict],
    blobs: dict[str, Any],
) -> str:
    return syncstate._sha256_text(
        syncstate._canonical_json(logical_unit_payload(header, bubble, blobs))
    )


_DIAGNOSTIC_HEADER_KEYS = frozenset({"type", "role"})


def diagnostic_unit_payload(
    header: dict,
    bubble: Optional[dict],
    blobs: dict[str, Any],
) -> dict[str, Any]:
    """Weaker diagnostic key. Never used to classify.

    Keeps semantic bubble body, role/type, toolCallId, tool I/O, and
    required blob digests. Drops only identity/transport fields already
    proven non-semantic.
    """
    header_n = _strip_compat_keys(
        syncstate._normalize_unit_object(syncstate._semantic_header(header), top=True),
        _COMPAT_HEADER_IDENTITY_KEYS,
    )
    role = {}
    if isinstance(header_n, dict):
        role = {
            key: header_n[key] for key in _DIAGNOSTIC_HEADER_KEYS if key in header_n
        }
    if bubble is None:
        bubble_part: dict[str, Any] = {"state": "missing"}
        ref_payload: Any = header
    else:
        body = _strip_compat_keys(
            syncstate._normalize_unit_object(bubble, top=True),
            _COMPAT_BUBBLE_IDENTITY_KEYS,
        )
        if isinstance(body, dict):
            for key in _DIAGNOSTIC_HEADER_KEYS:
                if key in body and key not in role:
                    role[key] = body[key]
        bubble_part = {"state": "present", "value": body}
        ref_payload = (header, bubble)
    payload: dict[str, Any] = {"role": role, "bubble": bubble_part}
    refs = syncstate._referenced_blob_ids(ref_payload, set(blobs))
    if refs:
        payload["blobs"] = {
            ref: syncstate._sha256_bytes(syncstate._blob_bytes(blobs[ref]))
            for ref in sorted(refs)
        }
    return payload


def diagnostic_fingerprint(
    header: dict,
    bubble: Optional[dict],
    blobs: dict[str, Any],
) -> str:
    return syncstate._sha256_text(
        syncstate._canonical_json(diagnostic_unit_payload(header, bubble, blobs))
    )


def make_compat_unit(
    index: int,
    header: dict,
    bubble: Optional[dict],
    blobs: dict[str, Any],
) -> CompatUnit:
    bid = header.get("bubbleId")
    if not bid:
        raise syncstate.ClassifyError("header is missing bubbleId")
    payload = logical_unit_payload(header, bubble, blobs)
    return CompatUnit(
        index=index,
        bubble_id=str(bid),
        fingerprint=syncstate._sha256_text(syncstate._canonical_json(payload)),
        canonical_hash=syncstate.unit_hash(header, bubble, blobs),
        has_body=bubble is not None,
        payload=payload,
        diagnostic_fingerprint=diagnostic_fingerprint(header, bubble, blobs),
    )


def snapshot_compat_units(snapshot: dict) -> list[CompatUnit]:
    composer = snapshot.get("composerData")
    if not isinstance(composer, dict):
        raise syncstate.ClassifyError("composerData is not an object")
    blobs = snapshot.get("contentBlobs") or {}
    if not isinstance(blobs, dict):
        blobs = {}
    units: list[CompatUnit] = []
    for index, header in enumerate(syncstate._headers(composer)):
        bid = header.get("bubbleId")
        if not bid:
            raise syncstate.ClassifyError("header is missing bubbleId")
        bubble = syncstate._bubble_from_snapshot(snapshot, bid)
        sem_header = syncstate._semantic_header(header)
        ref_payload: Any = sem_header if bubble is None else (sem_header, bubble)
        refs = syncstate._required_blob_ids(ref_payload, set(blobs))
        for ref in refs:
            if ref not in blobs:
                raise syncstate.ClassifyError(f"referenced blob missing: {ref}")
        unit_blobs = {ref: blobs[ref] for ref in refs}
        units.append(make_compat_unit(index, header, bubble, unit_blobs))
    return units


def local_compat_units(
    session: "syncstate.SyncReadSession", composer_id: str
) -> list[CompatUnit]:
    present, data = session.composer_cell(composer_id)
    if not present or not isinstance(data, dict):
        raise syncstate.ClassifyError("local composerData is unreadable")
    if "fullConversationHeadersOnly" not in data:
        raise syncstate.ClassifyError("local composerData is not a modern header list")
    available = session._available_blob_ids()
    discovered: set[str] = set()
    units: list[CompatUnit] = []
    for index, (header, bubble, blobs) in enumerate(
        session._iter_local_modern_units(composer_id, data, available, discovered)
    ):
        units.append(make_compat_unit(index, header, bubble, blobs))
    return units


def snapshot_stored_body_ids(snapshot: dict) -> list[str]:
    """Body IDs physically stored in the snapshot document."""
    ids: list[str] = []
    seen: set[str] = set()
    entries = snapshot.get("bubbleEntries")
    if isinstance(entries, dict):
        for key, value in entries.items():
            if isinstance(value, dict):
                bid = str(key)
                if bid not in seen:
                    seen.add(bid)
                    ids.append(bid)
    composer = snapshot.get("composerData")
    cmap = composer.get("conversationMap") if isinstance(composer, dict) else None
    if isinstance(cmap, dict):
        for key, value in cmap.items():
            if isinstance(value, dict):
                bid = str(key)
                if bid not in seen:
                    seen.add(bid)
                    ids.append(bid)
    return ids


def _parse_bubble_object(raw: Any) -> Optional[dict]:
    if raw is None:
        return None
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        parsed = json.loads(raw)
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def load_local_physical_digests(
    session: "syncstate.SyncReadSession",
    composer_id: str,
    wanted: Optional[set[str]] = None,
    *,
    conversational: bool = False,
) -> dict[str, str]:
    """Stream ``bubbleId:<cid>:*``. ``wanted is None`` keeps every digest."""
    digests: dict[str, str] = {}
    if wanted is not None and not wanted:
        return digests
    if session._cdb is None:
        return digests
    prefix = f"bubbleId:{composer_id}:"
    remaining = set(wanted) if wanted is not None else None
    try:
        conn = session._cdb._reader_conn()
        cursor = conn.execute(
            "SELECT key, value FROM cursorDiskKV WHERE key LIKE ?",
            (prefix + "%",),
        )
        for key, raw in cursor:
            if remaining is not None and not remaining:
                break
            if not isinstance(key, str) or not key.startswith(prefix):
                continue
            bid = key[len(prefix):]
            if remaining is not None and bid not in remaining:
                continue
            parsed = _parse_bubble_object(raw)
            if parsed is None:
                continue
            digests[bid] = _compat_body_digest(parsed, conversational=conversational)
            if remaining is not None:
                remaining.discard(bid)
    except Exception:
        return digests
    need_map = remaining is None or bool(remaining)
    if need_map:
        try:
            data = session.composer_data(composer_id)
        except syncstate.ClassifyError:
            return digests
        cmap = data.get("conversationMap") if isinstance(data, dict) else None
        if isinstance(cmap, dict):
            for bid, value in cmap.items():
                key = str(bid)
                if remaining is not None and key not in remaining:
                    continue
                if key in digests or not isinstance(value, dict):
                    continue
                digests[key] = _compat_body_digest(value, conversational=conversational)
                if remaining is not None:
                    remaining.discard(key)
    return digests


def snapshot_history_physically_intact_locally(
    session: "syncstate.SyncReadSession",
    composer_id: str,
    snapshot: dict,
    local_digests: Optional[dict[str, str]] = None,
) -> bool:
    """Every snapshot-stored body still exists locally with the same semantics.

    Necessary for treating local as an append-only continuation of the
    snapshot. Presence of leftover rows alone is not enough; a modified
    leftover body fails closed.
    """
    ids = snapshot_stored_body_ids(snapshot)
    if not ids:
        return False
    if local_digests is None:
        local_digests = load_local_physical_digests(session, composer_id, set(ids))
    for bid in ids:
        snap_bubble = syncstate._bubble_from_snapshot(snapshot, bid)
        local_digest = local_digests.get(bid)
        if snap_bubble is None or local_digest is None:
            return False
        if _compat_body_digest(snap_bubble) != local_digest:
            return False
    return True


def local_history_physically_intact_in_snapshot(
    session: "syncstate.SyncReadSession",
    composer_id: str,
    snapshot: dict,
    local_units: list[CompatUnit],
    local_digests: Optional[dict[str, str]] = None,
) -> bool:
    """Every local active body exists in the snapshot with the same semantics.

    Necessary for treating the snapshot as an append-only continuation
    of local (typical after the other machine exported leftover rows).
    """
    active = [unit for unit in local_units if unit.has_body]
    if not active:
        return False
    wanted = {unit.bubble_id for unit in active}
    if local_digests is None:
        local_digests = load_local_physical_digests(session, composer_id, wanted)
    for unit in active:
        snap_bubble = syncstate._bubble_from_snapshot(snapshot, unit.bubble_id)
        local_digest = local_digests.get(unit.bubble_id)
        if snap_bubble is None or local_digest is None:
            return False
        if _compat_body_digest(snap_bubble) != local_digest:
            return False
    return True


def _leftmost_map(source: list[str], dest: list[str]) -> Optional[list[int]]:
    positions: dict[str, deque[int]] = defaultdict(deque)
    for index, fingerprint in enumerate(dest):
        positions[fingerprint].append(index)
    mapping: list[int] = []
    last = -1
    for fingerprint in source:
        queue = positions.get(fingerprint)
        if not queue:
            return None
        while queue and queue[0] <= last:
            queue.popleft()
        if not queue:
            return None
        found = queue.popleft()
        mapping.append(found)
        last = found
    return mapping


def _rightmost_map(source: list[str], dest: list[str]) -> Optional[list[int]]:
    positions: dict[str, deque[int]] = defaultdict(deque)
    for index, fingerprint in enumerate(dest):
        positions[fingerprint].append(index)
    mapping = [0] * len(source)
    nxt = len(dest)
    for i in range(len(source) - 1, -1, -1):
        fingerprint = source[i]
        queue = positions.get(fingerprint)
        if not queue:
            return None
        while queue and queue[-1] >= nxt:
            queue.pop()
        if not queue:
            return None
        found = queue.pop()
        mapping[i] = found
        nxt = found
    return mapping


def unique_subsequence_map(
    source: list[str], dest: list[str]
) -> tuple[Optional[list[int]], list[str]]:
    """Leftmost monotone map if it is the unique embedding.

    Returns ``(mapping, ambiguous_fingerprints)``. ``mapping is None``
    means a source unit could not be placed. Non-empty ambiguous list
    means more than one monotone embedding exists.
    """
    if not source:
        return ([], [])
    left = _leftmost_map(source, dest)
    if left is None:
        return (None, [])
    right = _rightmost_map(source, dest)
    if right is None:
        return (None, [])
    if left != right:
        ambiguous = sorted({
            source[i] for i, (lo, hi) in enumerate(zip(left, right)) if lo != hi
        })
        return (None, ambiguous)
    return (left, [])


def _extras(
    mapping: list[int], dest_len: int
) -> tuple[list[int], list[int], list[int]]:
    used = set(mapping)
    extras = [i for i in range(dest_len) if i not in used]
    if not mapping:
        return extras, [], []
    lo, hi = mapping[0], mapping[-1]
    before = [i for i in extras if i < lo]
    inside = [i for i in extras if lo < i < hi]
    after = [i for i in extras if i > hi]
    return before, inside, after


def _empty_direction() -> tuple[
    Optional[syncstate.SyncRelation],
    Optional[list[int]],
    list[str],
    list[CompatUnit],
    list[CompatUnit],
]:
    return None, None, [], [], []


def _source_body_digest(
    unit: CompatUnit, source_digests: Optional[dict[str, str]]
) -> Optional[str]:
    if source_digests and unit.bubble_id in source_digests:
        return source_digests[unit.bubble_id]
    return _payload_body_digest(unit.payload)


def _evaluate_lineage_direction(
    source: list[CompatUnit],
    dest: list[CompatUnit],
    dest_digests: Optional[dict[str, str]],
    source_digests: Optional[dict[str, str]] = None,
    *,
    source_v: Optional[int] = None,
    dest_v: Optional[int] = None,
) -> tuple[Optional[syncstate.SyncRelation], Optional[list[int]], list[str], list[CompatUnit], list[CompatUnit]]:
    """Evaluate if dest is an append-only continuation of source.

    Physical proof is per-unit. There is no global same-ID digest gate.

      ACTIVE_REPRESENTED — unique logical embedding; same physical ID
        is not required. v17→v18 remaps still require the old source
        ID as a leftover body on dest. v18↔v18 remaps require a shared
        physical ancestor leftover instead.
      RETIRED_PRESERVED — old ID physically present on dest with a
        compatible digest, and not active in dest.
      MIGRATION_CONTEXT_PRUNED — same-ID v17→v18 monotone loss in the
        three observed context collections; treated as a hard anchor.
    """
    if not source or not dest:
        return _empty_direction()

    allow_prune = source_v == _COMPAT_FROM and dest_v == _COMPAT_TO
    both_v18 = source_v == _COMPAT_TO and dest_v == _COMPAT_TO

    s_by_bid: dict[str, list[int]] = defaultdict(list)
    for i, u in enumerate(source):
        s_by_bid[u.bubble_id].append(i)

    d_by_bid: dict[str, list[int]] = defaultdict(list)
    for j, u in enumerate(dest):
        d_by_bid[u.bubble_id].append(j)

    common_bids = set(s_by_bid) & set(d_by_bid)

    anchors: list[tuple[int, int]] = []
    prune_pairs: set[tuple[int, int]] = set()
    for bid in common_bids:
        s_indices = s_by_bid[bid]
        d_indices = d_by_bid[bid]
        if len(s_indices) != 1 or len(d_indices) != 1:
            return _empty_direction()
        s_idx = s_indices[0]
        d_idx = d_indices[0]
        if source[s_idx].fingerprint == dest[d_idx].fingerprint:
            anchors.append((s_idx, d_idx))
        elif allow_prune and is_migration_context_pruned(source[s_idx], dest[d_idx]):
            anchors.append((s_idx, d_idx))
            prune_pairs.add((s_idx, d_idx))
        else:
            return _empty_direction()

    anchors.sort(key=lambda p: p[0])

    for k in range(len(anchors) - 1):
        if anchors[k][1] >= anchors[k + 1][1]:
            return _empty_direction()

    tip_unit = source[-1]
    tip_fp = tip_unit.fingerprint

    if anchors and anchors[-1][0] == len(source) - 1:
        tip_candidates = [anchors[-1][1]]
    else:
        min_d = anchors[-1][1] if anchors else 0
        tip_candidates = [
            j for j in range(min_d, len(dest)) if dest[j].fingerprint == tip_fp
        ]

    if not tip_candidates:
        return _empty_direction()

    valid_results: list[tuple[int, list[int], set[int]]] = []
    ambiguous_fps: list[str] = []

    for t in tip_candidates:
        if any(d_idx > t for _, d_idx in anchors):
            continue

        points = [(-1, -1)] + [p for p in anchors if p[1] <= t]
        if points[-1] != (len(source) - 1, t):
            points.append((len(source) - 1, t))

        segment_ok = True
        h_to_s = [0] * (t + 1)

        for k in range(len(points) - 1):
            s_prev, d_prev = points[k]
            s_curr, d_curr = points[k + 1]

            sub_dest = [dest[j].fingerprint for j in range(d_prev + 1, d_curr)]
            sub_source = [source[i].fingerprint for i in range(s_prev + 1, s_curr)]

            if sub_dest:
                sub_map, amb = unique_subsequence_map(sub_dest, sub_source)
                if sub_map is None or amb:
                    if amb:
                        ambiguous_fps.extend(amb)
                    segment_ok = False
                    break
                for j, mapped_s in enumerate(sub_map):
                    h_to_s[d_prev + 1 + j] = s_prev + 1 + mapped_s

            if 0 <= d_curr <= t:
                h_to_s[d_curr] = s_curr

        if not segment_ok:
            continue

        mapped_s_indices = set(h_to_s)
        retired_s_indices = set(range(len(source))) - mapped_s_indices
        dest_append_fps = {dest[j].fingerprint for j in range(t + 1, len(dest))}

        retired_ok = True
        for s_idx in retired_s_indices:
            u = source[s_idx]
            if u.bubble_id in d_by_bid or u.fingerprint in dest_append_fps:
                retired_ok = False
                break
            if dest_digests is None:
                retired_ok = False
                break
            d_digest = dest_digests.get(u.bubble_id)
            if d_digest is None:
                retired_ok = False
                break
            if u.has_body:
                src_d = _source_body_digest(u, source_digests)
                if src_d is None or src_d != d_digest:
                    retired_ok = False
                    break
        if not retired_ok:
            continue

        remapped = False
        remap_ok = True
        for d_idx, s_idx in enumerate(h_to_s):
            if source[s_idx].bubble_id == dest[d_idx].bubble_id:
                continue
            remapped = True
            if both_v18:
                continue
            if dest_digests is None:
                remap_ok = False
                break
            leftover = dest_digests.get(source[s_idx].bubble_id)
            src_d = _source_body_digest(source[s_idx], source_digests)
            if leftover is None or src_d is None or leftover != src_d:
                remap_ok = False
                break
        if not remap_ok:
            continue

        if both_v18 and remapped:
            if not _has_shared_physical_ancestor(
                {u.bubble_id for u in source},
                {u.bubble_id for u in dest},
                source_digests,
                dest_digests,
            ):
                continue

        dest_equiv_fps = []
        for j in range(t + 1):
            s_idx = h_to_s[j]
            if (s_idx, j) in prune_pairs:
                dest_equiv_fps.append(source[s_idx].fingerprint)
            else:
                dest_equiv_fps.append(dest[j].fingerprint)
        act_fps = [source[i].fingerprint for i in sorted(mapped_s_indices)]
        if not anchors:
            all_d_fps = [u.fingerprint for u in dest]
            all_d_map, all_d_amb = unique_subsequence_map(act_fps, all_d_fps)
            if all_d_map is None or all_d_amb:
                ambiguous_fps.extend(all_d_amb)
                continue
        else:
            d_map, d_amb = unique_subsequence_map(act_fps, dest_equiv_fps)
            if d_map is None or d_amb:
                ambiguous_fps.extend(d_amb)
                continue

        valid_results.append((t, h_to_s, retired_s_indices))

    if len(valid_results) == 0:
        return None, None, ambiguous_fps, [], []
    if len(valid_results) > 1:
        return None, None, [tip_fp], [], []

    t, h_to_s, retired_s_indices = valid_results[0]

    if t == len(dest) - 1:
        relation = syncstate.SyncRelation.UP_TO_DATE
    else:
        relation = syncstate.SyncRelation.LOCAL_AHEAD

    active_source = [source[i] for i in sorted(set(h_to_s))]
    retired_source = [source[i] for i in sorted(retired_s_indices)]

    return relation, h_to_s, [], active_source, retired_source


def _collect_proof_trace(
    source: list[CompatUnit],
    dest: list[CompatUnit],
    dest_digests: Optional[dict[str, str]],
    source_digests: Optional[dict[str, str]] = None,
    *,
    source_v: Optional[int] = None,
    dest_v: Optional[int] = None,
) -> ProofTrace:
    """Walk every compatibility gate. Does not classify and does not stop early."""
    trace = ProofTrace()
    if not source or not dest:
        trace.final_reject_reason = "empty source or dest"
        trace.blockers.append(trace.final_reject_reason)
        return trace

    allow_prune = source_v == _COMPAT_FROM and dest_v == _COMPAT_TO
    both_v18 = source_v == _COMPAT_TO and dest_v == _COMPAT_TO

    s_by_bid: dict[str, list[int]] = defaultdict(list)
    for i, u in enumerate(source):
        s_by_bid[u.bubble_id].append(i)
    d_by_bid: dict[str, list[int]] = defaultdict(list)
    for j, u in enumerate(dest):
        d_by_bid[u.bubble_id].append(j)

    anchors: list[tuple[int, int]] = []
    prune_pairs: set[tuple[int, int]] = set()
    for bid in set(s_by_bid) & set(d_by_bid):
        s_indices = s_by_bid[bid]
        d_indices = d_by_bid[bid]
        if len(s_indices) != 1 or len(d_indices) != 1:
            trace.common_id_rejected += 1
            trace.rejected_duplicate_id += 1
            trace.blockers.append(f"duplicate common bubbleId {bid[:12]}")
            continue
        s_idx = s_indices[0]
        d_idx = d_indices[0]
        if source[s_idx].fingerprint == dest[d_idx].fingerprint:
            trace.common_id_exact += 1
            anchors.append((s_idx, d_idx))
        elif allow_prune and is_migration_context_pruned(source[s_idx], dest[d_idx]):
            trace.common_id_context_pruned += 1
            anchors.append((s_idx, d_idx))
            prune_pairs.add((s_idx, d_idx))
        else:
            trace.common_id_rejected += 1
            category = _same_id_reject_category(source[s_idx], dest[d_idx])
            if category == "header mismatch":
                trace.rejected_header_mismatch += 1
            elif category == "blobs mismatch":
                trace.rejected_blobs_mismatch += 1
            elif category == "context not subset":
                trace.rejected_context_not_subset += 1
            else:
                trace.rejected_other_body_mismatch += 1
            trace.blockers.append(f"rejected common-ID {bid[:12]}: {category}")

    anchors.sort(key=lambda p: p[0])
    monotone = True
    for k in range(len(anchors) - 1):
        if anchors[k][1] >= anchors[k + 1][1]:
            monotone = False
            break
    trace.anchors_monotone = monotone
    if not monotone:
        trace.blockers.append("anchors cross")

    tip_unit = source[-1]
    tip_fp = tip_unit.fingerprint
    exact_tips = [j for j, unit in enumerate(dest) if unit.fingerprint == tip_fp]
    prune_tips = [
        j
        for j, unit in enumerate(dest)
        if allow_prune and is_migration_context_pruned(tip_unit, unit)
    ]
    trace.tip_exact_candidates = len(exact_tips)
    trace.tip_prune_candidates = len(prune_tips)

    if anchors and anchors[-1][0] == len(source) - 1:
        prod_tips = [anchors[-1][1]]
    else:
        min_d = anchors[-1][1] if anchors else 0
        prod_tips = [j for j in range(min_d, len(dest)) if dest[j].fingerprint == tip_fp]
    trace.tip_found = bool(prod_tips)
    if not trace.tip_found:
        trace.blockers.append("source tip not found")

    valid_mappings: list[tuple[int, list[int], set[int]]] = []
    embedding_seen = False
    for t in prod_tips:
        if any(d_idx > t for _, d_idx in anchors):
            continue
        points = [(-1, -1)] + [p for p in anchors if p[1] <= t]
        if points[-1] != (len(source) - 1, t):
            points.append((len(source) - 1, t))
        segment_ok = True
        h_to_s = [0] * (t + 1)
        for k in range(len(points) - 1):
            s_prev, d_prev = points[k]
            s_curr, d_curr = points[k + 1]
            sub_dest = [dest[j].fingerprint for j in range(d_prev + 1, d_curr)]
            sub_source = [source[i].fingerprint for i in range(s_prev + 1, s_curr)]
            if sub_dest:
                embedding_seen = True
                sub_map, amb = unique_subsequence_map(sub_dest, sub_source)
                if sub_map is None or amb:
                    segment_ok = False
                    if amb:
                        if trace.embedding_status != "ambiguous":
                            trace.embedding_status = "ambiguous"
                        if not trace.failed_segment:
                            trace.failed_segment = (
                                f"source[{s_prev + 1}:{s_curr}] "
                                f"dest[{d_prev + 1}:{d_curr}] ambiguous"
                            )
                        trace.blockers.append("historical embedding ambiguous")
                    else:
                        if trace.embedding_status not in {"ambiguous"}:
                            trace.embedding_status = "missing"
                        if not trace.failed_segment:
                            trace.failed_segment = (
                                f"source[{s_prev + 1}:{s_curr}] "
                                f"dest[{d_prev + 1}:{d_curr}] missing"
                            )
                        trace.blockers.append("historical embedding missing")
                    break
                for j, mapped_s in enumerate(sub_map):
                    h_to_s[d_prev + 1 + j] = s_prev + 1 + mapped_s
            if 0 <= d_curr <= t:
                h_to_s[d_curr] = s_curr
        if not segment_ok:
            continue
        if trace.embedding_status == "not-attempted":
            dest_equiv_fps = []
            mapped_s_indices = set(h_to_s)
            for j in range(t + 1):
                s_idx = h_to_s[j]
                if (s_idx, j) in prune_pairs:
                    dest_equiv_fps.append(source[s_idx].fingerprint)
                else:
                    dest_equiv_fps.append(dest[j].fingerprint)
            act_fps = [source[i].fingerprint for i in sorted(mapped_s_indices)]
            if not anchors:
                all_d_map, all_d_amb = unique_subsequence_map(
                    act_fps, [u.fingerprint for u in dest]
                )
                if all_d_map is None or all_d_amb:
                    embedding_seen = True
                    if all_d_amb:
                        trace.embedding_status = "ambiguous"
                        trace.blockers.append("historical embedding ambiguous")
                    else:
                        trace.embedding_status = "missing"
                        trace.blockers.append("historical embedding missing")
                    continue
            else:
                d_map, d_amb = unique_subsequence_map(act_fps, dest_equiv_fps)
                if d_map is None or d_amb:
                    embedding_seen = True
                    if d_amb:
                        trace.embedding_status = "ambiguous"
                        trace.blockers.append("historical embedding ambiguous")
                    else:
                        trace.embedding_status = "missing"
                        trace.blockers.append("historical embedding missing")
                    continue
            trace.embedding_status = "unique"
        valid_mappings.append((t, h_to_s, set(range(len(source))) - set(h_to_s)))

    if prod_tips and not embedding_seen and valid_mappings:
        trace.embedding_status = "unique"
    elif prod_tips and not valid_mappings and trace.embedding_status == "not-attempted":
        trace.embedding_status = "missing"
        trace.blockers.append("historical embedding missing")

    chosen = valid_mappings[0] if len(valid_mappings) == 1 else None
    if len(valid_mappings) > 1:
        trace.blockers.append("ambiguous source-tip placement")

    mapped_s: set[int] = set(chosen[1]) if chosen else {s for s, _ in anchors}
    retired_indices = (
        chosen[2] if chosen else set(range(len(source))) - mapped_s
    )
    dest_append_fps = set()
    if chosen:
        dest_append_fps = {dest[j].fingerprint for j in range(chosen[0] + 1, len(dest))}
    elif anchors:
        dest_append_fps = {
            dest[j].fingerprint for j in range(anchors[-1][1] + 1, len(dest))
        }

    for s_idx in sorted(retired_indices):
        u = source[s_idx]
        trace.retirement_candidates += 1
        if u.bubble_id in d_by_bid or u.fingerprint in dest_append_fps:
            trace.retirement_still_active += 1
            trace.blockers.append(f"retired unit still active {u.bubble_id[:12]}")
            continue
        if dest_digests is None:
            trace.retirement_missing_body += 1
            trace.blockers.append("retirement dest digests unavailable")
            continue
        d_digest = dest_digests.get(u.bubble_id)
        if d_digest is None:
            trace.retirement_missing_body += 1
            trace.blockers.append(f"retired body missing {u.bubble_id[:12]}")
            continue
        if u.has_body:
            src_d = _source_body_digest(u, source_digests)
            if src_d is None or src_d != d_digest:
                trace.retirement_changed_body += 1
                trace.blockers.append(f"retired body changed {u.bubble_id[:12]}")
                continue
        trace.retirement_preserved += 1

    ancestor_ok = _has_shared_physical_ancestor(
        {u.bubble_id for u in source},
        {u.bubble_id for u in dest},
        source_digests,
        dest_digests,
    )
    remap_pairs: list[tuple[int, int]] = []
    if chosen:
        for d_idx, s_idx in enumerate(chosen[1]):
            if source[s_idx].bubble_id != dest[d_idx].bubble_id:
                remap_pairs.append((s_idx, d_idx))
    trace.remap_candidates = len(remap_pairs)
    for s_idx, d_idx in remap_pairs:
        if both_v18:
            if ancestor_ok:
                trace.remap_physical_ancestor_ok += 1
            else:
                trace.remap_failed += 1
            continue
        leftover = dest_digests.get(source[s_idx].bubble_id) if dest_digests else None
        src_d = _source_body_digest(source[s_idx], source_digests)
        if leftover is not None and src_d is not None and leftover == src_d:
            trace.remap_physical_ancestor_ok += 1
        else:
            trace.remap_failed += 1
            trace.blockers.append(
                f"remap leftover missing {source[s_idx].bubble_id[:12]}"
            )
    if both_v18 and remap_pairs and not ancestor_ok:
        trace.blockers.append("v18↔v18 shared physical ancestor missing")

    seen: list[str] = []
    for item in trace.blockers:
        if item not in seen:
            seen.append(item)
    trace.blockers = seen

    if trace.common_id_rejected:
        if trace.rejected_duplicate_id:
            reason = "duplicate common bubbleId"
        elif trace.rejected_header_mismatch:
            reason = "same-ID header mismatch"
        elif trace.rejected_blobs_mismatch:
            reason = "same-ID blobs mismatch"
        elif trace.rejected_context_not_subset:
            reason = "same-ID context not subset"
        else:
            reason = "same-ID other body mismatch"
    elif not monotone:
        reason = "anchors cross"
    elif not trace.tip_found:
        reason = "source tip not found"
    elif trace.embedding_status == "ambiguous":
        reason = "historical embedding ambiguous"
    elif trace.embedding_status == "missing":
        reason = "historical embedding missing"
    elif trace.retirement_still_active:
        reason = "retired unit still active"
    elif trace.retirement_missing_body:
        reason = "retired body missing"
    elif trace.retirement_changed_body:
        reason = "retired body changed"
    elif trace.remap_failed:
        reason = (
            "v18↔v18 shared physical ancestor missing"
            if both_v18
            else "remap leftover missing"
        )
    elif len(valid_mappings) > 1:
        reason = "ambiguous source-tip placement"
    elif len(valid_mappings) == 1:
        reason = ""
    else:
        reason = "compatibility proof failed"
    trace.final_reject_reason = reason
    return trace


def _unit_body_value(unit: CompatUnit) -> Optional[dict[str, Any]]:
    bubble = unit.payload.get("bubble")
    if not isinstance(bubble, dict) or bubble.get("state") != "present":
        return None
    value = bubble.get("value")
    return value if isinstance(value, dict) else None


def _unit_conversational_body_digest(unit: CompatUnit) -> Optional[str]:
    value = _unit_body_value(unit)
    if value is None:
        return None
    return syncstate._sha256_text(syncstate._canonical_json(value))


def _conversational_equal(left: CompatUnit, right: CompatUnit) -> bool:
    if left.payload.get("blobs") != right.payload.get("blobs"):
        return False
    left_header = _strip_ui_metadata(left.payload.get("header"))
    right_header = _strip_ui_metadata(right.payload.get("header"))
    if left_header != right_header:
        return False
    return (_unit_body_value(left) or {}) == (_unit_body_value(right) or {})


def _same_id_hard_conflict(
    left: CompatUnit,
    right: CompatUnit,
    *,
    left_v: Optional[int],
    right_v: Optional[int],
) -> bool:
    if left.fingerprint == right.fingerprint:
        return False
    if left_v == _COMPAT_FROM and right_v == _COMPAT_TO:
        if is_migration_context_pruned(left, right):
            return False
    if right_v == _COMPAT_FROM and left_v == _COMPAT_TO:
        if is_migration_context_pruned(right, left):
            return False
    if _conversational_equal(left, right):
        return False
    return True


def _unit_tool_call_id(unit: CompatUnit) -> Optional[str]:
    value = _unit_body_value(unit) or {}
    tid = value.get("toolCallId")
    if isinstance(tid, str) and tid:
        return tid
    tool = value.get("toolFormerData")
    if isinstance(tool, dict):
        nested = tool.get("toolCallId")
        if isinstance(nested, str) and nested:
            return nested
    return None


def _unit_tool_io(unit: CompatUnit) -> Any:
    value = _unit_body_value(unit) or {}
    if not isinstance(value, dict):
        return None
    return {
        "toolFormerData": value.get("toolFormerData"),
        "toolResults": value.get("toolResults"),
        "text": value.get("text"),
    }


def _hard_semantic_fork(
    local: list[CompatUnit],
    remote: list[CompatUnit],
    local_conv: dict[str, str],
    remote_conv: dict[str, str],
    *,
    local_v: Optional[int],
    remote_v: Optional[int],
) -> bool:
    l_by_id: dict[str, list[CompatUnit]] = defaultdict(list)
    r_by_id: dict[str, list[CompatUnit]] = defaultdict(list)
    for unit in local:
        l_by_id[unit.bubble_id].append(unit)
    for unit in remote:
        r_by_id[unit.bubble_id].append(unit)

    anchors: list[tuple[int, int]] = []
    for bid in set(l_by_id) & set(r_by_id):
        left_units = l_by_id[bid]
        right_units = r_by_id[bid]
        if len(left_units) != 1 or len(right_units) != 1:
            return True
        left = left_units[0]
        right = right_units[0]
        if _same_id_hard_conflict(
            left, right, left_v=local_v, right_v=remote_v
        ):
            return True
        anchors.append((left.index, right.index))
    anchors.sort()
    for idx in range(len(anchors) - 1):
        if anchors[idx][1] >= anchors[idx + 1][1]:
            return True

    for unit in remote:
        leftover = local_conv.get(unit.bubble_id)
        src = _unit_conversational_body_digest(unit)
        if leftover is None or src is None or leftover == src:
            continue
        if unit.bubble_id in l_by_id:
            continue
        return True
    for unit in local:
        leftover = remote_conv.get(unit.bubble_id)
        src = _unit_conversational_body_digest(unit)
        if leftover is None or src is None or leftover == src:
            continue
        if unit.bubble_id in r_by_id:
            continue
        return True

    tools: dict[str, Any] = {}
    for unit in (*local, *remote):
        tid = _unit_tool_call_id(unit)
        if tid is None:
            continue
        payload = _unit_tool_io(unit)
        prev = tools.get(tid)
        if prev is None:
            tools[tid] = payload
        elif prev != payload:
            return True

    local_fp_count = Counter(unit.fingerprint for unit in local)
    remote_fp_count = Counter(unit.fingerprint for unit in remote)
    unique_local = {
        unit.fingerprint: unit.index
        for unit in local
        if local_fp_count[unit.fingerprint] == 1
    }
    unique_remote = {
        unit.fingerprint: unit.index
        for unit in remote
        if remote_fp_count[unit.fingerprint] == 1
    }
    pairs = [
        (unique_local[fp], unique_remote[fp])
        for fp in unique_local.keys() & unique_remote.keys()
    ]
    pairs.sort()
    for idx in range(len(pairs) - 1):
        if pairs[idx][1] >= pairs[idx + 1][1]:
            return True
    if _in_place_historical_edit(local, remote, local_conv):
        return True
    if _in_place_historical_edit(remote, local, remote_conv):
        return True
    return False


def _explained_by_leftover(unit: CompatUnit, other_conv: dict[str, str]) -> bool:
    digest = other_conv.get(unit.bubble_id)
    if digest is None:
        return False
    own = _unit_conversational_body_digest(unit)
    return own is not None and own == digest


def _novel_counts(
    local: list[CompatUnit],
    remote: list[CompatUnit],
    local_conv: dict[str, str],
    remote_conv: dict[str, str],
) -> tuple[int, int]:
    common = {unit.bubble_id for unit in local} & {unit.bubble_id for unit in remote}
    local_remaining = [unit for unit in local if unit.bubble_id not in common]
    remote_remaining = [unit for unit in remote if unit.bubble_id not in common]
    local_fps = Counter(unit.fingerprint for unit in local_remaining)
    remote_fps = Counter(unit.fingerprint for unit in remote_remaining)
    for fp in set(local_fps) | set(remote_fps):
        paired = min(local_fps[fp], remote_fps[fp])
        local_fps[fp] -= paired
        remote_fps[fp] -= paired
    for unit in remote_remaining:
        if remote_fps[unit.fingerprint] <= 0:
            continue
        if _explained_by_leftover(unit, local_conv):
            remote_fps[unit.fingerprint] -= 1
    for unit in local_remaining:
        if local_fps[unit.fingerprint] <= 0:
            continue
        if _explained_by_leftover(unit, remote_conv):
            local_fps[unit.fingerprint] -= 1
    return sum(local_fps.values()), sum(remote_fps.values())


def _in_place_historical_edit(
    dest: list[CompatUnit],
    source: list[CompatUnit],
    dest_conv: dict[str, str],
) -> bool:
    """True when dest replaced an ancestral turn instead of retiring it."""
    common = {unit.bubble_id for unit in dest} & {unit.bubble_id for unit in source}
    source_fps = {unit.fingerprint for unit in source}
    dest_represented_idx = {
        unit.index
        for unit in dest
        if unit.bubble_id in common or unit.fingerprint in source_fps
    }
    dest_novel = [
        unit
        for unit in dest
        if unit.bubble_id not in common and unit.fingerprint not in source_fps
    ]
    leftover_source = [
        unit
        for unit in source
        if unit.bubble_id not in common
        and unit.fingerprint not in {other.fingerprint for other in dest}
        and _explained_by_leftover(unit, dest_conv)
    ]
    if not dest_novel or not leftover_source or not dest_represented_idx:
        return False
    last_source = len(source) - 1
    min_novel = min(unit.index for unit in dest_novel)
    if all(
        src.index < min_novel or src.index == last_source for src in leftover_source
    ):
        return False
    later_represented = max(dest_represented_idx)
    for src in leftover_source:
        if src.index == last_source:
            continue
        for novel in dest_novel:
            if novel.index <= src.index and later_represented > novel.index:
                return True
    return False


def _classify_count_and_veto(
    local: list[CompatUnit],
    remote: list[CompatUnit],
    local_conv: dict[str, str],
    remote_conv: dict[str, str],
    *,
    local_v: Optional[int],
    remote_v: Optional[int],
) -> Optional[syncstate.SyncRelation]:
    """Count gives the candidate; semantic contradictions and dual-novel veto it."""
    if not local or not remote:
        return None
    if _hard_semantic_fork(
        local,
        remote,
        local_conv,
        remote_conv,
        local_v=local_v,
        remote_v=remote_v,
    ):
        return None
    local_novel, remote_novel = _novel_counts(local, remote, local_conv, remote_conv)
    if local_novel and remote_novel:
        return None
    if (local_novel or remote_novel) and not _active_ancestry(local, remote):
        return None
    if local_novel:
        return syncstate.SyncRelation.LOCAL_AHEAD
    if remote_novel:
        return syncstate.SyncRelation.BEHIND
    return syncstate.SyncRelation.UP_TO_DATE


def _active_ancestry(local: list[CompatUnit], remote: list[CompatUnit]) -> bool:
    """True when the active lists still share an ID or a logical unit."""
    if {unit.bubble_id for unit in local} & {unit.bubble_id for unit in remote}:
        return True
    return bool(
        {unit.fingerprint for unit in local} & {unit.fingerprint for unit in remote}
    )


def classify_compat_lineage(
    local: list[CompatUnit],
    remote: list[CompatUnit],
    *,
    local_digests: Optional[dict[str, str]] = None,
    snapshot_digests: Optional[dict[str, str]] = None,
    snapshot_bodies_local: bool = False,
    local_bodies_snapshot: bool = False,
    local_schema_v: Optional[int] = None,
    snapshot_schema_v: Optional[int] = None,
    local_conv_digests: Optional[dict[str, str]] = None,
    snapshot_conv_digests: Optional[dict[str, str]] = None,
) -> Optional[syncstate.SyncRelation]:
    """Compatibility direction, or None to keep DIVERGED (fork / keep both).

    ``snapshot_bodies_local`` / ``local_bodies_snapshot`` and the strict
    digest maps are kept for call-site compatibility. Classification
    uses conversational leftover maps when provided.
    """
    del snapshot_bodies_local, local_bodies_snapshot
    if local_conv_digests is None:
        local_conv_digests = local_digests or {
            unit.bubble_id: digest
            for unit in local
            if (digest := _unit_conversational_body_digest(unit)) is not None
        }
    if snapshot_conv_digests is None:
        snapshot_conv_digests = snapshot_digests or {
            unit.bubble_id: digest
            for unit in remote
            if (digest := _unit_conversational_body_digest(unit)) is not None
        }
    return _classify_count_and_veto(
        local,
        remote,
        local_conv_digests,
        snapshot_conv_digests,
        local_v=local_schema_v,
        remote_v=snapshot_schema_v,
    )


def _common_bubble_diffs(
    local: list[CompatUnit], remote: list[CompatUnit]
) -> tuple[int, int, int, list[dict]]:
    local_by_id = {unit.bubble_id: unit for unit in local}
    remote_by_id = {unit.bubble_id: unit for unit in remote}
    common = sorted(set(local_by_id) & set(remote_by_id))
    different_canonical = 0
    different_logical = 0
    samples: list[dict] = []
    for bid in common:
        left = local_by_id[bid]
        right = remote_by_id[bid]
        if left.canonical_hash != right.canonical_hash:
            different_canonical += 1
        if left.fingerprint != right.fingerprint:
            different_logical += 1
            if len(samples) < _FIRST_CHANGED:
                samples.append({
                    "bubbleId": bid,
                    "local_index": left.index,
                    "snapshot_index": right.index,
                    "logical_changed": True,
                    "canonical_changed": left.canonical_hash != right.canonical_hash,
                })
        elif left.canonical_hash != right.canonical_hash and len(samples) < _FIRST_CHANGED:
            samples.append({
                "bubbleId": bid,
                "local_index": left.index,
                "snapshot_index": right.index,
                "logical_changed": False,
                "canonical_changed": True,
            })
    return len(common), different_canonical, different_logical, samples


def _fill_dest_mapping(
    report: LineageReport,
    mapping: Optional[list[int]],
    ambiguous: list[str],
    source: list[CompatUnit],
    dest: list[CompatUnit],
    dest_side: str,
    active_source: Optional[list[CompatUnit]] = None,
) -> None:
    report.ambiguous_duplicate_fingerprints = ambiguous
    if mapping is None:
        dest_fps = [unit.fingerprint for unit in dest]
        dest_counts = Counter(dest_fps)
        matched = 0
        missing = 0
        first_changed: list[dict] = []
        remaining = dest_counts.copy()
        for unit in source:
            if remaining.get(unit.fingerprint, 0) > 0:
                remaining[unit.fingerprint] -= 1
                matched += 1
            else:
                missing += 1
                if len(first_changed) < _FIRST_CHANGED:
                    first_changed.append({
                        "side": "source",
                        "index": unit.index,
                        "bubbleId": unit.bubble_id,
                        "reason": f"no unused {dest_side} logical equivalent",
                    })
        report.exact_logical_matches = matched
        report.source_missing_units = missing
        report.destination_extra_units = (
            sum(remaining.values()) if dest_counts else len(dest)
        )
        report.first_changed_logical_units = first_changed
        report.mapping_unique = False
        report.logical_source_tip_preserved = False
        report.source_tip_preserved = False
        return

    t = len(mapping) - 1
    mapped_source_indices = set(mapping)
    retired_source_indices = set(range(len(source))) - mapped_source_indices
    n_prune = 0
    for d_idx, s_idx in enumerate(mapping):
        if (
            0 <= s_idx < len(source)
            and d_idx < len(dest)
            and source[s_idx].bubble_id == dest[d_idx].bubble_id
            and source[s_idx].fingerprint != dest[d_idx].fingerprint
        ):
            n_prune += 1

    report.exact_logical_matches = len(mapping)
    report.source_missing_units = len(source) - len(mapped_source_indices)
    report.destination_extra_units = len(dest) - 1 - t
    report.extras_before = 0
    report.extras_inside = 0
    report.extras_after = len(dest) - 1 - t
    report.extras_before_anchor = 0
    report.extras_inside_anchor = 0
    report.extras_after_anchor = len(dest) - 1 - t
    report.destination_extras_after_source_tip = len(dest) - 1 - t
    report.mapping_unique = not ambiguous
    report.logical_source_tip_preserved = True
    report.source_tip_preserved = True
    report.active_represented = len(mapped_source_indices)
    report.retired_preserved = len(source) - len(mapped_source_indices)
    report.migration_equivalent_active_rewrites = n_prune
    report.semantically_changed_active = 0
    report.body_rewritten = 0
    report.unaccounted_missing = 0

    changed: list[dict] = []
    for idx in sorted(retired_source_indices)[:_FIRST_CHANGED]:
        unit = source[idx]
        changed.append({
            "side": "source",
            "index": unit.index,
            "bubbleId": unit.bubble_id,
            "reason": "retired from active list, physically preserved",
        })
    report.first_changed_logical_units = changed


def _changed_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if left == right:
        return []
    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            key_str = str(key)
            if "." in key_str or "[" in key_str or " " in key_str:
                child = f"{prefix}[{json.dumps(key_str)}]" if prefix else f"[{json.dumps(key_str)}]"
            else:
                child = f"{prefix}.{key_str}" if prefix else key_str
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_changed_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        paths = []
        limit = max(len(left), len(right))
        for index in range(limit):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            if index >= len(left) or index >= len(right):
                paths.append(child)
            else:
                paths.extend(_changed_paths(left[index], right[index], child))
        return paths
    return [prefix or "$"]


def _get_path_value(obj: Any, path: str) -> Any:
    if not path or obj is None:
        return obj
    tokens: list[Any] = []
    i = 0
    n = len(path)
    curr_part = ""
    while i < n:
        c = path[i]
        if c == ".":
            if curr_part:
                tokens.append(curr_part)
                curr_part = ""
            i += 1
        elif c == "[":
            if curr_part:
                tokens.append(curr_part)
                curr_part = ""
            close_idx = path.find("]", i)
            if close_idx == -1:
                tokens.append(path[i:])
                break
            inside = path[i + 1:close_idx]
            if (inside.startswith('"') and inside.endswith('"')) or (
                inside.startswith("'") and inside.endswith("'")
            ):
                try:
                    tokens.append(json.loads(inside))
                except Exception:
                    tokens.append(inside[1:-1])
            else:
                try:
                    tokens.append(int(inside))
                except ValueError:
                    tokens.append(inside)
            i = close_idx + 1
        else:
            curr_part += c
            i += 1
    if curr_part:
        tokens.append(curr_part)

    curr = obj
    for tok in tokens:
        if curr is None:
            return None
        if isinstance(tok, int):
            if not isinstance(curr, (list, tuple)) or tok < 0 or tok >= len(curr):
                return None
            curr = curr[tok]
        else:
            if not isinstance(curr, dict) or tok not in curr:
                return None
            curr = curr[tok]
    return curr


def _format_diff_value(val: Any, max_len: int = 160) -> Any:
    if val is None:
        return "<missing>"
    if isinstance(val, (dict, list)):
        try:
            s = json.dumps(val, sort_keys=True)
        except Exception:
            s = str(val)
        return s[:max_len] + ("..." if len(s) > max_len else "")
    if isinstance(val, str):
        return val[:max_len] + ("..." if len(val) > max_len else "")
    return val


def _normalize_bubble_for_compat(bubble: Optional[dict]) -> dict:
    if not isinstance(bubble, dict):
        return {}
    return migration_normalize_body(bubble)


def _load_local_raw_bubble(
    session: Optional["syncstate.SyncReadSession"],
    composer_id: str,
    bid: str,
    local_by_id: dict[str, CompatUnit],
    local_data: Any = None,
) -> Optional[dict]:
    unit = local_by_id.get(bid)
    if unit is not None and unit.has_body:
        val = unit.payload.get("bubble", {}).get("value")
        if isinstance(val, dict):
            return val
    if isinstance(local_data, dict):
        entries = local_data.get("bubbleEntries")
        if isinstance(entries, dict) and bid in entries and isinstance(entries[bid], dict):
            return entries[bid]
        cmap = local_data.get("conversationMap")
        if isinstance(cmap, dict) and bid in cmap and isinstance(cmap[bid], dict):
            return cmap[bid]
        cdata = local_data.get("composerData")
        if isinstance(cdata, dict):
            cmap2 = cdata.get("conversationMap")
            if isinstance(cmap2, dict) and bid in cmap2 and isinstance(cmap2[bid], dict):
                return cmap2[bid]
    if session is not None and session._cdb is not None:
        try:
            conn = session._cdb._reader_conn()
            row = conn.execute(
                "SELECT value FROM cursorDiskKV WHERE key = ?",
                (f"bubbleId:{composer_id}:{bid}",),
            ).fetchone()
            if row is not None and row[0] is not None:
                parsed = _parse_bubble_object(row[0])
                if isinstance(parsed, dict):
                    return parsed
        except Exception:
            pass
    if session is not None and composer_id:
        try:
            data = session.composer_data(composer_id)
            if isinstance(data, dict):
                cmap = data.get("conversationMap")
                if isinstance(cmap, dict) and isinstance(cmap.get(bid), dict):
                    return cmap[bid]
        except Exception:
            pass
    return None


def _load_snapshot_raw_bubble(
    snapshot: Optional[dict],
    bid: str,
    remote_by_id: dict[str, CompatUnit],
) -> Optional[dict]:
    if snapshot is not None:
        val = syncstate._bubble_from_snapshot(snapshot, bid)
        if isinstance(val, dict):
            return val
    unit = remote_by_id.get(bid)
    if unit is not None and unit.has_body:
        val = unit.payload.get("bubble", {}).get("value")
        if isinstance(val, dict):
            return val
    return None


def _structural_category(payload: dict[str, Any]) -> str:
    header = payload.get("header") if isinstance(payload.get("header"), dict) else {}
    bubble = payload.get("bubble") if isinstance(payload.get("bubble"), dict) else {}
    body = bubble.get("value") if bubble.get("state") == "present" else {}
    if not isinstance(body, dict):
        body = {}
    flags = {**header, **body}
    if flags.get("isSimulatedMsg") is True:
        return "simulated"
    kind = flags.get("backgroundTaskCompletionKind")
    if kind not in (None, "", 0, False):
        return "background_task"
    if body.get("toolResults") or header.get("toolResults"):
        return "tool_result"
    if body.get("toolFormerData") or body.get("toolCallId") or header.get("toolCallId"):
        return "tool_call"
    typ = body.get("type", header.get("type"))
    if typ == 1:
        return "user"
    if typ == 2:
        return "assistant_text"
    if flags.get("capabilityType") or flags.get("isAgentic"):
        return "internal_synthetic"
    return "other"


def _greedy_exact_remainder(
    source: list[CompatUnit], dest: list[CompatUnit]
) -> tuple[list[CompatUnit], list[CompatUnit]]:
    dest_by_fp: dict[str, deque[int]] = defaultdict(deque)
    for index, unit in enumerate(dest):
        dest_by_fp[unit.fingerprint].append(index)
    used = [False] * len(dest)
    unmatched: list[CompatUnit] = []
    for unit in source:
        queue = dest_by_fp.get(unit.fingerprint)
        if queue:
            used[queue.popleft()] = True
        else:
            unmatched.append(unit)
    unused = [dest[index] for index, taken in enumerate(used) if not taken]
    return unmatched, unused


def _payload_body_digest(payload: dict[str, Any]) -> Optional[str]:
    bubble = payload.get("bubble")
    if not isinstance(bubble, dict) or bubble.get("state") != "present":
        return None
    value = bubble.get("value")
    if not isinstance(value, dict):
        return None
    return syncstate._sha256_text(syncstate._canonical_json(value))


def _local_digest_index(
    local: list[CompatUnit],
    *,
    session: Optional["syncstate.SyncReadSession"] = None,
    composer_id: str = "",
    local_data: Any = None,
    conversational: bool = False,
) -> dict[str, str]:
    if session is not None and composer_id:
        return load_local_physical_digests(
            session, composer_id, wanted=None, conversational=conversational
        )
    index: dict[str, str] = {}
    if isinstance(local_data, dict):
        entries = local_data.get("bubbleEntries")
        if isinstance(entries, dict):
            for bid, bval in entries.items():
                if isinstance(bval, dict):
                    index[str(bid)] = _compat_body_digest(
                        bval, conversational=conversational
                    )
        cmap = local_data.get("conversationMap")
        if isinstance(cmap, dict):
            for bid, bval in cmap.items():
                if isinstance(bval, dict) and str(bid) not in index:
                    index[str(bid)] = _compat_body_digest(
                        bval, conversational=conversational
                    )
        cdata = local_data.get("composerData")
        if isinstance(cdata, dict):
            cmap2 = cdata.get("conversationMap")
            if isinstance(cmap2, dict):
                for bid, bval in cmap2.items():
                    if isinstance(bval, dict) and str(bid) not in index:
                        index[str(bid)] = _compat_body_digest(
                            bval, conversational=conversational
                        )
    for unit in local:
        if unit.bubble_id not in index:
            digest = (
                _unit_conversational_body_digest(unit)
                if conversational
                else _payload_body_digest(unit.payload)
            )
            if digest is not None:
                index[unit.bubble_id] = digest
    return index


def _snapshot_body_digests(
    remote: list[CompatUnit],
    snapshot: Optional[dict],
    *,
    conversational: bool = False,
) -> dict[str, str]:
    digests: dict[str, str] = {}
    if snapshot is not None:
        for bid in snapshot_stored_body_ids(snapshot):
            bubble = syncstate._bubble_from_snapshot(snapshot, bid)
            if isinstance(bubble, dict):
                digests[bid] = _compat_body_digest(bubble, conversational=conversational)
    for unit in remote:
        if unit.bubble_id in digests:
            continue
        digest = _unit_conversational_body_digest(unit) if conversational else _payload_body_digest(unit.payload)
        if digest is not None:
            digests[unit.bubble_id] = digest
    return digests


def _fill_forensic(
    report: LineageReport,
    source: list[CompatUnit],
    dest: list[CompatUnit],
    local: list[CompatUnit],
    remote: list[CompatUnit],
    *,
    snapshot: Optional[dict] = None,
    session: Optional["syncstate.SyncReadSession"] = None,
    composer_id: str = "",
    local_data: Any = None,
) -> None:
    unmatched, unused = _greedy_exact_remainder(source, dest)
    dest_by_diag: dict[str, deque[int]] = defaultdict(deque)
    for index, unit in enumerate(unused):
        dest_by_diag[unit.diagnostic_fingerprint].append(index)
    used_unused = [False] * len(unused)
    explained_ids: set[str] = set()
    zero_ids: set[str] = set()
    categories: Counter[str] = Counter()
    header_paths: Counter[str] = Counter()
    for unit in unmatched:
        categories[_structural_category(unit.payload)] += 1
        queue = dest_by_diag.get(unit.diagnostic_fingerprint)
        live = [i for i in (queue or []) if not used_unused[i]]
        if not live:
            report.body_candidates_zero += 1
            zero_ids.add(unit.bubble_id)
            continue
        if len(live) > 1:
            report.body_candidates_ambiguous += 1
            continue
        report.body_candidates_unique += 1
        dest_unit = unused[live[0]]
        used_unused[live[0]] = True
        explained_ids.add(unit.bubble_id)
        paths = _changed_paths(unit.payload, dest_unit.payload)
        header_only = all(
            path == "header" or path.startswith("header.") for path in paths
        )
        if header_only:
            report.unique_replacements_header_only += 1
            for path in paths:
                header_paths[path] += 1
        else:
            report.unique_replacements_body_diff += 1
    report.structural_categories = dict(sorted(categories.items()))
    report.header_path_histogram = dict(sorted(header_paths.items()))

    local_by_id = {unit.bubble_id: unit for unit in local}
    remote_by_id = {unit.bubble_id: unit for unit in remote}
    is_source_remote = (report.mapping_direction != "local_in_snapshot")
    source_units = remote if is_source_remote else local
    dest_units = local if is_source_remote else remote

    source_digests = (
        _snapshot_body_digests(remote, snapshot)
        if is_source_remote
        else _local_digest_index(local, session=session, composer_id=composer_id, local_data=local_data)
    )
    dest_index = (
        _local_digest_index(local, session=session, composer_id=composer_id, local_data=local_data)
        if is_source_remote
        else _snapshot_body_digests(remote, snapshot)
    )
    digest_to_dest: dict[str, list[str]] = defaultdict(list)
    for bid, digest in dest_index.items():
        digest_to_dest[digest].append(bid)

    active_ids = [unit.bubble_id for unit in source_units if unit.has_body]
    report.physical_snapshot_active_body_ids = len(active_ids)
    dest_active_bids = {unit.bubble_id for unit in dest_units}

    unmatched_indices = {u.index for u in unmatched}
    active_source = [u for u in source_units if u.index not in unmatched_indices]
    classifier_locked = report.lineage_relation is not None
    if not classifier_locked:
        report.active_represented = len(active_source)
        report.active_exact_preserved = len(active_source)

        retired_count = 0
        rewritten_count = 0
        unaccounted_count = 0
        for u in unmatched:
            bid = u.bubble_id
            src_d = source_digests.get(bid)
            dst_d = dest_index.get(bid)
            if dst_d is not None:
                if src_d is not None and dst_d == src_d:
                    retired_count += 1
                else:
                    rewritten_count += 1
            else:
                if src_d is not None and digest_to_dest.get(src_d):
                    pass
                else:
                    unaccounted_count += 1
        report.retired_preserved = retired_count
        report.retired_physically_preserved = retired_count
        report.body_rewritten = rewritten_count
        report.unaccounted_missing = unaccounted_count
        report.semantically_changed_active = rewritten_count
    else:
        report.active_exact_preserved = report.active_represented - report.migration_equivalent_active_rewrites
        report.retired_physically_preserved = report.retired_preserved

    physically_accounted: set[str] = set()
    changed_body_paths: Counter[str] = Counter()
    samples_by_key: dict[str, dict] = {}

    for bid in active_ids:
        src_digest = source_digests.get(bid)
        dst_digest = dest_index.get(bid)
        if dst_digest is not None:
            report.physical_same_id_present += 1
            if src_digest is not None and dst_digest == src_digest:
                report.physical_same_id_unchanged += 1
                physically_accounted.add(bid)
            else:
                report.physical_same_id_changed += 1
                if is_source_remote:
                    src_raw = _load_snapshot_raw_bubble(snapshot, bid, remote_by_id)
                    dst_raw = _load_local_raw_bubble(session, composer_id, bid, local_by_id, local_data=local_data)
                else:
                    src_raw = _load_local_raw_bubble(session, composer_id, bid, local_by_id, local_data=local_data)
                    dst_raw = _load_snapshot_raw_bubble(snapshot, bid, remote_by_id)
                norm_src = _normalize_bubble_for_compat(src_raw)
                norm_dst = _normalize_bubble_for_compat(dst_raw)
                paths = _changed_paths(norm_src, norm_dst)
                for p in paths:
                    changed_body_paths[p] += 1
                    sample_key = re.sub(r'\[\d+\]', '[]', p)
                    if sample_key not in samples_by_key:
                        s_val = _get_path_value(norm_src, p)
                        d_val = _get_path_value(norm_dst, p)
                        samples_by_key[sample_key] = {
                            "path_class": sample_key,
                            "path": p,
                            "bubbleId": bid,
                            "snapshot" if is_source_remote else "local": _format_diff_value(s_val),
                            "local" if is_source_remote else "snapshot": _format_diff_value(d_val),
                        }
            continue
        others = [
            other for other in digest_to_dest.get(src_digest or "", [])
            if other != bid
        ]
        if src_digest is not None and others:
            report.physical_same_body_other_id += 1
            physically_accounted.add(bid)
        else:
            report.physical_unaccounted += 1

    report.changed_physical_body_paths = dict(
        sorted(changed_body_paths.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    report.changed_physical_body_samples = [
        samples_by_key[k] for k in sorted(samples_by_key)
    ]

    if not classifier_locked:
        if active_source:
            active_source_fps = [u.fingerprint for u in active_source]
            dest_fps = [u.fingerprint for u in dest_units]
            mapping, ambiguous = unique_subsequence_map(active_source_fps, dest_fps)
            if mapping:
                lo = mapping[0]
                anchor_idx = mapping[-1]
                mapped_dest = set(mapping)
                report.extras_before_anchor = sum(1 for i in range(lo) if i not in mapped_dest)
                report.extras_inside_anchor = sum(1 for i in range(lo + 1, anchor_idx) if i not in mapped_dest)
                report.extras_after_anchor = sum(1 for i in range(anchor_idx + 1, len(dest_units)) if i not in mapped_dest)
                report.destination_extras_after_source_tip = report.extras_after_anchor
                report.logical_source_tip_preserved = (
                    source_units[-1].fingerprint == active_source[-1].fingerprint
                )
                report.source_tip_preserved = report.logical_source_tip_preserved
            else:
                report.logical_source_tip_preserved = False
                report.source_tip_preserved = False
        else:
            report.logical_source_tip_preserved = False
            report.source_tip_preserved = False

    report.forensic_unexplained_units = sum(
        1
        for unit in unmatched
        if unit.bubble_id in zero_ids
        and unit.bubble_id not in physically_accounted
    )


def diagnose_lineage(
    local: list[CompatUnit],
    remote: list[CompatUnit],
    *,
    composer_id: str = "",
    local_data: Any = None,
    remote_data: Any = None,
    snapshot: Optional[dict] = None,
    session: Optional["syncstate.SyncReadSession"] = None,
) -> LineageReport:
    wanted: set[str] = set(snapshot_stored_body_ids(snapshot)) if snapshot else set()
    wanted.update(unit.bubble_id for unit in remote if unit.has_body)
    wanted.update(unit.bubble_id for unit in local if unit.has_body)
    local_digests = (
        load_local_physical_digests(session, composer_id, wanted)
        if session and composer_id
        else _local_digest_index(local, session=session, composer_id=composer_id, local_data=local_data)
    )
    snap_digests = _snapshot_body_digests(remote, snapshot)
    local_conv = (
        load_local_physical_digests(
            session, composer_id, wanted, conversational=True
        )
        if session and composer_id
        else _local_digest_index(
            local,
            session=session,
            composer_id=composer_id,
            local_data=local_data,
            conversational=True,
        )
    )
    snap_conv = _snapshot_body_digests(remote, snapshot, conversational=True)

    snap_phys = (
        snapshot_history_physically_intact_locally(
            session, composer_id, snapshot, local_digests=local_digests
        )
        if session and snapshot and composer_id
        else False
    )
    loc_phys = (
        local_history_physically_intact_in_snapshot(
            session, composer_id, snapshot, local, local_digests=local_digests
        )
        if session and snapshot and composer_id
        else False
    )
    candidate = is_compatibility_lineage_candidate(local_data, remote_data)
    report = LineageReport(
        composer_id=composer_id,
        local_schema_v=schema_version(local_data),
        snapshot_schema_v=schema_version(remote_data),
        compatibility_candidate=candidate,
        snapshot_active_units=len(remote),
        local_active_units=len(local),
        snapshot_bodies_preserved_locally=snap_phys,
        local_bodies_preserved_in_snapshot=loc_phys,
    )

    local_v = schema_version(local_data)
    remote_v = schema_version(remote_data)
    rel_ahead, map_rl, amb_rl, act_rl, ret_rl = _evaluate_lineage_direction(
        remote,
        local,
        local_digests,
        snap_digests,
        source_v=remote_v,
        dest_v=local_v,
    )
    rel_behind, map_lr, amb_lr, act_lr, ret_lr = _evaluate_lineage_direction(
        local,
        remote,
        snap_digests,
        local_digests,
        source_v=local_v,
        dest_v=remote_v,
    )
    ahead_trace = _collect_proof_trace(
        remote,
        local,
        local_digests,
        snap_digests,
        source_v=remote_v,
        dest_v=local_v,
    )
    behind_trace = _collect_proof_trace(
        local,
        remote,
        snap_digests,
        local_digests,
        source_v=local_v,
        dest_v=remote_v,
    )

    relation = None
    if candidate:
        relation = classify_compat_lineage(
            local,
            remote,
            local_digests=local_digests,
            snapshot_digests=snap_digests,
            local_schema_v=local_v,
            snapshot_schema_v=remote_v,
            local_conv_digests=local_conv,
            snapshot_conv_digests=snap_conv,
        )
        report.lineage_relation = relation.value if relation is not None else None

    if relation in (
        syncstate.SyncRelation.UP_TO_DATE,
        syncstate.SyncRelation.LOCAL_AHEAD,
    ) and map_rl is not None:
        report.mapping_direction = "snapshot_in_local"
        _fill_dest_mapping(report, map_rl, amb_rl, remote, local, "local", active_source=act_rl)
    elif relation == syncstate.SyncRelation.BEHIND and map_lr is not None:
        report.mapping_direction = "local_in_snapshot"
        _fill_dest_mapping(report, map_lr, amb_lr, local, remote, "snapshot", active_source=act_lr)
    elif map_lr is not None and map_rl is None and not amb_rl:
        report.mapping_direction = "local_in_snapshot"
        _fill_dest_mapping(report, map_lr, amb_lr, local, remote, "snapshot", active_source=act_lr)
    else:
        report.mapping_direction = "snapshot_in_local"
        _fill_dest_mapping(report, map_rl, amb_rl, remote, local, "local", active_source=act_rl)

    common, diff_can, diff_log, samples = _common_bubble_diffs(local, remote)
    report.common_bubble_ids = common
    report.common_bubble_ids_different_canonical = diff_can
    report.common_bubble_ids_different_logical = diff_log
    report.common_bubble_semantic_diffs = samples
    if report.mapping_direction == "local_in_snapshot":
        source, dest = local, remote
        report.proof_trace = behind_trace.as_dict()
    else:
        source, dest = remote, local
        report.proof_trace = ahead_trace.as_dict()
    if relation is None:
        report.migration_equivalent_active_rewrites = int(
            report.proof_trace.get("common_id_context_pruned") or 0
        )
    _fill_forensic(
        report,
        source,
        dest,
        local,
        remote,
        snapshot=snapshot,
        session=session,
        composer_id=composer_id,
        local_data=local_data,
    )
    return report


def format_lineage_report(report: LineageReport) -> str:
    lines = [
        f"Composer: {report.composer_id or '(unknown)'}",
        f"Local _v: {report.local_schema_v}",
        f"Snapshot _v: {report.snapshot_schema_v}",
        (
            f"Compatibility candidate: "
            f"{'yes' if report.compatibility_candidate else 'no'}"
        ),
        "",
        f"snapshot active units: {report.snapshot_active_units}",
        f"local active units: {report.local_active_units}",
        f"exact logical matches: {report.exact_logical_matches}",
        f"source missing units: {report.source_missing_units}",
        f"destination extra units: {report.destination_extra_units}",
        f"ambiguous duplicate fingerprints: {len(report.ambiguous_duplicate_fingerprints)}",
        f"first changed logical units: {len(report.first_changed_logical_units)}",
        (
            f"extras before/inside/after the historical boundary: "
            f"{report.extras_before}/{report.extras_inside}/{report.extras_after}"
        ),
        f"common bubbleIds: {report.common_bubble_ids}",
        (
            f"common bubbleIds with different canonical hash: "
            f"{report.common_bubble_ids_different_canonical}"
        ),
        (
            f"common bubbleIds with different logical fingerprint: "
            f"{report.common_bubble_ids_different_logical}"
        ),
        f"mapping direction: {report.mapping_direction or 'none'}",
        (
            f"snapshot bodies preserved locally: "
            f"{'yes' if report.snapshot_bodies_preserved_locally else 'no'}"
        ),
        (
            f"local bodies preserved in snapshot: "
            f"{'yes' if report.local_bodies_preserved_in_snapshot else 'no'}"
        ),
        (
            "compatibility lineage: "
            + (
                "not-applicable"
                if not report.compatibility_candidate
                else (report.lineage_relation or "diverged")
            )
        ),
    ]
    if report.compatibility_candidate:
        lines.append(
            "sync action: "
            + ("same-CID" if report.lineage_relation else "preserve both")
        )
    if report.proof_trace:
        trace = report.proof_trace
        lines.extend([
            "",
            "Compatibility proof trace:",
            "  common-ID anchors:",
            f"    exact:                  {trace.get('common_id_exact', 0)}",
            f"    context-pruned:         {trace.get('common_id_context_pruned', 0)}",
            f"    rejected:               {trace.get('common_id_rejected', 0)}",
            "",
            "  rejected common IDs:",
            f"    header mismatch:        {trace.get('rejected_header_mismatch', 0)}",
            f"    blobs mismatch:         {trace.get('rejected_blobs_mismatch', 0)}",
            f"    context not subset:     {trace.get('rejected_context_not_subset', 0)}",
            f"    other body mismatch:    {trace.get('rejected_other_body_mismatch', 0)}",
            "",
            f"  anchors monotone:         {'yes' if trace.get('anchors_monotone') else 'no' if trace.get('anchors_monotone') is False else 'n/a'}",
            "",
            "  source tip:",
            f"    exact candidate:        {trace.get('tip_exact_candidates', 0)}",
            f"    prune candidate:        {trace.get('tip_prune_candidates', 0)}",
            f"    found:                  {'yes' if trace.get('tip_found') else 'no'}",
            "",
            "  historical embedding:",
            f"    status:                 {trace.get('embedding_status') or 'not-attempted'}",
            f"    failed segment:         {trace.get('failed_segment') or '-'}",
            "",
            "  retirement proof:",
            f"    candidates:             {trace.get('retirement_candidates', 0)}",
            f"    preserved:              {trace.get('retirement_preserved', 0)}",
            f"    missing body:           {trace.get('retirement_missing_body', 0)}",
            f"    changed body:           {trace.get('retirement_changed_body', 0)}",
            f"    still active:           {trace.get('retirement_still_active', 0)}",
            "",
            "  remap proof:",
            f"    candidates:             {trace.get('remap_candidates', 0)}",
            f"    physical ancestor ok:   {trace.get('remap_physical_ancestor_ok', 0)}",
            f"    failed:                 {trace.get('remap_failed', 0)}",
            "",
            f"  final reject reason:      {trace.get('final_reject_reason') or '-'}",
        ])
    lines.extend([
        "",
        f"source unmatched: {report.source_missing_units}",
        "body candidates:",
        f"  zero:       {report.body_candidates_zero}",
        f"  unique:     {report.body_candidates_unique}",
        f"  ambiguous:  {report.body_candidates_ambiguous}",
        "unique replacements:",
        f"  header-only differences: {report.unique_replacements_header_only}",
        f"  body differences:        {report.unique_replacements_body_diff}",
        "structural categories:",
        *(
            [f"  {name}: {count}" for name, count in report.structural_categories.items()]
            or ["  (none)"]
        ),
        "",
        f"snapshot active body IDs:             {report.physical_snapshot_active_body_ids}",
        f"same ID present locally:              {report.physical_same_id_present}",
        f"same-ID body semantically unchanged:  {report.physical_same_id_unchanged}",
        f"same-ID body changed:                 {report.physical_same_id_changed}",
        f"same semantic body under another ID:  {report.physical_same_body_other_id}",
        f"unaccounted:                          {report.physical_unaccounted}",
        f"forensic unexplained units:           {report.forensic_unexplained_units}",
        "",
        "History retirement accounting:",
        f"  source active units:                  {report.snapshot_active_units if report.mapping_direction != 'local_in_snapshot' else report.local_active_units}",
        f"  active represented (logical match):   {report.active_represented}",
        f"  migration-equivalent active rewrites: {report.migration_equivalent_active_rewrites}",
        f"  retired preserved (physical intact):  {report.retired_preserved}",
        f"  semantically changed/unaccounted:     {report.semantically_changed_active + report.unaccounted_missing}",
        f"  body rewritten (same ID modified):    {report.body_rewritten}",
        f"  unaccounted / missing:                {report.unaccounted_missing}",
        "",
        f"  logical source tip preserved:         {'yes' if report.logical_source_tip_preserved else 'no'}",
        f"  destination active units:             {report.local_active_units if report.mapping_direction != 'local_in_snapshot' else report.snapshot_active_units}",
        f"  destination extras before anchor:     {report.extras_before_anchor}",
        f"  destination extras inside history:    {report.extras_inside_anchor}",
        f"  destination extras after anchor:      {report.extras_after_anchor}",
    ])
    if report.changed_physical_body_paths:
        lines.append("")
        lines.append(f"changed physical bodies: {report.body_rewritten or report.physical_same_id_changed}")
        lines.append("")
        lines.append(f"{'path':<45} count")
        lines.append("-" * 55)
        for path, count in report.changed_physical_body_paths.items():
            lines.append(f"{path:<45} {count}")
    if report.changed_physical_body_samples:
        lines.append("")
        lines.append(f"Representative samples by path class ({len(report.changed_physical_body_samples)} classes):")
        for sample in report.changed_physical_body_samples:
            cls = sample.get("path_class", "")
            p = sample.get("path", "")
            bid = sample.get("bubbleId", "")
            s_val = sample.get("snapshot") if "snapshot" in sample else sample.get("source", "")
            d_val = sample.get("local") if "local" in sample else sample.get("dest", "")
            lines.append(f"  [{cls}]  (bubbleId: {bid[:16]}...)")
            lines.append(f"    path:     {p}")
            lines.append(f"    snapshot: {s_val}")
            lines.append(f"    local:    {d_val}")
    if report.header_path_histogram:
        lines.append("")
        lines.append("Header paths that change on unique replacements:")
        for path, count in report.header_path_histogram.items():
            lines.append(f"  {path}: {count}")
    if report.common_bubble_semantic_diffs:
        lines.append("")
        lines.append("Common bubbleId semantic diffs (first):")
        for item in report.common_bubble_semantic_diffs:
            lines.append(
                f"  {item['bubbleId'][:12]}...  "
                f"local[{item['local_index']}] snapshot[{item['snapshot_index']}]  "
                f"logical_changed={item['logical_changed']}  "
                f"canonical_changed={item['canonical_changed']}"
            )
    if report.first_changed_logical_units:
        lines.append("")
        lines.append("First changed logical units:")
        for item in report.first_changed_logical_units:
            lines.append(
                f"  {item['side']}[{item['index']}]  "
                f"{item.get('bubbleId', '')[:12]}...  {item['reason']}"
            )
    return "\n".join(lines)


def classify_after_canonical_diverged(
    session: "syncstate.SyncReadSession",
    rec: Any,
) -> Optional[syncstate.SyncRelation]:
    """Gated read-only lineage check. None keeps DIVERGED (fork). Never writes.

    Compatibility that is not recognized semantically stays DIVERGED.
    An internal exception or unreadable payload returns UNKNOWN so
    sync can abort without writes.
    """
    try:
        local_data = session.composer_data(rec.composer_id)
        if not isinstance(local_data, dict) or rec.path is None:
            return syncstate.SyncRelation.UNKNOWN
        try:
            snapshot = importer.read_snapshot_file(rec.path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError, gzip.BadGzipFile):
            return syncstate.SyncRelation.UNKNOWN
        remote_data = snapshot.get("composerData")
        if not is_compatibility_lineage_candidate(local_data, remote_data):
            return None
        syncstate._counts.compatibility_lineage_checks += 1
        try:
            local_units = local_compat_units(session, rec.composer_id)
            remote_units = snapshot_compat_units(snapshot)
        except syncstate.ClassifyError:
            return syncstate.SyncRelation.UNKNOWN

        wanted: set[str] = set(snapshot_stored_body_ids(snapshot))
        wanted.update(unit.bubble_id for unit in remote_units if unit.has_body)
        wanted.update(unit.bubble_id for unit in local_units if unit.has_body)
        local_digests = load_local_physical_digests(session, rec.composer_id, wanted)
        snapshot_digests = _snapshot_body_digests(remote_units, snapshot)
        local_conv = load_local_physical_digests(
            session, rec.composer_id, wanted, conversational=True
        )
        snapshot_conv = _snapshot_body_digests(
            remote_units, snapshot, conversational=True
        )

        return classify_compat_lineage(
            local_units,
            remote_units,
            local_digests=local_digests,
            snapshot_digests=snapshot_digests,
            local_schema_v=schema_version(local_data),
            snapshot_schema_v=schema_version(remote_data),
            local_conv_digests=local_conv,
            snapshot_conv_digests=snapshot_conv,
        )
    except syncstate.ClassifyError:
        return syncstate.SyncRelation.UNKNOWN
    except Exception:
        return syncstate.SyncRelation.UNKNOWN
