"""Incremental pull: classify via syncstate, stage candidates, batch import."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import db, dblock, importer, paths, syncstate
from .process import is_cursor_running


@dataclass
class PullResult:
    imported: int = 0
    failed: int = 0
    plan: Optional[syncstate.PullPlan] = None
    plans: list[syncstate.PullPlan] = field(default_factory=list)


def scoped_snapshot_index(
    project_path: str,
    source_host: Optional[str] = None,
) -> syncstate.SnapshotIndex:
    """Index only the snapshot directory for this origin."""
    found = importer.find_snapshot_dir_for_project(
        project_path, source_host=source_host
    )
    if found is not None:
        return syncstate.SnapshotIndex.build_for_project(found.name)
    project_id = paths.get_project_identifier(project_path, source_host=source_host)
    return syncstate.SnapshotIndex.build_for_project(project_id)


def snapshot_index_for_target(
    workspace: dict,
    selected: Optional[list[Path]],
) -> syncstate.SnapshotIndex:
    """Index the selected snapshot bucket, not the restore target path.

    ``pull -s`` can restore into a local path whose project ID differs from
    the bucket the user picked. Rediscovering from the target would miss
    those files.
    """
    if not selected:
        return scoped_snapshot_index(workspace["path"], workspace.get("host"))
    project_dirs = {Path(p).resolve().parent for p in selected}
    if len(project_dirs) != 1:
        return syncstate.SnapshotIndex()
    project_dir = next(iter(project_dirs))
    return syncstate.SnapshotIndex.build(
        snapshots_dir=project_dir.parent,
        project_identifier=project_dir.name,
    )


def resolve_pull_workspace(
    project_path: str,
    target_workspace_dir: Optional[Path] = None,
    source_host: Optional[str] = None,
) -> dict:
    ws_dir = target_workspace_dir
    if ws_dir is None:
        matches = paths.find_all_matching_workspaces(
            project_path, source_host=source_host
        )
        if matches:
            ws_dir = matches[0]["workspace_dir"]
            if source_host is None:
                source_host = matches[0].get("host")
    return {
        "path": project_path,
        "workspace_dir": ws_dir,
        "host": source_host,
        "type": "ssh" if source_host else "local",
    }


def format_pull_summary(plan: syncstate.PullPlan) -> list[str]:
    lines: list[str] = []
    n_sync = len(plan.synced)
    n_behind = len(plan.behind)
    n_missing = len(plan.missing_local)
    n_ahead = len(plan.ahead)
    n_div = len(plan.diverged)
    n_unk = len(plan.unknown)
    n_collision = len(plan.collisions)
    if n_sync:
        lines.append(f"{n_sync} already synced")
    if n_behind:
        lines.append(f"{n_behind} behind")
    if n_missing:
        lines.append(f"{n_missing} missing locally")
    if n_ahead:
        lines.append(f"{n_ahead} local ahead")
    if plan.never_pushed:
        lines.append(f"{plan.never_pushed} local only")
    if n_div:
        lines.append(f"{n_div} diverged")
    if n_unk:
        lines.append(f"{n_unk} unknown")
    if n_collision:
        lines.append(f"{n_collision} existing elsewhere")
    return lines


def _print_summary(plans: list[syncstate.PullPlan]) -> None:
    merged = syncstate.PullPlan()
    for plan in plans:
        merged.items.extend(plan.items)
        merged.never_pushed += plan.never_pushed
    for line in format_pull_summary(merged):
        print(line)
    print()


def _cursor_running_blocks(force: bool) -> bool:
    syncstate._counts.cursor_running_checks += 1
    if not force and is_cursor_running():
        print(
            "WARNING: Cursor is running. Close Cursor FIRST (Cmd+Q / quit),\n"
            "then import, then reopen Cursor. If you import while Cursor is\n"
            "running, Cursor will overwrite the sidebar registration on exit\n"
            "and the imported chats will disappear.\n"
            "Use --force to import anyway (not recommended).\n",
        )
        return True
    return False


def _import_staged(candidates: list[syncstate.PullItem]) -> tuple[int, int]:
    imported = 0
    failed = 0
    try:
        with importer.ImportSession() as batch:
            for item in candidates:
                ok = batch.import_snapshot(item)
                if ok is True:
                    imported += 1
                elif ok is False:
                    failed += 1
        return imported, failed
    finally:
        db.finish_cursor_writes()


def run_workspace_pull(
    project_path: str,
    target_workspace_dir: Optional[Path] = None,
    source_host: Optional[str] = None,
    force: bool = False,
    restore_all: bool = False,
    selected_paths: Optional[list[Path]] = None,
) -> PullResult:
    """Classify one origin, stage import candidates, then batch-import.

    No Cursor writes, backups, write lock, or running-check when there
    is nothing to import. Backend fetch is the caller's job.
    """
    return run_multi_target_pull(
        [
            (
                resolve_pull_workspace(
                    project_path, target_workspace_dir, source_host
                ),
                selected_paths,
            )
        ],
        force=force,
        restore_all=restore_all,
    )


def run_multi_target_pull(
    targets: list[tuple[dict, Optional[list[Path]]]],
    *,
    force: bool = False,
    restore_all: bool = False,
) -> PullResult:
    staging = Path(tempfile.mkdtemp(prefix="cursaves-pull-"))
    plans: list[syncstate.PullPlan] = []
    try:
        with dblock.repo_lock():
            with syncstate.SyncReadSession() as session:
                for workspace, selected in targets:
                    index = snapshot_index_for_target(workspace, selected)
                    plan = syncstate.build_pull_plan(
                        session,
                        index,
                        workspace,
                        restore_all=restore_all,
                        selected_paths=selected,
                    )
                    syncstate.stage_import_candidates(plan, staging)
                    plans.append(plan)

        print()
        _print_summary(plans)
        candidates = [
            item
            for plan in plans
            for item in plan.import_candidates
            if item.staged_path is not None
        ]
        if not candidates:
            print("Nothing to import.")
            return PullResult(plan=plans[0] if plans else None, plans=plans)

        if _cursor_running_blocks(force):
            return PullResult(plan=plans[0] if plans else None, plans=plans)

        n_div = sum(len(p.diverged) for p in plans)
        n_unk = sum(len(p.unknown) for p in plans)
        n_collision = sum(len(p.collisions) for p in plans)
        print(f"Importing {len(candidates)} conversation(s)...")
        imported, failed = _import_staged(candidates)
        print(f"  {imported} imported")
        if failed:
            print(f"  {failed} failed")
        if n_div:
            print(f"  {n_div} diverged — skipped")
        if n_unk:
            print(f"  {n_unk} unknown — skipped")
        if n_collision:
            print(f"  {n_collision} existing elsewhere — skipped")
        return PullResult(
            imported=imported,
            failed=failed,
            plan=plans[0] if plans else None,
            plans=plans,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
