"""Interactive TUI selection using InquirerPy (arrow keys, fuzzy search, checkboxes)."""

from __future__ import annotations

from typing import Any

from InquirerPy import inquirer
from InquirerPy.base.control import Choice


def select_one(
    choices: list[dict],
    message: str = "Select:",
    name_key: str = "name",
    value_key: str | None = None,
) -> dict | None:
    """Single-select from a list using arrow keys and fuzzy search.

    Each choice dict should have at least a `name_key` field for display.
    Returns the selected dict, or None if cancelled.
    """
    if not choices:
        return None

    inq_choices = []
    for c in choices:
        display = c.get(name_key, str(c))
        val = c if value_key is None else c.get(value_key)
        inq_choices.append(Choice(value=val, name=display))

    try:
        result = inquirer.fuzzy(
            message=message,
            choices=inq_choices,
            max_height="70%",
            mandatory=False,
        ).execute()
    except (KeyboardInterrupt, EOFError):
        return None

    return result


def select_many(
    choices: list[dict],
    message: str = "Select (space to toggle, enter to confirm):",
    name_key: str = "name",
    value_key: str | None = None,
    default_all: bool = False,
) -> list[dict]:
    """Multi-select using fuzzy search with arrow keys.

    Type to filter, space to toggle, enter to confirm.
    Ctrl+A to toggle all visible.

    Args:
        choices: List of dicts to display.
        message: Prompt message.
        name_key: Key in each dict to use for display.
        value_key: Key to use for the return value (None = return full dict).
        default_all: If True, all items are pre-selected.

    Returns list of selected items (empty if cancelled).
    """
    if not choices:
        return []

    inq_choices = []
    for c in choices:
        display = c.get(name_key, str(c))
        val = c if value_key is None else c.get(value_key)
        inq_choices.append(
            Choice(value=val, name=display, enabled=default_all)
        )

    try:
        result = inquirer.fuzzy(
            message=message,
            choices=inq_choices,
            max_height="70%",
            multiselect=True,
            mandatory=False,
            keybindings={"toggle": [{"key": "space"}]},
        ).execute()
    except (KeyboardInterrupt, EOFError):
        return []

    if result is None:
        return []
    if isinstance(result, list):
        return result
    return [result]


def confirm(message: str = "Continue?", default: bool = False) -> bool:
    """Simple yes/no confirmation."""
    try:
        return inquirer.confirm(message=message, default=default).execute()
    except (KeyboardInterrupt, EOFError):
        return False


def select_workspace(workspaces: list[dict]) -> dict | None:
    """Select a workspace from a list using fuzzy search.

    Each workspace dict should have 'path', 'host' (optional),
    and 'conversations' count.
    """
    import os

    if not workspaces:
        return None

    choices = []
    for ws in workspaces:
        name = os.path.basename(os.path.normpath(ws["path"])) or ws["path"]
        host = ws.get("host", "")
        convos = ws.get("conversations", 0)
        label = f"{name} ({host})" if host else name
        choices.append({
            "name": f"{label:<40} {convos:>3} refs",
            "_ws": ws,
        })

    selected = select_one(choices, message="Select workspace:", name_key="name")
    if selected is None:
        return None
    return selected["_ws"]


def select_conversations(
    conversations: list[dict],
    action: str = "push",
) -> list[str]:
    """Multi-select conversations using checkboxes.

    Each conversation dict should have 'id', 'name', 'messageCount'.
    Returns list of selected composer IDs.
    """
    if not conversations:
        return []

    choices = []
    for c in conversations:
        name = c.get("name", "Untitled")
        msgs = c.get("messageCount", 0)
        last = (c.get("lastUpdated", "") or "")[:16]
        display = f"{name:<38} {msgs:>4} msgs  {last}"
        choices.append({
            "name": display,
            "composerId": c["id"],
        })

    selected = select_many(
        choices,
        message=f"Select chats to {action} (space=toggle, type to filter):",
        name_key="name",
        default_all=True,
    )

    if not selected:
        return []
    return [s["composerId"] for s in selected]


def select_purge_chats(
    chats: list[dict],
) -> list[str]:
    """Multi-select chats for purging, grouped by workspace.

    Each chat dict has: composerId, name, messageCount, keyCount,
    workspace_label.
    Returns list of selected composer IDs.
    """
    if not chats:
        return []

    # Sort by workspace then by keyCount desc for natural grouping
    sorted_chats = sorted(
        chats,
        key=lambda c: (c["workspace_label"], -c["keyCount"]),
    )

    choices = []
    for c in sorted_chats:
        name = c.get("name") or "(unnamed)"
        if len(name) > 30:
            name = name[:27] + "..."
        msgs = c["messageCount"]
        keys = c["keyCount"]
        ws = c["workspace_label"]
        if len(ws) > 18:
            ws = ws[:15] + "..."
        display = f"{ws:<18} │ {name:<32} {msgs:>4} msgs  {keys:>5} keys"
        choices.append({
            "name": display,
            "composerId": c["composerId"],
        })

    selected = select_many(
        choices,
        message="Select chats to delete (space=toggle, type to filter):",
        name_key="name",
    )

    if not selected:
        return []
    return [s["composerId"] for s in selected]


def select_snapshots(
    snapshots: list[dict],
) -> list[Any]:
    """Multi-select snapshots for import.

    Each snapshot dict has: name, msgs, source, file.
    Returns the selected snapshot dicts.
    """
    if not snapshots:
        return []

    choices = []
    for s in snapshots:
        name = s.get("name", "Untitled")
        if len(name) > 34:
            name = name[:31] + "..."
        msgs = s.get("msgs", 0)
        source = s.get("source", "unknown")
        if len(source) > 14:
            source = source[:11] + "..."
        display = f"{name:<36} {msgs:>4} msgs  from {source}"
        choices.append({
            "name": display,
            "_snapshot": s,
        })

    selected = select_many(
        choices,
        message="Select chats to import (space=toggle, type to filter):",
        name_key="name",
        default_all=True,
    )

    if not selected:
        return []
    return [s["_snapshot"] for s in selected]
