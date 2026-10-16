"""Adapt an existing Windows Terminal Shift+Enter binding for this app only."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _install_binding(settings_path: Path) -> bool:
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot inspect Windows Terminal key bindings: {error}") from error

    action_list = settings.get("actions") or []
    bindings = settings.get("keybindings") or []
    if not isinstance(action_list, list) or not isinstance(bindings, list):
        raise ValueError("Windows Terminal key bindings have an invalid format.")
    actions = {action.get("id"): action.get("command") for action in action_list}
    for binding in bindings:
        keys = binding.get("keys")
        if keys != "shift+enter" and not (isinstance(keys, list) and "shift+enter" in keys):
            continue
        command = binding.get("command") or actions.get(binding.get("id"))
        if not isinstance(command, dict) or command.get("action") != "sendInput":
            return False
        if command.get("input") != "\x1b\r":
            return False
        try:
            from textual._ansi_sequences import ANSI_SEQUENCES_KEYS
            from textual.keys import Keys

            newline = Keys.ControlJ
        except (ImportError, AttributeError) as error:
            raise RuntimeError("This Textual version cannot handle the terminal's Shift+Enter binding.") from error

        current = ANSI_SEQUENCES_KEYS.get("\x1b\r")
        if current not in (None, (newline,)):
            raise RuntimeError("Shift+Enter already has a different keyboard mapping.")
        # Textual otherwise drops ESC and treats this terminal binding as plain Enter.
        ANSI_SEQUENCES_KEYS["\x1b\r"] = (newline,)
        return True
    return False


def enable_shift_enter() -> bool:
    if sys.platform != "win32" or not os.getenv("WT_SESSION"):
        return False
    local_app_data = os.getenv("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("LOCALAPPDATA is required to inspect the Windows Terminal binding.")
    root = Path(local_app_data) / "Packages"
    for package in ("Microsoft.WindowsTerminal_8wekyb3d8bbwe", "Microsoft.WindowsTerminalPreview_8wekyb3d8bbwe"):
        settings = root / package / "LocalState" / "settings.json"
        if settings.is_file():
            return _install_binding(settings)
    return False
