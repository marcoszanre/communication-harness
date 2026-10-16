"""Session log in JSONL (logs/YYYYMMDD-HHMMSS.jsonl), to review and improve the harness later.

Each line is an event: turn_start, model_call, tool, tasks, pushback, memory, filtered, turn_end.
Page content is not saved in full, only a short excerpt (preview).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock

HERE = Path(__file__).resolve().parent
LOGS_DIR = HERE / "logs"
PREVIEW = 300


class SessionLog:
    def __init__(self, folder: Path = LOGS_DIR) -> None:
        self.path = folder / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
        self._lock = Lock()

    def write(self, event: str, **data) -> None:
        record = {"ts": datetime.now().isoformat(timespec="seconds"), "event": event, **data}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")


class NullLog:
    path = None

    def write(self, event: str, **data) -> None:
        pass


def preview(text: str) -> str:
    return text[:PREVIEW] + ("..." if len(text) > PREVIEW else "")
