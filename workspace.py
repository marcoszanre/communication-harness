"""Per-session workspace: a folder on disk where the run keeps its assets.

workspaces/<session-id>/
  index.md        requests of the session and every saved file (one line each)
  pages/NN-*.md   text of each page the agent read (saved automatically)
  shots/NN.jpg    screenshots the agent looked at
  anything else   notes, summaries, drafts and results the model writes with the `files` tool

The folder is created on the first write. /new (or /clear) starts another session with another folder;
earlier sessions stay on disk and can be consulted with `files action=sessions`.
"""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACES_DIR = HERE / "workspaces"
MAX_READ = 8_000       # chars returned per files read (use start=N for the rest)
MAX_WRITE = 50_000
MAX_LIST = 60
MAX_SESSIONS = 10
TEXT_SUFFIXES = {".md", ".txt", ".json", ".csv", ".html", ".xml", ".yml", ".yaml", ".log"}

TOOL = {
    "type": "function",
    "function": {
        "name": "files",
        "description": (
            "Session folder on disk (read pages are saved in pages/, screenshots in shots/). write/append "
            "path+content; read path [start] (reopen a page already read instead of the browser); list; "
            "sessions (earlier ones, then list/read with session)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "read", "write", "append", "sessions"]},
                "path": {"type": "string"},
                "content": {"type": "string"},
                "session": {"type": "string"},
                "start": {"type": "integer"},
                "done": {"type": "integer",
                         "description": "Optional: number of a task this call finishes (e.g. writing the summary); "
                                        "it is closed in the same call."},
                "note": {"type": "string", "description": "With done: the facts that prove it."},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}


def new_session_id(root: Path) -> str:
    base = f"{datetime.now():%Y%m%d-%H%M%S}"
    session_id, n = base, 2
    while (root / session_id).exists():
        session_id, n = f"{base}-{n}", n + 1
    return session_id


def slug(text: str, size: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:size].strip("-") or "page"


class Workspace:
    def __init__(self, root: Path = WORKSPACES_DIR, session_id: str | None = None) -> None:
        self.root = root.resolve()  # real path (Windows 8.3 short names like USERNA~1 break the containment check)
        self.id = session_id or new_session_id(self.root)
        self.folder = self.root / self.id
        self._saved: dict[int, str] = {}

    def new(self) -> "Workspace":
        """Next session: another folder (this one stays on disk)."""
        return Workspace(self.root)

    # ---------- automatic assets ----------
    def _next_name(self, sub: str, stem: str, suffix: str) -> Path:
        folder = self.folder / sub
        folder.mkdir(parents=True, exist_ok=True)
        number = len(list(folder.iterdir())) + 1
        return folder / f"{number:02d}-{stem}{suffix}" if stem else folder / f"{number:02d}{suffix}"

    def _index(self, line: str) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        with (self.folder / "index.md").open("a", encoding="utf-8") as file:
            file.write(f"- {datetime.now():%H:%M:%S} {line}\n")

    def note_request(self, prompt: str) -> None:
        self._index("request: " + " ".join(prompt.split())[:300])

    def save_page(self, url: str, text: str) -> str:
        """Saves the text of a page read by the agent; returns the relative path (same text -> same file)."""
        key = hash((url, text))
        if key in self._saved:
            return self._saved[key]
        name = url.split("?")[0].split("#")[0].rstrip("/").rsplit("/", 1)[-1] or url
        path = self._next_name("pages", slug(name), ".md")
        path.write_text(f"Source: {url or '?'}\nSaved: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n{text}",
                        encoding="utf-8")
        relative = path.relative_to(self.folder).as_posix()
        self._saved[key] = relative
        self._index(f"page {relative} <- {url or '?'}")
        return relative

    def save_image(self, data_url: str, url: str = "") -> str:
        """Saves a screenshot (data URI); returns the relative path or '' on failure."""
        header, _, data = data_url.partition(",")
        if not data or ";base64" not in header:
            return ""
        suffix = ".png" if "png" in header else ".jpg"
        path = self._next_name("shots", "", suffix)
        path.write_bytes(base64.b64decode(data))
        relative = path.relative_to(self.folder).as_posix()
        self._index(f"screenshot {relative} <- {url or '?'}")
        return relative

    # ---------- the `files` tool ----------
    def apply(self, raw_arguments: str) -> str:
        try:
            arguments = json.loads(raw_arguments or "{}")
            if not isinstance(arguments, dict):
                return "Error: invalid arguments."
            action = arguments.get("action")
            if action == "sessions":
                return self.sessions()
            folder = self._session_folder(str(arguments.get("session") or ""))
            if action == "list":
                return self.list(folder)
            path = self._resolve(folder, str(arguments.get("path") or ""))
            if action == "read":
                return self.read(path, int(arguments.get("start") or 0))
            if action in ("write", "append"):
                if folder != self.folder:
                    return "Error: earlier sessions are read-only; write in this session."
                return self.write(path, str(arguments.get("content") or ""), append=action == "append")
            return "Error: action must be list, read, write, append or sessions."
        except (ValueError, OSError, TypeError) as error:
            return f"Error: {error}"

    def _session_folder(self, session: str) -> Path:
        if not session or session == self.id:
            return self.folder
        folder = (self.root / session).resolve()
        if folder.parent != self.root.resolve() or not folder.is_dir():
            raise ValueError(f"unknown session: {session}. Use action=sessions.")
        return folder

    @staticmethod
    def _resolve(folder: Path, relative: str) -> Path:
        relative = relative.strip().replace("\\", "/").lstrip("/")
        if not relative:
            raise ValueError("send path (e.g. summary.md).")
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError("path must stay inside the workspace.")
        return path

    def list(self, folder: Path | None = None) -> str:
        folder = folder or self.folder
        files = sorted(p for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else []
        if not files:
            return f"Workspace {folder.name}: empty."
        lines = [f"- {p.relative_to(folder).as_posix()} ({p.stat().st_size:,} bytes)" for p in files[:MAX_LIST]]
        more = f"\n(+{len(files) - MAX_LIST} more)" if len(files) > MAX_LIST else ""
        return f"Workspace {folder.name} ({len(files)} files):\n" + "\n".join(lines) + more

    @staticmethod
    def read(path: Path, start: int = 0) -> str:
        if not path.is_file():
            return f"Error: file not found: {path.name}. Use action=list."
        if path.suffix.lower() not in TEXT_SUFFIXES:
            return f"Error: {path.name} is not a text file (images cannot be read here)."
        text = path.read_text(encoding="utf-8", errors="replace")
        start = max(0, start)
        chunk = text[start:start + MAX_READ]
        rest = len(text) - start - len(chunk)
        return chunk + (f"\n[... {rest:,} more chars: read again with start={start + len(chunk)}]" if rest > 0 else "")

    def write(self, path: Path, content: str, append: bool = False) -> str:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            return f"Error: use a text extension ({', '.join(sorted(TEXT_SUFFIXES))})."
        if not content:
            return "Error: send content."
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a" if append else "w", encoding="utf-8") as file:
            file.write(content[:MAX_WRITE] + ("\n" if append and not content.endswith("\n") else ""))
        relative = path.relative_to(self.folder).as_posix()
        self._index(f"{'appended' if append else 'wrote'} {relative}")
        return f"Saved {relative} ({path.stat().st_size:,} bytes) in workspace {self.id}."

    def sessions(self) -> str:
        folders = sorted((p for p in self.root.iterdir() if p.is_dir()), reverse=True) if self.root.is_dir() else []
        lines = []
        for folder in folders[:MAX_SESSIONS]:
            index = folder / "index.md"
            first = next((line.split("request: ", 1)[1] for line in
                          (index.read_text(encoding="utf-8").splitlines() if index.is_file() else [])
                          if "request: " in line), "")
            count = sum(1 for p in folder.rglob("*") if p.is_file())
            current = " (this session)" if folder.name == self.id else ""
            lines.append(f"- {folder.name}{current}: {count} files; first request: {first[:100] or '-'}")
        return ("Sessions (newest first):\n" + "\n".join(lines)) if lines else "No sessions yet."


class NullWorkspace:
    """Tests and runs without a workspace: nothing is written."""

    id = "none"
    folder = None

    def new(self) -> "NullWorkspace":
        return self

    def note_request(self, prompt: str) -> None:
        pass

    def save_page(self, url: str, text: str) -> str:
        return ""

    def save_image(self, data_url: str, url: str = "") -> str:
        return ""

    def apply(self, raw_arguments: str) -> str:
        return "Error: workspace not available."
