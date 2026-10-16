"""Task plan for the request: the model creates and updates it with the `tasks` tool; the TUI shows progress."""

from __future__ import annotations

import json
import re

MAX_ITEMS = 8
ICONS = {"pending": "⏳", "in progress": "▶", "done": "✅", "failed": "⚠", "question": "❓", "cancelled": "—"}
ACTION_STATUS = {"start": "in progress", "done": "done", "failed": "failed", "ask": "question"}
KINDS = ("browser", "research", "write", "send", "verify", "playbook", "ask")
ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "maxLength": 300},
        "kind": {"type": "string", "enum": list(KINDS)},
        "recipient": {"type": "string", "description": "For send: the intended chat/person, or 'self'."},
        "expected_messages": {"type": "integer", "minimum": 1, "maximum": 10,
                              "description": "For send: default 1. Input chunks are never separate messages."},
    },
    "required": ["text", "kind"],
    "additionalProperties": False,
}


def task_kind(text: str) -> str:
    """Compatibility for old string plans: maintenance takes precedence over quoted action verbs."""
    if re.match(r"^\s*(update|revise|correct|save|learn|record|maintain)\b.*\b(playbook|rule|lesson|teach|learning)\b",
                text, re.I):
        return "playbook"
    if re.search(r"\b(ask\s+(?:the\s+)?user|ask\s+which|let\s+the\s+user\s+(?:choose|pick|confirm|select))\b",
                 text, re.I):
        return "ask"
    if re.match(r"^\s*(verify|check|confirm)\b", text, re.I):
        return "verify"
    if re.search(r"\b(invitation|invite|connection|request|follow|search)\b", text, re.I):
        return "browser"
    if re.search(r"\b(send|reply|submit)\b", text, re.I):
        return "send"
    if re.match(r"^\s*(write|draft|summarize|compose|type|fill)\b", text, re.I):
        return "write"
    if re.match(r"^\s*(research|review|collect)\b", text, re.I):
        return "research"
    return "browser"
LOW_CONFIDENCE = ("Error: not marked done - confidence is low. Verify first (read, find or screenshot); "
                  "if you still cannot confirm it, use action=failed or action=ask with a question for the user.")
NEEDS_NOTE = "Error: action={action} needs a note ({what})."

TOOL = {
    "type": "function",
    "function": {
        "name": "tasks",
        "description": (
            "Task plan (English). plan+items: typed objects with text and kind (browser, research, write, send, "
            "verify, playbook, ask); reach exactly what was asked. Use kind=ask when a step asks the user to "
            "choose or confirm an option via action=ask. "
            "Updating a playbook is kind=playbook, NEVER kind=send even if its rule mentions sending. "
            "For research done, sources lists the URLs actually read with verified publication dates. "
            "cancel stops all unfinished work without executing it. The first task starts by "
            "itself and done/failed start the next. ask pauses the plan until the user answers. "
            "done+index+note (facts SEEN) - better: done=N on your "
            "next browser/files call; several finished at once: done+indexes; failed+index+note (reason); "
            "ask+index+note (question for the user)+options (2-4 short answers, e.g. Yes/No)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["plan", "done", "failed", "ask", "cancel"]},
                "items": {"type": "array", "items": {"anyOf": [ITEM_SCHEMA, {"type": "string"}]},
                          "maxItems": MAX_ITEMS},
                "index": {"type": "integer"},
                "indexes": {"type": "array", "items": {"type": "integer"},
                            "description": "With done: several tasks finished by the same result, in one call."},
                "note": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"},
                            "description": "With ask: 2-4 short answers the user can pick."},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "sources": {"type": "array", "items": {"type": "string"},
                            "description": "For research done: selected source URLs, without duplicates."},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}


class Plan:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self.date_counts: dict[str, int] = {}
        self.partial_approved = False
        self.sources: dict[str, dict] = {}
        self.selected_sources: list[str] = []

    def clear(self) -> None:
        self.items = []
        self.date_counts = {}
        self.partial_approved = False
        self.sources = {}
        self.selected_sources = []

    def cancel(self, note: str = "Stopped by the user.") -> str:
        for item in self.items:
            if item["status"] in ("pending", "in progress", "question"):
                item.update(status="cancelled", note=note)
        return self.render()

    def pending(self) -> list[int]:
        return [n for n, item in enumerate(self.items, 1) if item["status"] in ("pending", "in progress")]

    def unfinished(self) -> list[int]:
        """Pending plus waiting for the user's answer: resumed on the next turn."""
        return [n for n, item in enumerate(self.items, 1) if item["status"] in ("pending", "in progress", "question")]

    def summary(self) -> str:
        if not self.items:
            return "no plan"
        done = sum(item["status"] == "done" for item in self.items)
        failed = sum(item["status"] == "failed" for item in self.items)
        asked = sum(item["status"] == "question" for item in self.items)
        return (f"{done}/{len(self.items)} done" + (f", {failed} failed" if failed else "")
                + (f", {asked} waiting for you" if asked else ""))

    def render(self) -> str:
        """Model version: plain text, with the notes of finished tasks."""
        lines = [f"Plan ({self.summary()}):"]
        if self.date_counts:
            lines.append("Publication-date coverage: " + json.dumps(self.date_counts, sort_keys=True)
                         + (" (partial coverage explicitly approved)" if self.partial_approved else ""))
        for n, item in enumerate(self.items, 1):
            note = f" - {item['note']}" if item["note"] else ""
            lines.append(f"{n}. [{item['status']}] [{item.get('kind', 'browser')}] {item['text']}{note}")
        return "\n".join(lines)

    def render_tui(self) -> str:
        """TUI version: icons and no long notes."""
        lines = [f"Plan - {self.summary()}"]
        for n, item in enumerate(self.items, 1):
            lines.append(f"{ICONS[item['status']]} {n}. {item['text']}")
        return "\n".join(lines)

    def apply(self, raw_arguments: str) -> str:
        """Runs the `tasks` tool and returns the updated plan (or an error for the model to fix)."""
        try:
            arguments = json.loads(raw_arguments or "{}")
        except ValueError:
            return "Error: invalid arguments."
        if not isinstance(arguments, dict):
            return "Error: invalid arguments."
        action = arguments.get("action")
        if action == "cancel":
            return self.cancel(str(arguments.get("note") or "Stopped by the user."))
        if action == "plan":
            raw_items = arguments.get("items")
            if not isinstance(raw_items, list) or len(raw_items) > MAX_ITEMS:
                return f"Error: items must be a list of at most {MAX_ITEMS} tasks."
            items = []
            for raw in raw_items:
                if isinstance(raw, str):
                    raw = {"text": raw, "kind": task_kind(raw)}
                if not isinstance(raw, dict) or raw.get("kind") not in KINDS or not isinstance(raw.get("text"), str):
                    return "Error: each task needs text and a valid kind."
                text = " ".join(raw["text"].split())
                if not text:
                    continue
                if len(text) > 300:
                    return "Error: task text exceeds 300 characters; shorten the label, not its requirements."
                count = raw.get("expected_messages", 1)
                if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 10:
                    return "Error: expected_messages must be from 1 to 10."
                recipient = raw.get("recipient", "")
                if not isinstance(recipient, str) or len(recipient) > 2000:
                    return "Error: recipient must be text of at most 2000 characters."
                kind = "playbook" if task_kind(text) == "playbook" else raw["kind"]
                items.append({"text": text, "kind": kind, "recipient": recipient,
                              "expected_messages": count, "status": "pending", "note": ""})
            if not items:
                return "Error: send items with at least one task."
            # Re-planning keeps the progress of tasks that are still in the plan (same text).
            old = {(item["text"], item.get("kind")): item for item in self.items}
            self.items = [dict(old.get((item["text"], item["kind"]), item)) for item in items]
            if not any(item["status"] == "in progress" for item in self.items):
                # Saves a model call: the first pending task starts with the plan.
                nxt = next((item for item in self.items if item["status"] == "pending"), None)
                if nxt:
                    nxt["status"] = "in progress"
            return self.render()
        if action not in ACTION_STATUS:
            return "Error: action must be plan, start, done, failed or ask."
        index = arguments.get("index")
        if not isinstance(index, int) or not 1 <= index <= len(self.items):
            return f"Error: index must be from 1 to {len(self.items)}." if self.items else "Error: create the plan first."
        note = " ".join(str(arguments.get("note") or "").split())[:600]
        if action in ("done", "failed", "ask") and not note:
            what = {"done": "the facts you saw", "failed": "the reason", "ask": "the question"}[action]
            return NEEDS_NOTE.format(action=action, what=what)
        if action == "done" and arguments.get("confidence") == "low":
            return LOW_CONFIDENCE
        item = self.items[index - 1]
        item["status"] = ACTION_STATUS[action]
        if action == "ask":
            for other in self.items:
                if other is not item and other["status"] == "in progress":
                    other["status"] = "pending"
        if note:
            item["note"] = note
        if action in ("done", "failed") and not any(i["status"] == "in progress" for i in self.items):
            # Saves a model call: the next pending task starts by itself.
            nxt = next((i for i in self.items if i["status"] == "pending"), None)
            if nxt:
                nxt["status"] = "in progress"
        return self.render()
