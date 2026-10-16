"""Optional planner: a configured cloud or local model maps requests to typed plans.

The configured executor then runs the plan in the browser. Planning is the most critical
step (what exactly to do, for whom, how many) and only one or two calls per request, so a stronger model is
cheap there. If the planner is off or fails, the executor plans by itself, as before.

BROWSER_HARNESS_PLANNER = deployment name; "0" turns it off.
BROWSER_HARNESS_PLANNER_REASONING = low | medium | high (default low).
BROWSER_HARNESS_PLANNER_SCOPE = actions (default: requests that act on others or bring a list) | all.
"""

from __future__ import annotations

import json
import os
import re
import time
from threading import Event
from typing import Any
from runtime_context import clock_context
from plan import ITEM_SCHEMA

PLANNER = os.getenv("BROWSER_HARNESS_PLANNER", "")
REASONING = os.getenv("BROWSER_HARNESS_PLANNER_REASONING", "low")
# actions = only requests that act on other people/accounts or bring a list (reads/summaries: the executor plans);
# all = every new request.
SCOPE = os.getenv("BROWSER_HARNESS_PLANNER_SCOPE", "actions")
MAX_TASKS = 8
MAX_OPTIONS = 4

RULES = """You are the PLANNER of a browser agent that works in the user's real browser (commands: read <url>,
find <text>, click, fill, press, goto, screenshot; files on disk; guides the user taught). Another model executes
your plan step by step. You only turn the user's request into the plan, or ask when it is not clear.

Rules:
- Tasks: concrete steps in request order, in English, at most 8, that reach EXACTLY what was asked, nothing more.
  Each task is an object with text and kind (browser, research, write, send, verify, playbook, ask).
  When the user asks to be asked/confirm/choose between found items before the next step, include a task with
  kind="ask" for that choice gate, followed by the task that acts on their choice.
  A correction to a playbook is kind=playbook, never send: do not execute the rule being edited.
  For send include recipient (or "self") and expected_messages (default 1). Compose long text in one draft;
  tool-input chunks do not mean separate messages. Never shorten or split the requested content unless asked.
  One task per item when the request has a list (e.g. one per contact). Each task at most ~120 characters: what to
  do and for whom (names, companies, language, length, where to save or send). Routes, URLs and rules that apply
  to several tasks go ONCE in "notes".
- Read the request as a person would: greetings and nicknames (mano, amigo, bro, pls) are not names; "the two"
  must match two items; the user's own chat/account is "the user's own".
- Ask instead of guessing when the request is ambiguous, contradicts itself, or misses something needed to act on
  other people or accounts (who, which company, which chat, how many). Ask ONE short question in the user's
  language with 2-4 short options (the most likely first). Never ask what the executor can see on the page.
- Use the lessons and guides below as the fastest route (e.g. direct URLs) and mention it in the task when useful.
- Use the authoritative local dates below. Preserve coverage per date, output language, a comprehensive summary
  PLUS any requested outcome paragraphs, and source URLs. Do not turn "with two outcome paragraphs" into
  "exactly two paragraphs total". Ask before substituting sources or reducing coverage.
- Reply ONLY with JSON, one of:
  {"plan": [{"text": "task 1", "kind": "research"}, {"text": "task 2", "kind": "send",
             "recipient": "self", "expected_messages": 1}], "notes": "short hints (optional)"}
  {"question": "...", "options": ["...", "..."]}"""


class Planner:
    def __init__(self, model: Any, scope: str = SCOPE) -> None:
        self.model = model
        self.scope = scope
        self.clock = ""
        self.cancel: Event | None = None

    @property
    def deployment(self) -> str:
        return getattr(self.model, "usage_name", None) or getattr(self.model, "deployment", None) or "planner"

    def close(self) -> None:
        close = getattr(self.model, "close", None)
        if close:
            close()

    def prompt(self, request: str, lessons: str, guides: str, history: str,
               answers: list[tuple[str, str]]) -> str:
        parts = [RULES,
                 self.clock or clock_context(),
                 "Lessons from earlier sessions:\n" + (lessons or "(none)"),
                 "Guides the user taught:\n" + (guides or "(none)"),
                 "Recent conversation:\n" + (history or "(none)"),
                 "Request:\n" + request]
        if answers:
            parts.append("The user already answered (do not ask again):\n"
                         + "\n".join(f"- Q: {question} -> A: {answer}" for question, answer in answers))
        return "\n\n".join(parts)

    def plan(self, request: str, lessons: str = "", guides: str = "", history: str = "",
             answers: list[tuple[str, str]] | None = None) -> tuple[dict | None, Any, int]:
        """(decision, turn, ms): decision = {"plan", "notes"} or {"question", "options"}; None if unusable."""
        started = time.monotonic()
        turn = self.model.complete([{"role": "user", "content": self.prompt(request, lessons, guides, history,
                                                                             answers or [])}],
                                   [], lambda kind, text: None, self.cancel or Event())
        return parse(turn.content), turn, int((time.monotonic() - started) * 1000)


def parse(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    question = " ".join(str(data.get("question") or "").split())
    if question:
        options = [" ".join(str(option).split())[:80] for option in data.get("options") or [] if str(option).strip()]
        return {"question": question[:400], "options": options[:MAX_OPTIONS]}
    plan = []
    for item in data.get("plan") or []:
        if isinstance(item, str):
            text = " ".join(item.split())
            if text:
                plan.append(text)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            text = " ".join(item["text"].split())
            if item.get("kind") not in ITEM_SCHEMA["properties"]["kind"]["enum"] or not text:
                return None
            if set(item) - set(ITEM_SCHEMA["properties"]):
                return None
            plan.append({**item, "text": text})
        else:
            return None
        if len(text) > 300:
            return None
    if not plan:
        return None
    return {"plan": plan[:MAX_TASKS], "notes": " ".join(str(data.get("notes") or "").split())[:500]}


def from_env():
    """The planner configured for this machine, or None (off: the executor plans)."""
    from model_config import Settings
    settings = Settings.load()
    model = settings.create_model("planner")
    return Planner(model, scope=settings.planner_scope) if model else None
