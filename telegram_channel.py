"""Local Telegram Bot API channel (Push + Pull) with 3 progressive capability levels.

Architecture overview (no cloud webhooks, no external event bus):
- Push (outbound): HTTP POST to `sendMessage`, `editMessageText`, `answerCallbackQuery`.
- Pull (inbound):  HTTP GET/POST to `getUpdates` from the local harness process.

Capability levels:
1. `status` (Level 1 — Task Update): Push only. Streams task start, live plan progress, and final outcome.
2. `hitl`   (Level 2 — HITL):        Level 1 + mirrors inline approvals (`ApprovalPanel`) and clarifying
                                     questions (`QuestionPanel`) to Telegram. First responder (TUI or
                                     Telegram) resolves the gate on both surfaces.
3. `tasks`  (Level 3 — Accept Tasks): Level 2 + accepts new natural-language tasks from Telegram when the
                                     harness is idle, enforcing surface-specific command guardrails and a
                                     Single-Active-Run (`busy-reject`) concurrency policy.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TELEGRAM_SETTINGS_PATH = Path(__file__).resolve().parent / "telegram.json"
TELEGRAM_API_BASE = "https://api.telegram.org"

TELEGRAM_LEVELS = ("off", "status", "hitl", "tasks")

TELEGRAM_LEVEL_CHOICES: tuple[tuple[str, str], ...] = (
    ("Off — External Telegram channel disabled", "off"),
    ("Level 1: Task Update — Push plan progress & final outcome only", "status"),
    ("Level 2: HITL — Push updates + approve actions & answer questions", "hitl"),
    ("Level 3: Accept Tasks — Full bidirectional: start tasks & resolve HITL", "tasks"),
)

TELEGRAM_LEVEL_LABELS: dict[str, str] = {
    "off": "Off",
    "status": "L1: Task Update",
    "hitl": "L2: HITL",
    "tasks": "L3: Accept Tasks",
}

TELEGRAM_LEVEL_SHORT: dict[str, str] = {
    "off": "off",
    "status": "L1:status",
    "hitl": "L2:hitl",
    "tasks": "L3:tasks",
}

# Local-only TUI commands that require physical terminal/browser interaction and are blocked on Telegram.
TUI_ONLY_COMMANDS = frozenset({
    "/teach",
    "/login",
    "/models",
    "/telegram",
    "/clear",
    "/reset",
    "/new",
    "/restart",
    "/close",
    "/quit",
    "/exit",
    "/playbooks",
    "/workspace",
    "/files",
    "/usage",
    "/memory",
})


def level_description(level: str) -> str:
    descriptions = {
        "off": "Channel disabled: no outbound notifications and no inbound polling.",
        "status": "Level 1 (Task Update — Push only): sends task start, live plan updates, and final outcome to Telegram.",
        "hitl": "Level 2 (HITL — Push + Pull gates): Level 1 + forwards approval and question prompts to Telegram buttons.",
        "tasks": "Level 3 (Accept Tasks — Full bidirectional): Level 2 + starts new tasks from Telegram when the harness is idle.",
    }
    return descriptions.get(level, descriptions["off"])


def markdown_to_telegram_html(text: str) -> str:
    """Converts CommonMark (bold, italic, inline/fenced code, headings, bullets, links) to safe Telegram HTML."""
    clipped = text[:3800]
    code_stash: list[str] = []

    def stash_pre(match: re.Match[str]) -> str:
        idx = len(code_stash)
        body = html.escape(match.group(1).strip("\r\n"), quote=False)
        code_stash.append(f"<pre>{body}</pre>")
        return f"\x00CODE{idx}\x00"

    def stash_inline(match: re.Match[str]) -> str:
        idx = len(code_stash)
        body = html.escape(match.group(1), quote=False)
        code_stash.append(f"<code>{body}</code>")
        return f"\x00CODE{idx}\x00"

    staged = re.sub(r"```(?:[^\n`]*)\n(.*?)```", stash_pre, clipped, flags=re.S)
    staged = re.sub(r"`([^`\n]+)`", stash_inline, staged)
    escaped = html.escape(staged, quote=False)

    # Links: [label](https://...)
    escaped = re.sub(
        r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)",
        lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>',
        escaped,
    )
    # Headings: ### Title -> <b>Title</b>
    escaped = re.sub(r"(?m)^#{1,6}\s+(.+)$", r"<b>\1</b>", escaped)
    # Bullet list items: - item -> • item
    escaped = re.sub(r"(?m)^(\s*)[-*]\s+", r"\1• ", escaped)
    # Bold: **text** or __text__
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped, flags=re.S)
    escaped = re.sub(r"__(.+?)__", r"<b>\1</b>", escaped, flags=re.S)
    # Italic: *text* (single asterisk within a line)
    escaped = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", escaped)

    for idx, snippet in enumerate(code_stash):
        escaped = escaped.replace(f"\x00CODE{idx}\x00", snippet)
    return escaped


def default_telegram_request(bot_token: str, method: str, payload: dict[str, Any] | None = None,
                             timeout: float = 10.0) -> dict[str, Any]:
    """Calls `https://api.telegram.org/bot<token>/<method>` using standard-library HTTPS."""
    token = bot_token.strip()
    if not token or ":" not in token or any(ch.isspace() for ch in token):
        raise ValueError("Telegram bot token must follow the '<bot_id>:<secret>' format from @BotFather.")
    url = f"{TELEGRAM_API_BASE}/bot{token}/{method}"
    body = json.dumps(payload or {}).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except HTTPError as error:
        detail = ""
        try:
            err_json = json.loads(error.read().decode("utf-8", errors="replace"))
            detail = err_json.get("description", "")
        except Exception:
            detail = str(error)
        raise RuntimeError(f"Telegram API error ({error.code}) on {method}: {detail or error.reason}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach Telegram API ({method}): {error.reason}") from error
    try:
        data = json.loads(raw)
    except ValueError as error:
        raise RuntimeError(f"Invalid JSON from Telegram API on {method}.") from error
    if not isinstance(data, dict) or not data.get("ok"):
        desc = data.get("description", "unknown error") if isinstance(data, dict) else "invalid payload"
        raise RuntimeError(f"Telegram API rejected {method}: {desc}")
    return data


@dataclass
class TelegramSettings:
    level: str = "off"
    bot_token: str = ""
    chat_id: str = ""
    bot_username: str = ""
    owner_label: str = ""

    @classmethod
    def load(cls, path: Path = TELEGRAM_SETTINGS_PATH) -> TelegramSettings:
        data: dict[str, Any] = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as error:
                raise ValueError(f"Cannot read Telegram settings: {error}") from error
            if not isinstance(data, dict) or set(data) - set(cls.__dataclass_fields__):
                raise ValueError("Telegram settings contain an invalid object or unknown fields.")
        level = str(data.get("level", "off")).strip().lower()
        bot_token = str(data.get("bot_token", "")).strip()
        chat_id = str(data.get("chat_id", "")).strip()
        bot_username = str(data.get("bot_username", "")).strip()
        owner_label = str(data.get("owner_label", "")).strip()

        env_token = os.getenv("TELEGRAM_BOT_TOKEN", os.getenv("BROWSER_HARNESS_TELEGRAM_TOKEN"))
        if env_token is not None:
            bot_token = env_token.strip()
        env_chat = os.getenv("TELEGRAM_CHAT_ID", os.getenv("BROWSER_HARNESS_TELEGRAM_CHAT_ID"))
        if env_chat is not None:
            chat_id = env_chat.strip()
        env_level = os.getenv("TELEGRAM_LEVEL", os.getenv("BROWSER_HARNESS_TELEGRAM_LEVEL"))
        if env_level is not None:
            level = env_level.strip().lower()

        settings = cls(
            level=level if level in TELEGRAM_LEVELS else "off",
            bot_token=bot_token,
            chat_id=chat_id,
            bot_username=bot_username,
            owner_label=owner_label,
        )
        return settings

    def validate(self) -> None:
        if self.level not in TELEGRAM_LEVELS:
            raise ValueError("Telegram level must be 'off', 'status', 'hitl', or 'tasks'.")
        token = self.bot_token.strip()
        if self.level != "off" and not token:
            raise ValueError("Enter the Bot Token from @BotFather to enable Telegram Level 1, 2, or 3.")
        if token and (":" not in token or any(ch.isspace() for ch in token) or token.startswith("http")):
            raise ValueError("Bot Token must follow the '<bot_id>:<secret>' format from @BotFather.")
        chat = str(self.chat_id).strip()
        if chat and not chat.lstrip("-").isdigit():
            raise ValueError("Authorized Chat ID must be a numeric Telegram chat ID (or leave blank to auto-detect).")

    def save(self, path: Path = TELEGRAM_SETTINGS_PATH) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @property
    def is_active(self) -> bool:
        return self.level != "off" and bool(self.bot_token.strip())

    @property
    def is_paired(self) -> bool:
        return self.is_active and bool(str(self.chat_id).strip())

    @property
    def can_push(self) -> bool:
        return self.is_paired and self.level in ("status", "hitl", "tasks")

    @property
    def can_hitl(self) -> bool:
        return self.is_paired and self.level in ("hitl", "tasks")

    @property
    def can_accept_tasks(self) -> bool:
        return self.is_paired and self.level == "tasks"

    @property
    def needs_polling(self) -> bool:
        if not self.bot_token.strip():
            return False
        # Always poll when waiting to auto-pair chat_id, or when Level 1/2/3 is active.
        return not str(self.chat_id).strip() or self.level in ("status", "hitl", "tasks")

    def masked_token(self) -> str:
        token = self.bot_token.strip()
        if not token:
            return "(not set)"
        if len(token) <= 12:
            return "***"
        return f"{token[:6]}...{token[-4:]}"

    def summary(self) -> str:
        if not self.is_active:
            return "tg off"
        tag = TELEGRAM_LEVEL_SHORT.get(self.level, "off")
        if not str(self.chat_id).strip():
            return f"tg {tag} (send /start)"
        target = self.bot_username or f"chat:{self.chat_id}"
        return f"tg {tag} ({target})"


class TelegramChannel:
    """Direct Push + Pull Telegram bridge for the local harness."""

    def __init__(self, settings: TelegramSettings | None = None,
                 request_fn: Callable[[str, str, dict[str, Any] | None], dict[str, Any]] | None = None) -> None:
        self.settings = settings or TelegramSettings()
        self._request_fn = request_fn or default_telegram_request
        self._poll_lock = Lock()
        self.last_update_id: int = 0
        self._status_message_id: int | None = None
        self._last_plan_text: str = ""
        self._active_task_prompt: str = ""
        self._active_hitl: dict[str, Any] | None = None

    def _api(self, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request_fn(self.settings.bot_token, method, payload or {})

    # ---------- pairing & diagnostics ----------
    def verify_and_pair(self, candidate: TelegramSettings | None = None) -> tuple[TelegramSettings, str]:
        """Verifies the bot token via `getMe`, auto-discovers `chat_id` from `/start` via `getUpdates`,
        and sends a confirmation message when paired."""
        cfg = candidate or self.settings
        cfg.validate()
        if not cfg.bot_token.strip():
            self.settings = cfg
            return cfg, "Telegram channel is Off."

        with self._poll_lock:
            me_resp = self._request_fn(cfg.bot_token, "getMe", {})
            bot_info = me_resp.get("result") or {}
            username = bot_info.get("username", "")
            bot_handle = f"@{username}" if username else (bot_info.get("first_name") or "bot")

            updates: list[dict[str, Any]] = []
            get_updates_payload: dict[str, Any] = {"timeout": 0, "limit": 50}
            if self.last_update_id > 0:
                get_updates_payload["offset"] = self.last_update_id + 1
            for attempt in range(2):
                try:
                    updates_resp = self._request_fn(cfg.bot_token, "getUpdates", get_updates_payload)
                    updates = updates_resp.get("result") or []
                    break
                except RuntimeError as error:
                    if "(409)" in str(error) and attempt == 0:
                        time.sleep(0.35)
                        continue
                    if "(409)" in str(error) and str(cfg.chat_id).strip():
                        updates = []
                        break
                    raise

            discovered_chat = str(cfg.chat_id).strip()
            discovered_owner = cfg.owner_label.strip()
            for update in updates:
                uid = int(update.get("update_id", 0))
                if uid > self.last_update_id:
                    self.last_update_id = uid
                msg = update.get("message") or update.get("edited_message") or (update.get("my_chat_member") or {})
                chat = msg.get("chat") or {}
                text = str(msg.get("text") or "").strip()
                chat_id = str(chat.get("id") or "").strip()
                if not chat_id:
                    continue
                if not discovered_chat or text.startswith("/start") or discovered_chat == chat_id:
                    discovered_chat = chat_id
                    sender = msg.get("from") or chat
                    uname = sender.get("username")
                    fname = " ".join(part for part in (sender.get("first_name"), sender.get("last_name")) if part).strip()
                    discovered_owner = f"@{uname}" if uname else (fname or f"chat:{chat_id}")

            if updates and self.last_update_id > 0:
                try:
                    self._request_fn(
                        cfg.bot_token,
                        "getUpdates",
                        {"offset": self.last_update_id + 1, "timeout": 0, "limit": 1},
                    )
                except RuntimeError:
                    pass

        if discovered_chat and not discovered_owner:
            try:
                chat_resp = self._request_fn(cfg.bot_token, "getChat", {"chat_id": int(discovered_chat)})
                chat_info = chat_resp.get("result") or {}
                uname = chat_info.get("username")
                fname = " ".join(part for part in (chat_info.get("first_name"), chat_info.get("last_name")) if part).strip()
                discovered_owner = f"@{uname}" if uname else (fname or f"chat:{discovered_chat}")
            except (RuntimeError, ValueError):
                discovered_owner = f"chat:{discovered_chat}"

        effective_level = "status" if cfg.level == "off" else cfg.level
        updated = TelegramSettings(
            level=effective_level,
            bot_token=cfg.bot_token.strip(),
            chat_id=discovered_chat,
            bot_username=bot_handle,
            owner_label=discovered_owner,
        )
        self.settings = updated

        if not updated.chat_id:
            link = f"https://t.me/{username}" if username else "your bot in Telegram"
            return (
                updated,
                f"Bot {bot_handle} verified! No chat paired yet: open {link} and send /start "
                "(the harness will bind your chat_id automatically).",
            )

        level_label = TELEGRAM_LEVEL_LABELS.get(updated.level, updated.level)
        if updated.level != "off":
            self.send_message(
                f"✅ Connected to Communication Harness\n"
                f"• Bot: {bot_handle}\n"
                f"• Active mode: {level_label}\n"
                f"• Policy: Single-Active-Run (local TUI commands stay on TUI)\n\n"
                f"{level_description(updated.level)}"
            )
        return (
            updated,
            f"Telegram ready: {bot_handle} paired with {updated.owner_label or updated.chat_id} "
            f"({level_label}).",
        )

    def send_message(self, text: str, reply_markup: dict[str, Any] | None = None,
                     chat_id: str | None = None) -> int | None:
        target = str(chat_id or self.settings.chat_id).strip()
        if not self.settings.bot_token.strip() or not target:
            return None
        resolved_chat: int | str = int(target) if target.lstrip("-").isdigit() else target
        payload: dict[str, Any] = {
            "chat_id": resolved_chat,
            "text": markdown_to_telegram_html(text),
            "parse_mode": "HTML",
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        try:
            response = self._api("sendMessage", payload)
        except RuntimeError as error:
            if "(400)" not in str(error) and "parse entities" not in str(error).lower():
                raise
            fallback: dict[str, Any] = {"chat_id": resolved_chat, "text": text[:4000]}
            if reply_markup is not None:
                fallback["reply_markup"] = reply_markup
            response = self._api("sendMessage", fallback)
        result = response.get("result") or {}
        msg_id = result.get("message_id")
        return int(msg_id) if isinstance(msg_id, int) else None

    def edit_message(self, message_id: int, text: str,
                     reply_markup: dict[str, Any] | None = None) -> bool:
        target = str(self.settings.chat_id).strip()
        if not self.settings.bot_token.strip() or not target:
            return False
        resolved_chat: int | str = int(target) if target.lstrip("-").isdigit() else target
        payload: dict[str, Any] = {
            "chat_id": resolved_chat,
            "message_id": message_id,
            "text": markdown_to_telegram_html(text),
            "parse_mode": "HTML",
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        try:
            self._api("editMessageText", payload)
            return True
        except RuntimeError as error:
            if "(400)" in str(error) or "parse entities" in str(error).lower():
                fallback: dict[str, Any] = {
                    "chat_id": resolved_chat,
                    "message_id": message_id,
                    "text": text[:4000],
                }
                if reply_markup is not None:
                    fallback["reply_markup"] = reply_markup
                try:
                    self._api("editMessageText", fallback)
                    return True
                except RuntimeError:
                    return False
            return False

    # ---------- Level 1: Task Update (Push) ----------
    def notify_task_started(self, prompt: str, source: str = "TUI") -> int | None:
        self._active_task_prompt = prompt.strip()
        self._last_plan_text = ""
        self._status_message_id = None
        if not self.settings.can_push:
            return None
        text = (
            f"**🚀 Task started ({source})**\n"
            f"**Request:** {self._active_task_prompt[:500]}\n\n"
            f"**Status:** Running..."
        )
        try:
            self._status_message_id = self.send_message(text)
        except RuntimeError:
            self._status_message_id = None
        return self._status_message_id

    def notify_plan_updated(self, plan_text: str) -> bool:
        cleaned = plan_text.strip()
        if not cleaned or cleaned == self._last_plan_text:
            return False
        self._last_plan_text = cleaned
        if not self.settings.can_push:
            return False
        header = (
            f"**🧭 Task in progress**\n**Request:** {self._active_task_prompt[:300]}\n\n"
            if self._active_task_prompt else ""
        )
        body = f"{header}{cleaned}"
        if self._status_message_id is not None:
            if self.edit_message(self._status_message_id, body):
                return True
        try:
            self._status_message_id = self.send_message(body)
            return self._status_message_id is not None
        except RuntimeError:
            return False

    def notify_task_finished(self, status_message: str, answer: str = "", failed: bool = False) -> int | None:
        self._status_message_id = None
        if not self.settings.can_push:
            return None
        icon = (
            "❌" if failed
            else "⏹" if status_message == "Stopped."
            else "⏸" if "Waiting" in status_message
            else "✅"
        )
        parts = [f"**{icon} Task finished: {status_message}**"]
        if self._active_task_prompt:
            parts.append(f"**Request:** {self._active_task_prompt[:300]}")
        if self._last_plan_text:
            parts.append(f"\n{self._last_plan_text[:1000]}")
        if answer.strip():
            parts.append(f"\n**Result:**\n{answer.strip()[:2200]}")
        try:
            return self.send_message("\n".join(parts))
        except RuntimeError:
            return None

    # ---------- Level 2: HITL (Push + Pull gates) ----------
    def send_approval_request(self, action: str) -> int | None:
        if not self.settings.can_hitl:
            return None
        keyboard = {
            "inline_keyboard": [[
                {"text": "✅ Approve", "callback_data": "hitl:approve"},
                {"text": "❌ Deny", "callback_data": "hitl:deny"},
            ]]
        }
        text = (
            "⚠️ Approval required\n\n"
            f"Action: {action[:800]}\n\n"
            "The safety check judged this browser action hard to undo. "
            "Tap a button below (or reply 'y' to approve / 'n' to deny):"
        )
        try:
            msg_id = self.send_message(text, reply_markup=keyboard)
        except RuntimeError:
            return None
        if msg_id is not None:
            self._active_hitl = {
                "kind": "approval",
                "message_id": msg_id,
                "text": text,
                "action": action,
            }
        return msg_id

    def send_question_request(self, question: str, options: list[str]) -> int | None:
        if not self.settings.can_hitl:
            return None
        rows = [
            [{"text": f"{idx}. {opt[:48]}", "callback_data": f"hitl:opt:{idx}"}]
            for idx, opt in enumerate(options, 1)
        ]
        rows.append([{"text": "💬 Answer later in chat", "callback_data": "hitl:later"}])
        lines = ["❓ The agent needs your answer\n", question[:800]]
        if options:
            lines.append("\nOptions:")
            for idx, opt in enumerate(options, 1):
                lines.append(f"  {idx}. {opt}")
        lines.append("\nTap an option below or reply with any custom answer:")
        text = "\n".join(lines)
        try:
            msg_id = self.send_message(text, reply_markup={"inline_keyboard": rows})
        except RuntimeError:
            return None
        if msg_id is not None:
            self._active_hitl = {
                "kind": "question",
                "message_id": msg_id,
                "text": text,
                "question": question,
                "options": list(options),
            }
        return msg_id

    def close_active_hitl(self, resolution_note: str) -> None:
        active = self._active_hitl
        self._active_hitl = None
        if not active or not self.settings.can_hitl:
            return
        msg_id = active.get("message_id")
        base_text = active.get("text", "")
        if isinstance(msg_id, int):
            self.edit_message(msg_id, f"{base_text}\n\n{resolution_note}", reply_markup={"inline_keyboard": []})

    # ---------- Level 2 & 3: Inbound Pull (`getUpdates`) ----------
    def poll_events(self, busy: bool = False, teaching: bool = False,
                    plan_text: str = "") -> list[dict[str, Any]]:
        """Polls `getUpdates` once (non-blocking) and enforces level + surface guardrails."""
        if not self.settings.needs_polling:
            return []
        if not self._poll_lock.acquire(blocking=False):
            return []
        try:
            payload: dict[str, Any] = {"timeout": 0, "limit": 25}
            if self.last_update_id > 0:
                payload["offset"] = self.last_update_id + 1
            response = self._api("getUpdates", payload)
            updates = response.get("result") or []
            events_out: list[dict[str, Any]] = []
            for update in updates:
                uid = int(update.get("update_id", 0))
                if uid > self.last_update_id:
                    self.last_update_id = uid
                event = self._handle_update(update, busy=busy, teaching=teaching, plan_text=plan_text)
                if event is not None:
                    events_out.append(event)
                    if event.get("type") == "task":
                        # Single-Active-Run: once one task is accepted in this batch, subsequent tasks see busy=True.
                        busy = True
            return events_out
        finally:
            self._poll_lock.release()

    def _handle_update(self, update: dict[str, Any], busy: bool, teaching: bool,
                       plan_text: str) -> dict[str, Any] | None:
        if "callback_query" in update:
            return self._handle_callback(update["callback_query"])
        if "message" in update:
            return self._handle_message(update["message"], busy=busy, teaching=teaching, plan_text=plan_text)
        return None

    def _handle_callback(self, callback: dict[str, Any]) -> dict[str, Any] | None:
        cb_id = callback.get("id")
        msg = callback.get("message") or {}
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id") or "").strip()
        data = str(callback.get("data") or "").strip()

        if cb_id:
            try:
                self._api("answerCallbackQuery", {"callback_query_id": cb_id})
            except RuntimeError:
                pass

        if not self.settings.chat_id or chat_id != str(self.settings.chat_id).strip():
            return None
        if not self.settings.can_hitl or not self._active_hitl:
            return None

        kind = self._active_hitl.get("kind")
        if kind == "approval" and data in ("hitl:approve", "hitl:deny"):
            approved = data == "hitl:approve"
            note = "✓ Approved via Telegram." if approved else "✗ Denied via Telegram."
            self.close_active_hitl(note)
            return {"type": "hitl_approval", "approved": approved}

        if kind == "question":
            options: list[str] = self._active_hitl.get("options") or []
            if data.startswith("hitl:opt:"):
                try:
                    idx = int(data.removeprefix("hitl:opt:"))
                except ValueError:
                    return None
                if 1 <= idx <= len(options):
                    answer = options[idx - 1]
                    self.close_active_hitl(f"→ Answered via Telegram: {answer}")
                    return {"type": "hitl_question", "answer": answer}
            elif data == "hitl:later":
                self.close_active_hitl("→ Deferred to chat via Telegram.")
                return {"type": "hitl_question", "answer": None}
        return None

    def _handle_message(self, message: dict[str, Any], busy: bool, teaching: bool,
                        plan_text: str) -> dict[str, Any] | None:
        chat = message.get("chat") or {}
        chat_id = str(chat.get("id") or "").strip()
        text = str(message.get("text") or "").strip()
        if not chat_id or not text:
            return None

        sender = message.get("from") or chat
        uname = sender.get("username")
        fname = " ".join(part for part in (sender.get("first_name"), sender.get("last_name")) if part).strip()
        sender_label = f"@{uname}" if uname else (fname or f"chat:{chat_id}")

        # Auto-pair on any first message in a private chat if no chat_id is bound yet.
        if not str(self.settings.chat_id).strip():
            self.settings.chat_id = chat_id
            self.settings.owner_label = sender_label
            if self.settings.level == "off":
                self.settings.level = "status"
            level_label = TELEGRAM_LEVEL_LABELS.get(self.settings.level, self.settings.level)
            try:
                self.send_message(
                    f"✅ Paired with Communication Harness!\n"
                    f"• Owner: {sender_label} (chat_id: {chat_id})\n"
                    f"• Active mode: {level_label}\n\n"
                    f"{level_description(self.settings.level)}"
                )
            except RuntimeError:
                pass
            return {"type": "paired", "chat_id": chat_id, "owner_label": sender_label}

        # Reject messages from any chat other than the paired owner.
        if chat_id != str(self.settings.chat_id).strip():
            return None

        head = text.split()[0].lower()
        # Strip @bot_username suffix if present (e.g. /status@my_bot)
        command = head.split("@")[0]

        if command == "/start":
            level_label = TELEGRAM_LEVEL_LABELS.get(self.settings.level, self.settings.level)
            self.send_message(
                f"👋 Communication Harness is connected.\n"
                f"• Active mode: {level_label}\n"
                f"• {level_description(self.settings.level)}\n"
                f"• Commands on Telegram: /status, /stop, /help"
            )
            return None

        if command == "/help":
            level_label = TELEGRAM_LEVEL_LABELS.get(self.settings.level, self.settings.level)
            self.send_message(
                f"🧭 Communication Harness — Telegram Help\n"
                f"Active mode: {level_label}\n\n"
                f"Capability levels (configured in TUI via /telegram):\n"
                f"• Level 1 (Task Update): Push status & final outcome only.\n"
                f"• Level 2 (HITL): Approve browser actions & answer agent questions.\n"
                f"• Level 3 (Accept Tasks): Send natural-language tasks when idle.\n\n"
                f"Telegram commands:\n"
                f"• /status — Show harness state and current plan\n"
                f"• /stop — Stop the currently running task (Level 2 & 3)\n"
                f"• /help — Show this guide\n\n"
                f"Note: Interactive commands (/teach, /login, /models, /telegram, /clear) are TUI-only."
            )
            return None

        if command == "/status":
            level_label = TELEGRAM_LEVEL_LABELS.get(self.settings.level, self.settings.level)
            state = ("Waiting for HITL response" if self._active_hitl
                     else "Recording local /teach demonstration" if teaching
                     else f"Busy ({self._active_task_prompt[:120]})" if busy
                     else "Idle (ready)")
            plan_block = f"\n\n{plan_text.strip()}" if plan_text.strip() else ""
            self.send_message(f"📊 Harness Status\n• Mode: {level_label}\n• State: {state}{plan_block}")
            return None

        if command == "/stop":
            if not self.settings.can_hitl:
                self.send_message(
                    "🔒 Level 1 (Task Update) is read-only. Switch to Level 2 (HITL) or Level 3 (Accept Tasks) "
                    "in the TUI to stop tasks from Telegram."
                )
                return None
            if busy or self._active_hitl:
                self.close_active_hitl("⏹ Stopped via Telegram.")
                self.send_message("⏹ Stop requested. Waiting for the current step to finish.")
                return {"type": "stop", "sender": sender_label}
            self.send_message("ℹ️ No task is currently running.")
            return None

        # If a HITL gate is currently open and Level 2/3 is active, route the reply to the open gate.
        if self._active_hitl and self.settings.can_hitl:
            kind = self._active_hitl.get("kind")
            if kind == "approval":
                lowered = text.lower()
                if lowered in ("y", "yes", "approve", "approved", "ok", "sim"):
                    self.close_active_hitl("✓ Approved via Telegram.")
                    return {"type": "hitl_approval", "approved": True}
                if lowered in ("n", "no", "deny", "denied", "cancel", "nao", "não"):
                    self.close_active_hitl("✗ Denied via Telegram.")
                    return {"type": "hitl_approval", "approved": False}
                self.send_message("⚠️ Waiting for approval: tap ✅ Approve / ❌ Deny above, or reply 'y' or 'n'.")
                return None
            if kind == "question":
                options: list[str] = self._active_hitl.get("options") or []
                answer = text
                if text.isdigit() and 1 <= int(text) <= len(options):
                    answer = options[int(text) - 1]
                self.close_active_hitl(f"→ Answered via Telegram: {answer}")
                return {"type": "hitl_question", "answer": answer}

        # No HITL gate is open: enforce Level permissions and surface guardrails.
        if self.settings.level == "status":
            self.send_message(
                "🔒 Level 1 (Task Update) is active: Telegram receives status updates only. "
                "Switch to Level 3 (Accept Tasks) in the TUI to send tasks."
            )
            return {"type": "level_rejected", "level": "status", "prompt": text}

        if self.settings.level == "hitl":
            self.send_message(
                "🔒 Level 2 (HITL) is active: Telegram can approve actions and answer agent questions when "
                "prompted, but cannot start new tasks. Switch to Level 3 (Accept Tasks) in the TUI to send tasks."
            )
            return {"type": "level_rejected", "level": "hitl", "prompt": text}

        if not self.settings.can_accept_tasks:
            return None

        # Level 3 (Accept Tasks) guardrails:
        if command.startswith("/"):
            if command in TUI_ONLY_COMMANDS:
                self.send_message(
                    f"🖥️ Local TUI command only: '{command}' requires interactive terminal/browser access "
                    f"and cannot be executed from Telegram."
                )
            else:
                self.send_message(
                    f"❓ Unknown Telegram command '{command}'. Allowed commands: /status, /stop, /help."
                )
            return {"type": "blocked_command", "command": command, "sender": sender_label}

        if teaching:
            self.send_message(
                "🎥 Local learning mode (/teach) is currently recording in the TUI. "
                "Finish or cancel /teach in the terminal before sending tasks from Telegram."
            )
            return {"type": "busy_rejected", "prompt": text, "reason": "teaching", "sender": sender_label}

        if busy:
            self.send_message(
                "⏳ Busy: the harness is currently running a task. "
                "Single-Active-Run policy is active (no queueing) — wait for the completion notice "
                "or send /stop to cancel the current task."
            )
            return {"type": "busy_rejected", "prompt": text, "reason": "busy", "sender": sender_label}

        return {"type": "task", "prompt": text, "sender": sender_label}
