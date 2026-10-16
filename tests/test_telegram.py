"""Unit and TUI integration tests for the 3-level local Telegram communication channel."""

from __future__ import annotations

import asyncio
import functools
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

os.environ["BROWSER_HARNESS_PLANNER"] = "0"

from textual.widgets import Button, Input, Select, Static

from foundry import ToolCall, Turn
from memory import NullMemory
from session_log import NullLog
from telegram_channel import TelegramChannel, TelegramSettings
from tui import ApprovalPanel, BrowserApp, ComposerInput, QuestionPanel, TelegramSettingsPanel
import workspace


class FakeTelegramAPI:
    """In-memory fake for Telegram Bot API (`getMe`, `getUpdates`, `sendMessage`, `editMessageText`, `answerCallbackQuery`)."""

    def __init__(self, username: str = "harness_demo_bot") -> None:
        self.username = username
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.updates: list[dict[str, Any]] = []
        self.sent_messages: list[dict[str, Any]] = []
        self.edited_messages: list[dict[str, Any]] = []
        self.answered_callbacks: list[str] = []
        self._next_msg_id = 100
        self._next_update_id = 1

    def push_user_message(self, text: str, chat_id: int | str = 777000,
                          username: str = "contoso_user") -> int:
        uid = self._next_update_id
        self._next_update_id += 1
        self.updates.append({
            "update_id": uid,
            "message": {
                "message_id": uid * 10,
                "from": {"id": int(chat_id), "username": username, "first_name": "Contoso"},
                "chat": {"id": int(chat_id), "type": "private", "username": username},
                "text": text,
            },
        })
        return uid

    def push_callback(self, data: str, chat_id: int | str = 777000,
                      message_id: int = 100, callback_id: str = "cb-1") -> int:
        uid = self._next_update_id
        self._next_update_id += 1
        self.updates.append({
            "update_id": uid,
            "callback_query": {
                "id": callback_id,
                "from": {"id": int(chat_id), "username": "contoso_user"},
                "message": {"message_id": message_id, "chat": {"id": int(chat_id)}},
                "data": data,
            },
        })
        return uid

    def __call__(self, token: str, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = dict(payload or {})
        self.calls.append((method, body))
        if method == "getMe":
            return {"ok": True, "result": {"id": 999, "is_bot": True, "username": self.username}}
        if method == "getChat":
            return {"ok": True, "result": {"id": body.get("chat_id", 888111), "username": "contoso_user"}}
        if method == "getUpdates":
            offset = int(body.get("offset", 0))
            visible = [u for u in self.updates if int(u["update_id"]) >= offset]
            return {"ok": True, "result": visible}
        if method == "sendMessage":
            msg_id = self._next_msg_id
            self._next_msg_id += 1
            record = {"message_id": msg_id, **body}
            self.sent_messages.append(record)
            return {"ok": True, "result": {"message_id": msg_id}}
        if method == "editMessageText":
            self.edited_messages.append(body)
            return {"ok": True, "result": True}
        if method == "answerCallbackQuery":
            self.answered_callbacks.append(str(body.get("callback_query_id", "")))
            return {"ok": True, "result": True}
        raise AssertionError(f"Unexpected Telegram method: {method}")


class FakeModel:
    def complete(self, messages, tools, emit, cancel, tool_choice="auto"):
        emit("answer", "Task completed.")
        return Turn(content="Task completed.", input_tokens=12, output_tokens=4)

    def close(self):
        pass


class PlanModel:
    def __init__(self) -> None:
        self.turns = [
            Turn(tool_calls=[ToolCall("t1", "tasks", '{"action":"plan","items":["open example.com","summarize"]}')]),
            Turn(tool_calls=[ToolCall("t2", "tasks", '{"action":"done","index":1,"note":"opened"}')]),
            Turn(tool_calls=[ToolCall("t3", "tasks", '{"action":"done","index":2,"note":"summarized"}')]),
            Turn(content="Summary ready."),
        ]

    def complete(self, messages, tools, emit, cancel, tool_choice="auto"):
        return self.turns.pop(0)

    def close(self):
        pass


class TelegramChannelTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.folder = Path(self._temp.name)
        patcher = mock.patch("tui.Workspace", functools.partial(workspace.Workspace, self.folder))
        settings_patcher = mock.patch("tui.SETTINGS_PATH", self.folder / "settings.json")
        tg_patcher = mock.patch("tui.TELEGRAM_SETTINGS_PATH", self.folder / "telegram.json")
        patcher.start()
        settings_patcher.start()
        tg_patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(settings_patcher.stop)
        self.addCleanup(tg_patcher.stop)
        self.addCleanup(self._temp.cleanup)

    def test_settings_validation_and_round_trip(self) -> None:
        path = self.folder / "telegram.json"
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = TelegramSettings(level="off")
            cfg.validate()
            self.assertEqual(cfg.summary(), "tg off")

            with self.assertRaisesRegex(ValueError, "Bot Token"):
                TelegramSettings(level="status", bot_token="").validate()
            with self.assertRaisesRegex(ValueError, "format"):
                TelegramSettings(level="status", bot_token="invalid_token").validate()
            with self.assertRaisesRegex(ValueError, "numeric"):
                TelegramSettings(level="status", bot_token="123456:ABCDEF", chat_id="not-a-number").validate()

            valid = TelegramSettings(
                level="tasks",
                bot_token="123456:ABCDEF_secret",
                chat_id="777000",
                bot_username="@harness_demo_bot",
                owner_label="@contoso_user",
            )
            valid.save(path)
            loaded = TelegramSettings.load(path)
            self.assertEqual(loaded, valid)
            self.assertEqual(loaded.summary(), "tg L3:tasks (@harness_demo_bot)")

    def test_verify_and_pair_discovers_chat_id_from_start_message(self) -> None:
        api = FakeTelegramAPI(username="contoso_harness_bot")
        api.push_user_message("/start", chat_id=888111, username="alex_demo")
        channel = TelegramChannel(
            TelegramSettings(level="status", bot_token="123456:ABCDEF_token"),
            request_fn=api,
        )
        updated, message = channel.verify_and_pair()
        self.assertEqual(updated.bot_username, "@contoso_harness_bot")
        self.assertEqual(updated.chat_id, "888111")
        self.assertEqual(updated.owner_label, "@alex_demo")
        self.assertIn("paired with @alex_demo", message)
        self.assertEqual(len(api.sent_messages), 1)
        self.assertIn("Connected to Communication Harness", api.sent_messages[0]["text"])
        # Historical /start update was consumed and must not re-trigger on next poll.
        self.assertEqual(channel.poll_events(), [])

    def test_level1_pushes_status_and_edits_plan_but_rejects_inbound_tasks(self) -> None:
        api = FakeTelegramAPI()
        cfg = TelegramSettings(
            level="status",
            bot_token="123456:ABCDEF_token",
            chat_id="777000",
            bot_username="@harness_demo_bot",
        )
        channel = TelegramChannel(cfg, request_fn=api)

        msg_id = channel.notify_task_started("Summarize Contoso news", source="TUI")
        self.assertIsNotNone(msg_id)
        self.assertEqual(len(api.sent_messages), 1)
        self.assertIn("Task started (TUI)", api.sent_messages[0]["text"])

        self.assertTrue(channel.notify_plan_updated("Plan - 0/2 done\n▶ 1. open example.com\n⏳ 2. summarize"))
        self.assertTrue(channel.notify_plan_updated("Plan - 1/2 done\n✅ 1. open example.com\n▶ 2. summarize"))
        self.assertEqual(len(api.edited_messages), 2)
        self.assertEqual(api.edited_messages[0]["message_id"], msg_id)

        channel.notify_task_finished("Done.", answer="Two articles summarized.", failed=False)
        self.assertEqual(len(api.sent_messages), 2)
        self.assertIn("Two articles summarized.", api.sent_messages[-1]["text"])

    def test_level2_hitl_resolves_approval_and_question_via_telegram_or_tui(self) -> None:
        api = FakeTelegramAPI()
        cfg = TelegramSettings(
            level="hitl",
            bot_token="123456:ABCDEF_token",
            chat_id="777000",
            bot_username="@harness_demo_bot",
        )
        channel = TelegramChannel(cfg, request_fn=api)

        # 1. Approval resolved via Telegram inline button callback
        msg_id = channel.send_approval_request("Send message to Contoso team")
        self.assertIsNotNone(msg_id)
        api.push_callback("hitl:approve", chat_id=777000, message_id=msg_id or 100, callback_id="cb-approve")
        events = channel.poll_events(busy=True)
        self.assertEqual(events, [{"type": "hitl_approval", "approved": True}])
        self.assertIn("cb-approve", api.answered_callbacks)
        self.assertIn("Approved via Telegram", api.edited_messages[-1]["text"])
        self.assertEqual(api.edited_messages[-1]["reply_markup"], {"inline_keyboard": []})

        # 2. Question resolved via Telegram option callback
        q_id = channel.send_question_request("Which Contoso channel?", ["General", "Engineering"])
        self.assertIsNotNone(q_id)
        api.push_callback("hitl:opt:2", chat_id=777000, message_id=q_id or 101, callback_id="cb-opt")
        events = channel.poll_events(busy=True)
        self.assertEqual(events, [{"type": "hitl_question", "answer": "Engineering"}])

        # 3. Attempting to send a new task in Level 2 is rejected
        api.push_user_message("open playwright.dev", chat_id=777000)
        events = channel.poll_events(busy=False)
        self.assertEqual(events[0]["type"], "level_rejected")
        self.assertIn("Level 2 (HITL) is active", api.sent_messages[-1]["text"])

    def test_level3_accepts_tasks_blocks_tui_only_commands_and_enforces_busy_reject(self) -> None:
        api = FakeTelegramAPI()
        cfg = TelegramSettings(
            level="tasks",
            bot_token="123456:ABCDEF_token",
            chat_id="777000",
            bot_username="@harness_demo_bot",
            owner_label="@contoso_user",
        )
        channel = TelegramChannel(cfg, request_fn=api)

        # Unauthorized chat is ignored completely
        api.push_user_message("open example.com", chat_id=999999, username="stranger")
        self.assertEqual(channel.poll_events(busy=False), [])

        # TUI-only commands are blocked on Telegram
        for cmd in ("/teach add contact", "/login linkedin.com", "/models", "/telegram", "/clear"):
            api.push_user_message(cmd, chat_id=777000)
            events = channel.poll_events(busy=False)
            self.assertEqual(events[0]["type"], "blocked_command")
            self.assertIn("Local TUI command only", api.sent_messages[-1]["text"])

        # Busy-reject when a task is already running (Single-Active-Run, no queue)
        api.push_user_message("summarize news", chat_id=777000)
        events = channel.poll_events(busy=True)
        self.assertEqual(events, [{
            "type": "busy_rejected",
            "prompt": "summarize news",
            "reason": "busy",
            "sender": "@contoso_user",
        }])
        self.assertIn("Single-Active-Run", api.sent_messages[-1]["text"])

        # Busy-reject when local /teach mode is recording in the TUI
        api.push_user_message("summarize news", chat_id=777000)
        events = channel.poll_events(busy=False, teaching=True)
        self.assertEqual(events[0]["reason"], "teaching")
        self.assertIn("Local learning mode (/teach)", api.sent_messages[-1]["text"])

        # When idle in Level 3, natural-language task is accepted
        api.push_user_message("open example.com and read the title", chat_id=777000)
        events = channel.poll_events(busy=False, teaching=False)
        self.assertEqual(events, [{
            "type": "task",
            "prompt": "open example.com and read the title",
            "sender": "@contoso_user",
        }])

    def test_tui_telegram_modal_and_end_to_end_level3_task_execution(self) -> None:
        api = FakeTelegramAPI(username="contoso_bot")
        api.push_user_message("/start", chat_id=777000, username="contoso_user")
        channel = TelegramChannel(TelegramSettings(level="off"), request_fn=api)
        tg_path = self.folder / "telegram.json"

        async def scenario() -> None:
            app = BrowserApp(
                model_factory=PlanModel,
                log=NullLog(),
                memory=NullMemory(),
                telegram_channel=channel,
                telegram_settings_path=tg_path,
            )
            async with app.run_test(size=(120, 42)) as pilot:
                # Open Telegram modal via button, select Level 3, enter token, click Save and check
                app._show_telegram_settings()
                await pilot.pause()
                panel = app.query_one(TelegramSettingsPanel)
                panel.on_button_pressed(Button.Pressed(panel.query_one("#tg-level-tasks", Button)))
                await pilot.pause()
                self.assertEqual(panel.query_one("#telegram-level", Select).value, "tasks")
                panel.query_one("#telegram-token", Input).value = "123456:ABCDEF_token"
                panel.on_button_pressed(Button.Pressed(panel.query_one("#telegram-check", Button)))
                for _ in range(40):
                    await pilot.pause(0.05)
                    if not app.busy:
                        break
                saved = json.loads(tg_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["level"], "tasks")
                self.assertEqual(saved["chat_id"], "777000")
                self.assertEqual(saved["bot_username"], "@contoso_bot")
                self.assertIn("tg L3:tasks (@contoso_bot)", str(app.query_one("#settings", Static).content))

                # Now send a task from Telegram in Level 3 and verify it runs in the TUI and pushes back to Telegram
                api.push_user_message("open example.com and summarize", chat_id=777000, username="contoso_user")
                app._process_telegram_updates()
                for _ in range(50):
                    await pilot.pause(0.05)
                    if not app.busy:
                        break
                texts = "\n".join(getattr(w.content, "markup", str(w.content)) for w in app.query(Static))
                self.assertIn("You [Telegram · @contoso_user]", texts)
                self.assertIn("open example.com and summarize", texts)
                self.assertTrue(any("Summary ready." in m["text"] for m in api.sent_messages))

        asyncio.run(scenario())

    def test_tui_and_telegram_hitl_race_free_resolution(self) -> None:
        api = FakeTelegramAPI()
        cfg = TelegramSettings(
            level="hitl",
            bot_token="123456:ABCDEF_token",
            chat_id="777000",
            bot_username="@harness_demo_bot",
        )
        channel = TelegramChannel(cfg, request_fn=api)

        async def scenario() -> tuple[list[bool], list[str | None]]:
            app = BrowserApp(
                model_factory=FakeModel,
                log=NullLog(),
                memory=NullMemory(),
                telegram_channel=channel,
                telegram_settings_path=self.folder / "telegram.json",
            )
            approvals: list[bool] = []
            answers: list[str | None] = []
            async with app.run_test(size=(110, 40)) as pilot:
                # Mount ApprovalPanel and resolve from Telegram
                channel.send_approval_request("Submit order")
                app._show_panel_inline(ApprovalPanel("Submit order", approvals.append), "wait")
                await pilot.pause()
                api.push_user_message("y", chat_id=777000)
                app._process_telegram_updates()
                await pilot.pause()
                panel = list(app.query(ApprovalPanel))[-1]
                self.assertTrue(panel.answered)

                # Mount QuestionPanel and resolve from Telegram
                channel.send_question_request("Which format?", ["Markdown", "JSON"])
                app._show_panel_inline(QuestionPanel("Which format?", ["Markdown", "JSON"], answers.append), "q")
                await pilot.pause()
                api.push_user_message("2", chat_id=777000)
                app._process_telegram_updates()
                await pilot.pause()
                qpanel = list(app.query(QuestionPanel))[-1]
                self.assertTrue(qpanel.answered)
            return approvals, answers

        self.assertEqual(asyncio.run(scenario()), ([True], ["JSON"]))

    def test_worker_thread_approve_and_ask_user_bridge_with_telegram(self) -> None:
        from textual import work

        api = FakeTelegramAPI()
        cfg = TelegramSettings(
            level="hitl",
            bot_token="123456:ABCDEF_token",
            chat_id="777000",
            bot_username="@harness_demo_bot",
        )
        channel = TelegramChannel(cfg, request_fn=api)
        outcomes: list[Any] = []

        async def scenario() -> None:
            app = BrowserApp(
                model_factory=FakeModel,
                log=NullLog(),
                memory=NullMemory(),
                telegram_channel=channel,
                telegram_settings_path=self.folder / "telegram.json",
            )

            def run_gates() -> None:
                outcomes.append(app._approve("Delete draft item"))
                outcomes.append(app._ask_user("Pick a destination", ["Archive", "Trash"]))
                outcomes.append(app._approve("Confirm final step"))

            async with app.run_test(size=(110, 40)) as pilot:
                app.run_worker(run_gates, thread=True)
                # Wait for first ApprovalPanel to mount, then approve via Telegram callback
                for _ in range(30):
                    await pilot.pause(0.05)
                    if channel._active_hitl and channel._active_hitl.get("kind") == "approval":
                        break
                msg_id = channel._active_hitl["message_id"]
                api.push_callback("hitl:approve", chat_id=777000, message_id=msg_id)

                # Wait for QuestionPanel to mount, then answer via Telegram text reply
                for _ in range(30):
                    await pilot.pause(0.05)
                    if channel._active_hitl and channel._active_hitl.get("kind") == "question":
                        break
                api.push_user_message("1", chat_id=777000)

                # Wait for second ApprovalPanel to mount in the TUI, then deny via TUI ('n' key)
                for _ in range(60):
                    await pilot.pause(0.05)
                    panels = list(app.query(ApprovalPanel))
                    if len(panels) == 2 and not panels[-1].answered and channel._active_hitl:
                        break
                await pilot.pause(0.05)
                await pilot.press("n")
                for _ in range(60):
                    await pilot.pause(0.05)
                    if len(outcomes) == 3:
                        break

        asyncio.run(scenario())
        self.assertEqual(outcomes, [True, "Archive", False])
        self.assertIsNone(channel._active_hitl)
        self.assertTrue(any("Denied in TUI" in m["text"] for m in api.edited_messages))

    def test_failed_plan_task_reports_failure_not_done(self) -> None:
        api = FakeTelegramAPI()
        cfg = TelegramSettings(
            level="tasks",
            bot_token="123456:ABCDEF_token",
            chat_id="777000",
            bot_username="@harness_demo_bot",
        )
        channel = TelegramChannel(cfg, request_fn=api)

        class FailedPlanModel:
            usage_name = "gpt-test"

            def __init__(self) -> None:
                self.step = 0

            def complete(self, messages, tools, on_event, cancel, tool_choice="auto"):
                self.step += 1
                if self.step == 1:
                    return Turn(tool_calls=[
                        ToolCall("c1", "tasks", json.dumps({"action": "plan", "items": ["Open CNN US"]}))
                    ])
                if self.step in (2, 3):
                    return Turn(tool_calls=[
                        ToolCall(f"c{self.step}", "tasks", json.dumps({
                            "action": "failed", "index": 1, "note": "Browser failed to start"
                        }))
                    ])
                return Turn(content="Stopped as requested.")

            def close(self) -> None:
                pass

        async def scenario() -> None:
            app = BrowserApp(
                model_factory=FailedPlanModel,
                log=NullLog(),
                memory=NullMemory(),
                telegram_channel=channel,
                telegram_settings_path=self.folder / "telegram.json",
            )
            async with app.run_test(size=(110, 40)) as pilot:
                app._start("Open CNN US", source="Telegram · @contoso_user")
                for _ in range(40):
                    await pilot.pause(0.05)
                    if not app.busy:
                        break

        asyncio.run(scenario())
        self.assertTrue(any("❌ Task finished: Task failed; see the plan." in m["text"] for m in api.sent_messages))
        self.assertFalse(any("✅ Task finished: Done." in m["text"] for m in api.sent_messages))

    def test_verify_and_pair_retries_transient_409_conflict_on_get_updates(self) -> None:
        api = FakeTelegramAPI(username="contoso_harness_bot")
        calls = {"getUpdates": 0}

        def flaky_request(token: str, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            if method == "getUpdates":
                calls["getUpdates"] += 1
                if calls["getUpdates"] == 1:
                    raise RuntimeError(
                        "Telegram API error (409) on getUpdates: Conflict: terminated by other getUpdates request"
                    )
            return api(token, method, payload)

        channel = TelegramChannel(
            TelegramSettings(level="status", bot_token="123456:ABCDEF_token", chat_id="888111"),
            request_fn=flaky_request,
        )
        updated, msg = channel.verify_and_pair()
        self.assertEqual(updated.chat_id, "888111")
        self.assertIn("Telegram ready", msg)
        self.assertGreaterEqual(calls["getUpdates"], 2)

    def test_markdown_to_telegram_html_formats_bold_bullets_and_escapes_html(self) -> None:
        api = FakeTelegramAPI()
        cfg = TelegramSettings(
            level="status",
            bot_token="123456:ABCDEF_token",
            chat_id="777000",
            bot_username="@harness_demo_bot",
        )
        channel = TelegramChannel(cfg, request_fn=api)
        channel.notify_task_finished(
            "Done.",
            answer=(
                'A manchete principal da CNN US é: **“Headline <1> & more”**\n\n'
                "- **Português (Brasil):** Texto em português.\n"
                "- **Espanhol:** Texto en español com `código`."
            ),
            failed=False,
        )
        sent = api.sent_messages[-1]
        self.assertEqual(sent.get("parse_mode"), "HTML")
        self.assertIn("<b>✅ Task finished: Done.</b>", sent["text"])
        self.assertIn("<b>“Headline &lt;1&gt; &amp; more”</b>", sent["text"])
        self.assertIn("• <b>Português (Brasil):</b> Texto em português.", sent["text"])
        self.assertIn("• <b>Espanhol:</b> Texto en español com <code>código</code>.", sent["text"])

    def test_browser_auto_closes_on_task_finish_but_stays_open_when_waiting(self) -> None:
        from threading import Event
        from agent import BrowserAgent

        class SequenceModel:
            usage_name = "gpt-test"

            def __init__(self, turns: list[Turn]) -> None:
                self.turns = list(turns)

            def complete(self, messages, tools, on_event, cancel, tool_choice="auto"):
                return self.turns.pop(0)

        closed_calls: list[str] = []
        events: list[tuple[str, str]] = []

        # 1) When a task pauses on a question (`waiting`), the browser stays OPEN so the user can answer/continue.
        waiting_model = SequenceModel([
            Turn(tool_calls=[ToolCall("t1", "tasks", json.dumps({"action": "plan", "items": ["Open CNN"]}))]),
            Turn(tool_calls=[ToolCall("b1", "browser", json.dumps({"command": "open", "args": ["https://www.cnn.com/"]}))]),
            Turn(tool_calls=[ToolCall("t2", "tasks", json.dumps({"action": "ask", "index": 1, "note": "Which section?"}))]),
            Turn(content="Which section do you want?"),
        ])
        agent_waiting = BrowserAgent(
            waiting_model,
            run_tool=lambda raw: "- Page URL: https://www.cnn.com/",
            ask_user=lambda q, o: None,
            close_browser=lambda: closed_calls.append("closed") or "Browser closed.",
        )
        agent_waiting.run("open cnn", lambda k, t: events.append((k, t)), Event())
        self.assertEqual(closed_calls, [])

        # 2) When a task finishes (`Done`), the browser closes automatically.
        done_model = SequenceModel([
            Turn(tool_calls=[ToolCall("t1", "tasks", json.dumps({"action": "plan", "items": ["Open CNN"]}))]),
            Turn(tool_calls=[ToolCall("b1", "browser", json.dumps({"command": "read", "args": ["https://www.cnn.com/"]}))]),
            Turn(content="Top headline is ready."),
        ])
        agent_done = BrowserAgent(
            done_model,
            run_tool=lambda raw: "- Page URL: https://www.cnn.com/\n### Text\nHeadline",
            close_browser=lambda: closed_calls.append("closed") or "Browser closed.",
        )
        agent_done.run("read cnn", lambda k, t: events.append((k, t)), Event())
        self.assertEqual(closed_calls, ["closed"])
        self.assertTrue(any("Browser closed automatically" in text for kind, text in events if kind == "note"))

    def test_site_scoped_memory_loads_only_requested_or_visited_site_and_updates_per_site(self) -> None:
        from threading import Event
        from agent import BrowserAgent
        from memory import Memory

        memory = Memory(self.folder / "memory")
        memory.save_long_term(
            "- CNN US: `read https://www.cnn.com/` for headlines.\n"
            "- LinkedIn add contact: `read https://www.linkedin.com/search/results/people/`.\n"
            "- WhatsApp Web: open https://web.whatsapp.com/ and `find (you)`."
        )
        # Verify per-site files were created under memory/sites/
        self.assertTrue((memory.sites_dir / "cnn.com.md").is_file())
        self.assertTrue((memory.sites_dir / "linkedin.com.md").is_file())
        self.assertTrue((memory.sites_dir / "web.whatsapp.com.md").is_file())

        # 1) Prompt mentioning CNN loads ONLY cnn.com into system_prompt (not LinkedIn or WhatsApp).
        cnn_prompt = memory.for_request("Give me the main headline from CNN")
        self.assertIn("https://www.cnn.com/", cnn_prompt)
        self.assertNotIn("linkedin.com", cnn_prompt)
        self.assertNotIn("whatsapp.com", cnn_prompt)

        # 2) Prompt not naming any site upfront does NOT bloat system_prompt; when the agent opens
        #    https://edition.cnn.com/world mid-turn, only cnn.com's memory is injected on-demand.
        class NavModel:
            usage_name = "gpt-test"

            def __init__(self) -> None:
                self.requests: list[list[dict]] = []
                self.turns = [
                    Turn(tool_calls=[ToolCall("t1", "tasks", json.dumps({"action": "plan", "items": ["Read portal"]}))]),
                    Turn(tool_calls=[ToolCall("b1", "browser", json.dumps({"command": "read", "args": ["https://edition.cnn.com/world"]}))]),
                    Turn(tool_calls=[ToolCall("b2", "browser", json.dumps({"command": "links", "args": []}))]),
                    Turn(content="Read the world section."),
                ]

            def complete(self, messages, tools, on_event, cancel, tool_choice="auto"):
                self.requests.append([dict(m) for m in messages])
                return self.turns.pop(0)

        nav_model = NavModel()
        events: list[tuple[str, str]] = []
        agent = BrowserAgent(
            nav_model,
            run_tool=lambda raw: "- Page URL: https://edition.cnn.com/world\n### Text\nWorld news",
            memory=memory,
        )
        agent.run("check the news portal", lambda k, t: events.append((k, t)), Event())
        # Initial system prompt did not load LinkedIn, WhatsApp, or CNN
        sys_msg = nav_model.requests[0][0]["content"]
        self.assertNotIn("linkedin.com", sys_msg)
        self.assertNotIn("whatsapp.com", sys_msg)
        self.assertNotIn("cnn.com", sys_msg)
        # First navigation to edition.cnn.com injected cnn.com site memory once
        first_nav_tool_msg = nav_model.requests[2][-1]["content"]
        self.assertIn("[harness] Site memory for cnn.com (from earlier sessions):", first_nav_tool_msg)
        self.assertIn("read https://www.cnn.com/", first_nav_tool_msg)
        self.assertNotIn("linkedin.com", first_nav_tool_msg)
        # Second browser call on the same site did NOT re-inject the site memory
        second_nav_tool_msg = nav_model.requests[3][-1]["content"]
        self.assertNotIn("[harness] Site memory for cnn.com", second_nav_tool_msg)
        self.assertTrue(any("Loaded site memory for cnn.com." in t for k, t in events if k == "note"))

        # 3) No 5-line cap per site and no premature review at 15 total lessons:
        many_cnn_lessons = "\n".join(
            f"- CNN US flow {i}: `read https://www.cnn.com/section-{i}` for topic {i}."
            for i in range(1, 13)
        )
        memory.save_long_term(many_cnn_lessons, sites={"cnn.com"})
        self.assertEqual(len(memory.for_site("cnn.com").splitlines()), 12)
        self.assertIn("https://www.linkedin.com/search/results/people/", memory.for_site("linkedin.com"))
        self.assertFalse(memory.needs_review())

    def test_ask_user_choice_guardrail_enforces_hitl_gate_instead_of_plain_text_question(self) -> None:
        from threading import Event
        from agent import BrowserAgent

        # Reproduces the scenario where the model tries `done=1` on `read` and then tries to ask in plain text
        # before finally opening `tasks action=ask` with options.
        class ChoiceGateModel:
            usage_name = "gpt-test"

            def __init__(self) -> None:
                self.requests: list[list[dict]] = []
                self.turns = [
                    Turn(tool_calls=[ToolCall("t1", "tasks", json.dumps({
                        "action": "plan",
                        "items": ["Open the news portal and find two AI articles, then ask the user which one they want summarized."],
                    }))]),
                    Turn(tool_calls=[ToolCall("b1", "browser", json.dumps({
                        "command": "read",
                        "args": ["https://news.example.com/ai/"],
                        "done": 1,
                        "note": "Found 2 AI articles",
                    }))]),
                    Turn(content="I found two AI articles:\n1. AI Search Links\n2. Model Training Policy\nWhich one would you like me to summarize: 1 or 2?"),
                    Turn(tool_calls=[ToolCall("t2", "tasks", json.dumps({
                        "action": "ask",
                        "index": 1,
                        "note": "Found two AI articles:\n1. AI Search Highlights Links\n2. Model Training Policy Update\nWhich one should I summarize?",
                        "options": ["1. AI Search Links", "2. Model Training Policy"],
                    }))]),
                    Turn(content="Summary of article 1: AI search responses will now highlight source links more prominently."),
                ]

            def complete(self, messages, tools, on_event, cancel, tool_choice="auto"):
                self.requests.append([dict(m) for m in messages])
                return self.turns.pop(0)

        asked_dialogs: list[tuple[str, list[str]]] = []
        closed_calls: list[str] = []
        model = ChoiceGateModel()
        agent = BrowserAgent(
            model,
            run_tool=lambda raw: "- Page URL: https://news.example.com/ai/\n### Text\n1. AI Search Links\n2. Model Training Policy",
            ask_user=lambda q, opts: asked_dialogs.append((q, opts)) or "1. AI Search Links",
            close_browser=lambda: closed_calls.append("closed") or "Browser closed.",
        )
        answer = agent.run(
            "Open https://news.example.com/ai/, find two AI news articles, and ask me which one to summarize.",
            lambda k, t: None,
            Event(),
        )
        # 1) Inline `done=1` on `read` was refused because `tasks action=ask` had not been called yet
        self.assertIn("Task 1 NOT marked done", model.requests[2][-1]["content"])
        self.assertIn("HITL choice gate", model.requests[2][-1]["content"])
        # 2) Plain-text question in step 2 was pushed back to force `tasks action=ask`
        self.assertIn("opening the HITL choice gate", model.requests[3][-1]["content"])
        # 3) `ask_user` dialog opened with multi-line note and the 2 button options, and resolved in the same turn
        self.assertEqual(len(asked_dialogs), 1)
        self.assertIn("1. AI Search Highlights Links\n2. Model Training Policy Update", asked_dialogs[0][0])
        self.assertEqual(asked_dialogs[0][1], ["1. AI Search Links", "2. Model Training Policy"])
        # 4) After the user picked option 1, the agent returned the summary and closed the task + browser
        self.assertIn("Summary of article 1", answer)
        self.assertEqual(agent.plan.summary(), "1/1 done")
        self.assertEqual(closed_calls, ["closed"])

        # 5) Language-agnostic check (generic plan item without the word 'ask'):
        #    When the model finishes browser actions with a question ('?') while HITL is active, the harness
        #    still intercepts it and routes through `tasks action=ask`.
        question_dialogs: list[tuple[str, list[str]]] = []
        question_model = ChoiceGateModel()
        question_model.turns = [
            Turn(tool_calls=[ToolCall("t1", "tasks", json.dumps({
                "action": "plan",
                "items": [{"text": "Find two AI articles on the portal", "kind": "research"}],
            }))]),
            Turn(tool_calls=[ToolCall("b1", "browser", json.dumps({
                "command": "read",
                "args": ["https://news.example.com/ai/"],
            }))]),
            Turn(content="Found two AI stories:\n1. Story A\n2. Story B\nWhich one should I summarize?"),
            Turn(tool_calls=[ToolCall("t2", "tasks", json.dumps({
                "action": "ask",
                "index": 1,
                "note": "Which of the two AI stories should I summarize?",
                "options": ["1. Story A", "2. Story B"],
            }))]),
            Turn(content="Summary of Story B: Enterprise AI adoption accelerated."),
        ]
        question_agent = BrowserAgent(
            question_model,
            run_tool=lambda raw: "- Page URL: https://news.example.com/ai/\n### Text\n1. Story A\n2. Story B",
            ask_user=lambda q, opts: question_dialogs.append((q, opts)) or "2. Story B",
        )
        q_answer = question_agent.run(
            "Check https://news.example.com/ai/ for two AI stories and let me pick one before summarizing.",
            lambda k, t: None,
            Event(),
        )
        self.assertEqual(len(question_dialogs), 1)
        self.assertEqual(question_dialogs[0][1], ["1. Story A", "2. Story B"])
        self.assertIn("Summary of Story B", q_answer)

        # 6) 3-step plan where Task 1 finds items, Task 2 is the ask step, and Task 3 summarizes:
        #    When the model calls `tasks action=ask` with `index=1` (while Task 1 is in progress), answering
        #    the dialog completes both Task 1 and Task 2 and advances directly to Task 3 without asking twice!
        three_step_dialogs: list[tuple[str, list[str]]] = []
        three_step_model = ChoiceGateModel()
        three_step_model.turns = [
            Turn(tool_calls=[ToolCall("t1", "tasks", json.dumps({
                "action": "plan",
                "items": [
                    {"text": "Open the news portal and find two AI news articles", "kind": "research"},
                    {"text": "Ask the user which of the two articles to summarize.", "kind": "ask"},
                    {"text": "Read the chosen article and write a one-paragraph summary.", "kind": "write"},
                ],
            }))]),
            Turn(tool_calls=[ToolCall("b1", "browser", json.dumps({
                "command": "read",
                "args": ["https://news.example.com/ai/"],
            }))]),
            Turn(tool_calls=[ToolCall("t2", "tasks", json.dumps({
                "action": "ask",
                "index": 1,
                "note": "Found two AI articles:\n1. Story A\n2. Story B",
                "options": ["Story A", "Story B"],
            }))]),
            Turn(tool_calls=[ToolCall("b2", "browser", json.dumps({
                "command": "read",
                "args": ["https://news.example.com/ai/story-b"],
            }))]),
            Turn(content="One-paragraph summary: Story B covers new AI search features."),
        ]
        three_step_agent = BrowserAgent(
            three_step_model,
            run_tool=lambda raw: "- Page URL: https://news.example.com/ai/\n### Text\nStory B",
            ask_user=lambda q, opts: three_step_dialogs.append((q, opts)) or "Story B",
        )
        three_step_ans = three_step_agent.run(
            "Open https://news.example.com/ai/, select 2 AI articles, ask me which one to summarize, and then summarize it.",
            lambda k, t: None,
            Event(),
        )
        self.assertEqual(len(three_step_dialogs), 1)  # Asked ONLY ONCE!
        self.assertEqual(three_step_agent.plan.summary(), "3/3 done")
        self.assertIn("Now task 3: Read the chosen article", three_step_model.requests[3][-1]["content"])
        self.assertIn("One-paragraph summary", three_step_ans)

        # 7) When the model closes Task 1 first and then calls `tasks action=ask` with `index=2`,
        #    answering Task 2 must mark ONLY Task 2 as `done` (2/3 done) and set Task 3 ("Read the chosen article...")
        #    to `in progress` — never marking Task 3 `done` prematurely while Task 3 is still running!
        step2_plan_snapshots: list[str] = []
        step2_model = ChoiceGateModel()
        step2_model.turns = [
            Turn(tool_calls=[ToolCall("t1", "tasks", json.dumps({
                "action": "plan",
                "items": [
                    {"text": "Open the news portal and locate two AI news articles.", "kind": "research"},
                    {"text": "Ask the user which of the two articles should be summarized.", "kind": "ask"},
                    {"text": "Read the selected article and produce a one-paragraph summary.", "kind": "write"},
                ],
            }))]),
            Turn(tool_calls=[ToolCall("b1", "browser", json.dumps({
                "command": "links",
                "args": [],
                "done": 1,
                "note": "Found 2 AI articles",
            }))]),
            Turn(tool_calls=[ToolCall("t2", "tasks", json.dumps({
                "action": "ask",
                "index": 2,
                "note": "Which article should I summarize?",
                "options": ["1. AI Policy Story", "2. AI Hardware Chip Story"],
            }))]),
            Turn(tool_calls=[ToolCall("b2", "browser", json.dumps({
                "command": "read",
                "args": ["https://news.example.com/ai/chip-story"],
            }))]),
            Turn(content="Summary: Next-generation AI accelerators are scheduled for production."),
        ]
        step2_agent = BrowserAgent(
            step2_model,
            run_tool=lambda raw: "- Page URL: https://news.example.com/ai/\n### Text\nAI Chip",
            ask_user=lambda q, opts: "2. AI Hardware Chip Story",
        )
        step2_agent.run(
            "Open https://news.example.com/ai/, select 2 AI articles, ask me which one to summarize, and then summarize it.",
            lambda k, t: step2_plan_snapshots.append(t) if k == "plan" else None,
            Event(),
        )
        # Right after answering `tasks ask 2`, the plan emitted must be `2/3 done` with Task 3 `▶` (in progress)!
        self.assertTrue(any(
            "Plan - 2/3 done" in snap
            and "✅ 2. Ask the user" in snap
            and "▶ 3. Read the selected article" in snap
            for snap in step2_plan_snapshots
        ))
        self.assertEqual(step2_agent.plan.summary(), "3/3 done")

    def test_composer_up_and_down_arrow_recalls_last_3_prompts_across_sessions(self) -> None:
        from memory import Memory
        from tui import ComposerInput

        mem_folder = self.folder / "memory"
        mem1 = Memory(mem_folder)
        for p in ("Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4"):
            mem1.record_prompt(p)
        # Only the last 3 unique prompts are retained across sessions
        self.assertEqual(mem1.recent_prompts(), ["Prompt 2", "Prompt 3", "Prompt 4"])

        async def scenario() -> tuple[str, str, str, str]:
            # New session loads the 3 saved prompts from disk automatically
            mem2 = Memory(mem_folder)
            app = BrowserApp(
                model_factory=FakeModel,
                log=NullLog(),
                memory=mem2,
                telegram_settings_path=self.folder / "telegram.json",
            )
            async with app.run_test(size=(110, 40)) as pilot:
                composer = app.query_one("#composer", ComposerInput)
                await pilot.press("up")
                after_up_1 = composer.text
                await pilot.press("up")
                after_up_2 = composer.text
                await pilot.press("up")
                after_up_3 = composer.text
                await pilot.press("down")
                after_down_1 = composer.text
                return after_up_1, after_up_2, after_up_3, after_down_1

        self.assertEqual(
            asyncio.run(scenario()),
            ("Prompt 4", "Prompt 3", "Prompt 2", "Prompt 3"),
        )


if __name__ == "__main__":
    unittest.main()
