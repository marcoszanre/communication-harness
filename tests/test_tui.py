import os

os.environ["BROWSER_HARNESS_PLANNER"] = "0"  # the planner is a real model: off in tests

import asyncio
import unittest

from textual.widgets import Markdown, Static

from foundry import ToolCall, Turn
from memory import NullMemory
from session_log import NullLog
from tui import BrowserApp, ComposerInput


class FakeModel:
    def complete(self, messages, tools, emit, cancel, tool_choice="auto"):
        emit("answer", "Pagina aberta.")
        return Turn(content="Pagina aberta.", input_tokens=10, output_tokens=3)

    def close(self):
        pass


class PlanModel:
    """Creates the plan first, then completes each task and answers."""

    def __init__(self):
        self.turns = [
            Turn(tool_calls=[ToolCall("t1", "tasks", '{"action":"plan","items":["abrir a.com","resumir"]}')]),
            Turn(tool_calls=[ToolCall("t2", "tasks", '{"action":"done","index":1,"note":"ok"}')]),
            Turn(tool_calls=[ToolCall("t3", "tasks", '{"action":"done","index":2,"note":"ok"}')]),
            Turn(content="Feito."),
        ]

    def complete(self, messages, tools, emit, cancel, tool_choice="auto"):
        return self.turns.pop(0)

    def close(self):
        pass


class TuiTests(unittest.TestCase):
    def setUp(self):
        # Workspaces of the tests go to a temporary folder, never to browser_harness/workspaces.
        import functools
        import tempfile
        from pathlib import Path
        from unittest import mock

        import workspace
        self._temp = tempfile.TemporaryDirectory()
        self.workspaces = Path(self._temp.name)
        patcher = mock.patch("tui.Workspace", functools.partial(workspace.Workspace, self.workspaces))
        settings_patcher = mock.patch("tui.SETTINGS_PATH", self.workspaces / "default-settings.json")
        tg_patcher = mock.patch("tui.TELEGRAM_SETTINGS_PATH", self.workspaces / "default-telegram.json")
        patcher.start()
        settings_patcher.start()
        tg_patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(settings_patcher.stop)
        self.addCleanup(tg_patcher.stop)
        self.addCleanup(self._temp.cleanup)

    def test_models_panel_validates_and_saves_independent_local_roles(self):
        import json
        from textual.widgets import Button, Input, Select
        from tui import ModelSettingsPanel

        async def scenario():
            path = self.workspaces / "settings.json"
            app = BrowserApp(model_factory=FakeModel, log=NullLog(), memory=NullMemory(), settings_path=path)
            async with app.run_test(size=(110, 60)) as pilot:
                app._show_model_settings()
                await pilot.pause()
                panel = app.query_one(ModelSettingsPanel)
                panel.query_one("#executor-provider", Select).value = "local"
                await pilot.pause()
                panel.on_button_pressed(Button.Pressed(panel.query_one("#models-save", Button)))
                self.assertIn("enter a model", str(panel.query_one("#model-settings-error", Static).content))
                self.assertFalse(path.exists())
                panel.query_one("#executor-model", Input).value = "example-local-executor"
                panel.query_one("#planner-provider", Select).value = "local"
                await pilot.pause()
                panel.query_one("#planner-model", Input).value = "example-local-planner"
                panel.on_button_pressed(Button.Pressed(panel.query_one("#models-save", Button)))
                await pilot.pause()
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(data["executor"]["provider"], "local")
                self.assertEqual(data["planner"]["model"], "example-local-planner")
                self.assertNotIn("api_key", data)
                self.assertIn("Settings saved", str(app.query_one("#settings", Static).content))
        asyncio.run(scenario())

    def test_models_panel_opens_centered_and_applies_ready_presets(self):
        import json
        from textual.widgets import Button, Input, Select
        from model_config import Settings
        from tui import ModelSettingsPanel

        async def scenario():
            path = self.workspaces / "settings.json"
            endpoint = "https://example.services.ai.azure.com/openai/v1/"
            initial = Settings(azure_endpoint=endpoint).with_preset("local")
            app = BrowserApp(model_factory=FakeModel, log=NullLog(), memory=NullMemory(),
                             settings=initial, settings_path=path)
            async with app.run_test(size=(120, 42)) as pilot:
                app._show_model_settings()
                await pilot.pause()
                panel = app.query_one(ModelSettingsPanel)
                dialog = panel.query_one("#models-dialog")
                self.assertNotIn(panel, app.query_one("#thread").children)
                self.assertEqual(dialog.scroll_y, 0)
                self.assertGreater(dialog.region.y, 0)
                self.assertLess(dialog.region.y + dialog.region.height, 42)
                save_btn = panel.query_one("#models-save", Button)
                check_btn = panel.query_one("#models-check", Button)
                cancel_btn = panel.query_one("#models-cancel", Button)
                self.assertEqual(save_btn.region.y, check_btn.region.y)
                self.assertEqual(check_btn.region.y, cancel_btn.region.y)
                for preset_id in ("preset-local", "preset-azure", "preset-hybrid", "preset-local-fast"):
                    panel.on_button_pressed(Button.Pressed(panel.query_one(f"#{preset_id}", Button)))
                    await pilot.pause()
                    for btn in panel.query(Button):
                        self.assertEqual(btn.region.height, 1, btn.id)
                        self.assertEqual(btn.content_region.height, 1, btn.id)
                        rendered = ["".join(seg.text for seg in line).strip()
                                    for line in btn.render_lines(btn.size.region)]
                        self.assertEqual(rendered, [str(btn.label)], btn.id)
                await pilot.hover("#models-check")
                await pilot.pause()
                self.assertEqual(check_btn.region.height, 1)
                self.assertEqual(check_btn.content_region.height, 1)
                panel.on_button_pressed(Button.Pressed(panel.query_one("#preset-azure", Button)))
                await pilot.pause()
                self.assertEqual(panel.query_one("#executor-provider", Select).value, "azure")
                self.assertEqual(panel.query_one("#executor-model", Input).value, "gpt-4.1-mini")
                self.assertEqual(panel.query_one("#executor-endpoint", Input).value, endpoint)
                self.assertEqual(panel.query_one("#planner-provider", Select).value, "azure")
                self.assertEqual(panel.query_one("#planner-model", Input).value, "gpt-4.1")
                panel.on_button_pressed(Button.Pressed(panel.query_one("#preset-local", Button)))
                await pilot.pause()
                self.assertEqual(panel.query_one("#executor-provider", Select).value, "local")
                self.assertEqual(panel.query_one("#executor-model", Input).value, "qwen2.5-1.5b")
                self.assertEqual(panel.query_one("#planner-provider", Select).value, "local")
                self.assertEqual(panel.query_one("#planner-model", Input).value, "qwen2.5-1.5b")
                panel.on_button_pressed(Button.Pressed(panel.query_one("#models-save", Button)))
                await pilot.pause()
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(data["executor"]["model"], "qwen2.5-1.5b")
                self.assertEqual(data["azure_endpoint"], endpoint)
        asyncio.run(scenario())

    def test_model_check_failure_is_visible_and_does_not_fall_back(self):
        from unittest import mock
        from model_config import ModelSettings, Settings

        async def scenario():
            settings = Settings(executor=ModelSettings(provider="local", model="example-local-model"))
            app = BrowserApp(log=NullLog(), memory=NullMemory(), settings=settings,
                             settings_path=self.workspaces / "settings.json")
            async with app.run_test() as pilot:
                app._save_model_settings((settings, True))
                for _ in range(40):
                    await pilot.pause(0.05)
                    if not app.busy:
                        break
                self.assertFalse(app.busy)
                text = "\n".join(str(widget.content) for widget in app.query(Static))
                self.assertIn("no cloud fallback", text)
                self.assertIn("synthetic model unavailable", text)
                self.assertIsNone(app.agent)
        with mock.patch("model_config.Settings.create_model", side_effect=RuntimeError("synthetic model unavailable")):
            asyncio.run(scenario())

    def test_workspace_command_and_new_session_folder(self):
        async def scenario():
            app = BrowserApp(model_factory=PlanModel, log=NullLog(), memory=NullMemory())
            async with app.run_test() as pilot:
                first = app.workspace.id
                app.query_one("#composer", ComposerInput).load_text("abra a.com e resuma")
                await pilot.press("enter")
                for _ in range(50):
                    await pilot.pause(0.05)
                    if not app.busy:
                        break
                self.assertTrue((self.workspaces / first / "index.md").is_file())
                app.query_one("#composer", ComposerInput).load_text("/new")
                await pilot.press("enter")
                await pilot.pause()
                self.assertNotEqual(app.workspace.id, first)
                self.assertIs(app.agent.workspace, app.workspace)
                app.query_one("#composer", ComposerInput).load_text("/files")
                await pilot.press("enter")
                await pilot.pause()
                texts = "\n".join(getattr(w.content, "markup", str(w.content)) for w in app.query(Static))
                texts += "\n".join(markdown.source for markdown in app.query(Markdown))
                self.assertIn(f"Session `{app.workspace.id}`", texts)
                self.assertIn(f"{first}: 1 files; first request: abra a.com e resuma", texts)

        asyncio.run(scenario())

    def test_plan_panel_shows_progress(self):
        async def scenario():
            app = BrowserApp(model_factory=PlanModel, log=NullLog(), memory=NullMemory())
            async with app.run_test() as pilot:
                app.query_one("#composer", ComposerInput).load_text("abra a.com e resuma")
                await pilot.press("enter")
                for _ in range(50):
                    await pilot.pause(0.05)
                    if not app.busy:
                        break
                panel = app.query_one("#plan", Static)
                self.assertTrue(panel.display)
                self.assertIn("Plan - 2/2 done", str(panel.content))
                self.assertIn("✅ 2. resumir", str(panel.content))

        asyncio.run(scenario())

    def test_prompt_streams_answer_and_usage(self):
        async def scenario():
            app = BrowserApp(model_factory=FakeModel, log=NullLog(), memory=NullMemory())
            async with app.run_test() as pilot:
                app.query_one("#composer", ComposerInput).load_text("abra example.com")
                await pilot.press("enter")
                for _ in range(50):
                    await pilot.pause(0.05)
                    if not app.busy:
                        break
                texts = [getattr(w.content, "markup", str(w.content)) for w in app.query(Static)]
                self.assertTrue(any("Pagina aberta." in text for text in texts))
                self.assertIn("In 10", str(app.query_one("#usage", Static).content))
                self.assertFalse(app.query_one("#composer", ComposerInput).disabled)

        asyncio.run(scenario())

    def test_clear_commands_reset_thread_and_counters(self):
        async def scenario():
            app = BrowserApp(model_factory=FakeModel, log=NullLog(), memory=NullMemory())
            async with app.run_test() as pilot:
                for command in ("/clear", "/reset", "/new"):
                    app.query_one("#composer", ComposerInput).load_text(command)
                    await pilot.press("enter")
                    await pilot.pause()
                    self.assertEqual(len(app.query_one("#thread").children), 1)
                    self.assertIn("ctx 0", str(app.query_one("#usage", Static).content))
                app.query_one("#composer", ComposerInput).load_text("/limpar")
                await pilot.press("enter")
                await pilot.pause()
                texts = [str(w.content) for w in app.query(Static)]
                self.assertTrue(any("Unknown command" in text for text in texts))

        asyncio.run(scenario())

    def test_teach_commands_guide_the_user(self):
        import tempfile
        from pathlib import Path

        from playbooks import Playbooks

        async def scenario(folder):
            app = BrowserApp(model_factory=FakeModel, log=NullLog(), memory=NullMemory(),
                             playbooks=Playbooks(Path(folder)))
            async with app.run_test() as pilot:
                for command in ("/teach", "/teach done", "/teach note x", "/playbooks"):
                    app.query_one("#composer", ComposerInput).load_text(command)
                    await pilot.press("enter")
                    await pilot.pause()
                texts = "\n".join(getattr(w.content, "markup", str(w.content)) for w in app.query(Static))
                self.assertIn("Use /teach <task> to start", texts)
                self.assertNotIn("latamairlines", texts)
                self.assertIn("No taught tasks yet", texts)

        with tempfile.TemporaryDirectory() as folder:
            asyncio.run(scenario(folder))

    def test_composer_grows_with_soft_wrapped_lines(self):
        async def scenario():
            app = BrowserApp(model_factory=FakeModel, log=NullLog(), memory=NullMemory())
            async with app.run_test(size=(60, 40)) as pilot:
                composer = app.query_one("#composer", ComposerInput)
                composer.insert("example " * 30)  # one long logical line, without Enter
                await pilot.pause()
                await pilot.pause()
                row = app.query_one("#composer-row")
                self.assertGreater(composer.wrapped_document.height, 1)
                self.assertEqual(row.size.height, composer.wrapped_document.height + 2)

        asyncio.run(scenario())

    def test_approval_is_inline_in_the_conversation_y_approves_escape_denies(self):
        from tui import ApprovalPanel

        async def scenario():
            app = BrowserApp(model_factory=FakeModel, log=NullLog(), memory=NullMemory())
            results = []
            async with app.run_test() as pilot:
                for key in ("y", "escape"):
                    app._show_panel_inline(ApprovalPanel('press Enter to submit "oi"', results.append), "wait")
                    await pilot.pause()
                    panel = list(app.query(ApprovalPanel))[-1]
                    self.assertIs(app.screen, app.screen_stack[0])  # no modal: the history stays on screen
                    self.assertIn(panel, app.query_one("#thread").children)
                    self.assertEqual(app.focused.id, "deny")  # Enter by accident does not approve
                    await pilot.press(key)
                    await pilot.pause()
                    self.assertTrue(panel.answered)
            return results

        self.assertEqual(asyncio.run(scenario()), [True, False])

    def test_question_inline_picks_an_option_types_another_or_answers_later(self):
        from textual.widgets import Input

        from tui import QuestionPanel

        async def scenario():
            app = BrowserApp(model_factory=FakeModel, log=NullLog(), memory=NullMemory())
            results = []
            async with app.run_test() as pilot:
                app._show_panel_inline(QuestionPanel("Which chat?", ["Me", "Team"], results.append), "q")
                await pilot.pause()
                await pilot.press("2")
                await pilot.pause()
                app._show_panel_inline(QuestionPanel("Which chat?", ["Me", "Team"], results.append), "q")
                await pilot.pause()
                list(app.query(QuestionPanel))[-1].query_one("#question-other", Input).focus()
                await pilot.press("A", "n", "a", "enter")
                await pilot.pause()
                app._show_panel_inline(QuestionPanel("Which chat?", [], results.append), "q")
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()
            return results

        self.assertEqual(asyncio.run(scenario()), ["Team", "Ana", None])

    def test_right_click_copies_a_selection_and_relative_links_are_fixed(self):
        from unittest import mock

        async def scenario():
            app = BrowserApp(model_factory=FakeModel, log=NullLog(), memory=NullMemory())
            copied = []
            async with app.run_test() as pilot:
                with mock.patch("tui.write_clipboard_text", copied.append), \
                        mock.patch.object(type(app.screen), "get_selected_text", lambda screen: "texto"):
                    self.assertTrue(app.copy_selection())
                with mock.patch.object(type(app.screen), "get_selected_text", lambda screen: None):
                    self.assertFalse(app.copy_selection())
                self.assertEqual(app._fix_links("[a.md](a.md) [s](https://s.com)"), "`a.md` [s](https://s.com)")
                self.assertEqual(
                    [button.id for button in app.query("#actions Button")],
                    ["close-browser", "models", "telegram", "stop"],
                )
            return copied

        self.assertEqual(asyncio.run(scenario()), ["texto"])

    def test_teach_is_one_command_opens_the_named_site_and_typed_text_is_a_note(self):
        # Offline: fake guides and a fake navigation - no real browser, no real account.
        from unittest import mock

        from playbooks import NoStartPage, Teaching

        class Memory(NullMemory):
            def long_term(self):
                return "- YouTube search: read https://www.youtube.com/results?search_query=<terms>"

        class Books:
            folder = None

            def __init__(self):
                self.starts, self.notes, self.stopped = 0, [], None

            def all(self):
                return []

            def start(self, task):
                self.starts += 1
                if self.starts == 1:
                    raise NoStartPage("no page")
                return Teaching(task, "https://www.youtube.com/", "YouTube")

            def add_note(self, teaching, text):
                self.notes.append(text)
                return f"on page: {text}"

            def stop(self, teaching, save=True, note="", write_guide=None):
                self.stopped = (save, note)
                return None

        async def scenario(books, opened):
            app = BrowserApp(model_factory=FakeModel, log=NullLog(), memory=Memory(), playbooks=books)
            async with app.run_test() as pilot:
                for text in ("/teach search a youtube tutorial", "click the first result", "/teach"):
                    app.query_one("#composer", ComposerInput).load_text(text)
                    await pilot.press("enter")
                    for _ in range(10):
                        await pilot.pause()
                return app.teaching

        books, opened = Books(), []
        with mock.patch("tui.navigate", lambda url: opened.append(url) or "- Page URL: " + url), \
                mock.patch("tui.navigation_error", lambda result: ""):
            still_recording = asyncio.run(scenario(books, opened))
        self.assertEqual(opened, ["https://youtube.com/"])  # the site named in the task, from the memory
        self.assertEqual(books.notes, ["click the first result"])  # typed while recording = a note, not a request
        self.assertEqual(books.stopped, (True, ""))  # /teach again finishes and saves
        self.assertIsNone(still_recording)

    def test_composer_shows_that_it_is_waiting_while_the_agent_works(self):
        async def scenario():
            app = BrowserApp(model_factory=FakeModel, log=NullLog(), memory=NullMemory())
            async with app.run_test() as pilot:
                composer = app.query_one("#composer", ComposerInput)
                app._set_busy(True)
                await pilot.pause()
                busy = (composer.disabled, composer.placeholder)
                app._set_busy(False)
                await pilot.pause()
                return busy, composer.placeholder

        (disabled, busy_text), idle_text = asyncio.run(scenario())
        self.assertTrue(disabled)
        self.assertIn("Working", busy_text)
        self.assertEqual(idle_text, "E.g.: open playwright.dev")


if __name__ == "__main__":
    unittest.main()
