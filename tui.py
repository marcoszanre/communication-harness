"""Full-screen TUI (raw mode) wired to the browser agent."""

from __future__ import annotations

import re
import json
from pathlib import Path
from threading import Event

from azure.core.exceptions import AzureError
from openai import OpenAIError
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Checkbox, Collapsible, Input, Markdown, Select, Static, TextArea

import planner as planner_module
from agent import BrowserAgent
from browser_tool import SESSION, close_browser, navigate, navigation_error, normalize_url
from clipboard import read_clipboard_text, write_clipboard_text
from foundry import Turn
from model_config import (
    AZURE_DEFAULT_EXECUTOR,
    AZURE_DEFAULT_PLANNER,
    AZURE_DEFAULT_REASONING,
    AZURE_MODEL_CHOICES,
    AZURE_MODEL_VALUES,
    LOCAL_DEFAULT_MODEL,
    LOCAL_MODEL_CHOICES,
    LOCAL_MODEL_VALUES,
    PRESET_CHOICES,
    SETTINGS_PATH,
    ModelSettings,
    Settings,
    is_valid_azure_endpoint,
)
from memory import Memory, site_keys
from playbooks import NoStartPage, Playbooks, Teaching
from session_log import SessionLog
from telegram_channel import (
    TELEGRAM_LEVEL_CHOICES,
    TELEGRAM_LEVEL_LABELS,
    TELEGRAM_SETTINGS_PATH,
    TelegramChannel,
    TelegramSettings,
    level_description,
)
from terminal_keys import enable_shift_enter
from usage import Usage
from workspace import Workspace

HELP = (
    "Ask in natural language, for example:\n"
    "- `open playwright.dev`\n"
    "- `click Docs and tell me the page title`\n"
    "- `summarize two AI news articles and save the summary to a Markdown file`\n\n"
    "Commands: `/help`  `/login <url>` (opens the page so you can sign in; the login is saved)  "
    "`/teach <task>` (learning mode: you show the task in the browser and the harness learns it; while "
    "recording, what you type is a note; `/teach` again saves, `/teach cancel` discards)  "
    "`/playbooks` (taught tasks)  `/workspace` (files of this session: saved pages, screenshots, results; also `/files`)  "
    "`/models` (configure and check planning/execution providers: Azure Foundry or Foundry Local)  "
    "`/telegram` (configure local Telegram Push/Pull channel: Off, L1 Task Update, L2 HITL, L3 Accept Tasks)  "
    "`/usage` (session usage per model: calls, tokens, reported cost)  "
    "`/memory` (shows the long-term memory)  `/clear` (new session: resets context, plan and counters and starts a new workspace folder; also `/reset`, `/new`)  "
    "`/restart` (clears and closes the browser)  `/close` (closes the browser)  `/quit` (also `/exit`)\n"
    "Shortcuts: Enter sends | Up/Down recalls last 3 prompts | Ctrl+J new line | Ctrl+X stops | Ctrl+Q quits"
)
COMMANDS = ("/help", "/models", "/telegram", "/login ", "/teach ", "/teach cancel", "/playbooks",
            "/workspace", "/usage", "/memory", "/clear", "/restart", "/close", "/quit")
ALIASES = {"/reset": "/clear", "/new": "/clear", "/exit": "/quit", "/files": "/workspace"}
IDLE_PLACEHOLDER = "E.g.: open playwright.dev"
MAX_COMPOSER_ROWS = 10  # visible text lines before the composer starts scrolling


class ComposerInput(TextArea):
    """Composer: Enter sends, Shift/Ctrl+Enter inserts a new line, Up/Down recalls the last 3 prompts."""

    class Submitted(Message):
        def __init__(self, composer: ComposerInput, value: str) -> None:
            super().__init__()
            self.composer = composer
            self.value = value

    def __init__(self, **kwargs) -> None:
        super().__init__(soft_wrap=True, tab_behavior="focus", highlight_cursor_line=False, **kwargs)
        self._history: list[str] = []
        self._history_index: int | None = None
        self._draft_before_history: str = ""

    def set_history(self, prompts: list[str]) -> None:
        self._history = [p for p in prompts if p.strip()][-3:]
        self._history_index = None
        self._draft_before_history = ""

    def _load_history_entry(self, text: str) -> None:
        self.load_text(text)
        lines = text.split("\n")
        self.move_cursor((max(0, len(lines) - 1), len(lines[-1]) if lines else 0))

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self._history_index = None
            self._draft_before_history = ""
            self.post_message(self.Submitted(self, self.text))
        elif event.key == "up" and self._history and (
            self._history_index is not None or self.cursor_location[0] == 0
        ):
            event.prevent_default()
            event.stop()
            if self._history_index is None:
                self._draft_before_history = self.text
                self._history_index = len(self._history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            self._load_history_entry(self._history[self._history_index])
        elif event.key == "down" and self._history_index is not None and (
            self.cursor_location[0] >= max(0, self.document.line_count - 1)
        ):
            event.prevent_default()
            event.stop()
            if self._history_index < len(self._history) - 1:
                self._history_index += 1
                self._load_history_entry(self._history[self._history_index])
            else:
                draft = self._draft_before_history
                self._history_index = None
                self._draft_before_history = ""
                self._load_history_entry(draft)
        elif event.key in {"ctrl+enter", "ctrl+j", "shift+enter", "alt+enter"}:
            event.prevent_default()
            event.stop()
            self.insert("\n")
        elif event.key == "tab" and self.text.startswith("/"):
            event.prevent_default()
            event.stop()
            match = next((c for c in COMMANDS if c.startswith(self.text) and c != self.text), None)
            if match:
                self.insert(match[len(self.text):])

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button != 3:
            return
        event.prevent_default()
        event.stop()
        # Right-click with text selected (in the box or in the conversation) copies it; otherwise it pastes.
        if self.app.copy_selection(self.selected_text):
            return
        try:
            text = read_clipboard_text()
        except (OSError, RuntimeError, ValueError) as error:
            self.app.notify(str(error), severity="warning")
            return
        self.focus()
        self.post_message(events.Paste(text))


class InlinePanel(Vertical):
    """A decision inside the conversation (not a modal): the history above stays visible and scrollable."""

    DEFAULT_CSS = """
    InlinePanel { height: auto; margin: 1 0; padding: 1 2; background: #1b1f27; border: round #d8b36a; }
    InlinePanel.question { border: round #7fa7d9; }
    InlinePanel .panel-title { color: #e6c687; text-style: bold; }
    InlinePanel.question .panel-title { color: #9cc0ee; }
    InlinePanel .panel-body { color: #eef1f5; background: #252a33; padding: 0 1; margin: 1 0; }
    InlinePanel .panel-hint { color: #939eae; }
    InlinePanel .panel-buttons { height: auto; margin: 1 0 0 0; }
    InlinePanel .panel-buttons Button { margin: 0 2 0 0; min-width: 16; }
    InlinePanel.question .panel-buttons { layout: vertical; }
    InlinePanel.question .panel-buttons Button { width: 100%; margin: 0; background: #252b34; color: #dfe5ee;
                                                 border: none; }
    InlinePanel.question .panel-buttons Button:focus { background: #33507a; }
    InlinePanel Input { margin: 1 0 0 0; }
    InlinePanel .panel-result { color: #a9c4ad; }
    """

    def __init__(self, finish, classes: str = "") -> None:
        super().__init__(classes=classes)
        self.finish = finish
        self.answered = False

    def _done(self, value, summary: str) -> None:
        if self.answered:
            return
        self.answered = True
        self.finish(value)
        self.remove_children()
        self.mount(Static(Text(summary), classes="panel-result"))

    def cancel(self) -> None:
        self._done(None, "Stopped.")


class ApprovalPanel(InlinePanel):
    """Approval of an irreversible browser action: the agent waits for the user."""

    BINDINGS = [Binding("y", "approve", "Approve"), Binding("n", "deny", "Deny"), Binding("escape", "deny", "Deny")]

    def __init__(self, action: str, finish) -> None:
        super().__init__(finish)
        self.action_text = action

    def compose(self) -> ComposeResult:
        yield Static("Approve this browser action?", classes="panel-title")
        yield Static(Text(self.action_text), classes="panel-body")
        yield Static("The safety check judged it hard to undo and not clearly requested. Scroll up to see the "
                     "conversation. y approves, n or Esc denies.", classes="panel-hint")
        with Horizontal(classes="panel-buttons"):
            yield Button("Approve (y)", id="approve", variant="success")
            yield Button("Deny (n)", id="deny", variant="error")

    def on_mount(self) -> None:
        self.query_one("#deny", Button).focus()  # Enter by accident must not approve

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        (self.action_approve if event.button.id == "approve" else self.action_deny)()

    def action_approve(self) -> None:
        self._done(True, f"✓ Approved: {self.action_text[:160]}")

    def action_deny(self) -> None:
        self._done(False, f"✗ Denied: {self.action_text[:160]}")

    def cancel(self) -> None:
        self._done(False, f"✗ Denied (stopped): {self.action_text[:160]}")


class QuestionPanel(InlinePanel):
    """HITL question of the agent: pick an option (1-4) or type another answer; Esc = answer later in the chat."""

    BINDINGS = [Binding("escape", "later", "Answer in the chat"),
                *(Binding(str(n), f"pick({n})", show=False) for n in range(1, 5))]

    def __init__(self, question: str, options: list[str], finish) -> None:
        super().__init__(finish, classes="question")
        self.question = question
        self.options = options

    def compose(self) -> ComposeResult:
        yield Static("The agent needs your answer", classes="panel-title")
        yield Static(Text(self.question), classes="panel-body")
        with Vertical(classes="panel-buttons"):
            for number, option in enumerate(self.options, 1):
                yield Button(f"{number}. {option}", id=f"option-{number}", compact=True)
        yield Input(placeholder="Other answer... (Enter sends)", id="question-other")
        keys = f"1-{len(self.options)} picks  |  " if self.options else ""
        yield Static(f"{keys}type another answer + Enter  |  Esc answers later in the chat", classes="panel-hint")

    def on_mount(self) -> None:
        self.query_one("#option-1" if self.options else "#question-other").focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_pick(int(event.button.id.removeprefix("option-")))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if event.value.strip():
            self._answer(event.value.strip())

    def action_pick(self, number: int) -> None:
        if 1 <= number <= len(self.options):
            self._answer(self.options[number - 1])

    def _answer(self, answer: str) -> None:
        self._done(answer, f"? {self.question[:160]}\n→ {answer}")

    def action_later(self) -> None:
        self._done(None, f"? {self.question[:160]}\n→ (answer in the chat)")


class ModelSettingsPanel(InlinePanel):
    """Centered modal-overlay model settings with ready-to-select presets and validation feedback."""

    BINDINGS = [Binding("escape", "cancel_settings", "Cancel")]

    DEFAULT_CSS = """
    ModelSettingsPanel {
        layer: modal;
        dock: top;
        width: 100%;
        height: 100%;
        align: center middle;
        background: rgba(10, 12, 16, 0.84);
        margin: 0;
        padding: 0 1;
        border: none;
    }
    ModelSettingsPanel #models-dialog {
        width: 112;
        max-width: 98%;
        height: auto;
        max-height: 96%;
        padding: 0 2;
        background: #1b1f27;
        border: round #7fa7d9;
        overflow-y: auto;
    }
    ModelSettingsPanel #models-dialog .panel-title {
        color: #9cc0ee;
        text-style: bold;
    }
    ModelSettingsPanel #models-dialog .preset-bar {
        layout: horizontal;
        height: auto;
        margin: 0;
    }
    ModelSettingsPanel #models-dialog .preset-bar Button {
        width: 1fr;
        height: 1;
        min-width: 16;
        padding: 0;
        margin: 0 1 0 0;
        border: none !important;
        background: #262d38;
        color: #d6deeb;
        text-style: bold;
        tint: transparent;
    }
    ModelSettingsPanel #models-dialog .preset-bar #preset-local-fast {
        margin: 0;
    }
    ModelSettingsPanel #models-dialog .preset-bar Button:hover {
        background: #333d4d;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    ModelSettingsPanel #models-dialog .preset-bar Button:focus {
        background: #33507a;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    ModelSettingsPanel #models-dialog .preset-bar Button.-active {
        border: none !important;
        tint: transparent;
    }
    ModelSettingsPanel #models-dialog .preset-bar Button.-success {
        background: #2ea043;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    ModelSettingsPanel #models-dialog .preset-bar Button.-success:hover,
    ModelSettingsPanel #models-dialog .preset-bar Button.-success:focus {
        background: #3fb950;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    ModelSettingsPanel #models-dialog #preset-select {
        margin: 0;
    }
    ModelSettingsPanel #models-dialog #preset-summary {
        color: #a9c4ad;
        background: #232933;
        padding: 0 1;
        margin: 0;
    }
    ModelSettingsPanel #models-dialog #roles-row {
        layout: horizontal;
        height: auto;
        margin: 0;
    }
    ModelSettingsPanel #models-dialog .role-card {
        width: 1fr;
        height: auto;
        padding: 0 1;
        margin: 0 1 0 0;
        background: #222731;
        border: round #3a4659;
    }
    ModelSettingsPanel #models-dialog #planner-card {
        margin: 0;
    }
    ModelSettingsPanel #models-dialog .field-label {
        color: #9fb0c7;
        margin: 0;
    }
    ModelSettingsPanel #models-dialog Select {
        margin: 0;
    }
    ModelSettingsPanel #models-dialog Input {
        margin: 0;
    }
    ModelSettingsPanel #models-dialog Checkbox {
        height: 1;
        border: none !important;
        padding: 0;
        margin: 0;
        background: transparent;
    }
    ModelSettingsPanel #models-dialog Checkbox:focus {
        border: none !important;
        background: #252d3a;
    }
    ModelSettingsPanel #models-dialog Checkbox > .toggle--button {
        color: #2b323f;
        background: #2b323f;
    }
    ModelSettingsPanel #models-dialog Checkbox.-on > .toggle--button {
        color: #3fb950;
        background: #2b323f;
        text-style: bold;
    }
    ModelSettingsPanel #models-dialog .panel-buttons {
        layout: horizontal;
        height: auto;
        margin: 0;
    }
    ModelSettingsPanel #models-dialog .panel-buttons Button {
        width: 1fr;
        height: 1;
        min-width: 16;
        padding: 0 1;
        margin: 0 1 0 0;
        border: none !important;
        background: #2b323f;
        color: #dce3ec;
        text-style: bold;
        tint: transparent;
    }
    ModelSettingsPanel #models-dialog .panel-buttons #models-cancel {
        margin: 0;
    }
    ModelSettingsPanel #models-dialog .panel-buttons Button:hover,
    ModelSettingsPanel #models-dialog .panel-buttons Button:focus {
        background: #3b4454;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    ModelSettingsPanel #models-dialog .panel-buttons Button.-active {
        border: none !important;
        tint: transparent;
    }
    ModelSettingsPanel #models-dialog .panel-buttons Button.-primary {
        background: #2f6feb;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    ModelSettingsPanel #models-dialog .panel-buttons Button.-primary:hover,
    ModelSettingsPanel #models-dialog .panel-buttons Button.-primary:focus {
        background: #388bfd;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    ModelSettingsPanel #models-dialog .panel-buttons Button.-success {
        background: #2ea043;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    ModelSettingsPanel #models-dialog .panel-buttons Button.-success:hover,
    ModelSettingsPanel #models-dialog .panel-buttons Button.-success:focus {
        background: #3fb950;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    """

    def __init__(self, settings: Settings, finish) -> None:
        super().__init__(finish)
        self.settings = settings
        self._planner_reasoning = settings.planner.reasoning
        self._deployment_provider = {
            "executor": settings.executor.provider,
            "planner": settings.planner.provider,
        }

    @staticmethod
    def _deployment_choices_for(provider: str, current_model: str) -> tuple[list[tuple[str, str]], str]:
        if provider == "local":
            base = list(LOCAL_MODEL_CHOICES)
            known = LOCAL_MODEL_VALUES
            custom_label = "Custom local model alias / ID..."
        elif provider == "azure":
            base = list(AZURE_MODEL_CHOICES)
            known = AZURE_MODEL_VALUES
            custom_label = "Custom Azure deployment..."
        else:
            return [("Disabled (executor plans)", "__off__")], "__off__"
        base.append((custom_label, "__custom__"))
        value = current_model if current_model in known else "__custom__"
        return base, value

    @staticmethod
    def _preset_summary_text(preset: str) -> str:
        descriptions = {
            "local": f"✓ Ready: Foundry Local on GPU ({LOCAL_DEFAULT_MODEL} for Plan + Run; local port auto-discovered).",
            "azure": f"✓ Ready: Foundry Normal on Azure ({AZURE_DEFAULT_PLANNER} Plan + {AZURE_DEFAULT_EXECUTOR} Run).",
            "hybrid": f"✓ Ready: Hybrid mode (Azure {AZURE_DEFAULT_PLANNER} Plan + Local {LOCAL_DEFAULT_MODEL} Run).",
            "local-fast": f"✓ Ready: Foundry Local Fast ({LOCAL_DEFAULT_MODEL} on GPU; separate planner off).",
        }
        return descriptions.get(preset, "Custom setup: select a preset above or configure each role below.")

    def compose(self) -> ComposeResult:
        active = self.settings.active_preset()
        with VerticalScroll(id="models-dialog"):
            yield Static("Planning and execution models — Ready-to-select presets", classes="panel-title")
            yield Static(
                "Pick a ready preset below or customize each role. Local ports are auto-discovered. Esc closes.",
                classes="panel-hint",
            )
            with Horizontal(classes="preset-bar"):
                yield Button("Foundry Local (GPU)", id="preset-local", compact=True,
                             variant="success" if active == "local" else "default")
                yield Button("Foundry Normal (Azure)", id="preset-azure", compact=True,
                             variant="success" if active == "azure" else "default")
                yield Button("Hybrid (Azure + Local)", id="preset-hybrid", compact=True,
                             variant="success" if active == "hybrid" else "default")
                yield Button("Local Fast (No Plan)", id="preset-local-fast", compact=True,
                             variant="success" if active == "local-fast" else "default")
            yield Select(PRESET_CHOICES, value=active, allow_blank=False, id="preset-select")
            yield Static(self._preset_summary_text(active), id="preset-summary")
            with Horizontal(id="roles-row"):
                for role in ("executor", "planner"):
                    spec = getattr(self.settings, role)
                    with Vertical(classes="role-card", id=f"{role}-card"):
                        yield Static(
                            "Executor (Browser Actions)" if role == "executor" else "Planner (Task Breakdown)",
                            classes="panel-title",
                        )
                        choices = [("Azure Foundry", "azure"), ("Foundry Local", "local")]
                        if role == "planner":
                            choices.insert(0, ("Disabled (executor plans)", "off"))
                        yield Static("Provider", classes="field-label")
                        yield Select(choices, value=spec.provider, allow_blank=False, id=f"{role}-provider")
                        dep_choices, dep_value = self._deployment_choices_for(spec.provider, spec.model)
                        yield Static("Catalog model", classes="field-label")
                        yield Select(dep_choices, value=dep_value, allow_blank=False,
                                     disabled=spec.provider == "off", id=f"{role}-deployment")
                        yield Static("Deployment / model ID", classes="field-label")
                        yield Input(value=spec.model, placeholder="Deployment name, local alias, or model ID",
                                    disabled=spec.provider == "off", id=f"{role}-model")
                        yield Static("Endpoint URL (Azure HTTPS or blank for local)", classes="field-label")
                        yield Input(value=spec.endpoint,
                                    placeholder="Azure HTTPS endpoint; leave blank for local discovery",
                                    disabled=spec.provider == "off", id=f"{role}-endpoint")
            yield Checkbox("Allow downloading uncached local models via Foundry Local CLI",
                           value=self.settings.allow_local_downloads, id="local-downloads")
            yield Static("", id="model-settings-error", classes="event-error")
            with Horizontal(classes="panel-buttons"):
                yield Button("Save", id="models-save", variant="primary", compact=True)
                yield Button("Save and check", id="models-check", variant="success", compact=True)
                yield Button("Cancel", id="models-cancel", compact=True)

    def _current_azure_endpoint(self) -> str:
        for role in ("executor", "planner"):
            value = self.query_one(f"#{role}-endpoint", Input).value.strip()
            if is_valid_azure_endpoint(value):
                return value
        return self.settings.known_azure_endpoint()

    def _candidate_settings(self) -> Settings:
        specs = {}
        for role in ("executor", "planner"):
            previous = getattr(self.settings, role)
            provider = str(self.query_one(f"#{role}-provider", Select).value)
            reasoning = (self._planner_reasoning or AZURE_DEFAULT_REASONING) if (
                role == "planner" and provider == "azure") else (previous.reasoning if role == "executor" else "")
            specs[role] = ModelSettings(
                provider=provider,
                model=self.query_one(f"#{role}-model", Input).value.strip(),
                endpoint=self.query_one(f"#{role}-endpoint", Input).value.strip(),
                reasoning=reasoning,
                max_tokens=previous.max_tokens,
            )
        azure_ep = self.settings.azure_endpoint
        for role in ("executor", "planner"):
            if specs[role].provider == "azure" and is_valid_azure_endpoint(specs[role].endpoint):
                azure_ep = specs[role].endpoint
                break
        return Settings(**specs, planner_scope=self.settings.planner_scope,
                        allow_local_downloads=self.query_one("#local-downloads", Checkbox).value,
                        azure_endpoint=azure_ep)

    def _detect_current_preset(self) -> str:
        return self._candidate_settings().active_preset()

    def _refresh_preset_ui(self) -> None:
        preset = self._detect_current_preset()
        preset_select = self.query_one("#preset-select", Select)
        with self.prevent(Select.Changed):
            if preset_select.value != preset:
                preset_select.value = preset
        self.query_one("#preset-summary", Static).update(Text(self._preset_summary_text(preset)))
        button_map = {
            "preset-local": "local",
            "preset-azure": "azure",
            "preset-hybrid": "hybrid",
            "preset-local-fast": "local-fast",
        }
        for button_id, key in button_map.items():
            btn = self.query_one(f"#{button_id}", Button)
            btn.variant = "success" if preset == key else "default"

    def _apply_preset(self, preset: str) -> None:
        applied = self.settings.with_preset(preset, self._current_azure_endpoint())
        self._planner_reasoning = applied.planner.reasoning
        with self.prevent(Select.Changed, Input.Changed):
            for role in ("executor", "planner"):
                spec = getattr(applied, role)
                provider_select = self.query_one(f"#{role}-provider", Select)
                deployment_select = self.query_one(f"#{role}-deployment", Select)
                model_input = self.query_one(f"#{role}-model", Input)
                endpoint_input = self.query_one(f"#{role}-endpoint", Input)
                if provider_select.value != spec.provider:
                    provider_select.value = spec.provider
                choices, dep_val = self._deployment_choices_for(spec.provider, spec.model)
                if self._deployment_provider.get(role) != spec.provider:
                    self._deployment_provider[role] = spec.provider
                    deployment_select.set_options(choices)
                if deployment_select.value != dep_val:
                    deployment_select.value = dep_val
                disabled = spec.provider == "off"
                deployment_select.disabled = model_input.disabled = endpoint_input.disabled = disabled
                if model_input.value != spec.model:
                    model_input.value = spec.model
                if endpoint_input.value != spec.endpoint:
                    endpoint_input.value = spec.endpoint
                model_input.cursor_position = 0
                endpoint_input.cursor_position = 0
        self.query_one("#model-settings-error", Static).update("")
        self._refresh_preset_ui()

    def _done(self, value, summary: str) -> None:
        if self.answered:
            return
        self.answered = True
        self.finish(value)
        self.remove()

    def action_cancel_settings(self) -> None:
        self._done(None, "Model settings unchanged.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        preset_map = {
            "preset-local": "local",
            "preset-azure": "azure",
            "preset-hybrid": "hybrid",
            "preset-local-fast": "local-fast",
        }
        if event.button.id in preset_map:
            self._apply_preset(preset_map[event.button.id])
            return
        if event.button.id == "models-cancel":
            self._done(None, "Model settings unchanged.")
            return
        candidate = self._candidate_settings()
        try:
            candidate.validate()
        except ValueError as error:
            self.query_one("#model-settings-error", Static).update(Text(str(error)))
            return
        self._done((candidate, event.button.id == "models-check"), "Model settings selected.")

    def on_mount(self) -> None:
        dialog = self.query_one("#models-dialog", VerticalScroll)
        dialog.scroll_home(animate=False)
        for inp in self.query(Input):
            inp.cursor_position = 0
        self.query_one("#executor-provider", Select).focus(scroll_visible=False)
        self.call_after_refresh(lambda: dialog.scroll_home(animate=False))

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "preset-select":
            preset = str(event.value)
            if preset in ("local", "azure", "hybrid", "local-fast") and self._detect_current_preset() != preset:
                self._apply_preset(preset)
            return
        if event.select.id in ("executor-deployment", "planner-deployment"):
            role = event.select.id.removesuffix("-deployment")
            selected = str(event.value)
            if selected not in ("__custom__", "__off__", "Select.BLANK"):
                model = self.query_one(f"#{role}-model", Input)
                endpoint = self.query_one(f"#{role}-endpoint", Input)
                provider = str(self.query_one(f"#{role}-provider", Select).value)
                with self.prevent(Input.Changed):
                    if model.value != selected:
                        model.value = selected
                    if provider == "azure" and not endpoint.value.strip():
                        endpoint.value = self._current_azure_endpoint()
                    elif provider == "local" and endpoint.value.startswith("https://"):
                        endpoint.value = ""
                self._refresh_preset_ui()
            return
        if event.select.id not in ("executor-provider", "planner-provider"):
            return
        role = event.select.id.removesuffix("-provider")
        provider = str(event.value)
        model = self.query_one(f"#{role}-model", Input)
        endpoint = self.query_one(f"#{role}-endpoint", Input)
        deployment = self.query_one(f"#{role}-deployment", Select)
        model.disabled = endpoint.disabled = deployment.disabled = provider == "off"
        current_model = model.value.strip()
        with self.prevent(Select.Changed, Input.Changed):
            if provider == "local":
                if endpoint.value.startswith("https://"):
                    endpoint.value = ""
                if current_model in AZURE_MODEL_VALUES:
                    current_model = LOCAL_DEFAULT_MODEL
                    model.value = current_model
            elif provider == "azure":
                if endpoint.value.startswith("http://"):
                    endpoint.value = ""
                if current_model in LOCAL_MODEL_VALUES:
                    current_model = AZURE_DEFAULT_EXECUTOR if role == "executor" else AZURE_DEFAULT_PLANNER
                    model.value = current_model
                if current_model and not endpoint.value.strip():
                    endpoint.value = self._current_azure_endpoint()
                if role == "planner" and not self._planner_reasoning:
                    self._planner_reasoning = AZURE_DEFAULT_REASONING
            choices, dep_val = self._deployment_choices_for(provider, current_model)
            if self._deployment_provider.get(role) != provider:
                self._deployment_provider[role] = provider
                deployment.set_options(choices)
            if deployment.value != dep_val:
                deployment.value = dep_val
        self._refresh_preset_ui()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id in ("executor-model", "planner-model"):
            role = event.input.id.removesuffix("-model")
            provider = str(self.query_one(f"#{role}-provider", Select).value)
            deployment = self.query_one(f"#{role}-deployment", Select)
            _, dep_val = self._deployment_choices_for(provider, event.value.strip())
            with self.prevent(Select.Changed):
                if deployment.value != dep_val:
                    deployment.value = dep_val
            self._refresh_preset_ui()
        elif event.input.id in ("executor-endpoint", "planner-endpoint"):
            self._refresh_preset_ui()


class TelegramSettingsPanel(InlinePanel):
    """Centered modal-overlay for configuring the 3-level local Telegram Push/Pull channel."""

    BINDINGS = [Binding("escape", "cancel_telegram", "Cancel")]

    DEFAULT_CSS = """
    TelegramSettingsPanel {
        layer: modal;
        dock: top;
        width: 100%;
        height: 100%;
        align: center middle;
        background: rgba(10, 12, 16, 0.84);
        margin: 0;
        padding: 0 1;
        border: none;
    }
    TelegramSettingsPanel #telegram-dialog {
        width: 108;
        max-width: 98%;
        height: auto;
        max-height: 96%;
        padding: 0 2;
        background: #1b1f27;
        border: round #7fa7d9;
        overflow-y: auto;
    }
    TelegramSettingsPanel #telegram-dialog .panel-title {
        color: #9cc0ee;
        text-style: bold;
    }
    TelegramSettingsPanel #telegram-dialog .level-bar {
        layout: horizontal;
        height: auto;
        margin: 0;
    }
    TelegramSettingsPanel #telegram-dialog .level-bar Button {
        width: 1fr;
        height: 1;
        min-width: 16;
        padding: 0;
        margin: 0 1 0 0;
        border: none !important;
        background: #262d38;
        color: #d6deeb;
        text-style: bold;
        tint: transparent;
    }
    TelegramSettingsPanel #telegram-dialog .level-bar #tg-level-tasks {
        margin: 0;
    }
    TelegramSettingsPanel #telegram-dialog .level-bar Button:hover {
        background: #333d4d;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    TelegramSettingsPanel #telegram-dialog .level-bar Button:focus {
        background: #33507a;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    TelegramSettingsPanel #telegram-dialog .level-bar Button.-success {
        background: #2ea043;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    TelegramSettingsPanel #telegram-dialog .level-bar Button.-success:hover,
    TelegramSettingsPanel #telegram-dialog .level-bar Button.-success:focus {
        background: #3fb950;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    TelegramSettingsPanel #telegram-dialog #telegram-level {
        margin: 0;
    }
    TelegramSettingsPanel #telegram-dialog #telegram-level-summary {
        color: #a9c4ad;
        background: #232933;
        padding: 0 1;
        margin: 0;
    }
    TelegramSettingsPanel #telegram-dialog #telegram-cards-row {
        layout: horizontal;
        height: auto;
        margin: 0;
    }
    TelegramSettingsPanel #telegram-dialog .tg-card {
        width: 1fr;
        height: auto;
        padding: 0 1;
        margin: 0 1 0 0;
        background: #222731;
        border: round #3a4659;
    }
    TelegramSettingsPanel #telegram-dialog #telegram-pairing-card {
        margin: 0;
    }
    TelegramSettingsPanel #telegram-dialog .field-label {
        color: #9fb0c7;
        margin: 0;
    }
    TelegramSettingsPanel #telegram-dialog Input {
        margin: 0;
    }
    TelegramSettingsPanel #telegram-dialog .panel-buttons {
        layout: horizontal;
        height: auto;
        margin: 0;
    }
    TelegramSettingsPanel #telegram-dialog .panel-buttons Button {
        width: 1fr;
        height: 1;
        min-width: 16;
        padding: 0 1;
        margin: 0 1 0 0;
        border: none !important;
        background: #2b323f;
        color: #dce3ec;
        text-style: bold;
        tint: transparent;
    }
    TelegramSettingsPanel #telegram-dialog .panel-buttons #telegram-cancel {
        margin: 0;
    }
    TelegramSettingsPanel #telegram-dialog .panel-buttons Button:hover,
    TelegramSettingsPanel #telegram-dialog .panel-buttons Button:focus {
        background: #3b4454;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    TelegramSettingsPanel #telegram-dialog .panel-buttons Button.-primary {
        background: #2f6feb;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    TelegramSettingsPanel #telegram-dialog .panel-buttons Button.-primary:hover,
    TelegramSettingsPanel #telegram-dialog .panel-buttons Button.-primary:focus {
        background: #388bfd;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    TelegramSettingsPanel #telegram-dialog .panel-buttons Button.-success {
        background: #2ea043;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    TelegramSettingsPanel #telegram-dialog .panel-buttons Button.-success:hover,
    TelegramSettingsPanel #telegram-dialog .panel-buttons Button.-success:focus {
        background: #3fb950;
        color: #ffffff;
        border: none !important;
        tint: transparent;
    }
    """

    def __init__(self, settings: TelegramSettings, finish) -> None:
        super().__init__(finish)
        self.settings = settings
        self._explicit_off = False

    def _pairing_status_text(self) -> str:
        bot = self.settings.bot_username or "(verified on 'Save and check')"
        owner = (f"{self.settings.owner_label} (chat_id: {self.settings.chat_id})"
                 if self.settings.chat_id and self.settings.owner_label
                 else f"chat_id: {self.settings.chat_id}" if self.settings.chat_id
                 else "Send /start to your bot in Telegram, then click 'Save and check'")
        return (
            f"• Bot handle: {bot}\n"
            f"• Paired owner: {owner}\n"
            f"• Transport: 100% local HTTP Push (POST) & Pull (getUpdates)\n"
            f"• Concurrency: Single-Active-Run (busy-reject, no task queue)\n"
            f"• Guardrails: /teach, /login, /models, /telegram stay local to TUI"
        )

    def compose(self) -> ComposeResult:
        lvl = self.settings.level
        with VerticalScroll(id="telegram-dialog"):
            yield Static("External Communication Channel — Telegram (3 Capability Levels)", classes="panel-title")
            yield Static(
                "Create a bot in @BotFather, send /start to your bot in Telegram, paste the token, and click 'Save and check'. Esc closes.",
                classes="panel-hint",
            )
            with Horizontal(classes="level-bar"):
                yield Button("Off", id="tg-level-off", compact=True,
                             variant="success" if lvl == "off" else "default")
                yield Button("L1: Task Update", id="tg-level-status", compact=True,
                             variant="success" if lvl == "status" else "default")
                yield Button("L2: HITL", id="tg-level-hitl", compact=True,
                             variant="success" if lvl == "hitl" else "default")
                yield Button("L3: Accept Tasks", id="tg-level-tasks", compact=True,
                             variant="success" if lvl == "tasks" else "default")
            yield Select(TELEGRAM_LEVEL_CHOICES, value=lvl, allow_blank=False, id="telegram-level")
            yield Static(f"✓ {level_description(lvl)}", id="telegram-level-summary")
            with Horizontal(id="telegram-cards-row"):
                with Vertical(classes="tg-card", id="telegram-credentials-card"):
                    yield Static("Bot Credentials (@BotFather)", classes="panel-title")
                    yield Static("Bot Token (<bot_id>:<secret>)", classes="field-label")
                    yield Input(value=self.settings.bot_token,
                                placeholder="Paste token from @BotFather (e.g. 123456:AAH...)",
                                id="telegram-token")
                    yield Static("Authorized Chat ID (optional — auto-discovered on /start)", classes="field-label")
                    yield Input(value=str(self.settings.chat_id),
                                placeholder="Leave blank to auto-detect from /start",
                                id="telegram-chat-id")
                with Vertical(classes="tg-card", id="telegram-pairing-card"):
                    yield Static("Pairing & Surface Guardrails", classes="panel-title")
                    yield Static(self._pairing_status_text(), id="telegram-pairing-status")
            yield Static("", id="telegram-settings-error", classes="event-error")
            with Horizontal(classes="panel-buttons"):
                yield Button("Save", id="telegram-save", variant="primary", compact=True)
                yield Button("Save and check", id="telegram-check", variant="success", compact=True)
                yield Button("Cancel", id="telegram-cancel", compact=True)

    def _set_level(self, level: str) -> None:
        level_select = self.query_one("#telegram-level", Select)
        with self.prevent(Select.Changed):
            if level_select.value != level:
                level_select.value = level
        self.query_one("#telegram-level-summary", Static).update(Text(f"✓ {level_description(level)}"))
        button_map = {
            "tg-level-off": "off",
            "tg-level-status": "status",
            "tg-level-hitl": "hitl",
            "tg-level-tasks": "tasks",
        }
        for button_id, key in button_map.items():
            btn = self.query_one(f"#{button_id}", Button)
            btn.variant = "success" if level == key else "default"
        self.query_one("#telegram-settings-error", Static).update("")

    def _candidate_settings(self) -> TelegramSettings:
        level = str(self.query_one("#telegram-level", Select).value)
        token = self.query_one("#telegram-token", Input).value.strip()
        chat_id = self.query_one("#telegram-chat-id", Input).value.strip()
        same_token = token == self.settings.bot_token.strip()
        same_chat = chat_id == str(self.settings.chat_id).strip()
        return TelegramSettings(
            level=level,
            bot_token=token,
            chat_id=chat_id,
            bot_username=self.settings.bot_username if same_token else "",
            owner_label=self.settings.owner_label if (same_token and same_chat) else "",
        )

    def _done(self, value, summary: str) -> None:
        if self.answered:
            return
        self.answered = True
        self.finish(value)
        self.remove()

    def action_cancel_telegram(self) -> None:
        self._done(None, "Telegram settings unchanged.")

    def on_mount(self) -> None:
        dialog = self.query_one("#telegram-dialog", VerticalScroll)
        dialog.scroll_home(animate=False)
        for inp in self.query(Input):
            inp.cursor_position = 0
        self.query_one("#telegram-level", Select).focus(scroll_visible=False)
        self.call_after_refresh(lambda: dialog.scroll_home(animate=False))

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "telegram-level":
            chosen = str(event.value)
            if chosen == "off":
                self._explicit_off = True
            self._set_level(chosen)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "telegram-token" and event.value.strip():
            current_level = str(self.query_one("#telegram-level", Select).value)
            if current_level == "off" and not self._explicit_off:
                self._set_level("status")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        level_map = {
            "tg-level-off": "off",
            "tg-level-status": "status",
            "tg-level-hitl": "hitl",
            "tg-level-tasks": "tasks",
        }
        if event.button.id in level_map:
            chosen = level_map[event.button.id]
            self._explicit_off = chosen == "off"
            self._set_level(chosen)
            return
        if event.button.id == "telegram-cancel":
            self._done(None, "Telegram settings unchanged.")
            return
        candidate = self._candidate_settings()
        try:
            candidate.validate()
        except ValueError as error:
            self.query_one("#telegram-settings-error", Static).update(Text(str(error)))
            return
        self._done((candidate, event.button.id == "telegram-check"), "Telegram settings selected.")


class BrowserApp(App):
    TITLE = "Communication Harness"
    CSS = """
    Screen { background: #101114; color: #d1d6df; layers: base modal; }
    #body { height: 1fr; min-height: 0; }
    #settings { height: 1; padding: 0 2; background: #191c22; color: #bec6d0; }
    #timeline { height: 1fr; background: #101114; scrollbar-size: 1 1; }
    #thread { width: 100%; height: auto; padding: 1 3; }
    .role { height: auto; color: #aab5c3; text-style: bold; margin: 1 0 0 0; }
    .role-user { color: #adbbe2; }
    .message { height: auto; padding: 0 1; margin: 0 0 1 0; color: #d6dae2; }
    Markdown.message { background: #101114; padding: 0 1; margin: 0 0 1 0; }
    .message-user { background: #191d24; padding: 1 2; }
    .event { height: auto; padding: 0 1; color: #aeb7c5; }
    .event-error { color: #d99c98; }
    .event-done { color: #a9c4ad; margin: 0 0 1 0; }
    .model-notes { height: auto; margin: 0 0 1 0; background: #181a20; color: #abb4c2; }
    .model-notes Static { height: auto; max-height: 18; overflow-y: auto; padding: 0 1; }
    #actions { height: 3; padding: 0 2; margin: 1 0 0 0; background: #14161b; }
    #actions Button { width: 20; margin-right: 1; background: #252b34; color: #c8d1dc; border: none; }
    #stop { color: #dab0ab; }
    #composer-row { height: 3; max-height: 12; padding: 0 2; background: #14161b; }
    #composer { width: 100%; height: 100%; background: #21252d; border: solid #4c5766; color: #f0f2f5; }
    #composer:disabled { background: #15171c; border: dashed #3a414c; color: #5d6572; }
    #footer-row { height: 1; background: #14161b; }
    #hint { width: 1fr; padding: 0 2; color: #939eae; }
    #usage { width: auto; padding: 0 2; color: #b1c3d8; }
    #plan { height: auto; max-height: 10; padding: 0 3; background: #161a21; color: #c9d3df;
            border-top: solid #2c3440; display: none; }
    """
    BINDINGS = [
        Binding("ctrl+x", "interrupt", "Stop", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("f1", "help", "Help"),
    ]

    def __init__(self, model_factory=None, log: SessionLog | None = None, memory: Memory | None = None,
                 playbooks: Playbooks | None = None, workspace: Workspace | None = None,
                 planner_factory=None, settings: Settings | None = None,
                 settings_path: Path | None = None, telegram_channel: TelegramChannel | None = None,
                 telegram_settings: TelegramSettings | None = None,
                 telegram_settings_path: Path | None = None) -> None:
        super().__init__()
        self.model_factory = model_factory
        self.planner_factory = planner_factory
        self.settings_path = settings_path or SETTINGS_PATH
        self.telegram_settings_path = telegram_settings_path or TELEGRAM_SETTINGS_PATH
        self.settings_error = ""
        try:
            self.model_settings = settings or Settings.load(self.settings_path)
        except ValueError as error:
            self.model_settings = Settings()
            self.settings_error = str(error)
        try:
            tg_cfg = telegram_settings or (telegram_channel.settings if telegram_channel else
                                           TelegramSettings.load(self.telegram_settings_path))
        except ValueError as error:
            tg_cfg = TelegramSettings()
            self.settings_error = f"{self.settings_error} {error}".strip()
        self.telegram = telegram_channel or TelegramChannel(tg_cfg)
        self.model_status = "not checked"
        self.session_log = log or SessionLog()
        self.memory = memory or Memory()
        self.playbooks = playbooks or Playbooks()
        self.workspace = workspace or Workspace()
        self.usage = Usage()
        self.teaching: Teaching | None = None
        self.agent: BrowserAgent | None = None
        self.busy = False
        self.cancel = Event()
        self.reasoning = ""
        self.answer = ""
        self.tokens = {"input": 0, "output": 0}
        self._live_answer: Markdown | None = None
        self._notes: Static | None = None
        self._render_pending = False
        self._task_source = "TUI"
        self._tg_polling = False
        self._tg_checking = False

    def _header_text(self) -> Text:
        title = Text("communication harness", style="bold #bfc7d6")
        title.append(
            f"   {self.model_settings.summary()}  |  {self.telegram.settings.summary()}  |  {self.model_status}",
            style="#b2bdcc",
        )
        return title

    def _refresh_header(self) -> None:
        self.query_one("#settings", Static).update(self._header_text())

    def compose(self) -> ComposeResult:
        with Vertical(id="body"):
            yield Static(self._header_text(), id="settings")
            with VerticalScroll(id="timeline"):
                yield Vertical(id="thread")
            yield Static("", id="plan")
            with Horizontal(id="actions"):
                yield Button("Close browser", id="close-browser")
                yield Button("Models", id="models")
                yield Button("Telegram", id="telegram")
                yield Button("Stop", id="stop", disabled=True)
            with Horizontal(id="composer-row"):
                yield ComposerInput(placeholder=IDLE_PLACEHOLDER, id="composer")
            with Horizontal(id="footer-row"):
                yield Static("Enter sends  |  Up/Down last 3 prompts  |  Ctrl+J new line  |  F1 help", id="hint")
                yield Static("In --  Out --", id="usage")

    def on_mount(self) -> None:
        try:
            if enable_shift_enter():
                self.query_one("#hint", Static).update(
                    "Enter sends  |  Up/Down last 3 prompts  |  Shift+Enter new line  |  F1 help"
                )
        except (OSError, RuntimeError, ValueError) as error:
            self.notify(str(error), severity="warning")
        recent_fn = getattr(self.memory, "recent_prompts", None)
        if callable(recent_fn):
            self.query_one("#composer", ComposerInput).set_history(recent_fn())
        self._chat_log("Assistant", "Hi! Tell me what to do and I will use the browser.\n\n" + HELP)
        if self.settings_error:
            self._log("ERROR", self.settings_error + " Use /models or /telegram to repair the configuration.")
        elif not self.model_factory and not self.model_settings.executor.model:
            self._log("STEP", "No executor configured. Use /models before running a browser task.")
        self.set_interval(1.5, self._on_telegram_timer)
        self.query_one("#composer", ComposerInput).focus()

    # ---------- rendering ----------
    def _scroll_if_at_end(self) -> None:
        timeline = self.query_one("#timeline", VerticalScroll)
        if timeline.is_vertical_scroll_end or timeline.max_scroll_y == 0:
            timeline.anchor()

    def _log(self, label: str, message: str) -> None:
        style = {"ERROR": "#d99c98", "DONE": "#a9c4ad", "TOOL": "#c9b98f"}.get(label, "#aab4c2")
        marker = {"ERROR": "!", "TOOL": ">", "DONE": "+"}.get(label, "-")
        classes = "event" + {"ERROR": " event-error", "DONE": " event-done"}.get(label, "")
        self._scroll_if_at_end()
        self.query_one("#thread", Vertical).mount(Static(Text(f"{marker} {message}", style=style), classes=classes))

    def _chat_log(self, label: str, message: str) -> Static | Markdown:
        user = label == "You" or label.startswith("You ")
        # Textual's Markdown widget: its blocks are selectable (right-click copies) and links open on click.
        body = (Static(Text(message), classes="message message-user") if user
                else Markdown(self._fix_links(message), classes="message", open_links=True))
        self._scroll_if_at_end()
        self.query_one("#thread", Vertical).mount(
            Static(Text(label), classes="role" + (" role-user" if user else "")), body,
        )
        return body

    def _new_block(self) -> None:
        """Closes the current text/reasoning block (before a tool call or a plan update)."""
        self._flush_answer()
        self.answer, self.reasoning = "", ""
        self._live_answer, self._notes = None, None

    def _event(self, kind: str, text: str) -> None:
        if kind == "reasoning":
            self.reasoning = (self.reasoning + text)[-6000:]
            if self._notes is None:
                self._notes = Static(Text(self.reasoning))
                self.query_one("#thread", Vertical).mount(
                    Collapsible(self._notes, title="Model reasoning", collapsed=True, classes="model-notes")
                )
            self._notes.update(Text(self.reasoning))
        elif kind == "answer":
            self.answer += text
            if self._live_answer is None:
                if not self.answer.strip():
                    return
                self._live_answer = self._chat_log("Assistant", "")
            if not self._render_pending:
                self._render_pending = True
                self.set_timer(0.05, self._flush_answer)
        elif kind == "tool":
            self._new_block()
            self._log("TOOL", text)
        elif kind == "tool_result":
            first = next((line for line in text.splitlines() if "Page URL" in line or line.startswith("Error")), "")
            if first:
                self._log("STEP", first.strip("- "))
        elif kind == "plan":
            # Fixed panel above the buttons: the user sees the plan and its progress at all times.
            self._new_block()
            panel = self.query_one("#plan", Static)
            panel.update(Text(text))
            panel.display = bool(text)
        elif kind == "note":
            self._log("STEP", text)

    def _flush_answer(self) -> None:
        self._render_pending = False
        if self._live_answer and self.answer:
            self._scroll_if_at_end()
            self._live_answer.update(self._fix_links(self.answer))

    def _record_usage(self, turn: Turn) -> None:
        self.tokens["input"] += turn.input_tokens or 0
        self.tokens["output"] += turn.output_tokens or 0
        # "ctx" is the latest prompt size; "In"/"Out" include every configured model role.
        self.query_one("#usage", Static).update(self.usage.footer(turn.input_tokens or 0))

    def copy_selection(self, text: str = "") -> bool:
        """Copies the selected text (of the composer, or of the conversation) to the clipboard, if any."""
        text = text or (self.screen.get_selected_text() or "")
        if not text.strip():
            return False
        try:
            write_clipboard_text(text)
        except (OSError, RuntimeError):
            self.copy_to_clipboard(text)  # OSC 52: the terminal copies it
        self.screen.clear_selection()
        self.notify(f"Copied {len(text):,} characters.")
        return True

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button == 3 and self.copy_selection():
            event.stop()

    def _fix_links(self, text: str) -> str:
        """Links to workspace files become file:/// links that open; other relative links become plain text."""
        def fix(match: re.Match) -> str:
            label, target = match.group(1), match.group(2)
            if re.match(r"^(https?|mailto|file):", target, re.I):
                return match.group(0)
            folder = self.workspace.folder.resolve() if self.workspace.folder else None
            path = (folder / target.strip("<>")).resolve() if folder else None
            if path and path.is_relative_to(folder) and path.is_file():
                return f"[{label}]({path.as_uri()})"
            return f"`{label}`" if label == target else f"{label} (`{target}`)"

        return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", fix, text)

    def _new_agent(self) -> BrowserAgent:
        status = lambda text: self.call_from_thread(self._model_feedback, text)
        model = self.model_factory() if self.model_factory else self.model_settings.create_model(
            "executor", status, self.cancel)
        try:
            planner = self.planner_factory() if self.planner_factory else None
            if not self.model_factory and not self.planner_factory:
                planning_model = self.model_settings.create_model("planner", status, self.cancel)
                planner = planner_module.Planner(planning_model, scope=self.model_settings.planner_scope) \
                    if planning_model else None
        except (ValueError, OSError, RuntimeError, OpenAIError, AzureError):
            model.close()
            raise
        return BrowserAgent(model, log=self.session_log, memory=self.memory,
                            playbooks=self.playbooks, workspace=self.workspace, usage=self.usage,
                            approve=self._approve, ask_user=self._ask_user, planner=planner)

    def _model_feedback(self, text: str) -> None:
        self.model_status = text
        self._refresh_header()
        self._log("STEP", text)

    def _show_model_settings(self) -> None:
        if self.busy:
            return
        for panel in self.query(ModelSettingsPanel):
            if not panel.answered:
                panel.focus()
                return
        self.mount(ModelSettingsPanel(self.model_settings, self._save_model_settings))

    def _show_telegram_settings(self) -> None:
        if self.busy:
            return
        for panel in self.query(TelegramSettingsPanel):
            if not panel.answered:
                panel.focus()
                return
        if self.telegram_settings_path.is_file():
            try:
                self.telegram.settings = TelegramSettings.load(self.telegram_settings_path)
                self._refresh_header()
            except ValueError:
                pass
        self.mount(TelegramSettingsPanel(self.telegram.settings, self._save_telegram_settings))

    def _save_telegram_settings(self, result) -> None:
        if result is None:
            self.query_one("#composer", ComposerInput).focus()
            return
        candidate, check = result
        try:
            candidate.save(self.telegram_settings_path)
            self.telegram.settings = candidate
            self.settings_error = ""
            self._refresh_header()
            self._log(
                "STEP",
                f"Telegram settings saved ({candidate.summary()}). "
                "telegram.json is local-only and ignored by Git.",
            )
            if check:
                self.cancel = Event()
                self._tg_checking = True
                self._set_busy(True)
                self._check_telegram(candidate)
            else:
                self.query_one("#composer", ComposerInput).focus()
        except (ValueError, OSError, RuntimeError) as error:
            self._log("ERROR", f"Could not save Telegram settings: {error}")

    @work(thread=True, exit_on_error=False)
    def _check_telegram(self, candidate: TelegramSettings) -> None:
        try:
            updated, status_message = self.telegram.verify_and_pair(candidate)
            updated.save(self.telegram_settings_path)
            self.call_from_thread(self._refresh_header)
            self.call_from_thread(self._finished, status_message, False)
        except (ValueError, OSError, RuntimeError) as error:
            self.call_from_thread(self._refresh_header)
            self.call_from_thread(self._finished, f"Telegram check failed: {error}", True)
        finally:
            self._tg_checking = False

    def _telegram_command(self, argument: str) -> None:
        arg = argument.strip().lower()
        if not arg:
            self._show_telegram_settings()
            return
        aliases = {
            "0": "off", "off": "off",
            "1": "status", "status": "status", "l1": "status",
            "2": "hitl", "hitl": "hitl", "l2": "hitl",
            "3": "tasks", "tasks": "tasks", "l3": "tasks",
        }
        if arg not in aliases:
            self._log("ERROR", "Use /telegram (opens dialog) or /telegram off|status|hitl|tasks")
            return
        target_level = aliases[arg]
        candidate = TelegramSettings(
            level=target_level,
            bot_token=self.telegram.settings.bot_token,
            chat_id=self.telegram.settings.chat_id,
            bot_username=self.telegram.settings.bot_username,
            owner_label=self.telegram.settings.owner_label,
        )
        try:
            candidate.validate()
        except ValueError as error:
            self._log("ERROR", str(error))
            self._show_telegram_settings()
            return
        self._save_telegram_settings((candidate, False))

    def _on_telegram_timer(self) -> None:
        if not self.telegram.settings.needs_polling or self._tg_polling or self._tg_checking:
            return
        if any(not panel.answered for panel in self.query(TelegramSettingsPanel)):
            return
        # When waiting on an inline HITL dialog, _wait_dialog already polls every 0.2 s.
        if self.telegram._active_hitl is not None:
            return
        self._tg_polling = True
        self._poll_telegram_worker()

    @work(thread=True, exit_on_error=False)
    def _poll_telegram_worker(self) -> None:
        try:
            plan_text = self.agent.plan.render_tui() if (self.agent and self.agent.plan.items) else ""
            events_out = self.telegram.poll_events(
                busy=self.busy,
                teaching=self.teaching is not None,
                plan_text=plan_text,
            )
            for tg_event in events_out:
                self.call_from_thread(self._handle_telegram_event, tg_event)
        except (ValueError, OSError, RuntimeError):
            pass
        finally:
            self._tg_polling = False

    def _process_telegram_updates(self) -> list[dict]:
        """Synchronously polls and dispatches pending Telegram updates on the main thread."""
        plan_text = self.agent.plan.render_tui() if (self.agent and self.agent.plan.items) else ""
        events_out = self.telegram.poll_events(
            busy=self.busy,
            teaching=self.teaching is not None,
            plan_text=plan_text,
        )
        for tg_event in events_out:
            self._handle_telegram_event(tg_event)
        return events_out

    def _handle_telegram_event(self, tg_event: dict) -> None:
        kind = tg_event.get("type")
        if kind == "paired":
            try:
                self.telegram.settings.save(self.telegram_settings_path)
            except (ValueError, OSError):
                pass
            self._refresh_header()
            self._log(
                "DONE",
                f"Telegram paired with {tg_event.get('owner_label')} (chat_id: {tg_event.get('chat_id')}).",
            )
            return
        if kind == "stop":
            self._log("STEP", f"Stop requested via Telegram ({tg_event.get('sender', 'Telegram')}).")
            self.action_interrupt()
            return
        if kind == "blocked_command":
            self._log("STEP", f"Blocked TUI-only command '{tg_event.get('command')}' sent from Telegram.")
            return
        if kind == "level_rejected":
            lvl = TELEGRAM_LEVEL_LABELS.get(str(tg_event.get("level")), str(tg_event.get("level")))
            self._log("STEP", f"Ignored Telegram task in {lvl} mode (switch to Level 3 to accept tasks).")
            return
        if kind == "busy_rejected":
            reason = tg_event.get("reason", "busy")
            self._log("STEP", f"Rejected Telegram task while {reason} (Single-Active-Run): {tg_event.get('prompt', '')[:80]}")
            return
        if kind == "hitl_approval":
            approved = bool(tg_event.get("approved"))
            for panel in reversed(list(self.query(ApprovalPanel))):
                if not panel.answered:
                    summary = (f"✓ Approved [Telegram]: {panel.action_text[:160]}"
                               if approved else f"✗ Denied [Telegram]: {panel.action_text[:160]}")
                    panel._done(approved, summary)
                    break
            return
        if kind == "hitl_question":
            ans = tg_event.get("answer")
            for panel in reversed(list(self.query(QuestionPanel))):
                if not panel.answered:
                    summary = (f"? {panel.question[:160]}\n→ {ans} [Telegram]"
                               if ans is not None else f"? {panel.question[:160]}\n→ (answer in the chat) [Telegram]")
                    panel._done(ans, summary)
                    break
            return
        if kind == "task":
            prompt = str(tg_event.get("prompt") or "").strip()
            sender = str(tg_event.get("sender") or "Telegram")
            if not prompt:
                return
            if self.busy or self.teaching is not None:
                self._log("STEP", f"Rejected Telegram task while busy (Single-Active-Run): {prompt[:80]}")
                return
            self.query_one("#timeline", VerticalScroll).anchor()
            self._chat_log(f"You [Telegram · {sender}]", prompt)
            self._start(prompt, source=f"Telegram · {sender}")

    def _save_model_settings(self, result) -> None:
        if result is None:
            self.query_one("#composer", ComposerInput).focus()
            return
        candidate, check = result
        try:
            candidate.save(self.settings_path)
            if self.agent:
                self.agent.model.close()
                if self.agent.planner:
                    self.agent.planner.close()
                self.agent.plan.cancel("Model settings changed.")
                self.agent = None
            self.model_settings = candidate
            self.settings_error = ""
            self._model_feedback("Settings saved; fresh agent context on the next task.")
            self._log("STEP", "Environment overrides apply at the next launch. settings.json is ignored by Git.")
            if check:
                self.cancel = Event()
                self._set_busy(True)
                self._check_models()
        except (ValueError, OSError, RuntimeError, OpenAIError, AzureError) as error:
            self._log("ERROR", f"Could not apply model settings: {error}")

    @work(thread=True, exit_on_error=False)
    def _check_models(self) -> None:
        models = []
        status = lambda text: self.call_from_thread(self._model_feedback, text)
        try:
            for role in ("executor", "planner"):
                model = self.model_settings.create_model(role, status, self.cancel)
                if model is None:
                    continue
                models.append(model)
                if role == "executor":
                    tools = [{"type": "function", "function": {"name": "diagnostic_ping",
                              "description": "Return a diagnostic result. This does not perform any action.",
                              "parameters": {"type": "object", "properties": {
                                  "ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}}}]
                    turn = model.complete([{"role": "user", "content": "Call diagnostic_ping with ok=true."}], tools,
                                          lambda kind, text: None, self.cancel,
                                          tool_choice={"type": "function", "function": {"name": "diagnostic_ping"}})
                    if (len(turn.tool_calls) != 1 or turn.tool_calls[0].name != "diagnostic_ping"
                            or json.loads(turn.tool_calls[0].arguments) != {"ok": True}):
                        raise RuntimeError("The executor did not return the required diagnostic tool arguments.")
                else:
                    planning = planner_module.Planner(model)
                    planning.cancel = self.cancel
                    decision, turn, _ = planning.plan("Plan one read-only task to open https://example.com/.")
                    if not decision or "plan" not in decision:
                        raise RuntimeError("The planner did not return a valid plan. Choose a more capable model.")
                    from plan import Plan
                    validation = Plan().apply(json.dumps({"action": "plan", "items": decision["plan"]}))
                    if validation.startswith("Error"):
                        raise RuntimeError("The planner returned invalid task metadata: " + validation)
                self.usage.add(model.usage_name, turn.input_tokens, turn.output_tokens, turn.cached_tokens)
                self.call_from_thread(self._record_usage, turn)
                status(f"{role.title()} inference check passed.")
            self.call_from_thread(self._finished, "Model checks passed. No browser action was performed.")
        except (ValueError, OSError, RuntimeError, OpenAIError, AzureError) as error:
            self.call_from_thread(self._model_feedback, "Model check failed; no cloud fallback was attempted.")
            self.call_from_thread(self._finished, str(error), True)
        finally:
            for model in models:
                model.close()

    def _wait_dialog(self, make_panel, note: str, on_open=None):
        """Runs in the agent thread: shows an inline panel and waits for its answer (TUI or Telegram)."""
        answer: dict = {}
        done = Event()
        resolved_by = {"surface": "TUI"}

        def finish(result) -> None:
            answer["value"] = result
            done.set()

        panel = make_panel(finish)
        self.call_from_thread(self._show_panel_inline, panel, note)
        if on_open is not None:
            on_open()
        try:
            while not done.wait(0.2):
                if self.cancel.is_set():
                    self.call_from_thread(self._close_dialogs)
                    self.telegram.close_active_hitl("⏹ Cancelled.")
                    done.wait(2)
                    return answer.get("value")
                if self.telegram.settings.can_hitl and self.telegram._active_hitl:
                    plan_text = self.agent.plan.render_tui() if (self.agent and self.agent.plan.items) else ""
                    for tg_event in self.telegram.poll_events(
                        busy=True,
                        teaching=self.teaching is not None,
                        plan_text=plan_text,
                    ):
                        if tg_event.get("type") == "stop":
                            resolved_by["surface"] = "Telegram"
                            self.call_from_thread(self.action_interrupt)
                        elif tg_event.get("type") == "hitl_approval" and isinstance(panel, ApprovalPanel):
                            resolved_by["surface"] = "Telegram"
                            approved = bool(tg_event.get("approved"))
                            summary = (f"✓ Approved [Telegram]: {panel.action_text[:160]}"
                                       if approved else f"✗ Denied [Telegram]: {panel.action_text[:160]}")
                            self.call_from_thread(panel._done, approved, summary)
                        elif tg_event.get("type") == "hitl_question" and isinstance(panel, QuestionPanel):
                            resolved_by["surface"] = "Telegram"
                            ans = tg_event.get("answer")
                            summary = (f"? {panel.question[:160]}\n→ {ans} [Telegram]"
                                       if ans is not None else f"? {panel.question[:160]}\n→ (answer in the chat) [Telegram]")
                            self.call_from_thread(panel._done, ans, summary)
                        else:
                            self.call_from_thread(self._handle_telegram_event, tg_event)
            return answer.get("value")
        finally:
            if self.telegram._active_hitl and resolved_by["surface"] == "TUI":
                val = answer.get("value")
                if isinstance(panel, ApprovalPanel):
                    note_text = "✓ Approved in TUI." if val else "✗ Denied in TUI."
                else:
                    note_text = f"→ Answered in TUI: {val}" if val is not None else "→ Deferred to chat in TUI."
                self.telegram.close_active_hitl(note_text)

    def _show_panel_inline(self, panel: InlinePanel, note: str) -> None:
        self._new_block()
        self._log("STEP", note)
        self.query_one("#timeline", VerticalScroll).anchor()
        self.query_one("#thread", Vertical).mount(panel)
        self.bell()

    def _close_dialogs(self) -> None:
        for panel in self.query(InlinePanel):
            if not panel.answered:
                panel.cancel()

    def _approve(self, action: str) -> bool:
        on_open = (lambda: self.telegram.send_approval_request(action)) if self.telegram.settings.can_hitl else None
        return bool(self._wait_dialog(lambda finish: ApprovalPanel(action, finish),
                                      f"Waiting for your approval: {action[:120]}", on_open=on_open))

    def _ask_user(self, question: str, options: list[str]) -> str | None:
        on_open = (lambda: self.telegram.send_question_request(question, options)) if self.telegram.settings.can_hitl else None
        return self._wait_dialog(lambda finish: QuestionPanel(question, options, finish),
                                 f"Question: {question[:120]}", on_open=on_open)

    def _show_usage(self) -> None:
        self._chat_log("Usage", f"**This session, per model**\n\n{self.usage.report()}\n\n"
                       "Cloud costs depend on your deployment prices; missing cost is unknown, not zero. "
                       "Local inference uses your device. All sessions: `python report.py`.")

    # ---------- execution ----------
    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        composer = self.query_one("#composer", ComposerInput)
        composer.disabled = busy
        # While the agent works the box says so (and why typing does nothing); the idle hint comes back after.
        composer.placeholder = ("Working... you can type again when it finishes (Ctrl+X stops)" if busy
                                else IDLE_PLACEHOLDER)
        self.query_one("#stop", Button).disabled = not busy
        self.query_one("#close-browser", Button).disabled = busy
        self.query_one("#models", Button).disabled = busy
        self.query_one("#telegram", Button).disabled = busy
        if not busy:
            composer.focus()

    def _start(self, prompt: str, source: str = "TUI") -> None:
        self.cancel = Event()
        self.answer, self.reasoning = "", ""
        self._live_answer, self._notes = None, None
        self._task_source = source
        self._set_busy(True)
        self._run_agent(prompt)

    @work(thread=True, exit_on_error=False)
    def _run_agent(self, prompt: str) -> None:
        def emit(kind: str, text: str) -> None:
            self.call_from_thread(self._event, kind, text)
            if kind == "plan":
                self.telegram.notify_plan_updated(text)

        self.telegram.notify_task_started(prompt, source=self._task_source)
        try:
            if self.agent is None:
                self.agent = self._new_agent()
            self.agent.on_usage = lambda turn: self.call_from_thread(self._record_usage, turn)
            result_text = self.agent.run(prompt, emit, self.cancel)
            pending = bool(self.agent.plan.pending())
            waiting = any(item["status"] == "question" for item in self.agent.plan.items)
            stopped = any(item["status"] == "cancelled" for item in self.agent.plan.items)
            failed_task = any(item["status"] == "failed" for item in self.agent.plan.items)
            final_msg = ("Pending tasks remain; see the plan." if pending
                         else "Waiting for your answer." if waiting
                         else "Stopped." if stopped
                         else "Task failed; see the plan." if failed_task
                         else "Done.")
            is_failed = pending or failed_task
            self.telegram.notify_task_finished(final_msg, answer=result_text or self.answer, failed=is_failed)
            self.call_from_thread(self._finished, final_msg, is_failed)
        except (ValueError, OSError, RuntimeError, InterruptedError, OpenAIError, AzureError) as error:
            if self.cancel.is_set() and self.agent:
                self.agent.plan.cancel()
                plan_text = self.agent.plan.render_tui()
                self.call_from_thread(self._event, "plan", plan_text)
                self.telegram.notify_plan_updated(plan_text)
            self.telegram.notify_task_finished(str(error), answer=self.answer, failed=True)
            self.call_from_thread(self._finished, str(error), True)

    def _finished(self, message: str, failed: bool = False) -> None:
        self._flush_answer()
        self._set_busy(False)
        self._log("ERROR" if failed else "DONE", message)

    @work(thread=True, exit_on_error=False)
    def _close_browser(self) -> None:
        if self.teaching:
            # Closing the browser ends the recording; nothing is saved.
            self.teaching = None
            self.call_from_thread(self._show_panel, "")
            self.call_from_thread(self._log, "STEP", "Learning mode stopped (browser closed); nothing saved.")
        result = close_browser()
        self.call_from_thread(self._log, "DONE", result.splitlines()[0] if result else "Browser closed.")

    @work(thread=True, exit_on_error=False)
    def _login(self, url: str) -> None:
        result = navigate(url)
        error = navigation_error(result)
        self.session_log.write("login", url=url, error=error or None)
        if error:
            self.call_from_thread(self._log, "ERROR", f"Could not open {url}: {error}")
            return
        self.call_from_thread(
            self._log, "DONE",
            f"Opened {url}. Sign in to your account in the browser window. The login is saved in the "
            ".profile/ profile and stays for the next sessions. Then just make your request here.",
        )

    # ---------- learning mode ----------
    def _show_panel(self, text: str) -> None:
        panel = self.query_one("#plan", Static)
        panel.update(Text(text))
        panel.display = bool(text)

    @work(thread=True, exit_on_error=False)
    def _teach_start(self, task: str) -> None:
        try:
            teaching = self.playbooks.start(task)
        except NoStartPage:
            # No page open yet: open the site the task names (a site the memory or a guide already knows).
            url = self._guess_site(task)
            if not url:
                self.call_from_thread(self._log, "ERROR", "Open the page where the task starts (e.g. /login "
                                      "<site>, or ask \"open <site>\"), then /teach <task> again.")
                return
            self.call_from_thread(self._log, "STEP", f"Opening {url} to start the recording...")
            error = navigation_error(navigate(url))
            if error:
                self.call_from_thread(self._log, "ERROR", f"Could not open {url}: {error}")
                return
            try:
                teaching = self.playbooks.start(task)
            except (RuntimeError, OSError, ValueError) as again:
                self.call_from_thread(self._log, "ERROR", str(again))
                return
        except (RuntimeError, OSError, ValueError) as error:
            self.call_from_thread(self._log, "ERROR", str(error))
            return
        self.teaching = teaching
        self.session_log.write("teach_start", task=teaching.task, url=teaching.start_url)
        self.call_from_thread(self._show_panel, f"● Recording: {teaching.task}\n"
                              "Do the task in the browser. Type in the box to explain a step (optional). "
                              "/teach again finishes and saves  |  /teach cancel discards.")
        self.call_from_thread(
            self._log, "DONE",
            f"Recording from {teaching.start_url}. Do the task in the browser (typed values are not saved). "
            "Anything you type here now is a note about the step you are on. When you reach the result, "
            "type /teach to finish and save.",
        )

    def _guess_site(self, task: str) -> str:
        """A site named in the task (e.g. "linkedin") that the memory or a taught guide already has."""
        words = set(re.findall(r"[a-z0-9]{3,}", task.lower()))
        hosts = site_keys(self.memory.long_term()) | {book.site for book in self.playbooks.all()}
        for host in sorted(hosts):
            labels = [label for label in host.split(".") if label not in ("www", "web", "com", "br", "m")]
            if any(label in words for label in labels):
                return f"https://{host}/"
        return ""

    @work(thread=True, exit_on_error=False)
    def _teach_stop(self, save: bool, note: str = "") -> None:
        teaching, self.teaching = self.teaching, None
        if save:
            self.call_from_thread(self._log, "STEP", "Writing the guide from your demonstration...")
        try:
            book = self.playbooks.stop(teaching, save=save, note=note, write_guide=self._write_guide)
        except (RuntimeError, OSError, ValueError) as error:
            self.call_from_thread(self._log, "ERROR", f"Could not stop the recording: {error}")
            return
        finally:
            self.call_from_thread(self._show_panel, "")
        self.session_log.write("teach_stop", task=teaching.task, saved=bool(book), notes=len(teaching.notes),
                               path=str(book.path) if book else None)
        if not book:
            self.call_from_thread(self._log, "DONE", "Recording discarded.")
            return
        self.call_from_thread(
            self._chat_log, "Learned",
            f"Saved **{book.task}** (`{book.path.relative_to(book.path.parents[2])}`). Next time a request "
            f"mentions {book.site}, the model gets this guide as a map (it still navigates and checks by "
            f"itself):\n\n{book.section('Guide')}",
        )

    def _write_guide(self, prompt: str) -> str:
        """One short call without tools (runs in the worker thread)."""
        if self.agent is None:
            self.agent = self._new_agent()
        turn = self.agent.model.complete([{"role": "user", "content": prompt}], [], lambda kind, text: None, Event())
        self.usage.add(self.agent.model_name, turn.input_tokens, turn.output_tokens, turn.cached_tokens)
        self.call_from_thread(self._record_usage, turn)
        return turn.content

    @work(thread=True, exit_on_error=False)
    def _teach_note(self, text: str) -> None:
        teaching = self.teaching
        if teaching:
            note = self.playbooks.add_note(teaching, text)
            self.call_from_thread(self._log, "DONE", f"Note saved {note}")

    def _teach(self, argument: str) -> None:
        """One command: /teach <task> starts; while recording, typed text is a note and /teach finishes."""
        word, _, rest = argument.strip().partition(" ")
        word = word.lower()
        if self.teaching:
            if word == "cancel":
                self._teach_stop(save=False)
            elif word == "note" and rest.strip():  # old form, still accepted
                self._teach_note(rest.strip())
            else:  # /teach, /teach done [note] or /teach <final note>: finish and save
                self._teach_stop(save=True, note=rest if word == "done" else argument.strip())
        elif word in ("done", "cancel", "note") or not argument.strip():
            self._log("ERROR", "Use /teach <task> to start, e.g. /teach add a LinkedIn contact")
        else:
            self._teach_start(argument.strip())

    def _list_playbooks(self) -> None:
        books = self.playbooks.all()
        if not books:
            self._chat_log("Playbooks", "No taught tasks yet. Open the start page and use `/teach <task>`.")
            return
        lines = [f"- **{book.task}** ({book.site}) - `{book.path.name}`" for book in books]
        self._chat_log("Playbooks", f"Folder: `{self.playbooks.folder}`\n\n" + "\n".join(lines))

    def _show_workspace(self) -> None:
        self._chat_log("Workspace", f"Session `{self.workspace.id}` - folder `{self.workspace.folder}`\n\n"
                       f"```\n{self.workspace.list()}\n\n{self.workspace.sessions()}\n```")

    # ---------- input ----------
    def action_interrupt(self) -> None:
        if self.busy:
            self.cancel.set()
            self._log("STEP", "Stop requested; waiting for the current step to finish.")

    def action_help(self) -> None:
        self._chat_log("Commands", HELP + f"\n\nSession log: `{self.session_log.path}`\n\n"
                       f"Workspace: `{self.workspace.folder}`")

    def _clear_context(self) -> None:
        """Resets conversation, plan and counters. Long-term memory and login stay."""
        self.query_one("#thread", Vertical).remove_children()
        if self.agent:
            self.agent.clear()
        self.query_one("#plan", Static).display = bool(self.teaching)  # the recording banner stays
        self.tokens = {"input": 0, "output": 0}
        self._log_session_usage()
        self.usage.reset()
        self.query_one("#usage", Static).update("ctx 0  In 0  Out 0")
        self.workspace = self.workspace.new()
        if self.agent:
            self.agent.workspace = self.workspace
        self.session_log.write("workspace", id=self.workspace.id)
        self._log("DONE", f"New session {self.workspace.id}: context, plan and counters reset; new workspace "
                  "folder. Long-term memory, login and earlier workspaces stay.")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "composer":
            # After the edit the document is rewrapped; measure on the next refresh.
            self.call_after_refresh(self._resize_composer)

    def on_resize(self, event: events.Resize) -> None:
        self.call_after_refresh(self._resize_composer)

    def _resize_composer(self) -> None:
        """Grows the composer with the VISUAL lines (soft wrap included), so no typed line is hidden."""
        composer = self.query_one("#composer", ComposerInput)
        rows = composer.wrapped_document.height or 1
        self.query_one("#composer-row", Horizontal).styles.height = min(MAX_COMPOSER_ROWS, rows) + 2
        composer.scroll_cursor_visible()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-browser":
            self._close_browser()
        elif event.button.id == "stop":
            self.action_interrupt()
        elif event.button.id == "models":
            self._show_model_settings()
        elif event.button.id == "telegram":
            self._show_telegram_settings()

    @on(ComposerInput.Submitted)
    def on_composer_submitted(self, event: ComposerInput.Submitted) -> None:
        text = event.value.strip()
        if not text or self.busy:
            return
        event.composer.load_text("")
        if not text.startswith("/"):
            record_fn = getattr(self.memory, "record_prompt", None)
            if callable(record_fn):
                event.composer.set_history(record_fn(text))
        self.query_one("#timeline", VerticalScroll).anchor()
        self._chat_log("You", text)
        if not text.startswith("/"):
            if self.teaching:
                self._teach_note(text)  # while recording, what the user types explains the current step
            else:
                self._start(text)
            return
        head = text.split()[0].lower()
        command = ALIASES.get(head, head)
        if command == "/quit":
            self.exit()
        elif command == "/help":
            self.action_help()
        elif command == "/models":
            self._show_model_settings()
        elif command == "/telegram":
            self._telegram_command(text[len(head):])
        elif command in ("/clear", "/restart"):
            self._clear_context()
            if command == "/restart":
                self._close_browser()
        elif command == "/memory":
            site_map = getattr(self.memory, "site_map", lambda: {})()
            if site_map:
                sections = [
                    f"**{site}** (`{self.memory.site_path(site) if hasattr(self.memory, 'site_path') else site}`):\n"
                    + "\n".join(lines)
                    for site, lines in sorted(site_map.items())
                ]
                body = "\n\n".join(sections)
            else:
                body = self.memory.long_term() or "(empty - it is created at the end of tasks)"
            self._chat_log(
                "Memory",
                f"**Site-scoped persistent memory** (`{getattr(self.memory, 'sites_dir', self.memory.long_term_path)}`):\n\n"
                f"{body}\n\n**This session:** `{self.memory.session_path}`",
            )
        elif command == "/close":
            self._close_browser()
        elif command == "/teach":
            self._teach(text[len(head):])
        elif command == "/playbooks":
            self._list_playbooks()
        elif command == "/workspace":
            self._show_workspace()
        elif command == "/usage":
            self._show_usage()
        elif command == "/login":
            url = normalize_url(text[len(head):])
            if not url:
                self._log("ERROR", "Use /login <url>, for example /login linkedin.com")
            else:
                self._login(url)
        else:
            self._log("ERROR", "Unknown command. Use /help.")

    def _log_session_usage(self) -> None:
        models = self.usage.snapshot()
        if models:
            self.session_log.write("session_usage", models=models, total=Usage.total(models))

    def on_unmount(self) -> None:
        self.cancel.set()
        self._log_session_usage()
        if self.agent:
            self.agent.model.close()
            if self.agent.planner:
                self.agent.planner.close()
