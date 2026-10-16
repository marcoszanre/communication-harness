"""Offline provider/settings tests; no cloud or local service is contacted."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest import mock

from foundry import Foundry
from local_foundry import local_endpoint, prepare_model
from model_config import ModelSettings, Settings


class Stream:
    def __init__(self, *chunks):
        self.chunks = chunks

    def __enter__(self):
        return iter(self.chunks)

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.chunks)


def chunk(content=None, tools=None, finish=None, usage=None):
    data = {}
    if content is not None:
        data["content"] = content
    if tools is not None:
        data["tool_calls"] = tools
    delta = SimpleNamespace(model_dump=lambda **kwargs: data)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=usage)


def client(*chunks):
    create = mock.Mock(return_value=Stream(*chunks))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=mock.Mock())


class SettingsTests(unittest.TestCase):
    def test_local_only_settings_need_no_azure_endpoint(self):
        settings = Settings(executor=ModelSettings(provider="local", model="local-chat"),
                            planner=ModelSettings(provider="local", model="local-planner"))
        settings.validate()
        self.assertIn("plan local: local-planner", settings.summary())

    def test_settings_round_trip_and_environment_overrides(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, {}, clear=True):
            path = Path(folder) / "settings.json"
            settings = Settings(executor=ModelSettings(provider="local", model="local-chat"))
            settings.save(path)
            self.assertEqual(Settings.load(path), settings)
            with mock.patch.dict(os.environ, {"BROWSER_HARNESS_EXECUTOR_MODEL": "another-local-model"}):
                self.assertEqual(Settings.load(path).executor.model, "another-local-model")
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_legacy_azure_environment_is_supported_without_hardcoded_deployments(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, {
            "BROWSER_HARNESS_DEPLOYMENT": "example-executor",
            "BROWSER_HARNESS_PLANNER": "example-planner",
            "BROWSER_HARNESS_FOUNDRY_ENDPOINT": "https://example.services.ai.azure.com/openai/v1/",
        }, clear=True):
            settings = Settings.load(Path(folder) / "settings.json")
            settings.validate()
            self.assertEqual(settings.executor.model, "example-executor")
            self.assertEqual(settings.planner.model, "example-planner")

    def test_no_model_is_guessed_when_configuration_is_missing(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, {}, clear=True):
            settings = Settings.load(Path(folder) / "settings.json")
            self.assertEqual(settings.executor.model, "")
            self.assertEqual(settings.planner.provider, "off")
            with self.assertRaisesRegex(ValueError, "enter a model"):
                settings.validate()

    def test_credentials_unknown_fields_and_invalid_scalar_types_are_rejected(self):
        for data in ({"api_key": "synthetic-value"}, {"executor": {"api_key": "synthetic-value"}},
                     {"executor": {"model": 123}}, {"allow_local_downloads": "yes"}):
            with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, {}, clear=True):
                path = Path(folder) / "settings.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    Settings.load(path)
        for endpoint in ("http://example.com", "https://user:password@example.com/v1",
                         "https://example.com/v1?token=synthetic"):
            with self.assertRaises(ValueError):
                ModelSettings(model="example-model", endpoint=endpoint).validate("executor")

    def test_local_provider_creation_does_not_construct_azure_credentials(self):
        runtime = {"endpoint": "http://127.0.0.1:12345/v1", "model": "local-chat:1",
                   "alias": "local-chat", "device": "Cpu", "images": False}
        settings = Settings(executor=ModelSettings(provider="local", model="local-chat"))
        with mock.patch("local_foundry.prepare_model", return_value=runtime), \
                mock.patch("foundry.DefaultAzureCredential") as credentials, mock.patch("foundry.OpenAI"):
            model = settings.create_model("executor")
        credentials.assert_not_called()
        self.assertEqual(model.provider, "local")
        self.assertFalse(model.supports_images)

    def test_ready_presets_and_azure_endpoint_retention(self):
        endpoint = "https://example.services.ai.azure.com/openai/v1/"
        base = Settings(azure_endpoint=endpoint)
        local_cfg = base.with_preset("local")
        self.assertEqual(local_cfg.active_preset(), "local")
        self.assertEqual(local_cfg.executor.provider, "local")
        self.assertEqual(local_cfg.executor.model, "qwen2.5-1.5b")
        self.assertEqual(local_cfg.planner.provider, "local")
        self.assertEqual(local_cfg.planner.model, "qwen2.5-1.5b")
        self.assertEqual(local_cfg.azure_endpoint, endpoint)
        local_cfg.validate()

        azure_cfg = local_cfg.with_preset("azure")
        self.assertEqual(azure_cfg.active_preset(), "azure")
        self.assertEqual(azure_cfg.executor.provider, "azure")
        self.assertEqual(azure_cfg.executor.model, "gpt-4.1-mini")
        self.assertEqual(azure_cfg.executor.endpoint, endpoint)
        self.assertEqual(azure_cfg.planner.provider, "azure")
        self.assertEqual(azure_cfg.planner.model, "gpt-4.1")
        self.assertEqual(azure_cfg.planner.reasoning, "low")
        azure_cfg.validate()

        hybrid_cfg = azure_cfg.with_preset("hybrid")
        self.assertEqual(hybrid_cfg.active_preset(), "hybrid")
        self.assertEqual(hybrid_cfg.executor.provider, "local")
        self.assertEqual(hybrid_cfg.executor.model, "qwen2.5-1.5b")
        self.assertEqual(hybrid_cfg.planner.provider, "azure")
        self.assertEqual(hybrid_cfg.planner.model, "gpt-4.1")
        hybrid_cfg.validate()

        fast_cfg = hybrid_cfg.with_preset("local-fast")
        self.assertEqual(fast_cfg.active_preset(), "local-fast")
        self.assertEqual(fast_cfg.planner.provider, "off")
        fast_cfg.validate()


class LocalRuntimeTests(unittest.TestCase):
    def commands(self, cached=True, tools=True, running=True):
        calls = []

        def run(args):
            calls.append(args)
            if args == ["server", "status"]:
                return {"running": running or ["server", "start"] in calls,
                        "webUrls": ["http://127.0.0.1:12345"]}
            if args[:2] == ["model", "info"]:
                return {"model": {"id": "local-chat:1", "alias": "local-chat", "type": "Chat",
                                  "device": "Cpu", "cached": cached, "supportsToolCalling": tools}}
            return {"success": True}
        return run, calls

    def test_dynamic_endpoint_and_model_id_are_discovered(self):
        run, calls = self.commands()
        runtime = prepare_model("local-chat", run=run, require_tools=True)
        self.assertEqual(runtime["endpoint"], "http://127.0.0.1:12345/v1")
        self.assertEqual(runtime["model"], "local-chat:1")
        self.assertIn(["model", "load", "local-chat:1"], calls)
        self.assertNotIn(["server", "stop"], calls)

    def test_stopped_server_is_started_not_restarted_or_stopped(self):
        run, calls = self.commands(running=False)
        prepare_model("local-chat", run=run)
        self.assertIn(["server", "start"], calls)

    def test_download_is_opt_in(self):
        run, calls = self.commands(cached=False)
        with self.assertRaisesRegex(RuntimeError, "not cached"):
            prepare_model("local-chat", run=run)
        self.assertFalse(any(args[:2] == ["model", "download"] for args in calls))
        prepare_model("local-chat", run=run, allow_download=True)
        self.assertIn(["model", "download", "local-chat:1"], calls)

    def test_executor_requires_native_tool_support_but_planner_does_not(self):
        run, calls = self.commands(tools=False)
        with self.assertRaisesRegex(RuntimeError, "tool-calling support"):
            prepare_model("local-chat", run=run, require_tools=True)
        self.assertFalse(any(args[:2] == ["model", "load"] for args in calls))
        prepare_model("local-chat", run=run)

    def test_nonloopback_credentials_and_non_http_are_rejected(self):
        for endpoint in ("https://localhost:12345", "http://example.com:12345", "http://localhost.example.com",
                         "http://user:password@localhost:12345", "file:///tmp/model", "http://localhost/x"):
            with self.assertRaises(ValueError, msg=endpoint):
                local_endpoint(endpoint)
        self.assertEqual(local_endpoint("http://[::1]:12345/"), "http://[::1]:12345/v1")

    def test_endpoint_must_match_the_foundry_service(self):
        run, _ = self.commands()
        with self.assertRaisesRegex(ValueError, "does not match"):
            prepare_model("local-chat", run=run, endpoint="http://127.0.0.1:54321")

    def test_a_model_name_is_not_a_shell_command(self):
        run, calls = self.commands()
        with self.assertRaises(ValueError):
            prepare_model("local-chat --unsafe", run=run)
        self.assertEqual(calls, [])


class StreamingProviderTests(unittest.TestCase):
    def test_cancelled_request_is_not_sent_to_any_provider(self):
        fake = client(chunk(content="OK", finish="stop"))
        model = Foundry(client=fake, deployment="local-chat", provider="local")
        cancel = Event()
        cancel.set()
        with self.assertRaises(InterruptedError):
            model.complete([], [], lambda *args: None, cancel)
        fake.chat.completions.create.assert_not_called()

    def test_local_uses_native_parameters_not_azure_reasoning(self):
        fake = client(chunk(content="OK", finish="stop"))
        model = Foundry(client=fake, deployment="local-chat", provider="local", reasoning="high", max_tokens=128)
        turn = model.complete([{"role": "user", "content": "OK"}], [], lambda *args: None, Event())
        self.assertEqual(turn.content, "OK")
        sent = fake.chat.completions.create.call_args.kwargs
        self.assertEqual(sent["max_tokens"], 128)
        self.assertNotIn("max_completion_tokens", sent)
        self.assertNotIn("reasoning_effort", sent)

    def test_azure_retains_its_completion_limit_parameter(self):
        fake = client(chunk(content="OK", finish="stop"))
        model = Foundry(client=fake, deployment="example-deployment")
        model.complete([], [], lambda *args: None, Event())
        self.assertIn("max_completion_tokens", fake.chat.completions.create.call_args.kwargs)

    def test_streamed_native_tool_arguments_are_accumulated(self):
        fake = client(chunk(tools=[{"index": 0, "id": "call-1",
                                   "function": {"name": "ping", "arguments": '{"ok":'}}]),
                      chunk(tools=[{"index": 0, "function": {"arguments": "true}"}}], finish="tool_calls"))
        model = Foundry(client=fake, deployment="local-chat", provider="local")
        turn = model.complete([], [{"type": "function", "function": {"name": "ping"}}],
                              lambda *args: None, Event())
        self.assertEqual(json.loads(turn.tool_calls[0].arguments), {"ok": True})

    def test_empty_truncated_and_incomplete_streams_are_errors(self):
        for parts in ([chunk(content="", finish="stop")], [chunk(content="partial")],
                      [chunk(content="partial", finish="length")]):
            model = Foundry(client=client(*parts), deployment="local-chat", provider="local")
            with self.assertRaises(RuntimeError):
                model.complete([], [], lambda *args: None, Event())

    def test_required_tool_call_cannot_be_replaced_by_a_text_claim(self):
        model = Foundry(client=client(chunk(content="I called ping", finish="stop")),
                        deployment="local-chat", provider="local")
        with self.assertRaisesRegex(RuntimeError, "required tool call"):
            model.complete([], [{"type": "function", "function": {"name": "ping"}}], lambda *args: None,
                           Event(), tool_choice={"type": "function", "function": {"name": "ping"}})

    def test_invalid_tool_names_or_arguments_never_reach_execution(self):
        for name, arguments in (("unknown", "{}"), ("ping", '{"partial":'), ("ping", "[]")):
            fake = client(chunk(tools=[{"index": 0, "id": "call",
                                        "function": {"name": name, "arguments": arguments}}], finish="tool_calls"))
            model = Foundry(client=fake, deployment="local-chat", provider="local")
            with self.assertRaises(RuntimeError):
                model.complete([], [{"type": "function", "function": {"name": "ping"}}],
                               lambda *args: None, Event())


if __name__ == "__main__":
    unittest.main()
