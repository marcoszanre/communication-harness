"""Foundry client (Chat Completions) with streaming and tool calls."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from threading import Event
from typing import Any, Callable

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI, OpenAIError

# Your Azure AI Foundry endpoint, e.g. https://<resource>.services.ai.azure.com/openai/v1/ (never hard-coded).
ENDPOINT = os.getenv("BROWSER_HARNESS_FOUNDRY_ENDPOINT", "").strip()
DEPLOYMENT = os.getenv("BROWSER_HARNESS_DEPLOYMENT", "")
EventSink = Callable[[str, str], None]


class ContentFiltered(RuntimeError):
    """The Azure content filter (e.g. Prompt Shields/'Jailbreak') blocked the call."""


@dataclass
class ToolCall:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class Turn:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None


class Foundry:
    def __init__(self, client: Any = None, deployment: str | None = None, reasoning: str | None = None, *,
                 endpoint: str | None = None, provider: str = "azure", max_tokens: int = 4000,
                 supports_images: bool = True) -> None:
        self.deployment = deployment or DEPLOYMENT
        if not self.deployment:
            raise RuntimeError("Configure an executor model with /models or BROWSER_HARNESS_DEPLOYMENT.")
        if provider not in ("azure", "local"):
            raise ValueError("Unknown model provider.")
        self.provider = provider
        self.usage_name = f"{provider}:{self.deployment}"
        self.supports_images = supports_images
        self.max_tokens = max_tokens
        # Reasoning for calls WITHOUT tools (e.g. the planner); with tools GPT models need "none".
        self.reasoning = reasoning
        self.credential = None
        if client is None:
            endpoint = endpoint or ENDPOINT
            if provider == "local":
                from local_foundry import local_endpoint
                client = OpenAI(base_url=local_endpoint(endpoint), api_key="none", timeout=180, max_retries=0)
            elif not endpoint:
                raise RuntimeError("Set BROWSER_HARNESS_FOUNDRY_ENDPOINT to your Azure AI Foundry endpoint "
                                   "(https://<resource>.services.ai.azure.com/openai/v1/) and sign in with az login.")
            else:
                from model_config import ModelSettings
                ModelSettings(model=self.deployment, endpoint=endpoint, reasoning=self.reasoning or "",
                              max_tokens=max_tokens).validate("executor")
                self.credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
                token = get_bearer_token_provider(self.credential, "https://ai.azure.com/.default")
                client = OpenAI(base_url=endpoint, api_key=token, timeout=120, max_retries=2)
        self.client = client

    def close(self) -> None:
        if self.credential:
            self.credential.close()
        self.client.close()

    def complete(self, messages: list[dict], tools: list[dict], emit: EventSink, cancel: Event,
                 tool_choice: str | dict = "auto") -> Turn:
        if cancel.is_set():
            raise InterruptedError("Stopped by the user.")
        try:
            return self._complete(messages, tools, emit, cancel, tool_choice)
        except OpenAIError as error:
            # Azure returns the block as an HTTP 400 error or as an error chunk in the middle of the stream.
            if "content_filter" in str(error):
                raise ContentFiltered(str(error)) from error
            raise

    def _complete(self, messages: list[dict], tools: list[dict], emit: EventSink, cancel: Event,
                  tool_choice: str | dict) -> Turn:
        turn = Turn()
        calls: dict[int, ToolCall] = {}
        extra: dict[str, Any] = {}
        if tools:
            extra.update(tools=tools, tool_choice=tool_choice)
        if self.provider == "azure" and (self.reasoning or self.deployment.startswith("gpt-")):
            # GPT models only accept tools in Chat Completions with reasoning_effort="none".
            extra["reasoning_effort"] = "none" if tools or not self.reasoning else self.reasoning
        limit = {"max_tokens" if self.provider == "local" else "max_completion_tokens": self.max_tokens}
        stream = self.client.chat.completions.create(
            model=self.deployment, messages=messages,
            stream=True, stream_options={"include_usage": True},
            **limit, **extra,
        )
        finished = False
        with stream:
            for chunk in stream:
                if cancel.is_set():
                    raise InterruptedError("Stopped by the user.")
                if chunk.usage is not None:
                    turn.input_tokens = chunk.usage.prompt_tokens
                    turn.output_tokens = chunk.usage.completion_tokens
                    details = chunk.usage.prompt_tokens_details
                    turn.cached_tokens = getattr(details, "cached_tokens", None) if details else None
                for choice in chunk.choices:
                    if choice.finish_reason == "length":
                        raise RuntimeError("The model reached its output limit. No incomplete tool call was executed.")
                    if choice.finish_reason:
                        finished = True
                    if choice.finish_reason == "content_filter":
                        raise ContentFiltered("Response blocked by the content filter.")
                    delta = choice.delta.model_dump(exclude_none=True)
                    if delta.get("reasoning_content"):
                        turn.reasoning += delta["reasoning_content"]
                        emit("reasoning", delta["reasoning_content"])
                    if delta.get("content"):
                        turn.content += delta["content"]
                        emit("answer", delta["content"])
                    # Each tool call's arguments arrive in chunks, grouped by index.
                    for part in delta.get("tool_calls") or []:
                        call = calls.setdefault(part.get("index", 0), ToolCall())
                        call.id = part.get("id") or call.id
                        function = part.get("function") or {}
                        call.name = function.get("name") or call.name
                        call.arguments += function.get("arguments") or ""
        turn.tool_calls = [calls[index] for index in sorted(calls)]
        if not finished or not (turn.content.strip() or turn.tool_calls):
            raise RuntimeError("The model returned an empty or incomplete response.")
        allowed = {tool["function"]["name"] for tool in tools}
        for call in turn.tool_calls:
            if not call.id or call.name not in allowed:
                raise RuntimeError("The model returned an invalid tool call.")
            try:
                arguments = json.loads(call.arguments)
            except ValueError as error:
                raise RuntimeError("The model returned incomplete tool arguments.") from error
            if not isinstance(arguments, dict):
                raise RuntimeError("Tool arguments must be a JSON object.")
        if isinstance(tool_choice, dict) and not any(
                call.name == tool_choice["function"]["name"] for call in turn.tool_calls):
            raise RuntimeError("The selected model did not make the required tool call. "
                               "Choose a more capable tool-calling executor in /models.")
        return turn
