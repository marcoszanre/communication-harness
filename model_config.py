"""Model settings and provider selection. Settings never contain credentials."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"
PROVIDERS = ("azure", "local")
LOCAL_DEFAULT_MODEL = os.getenv("BROWSER_HARNESS_LOCAL_MODEL", "qwen2.5-1.5b")
AZURE_DEFAULT_EXECUTOR = os.getenv("BROWSER_HARNESS_AZURE_EXECUTOR", "gpt-4.1-mini")
AZURE_DEFAULT_PLANNER = os.getenv("BROWSER_HARNESS_AZURE_PLANNER", "gpt-4.1")
AZURE_DEFAULT_REASONING = "low"

LOCAL_MODEL_CHOICES: tuple[tuple[str, str], ...] = (
    ("qwen2.5-1.5b (Foundry Local · 1.5B · tool calling)", "qwen2.5-1.5b"),
    ("phi-4-mini (Foundry Local · 3.8B · tool calling)", "phi-4-mini"),
    ("qwen2.5-7b (Foundry Local · 7B · tool calling)", "qwen2.5-7b"),
    ("qwen3-4b (Foundry Local · 4B · tool calling)", "qwen3-4b"),
    ("qwen3-8b (Foundry Local · 8B · tool calling)", "qwen3-8b"),
    ("qwen2.5-1.5b-instruct-generic-cpu:4 (CPU · 1.5B)", "qwen2.5-1.5b-instruct-generic-cpu:4"),
)

AZURE_MODEL_CHOICES: tuple[tuple[str, str], ...] = (
    ("gpt-4.1-mini (Fast Executor · Azure Foundry)", "gpt-4.1-mini"),
    ("gpt-4.1 (Reasoning Planner · Azure Foundry)", "gpt-4.1"),
    ("gpt-4o-mini (Fast Executor · Azure Foundry)", "gpt-4o-mini"),
    ("gpt-4o (General Purpose · Azure Foundry)", "gpt-4o"),
    ("o4-mini (Reasoning Planner · Azure Foundry)", "o4-mini"),
)

PRESET_CHOICES: tuple[tuple[str, str], ...] = (
    (f"Foundry Local — {LOCAL_DEFAULT_MODEL} (On-device · Plan + Run)", "local"),
    (f"Azure Foundry — {AZURE_DEFAULT_PLANNER} Plan + {AZURE_DEFAULT_EXECUTOR} Run", "azure"),
    (f"Hybrid — Azure Plan ({AZURE_DEFAULT_PLANNER}) + Local Run ({LOCAL_DEFAULT_MODEL})", "hybrid"),
    (f"Foundry Local Fast — {LOCAL_DEFAULT_MODEL} (Executor only · Planner off)", "local-fast"),
    ("Custom configuration", "custom"),
)

LOCAL_MODEL_VALUES = {value for _, value in LOCAL_MODEL_CHOICES}
AZURE_MODEL_VALUES = {value for _, value in AZURE_MODEL_CHOICES}


def validate_azure_endpoint(endpoint: str, label: str = "Azure") -> None:
    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"{label}: use an HTTPS Azure endpoint without embedded credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{label}: the endpoint must not contain a query or fragment.")
    if not parsed.hostname.endswith((".services.ai.azure.com", ".openai.azure.com",
                                     ".cognitiveservices.azure.com")):
        raise ValueError(f"{label}: use a public Azure Foundry/Azure OpenAI resource endpoint. "
                         "Azure bearer tokens are not sent to arbitrary hosts.")


def is_valid_azure_endpoint(endpoint: str) -> bool:
    if not endpoint or not isinstance(endpoint, str):
        return False
    try:
        validate_azure_endpoint(endpoint)
        return True
    except ValueError:
        return False


@dataclass
class ModelSettings:
    provider: str = "azure"
    model: str = ""
    endpoint: str = ""
    reasoning: str = ""
    max_tokens: int = 4000

    def validate(self, role: str) -> None:
        if any(not isinstance(value, str) for value in (self.provider, self.model, self.endpoint, self.reasoning)):
            raise ValueError(f"{role.title()}: provider, model, endpoint and reasoning must be text.")
        if self.endpoint:
            parsed = urlsplit(self.endpoint)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(f"{role.title()}: endpoints must not contain credentials, queries or fragments.")
        if self.provider == "off" and role == "planner":
            return
        if self.provider not in PROVIDERS:
            raise ValueError(f"{role.title()}: choose Azure Foundry or Foundry Local.")
        if not self.model.strip() or len(self.model) > 200:
            raise ValueError(f"{role.title()}: enter a model deployment, local alias, or model ID.")
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool) or not 1 <= self.max_tokens <= 32768:
            raise ValueError(f"{role.title()}: max_tokens must be an integer from 1 to 32768.")
        if self.reasoning not in ("", "none", "low", "medium", "high"):
            raise ValueError(f"{role.title()}: invalid reasoning effort.")
        if self.provider == "azure":
            validate_azure_endpoint(self.endpoint, role.title())
        elif self.endpoint:
            from local_foundry import local_endpoint
            local_endpoint(self.endpoint)


@dataclass
class Settings:
    executor: ModelSettings = field(default_factory=ModelSettings)
    planner: ModelSettings = field(default_factory=lambda: ModelSettings(provider="off"))
    planner_scope: str = "actions"
    allow_local_downloads: bool = False
    azure_endpoint: str = ""

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> Settings:
        data = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as error:
                raise ValueError(f"Cannot read model settings: {error}") from error
            if not isinstance(data, dict) or set(data) - {
                "executor", "planner", "planner_scope", "allow_local_downloads", "azure_endpoint"
            }:
                raise ValueError("Model settings contain an invalid object or unknown fields.")
        roles = {}
        for role in ("executor", "planner"):
            value = data.get(role, {})
            if not isinstance(value, dict) or set(value) - set(ModelSettings.__dataclass_fields__):
                raise ValueError(f"Invalid {role} settings. Credentials are not supported in settings.json.")
            roles[role] = ModelSettings(**value) if value else ModelSettings(
                provider="off" if role == "planner" else "azure")
            spec = roles[role]
            if any(not isinstance(v, str) for v in (spec.provider, spec.model, spec.endpoint, spec.reasoning)):
                raise ValueError(f"Invalid {role} settings: textual fields must be strings.")
            if not isinstance(spec.max_tokens, int) or isinstance(spec.max_tokens, bool):
                raise ValueError(f"Invalid {role} max_tokens: use an integer.")
            if (spec.provider not in (("azure", "local", "off") if role == "planner" else PROVIDERS)
                    or spec.reasoning not in ("", "none", "low", "medium", "high")
                    or not 1 <= spec.max_tokens <= 32768):
                raise ValueError(f"Invalid {role} provider, reasoning effort or output limit.")
            prefix = f"BROWSER_HARNESS_{role.upper()}_"
            legacy = "BROWSER_HARNESS_DEPLOYMENT" if role == "executor" else "BROWSER_HARNESS_PLANNER"
            model = os.getenv(prefix + "MODEL", os.getenv(legacy))
            provider = os.getenv(prefix + "PROVIDER")
            if model is not None:
                if role == "planner" and model in ("", "0", "off"):
                    roles[role].provider = "off"
                    roles[role].model = ""
                else:
                    roles[role].model = model.strip()
                    if role == "planner" and roles[role].provider == "off":
                        roles[role].provider = "azure"
            if provider is not None:
                roles[role].provider = provider.strip().lower()
            endpoint = os.getenv(prefix + "ENDPOINT")
            if endpoint is None and roles[role].provider == "azure":
                endpoint = os.getenv("BROWSER_HARNESS_FOUNDRY_ENDPOINT")
            if endpoint is not None:
                roles[role].endpoint = endpoint.strip()
            if role == "planner" and "BROWSER_HARNESS_PLANNER_REASONING" in os.environ:
                roles[role].reasoning = os.environ["BROWSER_HARNESS_PLANNER_REASONING"]
        scope = os.getenv("BROWSER_HARNESS_PLANNER_SCOPE", data.get("planner_scope", "actions"))
        downloads = data.get("allow_local_downloads", False)
        if "BROWSER_HARNESS_LOCAL_DOWNLOADS" in os.environ:
            downloads = os.environ["BROWSER_HARNESS_LOCAL_DOWNLOADS"] == "1"
        azure_endpoint = data.get("azure_endpoint", "")
        if not isinstance(azure_endpoint, str):
            raise ValueError("Use a string azure_endpoint.")
        azure_endpoint = azure_endpoint.strip()
        if azure_endpoint:
            validate_azure_endpoint(azure_endpoint, "Azure preset")
        if scope not in ("actions", "all") or not isinstance(downloads, bool):
            raise ValueError("Use planner_scope 'actions' or 'all' and a boolean allow_local_downloads.")
        return cls(**roles, planner_scope=scope, allow_local_downloads=downloads, azure_endpoint=azure_endpoint)

    def known_azure_endpoint(self) -> str:
        for candidate in (
            self.executor.endpoint if self.executor.provider == "azure" else "",
            self.planner.endpoint if self.planner.provider == "azure" else "",
            self.azure_endpoint,
            os.getenv("BROWSER_HARNESS_FOUNDRY_ENDPOINT", "").strip(),
            os.getenv("BROWSER_HARNESS_EXECUTOR_ENDPOINT", "").strip(),
            os.getenv("BROWSER_HARNESS_PLANNER_ENDPOINT", "").strip(),
        ):
            if is_valid_azure_endpoint(candidate):
                return candidate
        return ""

    def active_preset(self) -> str:
        ex, pl = self.executor, self.planner
        if ex.provider == "local" and ex.model == LOCAL_DEFAULT_MODEL:
            if pl.provider == "local" and pl.model == LOCAL_DEFAULT_MODEL:
                return "local"
            if pl.provider == "off":
                return "local-fast"
            if pl.provider == "azure" and pl.model == AZURE_DEFAULT_PLANNER:
                return "hybrid"
        if (ex.provider == "azure" and ex.model == AZURE_DEFAULT_EXECUTOR
                and pl.provider == "azure" and pl.model == AZURE_DEFAULT_PLANNER):
            return "azure"
        return "custom"

    def with_preset(self, preset: str, azure_endpoint: str = "") -> Settings:
        endpoint = (azure_endpoint or self.known_azure_endpoint()).strip()
        ex_tokens = self.executor.max_tokens
        pl_tokens = self.planner.max_tokens
        if preset == "local":
            executor = ModelSettings(provider="local", model=LOCAL_DEFAULT_MODEL, endpoint="",
                                     reasoning="", max_tokens=ex_tokens)
            planner = ModelSettings(provider="local", model=LOCAL_DEFAULT_MODEL, endpoint="",
                                    reasoning="", max_tokens=pl_tokens)
        elif preset == "local-fast":
            executor = ModelSettings(provider="local", model=LOCAL_DEFAULT_MODEL, endpoint="",
                                     reasoning="", max_tokens=ex_tokens)
            planner = ModelSettings(provider="off", model="", endpoint="",
                                    reasoning="", max_tokens=pl_tokens)
        elif preset == "azure":
            executor = ModelSettings(provider="azure", model=AZURE_DEFAULT_EXECUTOR, endpoint=endpoint,
                                     reasoning="", max_tokens=ex_tokens)
            planner = ModelSettings(provider="azure", model=AZURE_DEFAULT_PLANNER, endpoint=endpoint,
                                    reasoning=AZURE_DEFAULT_REASONING, max_tokens=pl_tokens)
        elif preset == "hybrid":
            executor = ModelSettings(provider="local", model=LOCAL_DEFAULT_MODEL, endpoint="",
                                     reasoning="", max_tokens=ex_tokens)
            planner = ModelSettings(provider="azure", model=AZURE_DEFAULT_PLANNER, endpoint=endpoint,
                                    reasoning=AZURE_DEFAULT_REASONING, max_tokens=pl_tokens)
        else:
            raise ValueError(f"Unknown model preset: {preset}")
        saved_azure = endpoint if is_valid_azure_endpoint(endpoint) else self.azure_endpoint
        return Settings(executor=executor, planner=planner, planner_scope=self.planner_scope,
                        allow_local_downloads=self.allow_local_downloads, azure_endpoint=saved_azure)

    def validate(self) -> None:
        self.executor.validate("executor")
        self.planner.validate("planner")
        if self.azure_endpoint:
            validate_azure_endpoint(self.azure_endpoint, "Azure preset")

    def save(self, path: Path = SETTINGS_PATH) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def summary(self) -> str:
        def role_text(role: ModelSettings) -> str:
            return "off" if role.provider == "off" else f"{role.provider}: {role.model or '(not configured)'}"
        return f"plan {role_text(self.planner)} | run {role_text(self.executor)}"

    def create_model(self, role: str, status: Callable[[str], None] = lambda text: None, cancel=None):
        from foundry import Foundry

        spec = getattr(self, role)
        spec.validate(role)
        if spec.provider == "off":
            return None
        endpoint, model, images = spec.endpoint, spec.model, True
        if spec.provider == "local":
            from local_foundry import prepare_model
            runtime = prepare_model(spec.model, status, allow_download=self.allow_local_downloads,
                                    endpoint=spec.endpoint, require_tools=role == "executor", cancel=cancel)
            endpoint, model, images = runtime["endpoint"], runtime["model"], runtime["images"]
            status(f"{role.title()} ready: Foundry Local / {runtime['alias']} / {runtime['device']}.")
        else:
            status(f"{role.title()} configured: Azure Foundry / {model}.")
        return Foundry(deployment=model, endpoint=endpoint, provider=spec.provider, reasoning=spec.reasoning,
                       max_tokens=spec.max_tokens, supports_images=images)
