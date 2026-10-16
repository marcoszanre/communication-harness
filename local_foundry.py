"""Discover and load Foundry Local through its CLI; inference stays on loopback."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from threading import Event
from typing import Callable
from urllib.parse import urlsplit, urlunsplit


def local_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1")
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Foundry Local requires an HTTP loopback endpoint without credentials.")
    if parsed.path.rstrip("/") not in ("", "/v1"):
        raise ValueError("Foundry Local endpoint must use the root or /v1 path.")
    return urlunsplit((parsed.scheme, parsed.netloc, "/v1", "", ""))


def foundry_json(args: list[str], cancel: Event | None = None) -> dict:
    executable = shutil.which("foundry")
    if not executable and os.name == "nt" and os.getenv("LOCALAPPDATA"):
        candidate = Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "WindowsApps" / "foundry.exe"
        if candidate.is_file():
            executable = str(candidate)
    if not executable:
        raise RuntimeError("Foundry Local CLI was not found. Install Foundry Local and reopen your terminal.")
    if cancel and cancel.is_set():
        raise InterruptedError("Stopped by the user.")
    with subprocess.Popen([executable, *args, "--output", "json"], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace") as process:
        deadline = time.monotonic() + (1800 if "download" in args else 120)
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                stopped = cancel and cancel.is_set()
                if stopped or time.monotonic() >= deadline:
                    process.terminate()
                    try:
                        process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
                    if stopped:
                        raise InterruptedError("Stopped by the user.")
                    raise RuntimeError(f"Foundry Local timed out during {' '.join(args[:2])}.")
    if process.returncode:
        raise RuntimeError(f"Foundry Local {' '.join(args[:2])} failed: {(stderr or stdout).strip()[:500]}")
    try:
        result = json.loads(stdout)
    except ValueError:
        result = None
        for line in reversed(stdout.splitlines()):
            try:
                result = json.loads(line)
                break
            except ValueError:
                continue
    if not isinstance(result, dict):
        raise RuntimeError("Foundry Local returned invalid JSON.")
    if result.get("success") is False:
        raise RuntimeError(str(result.get("message") or result.get("error") or "Foundry Local reported a failure."))
    return result


def prepare_model(model: str, status: Callable[[str], None] = lambda text: None, *,
                  allow_download: bool = False, endpoint: str = "", require_tools: bool = False,
                  cancel: Event | None = None, run=None) -> dict:
    if cancel and cancel.is_set():
        raise InterruptedError("Stopped by the user.")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:-]{0,199}", model):
        raise ValueError("Use a Foundry Local model alias or ID, not a path or command.")
    command = run or (lambda args: foundry_json(args, cancel))
    if endpoint:
        endpoint = local_endpoint(endpoint)
    status("Checking Foundry Local...")
    server = command(["server", "status"])
    if not server.get("running"):
        status("Starting the Foundry Local server...")
        command(["server", "start"])
        server = command(["server", "status"])
    urls = server.get("webUrls") or []
    if not server.get("running") or not urls:
        raise RuntimeError("Foundry Local is not ready. Check 'foundry server status'.")
    discovered = local_endpoint(urls[0])
    if endpoint and endpoint != discovered:
        raise ValueError("Configured local endpoint does not match the running Foundry Local server.")
    info = command(["model", "info", model]).get("model")
    if not isinstance(info, dict) or not info.get("id") or info.get("type") not in ("Chat", "Multimodal"):
        raise RuntimeError("The selected model is not a Foundry Local chat model. Run 'foundry model list'.")
    if require_tools and info.get("supportsToolCalling") is not True:
        raise RuntimeError("The executor needs a model with tool-calling support. Choose another local model.")
    if not info.get("cached"):
        if not allow_download:
            raise RuntimeError("The local model is not cached. Download it with 'foundry model download', "
                               "or explicitly enable model downloads in /models.")
        status(f"Downloading {info.get('alias') or model} ({info.get('fileSizeMb', '?')} MB)...")
        command(["model", "download", info["id"]])
    status(f"Loading {info.get('alias') or model} on {info.get('device', 'the selected device')}...")
    command(["model", "load", info["id"]])
    return {"endpoint": discovered, "model": info["id"], "alias": info.get("alias") or model,
            "device": info.get("device") or "unknown", "images": info.get("type") == "Multimodal"}
