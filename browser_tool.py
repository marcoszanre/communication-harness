"""The model's browser tool: one playwright-cli command per call."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
CLI_JS = HERE / "node_modules" / "@playwright" / "cli" / "playwright-cli.js"
SESSION = os.getenv("BROWSER_HARNESS_SESSION", "communication-harness")
MAX_OUTPUT = 6_000   # a full snapshot costs more than a screenshot (~1k tokens); keep it short
MAX_CHANGES = 40     # lines of "what changed" shown after a click
TIMEOUT_SECONDS = 60
MAX_ARGUMENT = 2000
MAX_DRAFT = 40000
VALIDATION_ERROR = "Error: [validation] "
SHOT = HERE / ".playwright-cli" / "look.jpg"
# Commands that act on the current page: the harness answers with what changed, not the whole page.
INTERACTIONS = {"click", "fill", "type", "press", "select", "hover", "check", "uncheck"}
NO_CHANGE = "### No visible change after this action"
INTERACTIVE = re.compile(r"^\s*- (link|button|menuitem\w*|tab|textbox|searchbox|combobox|checkbox|radio|"
                         r"option|switch|heading|/url:)")
_last = {"tree": "", "url": ""}  # last full snapshot seen, to show only the changes after a click
_draft: dict = {}
_browser_open = False

# Only navigation and interaction commands; eval, run-code, cookies, state etc. stay out.
COMMANDS = (
    "open", "goto", "read", "links", "snapshot", "find", "click", "fill", "type", "press",
    "select", "hover", "check", "uncheck", "go-back", "go-forward", "reload",
    "tab-list", "tab-new", "tab-select", "tab-close", "screenshot", "close", "wait", "compose", "verify-message",
)
NAVIGATIONS = {"open", "goto", "reload", "go-back", "go-forward", "tab-new"}
MAX_WAIT = 30  # seconds
# After a navigation, wait (inside the page, one call) until its text stops changing: single-page apps like
# WhatsApp keep loading after the "load" event.
SETTLE_JS = """async () => {
  let last = -1, stable = 0, waited = 0;
  while (waited < 8000 && stable < 1200) {
    const now = document.body ? document.body.innerText.length : 0;
    stable = (now === last && document.readyState === 'complete') ? stable + 300 : 0;
    last = now; await new Promise(r => setTimeout(r, 300)); waited += 300;
  }
  return JSON.stringify({waited});
}"""
# `wait <text>`: until the text (or an aria-label/placeholder/title with it) is on the page, up to MAX_WAIT.
WAIT_JS = """async () => {
  const t = %s.toLowerCase(), limit = %d;
  // The page and its same-origin iframes (LinkedIn shows dialogs inside one, e.g. "Send without a note").
  const docs = () => [document, ...Array.from(document.querySelectorAll('iframe')).map(f => {
    try { return f.contentDocument; } catch { return null; } })].filter(d => d && d.body);
  const has = d => d.body.innerText.toLowerCase().includes(t) ||
    Array.from(d.querySelectorAll('[aria-label],[placeholder],[title]')).some(e =>
      [e.getAttribute('aria-label'), e.getAttribute('placeholder'), e.getAttribute('title')]
        .join(' ').toLowerCase().includes(t));
  const found = () => docs().some(has);
  const start = Date.now();
  while (!found() && Date.now() - start < limit) await new Promise(r => setTimeout(r, 500));
  return JSON.stringify({found: found(), waited: Math.round((Date.now() - start) / 1000),
                         url: location.href, title: document.title});
}"""
URL_COMMANDS = {"open", "goto", "tab-new"}
TARGET_COMMANDS = {"click", "hover", "fill", "select", "check", "uncheck"}
ALLOWED_FLAGS = {"--submit"}
ARITIES = {
    "open": (1,), "goto": (1,), "read": (0, 1), "links": (0,), "snapshot": (0, 1, 2),
    "find": (1,), "click": (1,), "fill": (2, 3), "type": (1,), "press": (1,), "select": (2,),
    "hover": (1,), "check": (1,), "uncheck": (1,), "go-back": (0,), "go-forward": (0,), "reload": (0,),
    "tab-list": (0,), "tab-new": (0, 1), "tab-select": (1,), "tab-close": (0, 1),
    "screenshot": (0,), "close": (0,), "wait": (1,), "compose": (1,), "verify-message": (1,),
}


def normalize_command(command: str, args: list[str]) -> tuple[str, list[str]]:
    if isinstance(command, str) and isinstance(args, list) and " " in command.strip():
        first, *rest = command.split()
        return first, [*rest, *args]
    return command, args


def looks_like_label(target: str) -> bool:
    """True for 'Minhas viagens'; False for refs (e25, f60e25) and CSS/Playwright selectors."""
    return not re.fullmatch(r"(f\d+)?e\d+", target) and not re.search(r"[=#.\[\]>:]", target)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "browser",
        "description": (
            "Runs a playwright-cli command in the user's browser. Cheapest first: "
            "read [url] = TEXT of the main region (to read content; with a URL it opens the page and reads it "
            "in one call - use it for articles instead of goto + read); "
            "find <visible text> = matching elements with refs and link URLs (to locate a button, link, "
            "menu or section by its label; `find A | B` tries alternatives in one call, e.g. the label in "
            "two languages: find Connect | Conectar - then click <ref> or goto <url>); "
            "links = titles and URLs of a listing; "
            "screenshot = an image of the screen (~1k tokens): use it when unsure what is on the page, "
            "when a click did nothing, or before giving up; "
            "snapshot = full tree with refs (expensive, last choice). "
            "open <url> / goto <url> navigate (the open window is reused); click <ref>; "
            "fill <ref> <text> [--submit]; type <text>; press <key>; select <ref> <value>; hover <ref>; "
            "Ordinary args are limited to 2000 characters. For a MESSAGE use compose with args=[textbox ref], "
            "text=the complete draft (up to 40000 characters), recipient=the exact visible chat header, "
            "context_ref=that header's ref. It writes safe chunks into ONE draft, never sends or shortens it. "
            "Then press Enter ONCE and verify-message with args=[fresh textbox ref] and context_ref=the fresh "
            "chat-header ref. Verification checks the full text, recipient, new occurrence, and empty composer. "
            "go-back; reload; tab-list; tab-new [url]; tab-select <index>; close. "
            "wait <text> = waits (up to 30 s) until that text is on the page - for pages still loading; "
            "the harness already waits for a page to settle after open/goto. "
            "After click/fill/press you get only what changed on the page (e.g. the items of a menu that opened)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "enum": list(COMMANDS)},
                "args": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": MAX_ARGUMENT},
                    "maxItems": 3,
                    "description": "Positional arguments of the command, in order.",
                },
                "text": {"type": "string", "maxLength": MAX_DRAFT,
                         "description": "compose only: complete message, not one chunk; never shorten it to fit args."},
                "recipient": {"type": "string", "maxLength": MAX_ARGUMENT,
                              "description": "compose only: exact visible recipient/chat-header label."},
                "context_ref": {"type": "string", "maxLength": MAX_ARGUMENT,
                                "description": "compose/verify-message: ref of the current chat header from find."},
                "done": {"type": "integer",
                         "description": "Optional: number of the task this call's previous result finished; it is "
                                        "closed before the command runs (saves a separate tasks call)."},
                "note": {"type": "string", "description": "With done: the facts you SAW that prove it."},
            },
            "required": ["command", "args"],
            "additionalProperties": False,
        },
    },
}]


def base_argv() -> list[str]:
    """node playwright-cli.js -s=<session>, for the harness's own commands (no model input)."""
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js was not found on PATH.")
    if not CLI_JS.is_file():
        raise RuntimeError("playwright-cli is not installed. Run 'npm install' in browser_harness.")
    return [node, str(CLI_JS), f"-s={SESSION}"]


def build_argv(command: str, args: list[str]) -> list[str]:
    """Validates the model's request and builds the command line, without a shell."""
    command, args = normalize_command(command, args)
    if command not in COMMANDS:
        raise ValueError(f"Command not allowed: {command}")
    if not isinstance(args, list) or len(args) > 3:
        raise ValueError("args must be a list with at most 3 items.")
    if len(args) not in ARITIES[command]:
        raise ValueError(f"{command} needs {', '.join(map(str, ARITIES[command]))} positional argument(s).")
    if command == "fill" and len(args) == 3 and args[2] != "--submit":
        raise ValueError("fill's third argument may only be --submit.")
    literal_indexes = {"type": {0}, "fill": {1}, "select": {1}}.get(command, set())
    for index, arg in enumerate(args):
        if not isinstance(arg, str):
            raise ValueError("Each argument must be text.")
        if len(arg) > MAX_ARGUMENT:
            raise ValueError(f"Argument has {len(arg)} characters; maximum is {MAX_ARGUMENT}. "
                             "For a long message use compose with the separate text field; do not shorten or send chunks.")
        # Keeps the model from switching session (-s=), writing files (--filename) etc.
        if index not in literal_indexes and arg.startswith("-"):
            allowed = ((command == "fill" and index == 2 and arg == "--submit")
                       or (command == "snapshot" and re.fullmatch(r"--depth=\d{1,2}", arg)))
            if not allowed:
                raise ValueError(f"Option not allowed: {arg}")
    if command in URL_COMMANDS and args and not args[0].startswith(("http://", "https://")):
        raise ValueError("Use an http:// or https:// URL.")
    if command in TARGET_COMMANDS and (not args or not args[0].strip(" \"'")):
        # `click ""` became text="" and matched every element on the page (2304 in a LinkedIn run).
        raise ValueError(f"{command} needs a target: a ref from `find` (e.g. e25) or the element's visible text.")
    if command in TARGET_COMMANDS and args and looks_like_label(args[0]):
        # The model sometimes sends the visible label instead of a ref: turn it into a text selector.
        args = [f"text={json.dumps(args[0].strip(chr(34)))}", *args[1:]]
    if any(index < len(args) and args[index].startswith("-") for index in literal_indexes):
        flags = [arg for index, arg in enumerate(args) if index not in literal_indexes and arg in ALLOWED_FLAGS]
        positionals = [arg for index, arg in enumerate(args) if index in literal_indexes or arg not in ALLOWED_FLAGS]
        return [*base_argv(), command, *flags, "--", *positionals]
    return [*base_argv(), command, *args]


# FIXED harness JavaScript (the model cannot use eval). The selector goes in via json.dumps, as a string.
READ_JS = """() => {
  const pick = [%s, 'article', 'main', '[role=main]'].filter(Boolean);
  const el = pick.map(s => { try { return document.querySelector(s); } catch { return null; } })
                 .find(e => e && e.innerText.trim().length > 200) || document.body;
  let published = document.querySelector('meta[property="article:published_time"],meta[name="datePublished"]')
                    ?.content || '';
  const pageType = document.querySelector('meta[property="og:type"]')?.content || '';
  let article = pageType === 'article' || (!!published && pageType !== 'website');
  const walk = value => {
    if (!value || typeof value !== 'object') return '';
    if (Array.isArray(value)) { for (const item of value) { const date = walk(item); if (date) return date; } }
    const types = Array.isArray(value['@type']) ? value['@type'] : [value['@type'] || ''];
    if (types.some(type => /Article$|BlogPosting$/.test(type)) && typeof value.datePublished === 'string')
      return value.datePublished;
    if (value['@graph']) { const date = walk(value['@graph']); if (date) return date; }
    if (value.mainEntity) { const date = walk(value.mainEntity); if (date) return date; }
    return '';
  };
  if (!published) for (const script of document.querySelectorAll('script[type="application/ld+json"]')) {
    try { published = walk(JSON.parse(script.textContent)); } catch {}
    if (published) { article = true; break; }
  }
  const articles = document.querySelectorAll('article');
  if (!published && articles.length === 1 && articles[0].querySelector('h1')) {
    published = articles[0].querySelector('time[datetime]')?.dateTime || '';
    article = !!published;
  }
  let canonical = document.querySelector('link[rel=canonical]')?.href || location.href;
  try { const url = new URL(canonical); url.hash = ''; canonical = url.href; } catch {}
  return JSON.stringify({url: location.href, title: document.title, text: el.innerText, published, canonical, article});
}"""
LINKS_JS = """() => {
  const root = document.querySelector('main') || document.body, seen = new Set(), out = [];
  for (const a of root.querySelectorAll('a[href]')) {
    const text = a.innerText.replace(/\\s+/g, ' ').trim();
    if (text.length < 25 || !a.href.startsWith('http') || seen.has(a.href)) continue;
    seen.add(a.href); out.push([text.slice(0, 160), a.href]);
    if (out.length >= 40) break;
  }
  return JSON.stringify({url: location.href, title: document.title, links: out});
}"""
MAX_READ = 8_000
NOT_A_FIELD = "is not an <input>"
# Rich editors (WhatsApp, Teams, comment boxes) are contenteditable divs with a separate placeholder: when the
# model picks the placeholder, the harness focuses the editable field nearest to it and types there.
FOCUS_NEAREST_JS = """(el) => {
  const sel = '[contenteditable="true"],[contenteditable=""],[role=textbox],textarea,' +
              'input:not([type=hidden]):not([type=checkbox]):not([type=radio]):not([type=button]):not([type=submit])';
  let node = el, field = el.matches(sel) ? el : el.querySelector(sel);
  for (let i = 0; !field && node && i < 6; i++) { node = node.parentElement; field = node && node.querySelector(sel); }
  if (!field) return JSON.stringify({field: null});
  field.focus();
  // Typing replaces what is there, like fill does.
  if (field.select) field.select(); else window.getSelection().selectAllChildren(field);
  return JSON.stringify({field: (field.getAttribute('aria-label') || field.getAttribute('placeholder') ||
                                 field.tagName.toLowerCase()).slice(0, 80)});
}"""


def fill_nearest(target: str, text: str, submit: bool, reason: str = "was not a text field") -> str | None:
    """Focus the nearest editable field and type (new lines as Shift+Enter, so chats do not send early)."""
    data = run_eval(FOCUS_NEAREST_JS, target)
    if not isinstance(data, dict) or not data.get("field"):
        return None
    output = ""
    for number, line in enumerate(text.split("\n")):
        if number:
            output = _cli(build_argv("press", ["Shift+Enter"]))
        if line:
            output = _cli(build_argv("type", [line]))
        if output.startswith(("Error", "### Error")):
            return output
    if submit:
        output = _cli(build_argv("press", ["Enter"]))
    how = "focused the nearest one" if reason == "was not a text field" else "focused it"
    return (f"### {target} {reason}, so the harness {how} (\"{data['field']}\") and typed the text"
            f"{' and pressed Enter' if submit else ''}.\n{output}")


CONTEXT_JS = """el => {
  globalThis.__browserHarnessConversationHeader = el;
  return JSON.stringify({
  label: (el.getAttribute('aria-label') || el.innerText || el.textContent || '').trim(),
  url: location.href, title: document.title, page: document.body.innerText
  });
}"""
COMPOSER_JS = """el => {
  const header = globalThis.__browserHarnessConversationHeader;
  let root = el.parentElement;
  while (root && header && !root.contains(header)) root = root.parentElement;
  return JSON.stringify({
  editable: el.matches('input,textarea,[contenteditable="true"],[contenteditable=""],[role="textbox"]'),
  text: el.value === undefined ? el.innerText : el.value,
  contenteditable: el.isContentEditable, url: location.href, title: document.title,
  page: (root || document.body).innerText
  });
}"""


def normalized_text(text: str) -> str:
    return " ".join(text.split())


def text_digest(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def canonical_draft(text: str) -> str:
    return "\n".join(line.rstrip(" \t") for line in text.replace("\r\n", "\n").replace("\r", "\n")
                     .replace("\u00a0", " ").split("\n")).rstrip("\n")


def compose_message(target: str, text: str, recipient: str, context_ref: str) -> str:
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_DRAFT:
        raise ValueError(f"compose needs a nonempty text field of at most {MAX_DRAFT} characters.")
    if not recipient.strip() or not context_ref.strip():
        raise ValueError("compose needs recipient and context_ref from the visible chat header.")
    context = run_eval(CONTEXT_JS, context_ref)
    field = run_eval(COMPOSER_JS, target)
    if not isinstance(context, dict) or not isinstance(field, dict):
        raise RuntimeError("Could not inspect the recipient and message composer. Use find for fresh refs.")
    if normalized_text(context.get("label", "")) != normalized_text(recipient):
        raise ValueError("The visible chat header does not match recipient. Do not type; select the correct chat.")
    if not field.get("editable"):
        raise ValueError("compose requires the textbox itself, not its placeholder.")
    if _draft.get("submitted") and not _draft.get("verified"):
        raise ValueError("Verify the previous message before composing another; do not risk a duplicate send.")
    expected = normalized_text(text)
    baseline = normalized_text(field.get("page", context.get("page", ""))).count(expected)
    if field.get("contenteditable") and normalized_text(field.get("text", "")) == expected:
        baseline = max(0, baseline - 1)
    selector = build_argv("fill", [target, ""])[len(base_argv()) + 1]
    focused = run_eval(FOCUS_NEAREST_JS, selector)
    if not isinstance(focused, dict) or not focused.get("field"):
        raise RuntimeError("Could not focus the message textbox. No send was attempted.")
    for number, line in enumerate(text.replace("\r\n", "\n").replace("\r", "\n").split("\n")):
        if number:
            result = _cli(build_argv("press", ["Shift+Enter"]))
            if navigation_error(result):
                raise RuntimeError("Draft entry failed. No send was attempted: " + navigation_error(result))
        for start in range(0, len(line), MAX_ARGUMENT):
            result = _cli(build_argv("type", [line[start:start + MAX_ARGUMENT]]))
            if navigation_error(result):
                raise RuntimeError("Draft entry failed. No send was attempted: " + navigation_error(result))
    actual = run_eval(COMPOSER_JS, target)
    if not isinstance(actual, dict) or canonical_draft(actual.get("text", "")) != canonical_draft(text):
        _draft.clear()
        raise RuntimeError("The complete draft did not match the requested text. Do not send; inspect the composer.")
    _draft.clear()
    _draft.update(text=text, recipient=recipient, url=context.get("url"), baseline=baseline,
                  submitted=False, verified=False, target=target, context_ref=context_ref)
    return page_header(context) + f"### Draft composed: {len(text)} characters in one unsent message.\n" + (
        "- Draft SHA256: " + text_digest(text) + "\n- Draft recipient: " + recipient
        + "\n[harness] The complete draft was checked. Press Enter once, then verify-message.")


def verify_message(target: str, context_ref: str) -> str:
    if not _draft or not _draft.get("submitted"):
        raise ValueError("No composed message was submitted in this browser session.")
    if not context_ref:
        raise ValueError("verify-message needs the fresh chat-header context_ref.")
    context = run_eval(CONTEXT_JS, context_ref)
    if (not isinstance(context, dict) or context.get("url") != _draft["url"]
            or normalized_text(context.get("label", "")) != normalized_text(_draft["recipient"])):
        raise RuntimeError("Recipient verification failed. Do not resend; inspect the current chat.")
    expected = normalized_text(_draft["text"])
    deadline = time.monotonic() + 10
    while True:
        state = run_eval(COMPOSER_JS, target)
        if not isinstance(state, dict):
            raise RuntimeError("Could not verify the composer. Use a fresh textbox ref; do not resend.")
        occurrences = normalized_text(state.get("page", "")).count(expected) - _draft["baseline"]
        empty = not normalized_text(state.get("text", ""))
        if state.get("url") == _draft["url"] and state.get("editable") and empty and occurrences == 1:
            _draft["verified"] = True
            evidence = {"text_sha256": text_digest(_draft["text"]), "recipient": _draft["recipient"],
                        "new_messages": 1, "composer_empty": True, "url": _draft["url"]}
            return page_header(state) + "- Message evidence: " + json.dumps(evidence, ensure_ascii=False)
        if time.monotonic() >= deadline:
            raise RuntimeError("Message not verified: expected one new complete message and an empty composer. "
                               "Do not resend automatically; inspect the chat or ask the user.")
        time.sleep(0.25)


def run_eval(js: str, target: str = "") -> dict | str:
    """Runs the fixed JS on the page (or on the element `target`) and returns the decoded JSON, or the CLI error."""
    argv = [*base_argv(), "--raw", "eval", js, *([target] if target else [])]
    completed = subprocess.run(argv, cwd=HERE, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=TIMEOUT_SECONDS)
    output = (completed.stdout + completed.stderr).strip()
    try:
        value = json.loads(output)
        return json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return output or "(no output)"


def page_header(data: dict) -> str:
    title = re.sub(r"[\r\n]+", " ", str(data.get("title") or ""))
    return f"### Page\n- Page URL: {data.get('url', '')}\n- Page Title: {title}\n"


def read_text(selector: str = "") -> str:
    """`read`: plain text of the main region. Much cheaper than the accessibility tree."""
    data = run_eval(READ_JS % json.dumps(selector or ""))
    if not isinstance(data, dict):
        return data
    text = re.sub(r"\n\s*\n+", "\n\n", data.get("text") or "").strip()
    if len(text) > MAX_READ:
        text = text[:MAX_READ] + "\n...[text truncated; the rest was not read]"
    published = re.sub(r"[\r\n]+", " ", str(data.get("published") or ""))
    metadata = f"- Published date: {published}\n" if published else ""
    if data.get("article") is True:
        metadata += "- Source type: article\n"
    canonical = re.sub(r"[\r\n]+", "", str(data.get("canonical") or ""))
    if canonical and canonical != data.get("url"):
        metadata += "- Canonical URL: " + canonical + "\n"
    return page_header(data) + metadata + "### Text\n" + (text or "(page has no text)")


def list_links() -> str:
    """`links`: headlines/links of the main region with URLs, to visit each item with goto."""
    data = run_eval(LINKS_JS)
    if not isinstance(data, dict):
        return data
    origin = re.match(r"https?://[^/]+", data.get("url", ""))
    lines = []
    for text, href in data.get("links") or []:
        if origin and href.startswith(origin.group(0) + "/"):
            href = href[len(origin.group(0)):]
        lines.append(f"- {text} | {href}")
    return page_header(data) + "### Links\n" + ("\n".join(lines) or "(no links with text)")


def _cli(argv: list[str]) -> str:
    global _browser_open
    completed = subprocess.run(
        argv, cwd=HERE, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=TIMEOUT_SECONDS,
    )
    output = (completed.stdout + completed.stderr).strip()
    cmd = argv[3] if len(argv) > 3 else ""
    if cmd == "close":
        _browser_open = False
        _last.update(tree="", url="")
        _draft.clear()
    elif "is not open" in output:
        _browser_open = False
    elif not output.startswith(("Error", "### Error")):
        _browser_open = True
    return output


def wait_for(args: list[str]) -> str:
    """`wait <text>` (until it shows up, max 30 s) or `wait <seconds>`: for pages that are still loading."""
    target = " ".join(arg for arg in args if isinstance(arg, str)).strip()
    if not target:
        return "Error: use wait <text you expect> (e.g. wait Minhas viagens) or wait <seconds>."
    if target.isdigit():
        seconds = min(int(target), MAX_WAIT)
        time.sleep(seconds)
        data = run_eval("() => JSON.stringify({url: location.href, title: document.title})")
        return (page_header(data) if isinstance(data, dict) else "") + f"### Waited {seconds}s."
    data = run_eval(WAIT_JS % (json.dumps(target), MAX_WAIT * 1000))
    if not isinstance(data, dict):
        return data
    if data.get("found"):
        return page_header(data) + f'### Waited {data.get("waited")}s: "{target}" is on the page now.'
    return (f'Error: after {data.get("waited")}s, "{target}" is still not on the page (it may still be loading, '
            "need a login, or the label is different).\n" + page_header(data))


def run_cli(command: str, args: list[str]) -> str:
    submitting = ((command == "press" and args[:1] == ["Enter"]) or (
        command == "click" and args and re.search(r"\b(send|enviar|submit|post|publish)\b",
                                                 element_label(args[0]), re.I)))
    if submitting and _draft:
        if _draft.get("submitted"):
            return VALIDATION_ERROR + "This draft was already submitted. Verify it; do not send it again."
        field = run_eval(COMPOSER_JS, _draft["target"])
        context = run_eval(CONTEXT_JS, _draft["context_ref"])
        if (not isinstance(field, dict) or not isinstance(context, dict)
                or normalized_text(field.get("text", "")) != normalized_text(_draft["text"])
                or normalized_text(context.get("label", "")) != normalized_text(_draft["recipient"])
                or context.get("url") != _draft["url"]):
            return VALIDATION_ERROR + "The draft or recipient changed. No send was performed; compose with fresh refs."
    output = _run_cli(command, args)
    if command in NAVIGATIONS and not output.startswith(("Error", "### Error")):
        _draft.clear()
        run_eval(SETTLE_JS)  # let the page finish loading before the model looks at it
    if not output.startswith(("Error", "### Error")) and _draft:
        if submitting:
            _draft["submitted"] = True
        elif command in ("fill", "type"):
            _draft.clear()
    return output


def _run_cli(command: str, args: list[str]) -> str:
    if command == "wait":
        return wait_for(args)
    if command == "read":
        target = next((arg for arg in args[:1] if isinstance(arg, str) and not arg.startswith("-")), "")
        if target.startswith(("http://", "https://")):
            # `read <url>`: opens the page and reads it in one call (saves a model call per article).
            opened = run_cli("open", [target])
            if opened.startswith(("Error", "### Error")):
                return opened
            return read_text()
        return read_text(target)
    if command == "links":
        return list_links()
    if command == "find" and args and " | " in args[0]:
        # `find A | B`: alternatives (e.g. the label in two languages) tried in ONE model call.
        alternatives = [text.strip() for text in args[0].split(" | ") if text.strip()][:4]
        output = ""
        for text in alternatives:
            output = _finish("find", _cli(build_argv("find", [text, *args[1:]])))
            if "No matches found" not in output and not output.startswith(("Error", "### Error")):
                return f"### Matched \"{text}\" (tried: {' | '.join(alternatives)})\n{output}"
        return f"### None of these matched: {' | '.join(alternatives)}\n{output}"
    if command == "open" and args:
        # `open` restarts an already open browser; `goto` reuses the window (and its tabs).
        output = _cli(build_argv("goto", args))
        if "is not open" not in output:
            return _finish("goto", output)
    if command == "goto" and args:
        output = _cli(build_argv("goto", args))
        if "is not open" not in output:
            return _finish("goto", output)
        # The browser was closed: open it with the URL instead of failing (saves a model call).
        return _finish("open", _cli(build_argv("open", args)))
    if command == "fill" and len(args) >= 2:
        build_argv(command, args)
        target = f"text={json.dumps(args[0].strip(chr(34)))}" if looks_like_label(args[0]) else args[0]
        submit = "--submit" in args[2:]
        if "\n" in args[1]:
            # Multi-line text: typed line by line with Shift+Enter (Enter would send a chat message halfway).
            output = fill_nearest(target, args[1], submit, reason="has multi-line text")
            if output:
                return _finish("type", output)
        output = _cli(build_argv(command, args))
        if NOT_A_FIELD in output:
            output = fill_nearest(target, args[1], submit) or output
            return _finish("type", output)
        return _finish(command, output)
    return _finish(command, _cli(build_argv(command, args)))


def _finish(command: str, raw: str) -> str:
    """Remembers the page tree and, after an interaction, answers only with what changed."""
    tree = snapshot_tree(raw)
    url = page_url(raw)
    before, before_url = _last["tree"], _last["url"]
    if tree:
        _last.update(tree=tree, url=url or before_url)
    output = compact_snapshot(re.sub(r"### Snapshot\n- \[Snapshot\]\([^)]*\)\n?", "", raw)).strip()
    if command == "find":
        output = "\n".join(line if len(line) <= 300 or line.lstrip().startswith("- /url:")
                           else line[:260] + " ...[label shortened; use read for content]"
                           for line in output.splitlines())
    if command in INTERACTIONS and not output.startswith(("Error", "### Error")) and tree:
        if url and before_url and url != before_url:
            output += "\n### Navigated to a new page. Use `read` for its text or `find <text>` to locate elements."
        elif before:
            changed = changes(before, tree)
            output += ("\n### Changes on the page after this action (new or updated elements)\n" + changed
                       if changed else f"\n{NO_CHANGE}. Do not repeat it; try another way or take a screenshot.")
    if command == "snapshot" and len(output) > MAX_OUTPUT:
        output = interactive_only(output)
    if len(output) > MAX_OUTPUT:
        output = output[:MAX_OUTPUT] + (
            "\n...[output truncated. To locate an element use `find <visible text>`; to see the page, "
            "`screenshot`; to read content, `read`]"
        )
    return output or "(no output)"


def snapshot_tree(output: str) -> str:
    """Full page tree of a CLI output: inline (snapshot) or in the file it points to (open/goto/click)."""
    inline = re.search(r"```yaml\n(.*?)```", output, flags=re.S)
    if inline:
        return inline.group(1)
    link = re.search(r"\[Snapshot\]\(([^)]+)\)", output)
    if link:
        path = HERE / link.group(1).replace("\\", "/")
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


def element_label(target: str) -> str:
    """Line of the latest page tree for a ref (e.g. '- button "Send" [ref=e42]'), or the target itself."""
    if not re.fullmatch(r"(f\d+)?e\d+", target or ""):
        return target or ""
    line = next((line for line in _last["tree"].splitlines() if f"[ref={target}]" in line), "")
    return re.sub(r"\s*\[(ref=\w+|cursor=\w+)\]", "", line).strip()[:160] or target


def page_url(output: str) -> str:
    match = re.search(r"^- Page URL: (\S+)", output, flags=re.M)
    return match.group(1) if match else ""


def _key(line: str) -> str:
    return re.sub(r"\s*\[(ref=\w+|cursor=\w+|active)\]", "", line).strip()


def changes(before: str, after: str) -> str:
    """Lines of the new tree that were not in the old one (e.g. a menu that opened), with their refs."""
    remaining = Counter(_key(line) for line in before.splitlines())
    new = []
    for line in after.splitlines():
        key = _key(line)
        if remaining[key] > 0:
            remaining[key] -= 1
        elif key and not re.fullmatch(r"- (generic|group|list|listitem|region)?:?", key):
            new.append(line)
    if not new:
        return ""
    new = new[:MAX_CHANGES + 1]
    indent = min(len(line) - len(line.lstrip(" ")) for line in new)
    body = compact_snapshot("\n".join(line[indent:] for line in new[:MAX_CHANGES]))
    return body + ("\n... (more changed lines; use `find <text>`)" if len(new) > MAX_CHANGES else "")


def interactive_only(output: str) -> str:
    """Big snapshot: keeps only what can be clicked/filled, headings and link URLs."""
    head, _, tree = output.partition("### Snapshot")
    kept = [line for line in tree.splitlines() if INTERACTIVE.match(line)]
    return (head + "### Snapshot (big page: only links, buttons, fields and headings)\n" + "\n".join(kept)).strip()


def look() -> tuple[str, str | None]:
    """Screenshot of the viewport as a data URI, for the model to SEE the page (~1k tokens)."""
    SHOT.unlink(missing_ok=True)
    SHOT.parent.mkdir(exist_ok=True)
    output = _cli(build_argv("screenshot", []) + [f"--filename={SHOT.relative_to(HERE).as_posix()}"])
    if not SHOT.is_file():
        lines = [line for line in output.splitlines() if line.strip()]
        return (f"Error: screenshot failed. {lines[-1] if lines else ''}".strip(), None)
    data = base64.b64encode(SHOT.read_bytes()).decode()
    text = "### Screenshot\n- The image of the current screen is attached below."
    if _last["url"]:
        text += f"\n- Page URL: {_last['url']}"
    return text, f"data:image/jpeg;base64,{data}"


def compact_snapshot(text: str) -> str:
    """Removes what only costs tokens from a snapshot: cursor, empty containers and half the indentation.

    Same-site links get a relative /url; refs stay intact.
    """
    page = re.search(r"^- Page URL: (https?://[^/\s]+)", text, flags=re.M)
    origin = page.group(1) if page else None
    lines = []
    for line in text.splitlines():
        if re.match(r"^\s*- generic( \[ref=\w+\])?:?\s*$", line):
            continue
        if origin and re.match(r"^\s*- /url:", line):
            line = line.replace(f"/url: {origin}/", "/url: /")
        line = line.replace(" [cursor=pointer]", "")
        body = line.lstrip(" ")
        lines.append(" " * ((len(line) - len(body)) // 2) + body)
    return "\n".join(lines)


def summarize_output(text: str) -> str:
    """Short version of an old output: only URL/title, or the beginning of the text."""
    keep = [line for line in text.splitlines()
            if line.startswith(("- Page URL", "- Page Title", "- Saved to workspace"))]
    summary = "\n".join(keep) if keep else text[:200]
    return summary + "\n[content removed to save tokens; use your notes or the saved file]"


def dispatch(raw_arguments: str) -> str:
    """Runs the tool and returns text; errors go back to the model so it can correct itself."""
    try:
        arguments = json.loads(raw_arguments or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object.")
        command, args = normalize_command(arguments.get("command", ""), arguments.get("args", []))
        build_argv(command, args)
        if command in ("compose", "verify-message"):
            if len(args) != 1 or not args[0].strip():
                raise ValueError(f"{command} needs args=[textbox ref].")
            recipient, context_ref = arguments.get("recipient", ""), arguments.get("context_ref", "")
            if not isinstance(recipient, str) or not isinstance(context_ref, str):
                raise ValueError("recipient and context_ref must be text.")
            if len(recipient) > MAX_ARGUMENT or len(context_ref) > MAX_ARGUMENT:
                raise ValueError("recipient and context_ref exceed the argument limit.")
            return (compose_message(args[0], arguments.get("text"), recipient, context_ref) if command == "compose"
                    else verify_message(args[0], context_ref))
        return run_cli(command, args)
    except subprocess.TimeoutExpired:
        return f"Error: the command exceeded {TIMEOUT_SECONDS}s."
    except ValueError as error:
        return VALIDATION_ERROR + str(error)
    except (RuntimeError, OSError, AttributeError) as error:
        return f"Error: {error}"


def close_browser() -> str:
    global _browser_open
    _browser_open = False
    _last.update(tree="", url="")
    _draft.clear()
    return dispatch(json.dumps({"command": "close", "args": []}))


def close_browser_if_open() -> str:
    """Closes the browser only if it was opened in the current session."""
    if not _browser_open and not _last["url"]:
        return ""
    return close_browser()


def normalize_url(text: str) -> str:
    """Completes what the user typed: 'linkedin' -> https://www.linkedin.com, 'x.com/a' -> https://x.com/a."""
    text = text.strip()
    if not text:
        return ""
    if not re.match(r"https?://", text, flags=re.I):
        text = "https://" + text
    host = urlparse(text).hostname or ""
    if host and "." not in host and host != "localhost":
        text = text.replace(f"//{host}", f"//www.{host}.com", 1)
    return text


def navigation_error(result: str) -> str:
    """First useful line of a navigation error, or '' when it worked."""
    if not result.startswith("Error") and "### Error" not in result:
        return ""
    lines = [line.strip() for line in result.splitlines()]
    return next((line for line in lines if line.startswith("Error")), "Could not open the page.")


def navigate(url: str) -> str:
    """Opens the URL directly, without the model (used by /login)."""
    result = dispatch(json.dumps({"command": "goto", "args": [url]}))
    if "is not open" in result:
        result = dispatch(json.dumps({"command": "open", "args": [url]}))
    return result
