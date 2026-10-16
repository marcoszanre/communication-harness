"""Opt-in real Chromium test against a synthetic loopback chat, never a personal account."""

import json
import os
import re
import tempfile
import threading
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import browser_tool

TEXT = ("- Comprehensive synthetic summary: " + "Unicode content é🌍. " * 250
        + "\n\nOutcome paragraph one: all content must be preserved."
        + "\n\nOutcome paragraph two: input chunks must not become separate messages."
        + "\n--literal-text-not-a-cli-option")
HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Synthetic chat</title><meta property="article:published_time" content="2026-10-13T10:00:00-03:00">
</head><body><main><h1 id="recipient">Example (you)</h1><div id="messages"></div>
<div id="message" role="textbox" contenteditable="true" aria-label="Message" style="white-space:pre-wrap"></div>
<button id="send">Send</button></main><script>
const box = document.getElementById('message'), messages = document.getElementById('messages');
const add = text => { const item = document.createElement('p'); item.className = 'message';
                     item.style.whiteSpace = 'pre-wrap';
                     item.textContent = text; messages.appendChild(item); };
add(%s);
const send = () => { const text = box.innerText; if (!text.trim()) return;
                    box.textContent = ''; add(text); };
box.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); send(); }
});
document.getElementById('send').addEventListener('click', send);
</script></body></html>""" % json.dumps(TEXT, ensure_ascii=False)


@unittest.skipUnless(os.getenv("BROWSER_HARNESS_INTEGRATION") == "1",
                     "Set BROWSER_HARNESS_INTEGRATION=1 for the isolated real-browser test.")
class BrowserIntegrationTests(unittest.TestCase):
    def test_full_message_is_preserved_and_sent_exactly_once(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        session = "harness-regression-" + uuid.uuid4().hex
        previous = dict(browser_tool._last)
        browser_tool._draft.clear()
        self.addCleanup(browser_tool._draft.clear)
        self.addCleanup(browser_tool._last.update, previous)
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(browser_tool, "SESSION", session):
            config = Path(folder) / "browser.json"
            config.write_text(json.dumps({"browser": {"browserName": "chromium",
                                                      "launchOptions": {"headless": True},
                                                      "userDataDir": str(Path(folder) / "profile")}}), encoding="utf-8")
            url = f"http://127.0.0.1:{server.server_port}/"
            opened = browser_tool._cli([*browser_tool.base_argv(), "open", url, f"--config={config}"])
            self.assertFalse(browser_tool.navigation_error(opened), opened)
            browser_tool._finish("open", opened)
            try:
                header = self.ref("Example (you)", "heading")
                textbox = self.ref("Message", "textbox")
                with self.assertRaisesRegex(ValueError, "does not match recipient"):
                    browser_tool.compose_message(textbox, TEXT, "Wrong recipient", header)
                composed = browser_tool.compose_message(textbox, TEXT, "Example (you)", header)
                self.assertIn("one unsent message", composed)
                state = browser_tool.run_eval("""() => JSON.stringify({
                    count: document.querySelectorAll('.message').length,
                    text: document.getElementById('message').innerText
                })""")
                self.assertEqual(state["count"], 1, "composing must not send any input chunk")
                self.assertEqual(browser_tool.canonical_draft(state["text"]), browser_tool.canonical_draft(TEXT))
                with self.assertRaisesRegex(ValueError, "No composed message was submitted"):
                    browser_tool.verify_message(textbox, header)
                sent = browser_tool.run_cli("press", ["Enter"])
                self.assertFalse(browser_tool.navigation_error(sent), sent)
                verified = browser_tool.verify_message(self.ref("Message", "textbox"),
                                                        self.ref("Example (you)", "heading"))
                self.assertIn('"new_messages": 1', verified)
                state = browser_tool.run_eval("""() => JSON.stringify({
                    count: document.querySelectorAll('.message').length,
                    last: document.querySelector('.message:last-child').textContent,
                    text: document.getElementById('message').innerText
                })""")
                self.assertEqual(state["count"], 2, "one old message plus exactly one new message")
                self.assertEqual(browser_tool.canonical_draft(state["last"]), browser_tool.canonical_draft(TEXT))
                self.assertEqual(state["text"], "")
                self.assertIn("already submitted", browser_tool.run_cli("press", ["Enter"]))
                self.assertIn("- Published date: 2026-10-13", browser_tool.read_text())
            finally:
                closed = browser_tool.run_cli("close", [])
                self.assertFalse(browser_tool.navigation_error(closed), closed)

    def ref(self, label, role):
        result = browser_tool.run_cli("find", [label])
        for line in result.splitlines():
            if re.search(rf"- {role}\b", line):
                match = re.search(r"\[ref=((?:f\d+)?e\d+)\]", line)
                if match:
                    return match.group(1)
        self.fail(f"No {role} ref for {label}: {result}")


if __name__ == "__main__":
    unittest.main()
