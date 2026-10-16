import os

os.environ["BROWSER_HARNESS_PLANNER"] = "0"  # the planner is a real model: off in tests

import json
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest import mock

import browser_tool
import playbooks
import skills
from agent import ALREADY_READ, BLOCKED, FORCE_PLAN, BrowserAgent
from browser_tool import text_digest
from foundry import ContentFiltered, ToolCall, Turn
from memory import Memory
from plan import Plan
from session_log import SessionLog
from workspace import Workspace

PAGE_A = "- Page URL: https://a.com/"


def browser(command, *args, call_id="c"):
    return ToolCall(call_id, "browser", json.dumps({"command": command, "args": list(args)}))


def tasks(call_id="t", **arguments):
    return ToolCall(call_id, "tasks", json.dumps(arguments))


def noop(kind, text):
    pass


class FakeModel:
    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self.tool_choices = []
        self.tools = []

    def complete(self, messages, tools, emit, cancel, tool_choice="auto"):
        self.requests.append([dict(message) for message in messages])
        self.tool_choices.append(tool_choice)
        self.tools.append(tools)
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


class BrowserToolTests(unittest.TestCase):
    def test_fill_on_a_placeholder_focuses_the_nearest_field_and_types(self):
        calls = []

        def fake_cli(argv):
            calls.append(argv[-2:])
            if argv[-3] == "fill":
                return "### Error\nError: Element is not an <input>, <textarea>, <select> or [contenteditable]"
            return "### Ran Playwright code"

        with mock.patch.object(browser_tool, "_cli", fake_cli), \
                mock.patch.object(browser_tool, "run_eval", return_value={"field": "Type a message"}):
            out = browser_tool.run_cli("fill", ["e9", "hello"])
        self.assertIn('focused the nearest one ("Type a message")', out)
        self.assertEqual(calls[-1], ["type", "hello"])

    def test_multi_line_fill_uses_shift_enter_and_never_sends_early(self):
        calls = []
        with mock.patch.object(browser_tool, "_cli", lambda argv: calls.append(argv[-2:]) or "ok"), \
                mock.patch.object(browser_tool, "run_eval", return_value={"field": "Message"}):
            out = browser_tool.run_cli("fill", ["e9", "line 1\nline 2"])
        self.assertEqual(calls, [["type", "line 1"], ["press", "Shift+Enter"], ["type", "line 2"]])
        self.assertIn("has multi-line text", out)
        self.assertNotIn(["press", "Enter"], calls)

    def test_click_answers_with_changes_and_open_reuses_window(self):
        before = '- button "My trips" [ref=e25]\n- link "Help" [ref=e31]:\n  - /url: /help\n'
        after = before.replace("[ref=e25]", "[expanded] [ref=e25]") + (
            '- menu [ref=e872]:\n  - menuitem [ref=e874]:\n    - link [ref=e875]:\n'
            '      - /url: /trips\n      - strong [ref=e883]: Manage your trips\n')
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(browser_tool, "HERE", Path(folder)):
            (Path(folder) / "after.yml").write_text(after, encoding="utf-8")
            browser_tool._last.update(tree=before, url="https://a.com/")
            calls = []

            def fake_cli(argv):
                calls.append(argv[3])
                if argv[3] == "goto":
                    return "### Page\n- Page URL: https://a.com/\n### Snapshot\n```yaml\n" + before + "```"
                return "### Page\n- Page URL: https://a.com/\n### Snapshot\n- [Snapshot](after.yml)"

            with mock.patch.object(browser_tool, "_cli", fake_cli):
                out = browser_tool.run_cli("click", ["e25"])
                self.assertIn("Changes on the page", out)
                self.assertIn("strong [ref=e883]: Manage your trips", out)
                self.assertNotIn('link "Help"', out)
                self.assertNotIn("after.yml", out)
                self.assertIn(browser_tool.NO_CHANGE, browser_tool.run_cli("click", ["e25"]))
                browser_tool.run_cli("open", ["https://a.com/"])
                self.assertEqual(calls, ["click", "click", "goto"])

    def test_big_snapshot_keeps_only_interactive_lines(self):
        tree = "### Page\n### Snapshot\n" + '- paragraph: long text\n- link "Go" [ref=e2]:\n  - /url: /go\n' * 50
        out = browser_tool.interactive_only(tree)
        self.assertNotIn("paragraph", out)
        self.assertIn('link "Go" [ref=e2]', out)

    def test_builds_command_with_fixed_session(self):
        self.assertEqual(browser_tool.build_argv("click", ["Minhas viagens"])[-1], 'text="Minhas viagens"')
        self.assertEqual(browser_tool.build_argv("click", ["f60e25"])[-1], "f60e25")
        self.assertEqual(browser_tool.build_argv("fill", ["#q", "oi"])[-2:], ["#q", "oi"])
        argv = browser_tool.build_argv("open", ["https://example.com"])
        self.assertEqual(argv[2:], [f"-s={browser_tool.SESSION}", "open", "https://example.com"])

    def test_rejects_unsafe_requests(self):
        for command, args in [
            ("eval", ["document.cookie"]),
            ("run-code", ["x"]),
            ("goto", ["file:///C:/Windows/win.ini"]),
            ("snapshot", ["--filename=C:/temp/x.yml"]),
            ("click", ["-s=outra-sessao"]),
            ("click", ["a", "b", "c", "d"]),
            ("snapshot", ["--depth=4;x"]),
        ]:
            with self.assertRaises(ValueError, msg=command):
                browser_tool.build_argv(command, args)

    def test_accepts_submit_depth_and_command_with_spaces(self):
        self.assertEqual(browser_tool.build_argv("fill", ["e3", "x", "--submit"])[-3:], ["e3", "x", "--submit"])
        self.assertEqual(browser_tool.build_argv("snapshot --depth=4", [])[-2:], ["snapshot", "--depth=4"])

    def test_dispatch_returns_errors_as_text(self):
        self.assertTrue(browser_tool.dispatch("{").startswith("Error"))
        self.assertTrue(browser_tool.dispatch('{"command":"eval","args":[]}').startswith("Error"))

    def test_normalize_url_and_navigation_error(self):
        self.assertEqual(browser_tool.normalize_url("linkedin"), "https://www.linkedin.com")
        self.assertEqual(browser_tool.normalize_url(" linkedin/feed "), "https://www.linkedin.com/feed")
        self.assertEqual(browser_tool.normalize_url("linkedin.com"), "https://linkedin.com")
        self.assertEqual(browser_tool.normalize_url("http://localhost:3000"), "http://localhost:3000")
        failed = "### Browser `x` opened\n### Error\nError: net::ERR_NAME_NOT_RESOLVED at https://a/\nCall log:"
        self.assertEqual(browser_tool.navigation_error(failed), "Error: net::ERR_NAME_NOT_RESOLVED at https://a/")
        self.assertEqual(browser_tool.navigation_error("### Page\n- Page URL: https://a.com/"), "")

    def test_compact_snapshot_keeps_refs_and_drops_noise(self):
        raw = "\n".join([
            "- Page URL: https://example.com/home",
            "- main [ref=e1]:",
            "  - generic [ref=e2]:",
            "    - link \"Noticia\" [ref=e3] [cursor=pointer]:",
            "      - /url: https://example.com/noticia",
        ])
        self.assertEqual(browser_tool.compact_snapshot(raw), "\n".join([
            "- Page URL: https://example.com/home",
            "- main [ref=e1]:",
            "  - link \"Noticia\" [ref=e3]:",
            "   - /url: /noticia",
        ]))

    def test_read_and_links_format_text(self):
        page = {"url": "https://a.com/ia/", "title": "IA"}
        with mock.patch.object(browser_tool, "run_eval", return_value={**page, "text": "Titulo\n\n\n\nCorpo"}):
            self.assertEqual(browser_tool.run_cli("read", []),
                             "### Page\n- Page URL: https://a.com/ia/\n- Page Title: IA\n### Text\nTitulo\n\nCorpo")
        links = {**page, "links": [["Manchete longa sobre IA e dados", "https://a.com/n/1"],
                                   ["Outra manchete de outro site aqui", "https://b.com/x"]]}
        with mock.patch.object(browser_tool, "run_eval", return_value=links):
            out = browser_tool.run_cli("links", [])
        self.assertIn("- Manchete longa sobre IA e dados | /n/1", out)
        self.assertIn("| https://b.com/x", out)
        with mock.patch.object(browser_tool, "run_eval", return_value="### Error\nbrowser is not open"):
            self.assertTrue(browser_tool.run_cli("read", []).startswith("### Error"))


class SkillTests(unittest.TestCase):
    def test_catalog_index_and_section(self):
        self.assertIn("playwright-cli", skills.catalog())
        self.assertIn("Snapshots", skills.read_skill("playwright-cli"))
        section = skills.read_skill("playwright-cli", "snapshots")
        self.assertTrue(section.startswith("## Snapshots"))
        self.assertTrue(skills.read_skill("nao-existe").startswith("Error"))


class PlanTests(unittest.TestCase):
    def test_plan_lifecycle_and_validation(self):
        plan = Plan()
        self.assertTrue(plan.apply('{"action":"done","index":1}').startswith("Error"))
        plan.apply(json.dumps({"action": "plan", "items": ["Resumir CNN", "Resumir LinkedIn", " "]}))
        self.assertEqual(plan.pending(), [1, 2])
        plan.apply('{"action":"start","index":1}')
        out = plan.apply(json.dumps({"action": "done", "index": 1, "note": "Materia sobre Anthropic"}))
        self.assertIn("1. [done] [browser] Resumir CNN - Materia sobre Anthropic", out)
        self.assertEqual(plan.pending(), [2])
        self.assertTrue(plan.apply('{"action":"done","index":5}').startswith("Error"))
        plan.apply('{"action":"failed","index":2,"note":"sem login"}')
        self.assertEqual(plan.summary(), "1/2 done, 1 failed")
        self.assertIn("✅ 1. Resumir CNN", plan.render_tui())
        self.assertIn("⚠ 2. Resumir LinkedIn", plan.render_tui())

    def test_done_needs_evidence_and_confidence_and_ask_waits_for_user(self):
        plan = Plan()
        plan.apply(json.dumps({"action": "plan", "items": ["Open My trips", "Open booking details"]}))
        self.assertIn("needs a note", plan.apply('{"action":"done","index":1}'))
        self.assertIn("confidence is low",
                      plan.apply('{"action":"done","index":1,"note":"maybe","confidence":"low"}'))
        self.assertEqual(plan.pending(), [1, 2])
        plan.apply('{"action":"done","index":1,"note":"saw 1 trip","confidence":"high"}')
        plan.apply('{"action":"ask","index":2,"note":"Which booking?"}')
        self.assertEqual(plan.pending(), [])
        self.assertEqual(plan.unfinished(), [2])
        self.assertEqual(plan.summary(), "1/2 done, 1 waiting for you")
        self.assertIn("❓ 2. Open booking details", plan.render_tui())
        plan.apply(json.dumps({"action": "plan", "items": ["Open My trips", "Open booking details", "Report"]}))
        self.assertEqual(plan.summary(), "1/3 done, 1 waiting for you")


class AgentLoopTests(unittest.TestCase):
    def test_stuck_ladder_diagnosis_screenshot_reflect_then_ask(self):
        no_change = PAGE_A + "\n" + browser_tool.NO_CHANGE + ". Do not repeat it."
        model = FakeModel([Turn(tool_calls=[browser("click", f"e{i}", call_id=f"c{i}")]) for i in range(1, 6)]
                          + [Turn(tool_calls=[browser("find", "Trips", call_id="f1")]), Turn(content="ok")])
        shots = []
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A if "find" in raw else no_change,
                             look=lambda: (shots.append(1) or "### Screenshot", "data:image/jpeg;base64,AAA"))
        agent.run("x", noop, Event())
        self.assertIn("That did not work", model.requests[1][-1]["content"])
        second = model.requests[2]
        self.assertIn("screenshot of the current screen is attached", second[-2]["content"])
        self.assertEqual(second[-1]["content"][1]["image_url"]["url"], "data:image/jpeg;base64,AAA")
        self.assertIn("two DIFFERENT approaches", model.requests[3][-1]["content"])
        self.assertIn("two DIFFERENT approaches", model.requests[4][-1]["content"])
        ask = model.requests[5][-1]["content"]
        self.assertIn("Ask the user now", ask)
        self.assertNotIn("action=failed", ask)
        self.assertEqual(len(shots), 1)
        # The image does not stay in the context after the next step.
        self.assertIn("older screenshot removed", str(model.requests[4]))
        self.assertNotIn("base64,AAA", str(model.requests[4]))

    def test_errors_get_a_diagnosis_for_their_kind(self):
        errors = {"e1": "### Error\n<div> intercepts pointer events", "e2": "### Error\nNo matches found for x"}
        model = FakeModel([Turn(tool_calls=[browser("click", "e1")]),
                           Turn(content="ok")])
        agent = BrowserAgent(model, run_tool=lambda raw: errors["e1"])
        agent.run("x", noop, Event())
        self.assertIn("Something covers the element", model.requests[1][-1]["content"])
        from agent import diagnose
        self.assertIn("shorter or different word", diagnose(errors["e2"]))
        self.assertIn("not an editable field", diagnose("Error: Element is not an <input>, <textarea>"))

    def test_second_screenshot_without_action_is_refused(self):
        model = FakeModel([Turn(tool_calls=[browser("screenshot")]),
                           Turn(tool_calls=[browser("screenshot", call_id="c2")]),
                           Turn(content="ok")])
        shots = []
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A,
                             look=lambda: (shots.append(1) or "### Screenshot", "data:image/jpeg;base64,A"))
        agent.run("x", noop, Event())
        self.assertEqual(len(shots), 1)
        self.assertIn("still valid", model.requests[2][-1]["content"])

    def test_done_starts_next_task(self):
        plan = Plan()
        plan.apply(json.dumps({"action": "plan", "items": ["a", "b"]}))
        plan.apply('{"action":"done","index":1,"note":"ok","confidence":"high"}')
        self.assertEqual(plan.items[1]["status"], "in progress")

    def test_model_can_request_screenshot(self):
        model = FakeModel([Turn(tool_calls=[browser("screenshot")]), Turn(content="I see a menu.")])
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A,
                             look=lambda: ("### Screenshot", "data:image/jpeg;base64,BBB"))
        agent.run("x", noop, Event())
        self.assertEqual(model.requests[1][-1]["content"][1]["image_url"]["url"], "data:image/jpeg;base64,BBB")

    def test_done_right_after_a_failure_is_refused_once_and_failed_asks_first(self):
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["open trips"])]),
                           Turn(tool_calls=[browser("click", "e25")]),
                           Turn(tool_calls=[tasks(call_id="t2", action="done", index=1, note="x",
                                                  confidence="high")]),
                           Turn(tool_calls=[tasks(call_id="t3", action="failed", index=1, note="menu closed")]),
                           Turn(tool_calls=[tasks(call_id="t4", action="ask", index=1,
                                                  note="The menu does not open. Can you open it for me?")]),
                           Turn(content="Can you open the menu?")])
        agent = BrowserAgent(model, run_tool=lambda raw: "### Error\nTimeoutError")
        self.assertEqual(agent.run("x", noop, Event()), "Can you open the menu?")
        self.assertIn("last browser action failed", model.requests[3][-1]["content"])
        self.assertIn("ask the user first", model.requests[4][-1]["content"])
        self.assertEqual(agent.plan.summary(), "0/1 done, 1 waiting for you")

    def test_failed_is_accepted_when_the_model_insists(self):
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["a", "b"])]),
                           Turn(tool_calls=[tasks(call_id="t2", action="failed", index=1, note="no such page")]),
                           Turn(tool_calls=[tasks(call_id="t3", action="failed", index=1, note="no such page")]),
                           Turn(tool_calls=[tasks(call_id="t4", action="ask", index=2, note="Which one?")]),
                           Turn(content="Which one?")])
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A)
        agent.run("x", noop, Event())
        self.assertEqual(agent.plan.summary(), "0/2 done, 1 failed, 1 waiting for you")

    def test_guardrails_need_evidence_from_this_session_to_send_and_verify(self):
        # The hallucination: an OLD message on the page was taken as proof that it was sent.
        old = PAGE_A + "\n- paragraph: Resumo da CNN: Anderson Cooper (old message)"
        new_text = "Resumo da CNN: Google testa chips de IA no espaco"
        compose = ToolCall("compose", "browser", json.dumps({"command": "compose", "args": ["e9"],
                           "text": new_text, "recipient": "Example (you)", "context_ref": "e1"}))
        steps = [tasks(action="plan", items=["Open the self-chat and send the summary", "Verify the summary appears"]),
                 browser("click", "e1", call_id="b1"), browser("find", "Resumo da CNN", call_id="b2"),
                 tasks(call_id="t1", action="done", index=1, note="sent", confidence="high"),     # nothing typed
                 browser("fill", "e9", new_text, call_id="b3"),
                 tasks(call_id="t2", action="done", index=1, note="sent", confidence="high"),     # not submitted
                 compose, browser("press", "Enter", call_id="b4"),
                 tasks(call_id="t3", action="done", index=1, note="sent", confidence="high"),     # not verified
                 tasks(call_id="t4", action="done", index=2, note="seen", confidence="high"),     # not seen yet
                 browser("find", "Resumo da CNN: Google", call_id="b5"),                          # query echo only
                 tasks(call_id="t5", action="done", index=2, note="seen", confidence="high"),     # still not seen
                 browser("verify-message", "e9", call_id="b6"),
                 tasks(call_id="t7", action="done", index=1, note="complete new message verified", confidence="high"),
                 tasks(call_id="t6", action="done", index=2, note="seen", confidence="high")]     # ok
        model = FakeModel([Turn(tool_calls=[call]) for call in steps] + [Turn(content="Enviei.")])

        def run_tool(raw):
            command = json.loads(raw)["command"]
            if command == "compose":
                return PAGE_A + "\n- Draft SHA256: " + text_digest(new_text)
            if command == "verify-message":
                return PAGE_A + "\n- Message evidence: " + json.dumps({
                    "text_sha256": text_digest(new_text), "recipient": "Example (you)",
                    "new_messages": 1, "composer_empty": True, "url": "https://a.com/"})
            if command == "find":
                return 'Found 1 match for "Resumo da CNN: Google":\n' + old
            return old

        agent = BrowserAgent(model, run_tool=run_tool)
        self.assertEqual(agent.run("send the summary to my own chat", noop, Event()), "Enviei.")
        answers = [request[-1]["content"] for request in model.requests[1:]]
        outputs = "\n".join(str(answer) for answer in answers)
        self.assertIn("nothing was typed", outputs)
        self.assertIn("did not send it", outputs)
        self.assertIn("complete new message", outputs)
        self.assertIn("was not seen on the page", outputs)
        self.assertEqual(agent.plan.summary(), "2/2 done")

    def test_guardrail_answer_claiming_sent_without_typing_goes_back(self):
        model = FakeModel([Turn(tool_calls=[browser("open", "https://a.com/")]),
                           Turn(content="Enviei o resumo."),
                           Turn(content="Não enviei: não consegui digitar.")])
        answer = BrowserAgent(model, run_tool=lambda raw: PAGE_A).run("x", noop, Event())
        self.assertEqual(answer, "Não enviei: não consegui digitar.")
        self.assertIn("without a verified new message", model.requests[2][-1]["content"])

    def test_guardrail_ignores_negated_claims_and_ask_ends_the_turn(self):
        from agent import claims_sent
        self.assertFalse(claims_sent("Mas ainda não enviei o resumo. Posso mandar?"))
        self.assertFalse(claims_sent("The summary was not sent."))
        self.assertTrue(claims_sent("Enviei o resumo para o seu chat."))
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["Send the summary"])]),
                           Turn(tool_calls=[tasks(call_id="t2", action="ask", index=1, note="Send as new message?")]),
                           Turn(content="Ainda não enviei. Posso mandar como nova mensagem?")])
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A)
        self.assertEqual(agent.run("x", noop, Event()), "Ainda não enviei. Posso mandar como nova mensagem?")
        self.assertEqual(model.tool_choices, [FORCE_PLAN, "auto", "none"])  # after ask: only the question
        self.assertIn("write it to the user", model.requests[2][-1]["content"])

    def test_verify_task_is_closed_by_evidence_at_the_step_limit(self):
        text = "Resumo: Corinthians recebe averiguacao da PF"
        compose = ToolCall("compose", "browser", json.dumps({"command": "compose", "args": ["e9"],
                           "text": text, "recipient": "Example (you)", "context_ref": "e1"}))
        steps = [tasks(action="plan", items=["Send the summary", "Verify the new message appears"]),
                 compose, browser("press", "Enter", call_id="b2"), browser("verify-message", "e9", call_id="b3"),
                 tasks(call_id="t2", action="done", index=1, note="sent", confidence="high"),
                 ]
        model = FakeModel([Turn(tool_calls=[call]) for call in steps] + [Turn(content="Enviado e conferido.")])
        def run_tool(raw):
            command = json.loads(raw)["command"]
            if command == "compose":
                return PAGE_A + "\n- Draft SHA256: " + text_digest(text)
            if command == "verify-message":
                return PAGE_A + "\n- Message evidence: " + json.dumps({
                    "text_sha256": text_digest(text), "recipient": "Example (you)",
                    "new_messages": 1, "composer_empty": True, "url": "https://a.com/"})
            return PAGE_A
        agent = BrowserAgent(model, run_tool=run_tool, max_steps=6)
        agent.run("send the summary to self", noop, Event())
        self.assertEqual(agent.plan.summary(), "2/2 done")
        self.assertIn("verified by the harness", agent.plan.items[1]["note"])

    def test_first_step_forces_plan_and_plan_is_emitted(self):
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["abrir a.com"])]),
                           Turn(tool_calls=[browser("open", "https://a.com/")]),
                           Turn(tool_calls=[tasks(call_id="t2", action="done", index=1, note="aberto")]),
                           Turn(content="Abri a.com.")])
        events = []
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A)
        self.assertEqual(agent.run("abra a.com", lambda kind, text: events.append((kind, text)), Event()),
                         "Abri a.com.")
        self.assertEqual(model.tool_choices[:2], [FORCE_PLAN, "auto"])
        plans = [text for kind, text in events if kind == "plan"]
        self.assertIn("▶ 1. abrir a.com", plans[0])  # the plan starts its first task by itself
        self.assertIn("✅ 1. abrir a.com", plans[-1])

    def test_pending_tasks_push_back_before_answer(self):
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["cnn", "linkedin"])]),
                           Turn(content="Fiz so a cnn."),
                           Turn(tool_calls=[tasks(call_id="t2", action="done", index=1, note="ok")]),
                           Turn(tool_calls=[tasks(call_id="t3", action="done", index=2, note="ok")]),
                           Turn(content="Pronto, as duas.")])
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A)
        self.assertEqual(agent.run("x", noop, Event()), "Pronto, as duas.")
        self.assertIn("pending tasks", model.requests[2][-1]["content"])

    def test_old_plan_outputs_are_collapsed_but_latest_kept(self):
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["a"])]),
                           Turn(tool_calls=[browser("open", "https://a.com/")]),
                           Turn(tool_calls=[tasks(call_id="t2", action="done", index=1, note="fato")]),
                           Turn(tool_calls=[browser("read", call_id="c2")]),
                           Turn(content="ok")])
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A + "\n" + "x" * 3000)
        with mock.patch("agent.COMPACT_EVERY", 1):  # compaction every step, to check what it does
            agent.run("x", noop, Event())
        tool_messages = [m["content"] for m in model.requests[-1] if m["role"] == "tool"]
        self.assertEqual(tool_messages[0], "(plan updated below)")
        self.assertLess(len(tool_messages[1]), 200)
        self.assertIn("1. [done] [browser] a - fato", tool_messages[2])
        self.assertGreater(len(tool_messages[3]), 3000)

    def test_open_plan_continues_next_turn_without_forcing(self):
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["a", "b"])])]
                          + [Turn(content="parei")] * 3 + [Turn(content="continuando")] * 3)
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A, max_steps=5)
        agent.run("faca a e b", noop, Event())
        first_of_turn2 = len(model.requests)
        agent.run("continue", noop, Event())
        self.assertEqual(model.tool_choices[first_of_turn2], "auto")
        self.assertIn("Open plan", str(model.requests[first_of_turn2]))

    def test_tool_then_answer_keeps_reasoning_and_history(self):
        call = browser("open", "https://a.com/", call_id="call_1")
        model = FakeModel([Turn(reasoning="preciso abrir", tool_calls=[call]), Turn(content="Abri.")])
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A)
        self.assertEqual(agent.run("abra a.com", noop, Event()), "Abri.")
        second = model.requests[1]
        self.assertEqual(second[-2]["reasoning_content"], "preciso abrir")
        self.assertEqual(second[-1], {"role": "tool", "tool_call_id": "call_1", "content": PAGE_A})
        self.assertEqual([m["role"] for m in agent.history], ["user", "assistant"])

    def test_old_browser_outputs_are_summarized_and_reasoning_dropped(self):
        big = PAGE_A + "\n- Page Title: A\n" + "x" * 5000
        model = FakeModel([Turn(reasoning="r1", tool_calls=[browser("snapshot", "e1")]),
                           Turn(reasoning="r2", tool_calls=[browser("snapshot", "e2", call_id="c2")]),
                           Turn(content="ok")])
        with mock.patch("agent.COMPACT_EVERY", 1):
            BrowserAgent(model, run_tool=lambda raw: big).run("x", noop, Event())
        tools_sent = [m for m in model.requests[2] if m["role"] == "tool"]
        self.assertLess(len(tools_sent[0]["content"]), 200)
        self.assertEqual(tools_sent[1]["content"], big)
        reasoning = [m.get("reasoning_content") for m in model.requests[2] if m["role"] == "assistant"]
        self.assertEqual(reasoning, [None, "r2"])

    def test_repeated_read_is_answered_softly_and_failure_retried_once(self):
        click = browser("click", "e5")
        model = FakeModel([Turn(tool_calls=[browser("open", "https://a.com/")]),
                           Turn(tool_calls=[browser("find", "Menu")]), Turn(tool_calls=[browser("find", "Menu")]),
                           Turn(tool_calls=[click]), Turn(tool_calls=[click]), Turn(tool_calls=[click]),
                           Turn(content="ok")])
        executed = []

        def run_tool(raw):
            executed.append(json.loads(raw)["command"])
            return "### Error\nintercepts pointer events" if "click" in raw else PAGE_A + "\nmenu [ref=e5]"

        agent = BrowserAgent(model, run_tool=run_tool)
        agent.run("x", noop, Event())
        self.assertEqual(executed, ["open", "find", "click", "click"])
        repeated = model.requests[3][-1]["content"]
        self.assertTrue(repeated.startswith("[harness] Same result"))
        self.assertIn("menu [ref=e5]", repeated)  # the same answer again, not an error
        self.assertIn("already failed twice", model.requests[6][-1]["content"])

    def test_page_read_before_comes_back_from_the_workspace(self):
        # CNN bug: task 1 read the article, the text left the context, task 2 (summarize) needed it again.
        pages = {"a": "- Page URL: https://a.com/1\n### Text\narticle", "b": "- Page URL: https://a.com/2"}
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["ler 1", "resumir 1"])]),
                           Turn(tool_calls=[browser("goto", "https://a.com/1")]),
                           Turn(tool_calls=[browser("read", call_id="r1")]),
                           Turn(tool_calls=[browser("read", call_id="r2")]),
                           Turn(tool_calls=[browser("goto", "https://a.com/2", call_id="g2")]),
                           Turn(tool_calls=[browser("goto", "https://a.com/1", call_id="g3")]),
                           Turn(tool_calls=[browser("read", call_id="r3")]),
                           Turn(tool_calls=[tasks(call_id="t2", action="ask", index=1, note="x?")]),
                           Turn(content="ok")])
        executed = []

        def run_tool(raw):
            arguments = json.loads(raw)
            executed.append(arguments["command"])
            target = arguments["args"][0] if arguments["args"] else ""
            return pages["b"] if target.endswith("/2") else pages["a"]

        with tempfile.TemporaryDirectory() as folder:
            workspace = Workspace(Path(folder), "s1")
            BrowserAgent(model, run_tool=run_tool, workspace=workspace).run("x", noop, Event())
            self.assertEqual(executed, ["goto", "read", "goto", "goto"])  # the later reads use the saved file
            self.assertIn("[harness] Now: write a short note", model.requests[3][-1]["content"])
            self.assertIn("Saved to workspace: pages/01-1.md", model.requests[3][-1]["content"])
            self.assertIn("still in the conversation above", model.requests[4][-1]["content"])
            # Tasks pending: the latest page text stays in full, so coming back to the page needs no new read.
            self.assertIn("still in the conversation above", model.requests[7][-1]["content"])
            kept = [m["content"] for m in model.requests[7] if m.get("tool_call_id") == "r1"][0]
            self.assertIn("article", kept)
            self.assertIn("Saved to workspace: pages/01-1.md", kept)

    def test_compacted_page_text_comes_back_from_the_workspace(self):
        pages = {"a": "- Page URL: https://a.com/1\n### Text\narticle", "b": "- Page URL: https://a.com/2"}
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["ler 1", "resumir 1"])]),
                           Turn(tool_calls=[browser("read", "https://a.com/1", call_id="r1")]),
                           Turn(tool_calls=[browser("goto", "https://a.com/2", call_id="g2")]),
                           Turn(tool_calls=[ToolCall("r3", "files", json.dumps(
                               {"action": "read", "path": "pages/01-1.md"}))]),
                           Turn(tool_calls=[tasks(call_id="t2", action="ask", index=1, note="x?")]),
                           Turn(content="ok")])
        executed = []

        def run_tool(raw):
            arguments = json.loads(raw)
            executed.append(arguments["command"])
            return pages["b"] if arguments["args"][0].endswith("/2") else pages["a"]

        with tempfile.TemporaryDirectory() as folder, mock.patch("agent.COMPACT_EVERY", 1), \
                mock.patch("agent.KEEP_READS", 0):
            agent = BrowserAgent(model, run_tool=run_tool, workspace=Workspace(Path(folder), "s1"))
            agent.run("x", noop, Event())
        self.assertEqual(executed, ["read", "goto"])  # cached content requires an explicit files read
        again = model.requests[4][-1]["content"]
        self.assertIn("article", again)
        self.assertEqual(agent.turns[-1]["page"], "https://a.com/2", "reading a file must not change the browser page")
        summarized = [m["content"] for m in model.requests[4] if m.get("tool_call_id") == "r1"][0]
        self.assertIn("Saved to workspace: pages/01-1.md", summarized)
        self.assertNotIn("article", summarized)

    def test_pages_read_together_survive_the_next_compaction(self):
        # UOL/CNN session: 4 articles read in one response, then a size compaction kept only 2 and the model
        # re-opened all 4 from the workspace. Fresh reads are kept, and they do not re-trigger compaction.
        urls = [f"https://a.com/{i}" for i in range(4)]
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["read 4", "write article"])]),
                           Turn(tool_calls=[browser("read", url, call_id=f"r{i}") for i, url in enumerate(urls)]),
                           Turn(tool_calls=[tasks(call_id="t2", action="done", index=1, note="read")]),
                           Turn(tool_calls=[browser("goto", "https://a.com/x", call_id="g")]),
                           Turn(content="ok"), Turn(content="ok"), Turn(content="ok")])

        def run_tool(raw):
            target = (json.loads(raw)["args"] or [""])[0]
            return f"- Page URL: {target}\n### Text\n" + "article text " * 500

        with tempfile.TemporaryDirectory() as folder:
            BrowserAgent(model, run_tool=run_tool, workspace=Workspace(Path(folder), "s1")).run("x", noop, Event())
        for request in model.requests[2:4]:
            reads = [m["content"] for m in request if m.get("tool_call_id", "").startswith("r")]
            self.assertEqual(len(reads), 4)
            self.assertTrue(all("article text" in text for text in reads))
        self.assertEqual(model.requests[3][:len(model.requests[2])], model.requests[2])  # still append-only

    def test_context_is_append_only_between_compactions(self):
        # The prompt cache only matches an unchanged prefix: earlier messages must not be rewritten each step.
        model = FakeModel([Turn(tool_calls=[browser("goto", f"https://a.com/{i}", call_id=f"g{i}")])
                           for i in range(4)] + [Turn(content="ok")])
        BrowserAgent(model, run_tool=lambda raw: PAGE_A + "\n" + "x" * 1000).run("x", noop, Event())
        for before, after in zip(model.requests, model.requests[1:]):
            self.assertEqual(after[:len(before)], before)
    def test_no_step_limit_by_default_and_idle_loops_are_caught(self):
        # 40 useful steps: nothing cuts the turn.
        model = FakeModel([Turn(tool_calls=[browser("goto", f"https://a.com/{i}", call_id=f"g{i}")]) for i in range(40)]
                          + [Turn(content="done")])
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A)
        self.assertEqual(agent.run("x", noop, Event()), "done")
        self.assertEqual(len(model.requests), 41)
        # The same find again and again (no error, no progress) climbs to "reflect" after a few repeats.
        model = FakeModel([Turn(tool_calls=[browser("find", "Menu", call_id=f"f{i}")]) for i in range(5)]
                          + [Turn(content="ok")])
        BrowserAgent(model, run_tool=lambda raw: PAGE_A + "\nmenu").run("x", noop, Event())
        self.assertNotIn("two DIFFERENT", model.requests[4][-1]["content"])
        self.assertIn("two DIFFERENT approaches", model.requests[5][-1]["content"])

    def test_step_limit_asks_for_summary_without_tools(self):
        model = FakeModel([Turn(tool_calls=[browser("snapshot", f"e{i}", call_id=f"c{i}")]) for i in range(2)]
                          + [Turn(content="Parei na pagina A.")])
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A, max_steps=3)
        self.assertEqual(agent.run("loop", noop, Event()), "Parei na pagina A.")
        self.assertEqual(model.tool_choices, [FORCE_PLAN, "auto", "none"])
        self.assertIn("Step limit", model.requests[2][-1]["content"])

    def test_history_survives_errors_for_continue(self):
        model = FakeModel([Turn(tool_calls=[browser("open", "https://a.com/")]), RuntimeError("falha de rede"),
                           Turn(content="Continuei.")])
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A)
        with self.assertRaises(RuntimeError):
            agent.run("abra a.com e leia", noop, Event())
        agent.run("continue", noop, Event())
        sent = model.requests[-1]
        self.assertEqual(sent[1], {"role": "user", "content": "abra a.com e leia"})
        self.assertIn("falha de rede", sent[2]["content"])
        self.assertIn("page: https://a.com/", sent[2]["content"])

    def test_old_turns_are_collapsed_to_one_line(self):
        model = FakeModel([Turn(content=f"resposta {i} " + "x" * 1000) for i in range(6)])
        agent = BrowserAgent(model)
        for i in range(6):
            agent.run(f"pedido {i}", noop, Event())
        history = agent.history
        self.assertEqual(len(history), 2 + 3 * 2)
        self.assertIn("- pedido 2 -> final page", history[0]["content"])
        self.assertEqual(history[2], {"role": "user", "content": "pedido 3"})
        self.assertNotIn("resposta 0", str(history))

    def test_content_filter_drops_page_and_retries(self):
        model = FakeModel([Turn(tool_calls=[browser("snapshot")]), ContentFiltered("content_filter"),
                           Turn(content="bloqueada")])
        agent = BrowserAgent(model, run_tool=lambda raw: "texto suspeito")
        self.assertEqual(agent.run("x", noop, Event()), "bloqueada")
        self.assertEqual(model.requests[2][-1]["content"], BLOCKED)

    def test_cancel_stops_before_tool(self):
        cancel = Event()
        cancel.set()
        agent = BrowserAgent(FakeModel([Turn(tool_calls=[browser("snapshot")])]),
                             run_tool=lambda raw: self.fail("executou"))
        with self.assertRaises(InterruptedError):
            agent.run("x", noop, cancel)

    def test_session_log_records_turn(self):
        with tempfile.TemporaryDirectory() as folder:
            log = SessionLog(Path(folder))
            model = FakeModel([Turn(tool_calls=[browser("open", "https://a.com/")], input_tokens=100),
                               Turn(content="ok")])
            BrowserAgent(model, run_tool=lambda raw: PAGE_A, log=log).run("abra a.com", noop, Event())
            events = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([e["event"] for e in events],
                             ["turn_start", "model_call", "tool", "model_call", "turn_end", "turn_usage"])
            self.assertEqual(events[1]["input"], 100)
            self.assertEqual(events[-1]["models"]["llm"]["calls"], 2)
            self.assertEqual(events[-1]["total"]["input"], 100)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace = Workspace(self.root, "20260101-000000")

    def tearDown(self):
        self.temp.cleanup()

    def call(self, **arguments):
        return self.workspace.apply(json.dumps(arguments))

    def test_folder_is_created_on_first_write_and_files_round_trip(self):
        self.assertFalse(self.workspace.folder.exists())
        self.assertIn("empty", self.call(action="list"))
        self.assertIn("Saved notes/summary.md", self.call(action="write", path="notes/summary.md", content="one"))
        self.call(action="append", path="notes/summary.md", content="two")
        self.assertEqual(self.call(action="read", path="notes/summary.md"), "onetwo\n")
        listing = self.call(action="list")
        self.assertIn("notes/summary.md", listing)
        self.assertIn("index.md", listing)

    def test_paths_cannot_leave_the_workspace_and_only_text_is_written(self):
        self.assertIn("inside the workspace", self.call(action="write", path="../x.md", content="a"))
        self.assertIn("text extension", self.call(action="write", path="a.exe", content="a"))
        self.assertIn("not found", self.call(action="read", path="nope.md"))

    def test_long_files_are_read_in_chunks(self):
        self.call(action="write", path="big.txt", content="x" * 9000)
        first = self.call(action="read", path="big.txt")
        self.assertIn("start=8000", first)
        self.assertEqual(self.call(action="read", path="big.txt", start=8000), "x" * 1000)

    def test_pages_and_screenshots_are_saved_and_indexed(self):
        self.workspace.note_request("summarize the CNN article")
        self.assertEqual(self.workspace.save_page("https://cnn.com/2026/ai-hack?x=1", "text"), "pages/01-ai-hack.md")
        self.assertEqual(self.workspace.save_page("https://cnn.com/2026/ai-hack?x=1", "text"), "pages/01-ai-hack.md")
        self.assertEqual(self.workspace.save_image("data:image/jpeg;base64,/9j/", "https://cnn.com"), "shots/01.jpg")
        index = (self.workspace.folder / "index.md").read_text(encoding="utf-8")
        self.assertIn("request: summarize the CNN article", index)
        self.assertIn("page pages/01-ai-hack.md", index)
        self.assertIn("screenshot shots/01.jpg", index)
        self.assertIn("not a text file", self.call(action="read", path="shots/01.jpg"))

    def test_new_session_gets_another_folder_and_earlier_ones_can_be_consulted(self):
        self.workspace.note_request("first run")
        self.call(action="write", path="result.md", content="42")
        second = self.workspace.new()
        self.assertNotEqual(second.id, self.workspace.id)
        sessions = second.apply(json.dumps({"action": "sessions"}))
        self.assertIn(f"{self.workspace.id}: 2 files; first request: first run", sessions)
        self.assertEqual(second.apply(json.dumps({"action": "read", "session": self.workspace.id,
                                                  "path": "result.md"})), "42")
        self.assertIn("read-only", second.apply(json.dumps({"action": "write", "session": self.workspace.id,
                                                            "path": "x.md", "content": "a"})))
        self.assertIn("unknown session", second.apply(json.dumps({"action": "list", "session": "../x"})))

    def test_agent_writes_through_the_files_tool(self):
        call = ToolCall(id="f1", name="files", arguments=json.dumps(
            {"action": "write", "path": "summary.md", "content": "AI news"}))
        model = FakeModel([Turn(tool_calls=[call]), Turn(content="ok")])
        BrowserAgent(model, run_tool=lambda raw: PAGE_A, workspace=self.workspace).run("x", noop, Event())
        self.assertIn("Saved summary.md", model.requests[1][-1]["content"])
        self.assertEqual((self.workspace.folder / "summary.md").read_text(encoding="utf-8"), "AI news")


class PlaybookTests(unittest.TestCase):
    CODE = ("await page.getByRole('button', { name: 'Minhas viagens' }).click();\n"
            "await page.getByTestId('header__navbar__lnk-my-trips--menuitem__content').click();\n"
            "await page.getByRole('textbox', { name: 'Sobrenome' }).fill('Silva');\n"
            "await page.getByRole('textbox', { name: 'Sobrenome' }).press('Enter');\n"
            "await page.getByText('Ver mais detalhes').click();\n")

    def teaching(self):
        teaching = playbooks.Teaching("ver detalhes das minhas viagens", "https://www.latamairlines.com/br/pt?x=1", "LATAM")
        teaching.pages += ["https://www.latamairlines.com/br/pt (LATAM)",
                           "https://www.latamairlines.com/br/pt/minhas-viagens (Minhas Viagens)"]
        teaching.notes.append("on https://www.latamairlines.com/br/pt/minhas-viagens: the return is in Destination 2")
        return teaching

    def test_recording_becomes_actions_in_words_without_values(self):
        self.assertEqual(playbooks.to_actions(self.CODE), [
            'clicked button "Minhas viagens"',
            'clicked element with test id "header__navbar__lnk-my-trips--menuitem__content"',
            'typed <value> in textbox "Sobrenome"',
            'pressed Enter in textbox "Sobrenome"',
            'clicked text "Ver mais detalhes"',
        ])
        self.assertNotIn("Silva", playbooks.redact(self.CODE))

    def test_guide_is_written_by_the_model_from_actions_pages_and_notes(self):
        prompts = []

        def write_guide(prompt):
            prompts.append(prompt)
            return "Here:\n- Open the Minhas viagens menu, then Administrar suas viagens (/br/pt/minhas-viagens).\n" \
                   "- The return flight is in Destination 2."

        with tempfile.TemporaryDirectory() as folder:
            books = playbooks.Playbooks(Path(folder))
            book = books.save(self.teaching(), self.CODE, "Itinerario ... SECRET-CODE", write_guide)
            self.assertIn("Destination 2", prompts[0])           # the user's note goes to the guide writer
            self.assertIn('clicked button "Minhas viagens"', prompts[0])
            self.assertIn("/br/pt/minhas-viagens (Minhas Viagens)", prompts[0])
            self.assertEqual(book.section("Guide").splitlines()[1], "- The return flight is in Destination 2.")
            self.assertIn("the return is in Destination 2", book.section("User notes"))
            self.assertNotIn("x=1", book.text)
            self.assertNotIn("SECRET-CODE", book.text)          # the page excerpt is only used for the guide
            self.assertNotIn("Silva", book.text)
            self.assertEqual(book.path.parent.name, "www-latamairlines-com")
            # Only the guide goes to the agent: no recorded code, no selectors.
            text = books.prompt_text([book])
            self.assertIn("MAP, not a script", text)
            self.assertIn("Administrar suas viagens", text)
            self.assertNotIn("await page", text)
            self.assertNotIn("getByTestId", text)
            self.assertEqual(books.find("no site da latam, quais minhas viagens?"), [book])
            self.assertEqual(books.find("abra o linkedin"), [])

    def test_without_model_the_guide_falls_back_to_actions_and_notes(self):
        with tempfile.TemporaryDirectory() as folder:
            book = playbooks.Playbooks(Path(folder)).save(self.teaching(), self.CODE, "",
                                                          lambda prompt: (_ for _ in ()).throw(RuntimeError("x")))
            guide = book.section("Guide")
            self.assertIn('- The user clicked button "Minhas viagens".', guide)
            self.assertIn("Destination 2", guide)

    def test_agent_injects_matching_guide(self):
        with tempfile.TemporaryDirectory() as folder:
            books = playbooks.Playbooks(Path(folder))
            books.save(playbooks.Teaching("my trips", "https://www.latamairlines.com/", ""), self.CODE)
            model = FakeModel([Turn(content="ok"), Turn(content="ok")])
            events = []
            agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A, playbooks=books)
            agent.run("show my latam trips", lambda kind, text: events.append((kind, text)), Event())
            self.assertIn("showed you how to do a similar task", str(model.requests[0]))
            self.assertIn(("note", "Using what you taught: my trips (www.latamairlines.com)"), events)
            agent.run("open cnn", noop, Event())
            self.assertNotIn("showed you how", str(model.requests[1]))


class MemoryTests(unittest.TestCase):
    def test_session_note_and_long_term_filtering(self):
        with tempfile.TemporaryDirectory() as folder:
            memory = Memory(Path(folder))
            block = memory.append_turn("resuma a cnn", "ok", ["open https://a.com/", "read", "click e5"],
                                       ["click e5 (repeated)"], "https://a.com/", 4, 12000)
            self.assertIn("- worked: open https://a.com/; read\n", block + "\n")
            self.assertIn("- failed: click e5 (repeated)", block)
            self.assertIn("## ", memory.session_path.read_text(encoding="utf-8"))
            saved = memory.save_long_term("Aqui esta:\n- use read para ler\n* evite snapshot\ntexto solto")
            self.assertEqual(saved, "- use read para ler\n- evite snapshot")
            self.assertEqual(memory.long_term(), saved)
            self.assertIn("- use read para ler", memory.consolidate_prompt(block))

    def test_long_term_does_not_lose_lessons_and_session_has_task_results(self):
        with tempfile.TemporaryDirectory() as folder:
            memory = Memory(Path(folder))
            memory.save_long_term("- a\n- b\n- c")
            self.assertEqual(memory.save_long_term("- d"), "- a\n- b\n- c\n- d")
            self.assertEqual(memory.save_long_term("- a\n- c and b\n- d"), "- a\n- c and b\n- d")
            self.assertEqual(memory.save_long_term("- z"), "- a\n- c and b\n- d\n- z")
            block = memory.append_turn("x", "ok", ["read"], [], "https://a.com/", 1, 10,
                                       outcome="Plan (1/1 done):\n1. [done] Open My trips - via menu")
            self.assertIn("- task 1. [done] Open My trips - via menu", block)

    def test_turn_updates_long_term_memory_and_it_goes_to_system_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            memory = Memory(Path(folder))
            model = FakeModel([Turn(tool_calls=[browser("open", "https://a.com/")]),
                               Turn(tool_calls=[browser("read", call_id="c2")]),
                               Turn(content="resumo"),
                               Turn(content="- em a.com, goto + read resolve")])
            events = []
            agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A, memory=memory)
            agent.run("resuma a.com", lambda kind, text: events.append((kind, text)), Event())
            self.assertEqual(model.tools[-1], [])
            self.assertIn("Notes of the task", model.requests[-1][0]["content"])
            self.assertEqual(memory.long_term(), "- em a.com, goto + read resolve")
            self.assertIn(("note", "Long-term memory updated (1 lessons)."), events)
            self.assertIn("goto + read resolve", agent.system_prompt())

    def test_interrupted_turn_writes_session_note_without_consolidating(self):
        with tempfile.TemporaryDirectory() as folder:
            memory = Memory(Path(folder))
            cancel = Event()
            model = FakeModel([Turn(tool_calls=[browser("open", "https://a.com/")]),
                               Turn(tool_calls=[browser("read", call_id="c2")])])
            agent = BrowserAgent(model, run_tool=lambda raw: cancel.set() or PAGE_A, memory=memory)
            with self.assertRaises(InterruptedError):
                agent.run("x", noop, cancel)
            self.assertIn("stopped", memory.session_path.read_text(encoding="utf-8"))
            self.assertEqual(len(model.requests), 2)  # no extra consolidation call


if __name__ == "__main__":
    unittest.main()
