import os

os.environ["BROWSER_HARNESS_PLANNER"] = "0"  # the planner is a real model: off in tests

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from threading import Event

import report
import browser_tool
import plan as plan_module
from agent import BLOCKED_ACTION, CONFIRM_ACTION, BrowserAgent
from foundry import ToolCall, Turn
from memory import Memory
from session_log import SessionLog
from usage import Usage, table
from workspace import Workspace

PAGE_A = "- Page URL: https://a.com/"


def browser(command, *args, call_id="c"):
    return ToolCall(call_id, "browser", json.dumps({"command": command, "args": list(args)}))


def tasks(call_id="t", **arguments):
    return ToolCall(call_id, "tasks", json.dumps(arguments))


def noop(kind, text):
    pass


class FakeModel:
    deployment = "gpt-test"

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self.tool_choices = []

    def complete(self, messages, tools, emit, cancel, tool_choice="auto"):
        self.requests.append([dict(message) for message in messages])
        self.tool_choices.append(tool_choice)
        return self.turns.pop(0)


class ApprovalAgentTests(unittest.TestCase):
    def test_irreversible_actions_require_approval_and_denial_is_not_reasked(self):
        for answer in (True, False):
            ran, shown = [], []
            model = FakeModel([Turn(tool_calls=[browser("click", "Delete account")]),
                               Turn(tool_calls=[browser("click", "Delete account", call_id="c2")]),
                               Turn(content="No message was sent.")])
            agent = BrowserAgent(model, run_tool=lambda raw: ran.append(raw) or PAGE_A,
                                 approve=lambda what, answer=answer: shown.append(what) or answer)
            agent.run("delete the account", noop, Event())
            self.assertEqual(shown, ["click Delete account"])
            self.assertEqual(len(ran), 2 if answer else 0)
            if not answer:
                self.assertIn("DENIED", model.requests[1][-1]["content"])

    def test_headless_approval_is_explicit(self):
        model = FakeModel([Turn(tool_calls=[browser("click", "Delete account")]), Turn(content="Confirm deletion?")])
        agent = BrowserAgent(model, run_tool=lambda raw: self.fail("must not run"))
        agent.run("delete the account", noop, Event())
        self.assertEqual(model.requests[1][-1]["content"], CONFIRM_ACTION)

    def test_turn_usage_is_per_model_in_the_log_and_the_session_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            log, usage = SessionLog(Path(folder)), Usage()
            usage.add("previous-model", 400, 60)  # before the turn: not counted in it
            memory = Memory(Path(folder))
            model = FakeModel([Turn(content="ok", input_tokens=900, output_tokens=5, cached_tokens=100)])
            BrowserAgent(model, log=log, memory=memory, usage=usage).run("x", noop, Event())
            events = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[-1]["models"], {"gpt-test": {"calls": 1, "input": 900, "cached": 100,
                                                                 "output": 5, "cost": 0}})
            self.assertIn("- usage: gpt-test: 1 calls, 900 in (100 cached), 5 out",
                          memory.session_path.read_text(encoding="utf-8"))


class LeanLoopTests(unittest.TestCase):
    """Deterministic savings without auxiliary model calls."""

    def test_done_inside_a_browser_call_closes_the_task_and_runs_the_command(self):
        ran = []
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["open a.com", "read it"])]),
                           Turn(tool_calls=[browser("goto", "https://a.com/")]),
                           Turn(tool_calls=[ToolCall("c2", "browser", json.dumps(
                               {"command": "read", "args": [], "done": 1, "note": "a.com is open"}))]),
                           Turn(content="Resumo pronto.")])
        agent = BrowserAgent(model, run_tool=lambda raw: ran.append(json.loads(raw)["command"]) or PAGE_A)
        agent.run("x", noop, Event())
        self.assertEqual(ran, ["goto", "read"])
        self.assertIn("Done. Now task 2: read it", model.requests[3][-1]["content"])
        self.assertEqual(agent.plan.items[0]["note"], "a.com is open")
        # The final answer closed the last task: no extra tasks call and no pushback.
        self.assertEqual(agent.plan.summary(), "2/2 done")
        self.assertEqual(len(model.requests), 4)

    def test_inline_done_still_passes_the_guardrails(self):
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["send hello", "check it"])]),
                           Turn(tool_calls=[ToolCall("c1", "browser", json.dumps(
                               {"command": "read", "args": [], "done": 1, "note": "sent"}))]),
                           Turn(content="ok"), Turn(content="ok"), Turn(content="ok")])
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A)
        agent.run("x", noop, Event())
        self.assertIn("Task 1 NOT marked done", model.requests[2][-1]["content"])
        self.assertNotEqual(agent.plan.items[0]["status"], "done")

    def test_final_answer_does_not_close_a_task_it_reports_as_failed(self):
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["open a.com"])]),
                           Turn(tool_calls=[browser("goto", "https://a.com/")]),
                           Turn(content="Não consegui abrir a página."), Turn(content="ok"), Turn(content="ok")])
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A)
        agent.run("x", noop, Event())
        self.assertNotEqual(agent.plan.items[0]["status"], "done")

    def test_start_is_not_offered_to_the_model(self):
        self.assertNotIn("start", plan_module.TOOL["function"]["parameters"]["properties"]["action"]["enum"])
        properties = browser_tool.TOOLS[0]["function"]["parameters"]["properties"]
        self.assertIn("done", properties)

    def test_question_with_options_is_answered_in_the_same_turn(self):
        asked = []
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["pick a chat"])]),
                           Turn(tool_calls=[tasks(call_id="t2", action="ask", index=1, note="Which chat?",
                                                  options=["Me", "Team", "Me", "", "A", "B"])]),
                           Turn(tool_calls=[browser("goto", "https://a.com/")]),
                           Turn(content="Pronto: chat Me escolhido.")])
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A,
                             ask_user=lambda question, options: asked.append((question, options)) or "Me")
        answer = agent.run("x", noop, Event())
        self.assertEqual(asked, [("Which chat?", ["Me", "Team", "Me", "A"])])
        self.assertIn('answered your question in the dialog: "Me"', model.requests[2][-1]["content"])
        self.assertEqual(answer, "Pronto: chat Me escolhido.")
        self.assertIn("user: Me", agent.plan.items[0]["note"])

    def test_dismissed_question_ends_the_turn_like_before(self):
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["pick"])]),
                           Turn(tool_calls=[tasks(call_id="t2", action="ask", index=1, note="Which?")]),
                           Turn(content="Which one?")])
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A, ask_user=lambda q, o: None)
        self.assertEqual(agent.run("x", noop, Event()), "Which one?")
        self.assertEqual(agent.plan.summary(), "0/1 done, 1 waiting for you")

    def test_consolidation_only_after_a_failure_or_on_a_new_site(self):
        with tempfile.TemporaryDirectory() as folder:
            memory = Memory(Path(folder))
            memory.save_long_term("- on a.com, goto + read works")
            model = FakeModel([Turn(tool_calls=[browser("open", "https://a.com/")]),
                               Turn(tool_calls=[browser("read", call_id="c2")]), Turn(content="ok")])
            BrowserAgent(model, run_tool=lambda raw: PAGE_A, memory=memory).run("x", noop, Event())
            self.assertEqual(len(model.requests), 3)  # known site, no failure: no consolidation call
            model = FakeModel([Turn(tool_calls=[browser("open", "https://b.com/")]),
                               Turn(tool_calls=[browser("read", call_id="c2")]), Turn(content="ok"),
                               Turn(content="- on b.com, read works")])
            BrowserAgent(model, run_tool=lambda raw: "- Page URL: https://b.com/", memory=memory).run(
                "x", noop, Event())
            self.assertEqual(len(model.requests), 4)
            self.assertIn("b.com", memory.long_term())


class SelfImprovementTests(unittest.TestCase):
    def test_goto_opens_the_browser_when_it_is_closed(self):
        calls = []

        def fake_cli(argv):
            calls.append(argv[-2])
            return ("Error: Browser 'browser-harness' is not open. Run open" if argv[-2] == "goto"
                    else "### Page\n- Page URL: https://a.com/")

        with mock.patch.object(browser_tool, "_cli", fake_cli), mock.patch.object(browser_tool, "run_eval"), \
                mock.patch.object(browser_tool, "base_argv", lambda: ["node", "cli"]):
            output = browser_tool.run_cli("goto", ["https://a.com/"])
        self.assertEqual(calls, ["goto", "open"])
        self.assertIn("https://a.com/", output)

    def test_file_just_written_is_not_read_back(self):
        with tempfile.TemporaryDirectory() as folder:
            model = FakeModel([Turn(tool_calls=[ToolCall("w", "files", json.dumps(
                                   {"action": "write", "path": "r.md", "content": "resumo"}))]),
                               Turn(tool_calls=[ToolCall("r", "files", json.dumps({"action": "read", "path": "r.md"}))]),
                               Turn(content="ok")])
            agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A, workspace=Workspace(Path(folder)))
            agent.run("x", noop, Event())
            self.assertIn("not in the browser", model.requests[1][-1]["content"])
            self.assertIn("You wrote r.md in this turn (6 chars)", model.requests[2][-1]["content"])

    def test_several_tasks_and_files_calls_close_tasks_in_one_call(self):
        with tempfile.TemporaryDirectory() as folder:
            model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["find it", "read it", "save it"])]),
                               Turn(tool_calls=[browser("goto", "https://a.com/")]),
                               Turn(tool_calls=[tasks(call_id="t2", action="done", indexes=[1, 2], note="seen")]),
                               Turn(tool_calls=[ToolCall("w", "files", json.dumps(
                                   {"action": "write", "path": "r.md", "content": "x", "done": 3, "note": "saved"}))]),
                               Turn(content="ok")])
            agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A, workspace=Workspace(Path(folder)))
            agent.run("x", noop, Event())
        self.assertEqual(agent.plan.summary(), "3/3 done")
        self.assertIn("Every task is finished", model.requests[4][-1]["content"])

    def test_waste_is_measured_and_triggers_consolidation_on_a_known_site(self):
        with tempfile.TemporaryDirectory() as folder:
            memory = Memory(Path(folder))
            memory.save_long_term("- on a.com, read works")
            model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["open a", "read a"])]),
                               Turn(tool_calls=[browser("open", "https://a.com/")]),
                               Turn(tool_calls=[tasks(call_id="t2", action="done", index=1, note="open")]),
                               Turn(tool_calls=[browser("read", call_id="r1")]),
                               Turn(tool_calls=[browser("read", call_id="r2")]),
                               Turn(content="Resumo."),
                               Turn(content="- a.com: read https://a.com/ directly")])
            BrowserAgent(model, run_tool=lambda raw: PAGE_A, memory=memory).run("x", noop, Event())
            notes = memory.session_path.read_text(encoding="utf-8")
            self.assertIn("1 model calls only for `tasks`", notes)
            self.assertIn("asked again for text already available: read", notes)
            self.assertIn("wasted steps", model.requests[-1][0]["content"])  # sent to the consolidation
            self.assertIn("read https://a.com/ directly", memory.long_term())

    def test_memory_reviews_itself_when_it_grows_and_never_loses_a_site(self):
        from memory import REVIEW_AT, REVIEW_TARGET
        with tempfile.TemporaryDirectory() as folder:
            memory = Memory(Path(folder))
            lessons = [f"- lesson {i} on https://site{i % 3}.com/" for i in range(REVIEW_AT)]
            memory.long_term_path.parent.mkdir(parents=True, exist_ok=True)
            memory.long_term_path.write_text("\n".join(lessons) + "\n", encoding="utf-8")
            self.assertTrue(memory.needs_review())
            self.assertEqual(memory.save_review("- only https://site0.com/"), "")  # would lose site1, site2
            self.assertEqual(len(memory.long_term().splitlines()), REVIEW_AT)
            review = "- https://site0.com/: a\n- https://site1.com/: b\n- https://site2.com/: c"
            model = FakeModel([Turn(tool_calls=[browser("open", "https://new.com/")]),
                               Turn(tool_calls=[browser("read", call_id="c2")]), Turn(content="ok"),
                               Turn(content="\n".join(lessons + ["- new.com: read works"])), Turn(content=review)])
            events = []
            BrowserAgent(model, run_tool=lambda raw: "- Page URL: https://new.com/", memory=memory).run(
                "x", lambda kind, text: events.append(text), Event())
            self.assertIn(f"rewrite it as at most {REVIEW_TARGET} lines", model.requests[-1][0]["content"])
            self.assertEqual(memory.long_term(), review)
            self.assertTrue(any("reviewed by the harness" in text for text in events))



class InviteGuardrailTests(unittest.TestCase):
    def test_a_search_and_a_connection_invite_are_not_treated_as_sent_messages(self):
        # Invite flow (strings only, no real site): typed search text + Enter, open the profile, click Connect.
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=[
                               "Search the site for the person and verify the matching profile",
                               "If not connected, send a connection invitation and verify the new invitation"])]),
                           Turn(tool_calls=[browser("fill", "e1", "Joe Example", "--submit")]),
                           Turn(tool_calls=[ToolCall("c2", "browser", json.dumps(
                               {"command": "click", "args": ["e9"], "done": 1, "note": "profile matches"}))]),
                           Turn(content="Convite enviado; aparece como Pendente.")])
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A + "\n- button \"Pending\"")
        agent.run("add Joe Example", noop, Event())
        self.assertEqual(agent.plan.summary(), "2/2 done")
        self.assertNotIn("NOT marked done", str(model.requests))
        self.assertNotIn("nothing was typed", str(model.requests))  # "convite enviado" is not a false claim
        # A message task still needs the typed text: the guardrail was not weakened.
        self.assertIsNone(__import__("agent").text_entry("send a connection invitation"))
        self.assertIsNotNone(__import__("agent").text_entry("send the summary to my own chat"))


class PlaybookRuleTests(unittest.TestCase):
    def _book(self, folder):
        import playbooks
        books = playbooks.Playbooks(Path(folder))
        books.save(playbooks.Teaching("add a contact", "https://www.example.com/", "Example"), "")
        return books

    def test_a_user_rule_goes_into_the_guide_the_model_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            books = self._book(folder)
            result = books.apply(json.dumps({"action": "rule", "playbook": "add a contact",
                                             "rule": "after Send without a note it is done; never open the profile"}))
            self.assertTrue(result.startswith("Saved in the playbook www-example-com/add-a-contact.md"))
            book = books.all()[0]
            self.assertIn("- User rule: after Send without a note it is done", book.section("Guide"))
            self.assertIn("never open the profile", books.prompt_text([book]))
            self.assertIn("- rule (", book.section("User notes"))
            self.assertIn("add-a-contact.md", books.apply(json.dumps({"action": "list"})))

    def test_saying_the_playbook_was_updated_without_saving_it_is_caught(self):
        with tempfile.TemporaryDirectory() as folder:
            books = self._book(folder)
            model = FakeModel([Turn(tool_calls=[ToolCall("w", "files", json.dumps(
                                   {"action": "write", "path": "example-playbook.md", "content": "rule"}))]),
                               Turn(content="Fechado: atualizei o playbook."),
                               Turn(tool_calls=[ToolCall("p", "playbooks", json.dumps(
                                   {"action": "rule", "playbook": "add a contact", "rule": "never open the profile"}))]),
                               Turn(content="Fechado: atualizei o playbook."),
                               Turn(content="- example.com contacts: never open the profile")])
            with tempfile.TemporaryDirectory() as work:
                memory = Memory(Path(work) / "memory")
                answer = BrowserAgent(model, run_tool=lambda raw: PAGE_A, playbooks=books, memory=memory,
                                      workspace=Workspace(Path(work))).run("atualize seu teach", noop, Event())
                # The rule reaches the memory consolidation, which must not keep lessons against it.
                self.assertIn("user rule (overrides lessons): never open the profile", model.requests[-1][0]["content"])
            self.assertIn("no `playbooks action=rule` call succeeded", model.requests[2][-1]["content"])
            self.assertEqual(answer, "Fechado: atualizei o playbook.")
            self.assertIn("never open the profile", books.all()[0].section("Guide"))



    def test_answer_that_an_invite_was_sent_does_not_restart_the_task(self):
        # The 130k-token run: after "Send without a note" the answer said "enviado", guardrail 4 pushed back
        # and the model redid everything. Invites have no typed text: the answer is accepted as it is.
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["Find the person",
                                                                         "Send a connection invitation"])]),
                           Turn(tool_calls=[ToolCall("c1", "browser", json.dumps(
                               {"command": "click", "args": ["e9"], "done": 1, "note": "found"}))]),
                           Turn(content="Convite enviado sem nota para a pessoa certa."),
                           Turn(content="should not be needed")])
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A)
        self.assertEqual(agent.run("add X", noop, Event()), "Convite enviado sem nota para a pessoa certa.")
        self.assertEqual(len(model.requests), 3)
        self.assertEqual(agent.plan.summary(), "2/2 done")

    def test_a_message_claim_without_typing_is_still_caught(self):
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["Send the summary to my chat"])]),
                           Turn(content="Resumo enviado."), Turn(content="Não enviei ainda.")] + [Turn(content="ok")] * 3)
        BrowserAgent(model, run_tool=lambda raw: PAGE_A).run("x", noop, Event())
        self.assertIn("without a verified new message", model.requests[2][-1]["content"])

    def test_an_empty_click_target_is_refused_before_the_browser(self):
        with self.assertRaisesRegex(ValueError, "needs a target"):
            browser_tool.build_argv("click", [""])


class RouteLearningTests(unittest.TestCase):
    def test_a_longer_route_for_the_same_kind_of_task_is_reported_and_consolidated(self):
        # 2 invites: the first one with a detour (8 actions), the second one direct (3): the harness says so.
        detour = [browser(c, *a, call_id=f"d{i}") for i, (c, a) in enumerate([
            ("find", ["X"]), ("find", ["Y"]), ("click", ["e1"]), ("read", []), ("goto", ["https://a.com/p"]),
            ("find", ["Invite"]), ("click", ["e2"]), ("click", ["e3"])])]
        direct = [browser(c, *a, call_id=f"s{i}") for i, (c, a) in enumerate([
            ("read", ["https://a.com/p2"]), ("click", ["e4"]), ("click", ["e5"])])]
        with tempfile.TemporaryDirectory() as folder:
            memory = Memory(Path(folder))
            memory.save_long_term("- a.com: people search")
            model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["invite A", "invite B"])])]
                              + [Turn(tool_calls=[call]) for call in detour]
                              + [Turn(tool_calls=[tasks(call_id="t2", action="done", index=1, note="pending")])]
                              + [Turn(tool_calls=[call]) for call in direct]
                              + [Turn(content="Convites enviados."), Turn(content="- a.com: read the people URL")])
            BrowserAgent(model, run_tool=lambda raw: PAGE_A, memory=memory).run("invite A and B", noop, Event())
            notes = memory.session_path.read_text(encoding="utf-8")
            self.assertIn("the tasks took 8, 3 browser actions: the shortest one is the route to keep", notes)
            self.assertIn("read the people URL", memory.long_term())

    def test_using_a_taught_guide_always_refines_the_memory(self):
        import playbooks
        with tempfile.TemporaryDirectory() as folder:
            books = playbooks.Playbooks(Path(folder) / "books")
            books.save(playbooks.Teaching("add a contact", "https://www.example.com/", "Example"), "")
            memory = Memory(Path(folder) / "memory")
            memory.save_long_term("- example.com: add contact from the people results")
            model = FakeModel([Turn(tool_calls=[browser("read", "https://www.example.com/p")]),
                               Turn(tool_calls=[browser("click", "e1", call_id="c2")]), Turn(content="ok"),
                               Turn(content="- example.com: add contact from the people results")])
            BrowserAgent(model, run_tool=lambda raw: "- Page URL: https://www.example.com/p", memory=memory,
                         playbooks=books).run("add a contact on example", noop, Event())
            self.assertEqual(len(model.requests), 4)  # no failure, known site, but a guide was used: consolidated


class PlannerTests(unittest.TestCase):
    class FakePlanner:
        deployment = "gpt-test-planner"
        scope = "all"

        def __init__(self, *decisions):
            self.decisions = list(decisions)
            self.calls = []

        def plan(self, request, lessons="", guides="", history="", answers=None):
            self.calls.append((request, list(answers or [])))
            decision = self.decisions.pop(0)
            if isinstance(decision, Exception):
                raise decision
            return decision, Turn(input_tokens=900, output_tokens=120), 1500

    def test_the_planner_plans_and_the_executor_runs_it_without_planning_again(self):
        planner = self.FakePlanner({"plan": ["Invite Ana Lima (Microsoft)", "Invite Carlos Reis (Microsoft)"],
                                    "notes": "use the people search URL"})
        model = FakeModel([Turn(tool_calls=[browser("read", "https://a.com/p", call_id="r1")]),
                           Turn(tool_calls=[ToolCall("c1", "browser", json.dumps(
                               {"command": "click", "args": ["e1"], "done": 1, "note": "sent"}))]),
                           Turn(content="Convites enviados para os dois.")])
        usage = Usage()
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A, planner=planner, usage=usage)
        agent.run("add Ana Lima e Carlos Reis da Microsoft", noop, Event())
        self.assertEqual(model.tool_choices[0], "auto")  # the executor does not plan again
        first = model.requests[0]
        self.assertEqual(first[-2]["tool_calls"][0]["function"]["name"], "tasks")
        self.assertIn("Invite Ana Lima (Microsoft)", first[-1]["content"])
        self.assertIn("[planner] Notes: use the people search URL", first[-1]["content"])
        self.assertEqual(agent.plan.summary(), "2/2 done")
        self.assertEqual(usage.snapshot()["gpt-test-planner"]["input"], 900)

    def test_the_planner_asks_in_the_panel_and_plans_with_the_answer(self):
        planner = self.FakePlanner({"question": "Mano é um contato ou só um jeito de falar?",
                                    "options": ["Só os 2: Ana e Carlos", "Os 3"]},
                                   {"plan": ["Invite Ana Lima (Microsoft)", "Invite Carlos Reis (Microsoft)"]})
        asked = []
        model = FakeModel([Turn(content="ok")] * 4)
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A, planner=planner,
                             ask_user=lambda q, o: asked.append((q, o)) or "Só os 2: Ana e Carlos")
        agent.run("Mano, da Microsoft\nAna Lima\nCarlos Reis\nos dois", noop, Event())
        self.assertEqual(asked[0][1], ["Só os 2: Ana e Carlos", "Os 3"])
        self.assertEqual(planner.calls[1][1], [("Mano é um contato ou só um jeito de falar?", "Só os 2: Ana e Carlos")])
        self.assertIn("The user clarified", model.requests[0][-1]["content"])
        self.assertEqual(len(agent.plan.items), 2)

    def test_without_a_panel_the_question_is_the_answer_of_the_turn(self):
        planner = self.FakePlanner({"question": "Qual empresa?", "options": ["Microsoft", "Outra"]})
        model = FakeModel([])
        answer = BrowserAgent(model, run_tool=lambda raw: PAGE_A, planner=planner).run("add o Joao", noop, Event())
        self.assertEqual(answer, "Qual empresa?\n- Microsoft\n- Outra")
        self.assertEqual(model.requests, [])  # no executor call

    def test_if_the_planner_fails_the_executor_plans_as_before(self):
        planner = self.FakePlanner(RuntimeError("DeploymentNotFound"))
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["open a.com"])]),
                           Turn(tool_calls=[browser("goto", "https://a.com/")]), Turn(content="ok")])
        events = []
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE_A, planner=planner)
        agent.run("abra a.com", lambda kind, text: events.append(text), Event())
        self.assertEqual(model.tool_choices[0]["function"]["name"], "tasks")  # forced plan, as before
        self.assertTrue(any("Planner unavailable" in text for text in events))

    def test_with_the_actions_scope_reads_are_planned_by_the_executor(self):
        planner = self.FakePlanner({"plan": ["Invite Ana Lima (Microsoft)"]})
        planner.scope = "actions"
        model = FakeModel([Turn(tool_calls=[tasks(action="plan", items=["read the article"])])] + [Turn(content="ok")] * 3)
        BrowserAgent(model, run_tool=lambda raw: PAGE_A, planner=planner).run("resuma a noticia da cnn", noop, Event())
        self.assertEqual(planner.calls, [])  # a read: no stronger-model call
        model = FakeModel([Turn(content="ok")] * 3)
        BrowserAgent(model, run_tool=lambda raw: PAGE_A, planner=planner).run("adicione a Ana Lima da Microsoft",
                                                                              noop, Event())
        self.assertEqual(len(planner.calls), 1)  # acts on another person: the planner plans

    def test_parse_accepts_only_a_plan_or_a_question(self):
        import planner as planner_module
        self.assertEqual(planner_module.parse('Here: {"plan": ["a", " ", "b"]}'), {"plan": ["a", "b"], "notes": ""})
        self.assertEqual(planner_module.parse('{"question": "Qual?", "options": ["x", "y"]}')["options"], ["x", "y"])
        self.assertIsNone(planner_module.parse("no json"))
        self.assertIsNone(planner_module.parse('{"plan": []}'))



class BrowserToolLabelTests(unittest.TestCase):
    def test_find_tries_alternatives_in_one_call(self):
        calls = []

        def fake_cli(argv):
            calls.append(argv[-1])
            return ('### Result\nFound 1 match for "Conectar":\n- button "Conectar" [ref=e5]' if argv[-1] == "Conectar"
                    else '### Result\nNo matches found for "Connect".')

        with mock.patch.object(browser_tool, "_cli", fake_cli), \
                mock.patch.object(browser_tool, "base_argv", lambda: ["node", "cli"]):
            output = browser_tool._run_cli("find", ["Connect | Conectar"])
        self.assertEqual(calls, ["Connect", "Conectar"])
        self.assertIn('Matched "Conectar"', output)
        self.assertIn("ref=e5", output)

    def test_wait_also_looks_inside_same_origin_iframes(self):
        self.assertIn("contentDocument", browser_tool.WAIT_JS)


class UsageReportTests(unittest.TestCase):
    def test_footer_table_and_report_from_logs(self):
        usage = Usage()
        usage.add("azure:test-executor", 3000, 50, 1000)
        usage.add("local:test-planner", 400, 60)
        self.assertEqual(usage.footer(3000), "ctx 3,000  In 3,400  Out 110")
        self.assertIn("| **Total** | 2 | 3,400 | 1,000 | 110 |", table(usage.snapshot()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "s.jsonl"
            path.write_text("\n".join(json.dumps(e) for e in [
                {"event": "turn_start", "model": "azure:test-executor"},
                {"event": "model_call", "input": 3000, "output": 40, "cached": 0},
                {"event": "memory", "input": 500, "output": 30},
                {"event": "legacy-event", "input": 400, "output": 60},
                {"event": "legacy-event", "error": "down"}]), encoding="utf-8")
            usage, turns = report.read_session(path)
            rows = usage.snapshot()
            self.assertEqual(turns, 1)
            self.assertEqual(rows["azure:test-executor"]["calls"], 2)
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
