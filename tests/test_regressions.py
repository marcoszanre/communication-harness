"""Regression coverage for task intent, dates, research, and one-message composition."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from unittest import mock

import browser_tool
from agent import BrowserAgent, Progress
from foundry import ToolCall, Turn
from memory import NullMemory
from plan import Plan
from playbooks import Playbooks, Teaching
from runtime_context import clock_context, date_counts, publication_date

PAGE = "- Page URL: https://example.com/"


class Model:
    def __init__(self, turns=()):
        self.turns = list(turns)
        self.requests = []

    def complete(self, messages, tools, emit, cancel, tool_choice="auto"):
        self.requests.append(messages)
        return self.turns.pop(0)


def tool(name, **arguments):
    return ToolCall("call", name, json.dumps(arguments))


class IntentAndStopTests(unittest.TestCase):
    def test_validation_failure_cannot_be_closed_by_a_claim_or_repeated_done(self):
        agent, progress = BrowserAgent(Model()), Progress()
        agent.plan.apply(json.dumps({"action": "plan", "items": ["Open the page"]}))
        progress.record("type", browser_tool.VALIDATION_ERROR + "invalid argument")
        agent._close_with_answer("The page is open.", progress, lambda *args: None)
        self.assertNotEqual(agent.plan.items[0]["status"], "done")
        for _ in range(2):
            result = agent._apply_tasks(json.dumps({"action": "done", "index": 1, "note": "claimed"}),
                                       0, progress, lambda *args: None, "done")
            self.assertTrue(result.startswith("Error"))
        self.assertNotEqual(agent.plan.items[0]["status"], "done")

    def test_local_read_schema_is_grounded_in_the_users_explicit_url(self):
        from agent import TOOLS
        model = Model()
        model.provider = "local"
        agent = BrowserAgent(model)
        agent.plan_request = "Open https://example.com/ and report its title."
        agent.plan.apply(json.dumps({"action": "plan", "items": ["Read example.com"]}))
        tools = agent._tools_for_turn()
        schema = next(item for item in tools if item["function"]["name"] == "browser")
        properties = schema["function"]["parameters"]["properties"]
        self.assertEqual(properties["command"]["enum"], ["read"])
        self.assertEqual(properties["args"]["items"]["enum"], ["https://example.com/"])
        original = next(item for item in TOOLS if item["function"]["name"] == "browser")
        self.assertNotEqual(original["function"]["parameters"]["properties"]["command"]["enum"], ["read"])

    def test_update_containing_send_is_a_playbook_task(self):
        plan = Plan()
        plan.apply(json.dumps({"action": "plan", "items": [
            "Update the messaging playbook: send the summary as one message, not split messages."]}))
        self.assertEqual(plan.items[0]["kind"], "playbook")
        agent = BrowserAgent(Model(), run_tool=lambda raw: self.fail("browser must not run"))
        agent.plan = plan
        progress = Progress()
        progress.playbook_saved = True
        self.assertEqual(agent._entry_check({"action": "done", "index": 1}, progress), "")

    def test_structured_send_has_recipient_and_message_count(self):
        plan = Plan()
        result = plan.apply(json.dumps({"action": "plan", "items": [
            {"text": "Send the complete draft", "kind": "send", "recipient": "self", "expected_messages": 1}]}))
        self.assertFalse(result.startswith("Error"))
        self.assertEqual(plan.items[0]["recipient"], "self")
        self.assertEqual(plan.items[0]["expected_messages"], 1)

    def test_invalid_task_shapes_are_rejected(self):
        for value in ({}, [{"text": "x", "kind": "unknown"}], [123], ["x" * 301],
                      [{"text": "x", "kind": "send", "expected_messages": True}]):
            plan = Plan()
            self.assertTrue(plan.apply(json.dumps({"action": "plan", "items": value})).startswith("Error"))

    def test_cancel_never_changes_finished_tasks(self):
        plan = Plan()
        plan.apply(json.dumps({"action": "plan", "items": ["Open", "Read", "Save"]}))
        plan.apply(json.dumps({"action": "done", "index": 1, "note": "seen"}))
        plan.cancel()
        self.assertEqual([item["status"] for item in plan.items], ["done", "cancelled", "cancelled"])
        self.assertEqual(plan.unfinished(), [])

    def test_a_question_does_not_start_the_next_task(self):
        plan = Plan()
        plan.apply(json.dumps({"action": "plan", "items": ["Research", "Send the draft"]}))
        plan.apply(json.dumps({"action": "ask", "index": 1, "note": "Use fewer articles?"}))
        self.assertEqual([item["status"] for item in plan.items], ["question", "pending"])

    def test_batched_actions_after_a_question_are_not_executed(self):
        model = Model([Turn(tool_calls=[tool("tasks", action="plan", items=["Choose a chat", "Read"])]),
                       Turn(tool_calls=[tool("tasks", action="ask", index=1, note="Which chat?"),
                                        tool("browser", command="read", args=[])]),
                       Turn(content="Which chat?")])
        agent = BrowserAgent(model, run_tool=lambda raw: self.fail("must wait for the answer"))
        agent.run("Choose a chat", lambda *args: None, Event())
        self.assertIn("Not executed", str(model.requests[-1]))

    def test_a_new_request_does_not_resume_an_unrelated_open_plan(self):
        model = Model([Turn(tool_calls=[tool("tasks", action="plan", items=["Open the new page"])]),
                       Turn(tool_calls=[tool("browser", command="read", args=["https://example.com/"])]),
                       Turn(content="New page opened.")])
        agent = BrowserAgent(model, run_tool=lambda raw: PAGE)
        agent.plan.apply(json.dumps({"action": "plan", "items": ["Send an unrelated old message"]}))
        agent.run("Open https://example.com/", lambda *args: None, Event())
        self.assertEqual(agent.plan.items[0]["text"], "Open the new page")
        self.assertNotIn("Open plan from the previous turn", str(model.requests))

    def test_user_stop_needs_no_model_or_browser_call(self):
        model = Model()
        agent = BrowserAgent(model, run_tool=lambda raw: self.fail("browser must not run"))
        agent.plan.apply(json.dumps({"action": "plan", "items": ["Send a message"]}))
        result = agent.run("Stop; playbook update only", lambda *args: None, Event())
        self.assertIn("Stopped", result)
        self.assertEqual(model.requests, [])
        self.assertEqual(agent.plan.unfinished(), [])

    def test_stop_in_inline_answer_ends_before_another_model_call(self):
        model = Model([Turn(tool_calls=[tool("tasks", action="plan", items=["Select a chat"])]),
                       Turn(tool_calls=[tool("tasks", action="ask", index=1, note="Which chat?",
                                            options=["Self", "Other"])])])
        agent = BrowserAgent(model, ask_user=lambda question, options: "Keep playbook-only; stop",
                             run_tool=lambda raw: self.fail("browser must not run"))
        result = agent.run("Choose a chat", lambda *args: None, Event())
        self.assertIn("Stopped", result)
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(agent.plan.unfinished(), [])

    def test_maintenance_request_blocks_browser_even_if_plan_is_wrong(self):
        model = Model([Turn(tool_calls=[tool("tasks", action="plan", items=[
            {"text": "Send a message", "kind": "send"}])]),
            Turn(tool_calls=[tool("browser", command="press", args=["Enter"])]),
            Turn(content="No message was sent."), Turn(content="No message was sent."),
            Turn(content="No message was sent.")])
        agent = BrowserAgent(model, run_tool=lambda raw: self.fail("browser must not run"))
        agent.run("Update the playbook rule about sending", lambda *args: None, Event())
        self.assertEqual(agent.plan.items[0]["kind"], "playbook")
        self.assertIn("playbook-only request", str(model.requests))


class TimeAndResearchTests(unittest.TestCase):
    def context(self):
        return clock_context(datetime(2026, 10, 13, 23, 30, tzinfo=timezone(timedelta(hours=-3))))

    def test_local_day_is_not_utc_day(self):
        context = self.context()
        self.assertIn("Today: 2026-10-13", context)
        self.assertIn("Yesterday: 2026-10-12", context)
        self.assertIn("-03:00", context)

    def test_per_date_counts_are_preserved(self):
        self.assertEqual(date_counts("AI news from today and yesterday, two from each day", self.context()),
                         {"2026-10-13": 2, "2026-10-12": 2})
        self.assertEqual(date_counts("notícias de hoje e ontem, duas de cada dia", self.context()),
                         {"2026-10-13": 2, "2026-10-12": 2})
        self.assertEqual(date_counts("summarize the news", self.context()), {})

    def test_month_boundary(self):
        context = clock_context(datetime(2027, 1, 1, tzinfo=timezone.utc))
        self.assertIn("Yesterday: 2026-12-31", context)

    def test_publication_timestamp_uses_the_request_timezone(self):
        self.assertEqual(publication_date("2026-10-14T01:30:00Z", self.context()), "2026-10-13")
        self.assertEqual(publication_date("2026-10-14", self.context()), "2026-10-14")

    def test_planner_and_executor_receive_the_same_clock(self):
        from planner import Planner
        planner = Planner(Model())
        planner.clock = self.context()
        agent = BrowserAgent(Model())
        agent.clock = planner.clock
        self.assertIn(planner.clock, planner.prompt("read today", "", "", "", []))
        self.assertIn(planner.clock, agent.system_prompt())

    def test_source_date_comes_from_page_metadata_not_url(self):
        agent, progress = BrowserAgent(Model()), Progress()
        call = tool("browser", command="read", args=["https://example.com/2030/01/01/article"])
        agent._record_evidence(call, "read", "- Page URL: https://example.com/2030/01/01/article\n"
                               "- Page Title: Synthetic article\n- Source type: article\n"
                               "- Published date: 2026-10-13T10:00:00-03:00", progress)
        self.assertEqual(progress.sources["https://example.com/2030/01/01/article"]["date"], "2026-10-13")

    def test_a_missing_or_invalid_date_is_not_verified(self):
        for metadata in ("", "- Published date: not-a-date"):
            agent, progress = BrowserAgent(Model()), Progress()
            agent._record_evidence(tool("browser", command="read", args=[]), "read",
                                   PAGE + "\n- Page Title: Article\n" + metadata, progress)
            self.assertEqual(progress.sources, {})

    def test_listing_dates_do_not_count_as_an_article(self):
        agent, progress = BrowserAgent(Model()), Progress()
        agent._record_evidence(tool("browser", command="read", args=[]), "read",
                               PAGE + "\n- Page Title: Search results\n- Published date: 2026-10-13", progress)
        self.assertEqual(progress.sources, {})

    def test_coverage_is_per_date_not_total_articles(self):
        agent, progress = BrowserAgent(Model()), Progress()
        agent.plan.date_counts = {"2026-10-13": 2, "2026-10-12": 2}
        for index in range(4):
            progress.sources[f"https://example.com/{index}"] = {"date": "2026-10-13"}
        selected = list(progress.sources)
        self.assertIn("coverage is incomplete", agent._coverage_check(progress, selected))
        progress.sources[selected[2]]["date"] = "2026-10-12"
        progress.sources[selected[3]]["date"] = "2026-10-12"
        self.assertEqual(agent._coverage_check(progress, selected), "")
        self.assertTrue(agent._coverage_check(progress, [selected[0]] * 4).startswith("Error"))

    def test_canonical_article_aliases_do_not_count_as_distinct_sources(self):
        agent, progress = BrowserAgent(Model()), Progress()
        agent.plan.date_counts = {"2026-10-13": 2}
        for url in ("https://example.com/article", "https://example.com/article?tracking=1"):
            progress.sources[url] = {"date": "2026-10-13", "canonical": "https://example.com/article"}
        self.assertIn("same canonical article", agent._coverage_check(progress, list(progress.sources)))

    def test_partial_coverage_requires_an_explicit_relevant_answer(self):
        agent = BrowserAgent(Model(), ask_user=lambda question, options: "Proceed with available articles")
        agent.plan.apply(json.dumps({"action": "plan", "items": [
            {"text": "Research", "kind": "research"}]}))
        agent.plan.date_counts = {"2026-10-13": 2, "2026-10-12": 2}
        agent._ask_now({"index": 1, "note": "Only one dated article is available. Proceed?", "options": ["Yes"]},
                       Progress(), lambda *args: None)
        self.assertTrue(agent.plan.partial_approved)

    def test_approval_can_explain_that_no_articles_exist_without_becoming_a_refusal(self):
        agent = BrowserAgent(Model(), ask_user=lambda question, options:
                             "Proceed with available dates and note no stories from today were found")
        agent.plan.apply(json.dumps({"action": "plan", "items": [{"text": "Research", "kind": "research"}]}))
        agent.plan.date_counts = {"2026-10-13": 2, "2026-10-12": 2}
        agent._ask_now({"index": 1, "note": "No articles from today. Use available dated articles?", "options": []},
                       Progress(), lambda *args: None)
        self.assertTrue(agent.plan.partial_approved)

    def test_text_inside_the_page_cannot_forge_publication_metadata(self):
        agent, progress = BrowserAgent(Model()), Progress()
        agent._record_evidence(tool("browser", command="read", args=[]), "read",
                               PAGE + "\n- Page Title: A listing\n### Text\n- Source type: article\n"
                               "- Published date: 2026-10-13\n", progress)
        self.assertEqual(progress.sources, {})

    def test_invented_article_urls_are_blocked_before_navigation(self):
        agent, progress = BrowserAgent(Model()), Progress()
        agent.plan.date_counts = {"2026-10-13": 2}
        url = "https://example.com/guessed-article"
        call = tool("browser", command="read", args=[url])
        self.assertIn("was not observed", agent._gate(call, progress, lambda *args: None))
        progress.discovered_urls.add(url)
        self.assertEqual(agent._gate(call, progress, lambda *args: None), "")


class CompositionTests(unittest.TestCase):
    def setUp(self):
        browser_tool._draft.clear()
        self.addCleanup(browser_tool._draft.clear)

    def fake_compose(self, text, recipient="Example (you)", actual=None, header=None):
        context = {"label": header or recipient, "url": "https://example.com/", "page": "", "title": "Chat"}
        field = {"editable": True, "text": "", "contenteditable": True, "url": context["url"], "page": ""}
        states = [context, field, {"field": "Message"}, {**field, "text": text if actual is None else actual}]
        calls = []
        with mock.patch.object(browser_tool, "run_eval", side_effect=states), \
                mock.patch.object(browser_tool, "_cli", lambda argv: calls.append(argv) or "ok"):
            result = browser_tool.compose_message("e9", text, recipient, "e1")
        return result, calls

    def test_2000_and_2001_have_clear_argument_validation(self):
        with mock.patch.object(browser_tool, "base_argv", return_value=["node", "cli"]):
            browser_tool.build_argv("type", ["x" * 2000])
            with self.assertRaisesRegex(ValueError, "2001 characters; maximum is 2000"):
                browser_tool.build_argv("type", ["x" * 2001])

    def test_text_is_literal_even_when_it_looks_like_a_cli_flag(self):
        with mock.patch.object(browser_tool, "base_argv", return_value=["node", "cli", "-s=owned-session"]):
            self.assertEqual(browser_tool.build_argv("type", ["-s=other-session"])[-3:],
                             ["type", "--", "-s=other-session"])
            self.assertEqual(browser_tool.build_argv("fill", ["e9", "--literal", "--submit"])[-5:],
                             ["fill", "--submit", "--", "e9", "--literal"])

    def test_submit_flags_cannot_be_hidden_in_type_or_press_arguments(self):
        for command, args in (("type", ["partial message", "--submit"]), ("press", ["Enter", "--submit"])):
            with self.assertRaises(ValueError):
                browser_tool.build_argv(command, args)

    def test_command_normalization_is_shared_by_dispatch_and_guardrails(self):
        call = tool("browser", command="fill e9", args=["hello", "--submit"])
        self.assertEqual(BrowserAgent._args(call)["command"], "fill")
        self.assertEqual(BrowserAgent._args(call)["args"], ["e9", "hello", "--submit"])
        with mock.patch.object(browser_tool, "run_cli", return_value="ok") as run:
            browser_tool.dispatch(call.arguments)
        run.assert_called_once_with("fill", ["e9", "hello", "--submit"])

    def test_explicit_self_request_cannot_be_redirected_by_incorrect_plan_metadata(self):
        agent, progress = BrowserAgent(Model()), Progress()
        agent.plan_request = "Send the summary to myself."
        agent.plan.apply(json.dumps({"action": "plan", "items": [
            {"text": "Send the summary", "kind": "send", "recipient": "Other person"}]}))
        progress.typed_texts = ["text"]
        progress.draft = {"text": "text", "recipient": "Other person", "submitted": False}
        result = agent._gate(tool("browser", command="press", args=["Enter"]), progress, lambda *args: None)
        self.assertIn("does not match the planned recipient", result)

    def test_schema_advertises_the_limits_and_separate_long_text(self):
        properties = browser_tool.TOOLS[0]["function"]["parameters"]["properties"]
        self.assertEqual(properties["args"]["items"]["maxLength"], 2000)
        self.assertEqual(properties["text"]["maxLength"], 40000)

    def test_long_find_context_does_not_hide_the_textbox_ref(self):
        raw = ('### Result\nFound 2 matches:\n- paragraph [ref=e1]: ' + "long text " * 1000
               + '\n- textbox "Message" [ref=e9]\n')
        output = browser_tool._finish("find", raw)
        self.assertIn('textbox "Message" [ref=e9]', output)
        self.assertLess(len(output), 1000)

    def test_losing_paragraph_breaks_is_not_a_complete_draft(self):
        with self.assertRaisesRegex(RuntimeError, "complete draft did not match"):
            self.fake_compose("paragraph one\n\nparagraph two", actual="paragraph one paragraph two")

    def test_long_unicode_multiline_text_is_one_unsent_draft(self):
        text = "First paragraph " + "é🌍" * 2200 + "\n\nSecond paragraph\nOutcome paragraph"
        result, calls = self.fake_compose(text)
        typed = [argv[-1] for argv in calls if argv[-2] == "type"]
        self.assertTrue(all(len(value) <= 2000 for value in typed))
        self.assertEqual(sum(argv[-2:] == ["press", "Shift+Enter"] for argv in calls), 3)
        self.assertFalse(any(argv[-2:] == ["press", "Enter"] for argv in calls))
        self.assertEqual(browser_tool._draft["text"], text)
        self.assertIn("one unsent message", result)

    def test_wrong_header_prevents_typing(self):
        with self.assertRaisesRegex(ValueError, "does not match recipient"):
            self.fake_compose("hello", header="Wrong recipient")

    def test_partial_draft_is_not_accepted(self):
        with self.assertRaisesRegex(RuntimeError, "complete draft did not match"):
            self.fake_compose("complete text", actual="complete")
        self.assertEqual(browser_tool._draft, {})

    def test_repeat_submit_is_blocked_without_touching_the_browser(self):
        self.fake_compose("hello")
        browser_tool._draft["submitted"] = True
        with mock.patch.object(browser_tool, "_run_cli") as run:
            result = browser_tool.run_cli("press", ["Enter"])
        self.assertIn("already submitted", result)
        run.assert_not_called()

    def test_last_moment_recipient_change_blocks_send(self):
        self.fake_compose("hello")
        with mock.patch.object(browser_tool, "run_eval", side_effect=[
            {"text": "hello"}, {"label": "Wrong recipient", "url": "https://example.com/"}]), \
                mock.patch.object(browser_tool, "_run_cli") as run:
            result = browser_tool.run_cli("press", ["Enter"])
        self.assertIn("recipient changed", result)
        run.assert_not_called()

    def test_verification_counts_a_new_full_occurrence_not_an_old_message(self):
        self.fake_compose("hello")
        browser_tool._draft.update(submitted=True, baseline=1)
        state = {"editable": True, "text": "", "page": "hello hello",
                 "url": "https://example.com/", "title": "Chat"}
        with mock.patch.object(browser_tool, "run_eval", side_effect=[
            {"label": "Example (you)", "url": state["url"]}, state]):
            result = browser_tool.verify_message("e9", "e1")
        evidence = json.loads(result.split("- Message evidence: ")[1])
        self.assertEqual(evidence["new_messages"], 1)
        self.assertEqual(evidence["text_sha256"], browser_tool.text_digest("hello"))

    def test_empty_composer_and_exactly_one_new_message_are_required(self):
        for page, field in (("hello", "hello"), ("hello hello", ""), ("", "")):
            with self.subTest(page=page, field=field):
                self.fake_compose("hello")
                browser_tool._draft["submitted"] = True
                with mock.patch.object(browser_tool, "run_eval", side_effect=[
                    {"label": "Example (you)", "url": "https://example.com/"},
                    {"editable": True, "text": field, "page": page, "url": "https://example.com/"}]), \
                        mock.patch.object(browser_tool.time, "monotonic", side_effect=[0, 11]):
                    with self.assertRaisesRegex(RuntimeError, "Message not verified"):
                        browser_tool.verify_message("e9", "e1")
                browser_tool._draft.clear()

    def test_arbitrary_click_and_read_snippet_are_not_send_proof(self):
        agent, progress = BrowserAgent(Model()), Progress()
        progress.typed_texts = ["complete message"]
        agent._record_evidence(tool("browser", command="click", args=["e1"]), "click", PAGE, progress)
        agent._record_evidence(tool("browser", command="read", args=[]), "read",
                               PAGE + "\ncomplete message", progress)
        self.assertEqual(progress.submits, [])
        self.assertEqual(progress.verified_messages, [])

    def test_wrong_recipient_and_two_messages_cannot_finish_a_single_send(self):
        agent, progress = BrowserAgent(Model()), Progress()
        agent.plan.apply(json.dumps({"action": "plan", "items": [
            {"text": "Send", "kind": "send", "recipient": "Recipient", "expected_messages": 1}]}))
        progress.typed = [0]
        progress.submits = [1]
        progress.verified_messages = [{"position": 2, "task_index": 1, "recipient": "Other"}]
        self.assertIn("not verified", agent._entry_check({"action": "done", "index": 1}, progress))
        progress.verified_messages = [{"position": 2, "task_index": 1, "recipient": "Recipient"},
                                      {"position": 4, "task_index": 1, "recipient": "Recipient"}]
        self.assertIn("not verified", agent._entry_check({"action": "done", "index": 1}, progress))

    def test_message_proof_does_not_complete_a_different_send_task(self):
        agent, progress = BrowserAgent(Model()), Progress()
        agent.plan.apply(json.dumps({"action": "plan", "items": [
            {"text": "Send first", "kind": "send", "recipient": "Recipient"},
            {"text": "Send second", "kind": "send", "recipient": "Recipient"},
            {"text": "Verify second", "kind": "verify"}]}))
        progress.typed, progress.submits = [0], [1]
        progress.verified_messages = [{"position": 2, "task_index": 1, "recipient": "Recipient"}]
        progress.draft = {"verified": True}
        self.assertIn("not verified", agent._entry_check({"action": "done", "index": 2}, progress))
        agent._auto_verify(progress, lambda *args: None)
        self.assertNotEqual(agent.plan.items[2]["status"], "done")

    def test_read_only_verification_before_a_send_does_not_require_message_proof(self):
        agent, progress = BrowserAgent(Model()), Progress()
        agent.plan.apply(json.dumps({"action": "plan", "items": [
            {"text": "Verify article dates", "kind": "verify"}, {"text": "Send summary", "kind": "send"}]}))
        self.assertEqual(agent._entry_check({"action": "done", "index": 1}, progress), "")

    def test_validation_retries_do_not_take_screenshots_and_halt(self):
        calls = [tool("browser", command="type", args=["x" * 2001]) for _ in range(3)]
        model = Model([Turn(tool_calls=[call]) for call in calls])
        agent = BrowserAgent(model, run_tool=browser_tool.dispatch, look=lambda: self.fail("no screenshot"),
                             memory=NullMemory())
        result = agent.run("Fill a search field", lambda *args: None, Event())
        self.assertIn("repeated invalid tool arguments", result)
        self.assertEqual(len(model.requests), 3)

    def test_a_corrected_compose_payload_is_not_blocked_by_its_short_log_label(self):
        model = Model()
        agent, progress = BrowserAgent(model), Progress()
        failed, seen = {}, {}
        with mock.patch.dict(agent.handlers, {"browser": lambda raw: (
                browser_tool.VALIDATION_ERROR + "bad" if not json.loads(raw).get("text")
                else PAGE + "\n- Draft SHA256: " + browser_tool.text_digest("fixed"))}):
            for _ in range(2):
                agent._run_call(tool("browser", command="compose", args=["e9"], text=""), 0, progress,
                                failed, seen, lambda *args: None)
            result = agent._run_call(tool("browser", command="compose", args=["e9"], text="fixed",
                                         recipient="Recipient"), 0, progress, failed, seen, lambda *args: None)
        self.assertFalse(result.startswith("Error"))


class LearningTests(unittest.TestCase):
    def test_oversized_memory_is_not_cut_mid_instruction(self):
        from memory import Memory
        with tempfile.TemporaryDirectory() as folder:
            memory = Memory(Path(folder))
            memory.save_long_term("- Keep the complete message.")
            with self.assertRaises(ValueError):
                memory.save_long_term("- " + "x" * 301 + " do not split it")
            self.assertEqual(memory.long_term(), "- Keep the complete message.")

    def test_same_topic_replaces_instead_of_appending_conflicting_rules(self):
        with tempfile.TemporaryDirectory() as folder:
            books = Playbooks(Path(folder))
            books.save(Teaching("Send a message", "https://example.com/", "Example"), "")
            books.add_rule("Send a message", "Shorten the message", topic="message-delivery")
            result = books.add_rule("Send a message", "Compose all input chunks in one draft, then send once",
                                    topic="message-delivery", source="Paste the parts into one message; do not shorten it.")
            guide = books.all()[0].section("Guide")
            self.assertTrue(result.startswith("Saved"))
            self.assertNotIn("Shorten the message", guide)
            self.assertIn("do not shorten it", guide)
            self.assertEqual(guide.count("- User rule:"), 1)

    def test_legacy_rule_can_be_replaced_by_its_id_and_long_rules_are_not_truncated(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            books = Playbooks(Path(folder))
            books.save(Teaching("Send a message", "https://example.com/", "Example"), "")
            old = "Send chunks separately"
            books.add_rule("Send a message", old)
            key = hashlib.sha256(old.encode("utf-8")).hexdigest()[:12]
            self.assertIn(key, books.apply(json.dumps({"action": "list"})))
            rule = "Keep complete content. " * 30
            books.add_rule("Send a message", rule, replace=[key])
            guide = books.all()[0].section("Guide")
            self.assertNotIn(old, guide)
            self.assertIn(rule.strip(), guide)
            self.assertTrue(books.add_rule("Send a message", "x" * 2001).startswith("Error"))
