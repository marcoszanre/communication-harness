# Communication Harness — Extending the Browser Harness with a 3-Level Local Telegram Channel

**Communication Harness** builds directly on the **Browser Use Tool Harness** to explore the next architectural milestone in agent harness engineering: **decoupling human interaction from a single terminal window by adding an external communication channel**.

Without introducing any cloud webhook servers, reverse proxies, or abstract event-bus infrastructure, the harness connects its local **Textual Terminal UI (TUI)** and **Playwright browser loop** to **Telegram** using two direct HTTP primitives from the local process:

1. **Outbound Push (`POST /sendMessage`, `POST /editMessageText`)** — streaming task start, live plan progress, human-in-the-loop prompts, and final outcomes to your phone.
2. **Inbound Pull (`POST /getUpdates`)** — polling only when needed to receive human approvals, clarifying answers, or new remote tasks.

---

## Why Progressive Capability Levels?

Opening an external messaging channel to an autonomous browser agent immediately raises three systems-engineering questions:

1. **What is the external channel allowed to do?** (Observe only, resolve human-in-the-loop gates, or trigger new tasks?)
2. **Which commands belong only on the physical machine?** (Recording a browser demonstration with `/teach` or signing in via `/login` requires a human sitting in front of the local browser window.)
3. **What happens when a task is already running?** (Unlike coding agents where users steer or interrupt mid-thought, browser tasks mutate stateful web pages—queueing or mid-run steering creates unpredictable state collisions.)

To make these boundaries explicit and didactic, **Communication Harness** implements **three progressive capability levels** selectable directly from the TUI (`Telegram` button or `/telegram`):

| Level | Mode ID | Outbound Push | Inbound Pull (`getUpdates`) | What the User Can Do on Telegram |
|---|---|---|---|---|
| **Off** | `off` | Disabled | Disabled | Channel is completely inactive. |
| **Level 1: Task Update** | `status` | Task start, live plan edits, final result | Pairing only (`/start`) | **Read-only status monitor.** Follow long-running browser tasks from your phone. |
| **Level 2: HITL** | `hitl` | Level 1 + inline approval & question buttons | Callback buttons, text answers, `/status`, `/stop` | **Remote decision gate.** Approve/deny irreversible actions (`ApprovalPanel`) and answer clarifying questions (`QuestionPanel`) from Telegram or TUI—first responder wins. |
| **Level 3: Accept Tasks** | `tasks` | Level 2 (Full Push) | Level 2 + natural-language task prompts | **Full bidirectional remote control.** Start new browser tasks from Telegram when the harness is idle, using playbooks and memory learned locally in the TUI. |

---

## Architecture

```text
  ┌────────────────────────────────────────── Local Machine ──────────────────────────────────────────┐
  │                                                                                                   │
  │  ┌─────────────────────────────── Terminal UI (`tui.py` · Textual) ────────────────────────────┐  │
  │  │  Chat Timeline · Live Plan · Inline HITL (`ApprovalPanel` / `QuestionPanel`)                │  │
  │  │  Action Bar: [ Close browser ]  [ Models ]  [ Telegram ]  [ Stop ]                          │  │
  │  └───────────────┬───────────────────────────────────────────────────────┬─────────────────────┘  │
  │                  │                                                       │                        │
  │                  ▼                                                       ▼                        │
  │  ┌─────────────────────────────────────────┐           ┌───────────────────────────────────────┐  │
  │  │ Browser Agent (`agent.py`, `planner.py`)│           │ Local Telegram Bridge                 │  │
  │  │ • Azure AI Foundry (`gpt-6-luna` / etc.)│           │ (`telegram_channel.py`)               │  │
  │  │ • Foundry Local (on-device)             │           │ • Level 1: Push status & live plan    │  │
  │  │ • `playwright-cli` → Chromium           │           │ • Level 2: Race-free HITL sync        │  │
  │  │ • `memory/` & `playbooks/` (`/teach`)   │           │ • Level 3: Pull tasks + guardrails    │  │
  │  └─────────────────────────────────────────┘           └───────────┬───────────────────┬───────┘  │
  │                                                                    │                   ▲          │
  └────────────────────────────────────────────────────────────────────┼───────────────────┼──────────┘
                                                  Outbound HTTPS Push  │                   │ Inbound HTTPS Pull
                                                  (`sendMessage`,      │                   │ (`getUpdates`
                                                   `editMessageText`)  ▼                   │  with offset)
                                                           ┌───────────────────────────────────────┐
                                                           │ Telegram Bot API (`api.telegram.org`) │
                                                           │ Paired Owner Chat (`chat_id`)         │
                                                           └───────────────────────────────────────┘
```

### Core Engineering Design Choices

1. **Zero Cloud Infrastructure & Zero QR/OTP Ceremony.**
   - Create a bot in Telegram with `@BotFather`, tap `Start` (`/start`) in the bot chat, paste the token into the TUI's **Telegram** modal, and click **Save and check**.
   - The harness calls `getMe` (to verify the bot handle) and `getUpdates` (to read your `/start` message, extract your `chat_id` and `@username`, and advance `last_update_id` past historical messages), then pushes a confirmation card to Telegram.
2. **In-Place Live Plan Card (No Notification Spam).**
   - When a task starts, Level 1 sends a single `"🚀 Task started"` message and stores its `message_id`.
   - As the agent progresses through plan steps (`⏳` → `▶` → `✅`), the harness calls `editMessageText` on that same message instead of flooding your phone with separate notifications, followed by one final summary message when finished.
3. **Race-Free Dual-Surface HITL (Level 2 & Level 3).**
   - When the agent hits an approval gate (`_approve`) or asks a clarifying question (`_ask_user`), the TUI mounts its inline `ApprovalPanel` or `QuestionPanel` in the conversation thread *and* pushes an inline keyboard to Telegram.
   - Whichever surface you use first (pressing `y`/`n` in the TUI or tapping `✅ Approve` / `❌ Deny` on Telegram) resolves the gate immediately, unblocks the worker thread, and updates the other surface so stale buttons never remain open.
4. **Surface-Specific Command Guardrails (Level 3).**
   - Commands that require local physical interaction with the terminal or browser window (`/teach`, `/login`, `/models`, `/telegram`, `/clear`, `/restart`, `/close`, `/quit`) are restricted to the local TUI and blocked on Telegram with an explicit explanation.
   - Safe operational commands (`/status`, `/stop`, `/help`) are available directly on Telegram.
5. **Single-Active-Run (`busy-reject`) Concurrency (No Task Queueing).**
   - Browser automation is stateful: running or queueing overlapping tasks against the same browser session creates ambiguity around mid-run steering and interruption.
   - While `busy == True` (or while `/teach` is recording locally in the TUI), the TUI composer is disabled and any new task sent from Telegram receives an immediate `⏳ Busy` notice instructing the user to wait for completion or send `/stop`.
6. **Automatic Browser Lifecycle Cleanup (Auto-Close on Task Completion).**
   - Because `.playwright/cli.config.json` persists cookies and login state in `.profile/`, the harness automatically closes the Chromium window when a task finishes (`Done`, `Failed`, or `Stopped`), freeing system resources while keeping the browser open whenever a task pauses on a human-in-the-loop question (`waiting`) or has pending steps.
7. **Site-Scoped Persistent Memory (`memory/sites/<domain>.md`).**
   - Learnings are partitioned per domain in `memory/sites/<domain>.md` (synchronized with `memory/long_term.md`) with no artificial line cap per site—accumulating as many distinct site learnings as needed and only compacting when a site's memory grows very large.
   - Instead of injecting every site's lessons into every prompt, the harness loads **only** the target site's learnings when the site is named in the request or opened dynamically via `open` / `goto` / `read <url>`—keeping the context lean and consolidating updates per visited site at the end of the turn.

---

## Quick Start

The commands below use PowerShell on Windows (on macOS/Linux, substitute `source .venv/bin/activate`). No account, endpoint, deployment name, bot token, or machine-specific path is hardcoded in the repository.

```powershell
git clone https://github.com/marcoszanre/communication-harness.git
Set-Location communication-harness

# 1. Create the Python virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Install playwright-cli locally and download Playwright Chromium
npm ci
npx playwright-cli install-browser chromium

# 3. Launch the Communication Harness TUI
.\.venv\Scripts\python.exe communication_harness.py
```

---

## Setting Up the Telegram Channel in 60 Seconds

1. Open Telegram and message **`@BotFather`**:
   - Send `/newbot`, choose a display name and username (ending in `bot`), and copy the **Bot Token** (`<bot_id>:<secret>`).
   - Tap the `t.me/<your_bot>` link returned by `@BotFather` and press **Start** (`/start`).
2. In the Communication Harness TUI, click the **`Telegram`** button (next to `Close browser`, `Models`, and `Stop`) or type `/telegram`.
3. Select a capability level (`L1: Task Update`, `L2: HITL`, or `L3: Accept Tasks`), paste your **Bot Token** (masked on screen so it is safe during screen recordings), and click **`Save and check`**.
4. The harness verifies the bot, binds your `chat_id` from your `/start` message, saves `telegram.json` locally (ignored by Git), updates the top status bar (`tg L3:tasks (@your_bot)`), and sends a confirmation message to your Telegram chat.

You can also switch levels at any time from the composer:
- `/telegram` — opens the centered Telegram configuration modal
- `/telegram off` (or `/telegram 0`) — disables the external channel
- `/telegram status` (or `/telegram 1`) — switches to **Level 1: Task Update**
- `/telegram hitl` (or `/telegram 2`) — switches to **Level 2: HITL**
- `/telegram tasks` (or `/telegram 3`) — switches to **Level 3: Accept Tasks**

---

## Didactic Demo Script (Progressive 3-Level Walkthrough)

1. **Act 1 — Level 1 (`Task Update` · Push Only):**
   - Switch to `L1: Task Update` in the `Telegram` modal.
   - Start a multi-step research task in the TUI.
   - Watch the live plan update in place on Telegram and receive the final summary on completion.
   - Try typing a task in Telegram—observe the guardrail explaining that Level 1 is Push-only.
2. **Act 2 — Level 2 (`HITL` · Remote Approvals & Questions):**
   - Switch to `L2: HITL`.
   - Run a task that triggers a clarifying question (`QuestionPanel`) or an irreversible action approval (`ApprovalPanel`).
   - Tap `✅ Approve` (or pick an option) on your phone and watch the TUI inline panel resolve immediately (`✓ Approved [Telegram]`) as the browser continues.
3. **Act 3 — Level 3 (`Accept Tasks` · Full Bidirectional + Guardrails):**
   - Switch to `L3: Accept Tasks`.
   - Send `/teach something` from Telegram—observe the **Surface-Specific Guardrail** keeping `/teach` local-only.
   - Send a natural-language browser task from Telegram while the harness is idle—watch `You [Telegram · @user]` appear in the TUI timeline and the browser execute the task using your locally learned playbooks and memory.
   - While it is running, send a second task from Telegram—observe the **Single-Active-Run (`busy-reject`)** guardrail respond immediately without queueing.

---

## Testing & Verification

Run the complete offline unit and TUI test suite (180+ tests, including mocked Telegram Bot API tests for all 3 levels, pairing, HITL synchronization, surface guardrails, and `busy-reject`):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```
