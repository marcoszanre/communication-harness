"""Usage report from session logs: per session and per configured model.

    python report.py              # every session
    python report.py --last 5     # the 5 latest sessions
    python report.py logs\\x.jsonl  # specific files

Unknown legacy events are ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from usage import Usage, table

LOGS_DIR = Path(__file__).resolve().parent / "logs"


def read_session(path: Path) -> tuple[Usage, int]:
    """Usage per model of one log file, and its number of turns."""
    usage, turns, model = Usage(), 0, "llm"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind = event.get("event")
        if kind == "turn_start":
            turns += 1
            model = event.get("model") or model
        elif kind == "model_call" or (kind == "memory" and event.get("input") is not None):
            usage.add(event.get("model") or model, event.get("input"), event.get("output"), event.get("cached"))
    return usage, turns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--last", type=int, default=0, help="only the N latest sessions")
    args = parser.parse_args(argv)
    files = args.files or sorted(LOGS_DIR.glob("*.jsonl"))
    if args.last:
        files = files[-args.last:]
    if not files:
        print("No logs found.")
        return 1
    grand, all_turns = Usage(), 0
    for path in files:
        usage, turns = read_session(path)
        rows = usage.snapshot()
        if not rows:
            continue
        all_turns += turns
        for model, row in rows.items():
            with grand._lock:
                total = grand.models.setdefault(model, dict.fromkeys(row, 0))
                for key, value in row.items():
                    total[key] += value
        llm_calls = sum(row["calls"] for row in rows.values())
        print(f"\n## {path.stem} - {turns} turns, {int(llm_calls)} LLM calls"
              f"{f' ({llm_calls / turns:.1f}/turn)' if turns else ''}")
        print(table(rows))
    rows = grand.snapshot()
    llm = Usage.total(rows)
    print(f"\n# Total - {len(files)} sessions, {all_turns} turns")
    print(table(rows))
    if all_turns:
        print(f"\nPer turn: {llm['calls'] / all_turns:.1f} LLM calls, {llm['input'] / all_turns:,.0f} LLM input tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
