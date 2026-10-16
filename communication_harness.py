"""Starts the Communication Harness (Browser + 3-Level Telegram Channel) in an interactive terminal."""

from __future__ import annotations

import sys

from browser_tool import close_browser
from tui import BrowserApp


def main() -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("The TUI needs an interactive terminal.", file=sys.stderr)
        return 2
    try:
        BrowserApp().run()
    finally:
        # A visible browser does not close by itself; close the session on exit.
        print(close_browser().splitlines()[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
