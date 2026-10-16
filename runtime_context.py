"""Authoritative local time and conservative relative-date requirements."""

from __future__ import annotations

import re
from datetime import datetime, timedelta


def clock_context(now: datetime | None = None) -> str:
    local = now or datetime.now().astimezone()
    if local.tzinfo is None:
        local = local.astimezone()
    yesterday = local.date() - timedelta(days=1)
    return (f"Authoritative local time: {local.isoformat(timespec='seconds')}. "
            f"Today: {local.date().isoformat()}. Yesterday: {yesterday.isoformat()}. "
            "Use these civil dates for relative-date requests, not UTC or model knowledge. "
            "Verify publication dates on each source; never infer them from a URL.")


def date_counts(request: str, context: str) -> dict[str, int]:
    if not (re.search(r"\b(today|hoje)\b", request, re.I)
            and re.search(r"\b(yesterday|ontem)\b", request, re.I)):
        return {}
    match = re.search(
        r"\b(one|two|three|four|five|um|uma|dois|duas|tr[eê]s|quatro|cinco|\d{1,2})\s+"
        r"(?:(?:articles?|stories|news|not[ií]cias?)\s+)?"
        r"(?:(?:from|for|per|de|por)\s+)?(?:each\s+(?:day|date)|cada\s+dia)\b", request, re.I)
    if not match:
        return {}
    words = {"one": 1, "um": 1, "uma": 1, "two": 2, "dois": 2, "duas": 2,
             "three": 3, "três": 3, "tres": 3, "four": 4, "quatro": 4, "five": 5, "cinco": 5}
    word = match.group(1).lower()
    count = int(word) if word.isdigit() else words[word]
    dates = re.findall(r"(?:Today|Yesterday): (\d{4}-\d{2}-\d{2})", context)
    return dict.fromkeys(dates, count) if count > 0 and len(dates) == 2 else {}


def publication_date(value: str, context: str) -> str:
    published = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    match = re.search(r"Authoritative local time: (\S+)\.", context or clock_context())
    if published.tzinfo and match:
        request_time = datetime.fromisoformat(match.group(1))
        published = published.astimezone(request_time.tzinfo)
    return published.date().isoformat()
