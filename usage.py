"""Usage per configured model: calls, input (cached), output tokens and reported cost."""

from __future__ import annotations

from threading import Lock

FIELDS = ("calls", "input", "cached", "output", "cost")


class Usage:
    def __init__(self) -> None:
        self._lock = Lock()
        self.models: dict[str, dict[str, float]] = {}

    def add(self, model: str, input: int | None = 0, output: int | None = 0, cached: int | None = 0,
            cost: float | None = None) -> None:
        with self._lock:
            row = self.models.setdefault(model or "?", dict.fromkeys(FIELDS, 0))
            row["calls"] += 1
            row["input"] += input or 0
            row["output"] += output or 0
            row["cached"] += cached or 0
            row["cost"] += cost or 0.0

    def reset(self) -> None:
        with self._lock:
            self.models.clear()

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {model: dict(row) for model, row in self.models.items()}

    def since(self, before: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        """What was used after `before` (a snapshot), per model."""
        out = {}
        for model, row in self.snapshot().items():
            old = before.get(model, {})
            diff = {key: row[key] - old.get(key, 0) for key in FIELDS}
            if diff["calls"]:
                out[model] = {key: round(value, 8) if key == "cost" else int(value) for key, value in diff.items()}
        return out

    @staticmethod
    def total(rows: dict[str, dict[str, float]]) -> dict[str, float]:
        return {key: sum(row.get(key, 0) for row in rows.values()) for key in FIELDS}

    # ---------- views ----------
    def footer(self, ctx: int = 0) -> str:
        """TUI footer: latest context and totals across model roles."""
        rows = self.snapshot()
        llm = self.total(rows)
        text = f"ctx {ctx:,}  In {int(llm['input']):,}  Out {int(llm['output']):,}"
        return text

    @staticmethod
    def line(rows: dict[str, dict[str, float]]) -> str:
        """One line per turn for the session memory."""
        parts = []
        for model, row in rows.items():
            part = f"{model}: {int(row['calls'])} calls, {int(row['input']):,} in"
            if row.get("cached"):
                part += f" ({int(row['cached']):,} cached)"
            part += f", {int(row['output']):,} out"
            if row.get("cost"):
                part += f", ${row['cost']:.4f}"
            parts.append(part)
        return " | ".join(parts)

    def report(self) -> str:
        return table(self.snapshot())


def table(rows: dict[str, dict[str, float]]) -> str:
    """Markdown table: one row per model plus the total; unknown cost is shown as a dash."""
    if not rows:
        return "(no model calls yet)"
    lines = ["| Model | Calls | Input | Cached | Output | Cost (USD) |", "|---|---:|---:|---:|---:|---:|"]
    for model, row in sorted(rows.items(), key=lambda item: -item[1]["input"]):
        lines.append(_row(model, row))
    if len(rows) > 1:
        lines.append(_row("**Total**", Usage.total(rows)))
    return "\n".join(lines)


def _row(name: str, row: dict[str, float]) -> str:
    cost = f"{row['cost']:.6f}" if row.get("cost") else "-"
    return (f"| {name} | {int(row['calls']):,} | {int(row['input']):,} | {int(row['cached']):,} | "
            f"{int(row['output']):,} | {cost} |")
