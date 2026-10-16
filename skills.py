"""Agent Skills style skills, with progressive disclosure.

1. Catalog: only the name + description of each SKILL.md go into the system prompt (few tokens).
2. Index: the `skill` tool without a section returns the titles (## ...) and the reference files.
3. Content: the `skill` tool with a section returns only that part.

The playwright-cli skill was generated with `playwright-cli install --skills`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILLS_DIR = HERE / ".claude" / "skills"
MAX_SECTION = 8_000


@dataclass
class Skill:
    name: str
    description: str
    path: Path

    @property
    def body(self) -> str:
        text = self.path.read_text(encoding="utf-8")
        return re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)

    @property
    def references(self) -> list[str]:
        folder = self.path.parent / "references"
        return sorted(item.stem for item in folder.glob("*.md")) if folder.is_dir() else []


def load_skills(folder: Path = SKILLS_DIR) -> dict[str, Skill]:
    skills = {}
    for path in sorted(folder.glob("*/SKILL.md")):
        header = re.match(r"\A---\n(.*?)\n---\n", path.read_text(encoding="utf-8"), flags=re.S)
        fields = dict(re.findall(r"^(\w[\w-]*):\s*(.+)$", header.group(1), flags=re.M)) if header else {}
        name = fields.get("name", path.parent.name)
        skills[name] = Skill(name, fields.get("description", ""), path)
    return skills


def split_sections(body: str) -> dict[str, str]:
    """Splits the markdown at '## ' titles, ignoring lines inside code blocks."""
    sections: dict[str, list[str]] = {}
    current, in_code = "", False
    for line in body.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and line.startswith("## "):
            current = line[3:].strip()
        sections.setdefault(current, []).append(line)
    return {title: "\n".join(lines).strip() for title, lines in sections.items() if title}


SKILLS = load_skills()


def catalog() -> str:
    if not SKILLS:
        return ""
    lines = [f"- {skill.name}: {skill.description}" for skill in SKILLS.values()]
    return ("Available skills (read them with the `skill` tool only when needed):\n" + "\n".join(lines))


def read_skill(name: str, section: str = "") -> str:
    skill = SKILLS.get(name)
    if not skill:
        return f"Error: unknown skill: {name}. Available: {', '.join(SKILLS) or 'none'}."
    sections = split_sections(skill.body)
    if not section:
        return (f"Skill {name}. Sections: {'; '.join(sections)}.\n"
                f"References: {', '.join(skill.references)}.\n"
                "Ask for a section or reference by name.")
    if section in skill.references:
        text = (skill.path.parent / "references" / f"{section}.md").read_text(encoding="utf-8")
    else:
        match = next((title for title in sections if title.lower() == section.lower()), None)
        if not match:
            return f"Error: unknown section: {section}. Sections: {'; '.join(sections)}."
        text = sections[match]
    return text[:MAX_SECTION]


TOOL = {
    "type": "function",
    "function": {
        "name": "skill",
        "description": (
            "Reads an installed skill. Without `section` it returns the index; with `section` it returns only "
            "that section or reference. The skill was written for agents with a shell: here only "
            "the commands accepted by the `browser` tool work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": list(SKILLS) or [""]},
                "section": {"type": "string", "description": "Section title or reference name."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}


def dispatch(raw_arguments: str) -> str:
    try:
        arguments = json.loads(raw_arguments or "{}")
        return read_skill(str(arguments.get("name", "")), str(arguments.get("section") or ""))
    except (ValueError, OSError, AttributeError) as error:
        return f"Error: {error}"
