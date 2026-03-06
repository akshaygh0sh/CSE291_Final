from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any

from deceptive_text_env.types import ContradictionRecord, EnvironmentFactRecord, NPCStatementRecord


class StructuredMemoryArchitecture:
    def __init__(self) -> None:
        self.npc_statements: list[NPCStatementRecord] = []
        self.detected_contradictions: list[ContradictionRecord] = []
        self.environment_facts: list[EnvironmentFactRecord] = []
        self.reflection_notes: list[str] = []

    def reset(self) -> None:
        self.npc_statements.clear()
        self.detected_contradictions.clear()
        self.environment_facts.clear()
        self.reflection_notes.clear()

    def add_npc_statement(self, record: NPCStatementRecord) -> None:
        self.npc_statements.append(record)

    def add_contradiction(self, record: ContradictionRecord) -> None:
        self.detected_contradictions.append(record)

    def add_environment_fact(self, record: EnvironmentFactRecord) -> None:
        if any(existing.fact_id == record.fact_id and existing.value == record.value for existing in self.environment_facts):
            return
        self.environment_facts.append(record)

    def add_reflection(self, summary: str) -> None:
        if summary:
            self.reflection_notes.append(summary)

    def summarize(self, max_entries: int = 8) -> str:
        npc_section = self._format_npc_statements(max_entries)
        contradiction_section = self._format_contradictions(max_entries)
        fact_section = self._format_environment_facts(max_entries)
        reflection_section = self._format_reflections(max_entries)
        return (
            "NPC Statements:\n"
            f"{npc_section}\n\n"
            "Detected Contradictions:\n"
            f"{contradiction_section}\n\n"
            "Environment Facts:\n"
            f"{fact_section}\n\n"
            "Reflection Notes:\n"
            f"{reflection_section}"
        )

    def claims_by_fact(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in self.npc_statements:
            grouped[record.fact_id].append(asdict(record))
        return dict(grouped)

    def latest_environment_fact(self, fact_id: str) -> EnvironmentFactRecord | None:
        for record in reversed(self.environment_facts):
            if record.fact_id == fact_id:
                return record
        return None

    def _format_npc_statements(self, max_entries: int) -> str:
        if not self.npc_statements:
            return "- None"
        entries = self.npc_statements[-max_entries:]
        return "\n".join(
            f"- Turn {entry.turn_index}: {entry.npc_name} claimed {entry.fact_id} = {entry.claimed_value}"
            for entry in entries
        )

    def _format_contradictions(self, max_entries: int) -> str:
        if not self.detected_contradictions:
            return "- None"
        entries = self.detected_contradictions[-max_entries:]
        return "\n".join(
            f"- Turn {entry.turn_index}: {entry.details} (NPCs: {', '.join(entry.npc_names)})"
            for entry in entries
        )

    def _format_environment_facts(self, max_entries: int) -> str:
        if not self.environment_facts:
            return "- None"
        entries = self.environment_facts[-max_entries:]
        return "\n".join(
            f"- Turn {entry.turn_index}: {entry.fact_id} = {entry.value} ({entry.source})"
            for entry in entries
        )

    def _format_reflections(self, max_entries: int) -> str:
        if not self.reflection_notes:
            return "- None"
        return "\n".join(f"- {note}" for note in self.reflection_notes[-max_entries:])
