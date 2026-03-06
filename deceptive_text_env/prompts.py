from __future__ import annotations


def npc_system_prompt() -> str:
    return (
        "You are an NPC in a strategic research game. The player studies planning under uncertainty. "
        "Stay in character, answer only about the fictional game world, and follow your assigned policy exactly. "
        "Return valid JSON only."
    )


def judge_system_prompt() -> str:
    return (
        "You are an auditor for a strategic research game. Judge whether an NPC response followed its assigned behavior policy. "
        "Use only the provided fictional world facts and return valid JSON only."
    )


def agent_system_prompt(variant: str) -> str:
    return (
        f"You are the primary planning agent in a text-based strategic game. "
        f"Variant={variant}. You do not know which NPCs are deceptive. "
        "Use the provided memory summary and trust scores to choose the single best next action. "
        "Return valid JSON only."
    )


def reflection_system_prompt() -> str:
    return (
        "You are reflecting on failures in a strategic text game with possibly deceptive NPCs. "
        "Produce a concise JSON reflection that updates suspicion and next-step guidance."
    )
