from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deceptive_text_env.config import ModelConfig
from deceptive_text_env.llm import BaseLLMClient
from deceptive_text_env.prompts import npc_system_prompt
from deceptive_text_env.types import Claim, NPCMessage, VerifiedClaim
from deceptive_text_env.world.judge import JudgeModel
from deceptive_text_env.world.verifier import GroundedVerifier


@dataclass(slots=True)
class BaseNPC:
    name: str
    location: str
    llm_client: BaseLLMClient
    model_config: ModelConfig
    policy: str
    pivot_turn: int = 3
    turn_counter: int = 0
    knowledge_topics: list[str] = field(default_factory=lambda: [
        "location_of:sun_sigil",
        "location_of:moon_sigil",
        "location_of:river_sigil",
        "vault_order",
    ])

    def reset(self) -> None:
        self.turn_counter = 0

    def respond(
        self,
        *,
        topic: str,
        turn_index: int,
        verifier: GroundedVerifier,
        judge: JudgeModel,
        agent_trust: float,
    ) -> tuple[NPCMessage, list[VerifiedClaim]]:
        self.turn_counter += 1
        if topic not in self.knowledge_topics:
            topic = self.knowledge_topics[0]
        fact_id = topic
        true_value = verifier.get_true_value(fact_id)
        payload = {
            "npc_name": self.name,
            "policy": self.policy,
            "topic": topic,
            "fact_id": fact_id,
            "true_value": true_value,
            "candidates": self._candidate_values(topic, verifier),
            "turn_index": self.turn_counter,
            "global_turn_index": turn_index,
            "agent_trust": agent_trust,
            "pivot_turn": self.pivot_turn,
        }
        result = self.llm_client.generate_json(
            task="npc_response",
            system_prompt=npc_system_prompt(),
            user_prompt=self._user_prompt(topic),
            payload=payload,
            model_config=self.model_config,
        )
        message = self._build_message(topic=topic, turn_index=turn_index, result=result)
        verified_claims = verifier.verify_claims(message.claims)
        try:
            judgement = judge.audit_message(
                policy=self.policy,
                agent_trust=agent_trust,
                turn_index=self.turn_counter,
                pivot_turn=self.pivot_turn,
                verified_claims=verified_claims,
            )
        except RuntimeError:
            judgement = JudgeModel.fallback_audit(
                policy=self.policy,
                agent_trust=agent_trust,
                turn_index=self.turn_counter,
                pivot_turn=self.pivot_turn,
                verified_claims=verified_claims,
            )
        if not judgement.compliant:
            message = self._deterministic_fallback(topic=topic, turn_index=turn_index, verifier=verifier, agent_trust=agent_trust)
            verified_claims = verifier.verify_claims(message.claims)
        return message, verified_claims

    def _build_message(self, *, topic: str, turn_index: int, result: dict[str, Any]) -> NPCMessage:
        claims = [
            Claim(
                fact_id=str(item["fact_id"]),
                claimed_value=item["claimed_value"],
                natural_language=str(item["natural_language"]),
            )
            for item in result.get("claims", [])
        ]
        return NPCMessage(
            npc_name=self.name,
            policy=self.policy,
            topic=topic,
            text=str(result.get("text", "")),
            claims=claims,
            turn_index=turn_index,
        )

    def _candidate_values(self, topic: str, verifier: GroundedVerifier) -> list[Any]:
        if topic.startswith("location_of:"):
            return ["forest_shrine", "cave_pool", "river_dock", "ancient_vault"]
        if topic == "vault_order":
            true_order = verifier.get_true_value(topic)
            return [
                true_order,
                ["moon_sigil", "sun_sigil", "river_sigil"],
                ["river_sigil", "moon_sigil", "sun_sigil"],
                ["sun_sigil", "river_sigil", "moon_sigil"],
            ]
        return [verifier.get_true_value(topic)]

    def _user_prompt(self, topic: str) -> str:
        return (
            f"Answer a single topic for the fictional game world: {topic}. "
            "Follow your assigned policy exactly and return JSON with 'text' and 'claims'."
        )

    def _deterministic_fallback(
        self,
        *,
        topic: str,
        turn_index: int,
        verifier: GroundedVerifier,
        agent_trust: float,
    ) -> NPCMessage:
        true_value = verifier.get_true_value(topic)
        candidates = self._candidate_values(topic, verifier)
        false_candidates = [candidate for candidate in candidates if candidate != true_value]
        chosen_value: Any = true_value
        if self.policy == "deceptive" and agent_trust >= 0.65 and false_candidates:
            chosen_value = false_candidates[0]
        elif self.policy == "opportunistic" and self.turn_counter >= self.pivot_turn and false_candidates:
            chosen_value = false_candidates[0]
        text = self._format_claim_text(topic, chosen_value)
        return NPCMessage(
            npc_name=self.name,
            policy=self.policy,
            topic=topic,
            text=text,
            claims=[Claim(fact_id=topic, claimed_value=chosen_value, natural_language=text)],
            turn_index=turn_index,
        )

    def _format_claim_text(self, topic: str, value: Any) -> str:
        if topic.startswith("location_of:"):
            return f"The {topic.split(':', maxsplit=1)[1].replace('_', ' ')} is at {str(value).replace('_', ' ')}."
        if topic == "vault_order":
            pretty = ", ".join(item.replace("_", " ") for item in value)
            return f"The vault order is {pretty}."
        return str(value)


class TruthfulNPC(BaseNPC):
    def __init__(self, *, name: str, location: str, llm_client: BaseLLMClient, model_config: ModelConfig) -> None:
        super().__init__(name=name, location=location, llm_client=llm_client, model_config=model_config, policy="truthful")


class DeceptiveNPC(BaseNPC):
    def __init__(self, *, name: str, location: str, llm_client: BaseLLMClient, model_config: ModelConfig) -> None:
        super().__init__(name=name, location=location, llm_client=llm_client, model_config=model_config, policy="deceptive")


class OpportunisticNPC(BaseNPC):
    def __init__(self, *, name: str, location: str, llm_client: BaseLLMClient, model_config: ModelConfig, pivot_turn: int = 3) -> None:
        super().__init__(
            name=name,
            location=location,
            llm_client=llm_client,
            model_config=model_config,
            policy="opportunistic",
            pivot_turn=pivot_turn,
        )


def build_npc_roster(
    *,
    total_npcs: int,
    liar_ratio: float,
    llm_client: BaseLLMClient,
    model_config: ModelConfig,
    location: str = "village_square",
) -> list[BaseNPC]:
    liar_count = max(1, int(round(total_npcs * liar_ratio)))
    liar_count = min(liar_count, max(total_npcs - 1, 1))
    truthful_count = max(total_npcs - liar_count, 1)
    deceptive_count = liar_count // 2
    opportunistic_count = liar_count - deceptive_count

    names = [
        "Aster",
        "Bram",
        "Cyra",
        "Dorian",
        "Elara",
        "Fenn",
        "Galen",
        "Helia",
        "Isolde",
        "Jarek",
    ]
    roster: list[BaseNPC] = []
    index = 0

    for _ in range(truthful_count):
        roster.append(TruthfulNPC(name=names[index], location=location, llm_client=llm_client, model_config=model_config))
        index += 1
    for _ in range(deceptive_count):
        roster.append(DeceptiveNPC(name=names[index], location=location, llm_client=llm_client, model_config=model_config))
        index += 1
    for _ in range(opportunistic_count):
        roster.append(
            OpportunisticNPC(name=names[index], location=location, llm_client=llm_client, model_config=model_config, pivot_turn=3)
        )
        index += 1

    return roster[:total_npcs]
