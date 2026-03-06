from __future__ import annotations

import json
from typing import Any

from deceptive_text_env.config import ModelConfig
from deceptive_text_env.llm import BaseLLMClient
from deceptive_text_env.prompts import judge_system_prompt
from deceptive_text_env.types import NPCJudgement, VerifiedClaim


class JudgeModel:
    def __init__(self, llm_client: BaseLLMClient, model_config: ModelConfig) -> None:
        self.llm_client = llm_client
        self.model_config = model_config

    def audit_message(
        self,
        *,
        policy: str,
        agent_trust: float,
        turn_index: int,
        pivot_turn: int,
        verified_claims: list[VerifiedClaim],
    ) -> NPCJudgement:
        payload: dict[str, Any] = {
            "policy": policy,
            "agent_trust": agent_trust,
            "turn_index": turn_index,
            "pivot_turn": pivot_turn,
            "verified_claims": [
                {
                    "fact_id": claim.fact_id,
                    "claimed_value": claim.claimed_value,
                    "expected_value": claim.expected_value,
                    "is_true": claim.is_true,
                }
                for claim in verified_claims
            ],
        }
        result = self.llm_client.generate_json(
            task="judge_npc_response",
            system_prompt=judge_system_prompt(),
            user_prompt=(
                "Audit whether the NPC response matched its assigned deceptive or truthful game policy. "
                "Return JSON with compliant, issues, and summary."
            ),
            payload=payload,
            model_config=self.model_config,
        )
        return NPCJudgement(
            compliant=bool(result.get("compliant", True)),
            issues=[str(issue) for issue in result.get("issues", [])],
            summary=str(result.get("summary", "")),
        )

    @staticmethod
    def fallback_audit(
        *,
        policy: str,
        agent_trust: float,
        turn_index: int,
        pivot_turn: int,
        verified_claims: list[VerifiedClaim],
    ) -> NPCJudgement:
        truths = [claim.is_true for claim in verified_claims]
        compliant = True
        issues: list[str] = []
        if policy == "truthful" and not all(truths):
            compliant = False
            issues.append("Truthful policy violation.")
        if policy == "deceptive" and agent_trust >= 0.65 and any(truths):
            compliant = False
            issues.append("High-trust deceptive policy violation.")
        if policy == "opportunistic":
            if turn_index < pivot_turn and not all(truths):
                compliant = False
                issues.append("Pre-pivot opportunistic policy violation.")
            if turn_index >= pivot_turn and all(truths):
                compliant = False
                issues.append("Post-pivot opportunistic policy violation.")
        return NPCJudgement(compliant=compliant, issues=issues, summary=json.dumps({"issues": issues}))
