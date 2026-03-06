from __future__ import annotations

from typing import Any

from deceptive_text_env.types import Claim, VerifiedClaim


class GroundedVerifier:
    def __init__(self, immutable_world_facts: dict[str, Any]) -> None:
        self.immutable_world_facts = dict(immutable_world_facts)

    def verify_claims(self, claims: list[Claim]) -> list[VerifiedClaim]:
        verified: list[VerifiedClaim] = []
        for claim in claims:
            expected = self.immutable_world_facts.get(claim.fact_id)
            verified.append(
                VerifiedClaim(
                    fact_id=claim.fact_id,
                    claimed_value=claim.claimed_value,
                    natural_language=claim.natural_language,
                    expected_value=expected,
                    is_true=claim.claimed_value == expected,
                )
            )
        return verified

    def get_true_value(self, fact_id: str) -> Any:
        return self.immutable_world_facts[fact_id]

    def fact_exists(self, fact_id: str) -> bool:
        return fact_id in self.immutable_world_facts
