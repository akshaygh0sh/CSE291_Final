from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Claim:
    fact_id: str
    claimed_value: Any
    natural_language: str


@dataclass
class VerifiedClaim(Claim):
    expected_value: Any
    is_true: bool


@dataclass
class NPCMessage:
    npc_name: str
    policy: str
    topic: str
    text: str
    claims: list[Claim]
    turn_index: int


@dataclass
class NPCJudgement:
    compliant: bool
    issues: list[str]
    summary: str


@dataclass
class Observation:
    turn_index: int
    location: str
    description: str
    visible_npcs: list[str]
    accessible_locations: list[str]
    inventory: list[str]
    collected_sigils: list[str]
    pending_goal_text: str
    available_topics: list[str]
    last_event: str


@dataclass
class AgentAction:
    action_type: str
    target: str = ""
    topic: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    observation: Observation
    success: bool
    done: bool
    reward: float
    public_feedback: str
    npc_message: Optional[NPCMessage] = None
    hidden_verified_claims: list[VerifiedClaim] = field(default_factory=list)
    discovered_item: Optional[str] = None
    contradiction_fact_ids: list[str] = field(default_factory=list)


@dataclass
class NPCStatementRecord:
    turn_index: int
    npc_name: str
    topic: str
    fact_id: str
    claimed_value: Any
    statement_text: str
    trust_at_record: float


@dataclass
class ContradictionRecord:
    turn_index: int
    fact_id: str
    details: str
    npc_names: list[str]
    disproven_value: Any | None = None


@dataclass
class EnvironmentFactRecord:
    turn_index: int
    fact_id: str
    value: Any
    source: str


@dataclass
class EpisodeResult:
    agent_variant: str
    liar_ratio: float
    seed: int
    success: bool
    steps: int
    final_trust_scores: dict[str, float]
    inference_accuracy: float
    recovery_rate: Optional[float]
    hidden_roles: dict[str, str]
    trace: list[str]
