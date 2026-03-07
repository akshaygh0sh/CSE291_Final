from __future__ import annotations

from collections import defaultdict
from typing import Any

from deceptive_text_env.config import ExperimentConfig, ModelConfig
from deceptive_text_env.llm import BaseLLMClient
from deceptive_text_env.memory import StructuredMemoryArchitecture
from deceptive_text_env.prompts import agent_system_prompt, reflection_system_prompt
from deceptive_text_env.types import (
    AgentAction,
    ContradictionRecord,
    EnvironmentFactRecord,
    NPCStatementRecord,
    Observation,
    StepResult,
)


class BasePlanningAgent:
    def __init__(
        self,
        *,
        variant: str,
        llm_client: BaseLLMClient,
        model_config: ModelConfig,
        experiment_config: ExperimentConfig,
    ) -> None:
        self.variant = variant
        self.llm_client = llm_client
        self.model_config = model_config
        self.experiment_config = experiment_config
        self.memory = StructuredMemoryArchitecture()
        self.trust_scores: dict[str, float] = {}
        self.asked_pairs: set[tuple[str, str]] = set()
        self.failed_unlock_orders: list[list[str]] = []
        self.trace: list[str] = []
        self.recent_failures: list[str] = []
        self.active_hypothesis: dict[str, Any] | None = None
        self.recovery_open: dict[str, int] = {}
        self.recovery_durations: list[int] = []
        self.latest_reflection: dict[str, Any] = {}

    def reset(self, npc_names: list[str]) -> None:
        self.memory.reset()
        self.trust_scores = {name: self._initial_trust(name) for name in npc_names}
        self.asked_pairs.clear()
        self.failed_unlock_orders.clear()
        self.trace.clear()
        self.recent_failures.clear()
        self.active_hypothesis = None
        self.recovery_open.clear()
        self.recovery_durations.clear()
        self.latest_reflection = {}

    def select_action(self, observation: Observation) -> AgentAction:
        if self._should_reflect(observation):
            self._reflect(observation)

        payload = {
            "variant": self.variant,
            "turn_index": observation.turn_index,
            "location": observation.location,
            "description": observation.description,
            "inventory": observation.inventory,
            "collected_sigils": observation.collected_sigils,
            "pending_goal_text": observation.pending_goal_text,
            "available_npcs": observation.visible_npcs,
            "accessible_locations": observation.accessible_locations,
            "available_topics": observation.available_topics,
            "memory_summary": self.memory.summarize(),
            "known_claims": self.memory.claims_by_fact(),
            "environment_facts": [
                {"fact_id": record.fact_id, "value": record.value, "source": record.source}
                for record in self.memory.environment_facts
            ],
            "contradictions": [
                {
                    "fact_id": record.fact_id,
                    "details": record.details,
                    "npc_names": record.npc_names,
                    "disproven_value": record.disproven_value,
                }
                for record in self.memory.detected_contradictions
            ],
            "trust_scores": self.trust_scores,
            "asked_pairs": [list(item) for item in sorted(self.asked_pairs)],
            "failed_unlock_orders": self.failed_unlock_orders,
            "recent_failures": self.recent_failures[-4:],
            "reflection": self.latest_reflection,
        }
        result = self.llm_client.generate_json(
            task="agent_action",
            system_prompt=agent_system_prompt(self.variant),
            user_prompt=(
                "Choose one next action for the text game. Return JSON with action_type, target, topic, content, and metadata."
            ),
            payload=payload,
            model_config=self.model_config,
        )
        action = AgentAction(
            action_type=str(result.get("action_type", "talk")),
            target=str(result.get("target", "")),
            topic=str(result.get("topic", "")),
            content=str(result.get("content", "")),
            metadata=dict(result.get("metadata", {})),
        )
        self._record_hypothesis(action)
        self.trace.append(f"Turn {observation.turn_index}: action={action.action_type} target={action.target} topic={action.topic}")
        return action

    def process_step_result(self, result: StepResult) -> None:
        self.trace.append(result.public_feedback)
        if result.npc_message is not None:
            self._handle_npc_message(result)
        elif result.discovered_item is not None:
            self._handle_discovery(result)
        elif self.active_hypothesis and self.active_hypothesis.get("action_type") == "search" and not result.success:
            self._handle_failed_search(result)
        elif self.active_hypothesis and self.active_hypothesis.get("action_type") == "unlock":
            if result.success:
                self._handle_successful_unlock(result)
            else:
                self._handle_failed_unlock(result)

        if not result.success and result.public_feedback not in self.recent_failures:
            self.recent_failures.append(result.public_feedback)
        self.recent_failures = self.recent_failures[-8:]
        self._close_recovery_if_needed(result.observation.turn_index)

    def recovery_rate(self) -> float | None:
        if not self.recovery_durations:
            return None
        return sum(self.recovery_durations) / len(self.recovery_durations)

    def _initial_trust(self, npc_name: str) -> float:
        if self.variant == "naive":
            return 0.95
        return 0.50

    def _should_reflect(self, observation: Observation) -> bool:
        return (
            self.variant == "reflection_enhanced"
            and observation.turn_index > 0
            and observation.turn_index % self.experiment_config.reflection_interval == 0
        )

    def _reflect(self, observation: Observation) -> None:
        payload = {
            "turn_index": observation.turn_index,
            "memory_summary": self.memory.summarize(),
            "trust_scores": self.trust_scores,
            "recent_failures": self.recent_failures[-5:],
        }
        self.latest_reflection = self.llm_client.generate_json(
            task="agent_reflection",
            system_prompt=reflection_system_prompt(),
            user_prompt=(
                "Summarize likely deception patterns, suspicious NPCs, and the best next focus. Return JSON only."
            ),
            payload=payload,
            model_config=self.model_config,
        )
        self.memory.add_reflection(str(self.latest_reflection.get("summary", "")))

    def _record_hypothesis(self, action: AgentAction) -> None:
        if action.action_type in {"search", "unlock", "move"}:
            self.active_hypothesis = {
                "action_type": action.action_type,
                "fact_id": action.metadata.get("fact_id"),
                "candidate_value": action.metadata.get("candidate_value") or action.target,
                "submitted_order": [segment.strip() for segment in action.content.split(",") if segment.strip()],
            }
        else:
            self.active_hypothesis = {"action_type": action.action_type}
        if action.action_type == "talk" and action.target and action.topic:
            self.asked_pairs.add((action.target, action.topic))

    def _handle_npc_message(self, result: StepResult) -> None:
        assert result.npc_message is not None
        message = result.npc_message
        for claim in message.claims:
            self.memory.add_npc_statement(
                NPCStatementRecord(
                    turn_index=message.turn_index,
                    npc_name=message.npc_name,
                    topic=message.topic,
                    fact_id=claim.fact_id,
                    claimed_value=claim.claimed_value,
                    statement_text=message.text,
                    trust_at_record=self.trust_scores.get(message.npc_name, 0.5),
                )
            )
            self._detect_internal_contradictions(message.npc_name, claim.fact_id, claim.claimed_value, message.turn_index)

    def _handle_discovery(self, result: StepResult) -> None:
        sigil = str(result.discovered_item)
        fact_id = f"location_of:{sigil}"
        location = result.observation.location
        self.memory.add_environment_fact(
            EnvironmentFactRecord(
                turn_index=result.observation.turn_index,
                fact_id=fact_id,
                value=location,
                source="search_success",
            )
        )
        for record in self.memory.npc_statements:
            if record.fact_id != fact_id:
                continue
            if record.claimed_value == location:
                self._reward_npc(record.npc_name)
            else:
                self._penalize_npc(
                    record.npc_name,
                    reason=f"Claimed {fact_id}={record.claimed_value}, but the item was found at {location}.",
                    turn_index=result.observation.turn_index,
                )

    def _handle_failed_search(self, result: StepResult) -> None:
        fact_id = str(self.active_hypothesis.get("fact_id", ""))
        location = str(self.active_hypothesis.get("candidate_value", result.observation.location))
        contradicted_npcs = [
            record.npc_name
            for record in self.memory.npc_statements
            if record.fact_id == fact_id and record.claimed_value == location
        ]
        if contradicted_npcs:
            detail = f"Search disproved {fact_id} = {location}."
            self.memory.add_contradiction(
                ContradictionRecord(
                    turn_index=result.observation.turn_index,
                    fact_id=fact_id,
                    details=detail,
                    npc_names=sorted(set(contradicted_npcs)),
                    disproven_value=location,
                )
            )
            for npc_name in sorted(set(contradicted_npcs)):
                self._penalize_npc(npc_name, reason=detail, turn_index=result.observation.turn_index)

    def _handle_successful_unlock(self, result: StepResult) -> None:
        self.memory.add_environment_fact(
            EnvironmentFactRecord(
                turn_index=result.observation.turn_index,
                fact_id="vault_order",
                value=self.active_hypothesis.get("submitted_order", []),
                source="unlock_success",
            )
        )
        for record in self.memory.npc_statements:
            if record.fact_id == "vault_order" and record.claimed_value == self.active_hypothesis.get("submitted_order", []):
                self._reward_npc(record.npc_name)

    def _handle_failed_unlock(self, result: StepResult) -> None:
        submitted_order = list(self.active_hypothesis.get("submitted_order", []))
        if submitted_order and submitted_order not in self.failed_unlock_orders:
            self.failed_unlock_orders.append(submitted_order)
        contradicted_npcs = [
            record.npc_name
            for record in self.memory.npc_statements
            if record.fact_id == "vault_order" and record.claimed_value == submitted_order
        ]
        if contradicted_npcs:
            detail = f"Vault rejected order {submitted_order}."
            self.memory.add_contradiction(
                ContradictionRecord(
                    turn_index=result.observation.turn_index,
                    fact_id="vault_order",
                    details=detail,
                    npc_names=sorted(set(contradicted_npcs)),
                    disproven_value=submitted_order,
                )
            )
            for npc_name in sorted(set(contradicted_npcs)):
                self._penalize_npc(npc_name, reason=detail, turn_index=result.observation.turn_index)

    def _detect_internal_contradictions(self, npc_name: str, fact_id: str, claimed_value: Any, turn_index: int) -> None:
        differing_records = [
            record
            for record in self.memory.npc_statements
            if record.fact_id == fact_id and record.claimed_value != claimed_value
        ]
        if not differing_records:
            return
        involved = sorted({npc_name, *[record.npc_name for record in differing_records]})
        detail = f"Conflicting statements detected for {fact_id}."
        self.memory.add_contradiction(
            ContradictionRecord(
                turn_index=turn_index,
                fact_id=fact_id,
                details=detail,
                npc_names=involved,
                disproven_value=None,
            )
        )
        if self.variant in {"belief_tracking", "reflection_enhanced", "memory_augmented"}:
            self._apply_contradiction_decay(involved)

    def _apply_contradiction_decay(self, npc_names: list[str]) -> None:
        no_decay_variants = {"naive", "belief_no_decay"}
        for npc_name in npc_names:
            if self.variant in no_decay_variants:
                continue
            current = self.trust_scores.get(npc_name, 0.5)
            decayed = max(0.0, current - self.experiment_config.contradiction_decay)
            self.trust_scores[npc_name] = decayed

    def _reward_npc(self, npc_name: str) -> None:
        if self.variant == "naive":
            self.trust_scores[npc_name] = 0.99
            return
        if self.variant == "belief_no_decay":
            current = self.trust_scores.get(npc_name, 0.5)
            gain = self.experiment_config.trust_success_gain * (1.0 - current)
            self.trust_scores[npc_name] = min(1.0, current + gain)
            return
        current = self.trust_scores.get(npc_name, 0.5)
        gain = self.experiment_config.trust_success_gain * (1.0 - current)
        self.trust_scores[npc_name] = min(1.0, current + gain)

    def _penalize_npc(self, npc_name: str, *, reason: str, turn_index: int) -> None:
        if self.variant == "naive":
            self.trust_scores[npc_name] = max(0.75, self.trust_scores.get(npc_name, 0.95) - 0.05)
            return
        if self.variant == "belief_no_decay":
            # Ablation: no trust penalty on failure
            self.trace.append(f"(ablation: no decay) {npc_name}: {reason}")
            return
        current = self.trust_scores.get(npc_name, 0.5)
        decayed = max(0.0, current * (1.0 - self.experiment_config.trust_failure_decay) - 0.05)
        self.trust_scores[npc_name] = decayed
        if npc_name not in self.recovery_open:
            self.recovery_open[npc_name] = turn_index
        self.trace.append(f"Trust decay for {npc_name}: {reason}")

    def _close_recovery_if_needed(self, turn_index: int) -> None:
        closed: list[str] = []
        for npc_name, start_turn in self.recovery_open.items():
            if self.trust_scores.get(npc_name, 1.0) <= self.experiment_config.distrust_threshold:
                self.recovery_durations.append(turn_index - start_turn)
                closed.append(npc_name)
        for npc_name in closed:
            del self.recovery_open[npc_name]


class NaiveAgent(BasePlanningAgent):
    def __init__(self, *, llm_client: BaseLLMClient, model_config: ModelConfig, experiment_config: ExperimentConfig) -> None:
        super().__init__(variant="naive", llm_client=llm_client, model_config=model_config, experiment_config=experiment_config)


class MemoryAugmentedAgent(BasePlanningAgent):
    def __init__(self, *, llm_client: BaseLLMClient, model_config: ModelConfig, experiment_config: ExperimentConfig) -> None:
        super().__init__(variant="memory_augmented", llm_client=llm_client, model_config=model_config, experiment_config=experiment_config)


class BeliefTrackingAgent(BasePlanningAgent):
    def __init__(self, *, llm_client: BaseLLMClient, model_config: ModelConfig, experiment_config: ExperimentConfig) -> None:
        super().__init__(variant="belief_tracking", llm_client=llm_client, model_config=model_config, experiment_config=experiment_config)


class ReflectionEnhancedAgent(BasePlanningAgent):
    def __init__(self, *, llm_client: BaseLLMClient, model_config: ModelConfig, experiment_config: ExperimentConfig) -> None:
        super().__init__(variant="reflection_enhanced", llm_client=llm_client, model_config=model_config, experiment_config=experiment_config)


class BeliefNoDecayAgent(BasePlanningAgent):
    """Ablation: belief-tracking but trust never decreases on failure."""

    def __init__(self, *, llm_client: BaseLLMClient, model_config: ModelConfig, experiment_config: ExperimentConfig) -> None:
        super().__init__(variant="belief_no_decay", llm_client=llm_client, model_config=model_config, experiment_config=experiment_config)


class MemoryWithTrustAgent(BasePlanningAgent):
    """Ablation: memory-augmented + trust scores (but no reflection)."""

    def __init__(self, *, llm_client: BaseLLMClient, model_config: ModelConfig, experiment_config: ExperimentConfig) -> None:
        super().__init__(variant="memory_with_trust", llm_client=llm_client, model_config=model_config, experiment_config=experiment_config)


def build_agent(
    *,
    variant: str,
    llm_client: BaseLLMClient,
    model_config: ModelConfig,
    experiment_config: ExperimentConfig,
) -> BasePlanningAgent:
    registry = {
        "naive": NaiveAgent,
        "memory_augmented": MemoryAugmentedAgent,
        "belief_tracking": BeliefTrackingAgent,
        "reflection_enhanced": ReflectionEnhancedAgent,
        "belief_no_decay": BeliefNoDecayAgent,
        "memory_with_trust": MemoryWithTrustAgent,
    }
    if variant not in registry:
        raise ValueError(f"Unknown agent variant: {variant}")
    return registry[variant](llm_client=llm_client, model_config=model_config, experiment_config=experiment_config)
