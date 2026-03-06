from __future__ import annotations

import random
from typing import Iterable

from deceptive_text_env.agents import build_agent
from deceptive_text_env.config import FrameworkConfig
from deceptive_text_env.evaluation.metrics import aggregate_results, inference_accuracy
from deceptive_text_env.llm import create_llm_client
from deceptive_text_env.npcs import build_npc_roster
from deceptive_text_env.types import EpisodeResult
from deceptive_text_env.world import GroundedVerifier, JudgeModel, TextWorldEnvironment, build_world_facts


class EvaluationRunner:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.agent_client = create_llm_client(config.premium_agent_model)
        self.npc_client = create_llm_client(config.budget_npc_model)
        self.judge_client = create_llm_client(config.judge_model)

    def run_all(self, agent_variants: Iterable[str]) -> tuple[list[EpisodeResult], dict[str, dict[str, float]]]:
        results: list[EpisodeResult] = []
        for liar_ratio in self.config.experiment.liar_ratios:
            for variant in agent_variants:
                for run_index in range(self.config.experiment.runs_per_setting):
                    seed = self._seed_for(variant, liar_ratio, run_index)
                    results.append(self.run_episode(agent_variant=variant, liar_ratio=liar_ratio, seed=seed))
        return results, aggregate_results(results)

    def run_episode(self, *, agent_variant: str, liar_ratio: float, seed: int) -> EpisodeResult:
        random.seed(seed)
        world_facts = build_world_facts(self.config.world)
        verifier = GroundedVerifier(world_facts)
        judge = JudgeModel(self.judge_client, self.config.judge_model)
        npcs = build_npc_roster(
            total_npcs=self.config.experiment.total_npcs,
            liar_ratio=liar_ratio,
            llm_client=self.npc_client,
            model_config=self.config.budget_npc_model,
        )
        environment = TextWorldEnvironment(
            world_config=self.config.world,
            verifier=verifier,
            judge=judge,
            npcs=npcs,
            max_steps=self.config.experiment.max_steps,
        )
        agent = build_agent(
            variant=agent_variant,
            llm_client=self.agent_client,
            model_config=self.config.premium_agent_model,
            experiment_config=self.config.experiment,
        )

        observation = environment.reset()
        agent.reset([npc.name for npc in npcs])
        done = False
        steps = 0

        while not done and steps < self.config.experiment.max_steps:
            action = agent.select_action(observation)
            result = environment.step(action, agent.trust_scores)
            agent.process_step_result(result)
            observation = result.observation
            steps += 1
            done = result.done or steps >= self.config.experiment.max_steps

        hidden_roles = environment.hidden_role_map()
        score = inference_accuracy(agent.trust_scores, hidden_roles, self.config.experiment)
        return EpisodeResult(
            agent_variant=agent_variant,
            liar_ratio=liar_ratio,
            seed=seed,
            success=environment.state.completed,
            steps=steps,
            final_trust_scores=dict(agent.trust_scores),
            inference_accuracy=score,
            recovery_rate=agent.recovery_rate(),
            hidden_roles=hidden_roles,
            trace=list(agent.trace),
        )

    @staticmethod
    def _seed_for(agent_variant: str, liar_ratio: float, run_index: int) -> int:
        return abs(hash((agent_variant, liar_ratio, run_index))) % (2**31)
