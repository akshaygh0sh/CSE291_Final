from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable

from deceptive_text_env.config import ExperimentConfig
from deceptive_text_env.types import EpisodeResult


def inference_accuracy(
    trust_scores: dict[str, float],
    hidden_roles: dict[str, str],
    experiment_config: ExperimentConfig,
) -> float:
    errors: list[float] = []
    for npc_name, role in hidden_roles.items():
        target = experiment_config.role_target_trust.get(role, 0.5)
        errors.append(abs(trust_scores.get(npc_name, 0.5) - target))
    if not errors:
        return 0.0
    return max(0.0, 1.0 - (sum(errors) / len(errors)))


def aggregate_results(results: Iterable[EpisodeResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[EpisodeResult]] = defaultdict(list)
    for result in results:
        key = f"{result.agent_variant}@{result.liar_ratio:.1f}"
        grouped[key].append(result)

    summary: dict[str, dict[str, float]] = {}
    for key, bucket in grouped.items():
        recovery_values = [value for value in [result.recovery_rate for result in bucket] if value is not None]
        summary[key] = {
            "task_success_rate": mean(1.0 if result.success else 0.0 for result in bucket),
            "avg_steps": mean(result.steps for result in bucket),
            "avg_inference_accuracy": mean(result.inference_accuracy for result in bucket),
            "avg_recovery_rate": mean(recovery_values) if recovery_values else -1.0,
        }
    return summary
