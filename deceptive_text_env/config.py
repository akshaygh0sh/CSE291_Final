from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    provider: str
    model_name: str
    api_key_env_var: str = "TRITONAI_API_KEY"
    base_url: str = "https://tritonai-api.ucsd.edu/v1"
    temperature: float = 0.2
    timeout_seconds: int = 300
    max_tokens: int = 1024
    seed: int = 42


@dataclass
class WorldConfig:
    start_location: str = "village_square"
    location_graph: dict[str, list[str]] = field(
        default_factory=lambda: {
            "village_square": ["forest_shrine", "cave_pool", "river_dock", "ancient_vault"],
            "forest_shrine": ["village_square"],
            "cave_pool": ["village_square"],
            "river_dock": ["village_square"],
            "ancient_vault": ["village_square"],
        }
    )
    location_descriptions: dict[str, str] = field(
        default_factory=lambda: {
            "village_square": "A central plaza where wandering advisors gather around an old map table.",
            "forest_shrine": "A moss-covered shrine where sunbeams pierce the canopy in sharp golden lines.",
            "cave_pool": "A silent cave pool reflecting pale light from cracks in the stone ceiling.",
            "river_dock": "A weathered dock with old cargo crates and rushing water below.",
            "ancient_vault": "A sealed vault door marked with three sigil sockets and a rotating dial.",
        }
    )
    sigil_locations: dict[str, str] = field(
        default_factory=lambda: {
            "sun_sigil": "forest_shrine",
            "moon_sigil": "cave_pool",
            "river_sigil": "river_dock",
        }
    )
    vault_order: list[str] = field(default_factory=lambda: ["sun_sigil", "moon_sigil", "river_sigil"])
    decoy_items: dict[str, str] = field(
        default_factory=lambda: {
            "forest_shrine": "A cracked prayer bowl lies beside the altar.",
            "cave_pool": "An empty lantern flickers weakly near the water.",
            "river_dock": "A rotten coil of rope hangs from a piling.",
            "ancient_vault": "Dust gathers around ancient engravings that describe ritual order.",
        }
    )


@dataclass
class ExperimentConfig:
    max_steps: int = 24
    runs_per_setting: int = 10
    total_npcs: int = 6
    liar_ratios: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5])
    reflection_interval: int = 4
    trust_success_gain: float = 0.18
    trust_failure_decay: float = 0.42
    contradiction_decay: float = 0.12
    distrust_threshold: float = 0.30
    role_target_trust: dict[str, float] = field(
        default_factory=lambda: {
            "truthful": 1.0,
            "deceptive": 0.0,
            "opportunistic": 0.35,
            "partial_truth": 0.25,
            "coordinated_deceptive": 0.0,
        }
    )


@dataclass
class FrameworkConfig:
    premium_agent_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            provider="mock",
            model_name="gpt-4o",
            temperature=0.1,
        )
    )
    budget_npc_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            provider="mock",
            model_name="gpt-4o-mini",
            temperature=0.3,
        )
    )
    judge_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            provider="mock",
            model_name="gpt-4o-mini",
            temperature=0.0,
        )
    )
    world: WorldConfig = field(default_factory=WorldConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    metadata: dict[str, Any] = field(default_factory=dict)


def build_default_config() -> FrameworkConfig:
    """Default config using mock LLM for fast local testing."""
    return FrameworkConfig()


def build_tritonai_config() -> FrameworkConfig:
    """Config using the UCSD TritonAI API for real LLM experiments."""
    return FrameworkConfig(
        premium_agent_model=ModelConfig(
            provider="tritonai",
            model_name="api-gpt-oss-120b",
            api_key_env_var="TRITONAI_API_KEY",
            base_url="https://tritonai-api.ucsd.edu/v1",
            temperature=0.1,
            max_tokens=1024,
            seed=42,
        ),
        budget_npc_model=ModelConfig(
            provider="tritonai",
            model_name="api-gpt-oss-120b",
            api_key_env_var="TRITONAI_API_KEY",
            base_url="https://tritonai-api.ucsd.edu/v1",
            temperature=0.3,
            max_tokens=512,
            seed=42,
        ),
        judge_model=ModelConfig(
            provider="tritonai",
            model_name="api-gpt-oss-120b",
            api_key_env_var="TRITONAI_API_KEY",
            base_url="https://tritonai-api.ucsd.edu/v1",
            temperature=0.0,
            max_tokens=512,
            seed=42,
        ),
        experiment=ExperimentConfig(
            max_steps=24,
            runs_per_setting=3,
            total_npcs=6,
            liar_ratios=[0.0, 0.1, 0.3, 0.5],
        ),
    )


def build_hybrid_config() -> FrameworkConfig:
    """Real LLM agent + mock NPCs (cheap iteration)."""
    return FrameworkConfig(
        premium_agent_model=ModelConfig(
            provider="tritonai",
            model_name="api-gpt-oss-120b",
            api_key_env_var="TRITONAI_API_KEY",
            base_url="https://tritonai-api.ucsd.edu/v1",
            temperature=0.1,
            max_tokens=1024,
            seed=42,
        ),
        budget_npc_model=ModelConfig(
            provider="mock",
            model_name="mock-npc",
            temperature=0.3,
        ),
        judge_model=ModelConfig(
            provider="mock",
            model_name="mock-judge",
            temperature=0.0,
        ),
        experiment=ExperimentConfig(
            max_steps=24,
            runs_per_setting=3,
            total_npcs=6,
            liar_ratios=[0.0, 0.1, 0.3, 0.5],
        ),
    )


def build_hard_config() -> FrameworkConfig:
    """Hard mode: reduced step budget (18) + NPCs spread across locations.
    All components use real LLM."""
    return FrameworkConfig(
        premium_agent_model=ModelConfig(
            provider="tritonai",
            model_name="api-gpt-oss-120b",
            api_key_env_var="TRITONAI_API_KEY",
            base_url="https://tritonai-api.ucsd.edu/v1",
            temperature=0.1,
            max_tokens=1024,
            seed=42,
        ),
        budget_npc_model=ModelConfig(
            provider="tritonai",
            model_name="api-gpt-oss-120b",
            api_key_env_var="TRITONAI_API_KEY",
            base_url="https://tritonai-api.ucsd.edu/v1",
            temperature=0.3,
            max_tokens=512,
            seed=42,
        ),
        judge_model=ModelConfig(
            provider="tritonai",
            model_name="api-gpt-oss-120b",
            api_key_env_var="TRITONAI_API_KEY",
            base_url="https://tritonai-api.ucsd.edu/v1",
            temperature=0.0,
            max_tokens=512,
            seed=42,
        ),
        experiment=ExperimentConfig(
            max_steps=18,
            runs_per_setting=3,
            total_npcs=6,
            liar_ratios=[0.0, 0.1, 0.3, 0.5],
        ),
    )


def build_hard_hybrid_config() -> FrameworkConfig:
    """Hard mode with mock NPCs: reduced step budget (18) + NPCs spread across locations."""
    return FrameworkConfig(
        premium_agent_model=ModelConfig(
            provider="tritonai",
            model_name="api-gpt-oss-120b",
            api_key_env_var="TRITONAI_API_KEY",
            base_url="https://tritonai-api.ucsd.edu/v1",
            temperature=0.1,
            max_tokens=1024,
            seed=42,
        ),
        budget_npc_model=ModelConfig(
            provider="mock",
            model_name="mock-npc",
            temperature=0.3,
        ),
        judge_model=ModelConfig(
            provider="mock",
            model_name="mock-judge",
            temperature=0.0,
        ),
        experiment=ExperimentConfig(
            max_steps=18,
            runs_per_setting=3,
            total_npcs=6,
            liar_ratios=[0.0, 0.1, 0.3, 0.5],
        ),
    )
