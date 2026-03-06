from __future__ import annotations

import json

from deceptive_text_env import build_default_config
from deceptive_text_env.evaluation import EvaluationRunner


def main() -> None:
    config = build_default_config()
    runner = EvaluationRunner(config)
    variants = ["naive", "memory_augmented", "belief_tracking", "reflection_enhanced"]
    results, summary = runner.run_all(variants)

    print("=== Experiment Summary ===")
    print(json.dumps(summary, indent=2))
    print("\n=== Example Episode Trace ===")
    if results:
        example = results[0]
        print(f"Variant: {example.agent_variant} | liar_ratio={example.liar_ratio} | success={example.success}")
        for line in example.trace[:20]:
            print(f"- {line}")


if __name__ == "__main__":
    main()
