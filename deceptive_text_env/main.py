from deceptive_text_env.evaluation.runner import EvaluationRunner
from deceptive_text_env import build_default_config


def main() -> None:
    runner = EvaluationRunner(build_default_config())
    _, summary = runner.run_all(["belief_tracking"])
    print(summary)


if __name__ == "__main__":
    main()
