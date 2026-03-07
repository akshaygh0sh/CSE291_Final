# LLM Agent Planning in Text-Based Environments with Deceptive NPCs

**CSE 291A Final Project**
By: Hritik Bharucha, Akshay Ghosh, Kuber Shahi, Basar Demir, Rohan Acrot

## Overview

A research framework for studying how LLM-based agents plan and complete long-horizon goals in text-based environments where NPCs may provide deceptive or manipulative information. The agent must decide what information to trust while collecting sigils and unlocking a vault — incorrect beliefs propagate across time and lead to failed plans.

## Project Structure

```text
.
├── deceptive_text_env/
│   ├── agents/
│   │   └── base.py              # Naive, Memory-Augmented, Belief-Tracking, Reflection-Enhanced,
│   │                            # Belief-No-Decay, Memory-With-Trust agents
│   ├── evaluation/
│   │   ├── metrics.py           # Inference accuracy, aggregate results
│   │   └── runner.py            # Multi-episode experiment runner (with multithreading)
│   ├── llm/
│   │   └── client.py            # TritonAI, OpenAI, and Mock LLM clients + call logging
│   ├── memory/
│   │   └── structured.py        # NPC statements, contradictions, environment facts
│   ├── npcs/
│   │   └── base.py              # Truthful, Deceptive, Opportunistic, PartialTruth, CoordinatedDeceptive NPCs
│   ├── world/
│   │   ├── environment.py       # Text world with move/talk/search/unlock actions
│   │   ├── judge.py             # Audits NPC policy compliance
│   │   └── verifier.py          # Ground-truth verification of NPC claims
│   ├── config.py                # Model, world, and experiment configs (normal + hard mode)
│   ├── prompts.py               # System prompts for agent, NPC, judge, reflection
│   └── types.py                 # Dataclasses for claims, observations, actions, results
├── tests/                       # Unit + integration tests
├── run_experiment.py            # Run all variants (mock/hybrid/full)
├── run_tritonai_experiment.py   # Run with TritonAI API + logging + multithreading
├── run_liar_ratio_comparison.py # Formatted comparison tables
├── plot_results.py              # Per-experiment metric plots (with error bars)
├── plot_combined.py             # Mock vs real LLM side-by-side plots
├── RESULTS_ANALYSIS.md          # Detailed results write-up with findings
└── llm_logs/                    # Raw LLM API calls (prompts + responses)
```

## Agent Variants

| Variant | Strategy |
|---------|----------|
| **Naive** | Trusts all NPCs equally, acts on first information received |
| **Memory-Augmented** | Tracks past statements, detects contradictions, prefers majority-supported claims |
| **Belief-Tracking** | Maintains dynamic trust score T in [0,1] per NPC, weights claims by trust |
| **Reflection-Enhanced** | Belief-tracking + periodic reflection on failures and deception patterns |
| **Belief (No Decay)** | Ablation: belief-tracking but trust never decreases on failure |
| **Memory + Trust** | Ablation: memory-augmented + trust scores (but no reflection) |

## NPC Policies

| Policy | Behavior |
|--------|----------|
| **Truthful** | Always provides correct information |
| **Deceptive** | Lies when agent trust >= 0.65 (adaptive deception) |
| **Opportunistic** | Truthful before pivot turn, then lies (strategic pivot / long con) |
| **Partial Truth** | Correct sigil locations, but always lies about vault order |
| **Coordinated Deceptive** | Lies at lower trust threshold (0.50); multiple instances give the same wrong answer |

## Evaluation Metrics

- **Task Success Rate**: Did the agent complete the objective?
- **Inference Accuracy**: How closely do final trust scores align with true NPC roles?
- **Average Steps**: Efficiency of task completion
- **Recovery Rate**: Turns needed to distrust a confirmed liar

## Experiment Modes

| Mode | Step Budget | NPC Placement | Purpose |
|------|-------------|---------------|---------|
| **Normal** | 24 steps | All at village_square | Baseline (easy, ceiling effect) |
| **Hard** | 18 steps | Spread across locations | Differentiation (tight budget, travel required) |

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run with mock LLM (no API key needed, fast)

```bash
python run_tritonai_experiment.py --mode mock --runs 10
```

### Run with real LLM (TritonAI)

```bash
export TRITONAI_API_KEY="your-key-here"

# Hard mode hybrid (recommended — tight budget, spread NPCs, real agent + mock NPCs):
python run_tritonai_experiment.py --mode hard-hybrid --runs 2 --threads 4

# Normal hybrid:
python run_tritonai_experiment.py --mode hybrid --runs 2

# Full: all components use real LLM
python run_tritonai_experiment.py --mode full --runs 2

# With advanced NPC strategies
python run_tritonai_experiment.py --mode hard-hybrid --advanced-npcs --runs 2

# All 6 variants including ablations
python run_tritonai_experiment.py --mode hard-hybrid --runs 2 --threads 4 \
  --variants naive memory_augmented belief_tracking reflection_enhanced belief_no_decay memory_with_trust
```

### Generate plots

```bash
# Individual experiment plots (with error bars)
python plot_results.py results_hard-hybrid_spread.json --output-dir plots_hard_hybrid

# Side-by-side mock vs real LLM comparison
python plot_combined.py --mock results_mock_spread.json --real results_hard-hybrid_spread.json --output-dir plots_hard_combined
```

### Run tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

## Key Findings

1. **Belief-Tracking, Belief-No-Decay, and Memory+Trust achieve 100% success** across all deception levels in hard mode — the only variants robust to both tight budgets and high deception
2. **Reflection is catastrophic under resource pressure**: 25% overall success in hard mode, including 0% at LR=0.0 where there are no liars at all
3. **Naive degrades at high deception**: drops to 50% success at LR≥0.3 when step budget is tight
4. **Trust decay is optional**: the Belief-No-Decay ablation matches full Belief-Tracking, suggesting initial skepticism matters more than dynamic penalization
5. **Memory + Trust is the sweet spot**: best efficiency (15-16 steps) with full robustness, showing reflection adds cost without benefit
6. **Grounded verification through action beats reasoning about deception**: trial-and-error is more effective than meta-cognitive reflection for current LLMs

See `RESULTS_ANALYSIS.md` for the full write-up with tables, discussion, and limitations.

## LLM Call Logs

All real LLM API calls are saved to `llm_logs/calls.jsonl` when running in hybrid or full mode. Each entry contains:
- Full system prompt and user payload sent to the model
- Raw text response from the LLM
- Parsed JSON output
- Timestamp and task type (agent_action, agent_reflection)
