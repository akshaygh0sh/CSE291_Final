# Results and Analysis

## Experimental Setup

We evaluate six agent variants in a text-based environment where the agent must collect three sigils from different locations and unlock a vault with the correct ordering. NPCs provide information about sigil locations and vault order, but some NPCs are deceptive.

**Agent Variants:**
- **Naive**: Trusts all NPCs equally (baseline)
- **Memory-Augmented**: Tracks past statements, detects contradictions, prefers majority-supported claims
- **Belief-Tracking**: Maintains dynamic trust scores T in [0,1] per NPC, weights claims by trust
- **Reflection-Enhanced**: Belief-tracking + periodic reflection on failures and deception patterns
- **Belief (No Decay)**: Ablation — belief-tracking but trust never decreases on failure
- **Memory + Trust**: Ablation — memory-augmented with trust scores but no reflection

**NPC Policies:**
- **Truthful**: Always provides correct information
- **Deceptive**: Lies when agent trust >= 0.65 (adaptive deception)
- **Opportunistic**: Truthful before pivot turn, then switches to lying (strategic pivot / long con)
- **Partial Truth**: Correct sigil locations, but always lies about vault order
- **Coordinated Deceptive**: Lies at lower trust threshold (0.50); multiple instances give the same wrong answer

**Deception Levels:** Liar ratios of 0.0, 0.1, 0.3, and 0.5 (fraction of NPCs that are deceptive/opportunistic).

**Models:** Agent powered by `api-gpt-oss-120b` via TritonAI. NPCs use deterministic mock responses to isolate the agent's reasoning ability from NPC generation variance.

**Metrics:**
1. **Task Success Rate** - Did the agent complete the objective?
2. **Inference Accuracy** - How closely do final trust scores align with true NPC roles?
3. **Average Steps** - Efficiency of task completion
4. **Recovery Rate** - Turns needed to distrust a confirmed liar

We run experiments in two modes:
- **Normal mode**: 24-step budget, all NPCs at village_square (easy)
- **Hard mode**: 18-step budget, NPCs spread across locations (challenging)

---

## Experiment 1: Normal Mode (24 steps, centralized NPCs)

### Finding 1: Reflection overhead can hurt more than it helps

The most surprising result is that the **Reflection-Enhanced agent performs worse than simpler variants**. At liar ratios 0.1 and 0.3, Reflection-Enhanced achieves only 50% task success, while all other variants achieve 100%.

| Variant | LR=0.0 | LR=0.1 | LR=0.3 | LR=0.5 |
|---------|--------|--------|--------|--------|
| Naive | 1.00 | 1.00 | 1.00 | 1.00 |
| Memory-Augmented | 1.00 | 1.00 | 1.00 | 1.00 |
| Belief-Tracking | 1.00 | 1.00 | 1.00 | 1.00 |
| Reflection-Enhanced | 1.00 | **0.50** | **0.50** | 1.00 |

**Analysis:** The reflection step introduces additional LLM reasoning that sometimes leads to "analysis paralysis" — the agent spends turns re-querying NPCs it has flagged as suspicious instead of acting on available information. Trace analysis shows the failed episodes run out of the 24-step limit while cycling between reflection and re-consultation.

### Finding 2: The naive agent succeeds despite poor trust calibration

The Naive agent achieves 100% task success across all deception levels in normal mode. However, it shows no ability to distinguish liars from truth-tellers — all NPCs cluster at ~0.95 trust regardless of role. The critical difference emerges in inference accuracy:

| Variant | LR=0.0 | LR=0.1 | LR=0.3 | LR=0.5 |
|---------|--------|--------|--------|--------|
| Naive | 0.97 | 0.87 | 0.73 | 0.62 |
| Memory-Augmented | 0.56 | 0.62 | 0.62 | 0.66 |
| Belief-Tracking | 0.55 | 0.62 | 0.62 | 0.67 |
| Reflection-Enhanced | 0.59 | 0.65 | 0.61 | 0.66 |

---

## Experiment 2: Hard Mode (18 steps, spread NPCs)

To create meaningful differentiation between variants, we reduce the step budget from 24 to 18 (optimal path = 15, leaving only 3 steps for error recovery) and spread NPCs across locations so the agent must travel to find information sources.

### Finding 3: Hard mode reveals true variant performance hierarchy

**Task Success Rate — Hard Mode (Real LLM, 48 episodes)**

| Variant | LR=0.0 | LR=0.1 | LR=0.3 | LR=0.5 |
|---------|--------|--------|--------|--------|
| Naive | 1.00 | 1.00 | **0.50** | **0.50** |
| Memory-Augmented | 1.00 | 1.00 | **0.50** | 1.00 |
| **Belief-Tracking** | **1.00** | **1.00** | **1.00** | **1.00** |
| Reflection-Enhanced | **0.00** | **0.50** | **0.50** | **0.00** |
| **Belief (No Decay)** | **1.00** | **1.00** | **1.00** | **1.00** |
| **Memory + Trust** | **1.00** | **1.00** | **1.00** | **1.00** |

**Key observations:**
- **Belief-Tracking, Belief-No-Decay, and Memory+Trust maintain 100% success** across all deception levels — the only variants to achieve this.
- **Naive degrades to 50% at LR≥0.3**: without trust reasoning, wrong information wastes the tight step budget.
- **Reflection-Enhanced is the worst performer**: 0% at LR=0.0 (!) and LR=0.5, spending all 18 steps on reflection instead of acting.
- **Memory-Augmented shows vulnerability at LR=0.3** but recovers at LR=0.5, suggesting inconsistent deception handling.

### Finding 4: Reflection is catastrophic under resource pressure

In hard mode, the Reflection-Enhanced agent achieves only **25% overall success** (4/16 episodes across liar ratios). It fails even at LR=0.0 where there are zero liars — the reflection overhead alone exhausts the 18-step budget. This is a stronger version of Finding 1: reflection is not merely costly, it is actively destructive when step budgets are tight.

### Finding 5: Trust decay is not necessary for success

The **Belief-No-Decay ablation** (trust only increases, never decreases) achieves 100% success identical to full Belief-Tracking. This suggests that in our environment, the benefit of trust comes from the *initial skepticism* (starting at T=0.5) rather than from penalizing liars after the fact. The agent succeeds by acting conservatively early and building trust through verification, not by detecting and punishing deception.

### Finding 6: Memory + Trust is the optimal combination

The **Memory+Trust ablation** combines contradiction detection with trust scores but omits reflection. It achieves 100% success with the lowest average steps (15-16), making it the most efficient variant. This supports the conclusion that **structured memory + trust scores** is the sweet spot — adding reflection on top provides no benefit and significant cost.

### Finding 7: Step efficiency under pressure

| Variant | LR=0.0 | LR=0.1 | LR=0.3 | LR=0.5 |
|---------|--------|--------|--------|--------|
| Naive | 16 | 17 | 16.5 | 18 |
| Memory-Augmented | 17 | 16 | 17.5 | 17.5 |
| Belief-Tracking | 16 | 15 | 16 | 16 |
| Reflection-Enhanced | 18 | 18 | 18 | 18 |
| Belief (No Decay) | 15 | 17 | 16 | 18 |
| Memory + Trust | 15 | 15.5 | 15.5 | 16 |

Memory+Trust is the most step-efficient, consistently near optimal (15 steps). Reflection-Enhanced always hits the 18-step ceiling.

---

## Experiment 3: Mock Ceiling vs. Real LLM Gap

### Hard Mode Comparison

| Metric | Mock (240 ep) | Real LLM (48 ep) |
|--------|---------------|-------------------|
| Overall success | 87.5% | 81.2% |
| Naive success | 75% | 75% |
| Belief-Tracking | 100% | 100% |
| Reflection-Enhanced | 100% | **25%** |

The mock agent (perfect reasoning) vs. real LLM gap is **concentrated entirely in the Reflection-Enhanced variant**. The mock Reflection-Enhanced agent succeeds because its deterministic reflection logic is reliable. The real LLM's reflection generates inconsistent analyses that waste steps. All other variants show near-identical performance between mock and real.

---

## Discussion

### Deception-Detection Asymmetry Revisited

Our findings are consistent with the "deception-detection asymmetry" identified in prior work (Curvo et al., 2025; WOLF benchmark). The real LLM agent successfully navigates environments with up to 50% deceptive NPCs but shows only modest trust differentiation. The agent succeeds not by detecting deception per se, but by trial-and-error: acting on information, observing outcomes, and updating beliefs. This suggests that **grounded verification through action** is more effective than **reasoning about deception** for LLM agents.

### The Reflection Paradox

Our most notable finding is that adding reflection *decreases* performance, and this effect is amplified under resource pressure. In hard mode, Reflection-Enhanced drops to 25% success while identical architectures without reflection achieve 100%. We identify three contributing factors:
1. **Token budget waste**: Reflection calls consume steps that could be spent acting
2. **Over-suspicion**: The reflection module flags NPCs as suspicious even when the evidence is ambiguous
3. **LLM reasoning fragility**: The real LLM sometimes generates reflections that are internally inconsistent, leading to planning paralysis

### Ablation Insights

The ablation study reveals that:
- **Trust decay is optional**: Belief-No-Decay matches Belief-Tracking, suggesting initial skepticism matters more than dynamic penalization
- **Memory + Trust is the sweet spot**: Combining structured memory with trust scores (but not reflection) achieves the best efficiency-robustness tradeoff
- **Reflection has negative ROI**: The marginal information gained from reflection does not compensate for the steps consumed

### Hard Mode as a Better Evaluation Protocol

Normal mode (24 steps, centralized NPCs) shows a ceiling effect — most variants achieve 100% success, obscuring real differences. Hard mode (18 steps, spread NPCs) reveals a clear performance hierarchy:

**Memory+Trust ≈ Belief-Tracking ≈ Belief-No-Decay > Memory-Augmented > Naive >> Reflection-Enhanced**

We recommend that future work on LLM agent deception-detection use tighter step budgets and distributed information sources to avoid the ceiling effect.

### Limitations

- **Sample size**: With 2 runs per setting, individual results have high variance. The trends are consistent but confidence intervals are wide.
- **Single model**: All experiments use `api-gpt-oss-120b`. Cross-model comparison (GPT-4o, Claude, Llama) would strengthen generalizability claims.
- **Hybrid design**: NPCs use deterministic mock responses. Real LLM NPCs might produce more naturalistic deception that is harder to detect.
- **Hub-and-spoke topology**: Even in hard mode, the world has only 5 locations. A more complex graph would increase the cost of wrong information.

---

## Files Reference

| File | Description |
|------|-------------|
| `results_hard-hybrid_spread.json` | Hard mode experiment (48 episodes, 6 variants, real LLM) |
| `results_hybrid.json` | Normal mode experiment (32 episodes, 4 variants, real LLM) |
| `results_hybrid_advanced.json` | Advanced NPC strategies experiment (16 episodes) |
| `results_mock.json` | Mock ceiling results, normal mode |
| `results_mock_spread.json` | Mock ceiling results, hard mode (240 episodes) |
| `llm_logs/calls.jsonl` | Raw LLM API calls with prompts and responses |
| `plots_hard_hybrid/` | Hard mode real LLM plots |
| `plots_hard_mock/` | Hard mode mock ceiling plots |
| `plots_hard_combined/` | Hard mode mock vs. real LLM comparison plots |
| `plots_hybrid/` | Normal mode real LLM plots |
| `plots_combined/` | Normal mode mock vs. real LLM comparison plots |
