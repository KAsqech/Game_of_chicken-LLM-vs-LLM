# Game of Chicken: MBTI Personality and Strategic Decision-Making in LLM Agents

**CS 4701 — Practicum in Artificial Intelligence**
**Team:** Kamie Aran (ka447), Alif Abdullah (aa2298), Ginger McCoy (gmm225)

## Overview

This project investigates whether fine-tuning LLM agents on MBTI personality-typed text data produces more behaviorally distinct and strategically consistent agents than persona prompting alone. We use Llama 3 8B as a fixed base model and train separate LoRA/QLoRA adapters for each MBTI type using [Unsloth](https://github.com/unslothai/unsloth) for fast, memory-efficient fine-tuning, then evaluate agents through controlled, multi-run computational experiments across game-theoretic settings.

### Research Question

**Does fine-tuning produce more consistent and distinct agent behavior than prompting alone, and does that translate to measurably different strategic outcomes?**

We wanted to gobeyond the observation that different prompts produce different outputs. Fine-tuned agents internalize personality traits at the weight level rather than relying on instruction-following, which may produce deeper behavioral consistency, or it may not.

## AI Components

- **LoRA/QLoRA fine-tuning via Unsloth** — Training separate lightweight adapters for each MBTI type on personality-consistent text data (Kaggle MBTI dataset, MBTI subreddit posts). Unsloth provides 2-5x faster training and ~50% less memory usage compared to vanilla Hugging Face PEFT, making it feasible to train on Google Colab's free T4 tier.
- **Three-group controlled comparison** — Running identical experiments across fine-tuned agents, prompt-only agents, and neutral baseline agents to isolate the effect of personality conditioning method.
- **Behavioral evaluation pipeline** — Quantitative metrics to assess role-play fidelity, strategic differences, and payoff-level outcomes across multiple experimental runs.

## Games

| Game | Players Choose | Tension |
|------|---------------|---------|
| **Chicken** (primary) | Escalate or Yield | Brinksmanship — mutual escalation is the worst outcome, but unilateral escalation wins |
| **Prisoner's Dilemma** (stretch) | Cooperate or Defect | Self-interest — mutual defection is suboptimal but individually rational |

We focus on Chicken as the primary game to keep the fine-tuning workload manageable. Prisoner's Dilemma is a stretch goal for cross-game comparison.

## Experimental Design

### Three Agent Groups

Rather than a single-elimination tournament (which produces limited data), we run a controlled, multi-run experiment comparing three agent groups:

| Group | Model | Personality Source | Purpose |
|-------|-------|--------------------|---------|
| **Neutral baseline** | Llama 3 8B | None — no persona prompt, no adapter | Establishes default model behavior with no personality conditioning |
| **Prompt-only** | Llama 3 8B | MBTI persona in system prompt | Tests personality injected through instructions |
| **Fine-tuned** | Llama 3 8B + LoRA adapter | Personality baked into weights, no persona prompt | Tests personality internalized through training |

All groups use identical model parameters (temperature, token limits, etc.) and see the same game states. This three-way comparison isolates the effect of the conditioning method: any behavioral differences between prompt-only and fine-tuned agents are attributable to how the personality was introduced, not what personality was assigned.

### Experiment Structure

- **Repeated pairings** across all matchup combinations (within and across groups) over multiple runs to gather statistically meaningful data
- **Structured behavioral logging** of every decision, reasoning trace, payoff outcome, and game state (JSON/SQLite)
- **Randomized conditions** with controlled seeds for reproducibility
- Each MBTI type appears in all applicable groups, enabling direct type-level comparison across conditioning methods

### Fine-Tuning Pipeline

1. **Data collection** — Gather personality-typed text from the Kaggle MBTI dataset (~8,600 users with labeled forum posts) and/or MBTI subreddit posts. Clean and format as instruction-tuning data.
2. **LoRA adapter training** — For each MBTI type (starting with 4 representative types — ENTJ, ISFP, ENTP, ISFJ — expanding to 16 if time allows), fine-tune a separate LoRA adapter on Llama 3 8B using Unsloth + QLoRA on Google Colab. Unsloth handles 4-bit quantization, gradient checkpointing, and optimized training loops out of the box.
3. **Validation** — Verify that fine-tuned adapters produce personality-consistent text outside of the game context before running experiments.

MBTI is used as a structured personality abstraction rather than a validated psychological model.

## Evaluation

### Role-Play Fidelity
- **Behavioral consistency scoring** — Do agents act in line with their MBTI type's expected tendencies? (e.g., ENTJ escalates more, ISFP cooperates more)
- **Reasoning trace analysis** — Keyword/sentiment analysis of chain-of-thought outputs to assess personality alignment
- **Fine-tuned vs. prompted comparison** — Are fine-tuned agents more consistent in their personality-typed behavior than prompted agents?
- **Baseline controls** — Neutral agents and shuffled persona labels validate that observed differences stem from the MBTI conditioning

### Strategic and Payoff-Level Analysis
- Escalation/yielding frequencies and payoff distributions across all agent groups
- Statistical comparison of action distributions between fine-tuned, prompted, and neutral agents (pairwise tests)
- Analysis along MBTI dimensions (E/I, S/N, T/F, J/P) to identify which trait dimensions most strongly influence strategic behavior
- Edge case analysis: aggressive vs. aggressive matchups, passive exploitation patterns, cross-group matchups

### Prompt Optimization (Secondary)
- If initial fine-tuned adapters underperform, iterate on training data or hyperparameters
- Compare effect sizes: how much does prompt refinement improve prompt-only agents vs. how much does fine-tuning improve over prompting entirely?

## Repository Structure

```
src/
  agent.py              # LLM agent with observe-think-act loop (supports both adapter and prompt modes)
  chicken.py            # Game of Chicken engine
  experiment.py         # Controlled multi-run experiment orchestration
  evaluation.py         # Role-play fidelity metrics and payoff analysis
  analysis.py           # Statistical analysis and visualization
  main.py               # Entry point

fine_tuning/
  prepare_data.py       # Data cleaning and formatting for LoRA training
  train_lora.py         # Unsloth fine-tuning script
  validate_adapter.py   # Personality consistency validation for adapters

config/
  mbti_profiles.yaml    # MBTI trait definitions and expected behaviors
  payoff_matrix.yaml    # Payoff matrices
  model_config.yaml     # Base model and LoRA hyperparameters

prompts/
  mbti_prompts/         # 16 persona prompt templates (for prompt-only group)

data/
  training/             # MBTI-typed text data for fine-tuning
  results/              # Experiment logs (JSON/SQLite)
  analysis/             # Derived statistics and visualizations
```

## Setup

```bash
# Clone the repository
git clone https://github.com/KAsqech/Game_of_chicken-LLM-vs-LLM.git
cd Game_of_chicken-LLM-vs-LLM

# Install dependencies
pip install -r requirements.txt

# Key dependencies:
#   torch, unsloth, transformers, peft, bitsandbytes
#   ollama or vllm (for local inference)
#   pandas, numpy, scipy, matplotlib

# Pull the base model for local inference
ollama pull llama3:8b
```

### Fine-Tuning on Google Colab

Fine-tuning is done on Google Colab (free T4 GPU is sufficient with Unsloth):

```python
# Cell 1: Install Unsloth
!pip install unsloth

# Cell 2: Clone repo and prep data
!git clone https://github.com/KAsqech/Game_of_chicken-LLM-vs-LLM.git
%cd Game_of_chicken-LLM-vs-LLM

# Cell 3: Train an adapter
!python fine_tuning/train_lora.py --mbti_type ENTJ --epochs 3

# Cell 4: Save adapter to Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r outputs/lora_ENTJ /content/drive/MyDrive/adapters/
```

Adapters are small (~10–50MB) and can be committed to the repo or shared via Google Drive.