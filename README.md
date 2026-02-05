# Game_of_chicken-LLM-vs-LLM
We are attempting to build a tournament for the game of Chicken between 8 pairs of LLMs where each LLM has a different MBTI type in order to determine which traits correlate with victory. @ginger testing first committ!

# Methodology
-LLM-based agents with fixed MBTI personality prompts
-One-shot Game of Chicken interactions
-Fixed payoff matrix
-Randomized single-elimination tournament
-Outcomes logged and analyzed

MBTI is used as a structured personality abstraction rather than a validated psychological model.

# Repository structure
```text
src/
  agent.py
  chicken.py
  tournament.py
  analysis.py
  main.py

config/
  mbti_profiles.yaml
  payoff_nmatrix.yaml

prompts/
  mbti_prompts/

data/
  results.csv
