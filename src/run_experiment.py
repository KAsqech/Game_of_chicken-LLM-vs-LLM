from __future__ import annotations
import json
import random
from pathlib import Path
from itertools import product, combinations

from agent import Agent, AgentConfig, Method
from chicken import payoff  # uses your existing PAYOFFS/payoff()

def build_agents(
    mbti_types: list[str],
    methods: list[Method],
) -> list[Agent]:
    agents: list[Agent] = []
    for method, mbti in product(methods, mbti_types):
        cfg = AgentConfig(method=method, mbti=mbti)
        agents.append(Agent(cfg))
    return agents

def generate_matchups(agents: list[Agent]) -> list[tuple[Agent, Agent]]:
    """
    Generates unordered pairings of all agents (A,B) with A != B.
    This creates cross-type and cross-method matchups.
    If this is too many, you can sample later.
    """
    return list(combinations(agents, 2))

def run(
    *,
    mbti_types: list[str],
    methods: list[Method],
    repeats_per_matchup: int,
    seed: int,
    out_jsonl: Path,
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    agents = build_agents(mbti_types, methods)
    matchups = generate_matchups(agents)

    with out_jsonl.open("w", encoding="utf-8") as f:
        game_id = 0
        for (a, b) in matchups:
            for r in range(repeats_per_matchup):
                # Derive a per-game seed for reproducibility
                game_seed = rng.randrange(1_000_000_000)
                grng = random.Random(game_seed)

                action_a = a.act(opponent=b.cfg, rng=grng, context=None)
                action_b = b.act(opponent=a.cfg, rng=grng, context=None)

                pa, pb = payoff(action_a, action_b)

                row = {
                    "game_id": game_id,
                    "repeat": r,
                    "seed": seed,
                    "game_seed": game_seed,

                    "a_method": a.cfg.method,
                    "a_mbti": a.cfg.mbti,
                    "b_method": b.cfg.method,
                    "b_mbti": b.cfg.mbti,

                    "action_a": action_a,
                    "action_b": action_b,
                    "payoff_a": pa,
                    "payoff_b": pb,
                }
                f.write(json.dumps(row) + "\n")
                game_id += 1

if __name__ == "__main__":
    run(
        mbti_types=["ENTJ", "ISFP", "ENTP", "ISFJ"],   # start with 4
        methods=["neutral", "prompt", "lora"],
        repeats_per_matchup=50,                       # bump later
        seed=42,
        out_jsonl=Path("data/results/results.jsonl"),
    )
    print("Wrote data/results/results.jsonl")