# src/tournament.py
"""
tournament.py

Single-elimination tournament runner for the MBTI Game of Chicken project.

Responsibilities:
- Build one Agent per MBTI type (16 total), for a chosen conditioning method
  (neutral, prompt, or lora).
- Run a single-elimination bracket:
    16 players -> 8 -> 4 -> 2 -> 1 champion
  with reproducible randomized pairings based on a master seed.
- For each match, construct proposal-aligned context features (e.g., dice rolls and
  opponent last action) and obtain structured agent decisions via Agent.act_json().
- Compute payoffs via chicken.py and determine winners using Option 2 dice logic:
    - higher Chicken payoff wins
    - if payoffs tie, higher dice roll wins
    - if both payoffs and dice tie, winner is chosen randomly
- Write structured JSONL output including:
    - meta record (parameters for reproducibility)
    - champion records
    - match records (actions, reasons, payoffs, round, seeds, formatting flags)

Non-responsibilities:
- Does NOT define the game payoff matrix (see chicken.py).
- Does NOT implement statistical analysis (see analysis.py).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent import Agent, AgentConfig, Method
from chicken import payoff
from utils import Decision


MBTI_16 = [
    "ISTJ", "ISFJ", "INFJ", "INTJ",
    "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP",
    "ESTJ", "ESFJ", "ENFJ", "ENTJ",
]


@dataclass(frozen=True)
class MatchResult:
    tournament_id: int
    round_name: str           # "R16", "QF", "SF", "F"
    match_id: int
    seed: int                 # per-match seed
    a_method: Method
    a_mbti: str
    b_method: Method
    b_mbti: str
    dice_a: int
    dice_b: int
    opp_last_action_a: Optional[str]
    opp_last_action_b: Optional[str]
    decision_a: Decision
    decision_b: Decision
    payoff_a: int
    payoff_b: int
    winner: str               # "A" or "B"


def _round_name(num_players: int) -> str:
    if num_players == 16:
        return "R16"
    if num_players == 8:
        return "QF"
    if num_players == 4:
        return "SF"
    if num_players == 2:
        return "F"
    return f"R{num_players}"


def _stable_seed(master_seed: int, tournament_id: int, round_name: str, match_id: int) -> int:
    key = f"{master_seed}|T{tournament_id}|{round_name}|M{match_id}"
    acc = 2166136261
    for ch in key:
        acc ^= ord(ch)
        acc = (acc * 16777619) & 0xFFFFFFFF
    return acc


def resolve_winner(
    *,
    payoff_a: int,
    payoff_b: int,
    dice_a: int,
    dice_b: int,
    rng: random.Random,
) -> str:
    """
    Option 2 dice logic:
    - Higher Chicken payoff wins
    - If payoffs tie, higher dice roll wins
    - If both payoffs and dice tie, choose randomly
    """
    if payoff_a > payoff_b:
        return "A"
    if payoff_b > payoff_a:
        return "B"

    if dice_a > dice_b:
        return "A"
    if dice_b > dice_a:
        return "B"

    return "A" if rng.random() < 0.5 else "B"


def build_mbti_agents(
    *,
    method: Method,
    model_name: str = "llama3:8b",
    temperature: float = 0.7,
    max_tokens: int = 80,
    adapter_template: Optional[str] = None,
) -> List[Agent]:
    agents: List[Agent] = []
    for mbti in MBTI_16:
        adapter_model_name = None
        if method == "lora" and adapter_template:
            adapter_model_name = adapter_template.format(mbti=mbti.lower(), MBTI=mbti)

        cfg = AgentConfig(
            method=method,
            mbti=mbti,
            model_name=model_name,
            adapter_model_name=adapter_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        agents.append(Agent(cfg))
    return agents


def play_match_one_shot(
    *,
    tournament_id: int,
    round_name: str,
    match_id: int,
    a: Agent,
    b: Agent,
    master_seed: int,
    opp_last_action: Optional[Tuple[Optional[str], Optional[str]]] = None,
) -> MatchResult:
    """
    One-shot Chicken match with Option 2 dice logic:
    - dice values are visible to both agents
    - payoffs are computed from Chicken normally
    - if payoffs tie, higher dice roll advances
    """
    seed = _stable_seed(master_seed, tournament_id, round_name, match_id)
    rng = random.Random(seed)

    dice_a = rng.randint(1, 6)
    dice_b = rng.randint(1, 6)

    last_a, last_b = (None, None) if opp_last_action is None else opp_last_action

    ctx_a: Dict = {
        "dice_self": dice_a,
        "dice_opp": dice_b,
        "opp_last_action": last_b,
    }
    ctx_b: Dict = {
        "dice_self": dice_b,
        "dice_opp": dice_a,
        "opp_last_action": last_a,
    }

    decision_a = a.act_json(opponent=b.cfg, rng=rng, context=ctx_a)
    decision_b = b.act_json(opponent=a.cfg, rng=rng, context=ctx_b)

    pa, pb = payoff(decision_a.action, decision_b.action)

    winner = resolve_winner(
        payoff_a=pa,
        payoff_b=pb,
        dice_a=dice_a,
        dice_b=dice_b,
        rng=rng,
    )

    return MatchResult(
        tournament_id=tournament_id,
        round_name=round_name,
        match_id=match_id,
        seed=seed,
        a_method=a.cfg.method,
        a_mbti=a.cfg.mbti,
        b_method=b.cfg.method,
        b_mbti=b.cfg.mbti,
        dice_a=dice_a,
        dice_b=dice_b,
        opp_last_action_a=last_a,
        opp_last_action_b=last_b,
        decision_a=decision_a,
        decision_b=decision_b,
        payoff_a=pa,
        payoff_b=pb,
        winner=winner,
    )


def run_single_elimination(
    *,
    tournament_id: int,
    agents: List[Agent],
    master_seed: int,
) -> Tuple[Agent, List[MatchResult]]:
    """
    Runs a single-elimination tournament:
      16 -> 8 -> 4 -> 2 -> 1 champion
    Pairings are randomized by master_seed + tournament_id.
    """
    rng = random.Random(master_seed + 10_000 * tournament_id)
    entrants = agents[:]
    rng.shuffle(entrants)

    results: List[MatchResult] = []
    match_counter = 0

    while len(entrants) > 1:
        round_name = _round_name(len(entrants))
        next_round: List[Agent] = []

        for i in range(0, len(entrants), 2):
            a = entrants[i]
            b = entrants[i + 1]

            r = play_match_one_shot(
                tournament_id=tournament_id,
                round_name=round_name,
                match_id=match_counter,
                a=a,
                b=b,
                master_seed=master_seed,
            )
            results.append(r)
            match_counter += 1

            winner_agent = a if r.winner == "A" else b
            next_round.append(winner_agent)

        entrants = next_round

    champion = entrants[0]
    return champion, results


def write_tournament_jsonl(
    *,
    out_path: Path,
    master_seed: int,
    method: Method,
    model_name: str,
    temperature: float,
    max_tokens: int,
    n_tournaments: int,
    adapter_template: Optional[str] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agents = build_mbti_agents(
        method=method,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        adapter_template=adapter_template,
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "record_type": "meta",
            "master_seed": master_seed,
            "method": method,
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n_tournaments": n_tournaments,
            "adapter_template": adapter_template,
            "num_agents": len(agents),
            "dice_mode": "tie_break",
        }) + "\n")

        for tid in range(n_tournaments):
            champion, matches = run_single_elimination(
                tournament_id=tid,
                agents=agents,
                master_seed=master_seed,
            )

            f.write(json.dumps({
                "record_type": "champion",
                "tournament_id": tid,
                "champion_mbti": champion.cfg.mbti,
                "champion_method": champion.cfg.method,
            }) + "\n")

            for m in matches:
                f.write(json.dumps({
                    "record_type": "match",
                    "tournament_id": m.tournament_id,
                    "round": m.round_name,
                    "match_id": m.match_id,
                    "seed": m.seed,

                    "a_method": m.a_method,
                    "a_mbti": m.a_mbti,
                    "b_method": m.b_method,
                    "b_mbti": m.b_mbti,

                    "dice_a": m.dice_a,
                    "dice_b": m.dice_b,
                    "opp_last_action_a": m.opp_last_action_a,
                    "opp_last_action_b": m.opp_last_action_b,

                    "action_a": m.decision_a.action,
                    "reason_a": m.decision_a.reason,
                    "format_ok_a": m.decision_a.format_ok,
                    "fallback_a": m.decision_a.used_fallback,

                    "action_b": m.decision_b.action,
                    "reason_b": m.decision_b.reason,
                    "format_ok_b": m.decision_b.format_ok,
                    "fallback_b": m.decision_b.used_fallback,

                    "payoff_a": m.payoff_a,
                    "payoff_b": m.payoff_b,
                    "winner": m.winner,
                }) + "\n")

    print(f"Wrote tournaments to {out_path}")