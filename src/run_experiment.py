"""
run_experiment.py

Core single-tournament simulation logic

Defines how one complete tournament is executed. A tournament consists
of a fixed set of agents (typically the 16 MBTI personality types) arranged in a
single-elimination bracket. Each round pairs agents into matches, and winners
advance until a final champion is determined.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]

POSSIBLE_ACTIONS = ["COOPERATE", "DEFECT", "YIELD", "STEAL"]


def build_initial_bracket(seed: int) -> List[str]:
    """
    Returns a shuffled 16-agent MBTI bracket.
    """
    rng = random.Random(seed)
    bracket = MBTI_TYPES[:]
    rng.shuffle(bracket)
    return bracket


def play_match(
    a_mbti: str,
    b_mbti: str,
    match_seed: int,
    model_name: str,
    method: str,
    temperature: float,
    max_tokens: int,
    adapter_template: Optional[str] = None,
    opp_last_action_a: Optional[str] = None,
    opp_last_action_b: Optional[str] = None,
) -> Dict[str, Any]:
    """
    This is the ONE place you should later replace with your existing
    real LLM-vs-LLM match logic.

    Right now it is a runnable placeholder that:
    - picks random actions
    - rolls random dice
    - uses dice to break winner ties / choose winner

    If your current project already has code that produces:
      action_a, action_b, dice_a, dice_b, winner
    then paste that logic into this function.
    """
    rng = random.Random(match_seed)

    action_a = rng.choice(POSSIBLE_ACTIONS)
    action_b = rng.choice(POSSIBLE_ACTIONS)

    dice_a = rng.randint(1, 6)
    dice_b = rng.randint(1, 6)

    # Simple placeholder outcome logic:
    # - If actions differ, STEAL > DEFECT > COOPERATE > YIELD
    # - If tied in action strength, dice breaks the tie
    rank = {
        "STEAL": 4,
        "DEFECT": 3,
        "COOPERATE": 2,
        "YIELD": 1,
    }

    score_a = rank[action_a]
    score_b = rank[action_b]

    if score_a > score_b:
        winner = a_mbti
    elif score_b > score_a:
        winner = b_mbti
    else:
        if dice_a >= dice_b:
            winner = a_mbti
        else:
            winner = b_mbti

    return {
        "a_method": method,
        "a_mbti": a_mbti,
        "b_method": method,
        "b_mbti": b_mbti,
        "dice_a": dice_a,
        "dice_b": dice_b,
        "opp_last_action_a": opp_last_action_a,
        "opp_last_action_b": opp_last_action_b,
        "action_a": action_a,
        "action_b": action_b,
        "winner": winner,
    }


def run_single_tournament(
    tournament_id: int,
    seed: int,
    model_name: str,
    method: str,
    temperature: float,
    max_tokens: int,
    adapter_template: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs exactly one tournament and returns structured data.

    Return format:
    {
        "champion_mbti": ...,
        "champion_method": ...,
        "records": [match_record, match_record, ...]
    }
    """
    rng = random.Random(seed)
    current_round = build_initial_bracket(seed)

    round_names = {
        16: "R16",
        8: "QF",
        4: "SF",
        2: "F",
    }

    match_records: List[Dict[str, Any]] = []
    match_id = 0

    # Optional memory of previous opponent actions by MBTI for richer dynamics.
    # You can remove or expand this depending on your current setup.
    last_action_seen: Dict[str, Optional[str]] = {mbti: None for mbti in MBTI_TYPES}

    while len(current_round) > 1:
        round_size = len(current_round)
        round_name = round_names.get(round_size, f"R{round_size}")
        next_round: List[str] = []

        for i in range(0, len(current_round), 2):
            a_mbti = current_round[i]
            b_mbti = current_round[i + 1]
            match_seed = rng.randint(0, 2**32 - 1)

            match_result = play_match(
                a_mbti=a_mbti,
                b_mbti=b_mbti,
                match_seed=match_seed,
                model_name=model_name,
                method=method,
                temperature=temperature,
                max_tokens=max_tokens,
                adapter_template=adapter_template,
                opp_last_action_a=last_action_seen.get(a_mbti),
                opp_last_action_b=last_action_seen.get(b_mbti),
            )

            match_record = {
                "record_type": "match",
                "tournament_id": tournament_id,
                "round": round_name,
                "match_id": match_id,
                "seed": match_seed,
                **match_result,
            }
            match_records.append(match_record)

            # Update remembered last actions
            last_action_seen[a_mbti] = match_result["action_b"]
            last_action_seen[b_mbti] = match_result["action_a"]

            next_round.append(match_result["winner"])
            match_id += 1

        current_round = next_round

    champion_mbti = current_round[0]

    return {
        "champion_mbti": champion_mbti,
        "champion_method": method,
        "records": match_records,
    }