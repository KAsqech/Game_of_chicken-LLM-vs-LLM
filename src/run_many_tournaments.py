"""
run_many_tournaments.py

Batch execution and aggregation layer for MBTI LLM tournaments.

This module runs multiple independent tournaments using the core logic
defined in run_experiment.py and aggregates the results into a single
JSONL output file. It is designed to support experimental analysis of
emergent behavior across repeated simulations.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional

from run_experiment import run_single_tournament


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run many MBTI tournaments")
    parser.add_argument("--model", default="llama3:8b")
    parser.add_argument("--method", default="prompt")
    parser.add_argument("--n-tournaments", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapter-template", default=None)
    parser.add_argument("--output", default="results/many_tournaments.jsonl")
    return parser.parse_args()


def write_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_many_tournaments(
    model_name: str,
    method: str,
    n_tournaments: int,
    temperature: float,
    max_tokens: int,
    seed: int,
    adapter_template: Optional[str],
    output_path: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")

    rng = random.Random(seed)
    champion_counts = Counter()

    meta = {
        "record_type": "meta",
        "master_seed": seed,
        "method": method,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n_tournaments": n_tournaments,
        "adapter_template": adapter_template,
        "num_agents": 16,
        "dice_mode": "tie_break",
    }
    write_jsonl(output, meta)

    for tournament_id in range(n_tournaments):
        tournament_seed = rng.randint(0, 2**32 - 1)

        result = run_single_tournament(
            tournament_id=tournament_id,
            seed=tournament_seed,
            model_name=model_name,
            method=method,
            temperature=temperature,
            max_tokens=max_tokens,
            adapter_template=adapter_template,
        )

        champion_mbti = result["champion_mbti"]
        champion_method = result["champion_method"]
        champion_counts[champion_mbti] += 1

        write_jsonl(output, {
            "record_type": "champion",
            "tournament_id": tournament_id,
            "champion_mbti": champion_mbti,
            "champion_method": champion_method,
        })

        for rec in result["records"]:
            write_jsonl(output, rec)

    summary = {
        "record_type": "summary",
        "n_tournaments": n_tournaments,
        "champion_counts": dict(champion_counts),
        "champion_frequencies": {
            mbti: count / n_tournaments for mbti, count in champion_counts.items()
        },
    }
    write_jsonl(output, summary)

    print(f"Wrote results to {output}")
    print("Champion counts:")
    for mbti, count in champion_counts.most_common():
        print(f"  {mbti}: {count}")


def main() -> None:
    args = parse_args()
    run_many_tournaments(
        model_name=args.model,
        method=args.method,
        n_tournaments=args.n_tournaments,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        adapter_template=args.adapter_template,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()