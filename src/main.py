# src/main.py
"""
main.py

Command-line entry point for running MBTI Game of Chicken tournaments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tournament import write_tournament_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MBTI Game of Chicken tournament(s).")
    parser.add_argument("--model", default="llama3:8b", help="Model name (e.g., llama3:8b for Ollama).")
    parser.add_argument("--method", default="prompt", choices=["neutral", "prompt", "lora"], help="Conditioning method.")
    parser.add_argument("--tournaments", type=int, default=1, help="Number of tournaments to run.")
    parser.add_argument("--seed", type=int, default=42, help="Master seed for reproducibility.")
    parser.add_argument("--out", default="data/results/tournaments.jsonl", help="Output JSONL path.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature.")
    parser.add_argument("--max_tokens", type=int, default=80, help="Max tokens to generate.")
    parser.add_argument(
        "--adapter_template",
        default=None,
        help="Optional template for LoRA adapter model names. Example: 'lora-{MBTI}' or 'lora_{mbti}'.",
    )

    args = parser.parse_args()
    out_path = Path(args.out)

    write_tournament_jsonl(
        out_path=out_path,
        master_seed=args.seed,
        method=args.method,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n_tournaments=args.tournaments,
        adapter_template=args.adapter_template,
    )


if __name__ == "__main__":
    main()