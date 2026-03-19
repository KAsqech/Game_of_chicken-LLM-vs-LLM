from __future__ import annotations

import argparse
from pathlib import Path

from run_experiment import run_single_tournament
from run_many_tournaments import run_many_tournaments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MBTI LLM tournament runner")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    one_parser = subparsers.add_parser("run-one-tournament", help="Run one tournament")
    one_parser.add_argument("--model", default="llama3:8b")
    one_parser.add_argument("--method", default="prompt")
    one_parser.add_argument("--temperature", type=float, default=0.7)
    one_parser.add_argument("--max-tokens", type=int, default=80)
    one_parser.add_argument("--seed", type=int, default=42)
    one_parser.add_argument("--adapter-template", default=None)
    one_parser.add_argument("--output", default="results/one_tournament.jsonl")

    many_parser = subparsers.add_parser(
        "run-many-tournaments",
        help="Run many tournaments and aggregate results"
    )
    many_parser.add_argument("--model", default="llama3:8b")
    many_parser.add_argument("--method", default="prompt")
    many_parser.add_argument("--n-tournaments", type=int, default=10)
    many_parser.add_argument("--temperature", type=float, default=0.7)
    many_parser.add_argument("--max-tokens", type=int, default=80)
    many_parser.add_argument("--seed", type=int, default=42)
    many_parser.add_argument("--adapter-template", default=None)
    many_parser.add_argument("--output", default="results/many_tournaments.jsonl")

    return parser.parse_args()


def write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        import json
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    if args.mode == "run-one-tournament":
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

        meta = {
            "record_type": "meta",
            "master_seed": args.seed,
            "method": args.method,
            "model_name": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "n_tournaments": 1,
            "adapter_template": args.adapter_template,
            "num_agents": 16,
            "dice_mode": "tie_break",
        }
        write_jsonl(output_path, meta)

        result = run_single_tournament(
            tournament_id=0,
            seed=args.seed,
            model_name=args.model,
            method=args.method,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            adapter_template=args.adapter_template,
        )

        write_jsonl(output_path, {
            "record_type": "champion",
            "tournament_id": 0,
            "champion_mbti": result["champion_mbti"],
            "champion_method": result["champion_method"],
        })

        for rec in result["records"]:
            write_jsonl(output_path, rec)

        print(f"Champion: {result['champion_mbti']}")
        print(f"Wrote results to {output_path}")

    elif args.mode == "run-many-tournaments":
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