"""
CLI entrypoint for the MBTI tournament project.

Commands:
- run-many-tournaments
- run-all-conditions

Examples:
    python src/main.py run-many-tournaments \
        --n-tournaments 10 \
        --condition true_persona \
        --output results_true.jsonl

    python src/main.py run-many-tournaments \
        --n-tournaments 10 \
        --condition neutral \
        --output results_neutral.jsonl

    python src/main.py run-many-tournaments \
        --n-tournaments 10 \
        --condition shuffled_persona \
        --output results_shuffled.jsonl

    python src/main.py run-all-conditions \
        --n-tournaments 10 \
        --output results_all.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mbti_conditions import VALID_CONDITIONS
from run_many_tournaments import run_all_conditions, run_many_tournaments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MBTI tournament experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--n-tournaments", type=int, default=1)
    common.add_argument("--output", type=Path, required=True)
    common.add_argument("--model-name", type=str, default="llama3:8b")
    common.add_argument("--temperature", type=float, default=0.7)
    common.add_argument("--max-tokens", type=int, default=80)
    common.add_argument("--master-seed", type=int, default=42)
    common.add_argument(
        "--prompts-dir",
        type=Path,
        default=Path("prompts"),
        help="Directory containing MBTI persona prompt files and neutral.txt",
    )
    common.add_argument(
        "--adapter-template",
        type=str,
        default=None,
        help="Optional adapter template name/path used by your model wrapper.",
    )

    p_many = subparsers.add_parser(
        "run-many-tournaments",
        parents=[common],
        help="Run one condition across many tournaments.",
    )
    p_many.add_argument(
        "--condition",
        type=str,
        default="true_persona",
        choices=sorted(VALID_CONDITIONS),
        help="Experimental condition to run.",
    )

    p_all = subparsers.add_parser(
        "run-all-conditions",
        parents=[common],
        help="Run true_persona, neutral, and shuffled_persona into one JSONL file.",
    )
    p_all.add_argument(
        "--conditions",
        nargs="+",
        default=["true_persona", "neutral", "shuffled_persona"],
        choices=sorted(VALID_CONDITIONS),
        help="List of conditions to run.",
    )

    return parser


def print_counter(counter) -> None:
    print("Champion counts:")
    for key, value in sorted(counter.items(), key=lambda x: str(x[0])):
        print(f"  {key}: {value}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-many-tournaments":
        counts = run_many_tournaments(
            n_tournaments=args.n_tournaments,
            output_path=args.output,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            master_seed=args.master_seed,
            prompts_dir=args.prompts_dir,
            condition=args.condition,
            adapter_template=args.adapter_template,
        )
        print(f"Wrote results to {args.output}")
        print_counter(counts)
        return

    if args.command == "run-all-conditions":
        counts = run_all_conditions(
            n_tournaments=args.n_tournaments,
            output_path=args.output,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            master_seed=args.master_seed,
            prompts_dir=args.prompts_dir,
            conditions=args.conditions,
            adapter_template=args.adapter_template,
        )
        print(f"Wrote results to {args.output}")
        print_counter(counts)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()