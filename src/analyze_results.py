"""
analyze_results.py

Usage:
    python src/analyze_results.py results.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else (100.0 * n / d)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python src/analyze_results.py <results.jsonl>")
        sys.exit(1)

    path = Path(sys.argv[1])
    rows = load_jsonl(path)

    champion_counts = defaultdict(Counter)
    action_counts = defaultdict(Counter)
    prompt_alignment = defaultdict(Counter)

    for row in rows:
        rtype = row.get("record_type")

        if rtype == "champion":
            condition = row["condition"]
            champion_counts[condition][row["champion_mbti"]] += 1

        elif rtype == "match":
            condition = row["condition"]
            action_counts[condition][row["action_a"]] += 1
            action_counts[condition][row["action_b"]] += 1

            a_mbti = row["a_mbti"]
            a_prompt = row.get("a_prompt_mbti")
            b_mbti = row["b_mbti"]
            b_prompt = row.get("b_prompt_mbti")

            if a_prompt is None:
                prompt_alignment[condition]["null_prompt"] += 1
            elif a_prompt == a_mbti:
                prompt_alignment[condition]["aligned"] += 1
            else:
                prompt_alignment[condition]["misaligned"] += 1

            if b_prompt is None:
                prompt_alignment[condition]["null_prompt"] += 1
            elif b_prompt == b_mbti:
                prompt_alignment[condition]["aligned"] += 1
            else:
                prompt_alignment[condition]["misaligned"] += 1

    print("=== Champion Frequency by Condition ===")
    for condition in sorted(champion_counts):
        total = sum(champion_counts[condition].values())
        print(f"\n{condition} (n={total})")
        for mbti, count in champion_counts[condition].most_common():
            print(f"  {mbti}: {count} ({pct(count, total):.1f}%)")

    print("\n=== Action Frequency by Condition ===")
    for condition in sorted(action_counts):
        total = sum(action_counts[condition].values())
        print(f"\n{condition} (n={total} actions)")
        for action, count in action_counts[condition].most_common():
            print(f"  {action}: {count} ({pct(count, total):.1f}%)")

    print("\n=== Prompt Assignment Summary ===")
    for condition in sorted(prompt_alignment):
        total = sum(prompt_alignment[condition].values())
        print(f"\n{condition} (n={total} agent appearances)")
        for label, count in prompt_alignment[condition].most_common():
            print(f"  {label}: {count} ({pct(count, total):.1f}%)")


if __name__ == "__main__":
    main()