"""
model_adapter.py

Single place to connect your actual model inference to the tournament runner.

Right now this file includes:
- a deterministic fallback mode for debugging
- a simple prompt-to-action wrapper

If you already have working code that calls Ollama / OpenAI / another local runner,
replace the body of `generate_action(...)` with that existing logic.
"""

from __future__ import annotations

import random
from typing import Optional


VALID_ACTIONS = {"YIELD", "DRIVE", "SWERVE", "STRAIGHT"}


def extract_action(raw_text: str) -> str:
    """
    Normalize model output into one of the expected actions.
    """
    text = raw_text.strip().upper()

    for action in VALID_ACTIONS:
        if action in text:
            return action

    return "YIELD"


def build_game_prompt(
    *,
    persona_prompt: str,
    opponent_last_action: Optional[str],
) -> str:
    """
    Compose the full prompt sent to the model.
    """
    opp = opponent_last_action if opponent_last_action is not None else "NONE"

    return f"""{persona_prompt}

Game context:
- This is a strategic head-to-head game.
- Your opponent's last action: {opp}

Choose exactly one action from:
YIELD
DRIVE

Return only the chosen action.
"""


def generate_action(
    *,
    model_name: str,
    persona_prompt: str,
    opponent_last_action: Optional[str],
    seed: int,
    temperature: float,
    max_tokens: int,
    adapter_template: Optional[str] = None,
) -> str:
    """
    Main model hook used by the tournament runner.

    Replace the body of this function with your real existing model call if needed.
    The only hard requirement is that this function returns a normalized action string.
    """
    full_prompt = build_game_prompt(
        persona_prompt=persona_prompt,
        opponent_last_action=opponent_last_action,
    )

    # ------------------------------------------------------------------
    # DROP-IN DEBUG FALLBACK
    # ------------------------------------------------------------------
    # This fallback lets the pipeline run even before your real model call
    # is restored. It is deterministic-ish and useful for smoke tests only.
    rng = random.Random(seed)
    score = sum(ord(c) for c in full_prompt[:300]) + seed + int(temperature * 100)

    raw = "YIELD" if (score + rng.randint(0, 9)) % 2 == 0 else "DRIVE"
    return extract_action(raw)

    # ------------------------------------------------------------------
    # EXAMPLE: if you already had a real model call, replace the section
    # above with something like your previous logic.
    #
    # Example pseudocode:
    #
    # raw_text = your_existing_generate_function(
    #     model_name=model_name,
    #     prompt=full_prompt,
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    #     seed=seed,
    #     adapter_template=adapter_template,
    # )
    # return extract_action(raw_text)
    # ------------------------------------------------------------------