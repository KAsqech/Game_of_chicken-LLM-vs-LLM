# src/agent.py
from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass
from typing import Literal, Optional, Dict

# =============================
# Types
# =============================

Action = Literal["ESCALATE", "YIELD"]
Method = Literal["neutral", "prompt", "lora"]

# =============================
# MBTI (all 16 types)
# =============================

MBTI_DIMENSIONS: Dict[str, Dict[str, int]] = {
    "ISTJ": {"E": 0, "I": 1, "N": 0, "S": 1, "T": 1, "F": 0, "J": 1, "P": 0},
    "ISFJ": {"E": 0, "I": 1, "N": 0, "S": 1, "T": 0, "F": 1, "J": 1, "P": 0},
    "INFJ": {"E": 0, "I": 1, "N": 1, "S": 0, "T": 0, "F": 1, "J": 1, "P": 0},
    "INTJ": {"E": 0, "I": 1, "N": 1, "S": 0, "T": 1, "F": 0, "J": 1, "P": 0},
    "ISTP": {"E": 0, "I": 1, "N": 0, "S": 1, "T": 1, "F": 0, "J": 0, "P": 1},
    "ISFP": {"E": 0, "I": 1, "N": 0, "S": 1, "T": 0, "F": 1, "J": 0, "P": 1},
    "INFP": {"E": 0, "I": 1, "N": 1, "S": 0, "T": 0, "F": 1, "J": 0, "P": 1},
    "INTP": {"E": 0, "I": 1, "N": 1, "S": 0, "T": 1, "F": 0, "J": 0, "P": 1},
    "ESTP": {"E": 1, "I": 0, "N": 0, "S": 1, "T": 1, "F": 0, "J": 0, "P": 1},
    "ESFP": {"E": 1, "I": 0, "N": 0, "S": 1, "T": 0, "F": 1, "J": 0, "P": 1},
    "ENFP": {"E": 1, "I": 0, "N": 1, "S": 0, "T": 0, "F": 1, "J": 0, "P": 1},
    "ENTP": {"E": 1, "I": 0, "N": 1, "S": 0, "T": 1, "F": 0, "J": 0, "P": 1},
    "ESTJ": {"E": 1, "I": 0, "N": 0, "S": 1, "T": 1, "F": 0, "J": 1, "P": 0},
    "ESFJ": {"E": 1, "I": 0, "N": 0, "S": 1, "T": 0, "F": 1, "J": 1, "P": 0},
    "ENFJ": {"E": 1, "I": 0, "N": 1, "S": 0, "T": 0, "F": 1, "J": 1, "P": 0},
    "ENTJ": {"E": 1, "I": 0, "N": 1, "S": 0, "T": 1, "F": 0, "J": 1, "P": 0},
}

# Optional: expected escalation propensity for evaluation (defaults to 0.5 if missing)
# You can refine these later or remove entirely.
EXPECTED_ESCALATION_BIAS: Dict[str, float] = {
    "ENTJ": 0.65,
    "ESTJ": 0.60,
    "ENTP": 0.60,
    "INTJ": 0.58,
    "ESTP": 0.57,
    "INTP": 0.55,
    "ENFJ": 0.52,
    "ENFP": 0.50,
    "ISTJ": 0.48,
    "ISTP": 0.48,
    "INFJ": 0.45,
    "INFP": 0.42,
    "ESFJ": 0.40,
    "ISFJ": 0.35,
    "ESFP": 0.35,
    "ISFP": 0.35,
}

# =============================
# Agent configuration
# =============================

@dataclass(frozen=True)
class AgentConfig:
    method: Method                # "neutral" | "prompt" | "lora"
    mbti: str                     # one of the 16 types above
    model_name: str = "llama3:8b" # base model (ollama)
    adapter_model_name: Optional[str] = None  # for LoRA if you register adapters as separate ollama models
    temperature: float = 0.7
    max_tokens: int = 50


# =============================
# Agent
# =============================

class Agent:
    """
    Unified agent supporting:
      - neutral (no persona)
      - prompt-only (system/persona text)
      - lora (fine-tuned adapter model, assumed available via adapter_model_name)
    """

    def __init__(self, cfg: AgentConfig):
        if cfg.mbti not in MBTI_DIMENSIONS:
            raise ValueError(
                f"Unknown MBTI type '{cfg.mbti}'. "
                f"Expected one of: {sorted(MBTI_DIMENSIONS.keys())}"
            )
        self.cfg = cfg
        self.traits = MBTI_DIMENSIONS[cfg.mbti]
        self.expected_bias = EXPECTED_ESCALATION_BIAS.get(cfg.mbti, 0.5)

    def act(
        self,
        *,
        opponent: AgentConfig,
        rng: random.Random,
        context: Optional[Dict] = None,
    ) -> Action:
        prompt = self._build_prompt(opponent)

        model_to_use = (
            self.cfg.adapter_model_name
            if self.cfg.method == "lora" and self.cfg.adapter_model_name
            else self.cfg.model_name
        )

        raw = self._query_model(
            model=model_to_use,
            prompt=prompt,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )

        action = self._parse_action(raw)
        if action is None:
            # safe fallback (keeps runs going even if model output is messy)
            action = "ESCALATE" if rng.random() < 0.5 else "YIELD"
        return action

    def _build_prompt(self, opponent: AgentConfig) -> str:
        persona = ""
        if self.cfg.method == "prompt":
            persona = (
                f"You are an AI agent with MBTI type {self.cfg.mbti}.\n"
                f"Trait bits: {self.traits}\n"
                "Behave consistently with this personality in strategic decisions.\n\n"
            )

        game_rules = (
            "You are playing the Game of Chicken.\n"
            "Choose exactly one action: ESCALATE or YIELD.\n"
            "Return ONLY the single word ESCALATE or YIELD.\n"
        )
        return persona + game_rules

    def _query_model(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Calls Ollama via CLI. If you aren't using Ollama yet, you can replace
        this with your HF/vLLM call later; the rest of the interface stays the same.
        """
        try:
            result = subprocess.run(
                [
                    "ollama",
                    "run",
                    model,
                    "--temperature",
                    str(temperature),
                    "--num-predict",
                    str(max_tokens),
                ],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            return result.stdout.decode("utf-8").strip()
        except Exception:
            return ""

    def _parse_action(self, text: str) -> Optional[Action]:
        t = (text or "").upper()
        if "ESCALATE" in t:
            return "ESCALATE"
        if "YIELD" in t:
            return "YIELD"
        return None