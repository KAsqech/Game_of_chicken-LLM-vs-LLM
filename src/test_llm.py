from src.llm import LLMAgent

from prompts.mbti_prompts import enfp_prompt, intj_prompt
from prompts.game_prompts import chicken_game_prompt

enfp_agent = LLMAgent(enfp_prompt, chicken_game_prompt)
print(enfp_agent.get_action()["chosen_action"].content)