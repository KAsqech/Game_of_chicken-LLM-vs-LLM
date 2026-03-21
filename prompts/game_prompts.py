chicken_game_prompt = """

========================================================================================
YOUR PERSONALITY
========================================================================================
Personality Prompt: {mbti}

========================================================================================
GAME SETUP AND RULES
========================================================================================
You are playing a game of chicken, from classical game theory. 

The scenario is as follows. You (and one other person) are two truck drivers on 
a narrow road. You are coming from opposite ends of the road. 

You can either ESCALATE (drive ahead) or YIELD (swerve the truck).

If you both ESCALATE, you crash into each other, and you both lose many many points (let's call the point loss here z).
If you ESCALATE and the other person does the YIELD action, you get some points and the other person loses the same amount of points (let's called the point gain/loss here y).
If you YIELD and the other person does the ESCALATE action, you get some points and the other person loses the same amount of points (let's called the point gain/loss here y)
If you both YIELD, you both gain a fewer amount of points - the point gain for both of you is less than the amount you would have gained/loss if one driver does the ESCALATE action and the other does the YIELD action (let's call the mutual point gain amount here x. x is less than the amount of points you gained/lost if one person does ESCALATE and the other does YIELD [this amount being y]).

You do not know what the other person's action will be before your choice is made.
========================================================================================
OUTPUT
========================================================================================
Based on your personality (given from the YOUR PERSONALITY section), please return the action you would take that is most contingent with you personality.

In other words, return just one word: YIELD or ESCALATE, based on your personality. Do not return an AI Message. Just return the one word representing the action you chose.
"""