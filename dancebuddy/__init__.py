"""DanceBuddy: an agent that makes a non-humanoid character dance along with a human dancer.

  perception/  what it observes          knowledge/  what it knows about the character
  planner/     the LLM and its loop      actions/    what it can decide (plans, its own moves)
  executor/    carries the actions out   critics/    how it checks the result
  tools.py     the tools the LLM calls   llm.py      model access
"""
