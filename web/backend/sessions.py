"""In-memory game session store.

game_id (UUID string) → session dict:
  env          CheckersEnv instance
  mcts         MCTSSearch instance
  human_color  BLUE or RED constant
  ai_color     RED or BLUE constant
  is_wdl       bool — whether the loaded network is WDLAlphaZeroNetwork
  model_label  str  — display name for the loaded model
"""

from typing import Any

games: dict[str, dict[str, Any]] = {}
