"""Quick sanity check -- run with: .venv/Scripts/python web/backend/test_imports.py"""
import os, sys
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from checkers_game.constants import BLUE, RED, ALL_ACTIONS, encode_action
from rl.envs import CheckersEnv
from rl.networks import AlphaZeroNetwork, WDLAlphaZeroNetwork
from rl.algorithms.mcts.mcts_search import MCTSSearch
from web.backend.models_config import MODELS

env = CheckersEnv()
env.reset()
mask = env.get_action_mask()
legal = [ALL_ACTIONS[i] for i in range(len(mask)) if mask[i] == 1.0]
abs_state = env.get_absolute_board_state()

print("checkers_env    OK")
print(f"  Legal moves at start: {len(legal)}  (expected 7)")
print(f"  Board shape:          {abs_state.shape}  (expected (4, 8, 8))")
print(f"  BLUE={BLUE}, RED={RED}")
print("AlphaZeroNetwork  OK")
print("WDLAlphaZeroNetwork  OK")
print("MCTSSearch  OK")
print(f"models_config   OK — {[m['id'] for m in MODELS]}")
print("\nAll imports succeeded.")
