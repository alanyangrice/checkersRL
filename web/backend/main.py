"""FastAPI backend for the CheckersRL showcase website.

Run from repo root:
    uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload

Serves:
  /api/*          — game and stats endpoints
  /               — SvelteKit static build (web/frontend/build/)
"""

import os
import sys
import csv
import uuid
from pathlib import Path
from typing import Optional

# Must be set before any pygame import (constants.py calls pygame.font.init())
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Add repo root to path so checkers_game and rl are importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from checkers_game.constants import (
    BLUE, RED, NUM_ACTIONS, ALL_ACTIONS, encode_action,
)
from rl.envs import CheckersEnv
from rl.networks import AlphaZeroNetwork, WDLAlphaZeroNetwork
from rl.algorithms.mcts.mcts_search import MCTSSearch
from rl.networks import PPOPolicyNetwork
from rl.algorithms.ppo.agent import load_policy_state_dict

from web.backend.models_config import MODELS
from web.backend.sessions import games

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="CheckersRL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model registry — loaded once at startup
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {}


def _load_az_checkpoint(path: str) -> tuple:
    """Load an AZ checkpoint, auto-detect WDL vs scalar. Returns (network, is_wdl)."""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = checkpoint["model_state_dict"]
    
    # Check if the keys have a "net." prefix. If the checkpoint doesn't use the wrapper, we might need to adjust.
    # The new rl/networks/core.py structure wraps things in self.net.
    
    # Handle legacy checkpoints without the "net." prefix 
    if any(k.startswith("value_fc2.weight") for k in sd.keys()):
        is_wdl = sd["value_fc2.weight"].shape[0] == 3
        # Add the 'net.' prefix to all keys to be compatible with the new DualHeadResNet wrapper
        new_sd = {f"net.{k}": v for k, v in sd.items()}
        sd = new_sd
    else:
        # Checkpoint is already in the new format with 'net.' prefix
        is_wdl = sd["net.value_fc2.weight"].shape[0] == 3

    NetworkClass = WDLAlphaZeroNetwork if is_wdl else AlphaZeroNetwork
    network = NetworkClass((4, 8, 8), n_actions=NUM_ACTIONS).to(DEVICE)
    network.load_state_dict(sd)
    network.eval()
    return network, is_wdl


def _load_ppo_checkpoint(path: str):
    """Load a PPO policy checkpoint. Handles both full training checkpoints
    (saved as {epoch, model_state_dict, ...}) and raw state dicts."""
    network = PPOPolicyNetwork((4, 8, 8), n_actions=NUM_ACTIONS).to(DEVICE)
    data = torch.load(path, map_location=DEVICE, weights_only=False)
    state_dict = data["model_state_dict"] if isinstance(data, dict) and "model_state_dict" in data else data
    
    # Check if keys are missing the 'net.' prefix and apply it if needed
    if any(k.startswith("policy_fc.weight") for k in state_dict.keys()):
        new_sd = {f"net.{k}": v for k, v in state_dict.items()}
        state_dict = new_sd
        
    load_policy_state_dict(network, state_dict)
    network.eval()
    return network


@app.on_event("startup")
def load_models():
    for m in MODELS:
        ckpt_path = REPO_ROOT / m["checkpoint"]
        if not ckpt_path.exists():
            print(f"WARNING: checkpoint not found, skipping model '{m['id']}': {ckpt_path}")
            continue
        try:
            if m["type"] == "ppo":
                network = _load_ppo_checkpoint(str(ckpt_path))
                MODEL_REGISTRY[m["id"]] = {**m, "network": network}
                print(f"  Loaded {m['label']} (PPO)")
            else:
                network, is_wdl = _load_az_checkpoint(str(ckpt_path))
                MODEL_REGISTRY[m["id"]] = {**m, "network": network, "is_wdl": is_wdl}
                arch = "WDL" if is_wdl else "scalar"
                print(f"  Loaded {m['label']} ({arch})")
        except Exception as e:
            print(f"WARNING: failed to load '{m['id']}': {e}")

    print(f"Model registry ready: {list(MODEL_REGISTRY.keys())}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _color_str(color) -> str:
    return "blue" if color == BLUE else "red"


def _winner_str(winner) -> Optional[str]:
    if winner == BLUE:
        return "blue"
    if winner == RED:
        return "red"
    if winner == "Tie":
        return "tie"
    return None


def _board_state(env: CheckersEnv) -> list:
    """Return absolute (4, 8, 8) board as nested Python list."""
    return env.get_absolute_board_state().tolist()


def _legal_moves(env: CheckersEnv) -> list[dict]:
    """Return list of {from_sq, to_sq} dicts for all currently legal actions."""
    mask = env.get_action_mask()
    return [
        {"from_sq": from_sq, "to_sq": to_sq}
        for i, (from_sq, to_sq) in enumerate(ALL_ACTIONS)
        if mask[i] == 1.0
    ]


def _quick_eval(env: CheckersEnv, session: dict) -> float:
    """Single forward pass — returns value from AI's perspective."""
    network = session["mcts"].network if "mcts" in session else session["network"]
    state_t = torch.FloatTensor(env.get_board_state()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, v = network(state_t)
    raw = float(v.squeeze().item())  # from CURRENT player's perspective
    # Flip to AI's perspective when it's the human's turn
    if env.game.turn == session["human_color"]:
        return -raw
    return raw


def _run_ppo_turn(session: dict) -> tuple[Optional[str], float, bool, Optional[str], list]:
    """Execute the AI's complete turn using greedy PPO policy (no MCTS).

    Returns (move_notation, value, done, winner_str, hop_boards).
    """
    env: CheckersEnv = session["env"]
    network = session["network"]
    done = False
    winner = None
    value = 0.0
    last_move = None
    hop_boards: list = []

    while True:
        mask = env.get_action_mask()
        if mask.sum() == 0:
            _, _, done, _, info = env.step(0)
            winner = _winner_str(info.get("winner"))
            break

        state_t = torch.FloatTensor(env.get_board_state()).unsqueeze(0).to(DEVICE)
        mask_t  = torch.FloatTensor(mask).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, v = network(state_t)
        value = float(v.squeeze().item())

        masked_logits = logits + torch.where(
            mask_t > 0,
            torch.zeros_like(logits),
            torch.full_like(logits, -1e10),
        )
        action = int(masked_logits.argmax(dim=1).item())

        _, _, done, _, info = env.step(action)
        turn_complete = info.get("turn_complete", True)
        hop_boards.append(_board_state(env))

        if turn_complete or done:
            last_move = env.game.moves[-1] if env.game.moves else None
            if done:
                winner = _winner_str(info.get("winner"))
            break

    return last_move, value, done, winner, hop_boards


def _run_ai_turn(session: dict) -> tuple[Optional[str], float, bool, Optional[str], list]:
    """Execute the AI's complete turn (handles capture chains).

    Returns (move_notation, root_value, done, winner_str, hop_boards).
    hop_boards: list of board states after each hop (length == number of hops).
                Single move → 1 entry. Double capture → 2 entries.
    """
    env: CheckersEnv = session["env"]
    mcts: MCTSSearch = session["mcts"]
    done = False
    winner = None
    root_value = 0.0
    last_move = None
    hop_boards: list = []

    while True:
        mask = env.get_action_mask()
        if mask.sum() == 0:
            _, _, done, _, info = env.step(0)
            winner = _winner_str(info.get("winner"))
            break

        action, _, root_value = mcts.select_action(env, temperature=0.0)
        _, _, done, _, info = env.step(action)
        mcts.update_root(action)
        turn_complete = info.get("turn_complete", True)

        hop_boards.append(_board_state(env))  # board after this individual hop

        if turn_complete or done:
            last_move = env.game.moves[-1] if env.game.moves else None
            if done:
                winner = _winner_str(info.get("winner"))
            break

    return last_move, float(root_value), done, winner, hop_boards


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class NewGameRequest(BaseModel):
    model_id: str
    simulations: int = 100
    human_color: str = "blue"  # "blue" or "red"


class MoveRequest(BaseModel):
    game_id: str
    from_sq: int
    to_sq: int


class AIMoveRequest(BaseModel):
    game_id: str


class BoardResponse(BaseModel):
    game_id: str
    board: list                        # [4][8][8] — board after AI responded
    board_after_human: Optional[list]  # [4][8][8] — board after human's turn, before AI
    initial_board: Optional[list]      # [4][8][8] — starting position (new_game only)
    initial_value: Optional[float]     # eval of starting position (new_game only)
    post_ai_value: Optional[float]     # eval after AI moves (from AI's perspective)
    ai_boards: Optional[list]          # board after each AI hop for animation
    turn: str                    # "blue" | "red"
    legal_moves: list            # [{from_sq, to_sq}]
    done: bool
    winner: Optional[str]        # "blue" | "red" | "tie" | null
    value: Optional[float]       # root value from AI perspective (+1 AI winning)
    ai_move: Optional[str]       # last move notation e.g. "12-16" or "9x14x23"
    turn_complete: bool          # false during human capture chain
    human_color: str
    capturing_sq: Optional[int]  # board square of piece mid-capture-chain (human)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/models")
def get_models():
    """Return list of available models for the frontend dropdown."""
    return [
        {
            "id":                  m["id"],
            "label":               m["label"],
            "description":         m.get("description", ""),
            "supports_difficulty": m["supports_difficulty"],
        }
        for m in MODELS
        if m["id"] in MODEL_REGISTRY
    ]


@app.post("/api/new_game", response_model=BoardResponse)
def new_game(req: NewGameRequest):
    if req.model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not found")

    sims = max(10, min(req.simulations, 400))
    human_color = BLUE if req.human_color == "blue" else RED
    ai_color = RED if human_color == BLUE else BLUE

    env = CheckersEnv()
    env.reset()

    entry = MODEL_REGISTRY[req.model_id]
    game_id = str(uuid.uuid4())

    if entry["type"] == "ppo":
        games[game_id] = {
            "env":         env,
            "network":     entry["network"],
            "human_color": human_color,
            "ai_color":    ai_color,
            "is_ppo":      True,
            "model_label": entry["label"],
        }
    else:
        mcts = MCTSSearch(
            network=entry["network"],
            num_simulations=sims,
            device=DEVICE,
        )
        games[game_id] = {
            "env":         env,
            "mcts":        mcts,
            "human_color": human_color,
            "ai_color":    ai_color,
            "is_wdl":      entry["is_wdl"],
            "model_label": entry["label"],
        }

    ai_move = None
    value = None
    done = False
    winner = None

    # Capture starting position and its eval before any AI move
    starting_board = _board_state(env)
    initial_value  = _quick_eval(env, games[game_id])

    # If human plays Red, AI (Blue) moves first
    ai_hop_boards: list = []
    _ai_turn = _run_ppo_turn if games[game_id].get("is_ppo") else _run_ai_turn
    if human_color == RED:
        ai_move, value, done, winner, ai_hop_boards = _ai_turn(games[game_id])

    post_ai_value = _quick_eval(env, games[game_id]) if ai_move else None

    return BoardResponse(
        game_id=game_id,
        board=_board_state(env),
        board_after_human=None,
        initial_board=starting_board,
        initial_value=initial_value,
        post_ai_value=post_ai_value,
        ai_boards=ai_hop_boards if ai_hop_boards else None,
        turn=_color_str(env.game.turn),
        legal_moves=_legal_moves(env),
        done=done,
        winner=winner,
        value=value,
        ai_move=ai_move,
        turn_complete=True,
        human_color=req.human_color,
        capturing_sq=None,
    )


@app.post("/api/move", response_model=BoardResponse)
def make_move(req: MoveRequest):
    if req.game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found. Start a new game.")

    session = games[req.game_id]
    env: CheckersEnv = session["env"]

    if env.game.turn != session["human_color"] and not env._capture_in_progress:
        raise HTTPException(status_code=400, detail="Not the human's turn.")

    action = encode_action(req.from_sq, req.to_sq)
    if action == -1:
        raise HTTPException(status_code=400, detail="Invalid move geometry.")

    mask = env.get_action_mask()
    if mask[action] == 0:
        raise HTTPException(status_code=400, detail="Illegal move.")

    # Apply human move
    _, _, done, _, info = env.step(action)
    if "mcts" in session:
        session["mcts"].update_root(action)
    turn_complete = info.get("turn_complete", True)
    winner = _winner_str(info.get("winner")) if done else None

    # Mid capture chain — return restricted state, wait for next human step
    if not turn_complete:
        return BoardResponse(
            game_id=req.game_id,
            board=_board_state(env),
            board_after_human=None,
            initial_board=None,
            initial_value=None,
            post_ai_value=None,
            ai_boards=None,
            turn=_color_str(env.game.turn),
            legal_moves=_legal_moves(env),
            done=False,
            winner=None,
            value=None,
            ai_move=None,
            turn_complete=False,
            human_color="blue" if session["human_color"] == BLUE else "red",
            capturing_sq=env._capturing_piece_sq,
        )

    ai_move = None
    value = None

    # Snapshot board after human's turn
    board_after_human = _board_state(env)
    post_value = _quick_eval(env, session) if not done else None

    return BoardResponse(
        game_id=req.game_id,
        board=board_after_human,
        board_after_human=board_after_human,
        initial_board=None,
        initial_value=None,
        post_ai_value=None,
        ai_boards=None,
        turn=_color_str(env.game.turn),
        legal_moves=[] if done else _legal_moves(env),
        done=done,
        winner=winner,
        value=post_value,
        ai_move=None,
        turn_complete=True,
        human_color="blue" if session["human_color"] == BLUE else "red",
        capturing_sq=None,
    )


@app.post("/api/ai_move", response_model=BoardResponse)
def make_ai_move(req: AIMoveRequest):
    if req.game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found. Start a new game.")

    session = games[req.game_id]
    env: CheckersEnv = session["env"]

    if env.game.turn == session["human_color"]:
        raise HTTPException(status_code=400, detail="Not the AI's turn.")

    ai_hop_boards: list = []
    _ai_turn = _run_ppo_turn if session.get("is_ppo") else _run_ai_turn
    ai_move, value, done, winner, ai_hop_boards = _ai_turn(session)

    post_ai_value = _quick_eval(env, session) if not done else None

    return BoardResponse(
        game_id=req.game_id,
        board=_board_state(env),
        board_after_human=None,
        initial_board=None,
        initial_value=None,
        post_ai_value=post_ai_value,
        ai_boards=ai_hop_boards if ai_hop_boards else None,
        turn=_color_str(env.game.turn),
        legal_moves=[] if done else _legal_moves(env),
        done=done,
        winner=winner,
        value=value,
        ai_move=ai_move,
        turn_complete=True,
        human_color="blue" if session["human_color"] == BLUE else "red",
        capturing_sq=None,
    )


# ---------------------------------------------------------------------------
# Serve SvelteKit static build (must be last — catches all non-API routes)
# ---------------------------------------------------------------------------

FRONTEND_BUILD = REPO_ROOT / "web" / "frontend" / "build"
if FRONTEND_BUILD.exists():
    # Mount /_app/ separately for efficient static asset serving (JS, CSS, etc.)
    _app_dir = FRONTEND_BUILD / "_app"
    if _app_dir.exists():
        app.mount("/_app", StaticFiles(directory=str(_app_dir)), name="static_app")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve static files; fall back to 200.html for SPA client-side routing."""
        target = FRONTEND_BUILD / full_path
        if target.is_file():
            return FileResponse(str(target))
        fallback = FRONTEND_BUILD / "200.html"
        return FileResponse(str(fallback if fallback.exists() else FRONTEND_BUILD / "index.html"))
else:
    @app.get("/")
    def root():
        return {"message": "Frontend not built yet. Run: cd web/frontend && npm run build"}
