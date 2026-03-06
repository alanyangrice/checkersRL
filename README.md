# Checkers RL

A fully playable checkers game with reinforcement learning agents trained via **Proximal Policy Optimization (PPO)** self-play and **AlphaZero-style Monte Carlo Tree Search (MCTS)**. Play against a friend locally, challenge a trained AI, or interact through the web interface.

## Features

- **Interactive Checkers Game** — full American checkers rules with a Pygame GUI
- **PPO RL Agent** — trained through self-play with GPU-accelerated parallel game simulation
- **AlphaZero-style MCTS** — policy/value network guided tree search with scalar and WDL value heads
- **Multi-Agent League** — reward-diverse league training to prevent co-evolution collapse
- **GPU Training Server** — batched GPU inference server with zero-copy shared memory
- **Web Interface** — FastAPI + SvelteKit app to play against 6 trained models in the browser
- **Benchmark System** — automated evaluation vs. random and reference models each epoch
- **Game Replay** — replay recorded training games move-by-move with a visual interface

## Technologies


| Technology              | Purpose                                     |
| ----------------------- | ------------------------------------------- |
| **Python**              | Core language                               |
| **Pygame**              | Game rendering & user input                 |
| **PyTorch**             | Neural network + GPU training               |
| **NumPy**               | Board state encoding & array operations     |
| **Gymnasium**           | RL environment interface                    |
| **FastAPI / SvelteKit** | Web backend and frontend                    |
| **multiprocessing**     | Parallel game simulation across CPU workers |


## Project Structure

```
checkersRL/
├── checkers_game/                  # Core game engine (board, rules, move validation)
├── figures/                        # Training curve generation scripts
├── RL_models/
│   ├── checkers_env.py             # Gymnasium environment wrapper
│   ├── numpy_checkers_env.py       # Fast NumPy env for MCTS simulations
│   ├── play_agent.py               # Interactive play vs. trained agent
│   ├── replay_game.py              # Visual replay of recorded games
│   ├── az_vs_ppo.py                # Head-to-head evaluation: AlphaZero vs PPO
│   ├── PPO_Model/
│   │   ├── Agent.py                # PPO agent: GAE, DrAC augmentation, value clipping
│   │   ├── PolicyNetwork.py        # ResNet policy/value network
│   │   ├── Memory.py               # Experience buffer
│   │   ├── OpponentPool.py         # Past-checkpoint pool with PFSP sampling
│   │   ├── training_config.py      # Centralized hyperparameter configuration
│   │   ├── train.py                # Sequential single-process training
│   │   ├── train_parallel.py       # CPU-parallel training
│   │   ├── train_gpu_parallel.py   # GPU inference server + CPU workers (recommended)
│   │   ├── train_league.py         # Multi-agent league training
│   │   └── benchmark/              # CPU vs. GPU latency and epoch benchmarks
│   └── MCTS/
│       ├── mcts_node.py            # MCTSNode: PUCT scoring, visit counts
│       ├── mcts_search.py          # MCTS: select/expand/evaluate/backup loop
│       ├── AlphaZeroNetwork.py     # Scalar value network (tanh, MSE loss)
│       ├── WDLAlphaZeroNetwork.py  # WDL value network (softmax, cross-entropy)
│       ├── alphazero_trainer.py    # AlphaZero training loop
│       ├── train_gpu_parallel.py   # Parallel AlphaZero with GPU inference server
│       ├── evaluate.py             # Gating eval, value calibration, correctness tests
│       └── training_config.py      # AlphaZero hyperparameters and curriculum
└── web/
    ├── backend/                    # FastAPI server, model registry, game sessions
    └── frontend/                   # SvelteKit board UI
```

## Game Rules

Standard American checkers:

- 8×8 board with 12 pieces per side (Blue and Red)
- Pieces move diagonally forward; kings move in any diagonal direction
- Captures are mandatory when available, including multi-jump chains
- Pieces promote to kings upon reaching the opposite end of the board
- Win by capturing all opponent pieces or leaving them with no legal moves
- Tie after 250 total moves, 5-fold board repetition, or 40 consecutive turns without a capture or promotion

## How It Works

### Neural Network Architecture

An **AlphaZero-inspired ResNet** (~3.6M parameters) shared by both PPO and AlphaZero:

- **Input**: 4-channel 8×8 board — (my regular, my kings, opponent regular, opponent kings). Board is flipped vertically for Red so the agent always sees pieces moving forward.
- **Backbone**: 3×3 conv (4→256 channels) + 5 residual blocks (BatchNorm, skip connections)
- **Policy Head**: 1×1 conv → flatten → linear → 170 action logits, masked at −1e10 for illegal moves
- **Value Head**: PPO uses an unbounded scalar (MSE vs. GAE returns). AlphaZero offers a **scalar** variant (tanh ∈ [−1,1], MSE vs. {−1,0,+1} outcomes) or a **WDL** variant — softmax over [P(win), P(draw), P(loss)] with cross-entropy loss, enabling material-scaled draw contempt at inference.

### AlphaZero MCTS

Each move runs N simulations: **Select** (PUCT, c=1.5) → **Expand** → **Evaluate** (network value, no random rollouts) → **Backup**. Dirichlet noise (α=1.2, ε=0.35) is added at the root during self-play for exploration. Simulation count scales with a training curriculum — 75 sims/move in endgame positions, 200 in mid-game, 400 at full 12v12. Temperature is T=1.0 for the first 20 moves then T=0.4. The tree is reused between moves, preserving visit counts. A **regret buffer** collects positions where `|MCTS Q − outcome| > 0.5`; 20% of games per epoch start from these to focus training on misevaluated positions. Every 5 epochs, a gating test (50 games, ≥55% score) accepts or rejects the new checkpoint.

### GPU Training Architecture

- **Inference Server**: a thread in the main process batches forward-pass requests from CPU workers onto GPU via zero-copy shared memory
- **CPU Workers**: simulate games in parallel; opponent inference runs CPU-local; a dynamic task queue eliminates straggler delays
- **PPO Update**: runs on GPU in the main process after all games complete each epoch

### PPO Training Configuration


| Parameter        | Value                          |
| ---------------- | ------------------------------ |
| Learning rate    | 1e-4 (cosine annealed to 1e-6) |
| Discount (gamma) | 0.99                           |
| PPO clip range   | 0.2                            |
| GAE lambda       | 0.95                           |
| Mini-batch size  | 2048                           |
| Games per epoch  | 5000                           |


### AlphaZero Training Configuration


| Parameter                | Value                                            |
| ------------------------ | ------------------------------------------------ |
| Games per epoch          | 100 MCTS self-play games                         |
| Simulations per move     | 75 (endgame) → 200 (mid-game) → 400 (full 12v12) |
| Temperature              | T=1.0 for first 20 moves, then T=0.4             |
| Replay buffer            | 500,000 positions (FIFO)                         |
| Gradient steps per epoch | adaptive: clip(35 × newpositions / 256, 50, 500) |
| Gradient clip            | 1.0                                              |


### Reward Shaping (PPO mode)

All shaped rewards scaled by 0.5 so terminal outcomes dominate. Values below are the defaults for single-agent training; league agents override the tie penalty per-agent (tactical: −200, aggressive: −500).


| Signal                     | Reward                                                                       |
| -------------------------- | ---------------------------------------------------------------------------- |
| Win                        | +100                                                                         |
| Loss                       | −100                                                                         |
| King promotion             | +7.5                                                                         |
| Capture                    | +5.0                                                                         |
| Capture penalty (opponent) | −2.5 retroactive per captured piece                                          |
| Tie (default)              | −80 base − up to −48 stall (scales with material advantage and total pieces) |
| Time penalty               | −sqrt(moves)/10 per non-terminal move                                        |


### Multi-Agent League Play

Standard self-play causes **co-evolution collapse** — both agents converge to a mutual draw equilibrium because they share the same objective and play styles. This is documented in large-scale RL systems; AlphaStar (Vinyals et al., 2019) and OpenAI Five (Berner et al., 2019) both solved it via population diversity.

Our approach: **reward-diverse league training** with three agent types sharing a unified opponent pool. When any agent samples a pool opponent, it draws from all three agents' checkpoints — guaranteeing cross-style exposure every epoch.


| Agent type   | Reward profile                           | Emergent play style            |
| ------------ | ---------------------------------------- | ------------------------------ |
| `tactical`   | Balanced captures + king promotion       | Well-rounded baseline          |
| `terminal`   | Terminal only (win/loss/tie), no shaping | Long-horizon positional        |
| `aggressive` | 2× capture bonus, king promotion +25     | Piece-hungry, forces exchanges |


New agent types can be added by extending `LEAGUE_AGENTS` in `training_config.py`.

### Web Interface

- FastAPI backend (port 8000) serving 6 trained models: 2 AlphaZero variants (scalar, WDL) and 4 PPO variants (curriculum+self-play, and 3 league agents)
- SvelteKit frontend with HTML canvas board, model selector, MCTS simulation count slider, and eval bar
- Move responses include hop-by-hop board states for multi-capture chain animation

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- CUDA-capable GPU recommended for `train_gpu_parallel.py`

### Installation

```bash
git clone https://github.com/your-username/checkersRL.git
cd checkersRL
pip install -r requirements.txt
```

### Play Human vs Human

```bash
python -m checkers_game.main
```

### Train the Agent

Activate the virtual environment first (if using one):

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

```bash
# GPU-accelerated parallel training (recommended)
python -m RL_models.ppo.train_gpu_parallel

# CPU-only parallel training
python -m RL_models.ppo.train_parallel

# Sequential training (single process)
python -m RL_models.ppo.train

# AlphaZero: scalar value head
python -m RL_models.mcts.alphazero_trainer

# AlphaZero: WDL value head
python -m RL_models.mcts.alphazero_trainer --network-type wdl

# AlphaZero: GPU-accelerated parallel
python -m RL_models.mcts.train_gpu_parallel

# Multi-agent league play
python -m RL_models.ppo.train_league
```

All training scripts auto-resume from the latest checkpoint.

### Play Against the Agent

```bash
# Direct policy (fast) — loads latest checkpoint
python -m RL_models.play_agent

# Specific epoch
python -m RL_models.play_agent --epoch 50

# With MCTS search (stronger but slower)
python -m RL_models.play_agent --mcts --simulations 100
```

### AlphaZero vs PPO Head-to-Head

```bash
python -m RL_models.az_vs_ppo
```

### Replay a Training Game

```bash
python -m RL_models.replay_game --file RL_models/ppo/training_progress_detailed_parallel/detailed_games_epoch_50.zip --game 5

python -m RL_models.replay_game --moves "11-15, 24-20, 8-11, 28-24"
```

**Replay controls**: Arrow keys (step), Space (auto-play), Up/Down (speed), Home/End (jump), Q (quit)

### Run Benchmarks

```bash
python -m RL_models.ppo.benchmark.benchmark_inference
python -m RL_models.ppo.benchmark.benchmark_train   # requires checkpoint at ppo/PPO_saved_models_parallel/
```

### Start the Web Interface

```bash
# Backend
uvicorn web.backend.main:app --host 0.0.0.0 --port 8000

# Frontend (build once; served statically by the backend)
cd web/frontend && npm run build
```

### Run Tests

```bash
# With unittest (no extra deps)
python -m unittest tests.test_rl_models -v

# With pytest (install: pip install pytest)
python -m pytest tests/test_rl_models.py -v
```

### Lint and Format

```bash
# With ruff (install: pip install ruff)
ruff check RL_models/
ruff format RL_models/
```

## References

1. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.** (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347. [[paper]](https://arxiv.org/abs/1707.06347) — Core PPO training algorithm.
2. **Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P.** (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438. [[paper]](https://arxiv.org/abs/1506.02438) — GAE for lower-variance policy gradient.
3. **Silver, D., Hubert, T., Schrittwieser, J., et al.** (2018). *A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go Through Self-Play.* Science, 362(6419). arXiv:1712.01815. [[paper]](https://arxiv.org/abs/1712.01815) — AlphaZero: residual CNN architecture, MCTS self-play training, replay buffer and training step design.
4. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016. arXiv:1512.03385. [[paper]](https://arxiv.org/abs/1512.03385) — Residual blocks with skip connections.
5. **Raileanu, R., Goldstein, M., Yarats, D., Kostrikov, I., & Fergus, R.** (2021). *Automatic Data Augmentation for Generalization in Reinforcement Learning.* NeurIPS 2021. arXiv:2006.12862. [[paper]](https://arxiv.org/abs/2006.12862) — DrAC: observation augmentation in actor-critic training.
6. **Huang, S., & Ontanon, S.** (2020). *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms.* arXiv:2006.14171. [[paper]](https://arxiv.org/abs/2006.14171) — Consistent action masking during both collection and PPO update to preserve importance ratios.
7. **Coulom, R.** (2006). *Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search.* Computers and Games. [[paper]](https://link.springer.com/chapter/10.1007/978-3-540-75538-8_7) — Foundational MCTS with UCT.
8. **Rosin, C. D.** (2011). *Multi-armed Bandits with Episode Context.* Annals of Mathematics and Artificial Intelligence, 61(3), 203–230. — PUCT selection formula used in AlphaZero MCTS.
9. **Vinyals, O., Babuschkin, I., Czarnecki, W. M., et al.** (DeepMind). (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning.* Nature, 575(7782), 350–354. [[paper]](https://doi.org/10.1038/s41586-019-1724-z) — League play with agent-role diversity and PFSP opponent sampling to prevent co-evolution collapse.
10. **Berner, C., Brockman, G., Chan, B., et al.** (OpenAI). (2019). *Dota 2 with Large Scale Deep Reinforcement Learning.* arXiv:1912.06680. [[paper]](https://arxiv.org/abs/1912.06680) — Population-based training with diverse reward shaping and historical policy sampling (80% self-play / 20% historical) to prevent strategy cycling.
11. **Lanctot, M., Zambaldi, V., Gruslys, A., et al.** (2017). *A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning.* NeurIPS 2017. arXiv:1711.00832. [[paper]](https://arxiv.org/abs/1711.00832) — PSRO: game-theoretic foundation showing naive self-play converges to a pathological Nash equilibrium; best responses against the mixture of past policies provably converges to the true Nash.
12. **Lc0 Team.** (2020, April). *WDL Head.* Leela Chess Zero Blog. [[blog]](https://lczero.org/blog/2020/04/wdl-head/) — Separately predicting P(win), P(draw), P(loss) gives more calibrated evaluations in draw-heavy games and enables principled contempt via P(draw) for the material-ahead side. See also v0.30.0 release notes (Jul 2023) for WDL-based contempt implementation. [[release]](https://github.com/LeelaChessZero/lc0/releases/tag/v0.30.0)
13. **Willemsen, D., Baier, H., & Kaisers, M.** (2022). *Value targets in off-policy AlphaZero: a new greedy backup.* Neural Computing and Applications, 34(3), 1801–1814. — Soft-Z value blending (SOFT_Z_ALPHA=0.8): mixing MCTS Q-values with game outcomes as training targets to reduce on-policy bias from exploratory Dirichlet-noised self-play.
14. **Trudeau, F., & Bowling, M.** (2023). *Go-Exploit: Improving AlphaZero with Starting Positions from High-Regret States.* arXiv:2302.12359. [[paper]](https://arxiv.org/abs/2302.12359) — Regret-guided starting position diversity: replaying positions where the value estimate diverged most from the final outcome to direct training toward misevaluated states.
15. **Tsai, J. Y., et al.** (2026). *RGSC: Regret-Guided Self-play Curriculum.* arXiv:2602.20809. [[paper]](https://arxiv.org/abs/2602.20809) — Curriculum using high-regret states (|MCTS Q − outcome| > 0.5) as self-play starting positions for training position diversity.
16. **Joshi, A.** (2025). arXiv:2504.07757. [[paper]](https://arxiv.org/abs/2504.07757) — Corroborates simulation count non-monotonicity in AlphaZero evaluation; motivates matching EVAL_SIMULATIONS to the training simulation budget in gating tests.
17. **Wu, D.** (2019). *Accelerating Self-Play Learning in Go.* arXiv:1902.10565. [[paper]](https://arxiv.org/abs/1902.10565) — KataGo: higher replay ratios and increased training steps per epoch for improved sample efficiency in AlphaZero-style training.

## License

This project is open source. Feel free to use, modify, and distribute.