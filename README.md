# Checkers RL

A fully playable checkers game with a reinforcement learning agent trained via **Proximal Policy Optimization (PPO)** self-play. Play against a friend locally, challenge a trained AI opponent, or replay recorded training games.

## Features

- **Interactive Checkers Game** — full implementation of standard American checkers rules with a Pygame GUI
- **PPO RL Agent** — trained through self-play with GPU-accelerated parallel game simulation
- **AlphaZero-style MCTS** — alternative training approach using Monte Carlo Tree Search
- **GPU Training Server** — batched GPU inference server architecture for fast parallel training
- **Benchmark System** — automated evaluation vs random and reference models every N epochs
- **Game Replay** — replay recorded training games move-by-move with a visual interface
- **Play vs AI** — load any checkpoint and play against the agent interactively

## Technologies

| Technology | Purpose |
|---|---|
| **Python** | Core language |
| **Pygame** | Game rendering & user input |
| **PyTorch** | Neural network + GPU training |
| **NumPy** | Board state encoding & array operations |
| **Gymnasium** | RL environment interface |
| **multiprocessing** | Parallel game simulation across CPU workers |

## Project Structure

```
checkersRL/
├── requirements.txt
├── checkers_game/                  # Core game implementation
│   ├── main.py                     # Human vs human play entry point
│   ├── game.py                     # Game logic, turn management, win/tie detection
│   ├── board.py                    # Board state, move validation, captures
│   ├── piece.py                    # Piece class (regular & king)
│   ├── constants.py                # Colors, dimensions, board position mapping, action table
│   └── MoveNode.py                 # Tree structure for multi-capture sequences
│
└── RL_models/                      # Reinforcement learning components
    ├── checkers_env.py             # Gymnasium environment wrapper
    ├── play_agent.py               # Play against a trained agent interactively
    ├── replay_game.py              # Replay recorded training games visually
    ├── PPO_Model/
    │   ├── Agent.py                # PPO agent: GAE, action masking, augmentation
    │   ├── PolicyNetwork.py        # AlphaZero-inspired ResNet policy/value network
    │   ├── Memory.py               # Experience buffer (states, actions, rewards, masks)
    │   ├── OpponentPool.py         # Past-checkpoint opponent pool
    │   ├── training_config.py      # Centralized hyperparameter configuration
    │   ├── train.py                # Sequential CPU self-play training
    │   ├── train_parallel.py       # CPU-parallel self-play training
    │   ├── train_gpu_parallel.py   # GPU-accelerated training with inference server
    │   └── benchmark/
    │       ├── benchmark_inference.py  # CPU vs GPU latency benchmarks
    │       └── benchmark_train.py      # One-off epoch benchmark runner
    └── MCTS/
        ├── mcts_node.py            # MCTS tree node with PUCT scoring
        ├── mcts_search.py          # MCTS search algorithm (select/expand/evaluate/backup)
        └── alphazero_trainer.py    # AlphaZero training loop: MCTS self-play + supervised
```

## Game Rules

Standard American checkers:

- 8x8 board with 12 pieces per side (Blue and Red)
- Pieces move diagonally forward; kings move in any diagonal direction
- Captures are mandatory when available, including multi-jump chains
- Pieces promote to kings upon reaching the opposite end of the board
- Win by capturing all opponent pieces or leaving them with no legal moves
- Tie after 250 moves or 3 repeated board states

## How It Works

### Neural Network Architecture

An **AlphaZero-inspired residual CNN** (~3.6M parameters):

- **Input**: 4-channel 8x8 board — (my regular, my kings, opponent regular, opponent kings). Board is flipped vertically for Red so the agent always sees pieces moving in the same direction.
- **Backbone**: Initial 3x3 conv (4→256 channels) + 5 residual blocks (256 channels, BatchNorm, skip connections)
- **Policy Head**: 1x1 conv (256→2) + flatten + linear → 170 action logits
- **Value Head**: 1x1 conv (256→1) + flatten + 2-layer MLP → scalar value
- **Action Space**: 170 fixed (from_square, to_square) single-step pairs — moves and capture landings — with invalid-action masking

### GPU Training Architecture (train_gpu_parallel.py)

- **Inference Server**: a thread in the main process batches forward-pass requests from CPU workers and runs them on GPU
- **CPU Workers**: simulate games in parallel; send `(state, action_mask)` to the GPU server for agent moves; opponent inference runs CPU-local
- **Dynamic Scheduling**: shared task queue — faster workers get more tasks automatically, eliminating straggler delays
- **PPO Update**: runs on GPU in the main process after all games complete

### Training Configuration (training_config.py)

All hyperparameters are centralized:

| Parameter | Value |
|---|---|
| Learning rate | 1e-4 (cosine annealed to 1e-6) |
| Discount (gamma) | 0.95 |
| PPO clip range | 0.2 |
| GAE lambda | 0.95 |
| Mini-batch size | 2048 |
| Games per epoch | 5000 |
| Pool opponent prob | 15% |
| Pool epsilon | 15% |
| Epsilon decay | 1.0 → 0.08 over 100 epochs |

### Reward Shaping (PPO mode)

All shaped rewards scaled by 0.5 so terminal outcomes dominate.

| Signal | Reward |
|---|---|
| Win | +100 |
| Loss | -100 |
| King promotion | +7.5 (15 × 0.5) |
| Capture | +5.0 (10 × 0.5) |
| Capture penalty (opponent) | −2.5 retroactive (−5 × 0.5) |
| Blockout win | +100 (winner adjustment) |
| Blockout loss | −100 (loser adjustment) |
| Tie | −80 to −150 (scales with material advantage and total pieces) |
| Time penalty | −sqrt(moves)/10 per non-terminal move |

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

```bash
# GPU-accelerated parallel training (recommended)
python -m RL_models.PPO_Model.train_gpu_parallel

# CPU-only parallel training
python -m RL_models.PPO_Model.train_parallel

# Sequential training (single process)
python -m RL_models.PPO_Model.train

# AlphaZero: MCTS self-play training
python -m RL_models.MCTS.alphazero_trainer
```

### Play Against the Agent

```bash
# Direct policy (fast) — loads latest checkpoint
python -m RL_models.play_agent

# Specific epoch
python -m RL_models.play_agent --epoch 50

# With MCTS search (stronger but slower)
python -m RL_models.play_agent --mcts --simulations 100
```

### Replay a Training Game

```bash
# Replay game 5 from a training epoch CSV
python -m RL_models.replay_game --file RL_models/PPO_Model/training_progress_detailed_parallel/detailed_games_epoch_50.zip --game 5

# Replay a move string directly
python -m RL_models.replay_game --moves "11-15, 24-20, 8-11, 28-24"
```

**Replay controls**: Arrow keys (step), Space (auto-play), Up/Down (speed), Home/End (jump), Q (quit)

### Run Benchmarks

```bash
# Profile CPU vs GPU inference latency
python -m RL_models.PPO_Model.benchmark.benchmark_inference

# Run benchmark for a specific epoch
python -m RL_models.PPO_Model.benchmark.benchmark_train
```
 
## References

1. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.** (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347. [[paper]](https://arxiv.org/abs/1707.06347) — Core training algorithm.

2. **Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P.** (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438. [[paper]](https://arxiv.org/abs/1506.02438) — GAE for lower-variance policy gradient.

3. **Silver, D., Hubert, T., Schrittwieser, J., et al.** (2018). *A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go Through Self-Play.* Science, 362(6419). arXiv:1712.01815. [[paper]](https://arxiv.org/abs/1712.01815) — AlphaZero: residual CNN architecture, self-play training, 1x1 conv heads.

4. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016. arXiv:1512.03385. [[paper]](https://arxiv.org/abs/1512.03385) — Residual blocks with skip connections.

5. **Raileanu, R., Goldstein, M., Yarats, D., Kostrikov, I., & Fergus, R.** (2021). *Automatic Data Augmentation for Generalization in Reinforcement Learning.* NeurIPS 2021. arXiv:2006.12862. [[paper]](https://arxiv.org/abs/2006.12862) — DrAC: observation augmentation in actor-critic training.

6. **Huang, S., & Ontanon, S.** (2020). *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms.* arXiv:2006.14171. [[paper]](https://arxiv.org/abs/2006.14171) — Analysis of consistent action masking in PPO.

7. **Coulom, R.** (2006). *Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search.* Computers and Games. [[paper]](https://link.springer.com/chapter/10.1007/978-3-540-75538-8_7) — Foundational MCTS with UCT.

8. **Rosin, C. D.** (2011). *Multi-armed Bandits with Episode Context.* Annals of Mathematics and Artificial Intelligence, 61(3), 203–230. — PUCT selection formula used in AlphaZero MCTS.

## License

This project is open source. Feel free to use, modify, and distribute.
