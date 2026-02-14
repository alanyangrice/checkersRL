# Checkers RL

A fully playable checkers game with a reinforcement learning agent trained via **Proximal Policy Optimization (PPO)** self-play. Play against a friend locally or challenge a trained AI opponent.

## Features

- **Interactive Checkers Game** — full implementation of standard American checkers rules with a Pygame GUI
- **Reinforcement Learning Agent** — a PPO-based AI trained through self-play to learn checkers strategy
- **Parallel Training** — multiprocessing support for faster self-play training across CPU cores
- **Play vs AI** — load a trained model and play against the agent interactively

## Technologies

| Technology | Purpose |
|---|---|
| **Python** | Core language |
| **Pygame** | Game rendering & user input |
| **PyTorch** | Neural network (CNN) for the PPO agent |
| **NumPy** | Board state encoding & array operations |
| **Gymnasium** | RL environment interface |

## Project Structure

```
checkersRL/
├── requirements.txt                # Python dependencies
├── checkers_game/                  # Core game implementation
│   ├── main.py                     # Entry point for human vs human play
│   ├── game.py                     # Game logic, turn management, win/tie detection
│   ├── board.py                    # Board state, move validation, captures
│   ├── piece.py                    # Piece class (regular & king)
│   ├── constants.py                # Colors, dimensions, board position mapping
│   └── MoveNode.py                 # Tree structure for multi-capture sequences
│
├── RL_models/                      # Reinforcement learning components
│   ├── checkers_env.py             # Gymnasium environment wrapper for checkers
│   ├── play_agent.py               # Play against a trained agent
│   └── PPO_Model/                  # PPO implementation
│       ├── Agent.py                # PPO agent with GAE and action masking
│       ├── PolicyNetwork.py        # CNN policy/value network with batch norm
│       ├── Memory.py               # Experience replay buffer
│       ├── train.py                # Sequential self-play training
│       └── train_parallel.py       # Parallel self-play training
```

## Game Rules

Standard American checkers:

- 8x8 board with 12 pieces per side (Blue and Red)
- Pieces move diagonally forward; kings move diagonally in any direction
- Captures are mandatory when available, including multi-jump chains
- Pieces promote to kings upon reaching the opposite end of the board
- Win by capturing all opponent pieces or leaving them with no legal moves
- Ties after 250+ moves or 3 repeated board states

## How It Works

### Neural Network Architecture

The agent uses an **AlphaZero-inspired residual CNN** with:
- **Input**: 4-channel 8x8 board (current player regular, current player king, opponent regular, opponent king)
- **Backbone**: Initial conv layer + 5 residual blocks (256 channels each) with batch normalization and skip connections
- **Policy Head**: 1x1 conv (256 to 2 channels) + flatten + linear to 170 semantic actions
- **Value Head**: 1x1 conv (256 to 1 channel) + flatten + 2-layer MLP to scalar value
- **Action Space**: 170 fixed (from_square, to_square) single-step actions with invalid-move masking
- **~3.6M parameters** with efficient 1x1 conv heads (no FC bottleneck)

### Observation Normalization

The board state is always presented from the **current player's perspective** — channels represent "my pieces" and "opponent pieces" rather than fixed colors. When playing as Red, the board is flipped vertically so the agent always sees pieces moving in the same direction.

### Training Algorithm

**PPO with Generalized Advantage Estimation (GAE):**
- GAE (lambda=0.95) for lower-variance advantage estimates
- Cosine annealing learning rate scheduler
- Gradient clipping (max norm 0.5) for stability
- Epsilon-greedy exploration with decay
- Random noise data augmentation (DrAC-style) for observation robustness

### Reward Shaping

| Signal | Reward |
|---|---|
| Win | +100 |
| Loss | -100 |
| King promotion | +15 |
| Capture | +10 per piece |
| Center control | +0.5 |
| Back row defense | +0.5 |
| Tie | -20 |
| Undefended pieces | -5 |

## Getting Started

### Prerequisites

- Python 3.8+
- pip

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
# Sequential training
python -m RL_models.PPO_Model.train

# Parallel training (faster)
python -m RL_models.PPO_Model.train_parallel
```

### Play Against the Agent

```bash
python -m RL_models.play_agent
```

## References

This project draws on techniques from the following papers:

1. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.** (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347. [[paper]](https://arxiv.org/abs/1707.06347)
   - Core training algorithm (PPO with clipped surrogate objective).

2. **Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P.** (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438. [[paper]](https://arxiv.org/abs/1506.02438)
   - Generalized Advantage Estimation (GAE) for lower-variance policy gradient updates.

3. **Silver, D., Hubert, T., Schrittwieser, J., et al.** (2018). *A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go Through Self-Play.* Science, 362(6419), 1140-1144. [[paper]](https://arxiv.org/abs/1712.01815)
   - AlphaZero: inspiration for the residual CNN architecture, 1x1 conv heads, and self-play training framework.

4. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016. [[paper]](https://arxiv.org/abs/1512.03385)
   - Residual blocks with skip connections used in the policy/value network backbone.

5. **Raileanu, R., Goldstein, M., Yarats, D., Kostrikov, I., & Fergus, R.** (2021). *Automatic Data Augmentation for Generalization in Reinforcement Learning.* NeurIPS 2021. [[paper]](https://arxiv.org/abs/2006.12862)
   - DrAC (Data-regularized Actor-Critic): random noise augmentation applied to observations during PPO updates.

## License

This project is open source. Feel free to use, modify, and distribute.
