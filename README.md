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

The agent uses a **convolutional neural network** with:
- **Input**: 4-channel 8x8 board (current player regular, current player king, opponent regular, opponent king)
- **Backbone**: 4 convolutional layers (32 → 64 → 128 → 256 filters) with batch normalization
- **Policy Head**: 4 fully connected layers with dropout for action probabilities
- **Value Head**: 3 fully connected layers for state value estimation
- **Action masking**: Invalid moves are masked out before sampling

### Observation Normalization

The board state is always presented from the **current player's perspective** — channels represent "my pieces" and "opponent pieces" rather than fixed colors. When playing as Red, the board is flipped vertically so the agent always sees pieces moving in the same direction.

### Training Algorithm

**PPO with Generalized Advantage Estimation (GAE):**
- GAE (lambda=0.95) for lower-variance advantage estimates
- Cosine annealing learning rate scheduler
- Gradient clipping (max norm 0.5) for stability
- Epsilon-greedy exploration with decay

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

## License

This project is open source. Feel free to use, modify, and distribute.
