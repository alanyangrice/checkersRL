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
| **OpenAI Gym** | RL environment interface |

## Project Structure

```
checkersRL/
├── checkers_game/              # Core game implementation
│   ├── main.py                 # Entry point for human vs human play
│   ├── game.py                 # Game logic, turn management, win/tie detection
│   ├── board.py                # Board state, move validation, captures
│   ├── piece.py                # Piece class (regular & king)
│   ├── constants.py            # Colors, dimensions, board position mapping
│   └── MoveNode.py             # Tree structure for multi-capture sequences
│
├── RL_models/                  # Reinforcement learning components
│   ├── checkers_env.py         # Gym environment wrapper for checkers
│   ├── play_agent.py           # Play against a trained agent
│   └── PPO_Model/              # PPO implementation
│       ├── Agent.py            # PPO agent with action masking
│       ├── PolicyNetwork.py    # CNN policy/value network
│       ├── Memory.py           # Experience replay buffer
│       ├── train.py            # Sequential self-play training
│       └── train_parallel.py   # Parallel self-play training
```

## Game Rules

Standard American checkers:

- 8x8 board with 12 pieces per side (Blue and Red)
- Pieces move diagonally forward; kings move diagonally in any direction
- Captures are mandatory when available, including multi-jump chains
- Pieces promote to kings upon reaching the opposite end of the board
- Win by capturing all opponent pieces; ties after 250+ moves or 3 repeated board states

## How It Works

### Neural Network Architecture

The agent uses a **convolutional neural network** with:
- **Input**: 4-channel 8x8 board (red regular, red king, blue regular, blue king)
- **Backbone**: 4 convolutional layers (32 → 64 → 128 → 256 filters)
- **Heads**: Fully connected layers for policy (action probabilities) and value estimation
- **Action masking**: Invalid moves are masked out before sampling

### Reward Shaping

The agent receives shaped rewards to guide learning:

| Signal | Reward |
|---|---|
| Win | +100 |
| King promotion | +15 |
| Capture | +10 per piece |
| Center control | +0.5 |
| Back row defense | +0.5 |
| Tie | -20 |
| Undefended pieces | -5 |

### Training

Training is done via **self-play** — the agent plays as both Blue and Red, learning from both perspectives. Key hyperparameters:

- Learning rate: 1e-4
- Discount factor (γ): 0.95
- PPO clip: 0.2
- K epochs: 4
- Epsilon-greedy exploration with decay

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/your-username/checkersRL.git
cd checkersRL
pip install pygame torch numpy gym
```

### Play Human vs Human

```bash
python checkers_game/main.py
```

### Train the Agent

```bash
# Sequential training
python RL_models/PPO_Model/train.py

# Parallel training (faster)
python RL_models/PPO_Model/train_parallel.py
```

### Play Against the Agent

```bash
python RL_models/play_agent.py
```

> **Note:** You may need to update the model checkpoint path in `play_agent.py` to point to your saved model file.

## License

This project is open source. Feel free to use, modify, and distribute.
