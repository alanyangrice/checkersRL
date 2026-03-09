# 2. The Environment & Architecture

This section details formally defining the game and environment of American Checkers as a Reinforcement Learning problem and outlines the neural network architecture employed to evaluate board states and select actions.

### 2.1 The Game Mechanics

The environment adheres to the standard rules of American Checkers:

- The game is played on an 8×8 board with 12 pieces per side (Blue and Red).
- Regular pieces move diagonally forward, while kings may move diagonally in any direction.
- Captures are mandatory, including sequential multi-jump capture chains.
- To prevent infinite games, terminal tie conditions are enforced after 250 total moves, a 3-fold repetition of the board state made by the same player, or 40 consecutive moves without a capture or piece promotion.

### 2.2 Formulating the MDP

Checkers can be modeled as a two-player, zero-sum Markov Decision Process (MDP). An MDP is a mathematical framework used in reinforcement learning to describe an environment where outcomes are partly random and partly under the control of a decision-maker. It relies on the Markov property, meaning that the future state of the environment depends entirely on the current state and action, not on the sequence of events that preceded it. This formulation requires careful consideration of spatial symmetries and action semantics to facilitate effective learning:

- **State Space**: The state is represented as a 4-channel 8×8 board tensor from the perspective of the current player. The channels independently encode the positions of the current player's regular pieces, the current player's kings, the opponent's regular pieces, and the opponent's kings. To maintain spatial invariance, the board representation is flipped vertically for the Red player, ensuring both colors perceive their pieces as moving "forward" in the tensor representation. While a more compact 4×4×8 representation could be used—discarding the half of the checkerboard squares that pieces can never legally occupy—I opted to maintain the full 8×8 spatial structure to preserve standard 2D convolution semantics and allow for easier observability of the environment.
- **Action Space**: The action space is defined over **170 discrete, semantic moves**. Each action corresponds to a fixed `(from_square, to_square)` coordinate pair on the board, replacing my previous, flawed approach of using an ordinal action space (e.g., selecting an index from a variable-length list of legal moves). This position-invariant approach allows the neural network to consistently associate specific output nodes with specific physical movements.
- **Transitions**: The environment's state transition dynamics are entirely deterministic. The only source of stochasticity arises from the opponent's policy during self-play.
- **Rewards**: The agent optimizes for terminal outcomes (+100 for a win, -100 for a loss). A tie incurs a dynamically calculated penalty that depends on the agent's playstyle and performance—combining a configurable base penalty (ranging from -80 to -150) with additional negative scaling based on how many pieces were left on the board (stalling) and whether the agent failed to convert a material advantage. Additionally, intermediate shaped rewards are utilized during the PPO training phase to provide denser learning signals.

### 2.3 Neural Network Architecture

The agent relies on a deep residual neural network (ResNet), structurally inspired by the AlphaZero architecture (Silver et al., 2018) and incorporating residual blocks (He et al., 2016). The network contains approximately 3.6 million parameters and processes the board state tensor to produce both action probabilities and a state value evaluation:

- **Backbone**: The input tensor is passed through an initial 256-channel 3×3 convolutional layer, followed by 5 residual blocks. Each residual block incorporates batch normalization and skip connections to stabilize gradient flow through the deep network.
- **Policy Head**: The policy head processes the extracted features to output logits corresponding to the 170 semantic actions. Illegal actions for the given state are explicitly masked out by adding a large negative value (−1e10) before the softmax activation, strictly enforcing zero probability for invalid moves.
- **Value Head**: The value head computes an evaluation of the current board state's expected return. For the PPO implementations, this is an unbounded scalar. In subsequent AlphaZero experiments, the architecture supports both a bounded scalar (tanh) and a categorical Win/Draw/Loss (WDL) distribution to better capture the draw-heavy nature of checkers.