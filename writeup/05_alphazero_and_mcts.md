# 5. AlphaZero & MCTS Setup

While PPO successfully learned to play Checkers through the Multi-Agent League, it relied heavily on single-step value estimates, generalized advantage estimation, and careful reward shaping to solve the credit assignment problem. AlphaZero (Silver et al., 2018) takes a fundamentally different, and arguably more elegant, approach: explicit multi-step lookahead. 

By running Monte Carlo Tree Search (MCTS) at each decision point, the algorithm produces a much stronger action distribution than the raw policy network alone. This entirely eliminates the need for reward shaping—the only signal required is the final win, draw, or loss outcome of the game.

### 5.1 The MCTS Algorithm

At each move, the agent runs $N$ simulated games (rollouts) from the current board position to explore future possibilities. Each simulation traverses the search tree using four steps:

1. **Selection:** The tree is traversed from the root to a leaf node using the PUCT (Predictor + Upper Confidence bounds for Trees) formula (Rosin, 2011). At each node, the algorithm selects the child that maximizes:
   $$PUCT(s, a) = Q(s, a) + C_{PUCT} \cdot P(s, a) \frac{\sqrt{N(s)}}{1 + N(s, a)}$$
   This perfectly balances **exploitation** (choosing paths with high mean value, $Q(s,a)$) and **exploration** (giving unvisited paths a bonus that scales with the network's prior probability $P(s,a)$).
2. **Expansion:** Once an unexpanded leaf node is reached, the neural network evaluates the board state, producing a probability distribution of action priors and a scalar value estimate of the position. Child nodes are initialized for all legal moves.
3. **Evaluation:** Unlike traditional MCTS (Coulom, 2006) which runs random rollouts to the end of the game, AlphaZero uses the neural network's value estimate directly as the leaf evaluation.
4. **Backup:** The value is propagated back up the search path. Because Checkers features multi-jump captures where the same player might move multiple times in a row, the backup mechanism must carefully track the active player at each node, flipping the sign of the value *only* when the player-to-move changes across an edge. 

To ensure the agent explores a wide variety of strategies and doesn't deterministically collapse into a few narrow opening lines during self-play, **Dirichlet noise** is injected into the root node's priors. After all $N$ simulations complete, the action is sampled proportionally to the visit counts $N(a)^{1/\tau}$, where $\tau$ is a temperature parameter that dictates the final level of exploration.

### 5.2 The AlphaZero Training Loop

AlphaZero alternates between self-play data generation and network training. The continuous loop consists of:

1. **Self-Play:** The current network plays thousands of games against itself using MCTS. Each move produces a tuple of `(board_state, visit_counts, game_outcome)`. The softened visit count distribution over all legal moves becomes the **policy target** for training. It is a much stronger signal than the raw network prior because it has been refined by lookahead.
2. **Replay Buffer:** Unlike PPO which requires strict on-policy data and importance sampling, AlphaZero stores these tuples in a persistent replay buffer. This allows the network to train on each position multiple times. 
3. **Training Updates:** The network is trained using supervised learning. The Policy head minimizes the cross-entropy loss against the MCTS visit counts, while the Value head minimizes the error between its prediction and the actual game outcome.
4. **Gating:** To prevent catastrophic forgetting or policy degradation, the newly trained model is evaluated against a frozen reference checkpoint every few epochs. It must achieve a strict >55% win rate to be accepted. If it fails, the network weights and the replay buffer revert to the last accepted snapshot.

### 5.3 Parallel Inference Infrastructure

The primary bottleneck of AlphaZero is compute. Because MCTS requires hundreds of network evaluations per move (instead of PPO's one evaluation per step), generating self-play games sequentially is impractically slow. 

To solve this, I built a custom GPU-accelerated parallel infrastructure. A dedicated Main Process runs an `AlphaZeroInferenceServer` thread on the GPU. Dozens of CPU worker processes run the MCTS games and send batched leaf-evaluation requests to the GPU server via pre-allocated shared-memory NumPy arrays. The workers only pass their integer IDs through the Python `multiprocessing` queues, completely eliminating pickling overhead for the hundreds of thousands of IPC messages generated per epoch. This allowed for near 100% GPU utilization during the self-play phase.

### 5.4 Network Architectures

Over the course of the project, two distinct network architectures were designed and evaluated. Both shared the same ResNet backbone (He et al., 2016) (initial 3x3 conv into 5 residual blocks) and the same Policy head. They differed significantly in how they approached value prediction.

#### Network v1: The Scalar Architecture
The first iteration used a standard scalar value head. It compressed the 256-channel feature map through a 32-channel bottleneck (a deliberate upgrade to preserve positional capacity) before outputting a single scalar value bounded between `[-1, 1]` using a `tanh` activation. It was trained using Mean Squared Error (MSE) against the raw game outcome (+1 for a win, -1 for a loss).

#### Network v2: WDL & Soft-Z Blending
Drawing heavy inspiration from modern chess engines like Leela Chess Zero (Lc0), `Network v2` completely overhauled the value targets:

- **The WDL Head:** The scalar `tanh` head was replaced with a Win/Draw/Loss (WDL) softmax head that explicitly predicts the separate probabilities of `[P(win), P(draw), P(loss)]` (Lc0 Team, 2020). A standard scalar expected score conflates a volatile "50% win + 50% loss" position with a dead-locked "100% draw" position. The WDL head gives the network the vocabulary to distinguish between them, providing more calibrated evaluations in draw-heavy games like Checkers.
- **Soft-Z Value Blending:** A known issue with standard AlphaZero training is the on-policy bias of evaluating states reached under exploratory, Dirichlet-noised play rather than greedy deployment play. To combat this, `v2` training targets employ **Soft-Z value blending** (Willemsen et al., 2022). The training target is a weighted blend of the actual game outcome and the MCTS root Q-value: `Target = 0.8 * game_outcome + 0.2 * MCTS_Q`.
- **RGSC High-Regret Starts:** To forcefully inject diversity and prevent the agent from collapsing into repetitive opening lines, `v2` incorporates a Regret-Guided Self-Play Curriculum (Trudeau & Bowling, 2023; Tsai et al., 2026). During self-play, 20% of games are initialized not from the standard starting board, but from "high-regret" mid-game positions encountered in previous epochs (where the network's value estimate heavily diverged from the final game outcome).