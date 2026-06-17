# 6. AlphaZero & MCTS Training

Training the AlphaZero agent provided a stark contrast to the PPO methodology. By offloading the burden of immediate decision-making from the neural network to the Monte Carlo Tree Search (MCTS), the agent demonstrated incredible sample efficiency. While the PPO agent required nearly 50 epochs to confidently defeat a random baseline, the AlphaZero agent completely dominated the random baseline (saturating near a 100% win rate) by **Epoch 10**.

The training was structured around a 3-phase curriculum: **Phase 1 (Endgames)**, **Phase 2 (Midgames)**, and **Phase 3 (Full 12v12)**. I ran two full independent training runs across 250 epochs each to compare the standard **Scalar value head** against the **Win/Draw/Loss (WDL) value head**.

### 6.1 The Curriculum Progression & Loss Dynamics

During Phase 1 and Phase 2, both models learned rapidly. The policy loss dropped steeply in the first 15 epochs, while the value loss continued to decline through Phase 2, reaching its minimum around epoch 55–63 before the curriculum shift. Because the board positions were simplified (4–9 pieces), the MCTS could easily search to the end of the game and provide highly accurate value targets back to the network.

However, when the curriculum switched to the full 12v12 board at Epoch 66, both models hit the **Phase 3 Wall**. 

Average game lengths exploded, reaching upwards of 140 moves per game. This triggered a massive "hump" in the value loss for both architectures *(see Figure: Loss Comparison)*. Predicting the value of a full 12v12 Checkers board is notoriously difficult because long-horizon outcome supervision is inherently noisy—a single blunder on move 90 can invalidate a perfectly evaluated advantage on move 10. 

Over time, the loss trajectories diverged. The **Scalar model's** value loss (MSE) peaked at 0.217 around epoch 112 before substantially recovering to 0.124 by epoch 250. In contrast, the **WDL model's** value loss (Cross-Entropy) peaked later at 0.225 near epoch 138 and exhibited a much weaker recovery, ending at 0.192. Though MSE and cross-entropy magnitudes are not directly comparable, their differing recovery trajectories suggest that predicting categorical Win/Draw/Loss probabilities over a 140-move horizon remains challenging for longer than predicting a single smoothed scalar.

### 6.2 Model Gating & Evaluation Droughts

To prevent catastrophic forgetting, a new checkpoint was evaluated against a frozen reference model every 5 epochs. It had to achieve a strict gate score of `(win_rate + 0.5 * tie_rate) > 0.55` to be accepted. 

Both models suffered from severe **evaluation droughts**, where the network would stop improving and repeatedly fail to beat its past self. The Scalar model was particularly affected, with two prolonged stalls: epochs 40–65 (all 6 checkpoints rejected, with a gate win rate as low as 0.10 at epoch 45) and epochs 110–140 (all 7 checkpoints rejected), alongside persistent late-stage stagnation.

Despite having a strictly higher value loss, the **WDL model** proved slightly more robust in self-play. It achieved a 50% gate acceptance rate (compared to the Scalar model's 42%) and remained intermittently active and capable of beating its past self all the way through Epoch 240, demonstrating fewer prolonged stalls.

### 6.3 Value Polarization

The core difference between the two architectures is best illustrated by their **Value Polarization**—the spread ($\Delta$) between the network's raw evaluation of winning positions versus losing positions *(see Figure: Value Polarization Comparison)*. A wider spread means the network is highly confident and decisive.

In Phase 2, both models polarized rapidly, easily separating winning and losing states with a wide $\Delta \approx 1.45$ (Scalar peaking at 1.47 around epoch 50; WDL at 1.45 around epoch 59). However, the Phase 3 (full board) transition caused a sharp contraction in both models—confirming that the instability is driven by the curriculum distribution shift (full 12v12, longer games, higher MCTS budget) rather than the objective function alone. The fraction of decisive self-play games also dropped significantly (from ~79% to ~67% by epoch 250), meaning the average was increasingly diluted by ambiguous, drawish positions.
- The **Scalar model's** polarization contracted sharply and never recovered, ending training with a narrow spread of $\Delta = 0.477$. Calibration on clear synthetic positions (4v1 wins, 1v4 losses) remained saturated near $\pm 1$, indicating the head still recognized obvious wins and losses—but it struggled to confidently evaluate ambiguous 12v12 board states without MCTS lookahead.
- The **WDL model** proved significantly more resilient. It absorbed the shock of the Phase 3 shift and maintained a wider polarization gap throughout the rest of training, ending with $\Delta = 0.716$. 

Notably, both models exhibited persistent volatility when evaluating **equal-material positions** (3v3 boards) without MCTS. The Scalar model's equal-position predictions oscillated between $-0.412$ and $+0.540$, while the WDL model ranged from $+0.251$ to $-0.585$, despite formal calibration checks passing every epoch. This suggests that stable evaluation of ambiguous mid-game states remains fundamentally difficult for raw network predictions in both architectures—MCTS lookahead is doing the heavy lifting.

### 6.4 Conclusion: Scalar vs. WDL

The comparison revealed a clear trade-off. The **Scalar model** recovers more fully in terms of raw value loss, but its evaluations contract sharply under the complexity of the full board, leading to narrow polarization and severe gating droughts. 

The **WDL model** sustains a higher cross-entropy loss throughout Phase 3, but by forcing the network to explicitly model the "Draw" probability separate from the Win/Loss signal, the resulting evaluator maintains wider polarization and more consistent gating success. That said, neither architecture solved the fundamental challenge: both models rely heavily on MCTS lookahead to evaluate ambiguous positions, and both exhibit persistent instability on equal-material boards without search. The WDL head is the more resilient of the two, but the gap is one of degree rather than kind.