# 4. The PPO Training

Training the PPO agent was far from a straightforward process. Before I could even begin experimenting with complex league structures, I had to completely dismantle and rebuild the broken architecture from my initial 2024 attempt. 

The original agent never managed to beat a random baseline due to critical flaws: an ordinal action space that made position-invariant learning impossible, a missing loss penalty, and an indentation error that silently threw away 99% of the collected training data. However, the most damaging bug was an action masking mismatch between the data collection phase and the PPO update phase. This mismatch collapsed the importance sampling ratio $r_t(\theta)$ to near zero, completely flatlining the policy gradients. 

After rewriting the environment, fixing the math, and verifying that gradients were actually flowing, real training could finally begin.

### 4.1 The Curriculum Run & The Co-Evolution Trap

Training a naive agent directly on a full 12v12 board proved inefficient. To bootstrap the agent's understanding of basic mechanics and tactics, I implemented a two-phase curriculum. Phase 1 initialized games with randomized, asymmetric 4–9 piece mid-game positions. This forced the agent to learn both how to press a material advantage to win and how to defend a losing position. 

The strategy was highly successful. After transferring to the full 12v12 board at epoch 80, the agent rapidly adapted, reaching an 85.8% win rate against a random baseline in just 20 epochs.

However, around epoch 110, the agent fell into a massive **co-evolution trap**. The training data revealed a bizarre diagnostic signature: average episode lengths were *decreasing* while tie rates were *increasing*. The agents were not just playing passively—they had actively discovered how to deliberately trigger the 3-fold repetition tie rule using king backward-movement to end the game early.

This exploit was mathematically rational. With a discount factor of $\gamma=0.95$, a massive -200 tie penalty that occurs 60 moves in the future is heavily discounted (worth effectively -10). The agent realized that forcing an early tie now was mathematically superior to risking a distant loss. While bumping $\gamma$ to 0.995 corrected the mathematical incentive, the passive, draw-seeking behavior was already permanently baked into the opponent pool checkpoints. A structural fix was required.

### 4.2 The League Play Solution

To permanently cure the passive co-evolution, I discarded the standard self-play loop and initiated the multi-agent league, training the **Tactical**, **Terminal**, and **Aggressive** profiles simultaneously using Prioritized Fictitious Self-Play (PFSP).

The league was a massive success. Because the Aggressive agent was designed specifically to punish passive play, draw-seeking strategies were quickly exploited and eliminated from the pool. All three agents eventually reached a ~100% win rate against the random baseline, but more interestingly, they developed distinctly different playstyles:
- The **Aggressive** agent forced rapid exchanges, resulting in short, piece-hungry games averaging ~86 moves.
- The **Terminal** agent, receiving no intermediate rewards, developed long-horizon positional strategies, resulting in drawn-out games averaging ~116 moves.

### 4.4 Note on Ongoing Training

*(Note: The PPO training experiments are currently being re-run with updated hyperparameters. This section will be updated with the final metrics and narrative arcs once those runs complete.)*