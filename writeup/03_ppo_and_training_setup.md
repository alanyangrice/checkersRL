# 3. PPO & Training Setup

With the environment defined, the next step is the training engine. I initially chose Proximal Policy Optimization ([PPO](08_references.md)) due to its stability and strong performance in discrete action spaces, avoiding the instability of A2C or the continuous-action focus of SAC.

### 3.1 Proximal Policy Optimization

PPO stabilizes policy gradient updates via a clipped surrogate objective. It computes an importance sampling ratio between the new policy and the old policy, clipping it within a window (usually `[0.8, 1.2]`). This prevents destructive, overly large updates. 
If the new policy drifts too far from the policy used to collect the data, the gradient is zeroed out. This mechanism is powerful, but as we'll see in the debugging saga, it's highly sensitive to bugs in action masking.

### 3.2 Self-Play & Opponent Pooling

The agent acts as both players simultaneously, automatically generating its own curriculum as it improves. 
However, pure self-play often leads to **strategy cycling**—the agent becomes hyper-specialized in exploiting its own specific weaknesses but collapses against diverse playstyles. 
To mitigate this, we implemented **opponent pooling**. The agent stores past snapshots of itself and randomly selects them as opponents for a fraction of its games. This forces robustness. The pool opponents inject a 15% exploration noise (random moves) to prevent deterministic play from triggering the 3-fold repetition tie rule.

### 3.3 Action Masking

In Checkers, the number of legal moves changes every turn. We apply action masking by adding `-1e10` to the logits of invalid actions before the softmax, forcing their probabilities to zero.
**A critical rule:** The exact same masking must be applied during both data collection and the PPO update. A mismatch here destroys the importance ratio, completely flatlining the gradients (a lesson learned the hard way).

### 3.4 Reward Shaping

Terminal rewards are sparse and only arrive at the end of a 70+ move game. To speed up learning, we shaped the intermediate rewards:

- **Positives**: Capture bonus (+10), king promotion (+15)
- **Negatives**: Time penalty (-sqrt(moves)/10), retroactive capture penalty (-5 for leaving a piece vulnerable)
These are scaled down by 0.5 so that the terminal win/loss signals still dominate the agent's objective.
We also learned that discounting (`gamma=0.95`) shrinks distant terminal rewards so much that early-game tie exploits became mathematically rational for the agent.

### 3.5 Additional Techniques

- **Generalized Advantage Estimation (GAE)**: Balances bias and variance using exponentially-weighted multi-step TD errors.
- **Data-Regularized Actor-Critic (DrAC)**: Adds random Gaussian noise to observations, acting as a regularizer during the PPO batch updates.

