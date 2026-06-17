# 3. PPO & Training Setup

With the environment defined, the next step was designing the training engine. I initially chose Proximal Policy Optimization (PPO) (Schulman et al., 2017) due to its stability and strong performance in discrete action spaces, avoiding the instability often seen in A2C or the continuous-action focus of Soft Actor-Critic (SAC).

### 3.1 The PPO Algorithm & GAE

PPO stabilizes training by preventing destructively large policy updates. The final loss function is a combination of three components: policy loss, value loss, and an entropy bonus.

- **Generalized Advantage Estimation (GAE)** (Schulman et al., 2015): PPO relies on the Advantage function $\hat{A}_t$, which estimates how much better a specific action was compared to the average expected return of that state. To balance the bias and variance of these estimates, GAE is used, incorporating exponentially-weighted multi-step temporal difference errors.
- **Policy Loss (Clipped Surrogate Objective)**: PPO computes an importance sampling ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ representing the probability of taking an action under the new policy compared to the old policy used to collect the trajectory. It then clips this ratio within a bounded window (typically $[1-\epsilon, 1+\epsilon]$, where $\epsilon = 0.2$). The policy loss is defined as:
  $$L^{CLIP}(\theta) = -\mathbb{E}_t \left[ \min( \underbrace{r_t(\theta)\hat{A}_t}_{\substack{\text{normal} \\ \text{policy update}}}, \underbrace{\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t}_{\substack{\text{bounds update} \\ \text{to prevent instability}}} ) \right]$$
  If the new policy drifts too far from the collection policy, the gradient is effectively zeroed out. While this mechanism provides robust stability, it is highly sensitive to the exact match between data collection and update distributions.
- **Value Loss**: The value head is trained to predict the expected return by minimizing the Mean Squared Error (MSE) between its predictions and the empirical returns. To prevent unbounded value network updates when the agent encounters out-of-distribution states, value function clipping is critical. First, we define a clipped value prediction $V_{clip}(s_t) = V_{\theta_{old}}(s_t) + \text{clip}(V_\theta(s_t) - V_{\theta_{old}}(s_t), -\epsilon, \epsilon)$. Then, the loss is:
  $$L^{VF}(\theta) = \mathbb{E}_t \left[ \max( \underbrace{(V_\theta(s_t) - V_t^{target})^2}_{\text{standard MSE}}, \underbrace{(V_{clip}(s_t) - V_t^{target})^2}_{\substack{\text{clipped MSE to} \\ \text{bound updates}}} ) \right]$$
- **Entropy Bonus**: To encourage exploration and prevent the policy from collapsing prematurely into a deterministic, suboptimal strategy, an entropy maximization term $S[\pi_\theta](s_t)$ is added to the objective. 

### 3.2 PPO Implementation Details

Beyond the core objective, several critical implementation details were required to stabilize training and improve sample efficiency:

- **Action Masking**: To prevent invalid moves, a large negative constant (`-1e10`) is added to invalid action logits before softmax activation. Crucially, identical masking must be applied during both data collection and the gradient update; a mismatch will skew the importance sampling ratio $r_t(\theta)$ and flatline the policy gradients (Huang & Ontanon, 2020).
- **On-Policy Mini-batching**: To maximize sample efficiency, the rollouts collected by the current policy are shuffled and reused for $K$ epochs across smaller mini-batches.
- **Advantage Normalization**: The GAE advantage estimates $\hat{A}_t$ are normalized across the batch `(advantages - mean) / std` to stabilize the variance of the policy gradient, ensuring the learning rate does not cause gradient explosions on batches with unusually high or low returns.
- **Gradient Clipping**: In addition to clipping the PPO objective, a global norm clip (`nn.utils.clip_grad_norm_`) is applied to the backpropagated gradients as a final layer of stability.
- **Learning Rate Scheduling**: The optimizer utilizes a cosine annealing schedule (`CosineAnnealingLR`), decaying the learning rate from $10^{-4}$ down to $10^{-6}$ to force convergence as the policy matures.
- **Data-Regularized Actor-Critic (DrAC)** (Raileanu et al., 2021): To improve the agent's ability to generalize across novel board states, DrAC acts as a regularizer during the PPO batch updates by adding random Gaussian noise to observations and passing these augmented copies through the network alongside the original data.

### 3.3 Reward Shaping & Discounting

In addition to the environment's terminal outcomes, the agent utilizes intermediate shaped rewards to provide denser learning signals throughout the game. These include a capture bonus (+10), a king promotion bonus (+15), a retroactive capture penalty applied after an opponent successfully takes a piece (-5), and a per-move time penalty. These are scaled down by a factor of 0.5 to ensure the terminal win/loss signals remain the dominant objective. 

Furthermore, the discount factor $\gamma$ heavily dictates the mathematical reality of these rewards. I discovered that a standard discount factor (`gamma=0.95`) shrinks the value of distant terminal rewards so significantly that early-game tie exploits (like intentionally repeating board states to force a tie and escape a distant loss) became mathematically rational for the agent, requiring adjustment.

### 3.4 Self-Play & Multi-Agent League

The agent acts as both players simultaneously, automatically generating its own curriculum as it improves. However, pure self-play in zero-sum games often leads to **strategy cycling** or **co-evolution collapse**, where the agent becomes hyper-specialized in exploiting its own specific weaknesses but remains fragile against diverse playstyles (Lanctot et al., 2017). To combat this, several macro-level structural mechanisms were layered over the course of training:

- **Curriculum Learning**: To bootstrap the agent's understanding of the game mechanics, I designed a curriculum training phase where games initialize with randomized, asymmetric 4–9 piece mid-game positions. This guarantees an initial material advantage for one side, forcing the agent to learn both how to press an advantage to win and how to defend a losing position before transitioning to the balanced 12v12 full board.
- **Opponent Pooling**: Inspired by large-scale RL systems like OpenAI Five (Berner et al., 2019), the agent periodically saves snapshots of its policy. It randomly selects these historical models as opponents for a fraction of its training games, forcing the current policy to remain robust against a variety of past strategies.
- **Prioritized Opponent Sampling (PFSP)**: Rather than sampling past opponents uniformly, the opponent pool tracks historical win rates against each checkpoint. Opponents that the current agent struggles against receive a higher sampling weight (using a softmax distribution over $1 - \text{win\_rate}$). This mechanism is grounded in Prioritized Fictitious Self-Play (Vinyals et al., 2019) and ensures that degradation against hard opponents is detected and corrected immediately.
- **League Play**: To solve the passive co-evolution collapse, I introduced a reward-diverse multi-agent league, inspired by both AlphaStar's role-based exploiters (Vinyals et al., 2019) and OpenAI Five's population-based training (Berner et al., 2019). The system simultaneously trains three distinct agent profiles that share the same prioritized opponent pool. This diversity guarantees that passive, draw-seeking strategies are quickly exploited. The three profiles are:
  - **Tactical**: The well-rounded baseline agent. It receives standard shaped rewards for captures and king promotions.
  - **Terminal**: A purely positional agent with all intermediate shaped rewards disabled. To compensate for learning exclusively from terminal win/loss signals, its discount factor ($\gamma=0.995$) and entropy bonus are increased to encourage deep exploration of long-horizon strategies.
  - **Aggressive**: The designated "exploiter" of passive opponents. It receives amplified shaped rewards for captures and promotions, alongside heavier time penalties, explicitly encouraging a piece-hungry, exchange-heavy playstyle.

### 3.5 Training Configuration

The PPO training loop parameters and multi-agent league configurations were scaled to ensure stability while maintaining computational efficiency:

**Base PPO Configuration**
| Parameter | Value |
| :--- | :--- |
| Learning rate | $10^{-4}$ (cosine annealed to $10^{-6}$) |
| PPO clip range ($\epsilon$) | 0.2 |
| GAE lambda ($\lambda$) | 0.95 |
| Discount factor ($\gamma$) | 0.99 |
| Update epochs ($K$) | 4 |
| Mini-batch size | 2048 |
| Games per epoch | 5000 |

To ensure stability, the base PPO hyperparameters were carefully selected. The **learning rate** was annealed down to $10^{-6}$ to force convergence as the policy matured. The **PPO clip range (0.2)** and **GAE lambda (0.95)** were kept at their industry-standard values to balance update magnitudes and the bias-variance trade-off. Reusing the collected data for **4 update epochs ($K$)** prevented the network from over-optimizing on a single batch, while simulating **5,000 games per epoch** provided a massive, diverse sample size for stable gradient estimates.

**League Agent Configurations**
| Agent Type | Capture Bonus | King Bonus | Shaping Scale | Time Penalty Scale | Discount ($\gamma$) | Entropy Bonus |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tactical** | 10.0 | 15.0 | 0.5 | 1.0 | 0.99 | 0.01 |
| **Terminal** | 0.0 | 0.0 | 0.0 | 0.0 | 0.995 | 0.02 |
| **Aggressive** | 15.0 | 20.0 | 0.8 | 2.0 | 0.99 | 0.02 |

For the league agents, the hyperparameters were tuned specifically to force distinct playstyles. The **Terminal** agent's lack of shaped rewards required a longer planning horizon (**$\gamma=0.995$**) and a higher **entropy bonus (0.02)** to encourage deep exploration of positional moves. Conversely, the **Aggressive** agent's **shaping scale** and **time penalty** were significantly amplified to force rapid, piece-hungry exchanges, expressly designed to punish any passive, draw-seeking opponents.