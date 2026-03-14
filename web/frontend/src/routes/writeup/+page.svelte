<script lang="ts">
	import katex from 'katex';
	import 'katex/dist/katex.min.css';

	function math(tex: string, displayMode = false) {
		return katex.renderToString(tex, { throwOnError: false, displayMode });
	}
</script>

<svelte:head><title>CheckersRL — Writeup</title></svelte:head>

<div class="max-w-3xl flex flex-col gap-12 pb-20">
	<div>
		<h1 class="text-3xl font-bold text-gray-900">Project Writeup</h1>
		<p class="text-gray-500 mt-2">Currently in progress...</p>
	</div>

	<!-- Section 1 -->
	<section class="flex flex-col gap-4">
		<h2 class="text-2xl font-semibold text-gray-900 border-b border-gray-200 pb-2">1. Background</h2>

		<h3 class="text-xl font-medium text-gray-800 mt-4">Abstract</h3>
		<p class="text-gray-600 leading-relaxed">
			This project explores the development and training of autonomous agents to play standard American Checkers using Reinforcement Learning (RL). The implementation evaluates two distinct RL paradigms across four major training configurations: (1) Proximal Policy Optimization (PPO) utilizing self-play, curriculum learning, and opponent pooling; (2) PPO integrated into a reward-diverse multi-agent league system (featuring tactical, terminal, and aggressive agents) to combat the passive co-evolution collapse observed during standard self-play; (3) an AlphaZero-style Monte Carlo Tree Search (MCTS) utilizing a standard scalar value head; and (4) an AlphaZero MCTS utilizing a Win/Draw/Loss (WDL) value head. Furthermore, to scale the training efficiently, I designed a centralized GPU inference server (utilizing my Nvidia RTX 5090) alongside parallel CPU workers, which significantly accelerated game simulation. This writeup documents the architecture, the infrastructure design, and the iterative debugging process I went through to build an agent skilled at playing checkers.
		</p>

		<hr class="border-t border-gray-200 my-4" />

		<h3 class="text-xl font-medium text-gray-800 mt-4">Origin</h3>
		<p class="text-gray-600 leading-relaxed">
			The motivation for this project originated from observing the capabilities of Reinforcement Learning through creators like Code Bullet and the AlphaZero documentary on beating a Go grandmaster, as well as watching fascinating evolution and natural selection simulations by Primer. Seeing an agent start from zero knowledge and evolve complex, superhuman strategies through self-play and simulation demonstrated the power of these algorithms and inspired me to attempt building a similar system from the ground up, but much simpler. Hence, RL for checkers.
		</p>

		<h3 class="text-xl font-medium text-gray-800 mt-4">First Attempt (Sophomore Year, Fall 2024)</h3>
		<p class="text-gray-600 leading-relaxed">
			I initially tackled this project during my sophomore year as the final project for my STAT 413 class (Statistical Machine Learning) at Rice University. The goal was to build a checkers environment and use it to train an agent using RL.
		</p>
		<p class="text-gray-600 leading-relaxed">
			However, the implementation suffered from fundamental flaws due to my lack of RL knowledge. The agent's ability to learn was severely hampered by critical architectural errors. Key issues included the absence of proper loss penalties when losing, the use of ordinal, non-semantic action spaces (where identical network outputs mapped to entirely different board moves depending on the state), and silent bugs that flatlined the policy gradients (specifically, an action mask mismatch that collapsed the importance ratio, and a loop indentation error that threw away 99% of the collected training data before backpropagation). Compounding these issues was a lack of proper benchmarks and evaluation metrics, making it impossible for me at that time to diagnose these underlying bugs or even determine if the agent was learning. Ultimately, the agent failed to beat a random baseline. Acknowledging these limitations, I documented the findings for the final class submission and shelved the project.
		</p>

		<h3 class="text-xl font-medium text-gray-800 mt-4">The Return (Junior Winter, 2026)</h3>
		<p class="text-gray-600 leading-relaxed">
			In the fall of my junior year, I took COMP 442 (Reinforcement Learning), which provided the theoretical foundation needed to understand my previous mistakes. Concepts such as Markov Decision Processes, policy gradients, and importance sampling clarified exactly why the previous architecture was mathematically incapable of learning.
		</p>
		<p class="text-gray-600 leading-relaxed">
			Armed with this new knowledge, I restarted the project from scratch this winter while at my internship at RecruitU. This time, the objective was not just to train a capable agent at playing checkers, but to build robust, scalable training pipelines, and learn as much as possible. By developing custom parallelized infrastructure and methodically iterating through training configurations—starting from PPO with curriculum learning and opponent pooling, advancing to a reward-diverse PPO league system, and finally implementing and comparing AlphaZero MCTS with both scalar and WDL value heads—I am proud to say that I was able to achieve my goals.
		</p>
		<p class="text-gray-600 leading-relaxed">
			This writeup details the design principles, the engineering challenges, and the training insights from that effort.
		</p>
	</section>

	<!-- Section 2 -->
	<section class="flex flex-col gap-4">
		<h2 class="text-2xl font-semibold text-gray-900 border-b border-gray-200 pb-2">2. The Environment & Architecture</h2>
		<p class="text-gray-600 leading-relaxed mt-2">
			This section details formally defining the game and environment of American Checkers as a Reinforcement Learning problem and outlines the neural network architecture employed to evaluate board states and select actions.
		</p>

		<h3 class="text-xl font-medium text-gray-800 mt-4">2.1 The Game Mechanics</h3>
		<p class="text-gray-600 leading-relaxed">
			The environment adheres to the standard rules of American Checkers:
		</p>
		<ul class="list-disc list-outside text-gray-600 space-y-2 ml-5">
			<li class="leading-relaxed">The game is played on an {@html math("8 \\times 8")} board with 12 pieces per side (Blue and Red).</li>
			<li class="leading-relaxed">Regular pieces move diagonally forward, while kings may move diagonally in any direction.</li>
			<li class="leading-relaxed">Captures are mandatory, including sequential multi-jump capture chains.</li>
			<li class="leading-relaxed">To prevent infinite games, terminal tie conditions are enforced after 250 total moves, a 3-fold repetition of the board state made by the same player, or 40 consecutive moves without a capture or piece promotion.</li>
		</ul>

		<h3 class="text-xl font-medium text-gray-800 mt-4">2.2 Formulating the MDP</h3>
		<p class="text-gray-600 leading-relaxed">
			Checkers can be modeled as a two-player, zero-sum Markov Decision Process (MDP). An MDP is a mathematical framework used in reinforcement learning to describe an environment where outcomes are partly random and partly under the control of a decision-maker. It relies on the Markov property, meaning that the future state of the environment depends entirely on the current state and action, not on the sequence of events that preceded it. This formulation requires careful consideration of spatial symmetries and action semantics to facilitate effective learning:
		</p>
		<ul class="list-disc list-outside text-gray-600 space-y-2 ml-5">
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">State Space</strong>: The state is represented as a 4-channel {@html math("8 \\times 8")} board tensor from the perspective of the current player. The channels independently encode the positions of the current player's regular pieces, the current player's kings, the opponent's regular pieces, and the opponent's kings. To maintain spatial invariance, the board representation is flipped vertically for the Red player, ensuring both colors perceive their pieces as moving "forward" in the tensor representation. While a more compact {@html math("4 \\times 4 \\times 8")} representation could be used—discarding the half of the checkerboard squares that pieces can never legally occupy—I opted to maintain the full {@html math("8 \\times 8")} spatial structure to preserve standard 2D convolution semantics and allow for easier observability of the environment.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Action Space</strong>: The action space is defined over <strong class="font-semibold text-gray-900">170 discrete, semantic moves</strong>. Each action corresponds to a fixed <code class="bg-gray-100 px-1.5 py-0.5 rounded text-sm font-mono text-gray-800">(from_square, to_square)</code> coordinate pair on the board, replacing my previous, flawed approach of using an ordinal action space (e.g., selecting an index from a variable-length list of legal moves). This position-invariant approach allows the neural network to consistently associate specific output nodes with specific physical movements.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Transitions</strong>: The environment's state transition dynamics are entirely deterministic. The only source of stochasticity arises from the opponent's policy during self-play.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Rewards</strong>: The agent optimizes for terminal outcomes (+100 for a win, -100 for a loss). A tie incurs a dynamically calculated penalty that depends on the agent's playstyle and performance—combining a configurable base penalty (ranging from -80 to -150) with additional negative scaling based on how many pieces were left on the board (stalling) and whether the agent failed to convert a material advantage.</li>
		</ul>

		<h3 class="text-xl font-medium text-gray-800 mt-4">2.3 Neural Network Architecture</h3>
		<p class="text-gray-600 leading-relaxed">
			The agent relies on a deep residual neural network (ResNet), structurally inspired by the AlphaZero architecture (Silver et al., 2018) and incorporating residual blocks (He et al., 2016). The network contains approximately 3.6 million parameters and processes the board state tensor to produce both action probabilities and a state value evaluation:
		</p>
		<ul class="list-disc list-outside text-gray-600 space-y-2 ml-5">
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Backbone</strong>: The input tensor is passed through an initial 256-channel {@html math("3 \\times 3")} convolutional layer, followed by 5 residual blocks. Each residual block incorporates batch normalization and skip connections to stabilize gradient flow through the deep network.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Policy Head</strong>: The policy head processes the extracted features to output logits corresponding to the 170 semantic actions. Illegal actions for the given state are explicitly masked out by adding a large negative value (−1e10) before the softmax activation, strictly enforcing zero probability for invalid moves.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Value Head</strong>: The value head computes an evaluation of the current board state's expected return. For the PPO implementations, this is an unbounded scalar. In subsequent AlphaZero experiments, the architecture supports both a bounded scalar (tanh) and a categorical Win/Draw/Loss (WDL) distribution to better capture the draw-heavy nature of checkers.</li>
		</ul>
	</section>

	<!-- Section 3 -->
	<section class="flex flex-col gap-4">
		<h2 class="text-2xl font-semibold text-gray-900 border-b border-gray-200 pb-2">3. PPO & Training Setup</h2>
		<p class="text-gray-600 leading-relaxed mt-2">
			With the environment defined, the next step was designing the training engine. I initially chose Proximal Policy Optimization (PPO) (Schulman et al., 2017) due to its stability and strong performance in discrete action spaces, avoiding the instability often seen in A2C or the continuous-action focus of Soft Actor-Critic (SAC).
		</p>

		<h3 class="text-xl font-medium text-gray-800 mt-4">3.1 The PPO Algorithm & GAE</h3>
		<p class="text-gray-600 leading-relaxed">
			PPO stabilizes training by preventing destructively large policy updates. The final loss function is a combination of three components: policy loss, value loss, and an entropy bonus.
		</p>
		<ul class="list-disc list-outside text-gray-600 space-y-2 ml-5">
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Generalized Advantage Estimation (GAE)</strong> (Schulman et al., 2015): PPO relies on the Advantage function {@html math("\\hat{A}_t")}, which estimates how much better a specific action was compared to the average expected return of that state. To balance the bias and variance of these estimates, GAE is used, incorporating exponentially-weighted multi-step temporal difference errors.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Policy Loss (Clipped Surrogate Objective)</strong>: PPO computes an importance sampling ratio {@html math("r_t(\\theta) = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{old}}(a_t|s_t)}")} representing the probability of taking an action under the new policy compared to the old policy used to collect the trajectory. It then clips this ratio within a bounded window (typically [0.8, 1.2]). The policy loss is defined as:
				<div class="my-4 overflow-x-auto text-center">
					{@html math("L^{CLIP}(\\theta) = -\\mathbb{E}_t \\left[ \\min( \\underbrace{r_t(\\theta)\\hat{A}_t}_{\\substack{\\text{normal} \\\\ \\text{policy update}}}, \\underbrace{\\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon)\\hat{A}_t}_{\\substack{\\text{bounds update} \\\\ \\text{to prevent instability}}} ) \\right]", true)}
				</div>
				If the new policy drifts too far from the collection policy, the gradient is effectively zeroed out. While this mechanism provides robust stability, it is highly sensitive to the exact match between data collection and update distributions.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Value Loss</strong>: The value head is trained to predict the expected return by minimizing the Mean Squared Error (MSE) between its predictions and the empirical returns. To prevent unbounded value network updates when the agent encounters out-of-distribution states, value function clipping is critical. First, we define a clipped value prediction {@html math("V_{clip}(s_t) = V_{\\theta_{old}}(s_t) + \\text{clip}(V_\\theta(s_t) - V_{\\theta_{old}}(s_t), -\\epsilon, \\epsilon)")}. Then, the loss is:
				<div class="my-4 overflow-x-auto text-center">
					{@html math("L^{VF}(\\theta) = \\mathbb{E}_t \\left[ \\max( \\underbrace{(V_\\theta(s_t) - V_t^{target})^2}_{\\text{standard MSE}}, \\underbrace{(V_{clip}(s_t) - V_t^{target})^2}_{\\substack{\\text{clipped MSE to} \\\\ \\text{bound updates}}} ) \\right]", true)}
				</div>
			</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Entropy Bonus</strong>: To encourage exploration and prevent the policy from collapsing prematurely into a deterministic, suboptimal strategy, an entropy maximization term {@html math("S[\\pi_\\theta](s_t)")} is added to the objective.</li>
		</ul>

		<h3 class="text-xl font-medium text-gray-800 mt-4">3.2 PPO Implementation Details</h3>
		<p class="text-gray-600 leading-relaxed">
			Beyond the core objective, several critical implementation details were required to stabilize training and improve sample efficiency:
		</p>
		<ul class="list-disc list-outside text-gray-600 space-y-2 ml-5">
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Action Masking</strong>: To prevent invalid moves, a large negative constant (<code class="bg-gray-100 px-1.5 py-0.5 rounded text-sm font-mono text-gray-800">-1e10</code>) is added to invalid action logits before softmax activation. Crucially, identical masking must be applied during both data collection and the gradient update; a mismatch will skew the importance sampling ratio {@html math("r_t(\\theta)")} and flatline the policy gradients (Huang & Ontanon, 2020).</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">On-Policy Mini-batching</strong>: To maximize sample efficiency, the rollouts collected by the current policy are shuffled and reused for {@html math("K")} epochs across smaller mini-batches.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Advantage Normalization</strong>: The GAE advantage estimates {@html math("\\hat{A}_t")} are normalized across the batch to stabilize the variance of the policy gradient, ensuring the learning rate does not cause gradient explosions on batches with unusually high or low returns.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Gradient Clipping</strong>: In addition to clipping the PPO objective, a global norm clip is applied to the backpropagated gradients as a final layer of stability.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Learning Rate Scheduling</strong>: The optimizer utilizes a cosine annealing schedule, decaying the learning rate from 10<sup>-4</sup> down to 10<sup>-6</sup> to force convergence as the policy matures.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Data-Regularized Actor-Critic (DrAC)</strong> (Raileanu et al., 2021): To improve the agent's ability to generalize across novel board states, DrAC acts as a regularizer during the PPO batch updates by adding random Gaussian noise to observations and passing these augmented copies through the network alongside the original data.</li>
		</ul>

		<h3 class="text-xl font-medium text-gray-800 mt-4">3.3 Reward Shaping & Discounting</h3>
		<p class="text-gray-600 leading-relaxed">
			In addition to the environment's terminal outcomes, the agent utilizes intermediate shaped rewards to provide denser learning signals throughout the game. These include a capture bonus (+10), a king promotion bonus (+15), a retroactive capture penalty applied after an opponent successfully takes a piece (-5), and a per-move time penalty. These are scaled down by a factor of 0.5 to ensure the terminal win/loss signals remain the dominant objective.
		</p>
		<p class="text-gray-600 leading-relaxed">
			Furthermore, the discount factor heavily dictates the mathematical reality of these rewards. I discovered that a standard discount factor ({@html math("\\gamma=0.95")}) shrinks the value of distant terminal rewards so significantly that early-game tie exploits (like intentionally repeating board states to force a tie and escape a distant loss) became mathematically rational for the agent, requiring adjustment.
		</p>

		<h3 class="text-xl font-medium text-gray-800 mt-4">3.4 Self-Play & Multi-Agent League</h3>
		<p class="text-gray-600 leading-relaxed">
			The agent acts as both players simultaneously, automatically generating its own curriculum as it improves. However, pure self-play in zero-sum games often leads to <strong class="font-semibold text-gray-900">strategy cycling</strong> or <strong class="font-semibold text-gray-900">co-evolution collapse</strong>, where the agent becomes hyper-specialized in exploiting its own specific weaknesses but remains fragile against diverse playstyles (Lanctot et al., 2017). To combat this, several macro-level structural mechanisms were layered over the course of training:
		</p>
		<ul class="list-disc list-outside text-gray-600 space-y-2 ml-5">
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Curriculum Learning</strong>: To bootstrap the agent's understanding of the game mechanics, I designed a curriculum training phase where games initialize with randomized, asymmetric 4–9 piece mid-game positions. This guarantees an initial material advantage for one side, forcing the agent to learn both how to press an advantage to win and how to defend a losing position before transitioning to the balanced 12v12 full board.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Opponent Pooling</strong>: Inspired by large-scale RL systems like OpenAI Five (Berner et al., 2019), the agent periodically saves snapshots of its policy. It randomly selects these historical models as opponents for a fraction of its training games, forcing the current policy to remain robust against a variety of past strategies.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Prioritized Opponent Sampling (PFSP)</strong>: Rather than sampling past opponents uniformly, the opponent pool tracks historical win rates against each checkpoint. Opponents that the current agent struggles against receive a higher sampling weight (using a softmax distribution over {@html math("1 - \\text{win\\_rate}")}). This mechanism is grounded in Prioritized Fictitious Self-Play (Vinyals et al., 2019) and ensures that degradation against hard opponents is detected and corrected immediately.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">League Play</strong>: To solve the passive co-evolution collapse, I introduced a reward-diverse multi-agent league, inspired by both AlphaStar's role-based exploiters (Vinyals et al., 2019) and OpenAI Five's population-based training (Berner et al., 2019). The system simultaneously trains three distinct agent profiles that share the same prioritized opponent pool. This diversity guarantees that passive, draw-seeking strategies are quickly exploited. The three profiles are:
				<ul class="list-[circle] list-outside mt-2 ml-6 space-y-1 text-gray-600">
					<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Tactical</strong>: The well-rounded baseline agent. It receives standard shaped rewards for captures and king promotions.</li>
					<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Terminal</strong>: A purely positional agent with all intermediate shaped rewards disabled. To compensate for learning exclusively from terminal win/loss signals, its discount factor ({@html math("\\gamma=0.995")}) and entropy bonus are increased to encourage deep exploration of long-horizon strategies.</li>
					<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Aggressive</strong>: The designated "exploiter" of passive opponents. It receives amplified shaped rewards for captures and promotions, alongside heavier time penalties, explicitly encouraging a piece-hungry, exchange-heavy playstyle.</li>
				</ul>
			</li>
		</ul>

		<h3 class="text-xl font-medium text-gray-800 mt-4">3.5 Training Configuration</h3>
		<p class="text-gray-600 leading-relaxed">
			The PPO training loop parameters and multi-agent league configurations were scaled to ensure stability while maintaining computational efficiency:
		</p>

		<h4 class="text-lg font-medium text-gray-800 mt-4">Base PPO Configuration</h4>
		<div class="overflow-x-auto">
			<table class="min-w-full border-collapse border border-gray-200 text-sm text-left text-gray-600">
				<thead class="bg-gray-50 text-gray-800">
					<tr>
						<th class="border border-gray-200 px-4 py-2 font-medium">Parameter</th>
						<th class="border border-gray-200 px-4 py-2 font-medium">Value</th>
					</tr>
				</thead>
				<tbody>
					<tr>
						<td class="border border-gray-200 px-4 py-2">Learning rate</td>
						<td class="border border-gray-200 px-4 py-2">10<sup>-4</sup> (cosine annealed to 10<sup>-6</sup>)</td>
					</tr>
					<tr>
						<td class="border border-gray-200 px-4 py-2">PPO clip range ({@html math("\\epsilon")})</td>
						<td class="border border-gray-200 px-4 py-2">0.2</td>
					</tr>
					<tr>
						<td class="border border-gray-200 px-4 py-2">GAE lambda ({@html math("\\lambda")})</td>
						<td class="border border-gray-200 px-4 py-2">0.95</td>
					</tr>
					<tr>
						<td class="border border-gray-200 px-4 py-2">Discount factor ({@html math("\\gamma")})</td>
						<td class="border border-gray-200 px-4 py-2">0.99</td>
					</tr>
					<tr>
						<td class="border border-gray-200 px-4 py-2">Update epochs ({@html math("K")})</td>
						<td class="border border-gray-200 px-4 py-2">4</td>
					</tr>
					<tr>
						<td class="border border-gray-200 px-4 py-2">Mini-batch size</td>
						<td class="border border-gray-200 px-4 py-2">2048</td>
					</tr>
					<tr>
						<td class="border border-gray-200 px-4 py-2">Games per epoch</td>
						<td class="border border-gray-200 px-4 py-2">5000</td>
					</tr>
				</tbody>
			</table>
		</div>
		<p class="text-gray-600 leading-relaxed mt-2">
			To ensure stability, the base PPO hyperparameters were carefully selected. The <strong class="font-semibold text-gray-900">learning rate</strong> was annealed down to 10<sup>-6</sup> to force convergence as the policy matured. The <strong class="font-semibold text-gray-900">PPO clip range (0.2)</strong> and <strong class="font-semibold text-gray-900">GAE lambda (0.95)</strong> were kept at their industry-standard values to balance update magnitudes and the bias-variance trade-off. Reusing the collected data for <strong class="font-semibold text-gray-900">4 update epochs ({@html math("K")})</strong> prevented the network from over-optimizing on a single batch, while simulating <strong class="font-semibold text-gray-900">5,000 games per epoch</strong> provided a massive, diverse sample size for stable gradient estimates.
		</p>

		<h4 class="text-lg font-medium text-gray-800 mt-4">League Agent Configurations</h4>
		<div class="overflow-x-auto">
			<table class="min-w-full border-collapse border border-gray-200 text-sm text-left text-gray-600">
				<thead class="bg-gray-50 text-gray-800">
					<tr>
						<th class="border border-gray-200 px-4 py-2 font-medium">Agent Type</th>
						<th class="border border-gray-200 px-4 py-2 font-medium">Capture Bonus</th>
						<th class="border border-gray-200 px-4 py-2 font-medium">King Bonus</th>
						<th class="border border-gray-200 px-4 py-2 font-medium">Shaping Scale</th>
						<th class="border border-gray-200 px-4 py-2 font-medium">Time Penalty Scale</th>
						<th class="border border-gray-200 px-4 py-2 font-medium">Discount ({@html math("\\gamma")})</th>
						<th class="border border-gray-200 px-4 py-2 font-medium">Entropy Bonus</th>
					</tr>
				</thead>
				<tbody>
					<tr>
						<td class="border border-gray-200 px-4 py-2 font-medium text-gray-800">Tactical</td>
						<td class="border border-gray-200 px-4 py-2">10.0</td>
						<td class="border border-gray-200 px-4 py-2">15.0</td>
						<td class="border border-gray-200 px-4 py-2">0.5</td>
						<td class="border border-gray-200 px-4 py-2">1.0</td>
						<td class="border border-gray-200 px-4 py-2">0.99</td>
						<td class="border border-gray-200 px-4 py-2">0.01</td>
					</tr>
					<tr>
						<td class="border border-gray-200 px-4 py-2 font-medium text-gray-800">Terminal</td>
						<td class="border border-gray-200 px-4 py-2">0.0</td>
						<td class="border border-gray-200 px-4 py-2">0.0</td>
						<td class="border border-gray-200 px-4 py-2">0.0</td>
						<td class="border border-gray-200 px-4 py-2">0.0</td>
						<td class="border border-gray-200 px-4 py-2">0.995</td>
						<td class="border border-gray-200 px-4 py-2">0.02</td>
					</tr>
					<tr>
						<td class="border border-gray-200 px-4 py-2 font-medium text-gray-800">Aggressive</td>
						<td class="border border-gray-200 px-4 py-2">15.0</td>
						<td class="border border-gray-200 px-4 py-2">20.0</td>
						<td class="border border-gray-200 px-4 py-2">0.8</td>
						<td class="border border-gray-200 px-4 py-2">2.0</td>
						<td class="border border-gray-200 px-4 py-2">0.99</td>
						<td class="border border-gray-200 px-4 py-2">0.02</td>
					</tr>
				</tbody>
			</table>
		</div>
		<p class="text-gray-600 leading-relaxed mt-2">
			For the league agents, the hyperparameters were tuned specifically to force distinct playstyles. The <strong class="font-semibold text-gray-900">Terminal</strong> agent's lack of shaped rewards required a longer planning horizon (<strong class="font-semibold text-gray-900">{@html math("\\gamma=0.995")}</strong>) and a higher <strong class="font-semibold text-gray-900">entropy bonus (0.02)</strong> to encourage deep exploration of positional moves. Conversely, the <strong class="font-semibold text-gray-900">Aggressive</strong> agent's <strong class="font-semibold text-gray-900">shaping scale</strong> and <strong class="font-semibold text-gray-900">time penalty</strong> were significantly amplified to force rapid, piece-hungry exchanges, expressly designed to punish any passive, draw-seeking opponents.
		</p>
	</section>

</div>
