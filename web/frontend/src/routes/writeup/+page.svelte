<script lang="ts">
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
			<li class="leading-relaxed">The game is played on an 8×8 board with 12 pieces per side (Blue and Red).</li>
			<li class="leading-relaxed">Regular pieces move diagonally forward, while kings may move diagonally in any direction.</li>
			<li class="leading-relaxed">Captures are mandatory, including sequential multi-jump capture chains.</li>
			<li class="leading-relaxed">To prevent infinite games, terminal tie conditions are enforced after 250 total moves, a 3-fold repetition of the board state made by the same player, or 40 consecutive moves without a capture or piece promotion.</li>
		</ul>

		<h3 class="text-xl font-medium text-gray-800 mt-4">2.2 Formulating the MDP</h3>
		<p class="text-gray-600 leading-relaxed">
			Checkers can be modeled as a two-player, zero-sum Markov Decision Process (MDP). An MDP is a mathematical framework used in reinforcement learning to describe an environment where outcomes are partly random and partly under the control of a decision-maker. It relies on the Markov property, meaning that the future state of the environment depends entirely on the current state and action, not on the sequence of events that preceded it. This formulation requires careful consideration of spatial symmetries and action semantics to facilitate effective learning:
		</p>
		<ul class="list-disc list-outside text-gray-600 space-y-2 ml-5">
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">State Space</strong>: The state is represented as a 4-channel 8×8 board tensor from the perspective of the current player. The channels independently encode the positions of the current player's regular pieces, the current player's kings, the opponent's regular pieces, and the opponent's kings. To maintain spatial invariance, the board representation is flipped vertically for the Red player, ensuring both colors perceive their pieces as moving "forward" in the tensor representation. While a more compact 4×4×8 representation could be used—discarding the half of the checkerboard squares that pieces can never legally occupy—I opted to maintain the full 8×8 spatial structure to preserve standard 2D convolution semantics and allow for easier observability of the environment.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Action Space</strong>: The action space is defined over <strong class="font-semibold text-gray-900">170 discrete, semantic moves</strong>. Each action corresponds to a fixed <code class="bg-gray-100 px-1.5 py-0.5 rounded text-sm font-mono text-gray-800">(from_square, to_square)</code> coordinate pair on the board, replacing my previous, flawed approach of using an ordinal action space (e.g., selecting an index from a variable-length list of legal moves). This position-invariant approach allows the neural network to consistently associate specific output nodes with specific physical movements.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Transitions</strong>: The environment's state transition dynamics are entirely deterministic. The only source of stochasticity arises from the opponent's policy during self-play.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Rewards</strong>: The agent optimizes for terminal outcomes (+100 for a win, -100 for a loss). A tie incurs a dynamically calculated penalty that depends on the agent's playstyle and performance—combining a configurable base penalty (ranging from -80 to -150) with additional negative scaling based on how many pieces were left on the board (stalling) and whether the agent failed to convert a material advantage. Additionally, intermediate shaped rewards are utilized during the PPO training phase to provide denser learning signals.</li>
		</ul>

		<h3 class="text-xl font-medium text-gray-800 mt-4">2.3 Neural Network Architecture</h3>
		<p class="text-gray-600 leading-relaxed">
			The agent relies on a deep residual neural network (ResNet), structurally inspired by the AlphaZero architecture (Silver et al., 2018) and incorporating residual blocks (He et al., 2016). The network contains approximately 3.6 million parameters and processes the board state tensor to produce both action probabilities and a state value evaluation:
		</p>
		<ul class="list-disc list-outside text-gray-600 space-y-2 ml-5">
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Backbone</strong>: The input tensor is passed through an initial 256-channel 3×3 convolutional layer, followed by 5 residual blocks. Each residual block incorporates batch normalization and skip connections to stabilize gradient flow through the deep network.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Policy Head</strong>: The policy head processes the extracted features to output logits corresponding to the 170 semantic actions. Illegal actions for the given state are explicitly masked out by adding a large negative value (−1e10) before the softmax activation, strictly enforcing zero probability for invalid moves.</li>
			<li class="leading-relaxed"><strong class="font-semibold text-gray-900">Value Head</strong>: The value head computes an evaluation of the current board state's expected return. For the PPO implementations, this is an unbounded scalar. In subsequent AlphaZero experiments, the architecture supports both a bounded scalar (tanh) and a categorical Win/Draw/Loss (WDL) distribution to better capture the draw-heavy nature of checkers.</li>
		</ul>
	</section>

</div>
