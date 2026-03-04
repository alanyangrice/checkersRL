<script lang="ts">
	type Fig = { src: string; alt: string; caption: string; wide?: boolean };

	// ── AlphaZero Scalar (135 epochs) ──────────────────────────────────────
	// 1. Value polarization — immediate visual proof AZ is working
	// 2. Eval benchmarks  — performance evidence (gating + value calibration)
	// 3. Loss curves       — training stability
	// 4+5. Game complexity + Entropy — paired self-play diagnostics
	const azScalar: Fig[] = [
		{
			src: '/figures/az_scalar/AZ-S-2_value_polarization.png',
			alt: 'AlphaZero Scalar value head polarization',
			caption:
				'Root value averaged over winning vs. losing positions. The widening gap shows the value head learning to distinguish outcomes. Both curves converge back after epoch 66 when training dynamics shift.',
			wide: true,
		},
		{
			src: '/figures/az_scalar/AZ-S-5_eval_benchmarks.png',
			alt: 'AlphaZero Scalar evaluation benchmarks',
			caption:
				'Left: Model gating — each checkpoint is evaluated against the previous accepted model (▲ accepted, ▼ rejected). Gate win rate falls naturally as the reference strengthens. Right: Value head calibration on categorised positions — ideal targets are clear-win ≈ +1, clear-loss ≈ −1, equal ≈ 0.',
			wide: true,
		},
		{
			src: '/figures/az_scalar/AZ-S-1_loss_curves.png',
			alt: 'AlphaZero Scalar training loss curves',
			caption:
				'Policy, value, and total loss over 135 self-play epochs. Dashed vertical lines mark Phase 2 (epoch 16) and Phase 3 (epoch 66) — MCTS search-depth increases that cause the visible jumps in game length and epoch time.',
			wide: true,
		},
		{
			src: '/figures/az_scalar/AZ-S-3_game_complexity.png',
			alt: 'AlphaZero Scalar game complexity',
			caption:
				'Avg moves per game (left axis) and epoch wall-clock time (right axis). Both jump sharply at the Phase 2 and Phase 3 boundaries as MCTS runs deeper, producing longer strategic games.',
		},
		{
			src: '/figures/az_scalar/AZ-S-4_policy_entropy.png',
			alt: 'AlphaZero Scalar policy entropy decay',
			caption:
				'Policy entropy declines from ~1.31 → ~0.93 nats — the policy becomes more decisive without collapsing to near-determinism.',
		},
	];

	// ── AlphaZero WDL (ongoing) ────────────────────────────────────────────
	// 1. Value polarization — immediate visual hook
	// 2. Eval benchmarks  — performance evidence
	// 3. Loss curves       — training stability
	// 4+5. Game complexity + Entropy — paired diagnostics
	const azWdl: Fig[] = [
		{
			src: '/figures/az_wdl/AZ-W-2_value_polarization.png',
			alt: 'AlphaZero WDL value head polarization',
			caption:
				'Winner/loser polarization for the WDL model. The gap opens earlier and more cleanly than the scalar counterpart — consistent with the categorical WDL head providing a cleaner gradient signal.',
			wide: true,
		},
		{
			src: '/figures/az_wdl/AZ-W-5_eval_benchmarks.png',
			alt: 'AlphaZero WDL evaluation benchmarks',
			caption:
				'Left: Model gating over 9 checkpoints (epochs 5–45). Right: Value calibration — the WDL model\'s clear-win and clear-loss estimates reach higher magnitude earlier, consistent with the categorical head providing a cleaner gradient signal than MSE regression.',
			wide: true,
		},
		{
			src: '/figures/az_wdl/AZ-W-1_loss_curves.png',
			alt: 'AlphaZero WDL training loss curves',
			caption:
				'Policy, value, and total loss for the WDL model. Value loss decreases monotonically — unlike the scalar model which shows instability after epoch 66. Only Phase 2 (epoch 16) is marked; the run has not yet reached Phase 3.',
			wide: true,
		},
		{
			src: '/figures/az_wdl/AZ-W-3_game_complexity.png',
			alt: 'AlphaZero WDL game complexity',
			caption:
				'Game length and epoch time mirror the scalar model\'s Phase 2 transition at epoch 16, confirming the same MCTS configuration was applied.',
		},
		{
			src: '/figures/az_wdl/AZ-W-4_policy_entropy.png',
			alt: 'AlphaZero WDL policy entropy',
			caption:
				'Policy entropy follows a similar decay trajectory to the scalar model through the first 49 epochs.',
		},
	];

	// ── AlphaZero Comparison (epochs 1–49) ────────────────────────────────
	// C-2 first: value polarization directly answers whether WDL learns values better
	// C-1 second: loss curves as secondary stability evidence
	const azComparison: Fig[] = [
		{
			src: '/figures/az_comparison/AZ-C-2_value_polarization_comparison.png',
			alt: 'AlphaZero Scalar vs WDL value polarization comparison',
			caption:
				'Value polarization for both models. The shaded band between winner and loser curves shows the confidence gap — wider and earlier in training indicates faster value learning.',
			wide: true,
		},
		{
			src: '/figures/az_comparison/AZ-C-1_loss_comparison.png',
			alt: 'AlphaZero Scalar vs WDL loss comparison',
			caption:
				'Policy and value loss for both architectures over their shared first 49 epochs. WDL value loss (solid) descends steadily; scalar (dashed) is noisier and higher throughout, reflecting the harder MSE regression target.',
			wide: true,
		},
	];

	// ── PPO Curriculum + Self-Play (110 epochs) ───────────────────────────
	// 1. vs-Random   — headline: "did the agent learn?" (wide)
	// 2. vs-Reference — key narrative: passive co-evolution (wide)
	// 3+4. Tie rate + Episode length — paired passive-play diagnostics
	const ppoCurriculum: Fig[] = [
		{
			src: '/figures/ppo_curriculum/PPO-CS-1_vs_random_win_rate.png',
			alt: 'PPO Curriculum win rate vs random',
			caption:
				'Win rate against a fixed random opponent. The curriculum switch at epoch 81 (mid-game → full 12v12 board) produces a +16.4 pp gain in 10 epochs — vs. 50 epochs needed from a cold start in prior runs.',
			wide: true,
		},
		{
			src: '/figures/ppo_curriculum/PPO-CS-2_vs_reference.png',
			alt: 'PPO Curriculum vs reference agent',
			caption:
				'Win, loss, and tie rates against the self-play reference agent. Tie rate climbs to 37% by epoch 110 — agents learn to guarantee a draw via 3-fold repetition rather than risk losing.',
			wide: true,
		},
		{
			src: '/figures/ppo_curriculum/PPO-CS-4_tie_rate.png',
			alt: 'PPO Curriculum tie rate',
			caption:
				'Tie rate in self-play. Crosses the 5% reference near epoch 85 and reaches 24% by epoch 100, motivating the γ 0.95 → 0.995 fix in the next run.',
		},
		{
			src: '/figures/ppo_curriculum/PPO-CS-3_episode_length.png',
			alt: 'PPO Curriculum episode length',
			caption:
				'Average episode length. The jump at epoch 81 reflects the switch to full-board positions. Continued growth past epoch 90 is a diagnostic: passive agents extend games to delay terminal outcomes.',
		},
	];

	// ── PPO League Training (102 epochs) ──────────────────────────────────
	const ppoLeague: Fig[] = [
		{
			src: '/figures/ppo_league/PPO-L-1_vs_random_win_rate.png',
			alt: 'PPO League win rate vs random',
			caption:
				'Win rate against a random opponent for all three league agents. Aggressive leads early due to its 2× capture/king rewards. All three converge to near-100% by epoch 100.',
			wide: true,
		},
		{
			src: '/figures/ppo_league/PPO-L-2_heatmap_matrix.png',
			alt: 'PPO League head-to-head win rate matrix',
			caption:
				'Head-to-head win rates at four training snapshots. Row = agent, column = opponent. Outlined diagonal cells show each agent\'s win rate vs. its own checkpoint from 10 epochs prior. At epoch 70, Aggressive dominates (64% vs. Tactical, 63% vs. Terminal) before entropy collapse reduces it to near-parity by epoch 100.',
			wide: true,
		},
		{
			src: '/figures/ppo_league/PPO-L-4_tie_rate.png',
			alt: 'PPO League tie rate by agent type',
			caption:
				'Tie rate during self-play by agent type. Terminal\'s tie rate grows steadily — without shaping, tie avoidance is not incentivised. Aggressive\'s tie aversion reward keeps its rate near zero.',
		},
		{
			src: '/figures/ppo_league/PPO-L-3_episode_length.png',
			alt: 'PPO League episode length by agent type',
			caption:
				'Average episode length by agent. Terminal grows longest (123 moves at epoch 100) — without reward shaping, long-horizon reasoning is required to win. Aggressive stays shortest.',
		},
		{
			src: '/figures/ppo_league/PPO-L-5_reward_trajectories.png',
			alt: 'PPO League reward trajectories by agent type',
			caption:
				'Average epoch reward per agent (y-axes are not shared — reward functions are incomparable across agents). Vertical line at epoch 21 marks the curriculum switch from mid-game positions to full 12v12.',
			wide: true,
		},
	];

	// ── PPO Comparison ─────────────────────────────────────────────────────
	const ppoComparison: Fig = {
		src: '/figures/ppo_comparison/PPO-C-1_ppo_comparison.png',
		alt: 'PPO training comparison',
		caption:
			'Win rate vs. random across both PPO training regimes. Curriculum training (dashed) plateaus at ~86% and degrades by epoch 110 due to passive co-evolution. All three league agents (solid) reach near-100% and maintain it — reward diversity prevents the passive equilibrium from forming.',
	};
</script>

<svelte:head>
	<title>CheckersRL — Training Results</title>
</svelte:head>

<!-- Shared figure rendering helpers (inlined as reusable markup patterns) -->

<div class="flex flex-col gap-14">

	<!-- Header -->
	<div>
		<h1 class="text-2xl font-semibold text-gray-900">Training Results</h1>
		<p class="text-sm text-gray-500 mt-1">
			Publication-quality figures from four training runs. Click any figure to open it full-size.
		</p>
	</div>

	<!-- ================================================================= -->
	<!-- AlphaZero -->
	<!-- ================================================================= -->
	<section class="flex flex-col gap-8">

		<div>
			<h2 class="text-lg font-semibold text-gray-900">AlphaZero</h2>
			<p class="text-sm text-gray-500 mt-1">
				Self-play MCTS with a ResNet (5 residual blocks × 256 channels). Two value head
				variants compared — scalar MSE regression and Win/Draw/Loss categorical output.
			</p>
		</div>

		<!-- Scalar -->
		<div class="flex flex-col gap-4">
			<h3 class="text-xs font-semibold text-gray-500 uppercase tracking-wider border-b border-gray-100 pb-2">
				Scalar Value Head · 135 epochs
			</h3>
			<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
				{#each azScalar as fig}
					<figure class={fig.wide ? 'md:col-span-2' : ''}>
						<a href={fig.src} target="_blank" rel="noopener noreferrer"
							class="block rounded border border-gray-200 overflow-hidden hover:border-gray-400 transition-colors duration-150{fig.wide ? '' : ' aspect-[3/2]'}">
							<img
								src={fig.src} alt={fig.alt} loading="lazy"
								class={fig.wide ? 'w-full h-auto' : 'w-full h-full object-contain bg-white'}
							/>
						</a>
						<figcaption class="mt-2 text-xs text-gray-500 leading-relaxed">{fig.caption}</figcaption>
					</figure>
				{/each}
			</div>
		</div>

		<!-- WDL -->
		<div class="flex flex-col gap-4">
			<h3 class="text-xs font-semibold text-gray-500 uppercase tracking-wider border-b border-gray-100 pb-2">
				WDL Value Head · ongoing
			</h3>
			<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
				{#each azWdl as fig}
					<figure class={fig.wide ? 'md:col-span-2' : ''}>
						<a href={fig.src} target="_blank" rel="noopener noreferrer"
							class="block rounded border border-gray-200 overflow-hidden hover:border-gray-400 transition-colors duration-150{fig.wide ? '' : ' aspect-[3/2]'}">
							<img
								src={fig.src} alt={fig.alt} loading="lazy"
								class={fig.wide ? 'w-full h-auto' : 'w-full h-full object-contain bg-white'}
							/>
						</a>
						<figcaption class="mt-2 text-xs text-gray-500 leading-relaxed">{fig.caption}</figcaption>
					</figure>
				{/each}
			</div>
		</div>

		<!-- Comparison -->
		<div class="flex flex-col gap-4">
			<h3 class="text-xs font-semibold text-gray-500 uppercase tracking-wider border-b border-gray-100 pb-2">
				Scalar vs. WDL Comparison · epochs 1–49
			</h3>
			<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
				{#each azComparison as fig}
					<figure class={fig.wide ? 'md:col-span-2' : ''}>
						<a href={fig.src} target="_blank" rel="noopener noreferrer"
							class="block rounded border border-gray-200 overflow-hidden hover:border-gray-400 transition-colors duration-150{fig.wide ? '' : ' aspect-[3/2]'}">
							<img
								src={fig.src} alt={fig.alt} loading="lazy"
								class={fig.wide ? 'w-full h-auto' : 'w-full h-full object-contain bg-white'}
							/>
						</a>
						<figcaption class="mt-2 text-xs text-gray-500 leading-relaxed">{fig.caption}</figcaption>
					</figure>
				{/each}
			</div>
		</div>

	</section>

	<!-- ================================================================= -->
	<!-- PPO -->
	<!-- ================================================================= -->
	<section class="flex flex-col gap-8">

		<div>
			<h2 class="text-lg font-semibold text-gray-900">PPO</h2>
			<p class="text-sm text-gray-500 mt-1">
				Proximal Policy Optimisation with parallel self-play and opponent pools.
				Two training regimes: a curriculum with progressive board positions, and a
				reward-diverse league of three specialised agents.
			</p>
		</div>

		<!-- Curriculum -->
		<div class="flex flex-col gap-4">
			<h3 class="text-xs font-semibold text-gray-500 uppercase tracking-wider border-b border-gray-100 pb-2">
				Curriculum + Self-Play · 110 epochs
			</h3>
			<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
				{#each ppoCurriculum as fig}
					<figure class={fig.wide ? 'md:col-span-2' : ''}>
						<a href={fig.src} target="_blank" rel="noopener noreferrer"
							class="block rounded border border-gray-200 overflow-hidden hover:border-gray-400 transition-colors duration-150{fig.wide ? '' : ' aspect-[3/2]'}">
							<img
								src={fig.src} alt={fig.alt} loading="lazy"
								class={fig.wide ? 'w-full h-auto' : 'w-full h-full object-contain bg-white'}
							/>
						</a>
						<figcaption class="mt-2 text-xs text-gray-500 leading-relaxed">{fig.caption}</figcaption>
					</figure>
				{/each}
			</div>
		</div>

		<!-- League -->
		<div class="flex flex-col gap-4">
			<h3 class="text-xs font-semibold text-gray-500 uppercase tracking-wider border-b border-gray-100 pb-2">
				League Training · Tactical · Terminal · Aggressive · 102 epochs
			</h3>
			<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
				{#each ppoLeague as fig}
					<figure class={fig.wide ? 'md:col-span-2' : ''}>
						<a href={fig.src} target="_blank" rel="noopener noreferrer"
							class="block rounded border border-gray-200 overflow-hidden hover:border-gray-400 transition-colors duration-150{fig.wide ? '' : ' aspect-[3/2]'}">
							<img
								src={fig.src} alt={fig.alt} loading="lazy"
								class={fig.wide ? 'w-full h-auto' : 'w-full h-full object-contain bg-white'}
							/>
						</a>
						<figcaption class="mt-2 text-xs text-gray-500 leading-relaxed">{fig.caption}</figcaption>
					</figure>
				{/each}
			</div>
		</div>

	</section>

	<!-- ================================================================= -->
	<!-- PPO Comparison -->
	<!-- ================================================================= -->
	<section class="flex flex-col gap-4">
		<div>
			<h2 class="text-lg font-semibold text-gray-900">PPO Comparison</h2>
			<p class="text-sm text-gray-500 mt-1">Cross-run comparison on the shared vs-random baseline.</p>
		</div>
		<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
			<figure class="md:col-span-2">
				<a href={ppoComparison.src} target="_blank" rel="noopener noreferrer"
					class="block rounded border border-gray-200 overflow-hidden hover:border-gray-400 transition-colors duration-150">
					<img src={ppoComparison.src} alt={ppoComparison.alt} class="w-full h-auto" loading="lazy" />
				</a>
				<figcaption class="mt-2 text-xs text-gray-500 leading-relaxed">{ppoComparison.caption}</figcaption>
			</figure>
		</div>
	</section>

</div>
