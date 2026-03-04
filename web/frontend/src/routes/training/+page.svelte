<script lang="ts">
	import { onMount } from 'svelte';

	type Fig = { src: string; alt: string; caption: string; wide?: boolean };

	let lightbox: { src: string; alt: string } | null = null;

	function openLightbox(fig: Fig) {
		lightbox = { src: fig.src, alt: fig.alt };
	}

	function closeLightbox() {
		lightbox = null;
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') closeLightbox();
	}

	onMount(() => {
		window.addEventListener('keydown', handleKeydown);
		return () => window.removeEventListener('keydown', handleKeydown);
	});

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
			"Mean raw value-head output v_θ(s) (side-to-move perspective, evaluated once per move before MCTS, \
			averaged over all moves in decisive games) for games eventually won (green) vs lost (red); faint \
			traces = per-epoch raw means, thick = 5-epoch centered rolling average, grey band = gap between smoothed \
			curves, dotted lines = ±1 output bounds, dashed verticals = Phase 2 (epoch 16) and Phase 3 (epoch 66). \
			Curves start near-zero and undifferentiated at epoch 1 and grow strongly polarized through Phases 1–2 (mean \
			separation Δ = V_win − V_loss ≈ 1.2); at the Phase 3 boundary the winner value drops from +0.618 to +0.331 \
			and the loser from −0.690 to −0.459 in a single epoch (Δ ≈ 0.7), and the reduced separation persists through \
			epoch 135. Because this metric is v_θ directly rather than the post-search Q-value (logged separately as \
			avg_mcts_q_*), the contraction is attributable to a change in the network's value estimates on the on-policy \
			distribution — likely driven by the longer games (avg moves: ~74 → ~135) altering the game-stage mix and shrinking \
			the decisive-only sample — motivating the WDL head comparison.",
			wide: true,
		},
		{
			src: '/figures/az_scalar/AZ-S-5_eval_benchmarks.png',
			alt: 'AlphaZero Scalar evaluation benchmarks',
			caption:
			"Two panels over 27 checkpoints (every 5 epochs, 5–130). (A) Arena gate score vs the current reference model \
			(50 games; score = (W + 0.5T)/50; accept ≥ 0.55); green ▲ accepted, red ▼ rejected; dashed verticals at Phase 2 \
			(epoch 16) and Phase 3 (epoch 66). The gate accepts 15/27 checkpoints, with a mid-run stall at epochs 20–30 \
			(all rejected) and late stagnation from epoch 115 onward (epochs 120/125/130 all rejected; epoch 120 worst at 0.39). \
			(B) Raw network value v_θ (no MCTS) on synthetic 4v1 (clear-win), 1v4 (clear-loss), and 3v3 (equal) positions, \
			each averaged over 10 random boards. Clear-win peaks at +0.998 (epoch 110); clear-loss reaches −0.999 (epoch 125); \
			equal positions oscillate across Phase 3 (−0.408 to +0.251) despite the formal calibration check passing every epoch. \
			The vs-random score saturates at 40/40 from epoch 10 and is uninformative; the equal-position volatility is the more \
			sensitive diagnostic and motivates the WDL head, which models draw probability explicitly.",
			wide: true,
		},
		{
			src: '/figures/az_scalar/AZ-S-1_loss_curves.png',
			alt: 'AlphaZero Scalar training loss curves',
			caption:
			"Three stacked subplots over epochs 1–135 show policy loss (top), value loss on a log scale (middle), and total loss \
			(bottom); faint traces are raw per-epoch values and solid lines are 5-epoch centered rolling averages; dashed verticals \
			mark Phase 2 (epoch 16) and Phase 3 (epoch 66). Policy loss drops from 2.695 (epoch 1) to 1.282 (epoch 15), bumps at Phase \
			2 onset to 1.361 (epoch 16), and gradually improves to ~1.283 by epoch 135, with only negligible disruption at the Phase 3 \
			boundary; value loss reaches an early minimum near ~0.063 (≈epochs 11–13), rises to 0.086 at epoch 15, spikes sharply at \
			Phase 2 onset to 0.151 (epoch 16), recovers to a Phase 2 local minimum (~0.068 at epoch 50), then rises steadily after Phase \
			3 to 0.215 by epoch 135 — more than tripling from its minimum, with the rise already underway before the boundary (0.101 at \
			epoch 65). The two heads respond differently to each phase transition (curriculum shift + MCTS budget increase + LR warm \
			restart): policy loss recovers quickly and continues improving throughout, while value loss rises steadily in Phase 3 \
			because the scalar regression target becomes harder to optimise as full-board games grow longer and produce more varied \
			position evaluations; because total loss = policy + 3×value, the Phase 3 rise in total loss reflects value-head difficulty \
			rather than policy regression.",
			wide: true,
		},
		{
			src: '/figures/az_scalar/AZ-S-3_game_complexity.png',
			alt: 'AlphaZero Scalar game complexity',
			caption:
			"Dual-axis plot: left y-axis (blue, 5-epoch centered rolling average) = average moves per self-play game; right y-axis \
			(purple, dash-dot) = epoch wall-clock time in minutes (self-play dominated); dashed verticals at Phase 2 (epoch 16) and \
			Phase 3 (epoch 66). Both curves jump sharply at each phase boundary — configuration-driven (curriculum + MCTS budget), \
			not learning-driven — because epoch time scales as games × avg_moves × sims/move. Phase 1→2 (epoch 15→16): avg_moves 52.3→77.1 \
			(+47%), epoch time 89.5→275.7s (~3.1×). Phase 2→3 (epoch 65→66): avg_moves 70.6→112.1 (+59%), epoch time 382.5→1027.7s \
			(~2.7×). By epoch ~135, games average ~130–140 moves and epochs take ~15–17 min. The replay buffer (fixed 500k-position deque) \
			saturates naturally during Phase 3 (~epoch 100–105) and remains at capacity via FIFO replacement.",
		},
		{
			src: '/figures/az_scalar/AZ-S-4_policy_entropy.png',
			alt: 'AlphaZero Scalar policy entropy decay',
			caption:
			"Shannon entropy (nats) of the MCTS visit-count policy targets sampled from the replay buffer (not the network softmax), \
			shown as a 5-epoch centered rolling average; dotted line = initial entropy (1.311 nats); dashed verticals at Phase 2 \
			(epoch 16) and Phase 3 (epoch 66). Entropy drops sharply through Phase 1 and at the Phase-2 transition (1.311 → 1.131 \
			by epoch 16, −14%), plateaus through Phase 2 (~1.13–1.15), then decays steadily after Phase 3 — crossing 1.0 near epoch \
			93 and reaching 0.966 at epoch 135 (total −26%) — consistent with sharper visit distributions under increased search budget \
			and Phase-3 buffer turnover. The floor (~0.93 nats) is structurally maintained by Dirichlet root noise (α = 1.2, ε = 0.35) \
			and high-temperature opening sampling (T = 1.0 for the first 20 moves), keeping visit counts diverse regardless of policy \
			strength.",
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

	// ── AlphaZero Comparison (epochs 1–130) ────────────────────────────────
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
		<p class="text-sm text-gray-500 mt-1"></p>
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
					<button on:click={() => openLightbox(fig)}
						class="block w-full text-left rounded border border-gray-200 overflow-hidden hover:border-gray-400 transition-colors duration-150 cursor-zoom-in{fig.wide ? '' : ' aspect-[3/2]'}">
						<img
							src={fig.src} alt={fig.alt} loading="lazy"
							class={fig.wide ? 'w-full h-auto' : 'w-full h-full object-contain bg-white'}
						/>
					</button>
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
					<button on:click={() => openLightbox(fig)}
						class="block w-full text-left rounded border border-gray-200 overflow-hidden hover:border-gray-400 transition-colors duration-150 cursor-zoom-in{fig.wide ? '' : ' aspect-[3/2]'}">
						<img
							src={fig.src} alt={fig.alt} loading="lazy"
							class={fig.wide ? 'w-full h-auto' : 'w-full h-full object-contain bg-white'}
						/>
					</button>
					<figcaption class="mt-2 text-xs text-gray-500 leading-relaxed">{fig.caption}</figcaption>
				</figure>
			{/each}
			</div>
		</div>

		<!-- Comparison -->
		<div class="flex flex-col gap-4">
			<h3 class="text-xs font-semibold text-gray-500 uppercase tracking-wider border-b border-gray-100 pb-2">
				Scalar vs. WDL Comparison · epochs 1–130
			</h3>
			<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
			{#each azComparison as fig}
				<figure class={fig.wide ? 'md:col-span-2' : ''}>
					<button on:click={() => openLightbox(fig)}
						class="block w-full text-left rounded border border-gray-200 overflow-hidden hover:border-gray-400 transition-colors duration-150 cursor-zoom-in{fig.wide ? '' : ' aspect-[3/2]'}">
						<img
							src={fig.src} alt={fig.alt} loading="lazy"
							class={fig.wide ? 'w-full h-auto' : 'w-full h-full object-contain bg-white'}
						/>
					</button>
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
					<button on:click={() => openLightbox(fig)}
						class="block w-full text-left rounded border border-gray-200 overflow-hidden hover:border-gray-400 transition-colors duration-150 cursor-zoom-in{fig.wide ? '' : ' aspect-[3/2]'}">
						<img
							src={fig.src} alt={fig.alt} loading="lazy"
							class={fig.wide ? 'w-full h-auto' : 'w-full h-full object-contain bg-white'}
						/>
					</button>
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
					<button on:click={() => openLightbox(fig)}
						class="block w-full text-left rounded border border-gray-200 overflow-hidden hover:border-gray-400 transition-colors duration-150 cursor-zoom-in{fig.wide ? '' : ' aspect-[3/2]'}">
						<img
							src={fig.src} alt={fig.alt} loading="lazy"
							class={fig.wide ? 'w-full h-auto' : 'w-full h-full object-contain bg-white'}
						/>
					</button>
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
				<button on:click={() => openLightbox(ppoComparison)}
					class="block w-full text-left rounded border border-gray-200 overflow-hidden hover:border-gray-400 transition-colors duration-150 cursor-zoom-in">
					<img src={ppoComparison.src} alt={ppoComparison.alt} class="w-full h-auto" loading="lazy" />
				</button>
				<figcaption class="mt-2 text-xs text-gray-500 leading-relaxed">{ppoComparison.caption}</figcaption>
			</figure>
		</div>
	</section>

</div>

<!-- Lightbox overlay -->
{#if lightbox}
	<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4"
		on:click={closeLightbox}
	>
		<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
		<div class="relative max-w-[90vw] max-h-[90vh]" on:click|stopPropagation>
			<img
				src={lightbox.src}
				alt={lightbox.alt}
				class="block max-w-full max-h-[90vh] rounded shadow-2xl object-contain"
			/>
			<button
				on:click={closeLightbox}
				class="absolute -top-3 -right-3 w-7 h-7 flex items-center justify-center rounded-full bg-white text-gray-700 shadow hover:bg-gray-100 transition-colors text-sm font-medium leading-none"
				aria-label="Close"
			>✕</button>
		</div>
	</div>
{/if}
