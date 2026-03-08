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
			"Mean raw value-head output v_θ(s) (side-to-move, evaluated pre-MCTS, averaged over all moves in decisive games only) " +
			"for the eventual winner (green) and loser (red); faint traces show per-epoch means, thick lines a 5-epoch centered rolling average. " +
			"Polarization grows in Phases 1–2 but drops sharply at Phase-3 onset (e.g., epoch 65→66: winner +0.618 → +0.331, loser -0.690 → -0.459) " +
			"and remains reduced through epoch 250 (winner +0.175, loser -0.301). Because this measures raw on-policy evaluations, " +
			"the contraction is consistent with a distribution shift: Phase-3 games are much longer (~74 → ~135 moves) and more drawish " +
			"(84% → 43% decisive), potentially diluting the average with ambiguous early/midgame states. Calibration on clear synthetic " +
			"positions remains saturated near ±1, indicating the head still recognizes obvious wins/losses—motivating the WDL comparison " +
			"and move-stage analysis.",
			wide: true,
		},
		{
			src: '/figures/az_scalar/AZ-S-5_eval_benchmarks.png',
			alt: 'AlphaZero Scalar evaluation benchmarks',
			caption:
			"Two panels evaluating 50 checkpoints (every 5 epochs, 5–250). (A) Gate evaluation win rate vs. the current reference model \
			(50 games). While the y-axis plots absolute win rate, the acceptance criteria requires a gate score of \
			(W + 0.5T)/50 ≥ 0.55; green ▲ accepted, red ▼ rejected. Dashed verticals indicate the start of Phase 2 \
			(epoch 16) and Phase 3 (epoch 66). The gate accepts 21/50 checkpoints, highlighting severe mid-run stalls at epochs 40–65 \
			(all 6 rejected; lowest win rate 0.10 at epoch 45) and epochs 110–140 (all 7 rejected), alongside frequent late-stage rejections. \
			(B) Raw network value v_θ (no MCTS) on synthetic 4v1 (clear-win), 1v4 (clear-loss), and 3v3 (equal) positions, \
			each averaged over 10 random boards. Clear-win values saturate at +1.0 by epoch 165; clear-loss values smoothly reach \
			−0.999 by epoch 250. However, equal positions oscillate wildly throughout training (−0.412 at epoch 25 to +0.540 at epoch 115) \
			despite the formal calibration check passing every epoch. With the vs-random benchmark essentially saturating near 40/40 \
			from epoch 10, this equal-position volatility serves as a more sensitive diagnostic, suggesting that evaluation of \
			ambiguous mid-game states remains challenging.",
			wide: true,
		},
		{
			src: '/figures/az_scalar/AZ-S-1_loss_curves.png',
			alt: 'AlphaZero Scalar training loss curves',
			caption:
			"Three stacked subplots over epochs 1–250 show policy (top), log-scale value (middle), and total (bottom) losses, with " +
			"5-epoch rolling averages (solid) and Phase 2/3 boundaries at epochs 16 and 66 (dashed). Policy loss drops rapidly to " +
			"1.282 initially, bumps slightly at phase transitions, and slowly declines to ~1.268 by epoch 250. Value loss is highly " +
			"non-monotone: it hits an early minimum of ~0.063, spikes at Phase 2 (0.151), recovers, then surges after Phase 3 to peak " +
			"at 0.217 (epoch 112) before declining to 0.124. The Phase 3 hump in total loss is value-driven, reflecting temporary " +
			"difficulty predicting values under a harder distribution (longer games, deeper targets) before the network gradually adapts.",
			wide: true,
		},
		{
			src: '/figures/az_scalar/AZ-S-3_game_complexity.png',
			alt: 'AlphaZero Scalar game complexity',
			caption:
			"Dual-axis plot (epochs 0–250): left y-axis (blue, 5-epoch rolling mean) is average moves per self-play game; right y-axis (purple) " +
			"is epoch time. Dashed lines mark Phase 2 (epoch 16) and Phase 3 (epoch 66). Step increases at phase boundaries are driven by curriculum " +
			"and MCTS budget changes, as epoch time scales with games × avg_moves × sims/move. At Phase 1→2, moves jump 56.7→80.3 (+42%) and time " +
			"1.4→4.7 min (~3.3×). At Phase 2→3, moves jump 73.7→135.3 (+84%) and time 4.2→14.6 min (~3.5×). By epoch 250, games average 137 moves and " +
			"epochs take 16.1 min. The 500k-position replay buffer saturates at epoch 88, marking a stable, high-throughput data regime.",
		},
		{
			src: '/figures/az_scalar/AZ-S-4_policy_entropy.png',
			alt: 'AlphaZero Scalar policy entropy decay',
			caption:
			"Shannon entropy (nats) of MCTS visit-count policy targets sampled from the replay buffer, " +
			"shown as a 5-epoch rolling average; dotted line = initial (1.311 nats); dashed verticals at Phase 2 " +
			"(epoch 16) and Phase 3 (epoch 66). Entropy drops rapidly in Phase 1 (1.311 → 1.150), steps down at Phase 2 (1.131), " +
			"plateaus through Phase 2 (~1.11–1.18), then declines to a minimum of 0.932 at epoch 134 before rebounding to 1.015 " +
			"by epoch 250. The nonzero floor is maintained by Dirichlet root noise (α=1.2, ε=0.35) and high-temperature opening " +
			"sampling. The late-stage dynamics reflect changing target sharpness under higher search budgets and buffer distribution shifts.",
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
			"Mean raw WDL value-head output v_θ(s) = P(win) − P(loss) (side-to-move, evaluated once per move pre-MCTS, averaged over \
			decisive games only) for eventual winners (green) and losers (red); faint traces show per-epoch raw means, thick lines a \
			5-epoch centered rolling average. Values initialize near zero (epoch 1: −0.067 / −0.065) and polarize sharply (+0.539 / \
			−0.571 at epoch 2), strengthening through Phase 2 to a peak spread of Δ = 1.45 at epoch 59 (+0.681 / −0.773). At Phase-3 \
			onset (epoch 66), values contract sharply as in the scalar run (epoch 65→66: winner +0.572 → +0.448, loser −0.675 → −0.481), \
			collapsing Δ from 1.246 to 0.928. This shared contraction confirms the instability is driven by the Phase-3 distribution shift \
			(full 12v12 curriculum, longer games, increasing MCTS sims) rather than the objective function. Through extended training to epoch \
			250, polarization slowly decays further (winner +0.274, loser −0.442, Δ=0.716), alongside a reduction in decisive self-play \
			games (79% at epoch 65 → 67% at epoch 250).",
			wide: true,
		},
		{
			src: '/figures/az_wdl/AZ-W-5_eval_benchmarks.png',
			alt: 'AlphaZero WDL evaluation benchmarks',
			caption:
			"Two panels evaluating 50 checkpoints (every 5 epochs, 5–250). (A) Model gating vs the current reference: y-axis shows raw win \
			rate (W/50, ties counted as non-wins); accept/reject uses the tie-weighted gate score = (W + 0.5T)/50 ≥ 0.55; green ▲ accepted, \
			red ▼ rejected; dashed verticals at Phase 2 (epoch 16) and Phase 3 (epoch 66). Epoch 5 illustrates the distinction: raw win rate \
			= 0.48 (24W/26T) but gate score = 0.74, so the checkpoint passes. The gate accepts 25/50 checkpoints (50%), remaining intermittently \
			active throughout training; the last accepted checkpoint is epoch 240, while the worst performance occurs at epoch 170 (0.08 win \
			rate, 0.42 score). (B) Scalarized WDL value head output (P(win) − P(loss) + contempt × P(draw), no MCTS) on synthetic 4v1 \
			(clear-win), 1v4 (clear-loss), and 3v3 (equal) boards, each averaged over 10 random positions; phase boundaries marked identically \
			to (a). Clear-win peaks at +0.9998 (epoch 175) and clear-loss reaches −1.000 (epoch 250); vs-random averages ~0.998 and is \
			uninformative. Equal-position values oscillate persistently between +0.251 (epoch 95) and −0.585 (epoch 245), ending at −0.473; \
			nevertheless, the formal calibration check passes at every checkpoint. This persistent volatility indicates that ambiguous \
			equal-material states remain difficult to evaluate stably without MCTS, even with an explicit draw channel; under negative \
			contempt, part of the late negative drift may also reflect elevated P(draw) rather than pure miscalibration.",
			wide: true,
		},
		{
			src: '/figures/az_wdl/AZ-W-1_loss_curves.png',
			alt: 'AlphaZero WDL training loss curves',
			caption:
			"Three stacked subplots over epochs 1–250 show policy loss (top), value loss on a log scale (middle), and total loss (bottom); \
			faint traces are raw per-epoch values and solid lines are 5-epoch centered rolling averages; dashed verticals mark Phase 2 \
			(epoch 16) and Phase 3 (epoch 66). Policy loss drops from 2.176 (epoch 1) to 1.285 (epoch 10), bumps to 1.372 at Phase-2 onset, \
			and remains in the ~1.28–1.37 range, ending at 1.281 by epoch 250. Value loss traces a non-monotone path: it hits an early \
			minimum of 0.059 at epoch 55, spikes at the Phase-2 boundary (0.160), and surges significantly in Phase 3. Unlike the policy \
			loss, Phase 3 value loss continues rising to peak at 0.225 (epoch 138) before partially recovering to 0.192 at epoch 250. \
			The Phase-3 rise in total loss (total = policy + 3×value) is thus entirely value-driven, highlighting the sustained difficulty \
			of outcome supervision in long full-board games.",
			wide: true,
		},
		{
			src: '/figures/az_wdl/AZ-W-3_game_complexity.png',
			alt: 'AlphaZero WDL game complexity',
			caption:
			"Dual-axis plot (x = epochs 1–250): left y-axis shows average moves per self-play game (5-epoch centered rolling mean with raw \
			trace); right y-axis shows epoch wall-clock time (minutes, dash-dot); dashed verticals mark Phase 2 (epoch 16) and Phase 3 \
			(epoch 66). Phase transitions drive massive immediate jumps due to curriculum shifts and MCTS budget changes (75→200→400 sims/move): \
			Phase 1→2 (epoch 15→16) increases avg_moves 52.3→77.1 and epoch time 1.5→4.6 min (~3.1×); Phase 2→3 (epoch 65→66) increases \
			avg_moves 70.6→112.1 and epoch time 6.4→17.1 min (~2.7×). However, Phase 3 is not flat: game length drifts upward to peak at \
			141.5 moves (epoch 156), driving epoch time to a maximum of 22.7 min (epoch 213), before relaxing to 114.4 moves and 16.7 min \
			by epoch 250. The 500k-position replay buffer saturates at epoch 91.",
		},
		{
			src: '/figures/az_wdl/AZ-W-4_policy_entropy.png',
			alt: 'AlphaZero WDL policy entropy',
			caption:
			"Shannon entropy (nats) of the MCTS visit-count policy targets sampled from the replay buffer (faint = raw per-epoch, thick = \
			5-epoch centered rolling average); dotted line = initial entropy (1.360 nats); dashed verticals at Phase 2 (epoch 16) and Phase \
			3 (epoch 66). Entropy starts at 1.360 nats, drops to 1.139 by epoch 16, holds a plateau of ~1.14–1.19 through Phase 2, and then \
			declines overall through Phase 3 to reach a global minimum of 0.924 at epoch 170. By epoch 250, entropy partially rebounds to \
			0.984. The phase-wise structure is driven by configuration changes—MCTS budget increases sharpen visit distributions at each \
			boundary, and Phase 3 buffer turnover replaces plateau-era positions with longer full-board games. The nonzero floor is \
			maintained by Dirichlet root noise (α=1.2, ε=0.35) and high-temperature opening sampling.",
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
			"Mean raw pre-MCTS value-head output v_θ(s) (scalar: tanh output; WDL: P(win)−P(loss)) averaged over decisive-game moves for \
			eventual winners and losers across epochs 1–250; WDL in red (winner solid, loser dash-dot), scalar in blue (winner dashed, \
			loser dotted); shaded bands show polarization separation (Δ); 5-epoch centered rolling averages with faint raw traces. Both \
			models polarize rapidly after a single self-play epoch, with scalar attaining a slightly wider peak separation in Phase 2 \
			(Δ=1.47 at epoch 50 vs. WDL Δ=1.45 at epoch 59). However, WDL proves significantly more resilient to the Phase-3 curriculum \
			shift (epoch 66): the scalar spread collapses to 0.789, while WDL retains a spread of 0.928. This relative advantage persists \
			throughout the rest of training; by epoch 250, WDL maintains a much wider polarization gap (Δ=0.716) than the scalar model (Δ=0.477).",
			wide: true,
		},
		{
			src: '/figures/az_comparison/AZ-C-1_loss_comparison.png',
			alt: 'AlphaZero Scalar vs WDL loss comparison',
			caption:
			"Two-panel figure over epochs 1–250: (a) policy loss; (b) value loss on a log scale (scalar = MSE on tanh, WDL = cross-entropy \
			on W/D/L targets); WDL solid red, scalar dashed blue; 5-epoch centered rolling averages with faint raw traces. Policy losses \
			track nearly identically throughout training, remaining near ~1.27–1.28 by epoch 250, with WDL showing slightly larger bumps \
			at phase boundaries. Value loss tells a diverging story: while both objectives spike sharply at phase transitions, scalar value \
			loss forms a distinct Phase-3 hump (peaking at 0.217) before substantially recovering to 0.124 by epoch 250. In contrast, WDL \
			value loss rises later into Phase 3 (peaking at 0.225 near epoch 138) and exhibits a much weaker recovery, ending at 0.192. \
			Though cross-entropy and MSE magnitudes are not strictly comparable, their differing trajectories suggest WDL outcome prediction \
			remains challenging for longer during full-board play.",
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
			"x = evaluation checkpoint (every 10 epochs, 10–110); y = win rate vs a fixed random-action opponent (%); shaded area indicates \
			performance above the 50% baseline. The vertical dashed line at epoch 81 (between the 80 and 90 checkpoints) marks the curriculum \
			switch from randomised mid-game positions to the full 12v12 starting position. Win rate starts below baseline (45.2% at epoch 10), \
			rises non-monotonically to 64.8% by epoch 80, then jumps to 81.2% at epoch 90 (+16.4 pp) and peaks at 85.8% at epoch 100 before \
			declining to 81.6% at epoch 110. The pre-switch non-monotonicity (including the dip to 52.2% at epoch 70) is consistent with partial \
			transfer from the mid-game training distribution to full-start play; the post-peak decline aligns with increased draw-seeking and \
			passive co-evolution observed in self-play metrics (not shown). The 16.4 pp single-interval jump demonstrates the transfer benefit \
			of curriculum pre-training: in prior cold-start runs (not shown), the agent required ~50 epochs from scratch to cross the same \
			threshold.",
			wide: true,
		},
		{
			src: '/figures/ppo_curriculum/PPO-CS-2_vs_reference.png',
			alt: 'PPO Curriculum vs reference agent',
			caption:
			"Win rate (blue solid), loss rate (red dash-dot), and tie rate (purple dashed) evaluated every 10 epochs (10–110) against a \
			rolling reference that advances each benchmark (X vs X−10); dashed vertical at epoch 81 marks the curriculum switch to full \
			12v12 positions. Win rate peaks at 67.0% at epoch 80 (the last pre-switch checkpoint), then drops to 58.6% at epoch 100 and \
			40.6% at epoch 110; over the same interval, tie rate climbs from 6.0% to 22.6% to 37.0%, while loss rate decreases from 27.0% \
			to 22.4% — at epoch 110 the agent is more likely to draw (37.0%) than to lose (22.4%), a win-to-draw substitution rather than \
			general collapse. This pattern is consistent with passive co-evolution: as both the agent and the rolling reference train \
			together on the full-board distribution, they increasingly discover and exploit tie-seeking strategies; the behaviour is also \
			consistent with incentives under γ = 0.95, which heavily discounts long-horizon wins and can shift optimisation toward \
			lower-variance outcomes in long games.",
			wide: true,
		},
		{
			src: '/figures/ppo_curriculum/PPO-CS-4_tie_rate.png',
			alt: 'PPO Curriculum tie rate',
			caption:
			"Self-play tie rate (%) over epochs 1–110; faint trace = raw per-epoch, solid = 5-epoch rolling average; dotted horizontal \
			at 5%; dashed vertical at epoch 81 marks the curriculum switch. Tie rate holds stable at ~6–7.5% through epoch 71, then rises \
			sharply (8.18% at epoch 72; 13.54% at epoch 80), dips briefly at the switch (epoch 81: 9.26%), and resumes climbing to 24.10% \
			at epoch 100 and 27.84% at epoch 107. The epoch-100 training value (24.1%) closely matches the vs-reference benchmark tie rate \
			(22.6%), confirming the behaviour generalises beyond self-play partners. Because tie rate crosses ~10% around epoch 73–75 while \
			vs-random win rate is still improving, it is a more sensitive early-warning signal of passive co-evolution than win-rate metrics \
			alone.",
		},
		{
			src: '/figures/ppo_curriculum/PPO-CS-3_episode_length.png',
			alt: 'PPO Curriculum episode length',
			caption:
			"Average episode length (moves) over epochs 1–110 (faint = raw, thick = 5-epoch rolling average); dashed vertical at epoch 81 \
			marks the curriculum switch to full 12v12 starts. Length holds near ~51–54 moves through epoch 70 before rising to 64.9 at epoch \
			80, then jumps to 82.1 at the switch (+17.2 moves, configuration-driven). It continues growing to ~93–96 moves around epochs \
			100–101 before declining to 74.8 by epoch 110. The ~11-move post-switch rise is corroborated by concurrent tie-rate growth \
			(9.3% → 24.1%, epochs 81–100) and indicates passive co-evolution; the late decline reflects a shift to faster tie-forcing via \
			deliberate position repetition.",
		},
	];

	// ── PPO League Training (102 epochs) ──────────────────────────────────
	const ppoLeague: Fig[] = [
		{
			src: '/figures/ppo_league/PPO-L-1_vs_random_win_rate.png',
			alt: 'PPO League win rate vs random',
			caption:
			"Win rate of each league agent against a fixed random opponent, benchmarked every 10 epochs (500 games each); three lines — \
			Tactical (blue circles), Terminal (orange squares), Aggressive (red triangles) — with a 50% reference. At epoch 10, Aggressive \
			leads (57.4%) over Tactical (53.8%) and Terminal (52.0%), consistent with stronger shaping producing faster early gains, but by \
			epoch 20 all three cluster between 67.8–72.8%. Tactical reaches the highest absolute rate by epoch 50 (93.4%), while Terminal \
			shows the largest improvement over that interval (+22.0 pp, 67.8% → 89.8%) and Aggressive +18.6 pp (68.4% → 87.0%). All three \
			reach ≥97.8% by epoch 70 and saturate at 99.8–100% by epoch 80–100, at which point vs-random ceases to discriminate; early \
			differences likely reflect shaping and discount-horizon variations in learning speed rather than fundamentally different \
			strategies.",
			wide: true,
		},
		{
			src: '/figures/ppo_league/PPO-L-2_heatmap_matrix.png',
			alt: 'PPO League head-to-head win rate matrix',
			caption:
			"A 2×2 grid of 3×3 win-rate heatmaps (epochs 10, 40, 70, 100; 500 games per matchup); each cell reports the row agent's win rate \
			vs the column opponent (green = above 0.50, red = below); outlined diagonal cells show win rate against each agent's own checkpoint \
			from 10 epochs prior. At epoch 10, Aggressive leads cross-agent play (0.536 vs Tactical, 0.566 vs Terminal); by epoch 40 the advantage \
			erodes (0.528 vs Tactical, 0.472 vs Terminal); by epoch 70 the ordering has reversed — Tactical beats Aggressive 0.676 and Terminal \
			beats Aggressive 0.602, while Aggressive falls below 0.40 vs both (0.304 vs Tactical, 0.366 vs Terminal). At epoch 100, Aggressive \
			remains the weakest cross-agent performer (0.326 vs Tactical, 0.308 vs Terminal) despite continued self-improvement on the diagonal \
			(0.716, the highest of any agent at that checkpoint). The diagonal scores peak at epoch 70 for Tactical (0.730) and Terminal (0.660), \
			marking the fastest learning phase for those agents. The reversal indicates that strong capture-and-promotion shaping yields rapid \
			early gains but produces a play style that becomes systematically exploitable as Tactical and Terminal adapt; by epoch 100, Aggressive \
			is still improving against its own past self yet is consistently countered cross-agent, highlighting reward-style non-transitivity in \
			the league.",
			wide: true,
		},
		{
			src: '/figures/ppo_league/PPO-L-4_tie_rate.png',
			alt: 'PPO League tie rate by agent type',
			caption:
			"Self-play tie rate (%) across 5,000 training games per epoch (~1–100), with faint traces showing raw values and thick lines \
			showing smoothed trends; a dotted reference marks 5%. All three agents stay below 1% through ~epoch 50, after which their trajectories \
			diverge sharply: Terminal climbs rapidly, crossing 5% at ~epoch 84 and peaking at ~10.56% at epoch 93 before settling at ~8.34% by epoch \
			100; Tactical rises gradually to ~2.54%; Aggressive stays ≤0.8% throughout. The ordering Aggressive < Tactical < Terminal is consistent \
			across training and tracks differences in shaping density and urgency — Terminal's sparse outcome-only signal (no capture or promotion \
			rewards) leaves it prone to drifting into drawish equilibria as policies mature, while Aggressive's amplified capture/king rewards and \
			2× time penalty appear to suppress ties by keeping play decisive, despite all three agents sharing the same terminal draw penalty \
			(tie_base = −90).",
		},
		{
			src: '/figures/ppo_league/PPO-L-3_episode_length.png',
			alt: 'PPO League episode length by agent type',
			caption:
			"Average game length (moves) per epoch (1–102), 5-epoch centered rolling average, for Tactical (blue), Terminal (orange), and Aggressive \
			(red); dashed vertical at epoch 21 marks the curriculum switch from mid-game positions (4–9 pieces) to full 12v12. All three agents jump \
			from ~54 to ~72–73 moves at the switch, then diverge: Terminal peaks at ~122.5 moves around epoch 90 and remains ~116–117 by epochs \
			100–102, consistent with sparse outcome-only reward requiring extended positional play; Tactical plateaus at ~95–98 moves; Aggressive \
			is shortest at ~86–88 moves, driven by amplified capture rewards and a 2× time-penalty scale. The Terminal–Aggressive gap reaches ~36 \
			moves at peak and ~29 moves by epoch 100; benchmark self-play (terminal_vs_self = 145.2, tactical_vs_self = 112.6, aggressive_vs_self \
			= 96.8 at epoch 100) confirms the same ordering.",
		},
		{
			src: '/figures/ppo_league/PPO-L-5_reward_trajectories.png',
			alt: 'PPO League reward trajectories by agent type',
			caption:
			"Three vertically stacked subplots show average epoch reward (7-epoch rolling average over faint raw traces) for each agent across \
			epochs 1–100; y-axes are agent-specific and not comparable in magnitude. The dashed vertical at epoch 21 marks the curriculum switch \
			from mid-game starts to full 12v12 games, which produces a visible disruption in all three panels. Tactical rises steadily from ~25 to \
			~44–45, with a brief dip to ~20 just before the switch. Terminal, receiving only win/loss signal, traces the most volatile path: near \
			−2 to −5 in Phase 1, a sharp drop to ~−8 to −10 after the switch, recovery crossing zero around epoch 57–60, a peak near +9–10 around \
			epochs 75–80, then a partial decline to ~+5 by epoch 100. Aggressive, with amplified capture/king rewards and a time-pressure penalty, \
			shows the largest scale and smoothest growth, climbing from ~52–54 through ~92–93 by epoch 95 before plateauing. Together the panels \
			illustrate how reward shaping density governs both the magnitude and stability of the learning signal: dense shaping produces smooth, \
			high-magnitude returns, while sparse outcome-only rewards yield delayed, volatile improvement consistent with difficult long-horizon \
			credit assignment.",
			wide: true,
		},
	];

	// ── PPO Comparison ─────────────────────────────────────────────────────
	const ppoComparison: Fig = {
		src: '/figures/ppo_comparison/PPO-C-1_ppo_comparison.png',
		alt: 'PPO training comparison',
		caption:
		"Win rate vs. a fixed random-action opponent (%) evaluated every 10 epochs; dashed blue = solo Curriculum run (epochs 10–110), solid \
		lines = League agents Tactical (blue circles), Terminal (orange squares), Aggressive (red triangles) (epochs 10–100); dotted horizontals \
		at 50% (random baseline) and 100% (saturation). The solo curriculum run improves slowly and non-monotonically, peaking at 85.8% at epoch \
		100 before declining to 81.6% at epoch 110, consistent with passive co-evolution in single-agent self-play. League agents converge \
		substantially faster: Tactical reaches 99.4% by epoch 70, Terminal (98.6%) and Aggressive (97.8%) follow by epoch 80, and all three \
		saturate at 100% by epoch 80–100. Aggressive's early lead at epoch 10 (57.4% vs 45.2% curriculum) is consistent with higher-magnitude \
		shaping accelerating initial learning. Notably, Terminal saturates at 100% despite zero intermediate shaping rewards, suggesting that \
		cross-agent diversity — rather than reward richness alone — is a dominant contributor to the league's sample efficiency (supported by the \
		head-to-head and tie-rate figures).",
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
				Scalar Value Head
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
				WDL Value Head
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
				Scalar vs. WDL Comparison
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
				Curriculum + Self-Play
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
				League Training · Tactical · Terminal · Aggressive
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
