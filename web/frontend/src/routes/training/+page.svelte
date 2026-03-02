<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { Chart, registerables } from 'chart.js';
	import { api } from '$lib/api';
	import type { AZStats, PPOStats, BenchmarkStats } from '$lib/types';

	Chart.register(...registerables);

	// Chart canvas refs
	let azLossCanvas: HTMLCanvasElement;
	let azWinCanvas: HTMLCanvasElement;
	let azEntropyCanvas: HTMLCanvasElement;
	let azGateCanvas: HTMLCanvasElement;
	let ppoRewardCanvas: HTMLCanvasElement;
	let ppoWinCanvas: HTMLCanvasElement;
	let leagueSeriesCanvas: HTMLCanvasElement;

	let charts: Chart[] = [];
	let loading = true;
	let error = '';

	const CHART_DEFAULTS = {
		responsive: true,
		maintainAspectRatio: false,
		animation: { duration: 400 },
		plugins: {
			legend: {
				labels: { color: '#374151', font: { size: 11 } },
			},
			tooltip: {
				backgroundColor: '#fff',
				borderColor: '#E5E7EB',
				borderWidth: 1,
				titleColor: '#111827',
				bodyColor: '#6B7280',
			},
		},
		scales: {
			x: {
				ticks: { color: '#9CA3AF', font: { size: 10 } },
				grid: { color: '#F3F4F6' },
			},
			y: {
				ticks: { color: '#9CA3AF', font: { size: 10 } },
				grid: { color: '#F3F4F6' },
			},
		},
	} as const;

	function destroyCharts() {
		charts.forEach((c) => c.destroy());
		charts = [];
	}

	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	function makeChart(canvas: HTMLCanvasElement, config: any): Chart {
		const chart = new Chart(canvas, config);
		charts.push(chart);
		return chart;
	}

	onMount(async () => {
		try {
			const [az, ppo, bench] = await Promise.all([
				api.statsAlphaZero(),
				api.statsPPO(),
				api.statsBenchmarks(),
			]);
			buildCharts(az, ppo, bench);
		} catch (e: unknown) {
			error = e instanceof Error ? e.message : 'Failed to load stats.';
		} finally {
			loading = false;
		}
	});

	onDestroy(destroyCharts);

	function buildCharts(az: AZStats, ppo: PPOStats, bench: BenchmarkStats) {
		// 1. AZ Loss curves
		makeChart(azLossCanvas, {
			type: 'line',
			data: {
				labels: az.epochs,
				datasets: [
					{
						label: 'Policy Loss',
						data: az.policy_loss,
						borderColor: '#60A5FA',
						backgroundColor: 'rgba(96,165,250,0.1)',
						tension: 0.3,
						pointRadius: 0,
						borderWidth: 2,
					},
					{
						label: 'Value Loss',
						data: az.value_loss,
						borderColor: '#F87171',
						backgroundColor: 'rgba(248,113,113,0.1)',
						tension: 0.3,
						pointRadius: 0,
						borderWidth: 2,
					},
				],
			},
			options: {
				...CHART_DEFAULTS,
				plugins: { ...CHART_DEFAULTS.plugins, legend: { ...CHART_DEFAULTS.plugins.legend } },
				scales: {
					...CHART_DEFAULTS.scales,
					y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: 'Loss', color: '#6B7280' } },
					x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: 'Epoch', color: '#6B7280' } },
				},
			},
		});

		// 2. AZ Win / tie / loss rates
		makeChart(azWinCanvas, {
			type: 'line',
			data: {
				labels: az.epochs,
				datasets: [
					{
						label: 'Blue Win Rate',
						data: az.win_rate,
						borderColor: '#34D399',
						tension: 0.3,
						pointRadius: 0,
						borderWidth: 2,
						fill: false,
					},
					{
						label: 'Tie Rate',
						data: az.tie_rate,
						borderColor: '#FBBF24',
						tension: 0.3,
						pointRadius: 0,
						borderWidth: 2,
						fill: false,
					},
					{
						label: 'Red Win Rate',
						data: az.loss_rate,
						borderColor: '#F87171',
						tension: 0.3,
						pointRadius: 0,
						borderWidth: 2,
						fill: false,
					},
				],
			},
			options: {
				...CHART_DEFAULTS,
				scales: {
					...CHART_DEFAULTS.scales,
					y: {
						...CHART_DEFAULTS.scales.y,
						min: 0,
						max: 1,
						title: { display: true, text: 'Rate', color: '#6B7280' },
					},
					x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: 'Epoch', color: '#6B7280' } },
				},
			},
		});

		// 3. Policy entropy
		makeChart(azEntropyCanvas, {
			type: 'line',
			data: {
				labels: az.epochs,
				datasets: [
					{
						label: 'Policy Entropy (nats)',
						data: az.entropy,
						borderColor: '#A78BFA',
						backgroundColor: 'rgba(167,139,250,0.1)',
						tension: 0.3,
						pointRadius: 0,
						borderWidth: 2,
					},
				],
			},
			options: {
				...CHART_DEFAULTS,
				scales: {
					...CHART_DEFAULTS.scales,
					y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: 'Entropy (nats)', color: '#6B7280' } },
					x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: 'Epoch', color: '#6B7280' } },
				},
			},
		});

		// 4. AZ Gate win rate + accepted epochs
		const gateAcceptedEpochs = bench.alphazero.epochs.filter(
			(_, i) => bench.alphazero.gate_accepted[i]
		);
		makeChart(azGateCanvas, {
			type: 'line',
			data: {
				labels: bench.alphazero.epochs,
				datasets: [
					{
						label: 'Gate Win Rate vs Prev',
						data: bench.alphazero.gate_win_rate,
						borderColor: '#34D399',
						tension: 0.2,
						pointRadius: 3,
						borderWidth: 2,
						fill: false,
					},
					{
						label: 'vs Random Score',
						data: bench.alphazero.vs_random_score,
						borderColor: '#60A5FA',
						tension: 0.2,
						pointRadius: 3,
						borderWidth: 2,
						fill: false,
					},
				],
			},
			options: {
				...CHART_DEFAULTS,
				scales: {
					...CHART_DEFAULTS.scales,
					y: {
						...CHART_DEFAULTS.scales.y,
						min: 0,
						max: 1,
						title: { display: true, text: 'Score / Rate', color: '#6B7280' },
					},
					x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: 'Eval Epoch', color: '#6B7280' } },
				},
				plugins: {
					...CHART_DEFAULTS.plugins,
					annotation: undefined,
					tooltip: {
						...CHART_DEFAULTS.plugins.tooltip,
						callbacks: {
							afterLabel: (ctx: { dataIndex: number }) => {
								const ep = bench.alphazero.epochs[ctx.dataIndex];
								return gateAcceptedEpochs.includes(ep) ? '✓ Model accepted' : '✗ Rejected';
							},
						},
					},
				},
			},
		});

		// 5. PPO reward curve
		makeChart(ppoRewardCanvas, {
			type: 'line',
			data: {
				labels: ppo.epochs,
				datasets: [
					{
						label: 'Avg Epoch Reward',
						data: ppo.avg_reward,
						borderColor: '#60A5FA',
						backgroundColor: 'rgba(96,165,250,0.08)',
						tension: 0.3,
						pointRadius: 0,
						borderWidth: 2,
					},
				],
			},
			options: {
				...CHART_DEFAULTS,
				scales: {
					...CHART_DEFAULTS.scales,
					y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: 'Reward', color: '#6B7280' } },
					x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: 'Epoch', color: '#6B7280' } },
				},
			},
		});

		// 6. PPO win rates
		makeChart(ppoWinCanvas, {
			type: 'line',
			data: {
				labels: ppo.epochs,
				datasets: [
					{
						label: 'Blue Win Rate',
						data: ppo.win_rate_blue,
						borderColor: '#60A5FA',
						tension: 0.3,
						pointRadius: 0,
						borderWidth: 2,
						fill: false,
					},
					{
						label: 'Tie Rate',
						data: ppo.tie_rate,
						borderColor: '#FBBF24',
						tension: 0.3,
						pointRadius: 0,
						borderWidth: 2,
						fill: false,
					},
				],
			},
			options: {
				...CHART_DEFAULTS,
				scales: {
					...CHART_DEFAULTS.scales,
					y: { ...CHART_DEFAULTS.scales.y, min: 0, max: 1, title: { display: true, text: 'Rate', color: '#6B7280' } },
					x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: 'Epoch', color: '#6B7280' } },
				},
			},
		});

		// 7. PPO League win rate series vs random
		if (bench.ppo_league?.series) {
			const s = bench.ppo_league.series;
			makeChart(leagueSeriesCanvas, {
				type: 'line',
				data: {
					labels: s.epochs,
					datasets: [
						{
							label: 'Tactical vs Random',
							data: s.tactical_vs_random,
							borderColor: '#34D399',
							tension: 0.2,
							pointRadius: 2,
							borderWidth: 2,
							fill: false,
						},
						{
							label: 'Terminal vs Random',
							data: s.terminal_vs_random,
							borderColor: '#60A5FA',
							tension: 0.2,
							pointRadius: 2,
							borderWidth: 2,
							fill: false,
						},
						{
							label: 'Aggressive vs Random',
							data: s.aggressive_vs_random,
							borderColor: '#F87171',
							tension: 0.2,
							pointRadius: 2,
							borderWidth: 2,
							fill: false,
						},
					],
				},
				options: {
					...CHART_DEFAULTS,
					scales: {
						...CHART_DEFAULTS.scales,
						y: { ...CHART_DEFAULTS.scales.y, min: 0, max: 1, title: { display: true, text: 'Win Rate', color: '#6B7280' } },
						x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: 'Epoch', color: '#6B7280' } },
					},
				},
			});
		}
	}
</script>

<svelte:head>
	<title>CheckersRL — Training Results</title>
</svelte:head>

{#if loading}
	<p class="text-sm text-gray-400">Loading training data…</p>
{:else if error}
	<p class="text-sm text-red-600">{error}</p>
{:else}
	<div class="flex flex-col gap-10">

		<section class="flex flex-col gap-4">
			<div>
				<h2 class="text-lg font-medium text-gray-900">AlphaZero</h2>
				<p class="text-sm text-gray-500 mt-1">Trained via self-play MCTS. 5 residual blocks × 256 channels, Win/Draw/Loss value head.</p>
			</div>
			<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
				<div class="border border-gray-200 rounded p-4">
					<p class="text-sm text-gray-700 mb-3">Training loss</p>
					<div class="h-48"><canvas bind:this={azLossCanvas} /></div>
				</div>
				<div class="border border-gray-200 rounded p-4">
					<p class="text-sm text-gray-700 mb-3">Self-play win / tie / loss rates</p>
					<div class="h-48"><canvas bind:this={azWinCanvas} /></div>
				</div>
				<div class="border border-gray-200 rounded p-4">
					<p class="text-sm text-gray-700 mb-1">Policy entropy</p>
					<p class="text-xs text-gray-400 mb-3">Higher = more exploratory self-play</p>
					<div class="h-44"><canvas bind:this={azEntropyCanvas} /></div>
				</div>
				<div class="border border-gray-200 rounded p-4">
					<p class="text-sm text-gray-700 mb-1">Evaluation benchmarks</p>
					<p class="text-xs text-gray-400 mb-3">Gate win rate vs previous best; vs-random score</p>
					<div class="h-44"><canvas bind:this={azGateCanvas} /></div>
				</div>
			</div>
		</section>

		<section class="flex flex-col gap-4">
			<div>
				<h2 class="text-lg font-medium text-gray-900">PPO League</h2>
				<p class="text-sm text-gray-500 mt-1">Three specialised agents (Tactical, Terminal, Aggressive) trained with PPO against a self-play opponent pool.</p>
			</div>
			<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
				<div class="border border-gray-200 rounded p-4">
					<p class="text-sm text-gray-700 mb-3">Average epoch reward</p>
					<div class="h-48"><canvas bind:this={ppoRewardCanvas} /></div>
				</div>
				<div class="border border-gray-200 rounded p-4">
					<p class="text-sm text-gray-700 mb-3">Win rates over training</p>
					<div class="h-48"><canvas bind:this={ppoWinCanvas} /></div>
				</div>
				<div class="border border-gray-200 rounded p-4 md:col-span-2">
					<p class="text-sm text-gray-700 mb-1">League vs random opponent</p>
					<p class="text-xs text-gray-400 mb-3">Strength progression across all three agents</p>
					<div class="h-48"><canvas bind:this={leagueSeriesCanvas} /></div>
				</div>
			</div>
		</section>

	</div>
{/if}
