<script lang="ts">
	import { onMount } from 'svelte';
	import Board from '$lib/Board.svelte';
	import { api } from '$lib/api';
	import type { ModelInfo, BoardResponse, Color, Winner } from '$lib/types';

	// ── Setup ──────────────────────────────────────────────────────────────
	let models: ModelInfo[] = [];
	let selectedModelId = '';
	let simulations = 100;
	let humanColor: Color = 'blue';

	// ── Live game state ────────────────────────────────────────────────────
	let gameId: string | null = null;
	let boardState: number[][][] = Array(4).fill(null).map(() => Array(8).fill(null).map(() => Array(8).fill(0)));
	let legalMoves: { from_sq: number; to_sq: number }[] = [];
	let currentTurn: Color = 'blue';
	let done = false;
	let winner: Winner = null;
	let turnComplete = true;
	let capturingSq: number | null = null;
	let thinking = false;
	let error = '';
	let aiValue: number | null = null;
	let aiMovePath: number[] = [];
	let lastAiMove: string | null = null;
	let notification = '';
	let notifTimeout: ReturnType<typeof setTimeout>;
	let rulesOpen = false;
	let showEndOverlay = false; // only true after all animations finish

	// ── History ────────────────────────────────────────────────────────────
	interface Snapshot {
		board: number[][][];
		aiMovePath: number[];
		notation: string;
		value: number | null;
		by: 'human' | 'ai';
	}
	let snapshots: Snapshot[] = [];
	let viewIndex: number | null = null; // null = live

	// null = unset (will be treated as last); treat last snapshot as "present"
	$: atLive = viewIndex === null || viewIndex === snapshots.length - 1;
	$: effectiveIndex = viewIndex ?? snapshots.length - 1;

	// Animation state (overrides snapshot display during AI multi-capture)
	let animating = false;
	let animBoard: number[][][] | null = null;
	let animPath: number[] = [];

	$: displayBoard    = animating ? (animBoard ?? boardState) : (snapshots.length > 0 ? snapshots[effectiveIndex].board      : boardState);
	$: displayAiPath   = animating ? animPath                  : (snapshots.length > 0 ? snapshots[effectiveIndex].aiMovePath : aiMovePath);
	$: displayValue    = snapshots.length > 0 ? snapshots[effectiveIndex].value      : aiValue;
	$: displayNotation = snapshots.length > 0 ? snapshots[effectiveIndex].notation   : lastAiMove;

	function sleep(ms: number) { return new Promise<void>(r => setTimeout(r, ms)); }

	async function animateHops(hopBoards: Board[], fullPath: number[]) {
		if (hopBoards.length <= 1) return; // single move, no animation needed
		animating = true;
		for (let i = 0; i < hopBoards.length; i++) {
			animBoard = hopBoards[i];
			animPath  = fullPath.slice(0, i + 2); // orange→from, green→current hop
			if (i < hopBoards.length - 1) await sleep(500);
		}
		animating = false;
		animBoard = null;
		animPath  = [];
	}

	function stepBack()    {
		const cur = viewIndex ?? snapshots.length - 1;
		if (cur > 0) viewIndex = cur - 1;
	}
	function stepForward() {
		if (viewIndex === null) return;
		viewIndex = viewIndex < snapshots.length - 1 ? viewIndex + 1 : null;
	}

	function onKey(e: KeyboardEvent) {
		if (!gameId) return;
		if (e.key === 'ArrowLeft')  { e.preventDefault(); stepBack(); }
		if (e.key === 'ArrowRight') { e.preventDefault(); stepForward(); }
	}

	// ── Square → [row, col] — needed to detect captures (row diff == 2) ───
	const SQ_POS: [number, number][] = [
		[0,1],[0,3],[0,5],[0,7],[1,0],[1,2],[1,4],[1,6],[2,1],[2,3],[2,5],[2,7],
		[3,0],[3,2],[3,4],[3,6],[4,1],[4,3],[4,5],[4,7],[5,0],[5,2],[5,4],[5,6],
		[6,1],[6,3],[6,5],[6,7],[7,0],[7,2],[7,4],[7,6],
	];
	function isCapture(from_sq: number, to_sq: number): boolean {
		return Math.abs(SQ_POS[from_sq - 1][0] - SQ_POS[to_sq - 1][0]) === 2;
	}

	// Track the human's turn for snapshotting
	let preMoveBoard: number[][][] | null = null; // board before human's first step
	let humanPath: number[] = [];                 // squares visited this turn [from, hop1, hop2,…]
	let humanTurnIsCapture = false;

	// ── Derived display ────────────────────────────────────────────────────
	$: isHumanTurn = gameId !== null && !done && turnComplete && currentTurn === humanColor && atLive;
	$: supportsDifficulty = models.find((m) => m.id === selectedModelId)?.supports_difficulty ?? true;

	function parsePath(mv: string): number[] {
		const sep = mv.includes('x') ? 'x' : '-';
		return mv.split(sep).map(Number).filter(n => !isNaN(n));
	}

	function evalString(v: number): string {
		const human = -v;
		const sign = human >= 0 ? '+' : '−';
		return `${sign}${Math.abs(human).toFixed(2)}`;
	}
	$: evalColor = displayValue === null ? '' : displayValue < -0.15 ? 'text-blue-600' : displayValue > 0.15 ? 'text-red-600' : 'text-gray-500';
	$: evalWho   = displayValue === null ? '' : displayValue < -0.15 ? 'you' : displayValue > 0.15 ? 'ai' : 'even';

	$: historyLabel = (() => {
		if (!snapshots.length) return '';
		const i = viewIndex ?? snapshots.length - 1;
		return `${i + 1} / ${snapshots.length}`;
	})();

	function notify(msg: string) {
		notification = msg;
		clearTimeout(notifTimeout);
		notifTimeout = setTimeout(() => (notification = ''), 4000);
	}

	// ── Lifecycle ──────────────────────────────────────────────────────────
	onMount(async () => {
		try {
			models = await api.getModels();
			if (models.length > 0) selectedModelId = models[0].id;
		} catch {
			error = 'Could not reach the server. Is the backend running?';
		}
	});

	// ── Actions ────────────────────────────────────────────────────────────
	async function startGame() {
		error = '';
		thinking = true;
		snapshots = [];
		viewIndex = null;
		aiValue = null;
		aiMovePath = [];
		lastAiMove = null;
		done = false;
		winner = null;
		notification = '';
		capturingSq = null;
		preMoveBoard = null;
		humanPath = [];
		showEndOverlay = false;

		try {
			const resp = await api.newGame(selectedModelId, simulations, humanColor);
			applyLiveState(resp);

			// Always start history with the default board position + its eval
			const startSnap: Snapshot = {
				board:       resp.initial_board ?? resp.board,
				aiMovePath:  [],
				notation:    'Start',
				value:       resp.initial_value,
				by:          'human',
			};

			if (resp.ai_move) {
				// Human plays Red → AI moved first; brief pause then animate
				await sleep(500);
				const fullPath = parsePath(resp.ai_move);
				if (resp.ai_boards && resp.ai_boards.length > 1) {
					await animateHops(resp.ai_boards as number[][][], fullPath);
				}
				snapshots = [
					startSnap,
					{
						board:       resp.board,
						aiMovePath:  fullPath,
						notation:    `AI: ${resp.ai_move}`,
						value:       resp.post_ai_value,
						by:          'ai',
					},
				];
				notify(`AI played ${resp.ai_move}`);
			} else {
				snapshots = [startSnap];
			}
		} catch (e: unknown) {
			error = e instanceof Error ? e.message : 'Failed to start game.';
		} finally {
			thinking = false;
		}
	}

	async function handleMove(event: CustomEvent<{ from_sq: number; to_sq: number }>) {
		if (!gameId || thinking || !atLive) return;
		const { from_sq, to_sq } = event.detail;

		// Start of human turn (not continuing a capture chain)
		if (capturingSq === null) {
			preMoveBoard = boardState;
			humanPath = [from_sq, to_sq];
			humanTurnIsCapture = isCapture(from_sq, to_sq);
		} else {
			// Continuing a capture chain — extend the path
			humanPath = [...humanPath, to_sq];
			humanTurnIsCapture = true;
		}

		thinking = true;
		error = '';
		try {
			const resp = await api.move(gameId, from_sq, to_sq);
			applyLiveState(resp);

			if (resp.turn_complete) {
				const sep = humanTurnIsCapture ? 'x' : '-';
				const humanNotation = humanPath.join(sep);

				// 1. Human snapshot
				if (resp.board_after_human) {
					snapshots = [...snapshots, {
						board:       resp.board_after_human,
						aiMovePath:  [...humanPath],
						notation:    `You: ${humanNotation}`,
						value:       resp.value,
						by:          'human',
					}];
				}

				// 2. Brief pause so the human's move is visible before the AI responds
				await sleep(500);

				// 3. Animate AI multi-capture hops, then push AI snapshot
				if (resp.ai_move) {
					const fullPath = parsePath(resp.ai_move);
					if (resp.ai_boards && resp.ai_boards.length > 1) {
						await animateHops(resp.ai_boards as number[][][], fullPath);
					}
					snapshots = [...snapshots, {
						board:       resp.board,
						aiMovePath:  fullPath,
						notation:    `AI: ${resp.ai_move}`,
						value:       resp.post_ai_value,
						by:          'ai',
					}];
					lastAiMove = resp.ai_move;
					aiMovePath = fullPath;
					notify(resp.done ? `AI played ${resp.ai_move} — ${winnerText(resp.winner)}` : `AI played ${resp.ai_move}`);
				} else if (resp.done) {
					notify(winnerText(resp.winner));
					showEndOverlay = true;
				}

				preMoveBoard = null;
				humanPath = [];
				viewIndex = null;
				if (resp.done) showEndOverlay = true;
			}
		} catch (e: unknown) {
			error = e instanceof Error ? e.message : 'Move failed.';
		} finally {
			thinking = false;
		}
	}

	/** Apply server response to live state only — no snapshot logic here */
	function applyLiveState(resp: BoardResponse) {
		boardState   = resp.board;
		legalMoves   = resp.legal_moves;
		currentTurn  = resp.turn;
		done         = resp.done;
		winner       = resp.winner;
		turnComplete = resp.turn_complete;
		humanColor   = resp.human_color;
		capturingSq  = resp.capturing_sq;
		gameId       = resp.game_id;
		if (resp.value !== null) aiValue = resp.value;
		if (!resp.turn_complete) aiMovePath = []; // clear AI highlight mid-chain
	}

	function winnerText(w: Winner): string {
		if (w === 'blue') return humanColor === 'blue' ? 'You win!' : 'Blue wins';
		if (w === 'red')  return humanColor === 'red'  ? 'You win!' : 'Red wins';
		if (w === 'tie')  return 'Draw';
		return '';
	}
</script>

<svelte:head><title>CheckersRL</title></svelte:head>
<svelte:window on:keydown={onKey} />

<div class="flex flex-col xl:flex-row gap-6 xl:gap-8 items-center xl:items-start justify-center">

	<!-- ── Board column ────────────────────────────────────────────────── -->
	<div class="flex flex-col gap-2 w-full xl:flex-shrink-0 mx-auto xl:mx-0" style="max-width: min(480px, 100%)">
		<div class="relative border border-gray-200 rounded" style="width:100%; aspect-ratio:1">
			<Board
				board={displayBoard}
				legalMoves={atLive ? legalMoves : []}
				{humanColor}
				{isHumanTurn}
				aiMovePath={displayAiPath}
				{capturingSq}
				disabled={thinking || done || !gameId || !atLive}
				on:move={handleMove}
			/>
			{#if showEndOverlay && winner && atLive}
				<div class="absolute inset-0 bg-white/85 flex flex-col items-center justify-center gap-3 rounded">
					<p class="text-xl font-medium">{winnerText(winner)}</p>
					<button class="btn-primary" on:click={startGame}>Play again</button>
				</div>
			{/if}
			{#if !gameId}
				<div class="absolute inset-0 bg-white/50 flex items-center justify-center rounded text-sm text-gray-400">
					Set up a game and click Start
				</div>
			{/if}
			{#if viewIndex !== null && viewIndex < snapshots.length - 1}
				<div class="absolute top-2 left-2 bg-white/90 border border-gray-200 rounded px-2 py-0.5 text-xs text-gray-500">
					viewing history
				</div>
			{/if}
		</div>

		<!-- History navigator -->
		{#if snapshots.length > 0}
			<div class="flex items-center gap-2">
				<button class="btn-secondary btn-sm px-2.5 disabled:opacity-30" on:click={stepBack}
					disabled={effectiveIndex === 0} title="Previous (←)">←</button>
				<span class="text-xs text-gray-400 font-mono flex-1 text-center">{historyLabel}</span>
				<button class="btn-secondary btn-sm px-2.5 disabled:opacity-30" on:click={stepForward}
					disabled={atLive} title="Next (→)">→</button>
			</div>
		{/if}

		<!-- Notification & legend -->
		<div class="flex items-center justify-between text-xs text-gray-500 min-h-[1.25rem]">
			<span>{atLive ? notification : (displayNotation ?? '')}</span>
			{#if gameId}
				<div class="flex gap-3">
					<span><span class="inline-block w-2 h-2 rounded-full bg-blue-600 mr-1 align-middle"/>Blue — {humanColor === 'blue' ? 'You' : 'AI'}</span>
					<span><span class="inline-block w-2 h-2 rounded-full bg-red-700 mr-1 align-middle"/>Red — {humanColor === 'red' ? 'You' : 'AI'}</span>
				</div>
			{/if}
		</div>
	</div>

	<!-- ── Controls column ─────────────────────────────────────────────── -->
	<div class="flex flex-col gap-5 w-full xl:w-56 xl:flex-shrink-0 max-w-sm xl:max-w-none mx-auto xl:mx-0">

		<!-- Setup -->
		<div class="flex flex-col gap-3">
			<div class="flex flex-col gap-1">
				<label class="section-label" for="model-select">Model</label>
				{#if models.length === 0}
					<p class="text-sm text-gray-400">Loading…</p>
				{:else}
					<select id="model-select" bind:value={selectedModelId}
						class="border border-gray-300 rounded px-2.5 py-1.5 text-sm text-gray-900 bg-white w-full focus:outline-none focus:border-gray-500">
						{#each models as m}
							<option value={m.id}>{m.label}</option>
						{/each}
					</select>
				{/if}
			</div>

			{#if supportsDifficulty}
				<div class="flex flex-col gap-1">
					<div class="flex justify-between">
						<label class="section-label" for="sims">Difficulty</label>
						<span class="text-xs text-gray-400">{simulations} sims</span>
					</div>
					<input id="sims" type="range" min="25" max="200" step="25" bind:value={simulations} class="w-full accent-gray-800"/>
					<div class="flex justify-between text-xs text-gray-400"><span>Easy</span><span>Hard</span></div>
				</div>
			{/if}

			<div class="flex flex-col gap-1">
				<span class="section-label">Your color</span>
				<div class="flex gap-2">
					<button on:click={() => humanColor = 'blue'}
						class="flex-1 text-sm py-1 border rounded {humanColor === 'blue' ? 'border-gray-900 bg-gray-900 text-white' : 'border-gray-300 text-gray-600 hover:border-gray-500'}">Blue</button>
					<button on:click={() => humanColor = 'red'}
						class="flex-1 text-sm py-1 border rounded {humanColor === 'red' ? 'border-gray-900 bg-gray-900 text-white' : 'border-gray-300 text-gray-600 hover:border-gray-500'}">Red</button>
				</div>
			</div>

			<button class="btn-primary w-full" on:click={startGame} disabled={models.length === 0 || thinking}>
				{gameId ? 'New game' : 'Start game'}
			</button>
		</div>

		<!-- Eval -->
		{#if gameId && displayValue !== null}
			<div class="flex flex-col gap-0.5">
				<span class="section-label">Evaluation
					{#if !atLive}<span class="text-gray-400 normal-case font-normal">(move {(viewIndex ?? snapshots.length - 1) + 1})</span>{/if}
				</span>
				<div class="flex items-baseline gap-2">
					<span class="text-2xl font-mono font-medium {evalColor}">{evalString(displayValue)}</span>
					<span class="text-xs text-gray-400">
						{#if evalWho === 'you'}your favor{:else if evalWho === 'ai'}AI's favor{:else}equal{/if}
					</span>
				</div>
				{#if displayNotation}
					<p class="text-xs text-gray-400 font-mono mt-0.5">{displayNotation}</p>
				{/if}
			</div>
		{/if}

		<!-- Move log -->
		{#if snapshots.length > 0}
			<div class="flex flex-col gap-1">
				<span class="section-label">Move log</span>
				<div class="max-h-40 overflow-y-auto flex flex-col gap-0.5 text-xs font-mono">
					{#each snapshots as snap, i}
						<button
							class="text-left px-1 rounded transition-colors {
								effectiveIndex === i
									? 'bg-gray-100 text-gray-900'
									: snap.by === 'human' ? 'text-gray-400 hover:text-gray-700' : 'text-gray-600 hover:text-gray-900'
							}"
							on:click={() => viewIndex = i === snapshots.length - 1 ? null : i}
						>
							{i + 1}. {snap.notation}{#if snap.value !== null} <span class="text-gray-400">({evalString(snap.value)})</span>{/if}
						</button>
					{/each}
				</div>
			</div>
		{/if}

		<!-- Error -->
		{#if error}
			<p class="text-sm text-red-600">{error}</p>
		{/if}

		<!-- Rules -->
		<div class="border-t border-gray-100 pt-4">
			<button class="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 w-full text-left"
				on:click={() => rulesOpen = !rulesOpen}>
				<span class="transition-transform {rulesOpen ? 'rotate-90' : ''} inline-block">▶</span>
				Rules
			</button>
			{#if rulesOpen}
				<ul class="mt-2 text-xs text-gray-500 space-y-1 pl-3">
					<li>Blue moves first. Pieces move diagonally forward one square.</li>
					<li>Captures are mandatory — you must jump if able.</li>
					<li>Multi-jump: if a capture lands where another jump is possible, you must continue.</li>
					<li>A piece reaching the far row becomes a king and can move in any diagonal direction.</li>
					<li>If you have no legal moves, you lose.</li>
					<li><strong class="text-gray-600">Tie conditions:</strong></li>
					<li class="pl-2">250 total moves played.</li>
					<li class="pl-2">40 consecutive moves without a capture or promotion.</li>
					<li class="pl-2">Same position repeated 3 times.</li>
				</ul>
			{/if}
		</div>

		{#if !gameId}
			<div class="text-sm text-gray-500 space-y-1">
				<p>Select a model and click <strong class="text-gray-800">Start game</strong>.</p>
				<p>Click a piece to select it, then click a highlighted square to move.</p>
				<p>Blue starts at the top of the board.</p>
			</div>
		{/if}
	</div>

</div>
