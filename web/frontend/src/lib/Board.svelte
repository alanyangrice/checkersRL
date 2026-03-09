<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import type { Board, Move, Color } from './types';

	export let board: Board = Array(4).fill(null).map(() => Array(8).fill(null).map(() => Array(8).fill(0)));
	export let legalMoves: Move[] = [];
	export let humanColor: Color = 'blue';
	export let isHumanTurn: boolean = true;
	export let disabled: boolean = false;
	/** Squares in the AI's last move path, e.g. [9, 14, 23] for "9x14x23" */
	export let aiMovePath: number[] = [];
	/** Board square of the piece locked into a capture chain (human mid-turn) */
	export let capturingSq: number | null = null;

	const dispatch = createEventDispatcher<{ move: Move }>();

	let canvas: HTMLCanvasElement;
	const SIZE = 480;

	// Position map: squares 1–32 → [row, col]
	const POS: [number, number][] = [
		[0,1],[0,3],[0,5],[0,7],
		[1,0],[1,2],[1,4],[1,6],
		[2,1],[2,3],[2,5],[2,7],
		[3,0],[3,2],[3,4],[3,6],
		[4,1],[4,3],[4,5],[4,7],
		[5,0],[5,2],[5,4],[5,6],
		[6,1],[6,3],[6,5],[6,7],
		[7,0],[7,2],[7,4],[7,6],
	];

	function sqToRC(sq: number): [number, number] { return POS[sq - 1]; }
	function rcToSq(r: number, c: number): number | null {
		const i = POS.findIndex(([pr, pc]) => pr === r && pc === c);
		return i === -1 ? null : i + 1;
	}
	function isDark(r: number, c: number) { return (r % 2 === 0) ? (c % 2 === 1) : (c % 2 === 0); }

	// When capturingSq changes (mid-chain), auto-select that piece
	let selectedSq: number | null = null;
	$: if (capturingSq !== null) selectedSq = capturingSq;
	$: if (capturingSq === null && !isHumanTurn) selectedSq = null;

	$: destinations = new Set(legalMoves.filter(m => m.from_sq === selectedSq).map(m => m.to_sq));
	$: movableSqs  = new Set(legalMoves.map(m => m.from_sq));

	// AI move path sets for highlighting
	$: aiFrom = aiMovePath.length > 0 ? aiMovePath[0] : null;
	$: aiTo   = aiMovePath.length > 0 ? aiMovePath[aiMovePath.length - 1] : null;
	$: aiMid  = new Set(aiMovePath.slice(1, -1));

	function pieceAt(r: number, c: number): { color: Color; king: boolean } | null {
		if (board[0][r][c]) return { color: 'blue', king: false };
		if (board[1][r][c]) return { color: 'blue', king: true };
		if (board[2][r][c]) return { color: 'red',  king: false };
		if (board[3][r][c]) return { color: 'red',  king: true };
		return null;
	}

	function draw(ctx: CanvasRenderingContext2D) {
		const cell = SIZE / 8;
		ctx.clearRect(0, 0, SIZE, SIZE);

		for (let r = 0; r < 8; r++) {
			for (let c = 0; c < 8; c++) {
				const dark = isDark(r, c);
				// Base square colour
				ctx.fillStyle = dark ? '#B58863' : '#F0D9B5';
				ctx.fillRect(c * cell, r * cell, cell, cell);

				if (!dark) continue;
				const sq = rcToSq(r, c)!;
				const x = c * cell, y = r * cell;

				// ── Highlights ──────────────────────────────────────────
				// Selected piece
				if (sq === selectedSq) {
					ctx.fillStyle = 'rgba(234,179,8,0.5)';
					ctx.fillRect(x, y, cell, cell);
				}
				// Legal destination dots
				if (destinations.has(sq)) {
					ctx.fillStyle = 'rgba(34,197,94,0.3)';
					ctx.fillRect(x, y, cell, cell);
					ctx.beginPath();
					ctx.arc(x + cell/2, y + cell/2, cell * 0.13, 0, Math.PI * 2);
					ctx.fillStyle = 'rgba(22,163,74,0.85)';
					ctx.fill();
				}
				// AI last-move: from square (orange tint)
				if (sq === aiFrom) {
					ctx.fillStyle = 'rgba(249,115,22,0.35)';
					ctx.fillRect(x, y, cell, cell);
				}
				// AI last-move: to square (green tint)
				if (sq === aiTo && sq !== aiFrom) {
					ctx.fillStyle = 'rgba(34,197,94,0.4)';
					ctx.fillRect(x, y, cell, cell);
				}
				// AI last-move: intermediate capture squares
				if (aiMid.has(sq)) {
					ctx.fillStyle = 'rgba(234,179,8,0.25)';
					ctx.fillRect(x, y, cell, cell);
				}
				// Movable piece indicator (subtle ring, no piece selected yet)
				if (selectedSq === null && movableSqs.has(sq) && isHumanTurn && !disabled) {
					const p = pieceAt(r, c);
					if (p && p.color === humanColor) {
						ctx.beginPath();
						ctx.arc(x + cell/2, y + cell/2, cell * 0.37, 0, Math.PI * 2);
						ctx.strokeStyle = 'rgba(234,179,8,0.55)';
						ctx.lineWidth = 2;
						ctx.stroke();
					}
				}
			}
		}

		// ── Pieces ────────────────────────────────────────────────────
		for (let r = 0; r < 8; r++) {
			for (let c = 0; c < 8; c++) {
				const p = pieceAt(r, c);
				if (!p) continue;
				const cx = c * (SIZE/8) + (SIZE/8)/2;
				const cy = r * (SIZE/8) + (SIZE/8)/2;
				const rad = (SIZE/8) * 0.37;
				const cell = SIZE / 8;

				// Subtle drop shadow (soft, flat — not offset)
				ctx.beginPath();
				ctx.arc(cx, cy + 1.5, rad, 0, Math.PI * 2);
				ctx.fillStyle = 'rgba(0,0,0,0.15)';
				ctx.fill();

				// Flat disc — solid colour with a very slight top-to-bottom tint
				const g = ctx.createLinearGradient(cx, cy - rad, cx, cy + rad);
				if (p.color === 'blue') {
					g.addColorStop(0, '#3B82F6'); g.addColorStop(1, '#2563EB');
				} else {
					g.addColorStop(0, '#EF4444'); g.addColorStop(1, '#DC2626');
				}
				ctx.beginPath();
				ctx.arc(cx, cy, rad, 0, Math.PI * 2);
				ctx.fillStyle = g;
				ctx.fill();

				// Outer rim
				ctx.beginPath();
				ctx.arc(cx, cy, rad, 0, Math.PI * 2);
				ctx.strokeStyle = p.color === 'blue' ? '#1D4ED8' : '#B91C1C';
				ctx.lineWidth = 2;
				ctx.stroke();

				// Inner ring — gives the recessed-edge texture of a real checker
				ctx.beginPath();
				ctx.arc(cx, cy, rad * 0.72, 0, Math.PI * 2);
				ctx.strokeStyle = p.color === 'blue' ? 'rgba(29,78,216,0.45)' : 'rgba(185,28,28,0.45)';
				ctx.lineWidth = 1;
				ctx.stroke();

				// King crown
				if (p.king) {
					ctx.font = `bold ${Math.round(cell * 0.55)}px serif`;
					ctx.textAlign = 'center';
					ctx.textBaseline = 'middle';
					ctx.fillStyle = 'rgba(255,255,255,0.9)';
					ctx.fillText('♛', cx, cy + 2);
				}
			}
		}

		// Border
		ctx.strokeStyle = '#9CA3AF';
		ctx.lineWidth = 1;
		ctx.strokeRect(0.5, 0.5, SIZE - 1, SIZE - 1);
	}

	function render() {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (ctx) draw(ctx);
	}

	$: board, legalMoves, selectedSq, isHumanTurn, disabled, aiMovePath, capturingSq, render();
	onMount(render);

	function handleClick(e: MouseEvent) {
		if (disabled) return;
		// During a capture chain, only destination clicks matter
		const canInteract = isHumanTurn || capturingSq !== null;
		if (!canInteract) return;

		const rect = canvas.getBoundingClientRect();
		const cell = SIZE / rect.width;
		const x = (e.clientX - rect.left) * cell;
		const y = (e.clientY - rect.top) * cell;
		const col = Math.floor(x / (SIZE / 8));
		const row = Math.floor(y / (SIZE / 8));
		if (row < 0 || row > 7 || col < 0 || col > 7 || !isDark(row, col)) {
			if (capturingSq === null) selectedSq = null;
			return;
		}
		const sq = rcToSq(row, col);
		if (!sq) return;

		// If this square is a legal destination from the selected piece → dispatch move
		if (selectedSq !== null && destinations.has(sq)) {
			dispatch('move', { from_sq: selectedSq, to_sq: sq });
			if (capturingSq === null) selectedSq = null;
			return;
		}

		// Can't change selection mid-capture-chain
		if (capturingSq !== null) return;

		// Select a movable piece of the human's colour
		const p = pieceAt(row, col);
		if (p && p.color === humanColor && movableSqs.has(sq)) {
			selectedSq = sq;
		} else {
			selectedSq = null;
		}
	}
</script>

<canvas
	bind:this={canvas}
	width={SIZE}
	height={SIZE}
	style="display:block; width:100%; height:100%;"
	class:cursor-pointer={isHumanTurn && !disabled}
	class:cursor-default={!isHumanTurn || disabled}
	on:click={handleClick}
	role="grid"
	aria-label="Checkers board"
/>
