import type { ModelInfo, BoardResponse, AZStats, PPOStats, BenchmarkStats } from './types';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
	const res = await fetch(path, options);
	if (!res.ok) {
		const body = await res.text();
		let detail = body;
		try {
			detail = JSON.parse(body)?.detail ?? body;
		} catch {}
		throw new Error(detail || `HTTP ${res.status}`);
	}
	return res.json() as Promise<T>;
}

export const api = {
	getModels(): Promise<ModelInfo[]> {
		return request<ModelInfo[]>('/api/models');
	},

	newGame(modelId: string, simulations: number, humanColor: 'blue' | 'red'): Promise<BoardResponse> {
		return request<BoardResponse>('/api/new_game', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				model_id: modelId,
				simulations,
				human_color: humanColor,
			}),
		});
	},

	move(gameId: string, fromSq: number, toSq: number): Promise<BoardResponse> {
		return request<BoardResponse>('/api/move', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ game_id: gameId, from_sq: fromSq, to_sq: toSq }),
		});
	},

	statsAlphaZero(): Promise<AZStats> {
		return request<AZStats>('/api/training/alphazero');
	},

	statsPPO(): Promise<PPOStats> {
		return request<PPOStats>('/api/training/ppo');
	},

	statsBenchmarks(): Promise<BenchmarkStats> {
		return request<BenchmarkStats>('/api/training/benchmarks');
	},
};
