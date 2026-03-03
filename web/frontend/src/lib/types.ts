export interface ModelInfo {
	id: string;
	label: string;
	supports_difficulty: boolean;
}

export interface Move {
	from_sq: number;
	to_sq: number;
}

/** Board is a 4×8×8 nested array (absolute coordinates).
 *  board[0][r][c] = 1 → blue regular piece at row r, col c
 *  board[1][r][c] = 1 → blue king
 *  board[2][r][c] = 1 → red regular piece
 *  board[3][r][c] = 1 → red king
 */
export type Board = number[][][];

export type Color = 'blue' | 'red';
export type Winner = 'blue' | 'red' | 'tie' | null;

export interface BoardResponse {
	game_id: string;
	board: Board;                        // after AI responded
	board_after_human: Board | null;     // after human's turn, before AI
	initial_board: Board | null;         // starting position (new_game only)
	initial_value: number | null;        // eval of starting position
	post_ai_value: number | null;        // eval after AI moves (AI perspective)
	ai_boards: Board[] | null;           // board after each AI hop (for animation)
	turn: Color;
	legal_moves: Move[];
	done: boolean;
	winner: Winner;
	value: number | null;
	ai_move: string | null;
	turn_complete: boolean;
	human_color: Color;
	capturing_sq: number | null;  // which square is mid-capture-chain (human)
}

// Stats shapes
export interface AZStats {
	epochs: number[];
	policy_loss: number[];
	value_loss: number[];
	total_loss: number[];
	win_rate: number[];
	tie_rate: number[];
	loss_rate: number[];
	entropy: number[];
	avg_moves: number[];
}

export interface PPOStats {
	epochs: number[];
	avg_reward: number[];
	win_rate_blue: number[];
	win_rate_red: number[];
	tie_rate: number[];
	avg_ep_length: number[];
}

export interface BenchmarkStats {
	alphazero: {
		epochs: number[];
		gate_win_rate: number[];
		gate_score: number[];
		gate_accepted: boolean[];
		vs_random_score: number[];
	};
	ppo_league: {
		epoch?: number;
		tactical_vs_random?: number;
		terminal_vs_random?: number;
		aggressive_vs_random?: number;
		tactical_vs_terminal?: number;
		tactical_vs_aggressive?: number;
		terminal_vs_aggressive?: number;
		series?: {
			epochs: number[];
			tactical_vs_random: number[];
			terminal_vs_random: number[];
			aggressive_vs_random: number[];
		};
	};
}
