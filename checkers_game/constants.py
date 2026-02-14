import pygame

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 128, 0)
YELLOW = (255, 255, 0)
GRAY = (50, 50, 50)

# Constants
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

# Fonts
pygame.font.init()
font = pygame.font.SysFont('Arial', 24)
large_font = pygame.font.SysFont('Arial', 48, bold=True)
medium_font = pygame.font.SysFont('Arial', 32)

# Board position mapping (1-32 numbering for dark squares)
_POSITION_MAP = [
    (0, 1), (0, 3), (0, 5), (0, 7),
    (1, 0), (1, 2), (1, 4), (1, 6),
    (2, 1), (2, 3), (2, 5), (2, 7),
    (3, 0), (3, 2), (3, 4), (3, 6),
    (4, 1), (4, 3), (4, 5), (4, 7),
    (5, 0), (5, 2), (5, 4), (5, 6),
    (6, 1), (6, 3), (6, 5), (6, 7),
    (7, 0), (7, 2), (7, 4), (7, 6)
]

# Helper functions
def board_number_to_position(num):
    return _POSITION_MAP[num - 1]

def position_to_board_number(row, col):
    if (row, col) in _POSITION_MAP:
        return _POSITION_MAP.index((row, col)) + 1
    else:
        return None


# ---------------------------------------------------------------------------
# Semantic action space: pre-computed (from_sq, to_sq) single-step pairs
# ---------------------------------------------------------------------------
NUM_SQUARES = 32

def _build_action_table():
    """Pre-compute all geometrically valid single-step (from_sq, to_sq) pairs.

    Includes:
      - Regular moves: 1 diagonal step (|dr|=1, |dc|=1)
      - Capture landings: 2 diagonal steps (|dr|=2, |dc|=2)

    Returns a deduplicated list of (from_sq, to_sq) tuples.
    """
    seen = set()
    actions = []

    for from_sq in range(1, NUM_SQUARES + 1):
        from_row, from_col = board_number_to_position(from_sq)
        for d_row in [-1, 1]:
            for d_col in [-1, 1]:
                # Regular move targets (1 step diagonal)
                r1, c1 = from_row + d_row, from_col + d_col
                if 0 <= r1 < ROWS and 0 <= c1 < COLS:
                    to_sq = position_to_board_number(r1, c1)
                    if to_sq is not None and (from_sq, to_sq) not in seen:
                        seen.add((from_sq, to_sq))
                        actions.append((from_sq, to_sq))

                # Capture landing targets (2 step diagonal)
                r2, c2 = from_row + 2 * d_row, from_col + 2 * d_col
                if 0 <= r2 < ROWS and 0 <= c2 < COLS:
                    to_sq = position_to_board_number(r2, c2)
                    if to_sq is not None and (from_sq, to_sq) not in seen:
                        seen.add((from_sq, to_sq))
                        actions.append((from_sq, to_sq))

    return actions


# Built once at import time
ALL_ACTIONS = _build_action_table()
ACTION_TO_INDEX = {pair: i for i, pair in enumerate(ALL_ACTIONS)}
NUM_ACTIONS = len(ALL_ACTIONS)


def encode_action(from_sq, to_sq):
    """Convert a (from_square, to_square) pair to its fixed action index.
    Returns -1 if the pair is not a valid single-step action."""
    return ACTION_TO_INDEX.get((from_sq, to_sq), -1)


def decode_action(action_idx):
    """Convert a fixed action index back to its (from_square, to_square) pair."""
    return ALL_ACTIONS[action_idx]
