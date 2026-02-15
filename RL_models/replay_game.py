"""Game replay viewer — replay training games visually using pygame.

Loads game move logs from the detailed training CSVs (or ZIPs) and
plays them back on a rendered checkers board.

Usage:
    # Replay a specific game from a CSV file
    python -m RL_models.replay_game --file path/to/detailed_games.csv --game 5

    # Replay from a ZIP archive
    python -m RL_models.replay_game --file path/to/detailed_games.zip --game 10

    # Replay a move string directly
    python -m RL_models.replay_game --moves "11-15, 24-20, 8-11, 28-24"

Controls:
    Right Arrow  : Next move
    Left Arrow   : Previous move
    Home         : Jump to start
    End          : Jump to last move
    Space        : Toggle auto-play
    Up Arrow     : Speed up auto-play
    Down Arrow   : Slow down auto-play
    Q / Escape   : Quit

Note:
    Games from curriculum epochs (< 150) use random starting positions that
    are NOT recorded in the CSV. Replaying those games on a standard board
    will look incorrect. Use this tool with games from epoch 150+ for
    accurate replays, or pass a known move string with --moves.
"""

import argparse
import copy
import csv
import io
import sys
import zipfile

import pygame

from checkers_game.board import Board
from checkers_game.constants import (
    WIDTH, HEIGHT, ROWS, COLS, SQUARE_SIZE,
    RED, BLUE, WHITE, BLACK, GREEN,
    board_number_to_position, font,
)


# ── Layout constants ──────────────────────────────────────────────────
HUD_HEIGHT = 90
HUD_BG = (30, 30, 30)
HIGHLIGHT_FROM = (255, 215, 0)   # Gold — source square
HIGHLIGHT_TO = (0, 200, 100)     # Green — destination square
WINDOW_HEIGHT = HEIGHT + HUD_HEIGHT
INFO_FONT = None   # Initialised after pygame.init()
SMALL_FONT = None


# ── Move parsing ──────────────────────────────────────────────────────

def parse_move_string(move_str):
    """Parse a single move like '14-10' or '9x18x25' into step tuples.

    Returns a list of (from_sq, to_sq, is_capture) for each hop.
    """
    move_str = move_str.strip()
    if not move_str:
        return []

    if "x" in move_str:
        squares = list(map(int, move_str.split("x")))
        return [(squares[i], squares[i + 1], True) for i in range(len(squares) - 1)]
    elif "-" in move_str:
        parts = move_str.split("-")
        return [(int(parts[0]), int(parts[1]), False)]
    return []


def parse_all_moves(moves_field):
    """Split the CSV moves field into individual move strings (one per turn)."""
    if not moves_field or not moves_field.strip():
        return []
    return [m.strip() for m in moves_field.split(",") if m.strip()]


# ── CSV / ZIP loading ─────────────────────────────────────────────────

def load_game_from_file(file_path, game_row):
    """Load a single game record from a CSV or ZIP file.

    Args:
        file_path: Path to .csv or .zip file.
        game_row:  1-based row number in the CSV (excluding header).

    Returns:
        dict with keys: game_number, moves, winner, row_index.
    """
    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return None
            with zf.open(csv_names[0]) as f:
                return _find_game_in_csv(io.TextIOWrapper(f, encoding="utf-8"), game_row)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return _find_game_in_csv(f, game_row)


def _find_game_in_csv(file_obj, game_row):
    """Return the game dict at the given 1-based row, or None."""
    reader = csv.DictReader(file_obj)
    for row_num, row in enumerate(reader, start=1):
        if row_num == game_row:
            blue_win = int(row.get("blue_win", 0))
            red_win = int(row.get("red_win", 0))
            winner = "Blue" if blue_win else ("Red" if red_win else "Tie")
            return {
                "game_number": row.get("game_number", row_num),
                "moves": row.get("moves", ""),
                "winner": winner,
                "row_index": row_num,
            }
    return None


def count_games_in_file(file_path):
    """Return the total number of game rows in a CSV/ZIP file."""
    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return 0
            with zf.open(csv_names[0]) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                return sum(1 for _ in reader)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return sum(1 for _ in reader)


# ── Board replay logic ───────────────────────────────────────────────

class GameReplay:
    """Pre-computes board snapshots so we can step forward and backward."""

    def __init__(self, move_strings):
        self.move_strings = move_strings
        self.boards = []       # len = total_moves + 1 (index 0 = start)
        self.turn_colors = []  # len = total_moves
        self.error_at = None   # Move index where replay broke (curriculum)
        self._build_snapshots()
        self.current_step = 0

    def _build_snapshots(self):
        board = Board()
        self.boards = [copy.deepcopy(board)]
        current_turn = BLUE

        for i, move_str in enumerate(self.move_strings):
            steps = parse_move_string(move_str)
            try:
                for from_sq, to_sq, is_capture in steps:
                    fr, fc = board_number_to_position(from_sq)
                    tr, tc = board_number_to_position(to_sq)
                    piece = board.get_piece(fr, fc)
                    if piece == 0:
                        raise ValueError(
                            f"No piece at square {from_sq} ({fr},{fc}) for move '{move_str}'"
                        )
                    if is_capture:
                        board.capture_piece(fr, fc, tr, tc)
                    else:
                        board.move_piece(fr, fc, tr, tc)
            except Exception as e:
                print(f"Warning: Replay failed at move {i + 1} ({move_str}): {e}")
                print("  This is likely a curriculum game with a non-standard starting board.")
                self.error_at = i + 1
                self.move_strings = self.move_strings[:i]
                break

            self.boards.append(copy.deepcopy(board))
            self.turn_colors.append(current_turn)
            current_turn = BLUE if current_turn == RED else RED

    @property
    def total_moves(self):
        return len(self.turn_colors)

    def get_board(self):
        return self.boards[self.current_step]

    def get_current_move_str(self):
        if self.current_step == 0:
            return ""
        return self.move_strings[self.current_step - 1]

    def get_turn_color_name(self):
        if self.current_step == 0:
            return "Blue"
        color = self.turn_colors[self.current_step - 1]
        return "Blue" if color == BLUE else "Red"

    def next(self):
        if self.current_step < self.total_moves:
            self.current_step += 1
            return True
        return False

    def prev(self):
        if self.current_step > 0:
            self.current_step -= 1
            return True
        return False

    def go_start(self):
        self.current_step = 0

    def go_end(self):
        self.current_step = self.total_moves


# ── Rendering helpers ─────────────────────────────────────────────────

def draw_board_no_update(screen, board):
    """Render the board without calling pygame.display.update()."""
    board.draw_squares(screen)
    for row in range(ROWS):
        for col in range(COLS):
            piece = board.get_piece(row, col)
            if piece != 0:
                piece.draw(screen)


def draw_move_highlight(screen, replay):
    """Highlight the from/to squares of the current move."""
    if replay.current_step == 0:
        return

    move_str = replay.get_current_move_str()
    if "x" in move_str:
        squares = list(map(int, move_str.split("x")))
    elif "-" in move_str:
        squares = list(map(int, move_str.split("-")))
    else:
        return

    # Highlight first square (from) in gold
    fr, fc = board_number_to_position(squares[0])
    pygame.draw.rect(screen, HIGHLIGHT_FROM,
                     (fc * SQUARE_SIZE, fr * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4)

    # Highlight last square (to) in green
    tr, tc = board_number_to_position(squares[-1])
    pygame.draw.rect(screen, HIGHLIGHT_TO,
                     (tc * SQUARE_SIZE, tr * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4)


def draw_hud(screen, replay, game_info, auto_play, auto_speed):
    """Draw the info panel below the board."""
    hud_rect = pygame.Rect(0, HEIGHT, WIDTH, HUD_HEIGHT)
    pygame.draw.rect(screen, HUD_BG, hud_rect)

    y = HEIGHT + 6

    # Row 1: move counter + current move
    step_text = f"Move {replay.current_step}/{replay.total_moves}"
    if replay.current_step > 0:
        who = replay.get_turn_color_name()
        move = replay.get_current_move_str()
        step_text += f"   {who}: {move}"
    else:
        step_text += "   Game start"
    screen.blit(INFO_FONT.render(step_text, True, WHITE), (10, y))

    # Winner (right side)
    winner = game_info.get("winner", "?")
    winner_color = (100, 255, 100) if winner != "Tie" else (255, 255, 100)
    screen.blit(INFO_FONT.render(f"Winner: {winner}", True, winner_color), (WIDTH - 160, y))

    # Row 2: auto-play state + game number
    y += 28
    game_num = game_info.get("game_number", "?")
    screen.blit(INFO_FONT.render(f"Game #{game_num}", True, (180, 180, 180)), (10, y))

    auto_text = f"Auto: ON ({auto_speed:.1f}s)" if auto_play else "Auto: OFF"
    auto_color = (255, 200, 100) if auto_play else (120, 120, 120)
    screen.blit(INFO_FONT.render(auto_text, True, auto_color), (WIDTH - 200, y))

    # Row 3: controls
    y += 26
    hint = "</>: Step  |  Space: Auto  |  Home/End  |  Up/Down: Speed  |  Q: Quit"
    screen.blit(SMALL_FONT.render(hint, True, (90, 90, 90)), (10, y))

    # Warning for curriculum games
    if replay.error_at is not None:
        y += 20
        warn = f"Replay stopped at move {replay.error_at} (curriculum random board)"
        screen.blit(SMALL_FONT.render(warn, True, (255, 80, 80)), (10, y))


# ── Main replay loop ─────────────────────────────────────────────────

def run_replay(replay, game_info):
    """Open the pygame window and run the interactive replay."""
    pygame.init()

    global INFO_FONT, SMALL_FONT
    INFO_FONT = pygame.font.SysFont("Arial", 22)
    SMALL_FONT = pygame.font.SysFont("Arial", 16)

    screen = pygame.display.set_mode((WIDTH, WINDOW_HEIGHT))
    game_num = game_info.get("game_number", "?")
    pygame.display.set_caption(f"Checkers Replay - Game #{game_num}")

    clock = pygame.time.Clock()
    auto_play = False
    auto_speed = 1.0
    auto_timer = 0.0

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_RIGHT:
                    replay.next()
                elif event.key == pygame.K_LEFT:
                    replay.prev()
                elif event.key == pygame.K_HOME:
                    replay.go_start()
                elif event.key == pygame.K_END:
                    replay.go_end()
                elif event.key == pygame.K_SPACE:
                    auto_play = not auto_play
                    auto_timer = 0.0
                elif event.key == pygame.K_UP:
                    auto_speed = max(0.1, round(auto_speed - 0.2, 1))
                elif event.key == pygame.K_DOWN:
                    auto_speed = min(5.0, round(auto_speed + 0.2, 1))

        # Auto-play stepping
        if auto_play:
            auto_timer += dt
            if auto_timer >= auto_speed:
                auto_timer = 0.0
                if not replay.next():
                    auto_play = False

        # Draw
        screen.fill(BLACK)
        draw_board_no_update(screen, replay.get_board())
        draw_move_highlight(screen, replay)
        draw_hud(screen, replay, game_info, auto_play, auto_speed)
        pygame.display.update()

    pygame.quit()


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replay checkers games from training logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  Right / Left Arrow : Step forward / backward
  Home / End         : Jump to start / end
  Space              : Toggle auto-play
  Up / Down Arrow    : Adjust auto-play speed
  Q / Escape         : Quit

Examples:
  python -m RL_models.replay_game -f detailed_games_epoch_150.csv -g 5
  python -m RL_models.replay_game -f detailed_games_epoch_10.zip -g 42
  python -m RL_models.replay_game --moves "11-15, 24-20, 8-11, 28-24"
""",
    )
    parser.add_argument("--file", "-f", type=str,
                        help="Path to a detailed CSV or ZIP log file")
    parser.add_argument("--game", "-g", type=int, default=1,
                        help="Game row number to replay (1-based, default: 1)")
    parser.add_argument("--moves", "-m", type=str,
                        help="Comma-separated move string to replay directly")

    args = parser.parse_args()

    if args.moves:
        move_strings = parse_all_moves(args.moves)
        game_info = {"game_number": "manual", "winner": "?"}
    elif args.file:
        total = count_games_in_file(args.file)
        if args.game < 1 or args.game > total:
            print(f"Error: --game {args.game} out of range (file has {total} games)")
            sys.exit(1)

        game = load_game_from_file(args.file, game_row=args.game)
        if not game:
            print(f"Could not load game {args.game} from {args.file}")
            sys.exit(1)

        move_strings = parse_all_moves(game["moves"])
        game_info = game
        print(f"Game #{game['game_number']} | Winner: {game['winner']} | {len(move_strings)} moves")
    else:
        parser.print_help()
        print("\nError: provide either --file or --moves")
        sys.exit(1)

    if not move_strings:
        print("No moves to replay.")
        sys.exit(1)

    print(f"Loaded {len(move_strings)} moves. Opening replay window...")

    replay = GameReplay(move_strings)
    run_replay(replay, game_info)


if __name__ == "__main__":
    main()

