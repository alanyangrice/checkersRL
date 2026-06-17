from __future__ import annotations

import math
import csv
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[1]
W, H = 960, 540
FPS_MS = 55

BG = "#f8fafc"
INK = "#111827"
MUTED = "#6b7280"
SUBTLE = "#e5e7eb"
CARD = "#ffffff"
BLUE = "#2563eb"
RED = "#b91c1c"
GREEN = "#16a34a"
AMBER = "#d97706"
PURPLE = "#7c3aed"
SLATE = "#334155"


SQ_POS = [
    (0, 1), (0, 3), (0, 5), (0, 7), (1, 0), (1, 2), (1, 4), (1, 6),
    (2, 1), (2, 3), (2, 5), (2, 7), (3, 0), (3, 2), (3, 4), (3, 6),
    (4, 1), (4, 3), (4, 5), (4, 7), (5, 0), (5, 2), (5, 4), (5, 6),
    (6, 1), (6, 3), (6, 5), (6, 7), (7, 0), (7, 2), (7, 4), (7, 6),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


F10 = font(10)
F12 = font(12)
F14 = font(14)
F16 = font(16)
F18 = font(18)
F20B = font(20, True)
F24B = font(24, True)
F30B = font(30, True)


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def col(rows: list[dict[str, str]], name: str, scale: float = 1.0) -> list[float]:
    return [float(row[name]) * scale for row in rows if row.get(name) not in (None, "")]


def moving_average(values: list[float], window: int = 5) -> list[float]:
    if window <= 1 or len(values) < window:
        return values
    out = []
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str | None = None, radius: int = 18, width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def shadowed_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int = 22) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle((x1 + 3, y1 + 5, x2 + 3, y2 + 5), radius=radius, fill="#dbe3ef")
    rounded(draw, box, CARD, "#e2e8f0", radius)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], s: str, fill: str = INK, fnt=F14) -> None:
    draw.text(xy, s, font=fnt, fill=fill)


def center_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], s: str, fill: str = INK, fnt=F14) -> None:
    bbox = draw.textbbox((0, 0), s, font=fnt)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x1, y1, x2, y2 = box
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2 - 1), s, font=fnt, fill=fill)


def pill(draw: ImageDraw.ImageDraw, xy: tuple[int, int], s: str, color: str, fnt=F12) -> None:
    x, y = xy
    bbox = draw.textbbox((0, 0), s, font=fnt)
    w = bbox[2] - bbox[0] + 18
    h = bbox[3] - bbox[1] + 10
    rounded(draw, (x, y, x + w, y + h), "#f8fafc", color, radius=999)
    center_text(draw, (x, y, x + w, y + h), s, color, fnt)


def draw_header(draw: ImageDraw.ImageDraw, subtitle: str) -> None:
    text(draw, (42, 28), "CheckersRL", INK, F30B)
    text(draw, (44, 65), subtitle, MUTED, F14)


def board_center(origin: tuple[int, int], size: int, sq: int) -> tuple[float, float]:
    row, col = SQ_POS[sq - 1]
    cell = size / 8
    return origin[0] + col * cell + cell / 2, origin[1] + row * cell + cell / 2


def draw_board(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    size: int,
    blue_sqs: list[int],
    red_sqs: list[int],
    kings: set[int] | None = None,
    highlight_path: list[int] | None = None,
) -> None:
    kings = kings or set()
    highlight_path = highlight_path or []
    ox, oy = origin
    cell = size / 8
    rounded(draw, (ox - 10, oy - 10, ox + size + 10, oy + size + 10), "#e2e8f0", None, 18)
    for r in range(8):
        for c in range(8):
            fill = "#f1f5f9" if (r + c) % 2 == 0 else "#64748b"
            draw.rectangle((ox + c * cell, oy + r * cell, ox + (c + 1) * cell, oy + (r + 1) * cell), fill=fill)
    for sq in highlight_path:
        x, y = board_center(origin, size, sq)
        draw.ellipse((x - cell * 0.39, y - cell * 0.39, x + cell * 0.39, y + cell * 0.39), outline="#f59e0b", width=4)
    if len(highlight_path) > 1:
        pts = [board_center(origin, size, sq) for sq in highlight_path]
        draw.line(pts, fill="#f59e0b", width=5, joint="curve")
    for sq, color, stroke in [(s, BLUE, "#1d4ed8") for s in blue_sqs] + [(s, RED, "#991b1b") for s in red_sqs]:
        x, y = board_center(origin, size, sq)
        draw.ellipse((x - cell * 0.31 + 2, y - cell * 0.31 + 3, x + cell * 0.31 + 2, y + cell * 0.31 + 3), fill="#1e293b")
        draw.ellipse((x - cell * 0.31, y - cell * 0.31, x + cell * 0.31, y + cell * 0.31), fill=color, outline=stroke, width=3)
        draw.ellipse((x - cell * 0.19, y - cell * 0.19, x + cell * 0.19, y + cell * 0.19), outline="#ffffff", width=2)
        if sq in kings:
            center_text(draw, (int(x - 14), int(y - 12), int(x + 14), int(y + 12)), "K", "#ffffff", F16)


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    values: list[float],
    color: str,
    width: int = 4,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    x1, y1, x2, y2 = box
    if not values:
        return
    vmin = min(values) if y_min is None else y_min
    vmax = max(values) if y_max is None else y_max
    if vmax - vmin < 1e-6:
        vmax += 1
    pts = []
    for i, v in enumerate(values):
        x = lerp(x1, x2, i / max(1, len(values) - 1))
        y = lerp(y2, y1, (v - vmin) / (vmax - vmin))
        pts.append((x, y))
    if len(pts) > 1:
        draw.line(pts, fill=color, width=width, joint="curve")
    for x, y in pts[-2:]:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)


def draw_y_ticks(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    ticks: list[float],
    y_min: float,
    y_max: float,
    suffix: str = "",
) -> None:
    x1, y1, x2, y2 = box
    for tick in ticks:
        y = lerp(y2, y1, (tick - y_min) / (y_max - y_min))
        draw.line((x1 - 4, y, x1, y), fill="#94a3b8", width=1)
        label = f"{int(tick)}{suffix}"
        bbox = draw.textbbox((0, 0), label, font=F10)
        text(draw, (int(x1 - 8 - (bbox[2] - bbox[0])), int(y - 6)), label, MUTED, F10)


def board_lists_from_state(state) -> tuple[list[int], list[int], set[int]]:
    blue: list[int] = []
    red: list[int] = []
    kings: set[int] = set()
    for sq, (row, col) in enumerate(SQ_POS, start=1):
        if state[0][row][col] > 0:
            blue.append(sq)
        elif state[1][row][col] > 0:
            blue.append(sq)
            kings.add(sq)
        elif state[2][row][col] > 0:
            red.append(sq)
        elif state[3][row][col] > 0:
            red.append(sq)
            kings.add(sq)
    return blue, red, kings


def split_move(move: str) -> list[int]:
    sep = "x" if "x" in move else "-"
    return [int(part) for part in move.split(sep)]


def load_wdl_network():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import torch
    from checkers_game.constants import NUM_ACTIONS
    from rl.networks import WDLAlphaZeroNetwork

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(repo_path("displayed_model", "az-wdl", "best.pt"), map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("value_fc2.weight") for k in state_dict.keys()):
        state_dict = {f"net.{k}": v for k, v in state_dict.items()}
    network = WDLAlphaZeroNetwork((4, 8, 8), n_actions=NUM_ACTIONS).to(device)
    network.load_state_dict(state_dict)
    network.eval()
    return network, device


def build_recorded_game_timeline(max_moves: int = 8) -> list[dict]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import torch
    from checkers_game.constants import BLUE, encode_action
    from rl.envs import CheckersEnv

    rows = read_csv(repo_path("rl", "scripts", "play_agent_games.csv"))
    game = next(row for row in rows if row["game_number"] == "4")
    moves = [m.strip() for m in game["moves"].split(",")]

    network, device = load_wdl_network()
    env = CheckersEnv()
    env.reset()
    timeline = []

    def blue_eval() -> dict[str, float]:
        state_t = torch.FloatTensor(env.get_board_state()).unsqueeze(0).to(device)
        with torch.no_grad():
            _, wdl_t, _ = network.forward_wdl(state_t)
        win, draw_prob, loss = [float(x) for x in wdl_t[0].tolist()]
        if env.game.turn == BLUE:
            blue_win = win
            red_win = loss
        else:
            blue_win = loss
            red_win = win
        return {
            "blue_win": blue_win,
            "draw": draw_prob,
            "red_win": red_win,
            "value": blue_win - red_win,
        }

    timeline.append(
        {
            "move": "Start",
            "state": env.get_absolute_board_state().tolist(),
            **blue_eval(),
            "turn": "Blue" if env.game.turn == BLUE else "Red",
        }
    )

    for move in moves[:max_moves]:
        parts = split_move(move)
        for src, dst in zip(parts, parts[1:]):
            action = encode_action(src, dst)
            if action < 0:
                raise ValueError(f"Cannot encode recorded move hop {src}->{dst} from {move}")
            _, _, done, _, _ = env.step(action)
            if done:
                break
        timeline.append(
            {
                "move": move,
                "state": env.get_absolute_board_state().tolist(),
                **blue_eval(),
                "turn": "Blue" if env.game.turn == BLUE else "Red",
            }
        )
    return timeline


def save_gif(frames: list[Image.Image], name: str, duration_ms: int = FPS_MS) -> None:
    frames[0].save(
        OUT_DIR / name,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=True,
    )


def make_gameplay_gif() -> None:
    frames: list[Image.Image] = []
    timeline = build_recorded_game_timeline(max_moves=8)
    total = 96

    for frame in range(total):
        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)
        shadowed_card(draw, (42, 58, 508, 505))

        phase = frame / total * (len(timeline) - 1)
        idx = min(len(timeline) - 2, int(phase))
        local = ease(phase - idx)
        cur = timeline[idx]
        nxt = timeline[idx + 1]
        display = nxt if local > 0.72 else cur
        blue, red, kings = board_lists_from_state(display["state"])
        active_path = [] if nxt["move"] == "Start" else split_move(nxt["move"])
        draw_board(draw, (110, 117), 330, blue, red, kings=kings, highlight_path=active_path)

        shadowed_card(draw, (548, 58, 918, 505))
        text(draw, (578, 85), "Live Agent Match", INK, F24B)
        text(draw, (580, 127), "Recorded game #4", BLUE, F14)
        text(draw, (720, 127), "AlphaZero WDL checkpoint", PURPLE, F14)
        text(draw, (580, 179), "Win probabilities", MUTED, F12)
        blue_win = lerp(cur["blue_win"], nxt["blue_win"], local)
        draw_prob = lerp(cur["draw"], nxt["draw"], local)
        red_win = lerp(cur["red_win"], nxt["red_win"], local)
        bar_x1, bar_y1, bar_x2, bar_y2 = 580, 203, 886, 225
        bar_w = bar_x2 - bar_x1
        blue_w = int(bar_w * blue_win)
        draw_w = int(bar_w * draw_prob)
        rounded(draw, (bar_x1, bar_y1, bar_x2, bar_y2), "#e5e7eb", None, 999)
        draw.rounded_rectangle((bar_x1, bar_y1, bar_x2, bar_y2), radius=999, fill="#e5e7eb")
        draw.rectangle((bar_x1, bar_y1, bar_x1 + blue_w, bar_y2), fill=BLUE)
        draw.rectangle((bar_x1 + blue_w, bar_y1, bar_x1 + blue_w + draw_w, bar_y2), fill="#cbd5e1")
        draw.rectangle((bar_x1 + blue_w + draw_w, bar_y1, bar_x2, bar_y2), fill=RED)
        text(draw, (580, 234), f"Blue win {blue_win * 100:.0f}%", BLUE, F12)
        center_text(draw, (684, 233, 784, 257), f"draw {draw_prob * 100:.0f}%", MUTED, F12)
        text(draw, (810, 234), f"Red win {red_win * 100:.0f}%", RED, F12)
        text(draw, (580, 312), "Move log", MUTED, F12)
        shown = timeline[1 : idx + 2][-4:]
        for i, entry in enumerate(shown):
            y = 339 + i * 24
            rounded(draw, (580, y, 886, y + 23), "#f8fafc" if i == len(shown) - 1 else "#ffffff", "#e5e7eb", 8)
            row_color = INK if i == len(shown) - 1 else MUTED
            text(draw, (594, y + 3), entry["move"], row_color, F14)
            wdl = f"B {entry['blue_win'] * 100:.0f}  D {entry['draw'] * 100:.0f}  R {entry['red_win'] * 100:.0f}"
            text(draw, (742, y + 5), wdl, row_color, F12)
        frames.append(img)
    save_gif(frames, "checkersrl-gameplay.gif", duration_ms=95)


def make_ppo_training_gif() -> None:
    frames: list[Image.Image] = []
    total = 86
    bench = read_csv(repo_path("rl", "training_results", "ppo", "benchmark_league.csv"))
    tactical = col(bench, "tactical_vs_random_win", 100)
    terminal = col(bench, "terminal_vs_random_win", 100)
    aggressive = col(bench, "aggressive_vs_random_win", 100)
    tactical_prog = read_csv(repo_path("rl", "training_results", "ppo", "training_progress_tactical.csv"))
    terminal_prog = read_csv(repo_path("rl", "training_results", "ppo", "training_progress_terminal.csv"))
    aggressive_prog = read_csv(repo_path("rl", "training_results", "ppo", "training_progress_aggressive.csv"))
    tie_tactical = moving_average(col(tactical_prog, "tie_rate", 100), 7)
    tie_terminal = moving_average(col(terminal_prog, "tie_rate", 100), 7)
    tie_aggressive = moving_average(col(aggressive_prog, "tie_rate", 100), 7)
    len_tactical = moving_average(col(tactical_prog, "average_episode_length"), 7)
    len_terminal = moving_average(col(terminal_prog, "average_episode_length"), 7)
    len_aggressive = moving_average(col(aggressive_prog, "average_episode_length"), 7)

    for frame in range(total):
        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)
        shadowed_card(draw, (42, 58, 918, 505))
        text(draw, (78, 82), "PPO League vs Random", INK, F24B)

        progress = ease(frame / (total - 1))
        n = max(2, int(progress * len(tactical)))
        pn = max(3, int(progress * len(tie_tactical)))
        plot = (94, 174, 566, 404)
        draw.rectangle(plot, fill="#f8fafc", outline="#e2e8f0")
        for tick in [40, 55, 70, 85, 100]:
            y = lerp(plot[3], plot[1], (tick - 40) / 60)
            draw.line((plot[0], y, plot[2], y), fill="#e2e8f0", width=1)
        draw_y_ticks(draw, plot, [40, 55, 70, 85, 100], 40, 100, "%")
        text(draw, (94, 414), "epoch", MUTED, F12)
        text(draw, (52, 172), "win", MUTED, F12)
        draw_polyline(draw, plot, tactical[:n], BLUE, y_min=40, y_max=100)
        draw_polyline(draw, plot, terminal[:n], AMBER, y_min=40, y_max=100)
        draw_polyline(draw, plot, aggressive[:n], RED, y_min=40, y_max=100)
        text(draw, (94, 132), "Tactical", BLUE, F14)
        text(draw, (170, 132), "Terminal", AMBER, F14)
        text(draw, (252, 132), "Aggressive", RED, F14)

        text(draw, (620, 82), "Diagnostics", INK, F20B)
        rounded(draw, (620, 122, 876, 258), "#f8fafc", "#e2e8f0", 16)
        text(draw, (650, 140), "Self-play Tie Rate", MUTED, F12)
        tie_plot = (666, 174, 856, 236)
        for tick in [0, 10, 20, 30]:
            y = lerp(tie_plot[3], tie_plot[1], tick / 30)
            draw.line((tie_plot[0], y, tie_plot[2], y), fill="#e2e8f0", width=1)
        draw_y_ticks(draw, tie_plot, [0, 10, 20, 30], 0, 30, "%")
        draw_polyline(draw, tie_plot, tie_tactical[:pn], BLUE, 3, y_min=0, y_max=30)
        draw_polyline(draw, tie_plot, tie_terminal[:pn], AMBER, 3, y_min=0, y_max=30)
        draw_polyline(draw, tie_plot, tie_aggressive[:pn], RED, 3, y_min=0, y_max=30)
        rounded(draw, (620, 284, 876, 430), "#f8fafc", "#e2e8f0", 16)
        text(draw, (650, 302), "Episode Length", MUTED, F12)
        len_plot = (666, 338, 856, 408)
        for tick in [40, 70, 100, 130]:
            y = lerp(len_plot[3], len_plot[1], (tick - 35) / 95)
            draw.line((len_plot[0], y, len_plot[2], y), fill="#e2e8f0", width=1)
        draw_y_ticks(draw, len_plot, [40, 70, 100, 130], 35, 130)
        draw_polyline(draw, len_plot, len_tactical[:pn], BLUE, 3, y_min=35, y_max=130)
        draw_polyline(draw, len_plot, len_terminal[:pn], AMBER, 3, y_min=35, y_max=130)
        draw_polyline(draw, len_plot, len_aggressive[:pn], RED, 3, y_min=35, y_max=130)
        epoch = int(float(bench[n - 1]["epoch"]))
        center_text(draw, (685, 455, 815, 485), f"epoch {epoch:03d}", INK, F20B)
        frames.append(img)
    save_gif(frames, "checkersrl-ppo-training-dashboard.gif")


def make_mcts_training_gif() -> None:
    frames: list[Image.Image] = []
    total = 86
    scalar_eval = read_csv(repo_path("rl", "training_results", "mcts", "alphazero_eval_benchmarks_scalar.csv"))
    wdl_eval = read_csv(repo_path("rl", "training_results", "mcts", "alphazero_eval_benchmarks_wdl.csv"))
    scalar_prog = read_csv(repo_path("rl", "training_results", "mcts", "alphazero_training_progress_parallel_scalar.csv"))
    wdl_prog = read_csv(repo_path("rl", "training_results", "mcts", "alphazero_training_progress_parallel_wdl.csv"))
    scalar_gate = col(scalar_eval, "gate_score", 100)
    wdl_gate = col(wdl_eval, "gate_score", 100)
    scalar_entropy = moving_average(col(scalar_prog, "policy_entropy_nats"), 5)
    wdl_entropy = moving_average(col(wdl_prog, "policy_entropy_nats"), 5)
    scalar_moves = moving_average(col(scalar_prog, "avg_moves"), 5)
    wdl_moves = moving_average(col(wdl_prog, "avg_moves"), 5)

    for frame in range(total):
        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)
        draw_header(draw, "AlphaZero dashboard from MCTS training and eval CSVs")
        shadowed_card(draw, (42, 105, 918, 505))
        text(draw, (78, 132), "AlphaZero Scalar vs WDL", INK, F24B)
        text(draw, (78, 164), "Actual gate_score from eval benchmarks", MUTED, F14)

        progress = ease(frame / (total - 1))
        n = max(2, int(progress * min(len(scalar_gate), len(wdl_gate))))
        pn = max(3, int(progress * min(len(scalar_entropy), len(wdl_entropy))))
        plot = (78, 220, 566, 430)
        draw.rectangle(plot, fill="#f8fafc", outline="#e2e8f0")
        for i in range(5):
            y = plot[1] + i * (plot[3] - plot[1]) / 4
            draw.line((plot[0], y, plot[2], y), fill="#e2e8f0", width=1)
        text(draw, (78, 440), "checkpoint", MUTED, F12)
        text(draw, (42, 218), "gate %", MUTED, F12)
        draw.line((plot[0], plot[3] - (55 - 35) / (85 - 35) * (plot[3] - plot[1]), plot[2], plot[3] - (55 - 35) / (85 - 35) * (plot[3] - plot[1])), fill="#cbd5e1", width=2)
        draw_polyline(draw, plot, scalar_gate[:n], BLUE, y_min=35, y_max=85)
        draw_polyline(draw, plot, wdl_gate[:n], RED, y_min=35, y_max=85)
        pill(draw, (78, 188), "Scalar", BLUE)
        pill(draw, (158, 188), "WDL", RED)
        pill(draw, (220, 188), "55% accept line", SLATE)

        text(draw, (620, 132), "Diagnostics", INK, F20B)
        rounded(draw, (620, 172, 876, 278), "#f8fafc", "#e2e8f0", 16)
        text(draw, (640, 190), "Policy Entropy (nats)", MUTED, F12)
        draw_polyline(draw, (640, 220, 856, 258), scalar_entropy[:pn], BLUE, 3, y_min=0.85, y_max=1.4)
        draw_polyline(draw, (640, 220, 856, 258), wdl_entropy[:pn], RED, 3, y_min=0.85, y_max=1.4)
        rounded(draw, (620, 304, 876, 430), "#f8fafc", "#e2e8f0", 16)
        text(draw, (640, 322), "Self-play Avg Moves", MUTED, F12)
        draw_polyline(draw, (640, 354, 856, 410), scalar_moves[:pn], BLUE, 3, y_min=35, y_max=150)
        draw_polyline(draw, (640, 354, 856, 410), wdl_moves[:pn], RED, 3, y_min=35, y_max=150)
        epoch = int(float(scalar_eval[n - 1]["epoch"]))
        center_text(draw, (685, 455, 815, 485), f"epoch {epoch:03d}", INK, F20B)
        frames.append(img)
    save_gif(frames, "checkersrl-mcts-training-dashboard.gif")


def make_infrastructure_gif() -> None:
    frames: list[Image.Image] = []
    total = 88
    worker_y = [160, 235, 310, 385]

    for frame in range(total):
        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)
        draw_header(draw, "Parallel CPU self-play with batched GPU inference")
        shadowed_card(draw, (42, 105, 918, 505))
        text(draw, (78, 130), "Training Infrastructure", INK, F24B)
        text(draw, (78, 162), "Workers simulate games; the GPU server batches neural net calls", MUTED, F14)

        for i, y in enumerate(worker_y):
            rounded(draw, (82, y, 238, y + 48), "#eff6ff", "#bfdbfe", 14)
            center_text(draw, (82, y, 238, y + 48), f"CPU Worker {i + 1}", BLUE, F14)

        rounded(draw, (390, 205, 570, 345), "#f8fafc", "#cbd5e1", 20)
        center_text(draw, (390, 220, 570, 255), "Batch Queue", SLATE, F18)
        for i in range(4):
            rounded(draw, (420, 272 + i * 14, 540, 280 + i * 14), "#dbeafe", None, 999)

        rounded(draw, (690, 205, 850, 345), "#111827", "#0f172a", 22)
        center_text(draw, (690, 235, 850, 270), "GPU", "#ffffff", F24B)
        center_text(draw, (690, 274, 850, 302), "batched inference", "#cbd5e1", F12)

        cycle = (frame % 44) / 44
        for i, y in enumerate(worker_y):
            offset = (cycle + i * 0.15) % 1.0
            if offset < 0.55:
                t = ease(offset / 0.55)
                x = lerp(238, 390, t)
                yy = lerp(y + 24, 275, t)
                color = BLUE
            else:
                t = ease((offset - 0.55) / 0.45)
                x = lerp(570, 690, t)
                yy = lerp(275, 275, t)
                color = GREEN
            draw.line((238, y + 24, 390, 275), fill="#dbeafe", width=2)
            draw.line((570, 275, 690, 275), fill="#dcfce7", width=3)
            draw.ellipse((x - 8, yy - 8, x + 8, yy + 8), fill=color, outline="#ffffff", width=2)

        wave = 0.5 + 0.5 * math.sin(frame / total * math.tau * 3)
        rounded(draw, (706, 314, 834, 326), "#334155", None, 999)
        rounded(draw, (706, 314, int(706 + 128 * wave), 326), GREEN, None, 999)

        rounded(draw, (390, 395, 850, 455), "#f8fafc", "#e2e8f0", 16)
        text(draw, (414, 414), "Main process updates PPO / AlphaZero networks after rollout collection", MUTED, F14)
        frames.append(img)
    save_gif(frames, "checkersrl-training-infrastructure.gif")


if __name__ == "__main__":
    make_gameplay_gif()
    make_ppo_training_gif()
    make_mcts_training_gif()
    make_infrastructure_gif()
    print("Generated:")
    for path in sorted(OUT_DIR.glob("*.gif")):
        print(f" - {path.name}")
