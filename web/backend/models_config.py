"""Model registry config — edit this file to change which models appear in the dropdown.

Each entry's 'checkpoint' path is relative to the repo root (displayed_model/<run>/best.pt).
'supports_difficulty' controls whether the simulations slider is shown.

To update a model, copy the new checkpoint to displayed_model/<run>/best.pt and restart.
"""

MODELS = [
    {
        "id":                  "az-wdl",
        "label":               "AlphaZero WDL",
        "description":         "AlphaZero with Win/Draw/Loss categorical value head",
        "type":                "alphazero",
        "checkpoint":          "displayed_model/az-wdl/best.pt",
        "supports_difficulty": False,
    },
    {
        "id":                  "az-scalar",
        "label":               "AlphaZero Scalar",
        "description":         "AlphaZero with scalar MSE regression value head",
        "type":                "alphazero",
        "checkpoint":          "displayed_model/az-scalar/best.pt",
        "supports_difficulty": False,
    },
    {
        "id":                  "ppo-cs",
        "label":               "PPO Curriculum",
        "description":         "PPO trained with progressive curriculum + self-play",
        "type":                "ppo",
        "checkpoint":          "displayed_model/ppo-cs/best.pt",
        "supports_difficulty": False,
    },
    {
        "id":                  "ppo-league-aggressive",
        "label":               "PPO League — Aggressive",
        "description":         "PPO league agent with capture/king reward shaping",
        "type":                "ppo",
        "checkpoint":          "displayed_model/ppo-league/best_aggressive.pt",
        "supports_difficulty": False,
    },
    {
        "id":                  "ppo-league-tactical",
        "label":               "PPO League — Tactical",
        "description":         "PPO league agent with positional/advancement reward shaping",
        "type":                "ppo",
        "checkpoint":          "displayed_model/ppo-league/best_tactical.pt",
        "supports_difficulty": False,
    },
    {
        "id":                  "ppo-league-terminal",
        "label":               "PPO League — Terminal",
        "description":         "PPO league agent with win/loss only terminal rewards",
        "type":                "ppo",
        "checkpoint":          "displayed_model/ppo-league/best_terminal.pt",
        "supports_difficulty": False,
    },
]
