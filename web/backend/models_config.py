"""Model registry config — edit this file to change which models appear in the dropdown.

Each entry's 'checkpoint' path is relative to the repo root.
'supports_difficulty' controls whether the simulations slider is shown.
"""

MODELS = [
    {
        "id":                  "az_v2_latest",
        "label":               "AlphaZero — Latest",
        "type":                "alphazero",
        "checkpoint":          "RL_models/MCTS/alphazero_checkpoints/az_epoch_93.pt",
        "az_version":          "v2",
        "supports_difficulty": True,
    },
    {
        "id":                  "az_v2_mid",
        "label":               "AlphaZero — Mid Training",
        "type":                "alphazero",
        "checkpoint":          "RL_models/MCTS/alphazero_checkpoints/az_epoch_60.pt",
        "az_version":          "v2",
        "supports_difficulty": True,
    },
    {
        "id":                  "az_v2_early",
        "label":               "AlphaZero — Early Training",
        "type":                "alphazero",
        "checkpoint":          "RL_models/MCTS/alphazero_checkpoints/az_epoch_30.pt",
        "az_version":          "v2",
        "supports_difficulty": True,
    },
    {
        "id":                  "az_v1_latest",
        "label":               "AlphaZero v1 — Scalar Network",
        "type":                "alphazero",
        "checkpoint":          "RL_models/MCTS/alphazero_checkpoints_v1/az_epoch_135.pt",
        "az_version":          "v1",
        "supports_difficulty": True,
    },
]
