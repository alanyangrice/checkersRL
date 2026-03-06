"""One-off script to run the benchmark for epoch 60 and append to benchmark_parallel.csv."""
import os
import csv
import torch
from multiprocessing import cpu_count

from rl.algorithms.ppo.agent import PPOAgent
from rl.eval.ppo_evaluate import run_benchmark
from checkers_game.constants import NUM_ACTIONS

if __name__ == "__main__":
    epoch = 60
    benchmark_games = 500
    n_actions = NUM_ACTIONS
    input_shape = (4, 8, 8)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "training_results", "ppo", "ppo_saved_models_parallel")
    checkpoint_path = os.path.join(model_dir, f"agent_epoch_{epoch}.pt")
    reference_model_path = os.path.join(model_dir, "reference_model.pt")
    benchmark_csv_path = os.path.join(base_dir, "training_results", "ppo", "benchmark_parallel.csv")

    # Load agent from checkpoint
    agent = PPOAgent(input_shape, n_actions)
    checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    agent.policy.load_state_dict(checkpoint["model_state_dict"])
    agent.policy.eval()

    num_processes = max(2, int(cpu_count() * 0.5))
    print(f"Running benchmark for epoch {epoch} ({benchmark_games} games each, {num_processes} processes)...")

    from rl.algorithms.ppo.parallel_infra import WorkerContext
    from rl.configs.ppo_config import PPOConfig
    
    with WorkerContext(agent.get_policy_state_dict(), agent.device, n_actions, num_processes, config=PPOConfig()) as ctx:
        bench = run_benchmark(ctx, agent.device, n_actions, num_processes, reference_model_path, num_games=benchmark_games)

    vr = bench["vs_random"]
    print(f"  vs Random:    Win {vr['win_rate']:.1%}  Loss {vr['loss_rate']:.1%}  "
          f"Tie {vr['tie_rate']:.1%}  AvgSteps {vr['avg_steps']:.0f}")

    vref = bench.get("vs_reference", {})
    if vref:
        print(f"  vs Reference: Win {vref['win_rate']:.1%}  Loss {vref['loss_rate']:.1%}  "
              f"Tie {vref['tie_rate']:.1%}  AvgSteps {vref['avg_steps']:.0f}")

    # Append to benchmark CSV
    with open(benchmark_csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            epoch,
            vr["win_rate"], vr["loss_rate"], vr["tie_rate"], vr["avg_steps"],
            vref.get("win_rate", ""), vref.get("loss_rate", ""),
            vref.get("tie_rate", ""), vref.get("avg_steps", ""),
        ])

    print(f"Results appended to {benchmark_csv_path}")
