"""
train_belyse.py — Train belyse's 10 DQN experiments on Atari Galaxian.

Usage:
    python train_belyse.py                          # Run all 10 experiments (500K steps each)
    python train_belyse.py --experiment 1           # Run only experiment 1
    python train_belyse.py --timesteps 100000      # Custom timestep count
"""

import argparse
import os
import time

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage

from tools.env_factory import make_cnn_env, make_mlp_env
from Atari.workflows.hyperparams_belyse import EXPERIMENTS_BELYSE, get_dqn_kwargs, get_experiment
from tools.results_tracker import ResultsTracker


def parse_args():
    parser = argparse.ArgumentParser(description="DQN Galaxian Hyperparameter Sweep - belyse")
    parser.add_argument(
        "--experiment", type=int, default=None,
        help="Run a single experiment by ID (1-10). Default: run all.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Total training timesteps per experiment (default: 500000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def create_env(exp, seed, monitor_dir=None):
    """Create the appropriate vectorized environment for the experiment."""
    if exp["policy"] == "CnnPolicy":
        return make_cnn_env(seed=seed, monitor_dir=monitor_dir)
    else:
        return make_mlp_env(seed=seed, monitor_dir=monitor_dir)


def run_experiment(exp, total_timesteps, seed):
    """Train and evaluate one DQN experiment. Returns (mean_reward, std_reward, mean_ep_len, elapsed_s)."""
    exp_id = exp["id"]
    print(f"\n{'='*60}")
    print(f"  belyse - Experiment {exp_id}: {exp['description']}")
    print(f"  Policy={exp['policy']}  lr={exp['learning_rate']}  gamma={exp['gamma']}")
    print(f"  batch={exp['batch_size']}  expl_frac={exp['exploration_fraction']}  expl_final={exp['exploration_final_eps']}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"{'='*60}\n")

    model_dir = f"models/belyse/experiment_{exp_id}"
    log_dir = f"logs/belyse/experiment_{exp_id}"
    monitor_dir = f"logs/belyse/experiment_{exp_id}/monitor"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)

    # Create training and evaluation environments
    train_env = create_env(exp, seed=seed, monitor_dir=monitor_dir)
    eval_env = create_env(exp, seed=seed + 100)

    # Match VecTransposeImage wrapping so eval callback doesn't warn
    if exp["policy"] == "CnnPolicy":
        eval_env = VecTransposeImage(eval_env)

    # Build DQN agent
    dqn_kwargs = get_dqn_kwargs(exp)
    model = DQN(
        policy=exp["policy"],
        env=train_env,
        **dqn_kwargs,
        tensorboard_log=log_dir,
        verbose=1,
        seed=seed,
    )

    # Evaluation callback — saves best model during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=f"{log_dir}/eval",
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # Train
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )
    elapsed = time.time() - start_time

    # Final evaluation (20 episodes)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )

    # Compute mean episode length from evaluation
    ep_lengths = []
    for _ in range(20):
        obs = eval_env.reset()
        ep_len = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_len += 1
        ep_lengths.append(ep_len)
    mean_ep_length = np.mean(ep_lengths)

    # Save final model
    model.save(f"{model_dir}/final_model")

    # Clean up environments
    train_env.close()
    eval_env.close()

    print(f"\n  belyse Experiment {exp_id} complete:")
    print(f"    Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"    Mean episode length: {mean_ep_length:.1f}")
    print(f"    Training time: {elapsed/60:.1f} min")

    return mean_reward, std_reward, mean_ep_length, elapsed


def main():
    args = parse_args()
    tracker = ResultsTracker()

    # Select experiments to run
    if args.experiment is not None:
        exp_ids = [args.experiment]
    else:
        exp_ids = list(range(1, len(EXPERIMENTS_BELYSE) + 1))

    print(f"\n{'='*60}")
    print(f"  belyse's DQN Hyperparameter Tuning")
    print(f"  Running experiments: {exp_ids}")
    print(f"  Timesteps per experiment: {args.timesteps:,}")
    print(f"  Seed: {args.seed}")
    print(f"{'='*60}\n")

    all_results = []
    for exp_id in exp_ids:
        exp = get_experiment(exp_id)
        mean_reward, std_reward, mean_ep_length, elapsed = run_experiment(
            exp, args.timesteps, args.seed
        )
        tracker.add_result(exp_id, exp, mean_reward, std_reward, mean_ep_length, elapsed)
        all_results.append((exp_id, mean_reward))

    # Print summary table
    print(f"\n{'='*60}")
    print("  RESULTS TABLE - belyse's Experiments")
    print(f"{'='*60}\n")
    print(tracker.get_table())

    # Find and report best experiment
    best = tracker.get_best()
    if best:
        print(f"\n✓ Best experiment: #{best['Exp']} ({best['Description']})")
        print(f"  Mean reward: {best['Mean Reward']:.2f} +/- {best['Std Reward']:.2f}")
        
        # Copy best model to standard location
        best_model_src = f"models/belyse/experiment_{best['Exp']}/best_model.zip"
        best_model_dst = "models/belyse/dqn_model.zip"
        if os.path.exists(best_model_src):
            import shutil
            shutil.copy(best_model_src, best_model_dst)
            print(f"  Copied to: {best_model_dst}")

    # Save results CSV
    tracker.save_csv("results/belyse_results.csv")

    print(f"\n{'='*60}")
    print("  Training Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
