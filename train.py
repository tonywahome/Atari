"""
train.py — Train a DQN agent to play Atari Galaxian using Stable Baselines 3.

Runs 10 hyperparameter experiments, logs training to TensorBoard,
evaluates each configuration, and saves the best model as models/dqn_model.zip.

Usage:
    python train.py                          # Run all 10 experiments (500K steps each)
    python train.py --experiment 1           # Run only experiment 1
    python train.py --timesteps 100000      # Custom timestep count
    python train.py --experiment 1 --timesteps 1000  # Quick smoke test
"""

import argparse
import os
import shutil
import time

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage

from tools.env_factory import make_cnn_env, make_mlp_env
from Atari.workflows.hyperparams import EXPERIMENTS, get_dqn_kwargs, get_experiment
from tools.results_tracker import ResultsTracker


def parse_args():
    parser = argparse.ArgumentParser(description="DQN Galaxian Hyperparameter Sweep")
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
    print(f"  Experiment {exp_id}: {exp['description']}")
    print(f"  Policy={exp['policy']}  lr={exp['learning_rate']}  gamma={exp['gamma']}")
    print(f"  batch={exp['batch_size']}  expl_frac={exp['exploration_fraction']}  expl_final={exp['exploration_final_eps']}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"{'='*60}\n")

    model_dir = f"models/experiment_{exp_id}"
    log_dir = f"logs/experiment_{exp_id}"
    monitor_dir = f"logs/experiment_{exp_id}/monitor"
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
    obs = eval_env.reset()
    ep_len = 0
    for _ in range(20):
        done = False
        obs = eval_env.reset()
        ep_len = 0
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

    print(f"\n  Experiment {exp_id} complete:")
    print(f"    Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"    Mean episode length: {mean_ep_length:.1f}")
    print(f"    Training time: {elapsed/60:.1f} min")

    return mean_reward, std_reward, mean_ep_length, elapsed


def main():
    args = parse_args()
    tracker = ResultsTracker()

    # Select experiments to run
    if args.experiment is not None:
        experiments = [get_experiment(args.experiment)]
    else:
        experiments = EXPERIMENTS

    print(f"Running {len(experiments)} experiment(s) with {args.timesteps:,} timesteps each.\n")

    # Run each experiment
    for exp in experiments:
        mean_reward, std_reward, mean_ep_length, elapsed = run_experiment(
            exp, total_timesteps=args.timesteps, seed=args.seed,
        )
        tracker.add_result(
            exp_id=exp["id"],
            hyperparams=exp,
            mean_reward=mean_reward,
            std_reward=std_reward,
            mean_ep_length=mean_ep_length,
            training_time_s=elapsed,
        )

    # Print and save results
    print(f"\n{'='*60}")
    print("  EXPERIMENT RESULTS")
    print(f"{'='*60}\n")
    print(tracker.get_table())
    tracker.save_csv()

    # Identify and save the best model
    best = tracker.get_best()
    if best is not None:
        best_id = best["Exp"]
        print(f"\nBest experiment: #{best_id} (Mean Reward: {best['Mean Reward']})")

        # Copy the best model to models/dqn_model.zip
        best_model_path = f"models/experiment_{best_id}/best_model.zip"
        if not os.path.exists(best_model_path):
            best_model_path = f"models/experiment_{best_id}/final_model.zip"

        os.makedirs("models", exist_ok=True)
        dest = "models/dqn_model.zip"
        shutil.copy2(best_model_path, dest)
        print(f"Best model saved to {dest}")


if __name__ == "__main__":
    main()
