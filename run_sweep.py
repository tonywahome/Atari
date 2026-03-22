"""Lean training sweep — trains all 10 experiments with minimal overhead."""

import os
import time
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage

from tools.env_factory import make_cnn_env, make_mlp_env
from Atari.workflows.hyperparams import EXPERIMENTS, get_dqn_kwargs
from tools.results_tracker import ResultsTracker

TIMESTEPS = 10_000
SEED = 42


def run_one(exp):
    exp_id = exp["id"]
    print(f"\n{'='*50}")
    print(f"  Exp {exp_id}: {exp['description']}")
    print(f"{'='*50}")

    # Create environments
    if exp["policy"] == "CnnPolicy":
        train_env = make_cnn_env(seed=SEED)
        eval_env = make_cnn_env(seed=SEED + 100)
        eval_env = VecTransposeImage(eval_env)
    else:
        train_env = make_mlp_env(seed=SEED)
        eval_env = make_mlp_env(seed=SEED + 100)

    # Build and train
    model = DQN(
        policy=exp["policy"],
        env=train_env,
        **get_dqn_kwargs(exp),
        verbose=0,
        seed=SEED,
    )

    start = time.time()
    model.learn(total_timesteps=TIMESTEPS)
    elapsed = time.time() - start

    # Evaluate (10 episodes)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

    # Quick episode length estimate (5 episodes)
    ep_lengths = []
    for _ in range(5):
        obs = eval_env.reset()
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = eval_env.step(action)
            steps += 1
        ep_lengths.append(steps)

    # Save model
    model_dir = f"models/experiment_{exp_id}"
    os.makedirs(model_dir, exist_ok=True)
    model.save(f"{model_dir}/best_model")

    train_env.close()
    eval_env.close()

    print(f"  -> Reward: {mean_reward:.1f} +/- {std_reward:.1f} | Ep len: {np.mean(ep_lengths):.0f} | Time: {elapsed:.0f}s")
    return mean_reward, std_reward, np.mean(ep_lengths), elapsed


def main():
    tracker = ResultsTracker()
    print(f"Running {len(EXPERIMENTS)} experiments @ {TIMESTEPS:,} timesteps each\n")

    for exp in EXPERIMENTS:
        mean_r, std_r, ep_len, elapsed = run_one(exp)
        tracker.add_result(
            exp_id=exp["id"],
            hyperparams=exp,
            mean_reward=mean_r,
            std_reward=std_r,
            mean_ep_length=ep_len,
            training_time_s=elapsed,
        )

    print(f"\n{'='*50}")
    print("  ALL RESULTS")
    print(f"{'='*50}\n")
    print(tracker.get_table())
    tracker.save_csv()

    best = tracker.get_best()
    if best:
        print(f"\nBest: Experiment #{best['Exp']} — Mean Reward: {best['Mean Reward']}")


if __name__ == "__main__":
    main()
