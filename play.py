"""
play.py — Load a trained DQN model and play Atari Galaxian with real-time rendering.

Uses a greedy policy (deterministic=True) so the agent always picks
the action with the highest Q-value.

Usage:
    python play.py                                # Play with default model
    python play.py --model models/experiment_3/best_model.zip  # Specific model
    python play.py --episodes 10                  # Play 10 episodes
"""

import argparse
import time

import numpy as np
from stable_baselines3 import DQN

from tools.env_factory import make_play_env


def parse_args():
    parser = argparse.ArgumentParser(description="Play Galaxian with a trained DQN agent")
    parser.add_argument(
        "--model", type=str, default="models/dqn_model.zip",
        help="Path to the trained model (default: models/dqn_model.zip).",
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of episodes to play (default: 5).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the trained model
    print(f"Loading model from {args.model}...")
    model = DQN.load(args.model)

    # Detect policy type to create the matching environment
    policy_name = model.policy_class.__name__
    if "Cnn" in policy_name:
        policy_type = "CnnPolicy"
    else:
        policy_type = "MlpPolicy"
    print(f"Detected policy: {policy_type}")

    # Create rendering environment
    env = make_play_env(policy_type)

    print(f"\nPlaying {args.episodes} episode(s) of Galaxian...\n")

    total_rewards = []
    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            # Greedy Q-policy: deterministic=True selects the highest Q-value action
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            total_reward += rewards[0]
            steps += 1
            if dones[0]:
                break

        total_rewards.append(total_reward)
        print(f"  Episode {episode}: Reward = {total_reward:.0f}, Steps = {steps}")

    env.close()

    print(f"\nSummary over {args.episodes} episodes:")
    print(f"  Mean Reward: {np.mean(total_rewards):.2f}")
    print(f"  Std Reward:  {np.std(total_rewards):.2f}")
    print(f"  Min Reward:  {np.min(total_rewards):.0f}")
    print(f"  Max Reward:  {np.max(total_rewards):.0f}")


if __name__ == "__main__":
    main()
