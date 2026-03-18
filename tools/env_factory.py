"""Environment creation helpers for DQN Galaxian training and evaluation."""

import gymnasium as gym
import ale_py
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Register ALE environments with Gymnasium (required for newer ale_py versions)
gym.register_envs(ale_py)


def make_cnn_env(seed=0, monitor_dir=None):
    """Create a vectorized Atari environment with image observations for CnnPolicy.

    Uses ALE/Galaxian-v5 with frameskip=1 so SB3's AtariWrapper handles
    frame-skipping (avoids double skip). Stacks 4 grayscale 84x84 frames.
    """
    vec_env = make_atari_env(
        "ALE/Galaxian-v5",
        n_envs=1,
        seed=seed,
        monitor_dir=monitor_dir,
        env_kwargs={"frameskip": 1},
    )
    vec_env = VecFrameStack(vec_env, n_stack=4)
    return vec_env


def make_mlp_env(seed=0, monitor_dir=None):
    """Create a vectorized Atari environment with RAM observations for MlpPolicy.

    Uses ALE/Galaxian-v5 with obs_type='ram' which provides a flat 128-byte
    observation vector. No image preprocessing or frame stacking needed.
    """
    def _make():
        env = gym.make("ALE/Galaxian-v5", obs_type="ram")
        env = Monitor(env, filename=monitor_dir)
        env.reset(seed=seed)
        return env

    vec_env = DummyVecEnv([_make])
    return vec_env


def make_play_env(policy_type):
    """Create a non-vectorized rendering environment for play.py.

    Args:
        policy_type: "CnnPolicy" or "MlpPolicy"

    Returns:
        A gymnasium environment with render_mode="human" and appropriate wrappers.
    """
    if policy_type == "CnnPolicy":
        env = gym.make("ALE/Galaxian-v5", frameskip=1, render_mode="human")
        env = AtariWrapper(env)
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    else:
        env = gym.make("ALE/Galaxian-v5", obs_type="ram", render_mode="human")
    return env
