"""Hyperparameter configurations for the 10 DQN experiments."""

# Shared defaults across all experiments
SHARED_DEFAULTS = {
    "buffer_size": 10_000,
    "learning_starts": 10_000,
    "target_update_interval": 1_000,
    "train_freq": 4,
    "gradient_steps": 1,
    "exploration_initial_eps": 1.0,
}

# 10 experiment configurations — vary one/two params from baseline each time
EXPERIMENTS = [
    {
        "id": 1,
        "policy": "CnnPolicy",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.10,
        "exploration_final_eps": 0.01,
        "description": "Baseline (RL Zoo Atari defaults)",
    },
    {
        "id": 2,
        "policy": "CnnPolicy",
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.10,
        "exploration_final_eps": 0.01,
        "description": "Higher learning rate",
    },
    {
        "id": 3,
        "policy": "CnnPolicy",
        "learning_rate": 1e-5,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.10,
        "exploration_final_eps": 0.01,
        "description": "Lower learning rate",
    },
    {
        "id": 4,
        "policy": "CnnPolicy",
        "learning_rate": 1e-4,
        "gamma": 0.95,
        "batch_size": 32,
        "exploration_fraction": 0.10,
        "exploration_final_eps": 0.01,
        "description": "Lower gamma (myopic)",
    },
    {
        "id": 5,
        "policy": "CnnPolicy",
        "learning_rate": 1e-4,
        "gamma": 0.999,
        "batch_size": 32,
        "exploration_fraction": 0.10,
        "exploration_final_eps": 0.01,
        "description": "Higher gamma (farsighted)",
    },
    {
        "id": 6,
        "policy": "CnnPolicy",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "exploration_fraction": 0.10,
        "exploration_final_eps": 0.01,
        "description": "Larger batch size",
    },
    {
        "id": 7,
        "policy": "CnnPolicy",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "exploration_fraction": 0.10,
        "exploration_final_eps": 0.01,
        "description": "Even larger batch size",
    },
    {
        "id": 8,
        "policy": "CnnPolicy",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.30,
        "exploration_final_eps": 0.05,
        "description": "More exploration (longer decay, higher floor)",
    },
    {
        "id": 9,
        "policy": "CnnPolicy",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.05,
        "exploration_final_eps": 0.01,
        "description": "Less exploration (shorter annealing)",
    },
    {
        "id": 10,
        "policy": "MlpPolicy",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.10,
        "exploration_final_eps": 0.01,
        "description": "MLP on RAM observations (CnnPolicy vs MlpPolicy comparison)",
    },
]


def get_experiment(exp_id):
    """Return a single experiment config by ID (1-indexed)."""
    for exp in EXPERIMENTS:
        if exp["id"] == exp_id:
            return exp
    raise ValueError(f"Experiment {exp_id} not found. Valid IDs: 1-{len(EXPERIMENTS)}")


def get_dqn_kwargs(exp):
    """Build the kwargs dict for DQN() from an experiment config."""
    return {
        "learning_rate": exp["learning_rate"],
        "gamma": exp["gamma"],
        "batch_size": exp["batch_size"],
        "exploration_fraction": exp["exploration_fraction"],
        "exploration_initial_eps": SHARED_DEFAULTS["exploration_initial_eps"],
        "exploration_final_eps": exp["exploration_final_eps"],
        "buffer_size": SHARED_DEFAULTS["buffer_size"],
        "learning_starts": SHARED_DEFAULTS["learning_starts"],
        "target_update_interval": SHARED_DEFAULTS["target_update_interval"],
        "train_freq": SHARED_DEFAULTS["train_freq"],
        "gradient_steps": SHARED_DEFAULTS["gradient_steps"],
    }
