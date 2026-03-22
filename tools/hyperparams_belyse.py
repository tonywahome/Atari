"""Hyperparameter configurations for belyse's 10 DQN experiments."""

# Shared defaults across all experiments
SHARED_DEFAULTS = {
    "buffer_size": 50_000,
    "learning_starts": 1_000,
    "target_update_interval": 1_000,
    "train_freq": 4,
    "gradient_steps": 1,
    "exploration_initial_eps": 1.0,
}

# belyse's 10 experiment configurations — diverse exploration of hyperparameter space
EXPERIMENTS_BELYSE = [
    {
        "id": 1,
        "policy": "CnnPolicy",
        "learning_rate": 3e-4,
        "gamma": 0.97,
        "batch_size": 48,
        "exploration_fraction": 0.35,
        "exploration_final_eps": 0.01,
        "description": "High lr with conservative exploration (CNN variant A)",
    },
    {
        "id": 2,
        "policy": "CnnPolicy",
        "learning_rate": 5e-5,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.45,
        "exploration_final_eps": 0.04,
        "description": "Very low lr with strong exploration",
    },
    {
        "id": 3,
        "policy": "CnnPolicy",
        "learning_rate": 1e-4,
        "gamma": 0.9,
        "batch_size": 32,
        "exploration_fraction": 0.30,
        "exploration_final_eps": 0.02,
        "description": "Low gamma (short-term focused)",
    },
    {
        "id": 4,
        "policy": "CnnPolicy",
        "learning_rate": 2e-4,
        "gamma": 0.999,
        "batch_size": 32,
        "exploration_fraction": 0.25,
        "exploration_final_eps": 0.01,
        "description": "High gamma with fast learning (long-term + quick adaptation)",
    },
    {
        "id": 5,
        "policy": "CnnPolicy",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 16,
        "exploration_fraction": 0.40,
        "exploration_final_eps": 0.03,
        "description": "Small batch size with extended exploration",
    },
    {
        "id": 6,
        "policy": "CnnPolicy",
        "learning_rate": 1e-4,
        "gamma": 0.98,
        "batch_size": 96,
        "exploration_fraction": 0.20,
        "exploration_final_eps": 0.02,
        "description": "Large batch with minimal exploration (stability focused)",
    },
    {
        "id": 7,
        "policy": "CnnPolicy",
        "learning_rate": 4e-4,
        "gamma": 0.96,
        "batch_size": 64,
        "exploration_fraction": 0.45,
        "exploration_final_eps": 0.05,
        "description": "Aggressive learning + exploration (high variance strategy)",
    },
    {
        "id": 8,
        "policy": "CnnPolicy",
        "learning_rate": 8e-5,
        "gamma": 0.995,
        "batch_size": 32,
        "exploration_fraction": 0.25,
        "exploration_final_eps": 0.015,
        "description": "Conservative tuning (low lr, high gamma, minimal exploration)",
    },
    {
        "id": 9,
        "policy": "MlpPolicy",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.35,
        "exploration_final_eps": 0.02,
        "description": "MLP baseline (RAM observations, balanced params)",
    },
    {
        "id": 10,
        "policy": "MlpPolicy",
        "learning_rate": 2e-4,
        "gamma": 0.97,
        "batch_size": 64,
        "exploration_fraction": 0.40,
        "exploration_final_eps": 0.03,
        "description": "MLP with higher lr and larger batch (faster training)",
    },
]


def get_experiment(exp_id):
    """Return a single experiment config by ID (1-indexed)."""
    for exp in EXPERIMENTS_BELYSE:
        if exp["id"] == exp_id:
            return exp
    raise ValueError(f"Experiment {exp_id} not found. Valid IDs: 1-{len(EXPERIMENTS_BELYSE)}")


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
