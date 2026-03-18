# Training Workflow — DQN Galaxian

## Prerequisites

- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`

## Quick Start

```bash
# Run all 10 experiments (500K steps each — several hours total)
python train.py

# Run a single experiment for testing
python train.py --experiment 1 --timesteps 1000

# Run a specific experiment with custom timesteps
python train.py --experiment 3 --timesteps 200000
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiment` | None (all) | Run a single experiment by ID (1-10) |
| `--timesteps` | 500000 | Total training timesteps per experiment |
| `--seed` | 42 | Random seed for reproducibility |

## What Happens During Training

1. For each experiment, a DQN agent is created with that experiment's hyperparameters
2. The agent trains on Galaxian using either CnnPolicy (image obs) or MlpPolicy (RAM obs)
3. Every 10,000 steps the agent is evaluated for 10 episodes — the best checkpoint is saved
4. After training, a final evaluation of 20 episodes is run
5. Results are printed as a table and saved to `results/experiment_results.csv`
6. The best-performing model is copied to `models/dqn_model.zip`

## Viewing Training Curves

```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser to see reward trends and episode lengths.

## Output Files

- `models/experiment_{id}/best_model.zip` — Best checkpoint per experiment
- `models/experiment_{id}/final_model.zip` — Final model per experiment
- `models/dqn_model.zip` — Best overall model (used by play.py)
- `logs/` — TensorBoard logs
- `results/experiment_results.csv` — Results table

## Hyperparameter Experiments

See `tools/hyperparams.py` for all 10 configurations. The experiments systematically vary:
- Learning rate (experiments 1-3)
- Gamma / discount factor (experiments 4-5)
- Batch size (experiments 6-7)
- Exploration schedule (experiments 8-9)
- CnnPolicy vs MlpPolicy (experiment 10)
