# Evaluation Workflow — DQN Galaxian

## Prerequisites

- A trained model (run `python train.py` first)
- Dependencies installed: `pip install -r requirements.txt`

## Quick Start

```bash
# Play with the best model (default: models/dqn_model.zip)
python play.py

# Play with a specific experiment's model
python play.py --model models/experiment_1/best_model.zip

# Play more episodes
python play.py --episodes 10
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `models/dqn_model.zip` | Path to trained model |
| `--episodes` | 5 | Number of episodes to play |

## What to Observe

- **Early training models**: The agent moves randomly, shoots occasionally, dies quickly
- **Mid training models**: The agent starts dodging enemy fire and shooting back
- **Well-trained models**: The agent actively targets enemies and avoids projectiles

## How It Works

1. The trained DQN model is loaded from disk
2. The policy type (CnnPolicy or MlpPolicy) is auto-detected
3. A Galaxian environment is created with `render_mode="human"` for real-time display
4. The agent plays using a greedy policy (`deterministic=True`) — always selecting the action with the highest Q-value
5. Per-episode reward and step count are printed, followed by a summary
