"""Track and tabulate experiment results across the hyperparameter sweep."""

import os
import pandas as pd
from tabulate import tabulate


class ResultsTracker:
    """Accumulates experiment results and produces formatted tables."""

    def __init__(self):
        self.results = []

    def add_result(self, exp_id, hyperparams, mean_reward, std_reward,
                   mean_ep_length, training_time_s):
        """Record one experiment's outcome."""
        self.results.append({
            "Exp": exp_id,
            "Policy": hyperparams["policy"],
            "lr": hyperparams["learning_rate"],
            "gamma": hyperparams["gamma"],
            "batch_size": hyperparams["batch_size"],
            "expl_fraction": hyperparams["exploration_fraction"],
            "expl_initial_eps": hyperparams.get("exploration_initial_eps", 1.0),
            "expl_final_eps": hyperparams["exploration_final_eps"],
            "Mean Reward": round(mean_reward, 2),
            "Std Reward": round(std_reward, 2),
            "Mean Ep Length": round(mean_ep_length, 2),
            "Time (min)": round(training_time_s / 60, 1),
            "Description": hyperparams.get("description", ""),
        })

    def get_table(self):
        """Return a formatted markdown table of all results."""
        if not self.results:
            return "No results recorded yet."
        df = pd.DataFrame(self.results)
        return tabulate(df, headers="keys", tablefmt="pipe", showindex=False)

    def get_dataframe(self):
        """Return results as a pandas DataFrame."""
        return pd.DataFrame(self.results)

    def save_csv(self, path="results/experiment_results.csv"):
        """Save results to CSV file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv(path, index=False)
        print(f"Results saved to {path}")

    def get_best(self):
        """Return the experiment dict with the highest mean reward."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r["Mean Reward"])
