import os
import random
import time
from typing import List

import numpy as np
import pandas as pd
import torch

from dataloader import get_cali_housing_example
from metrics import score
from preprocessing import standardize_data
from trainer import optimize_weights, predict


def main(n_seeds: int = 10, n_iter: int = 500, out_dir: str = "results") -> None:
    os.makedirs(out_dir, exist_ok=True)

    x, selections = get_cali_housing_example()
    x = torch.tensor(x)
    x, _, _ = standardize_data(x)

    results: List[dict] = []

    seeds = list(range(n_seeds))
    for sel_idx in range(selections.shape[0]):
        labels = selections[sel_idx]
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            mu, inverse_radius = optimize_weights(x, labels, n_iter=n_iter)

            preds = predict(x, inverse_radius, mu)
            actuals = torch.tensor(labels).float()

            accuracy, precision, recall, f1 = score(preds, actuals)

            results.append(
                {
                    "selection": int(sel_idx),
                    "seed": int(seed),
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }
            )

    df = pd.DataFrame(results)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(out_dir, f"experiment_randomness_{timestamp}.csv")
    df.to_csv(csv_path, index=False)

    metric_cols = ["accuracy", "precision", "recall", "f1"]
    summary = df.groupby("selection")[metric_cols].agg(["mean", "std"])
    summary_path = os.path.join(out_dir, f"summary_randomness_{timestamp}.csv")
    summary.to_csv(summary_path)

    print(f"Results saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main(n_seeds=10, n_iter=500)
