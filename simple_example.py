import torch

from dataloader import get_cali_housing_example
from predicates import create_predicates
from preprocessing import standardize_data
from trainer import optimize_weights


def main():
    x, labels = get_cali_housing_example()
    feature_min = x.min(0).tolist()
    feature_max = x.max(0).tolist()
    x = torch.tensor(x, dtype=torch.float32)
    x, feature_mean, feature_std = standardize_data(x)
    for i in range(5):
        learned_mean, learned_inverse_radius = optimize_weights(
            x, labels[i], n_iter=1000, inverse_radius_init=0.4
        )
        create_predicates(
            learned_mean,
            learned_inverse_radius,
            feature_mean,
            feature_std,
            feature_min,
            feature_max,
        )


if __name__ == "__main__":
    main()
