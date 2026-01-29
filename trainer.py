import torch
from torch import nn, optim


def predict(x, inverse_radius_param, mu_param):
    b = 3
    return 1 / (1 + ((inverse_radius_param.abs() * (x - mu_param).abs()).pow(b)).sum(1))


def optimize_weights(x, labels, n_iter=1000, inverse_radius_init=0.4):
    n_points, n_features = x.shape

    mu_init = x[labels].mean(0)
    inverse_radius_parameter = nn.Parameter(
        inverse_radius_init + 0.1 * (2 * torch.rand(n_features, dtype=x.dtype) - 1)
    )
    mu_param = nn.Parameter(
        mu_init + 0.1 * (2 * torch.rand(n_features, dtype=x.dtype) - 1)
    )

    n_selected = labels.sum()
    n_unselected = n_points - n_selected
    instance_weight = torch.ones(x.shape[0])
    instance_weight[labels] = n_points / n_selected
    instance_weight[~labels] = n_points / n_unselected
    bce = nn.BCELoss(weight=instance_weight)
    optimizer = optim.SGD(
        [
            {"params": mu_param, "weight_decay": 0},
            {"params": inverse_radius_parameter, "weight_decay": 0.01},
        ],
        lr=1e-2,
        momentum=0.9,
    )

    y = torch.tensor(labels, dtype=torch.float32)

    for _ in range(n_iter):
        pred = predict(x, inverse_radius_parameter, mu_param)
        loss = bce(pred, y)
        loss += (mu_param - mu_init).pow(2).mean() * 20
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    inverse_radius_parameter = inverse_radius_parameter.detach()
    mu_param = mu_param.detach()

    return mu_param, inverse_radius_parameter
