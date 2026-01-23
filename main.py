import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_california_housing
from torch import nn, optim


def predict(x, a, mu):
    b = 3
    return 1 / (1 + ((a.abs() * (x - mu).abs()).pow(b)).sum(1))


def compute_predicate(x0, selected, n_iter=1000, mu_init=None, a_init=0.4):
    n_points, n_features = x0.shape
    vmin = x0.min(0)
    vmax = x0.max(0)
    x = torch.from_numpy(x0.astype(np.float32))
    label = torch.from_numpy(selected).float()

    mean = x.mean(0)
    scale = x.std(0) + 0.1
    x = (x - mean) / scale

    center_selected = x[selected].mean(0)
    if mu_init is None:
        mu_init = center_selected
    a = a_init + 0.1 * (2 * torch.rand(n_features) - 1)
    mu = mu_init + 0.1 * (2 * torch.rand(x.shape[1]) - 1)
    a.requires_grad_(True)
    mu.requires_grad_(True)

    n_selected = selected.sum()
    n_unselected = n_points - n_selected
    instance_weight = torch.ones(x.shape[0])
    instance_weight[selected] = n_points / n_selected
    instance_weight[~selected] = n_points / n_unselected
    bce = nn.BCELoss(weight=instance_weight)
    optimizer = optim.SGD(
        [
            {"params": mu, "weight_decay": 0},
            {"params": a, "weight_decay": 0.01},
        ],
        lr=1e-2,
        momentum=0.9,
    )

    for e in range(n_iter):
        pred = predict(x, a, mu)
        loss = bce(pred, label)
        loss += (mu - center_selected).pow(2).mean() * 20
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if e % (n_iter // 5) == 0:
            print(f"[{e:>4}] loss {loss.item()}")
    a.detach_()
    mu.detach_()

    pred = (pred > 0.5).float()
    correct = (pred == label).float().sum().item()
    total = selected.shape[0]
    accuracy = correct / total
    tp = ((pred == 1).float() * (label == 1).float()).sum().item()
    fp = ((pred == 1).float() * (label == 0).float()).sum().item()
    fn = ((pred == 0).float() * (label == 1).float()).sum().item()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 1 / (1 / precision + 1 / recall)
    print(
        f"""
            accuracy = {correct/total}
            precision = {precision}
            recall = {recall}
            f1 = {f1}
        """
    )

    r = 1 / a.abs()
    predicates = []
    for k in range(mu.shape[0]):
        r_k = (r[k] * scale[k]).item()
        mu_k = (mu[k] * scale[k] + mean[k]).item()
        ci = [mu_k - r_k, mu_k + r_k]
        assert ci[0] < ci[1], "ci[0] is not less than ci[1]"
        if ci[0] < vmin[k]:
            ci[0] = vmin[k]
        if ci[1] > vmax[k]:
            ci[1] = vmax[k]
        should_include = not (ci[0] <= vmin[k] and ci[1] >= vmax[k])
        if should_include:
            predicates.append(dict(dim=k, interval=ci))
    for p in predicates:
        print(p)
    return predicates, mu, a, [accuracy, precision, recall, f1]


if __name__ == '__main__':
    housing = fetch_california_housing(as_frame=True)
    df = housing['data']
    x0 = df.to_numpy()
    t = 5
    selected = np.zeros([t,x0.shape[0]]).astype(np.bool)
    selected[0, df['AveRooms']<6.0] = 1
    selected[1, df['AveBedrms']<1.3] = 1
    selected[2, df['Population']<400.0] = 1
    selected[3, df['AveOccup']<2.3] = 1
    selected[4, df['MedInc']<5.0] = 1

    print('selected', selected.sum(1), 'points in the sequence...')

    for i in range(5):
        predicate, mu, a, quality = compute_predicate(x0, selected[i], n_iter=1000, mu_init=None, a_init=0.4)
