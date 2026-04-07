from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class AxisAlignedBox(nn.Module):
    def __init__(
        self,
        n_features,
        sparsity_lambda=1e-3,
        k=10.0,
    ):
        super().__init__()

        self.sparsity_lambda = sparsity_lambda
        self.k = k

        self.center = nn.Parameter(torch.randn(n_features))

        self.half_width_uncontrained = nn.Parameter(torch.randn(n_features))
        self.feature_gate_logits = nn.Parameter(torch.ones(n_features))

    def forward(self, x):
        half_width_positive = F.softplus(self.half_width_uncontrained)
        lower = self.center - half_width_positive
        upper = self.center + half_width_positive

        left = torch.sigmoid(self.k * (x - lower))
        right = torch.sigmoid(self.k * (upper - x))
        inside = left * right

        feature_gate = torch.sigmoid(self.feature_gate_logits)

        weighted = inside * feature_gate

        return torch.sum(weighted, dim=1) / torch.sum(feature_gate)

    def sparsity_penalty(self):
        return self.sparsity_lambda * torch.sum(torch.sigmoid(self.feature_gate_logits))


class SparseUnionAxisBoxes(nn.Module):
    def __init__(
        self,
        n_features,
        n_boxes=6,
        feature_lambda=1e-3,
        box_lambda=1e-2,
        k=10.0,
    ):
        super().__init__()

        self.boxes = nn.ModuleList(
            [AxisAlignedBox(n_features, feature_lambda, k=k) for _ in range(n_boxes)]
        )

        self.box_gate_logits = nn.Parameter(torch.ones(n_boxes))

        self.box_lambda = box_lambda

    def forward(self, x):
        box_probs = torch.stack([box(x) for box in self.boxes], dim=1)

        box_gate = torch.sigmoid(self.box_gate_logits)
        weighted = box_gate * box_probs

        return 1 - torch.prod(1 - weighted, dim=1)

    def sparsity_penalty(self):
        feature_penalty = torch.sum(
            torch.stack(
                [cast(AxisAlignedBox, box).sparsity_penalty() for box in self.boxes]
            )
        )

        box_gate = torch.sigmoid(self.box_gate_logits)
        box_penalty = self.box_lambda * torch.sum(box_gate)

        return feature_penalty + box_penalty


def train_model(
    model,
    x,
    y,
    lr=1e-2,
    epochs=500,
    use_balanced_loss=False,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    instance_weights = None
    if use_balanced_loss:
        n_points = y.shape[0]
        n_selected = y.sum()
        n_unselected = n_points - n_selected
        instance_weights = torch.ones_like(y)
        instance_weights[y == 1] = float(n_points) / float(n_selected)
        instance_weights[y == 0] = float(n_points) / float(n_unselected)

    for epoch in range(epochs):
        optimizer.zero_grad()

        preds = model(x)
        bce = F.binary_cross_entropy(preds, y, weight=instance_weights)

        loss = bce + model.sparsity_penalty()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                active = int(torch.sum(torch.sigmoid(model.box_gate_logits) > 0.1))
            loss_value = float(loss.detach())
            print(f"epoch {epoch:4d} | loss {loss_value:.4f} | active boxes {active}")

    return model
