import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.k = k
        self.feature_lambda = feature_lambda
        self.box_lambda = box_lambda

        self.centers = nn.Parameter(torch.randn(n_boxes, n_features))
        self.half_width_unconstrained = nn.Parameter(torch.randn(n_boxes, n_features))
        self.feature_gate_logits = nn.Parameter(torch.ones(n_boxes, n_features))

        self.box_gate_logits = nn.Parameter(torch.ones(n_boxes))

    def forward(self, x):
        x = x.unsqueeze(1)

        half_width = F.softplus(self.half_width_unconstrained)
        lower = self.centers - half_width
        upper = self.centers + half_width

        left = torch.sigmoid(self.k * (x - lower))
        right = torch.sigmoid(self.k * (upper - x))
        inside = left * right

        feature_gate = torch.sigmoid(self.feature_gate_logits)
        weighted = inside * feature_gate
        box_probs = weighted.sum(dim=2) / feature_gate.sum(dim=1)

        box_gate = torch.sigmoid(self.box_gate_logits)
        gated = box_gate * box_probs
        return 1 - torch.prod(1 - gated, dim=1)

    def sparsity_penalty(self, box_lambda_scale=1.0):
        feature_penalty = self.feature_lambda * torch.sum(
            torch.sigmoid(self.feature_gate_logits)
        )
        box_penalty = (
            box_lambda_scale
            * self.box_lambda
            * torch.sum(torch.sigmoid(self.box_gate_logits))
        )
        return feature_penalty + box_penalty


def train_model(
    model,
    x,
    y,
    lr=1e-2,
    epochs=500,
    use_balanced_loss=False,
    box_lambda_warmup_epochs=250,
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

    warmup_epochs = max(1, box_lambda_warmup_epochs)

    for epoch in range(epochs):
        optimizer.zero_grad()

        box_lambda_scale = min(1.0, epoch / warmup_epochs)

        preds = model(x)
        bce = F.binary_cross_entropy(preds, y, weight=instance_weights)

        loss = bce + model.sparsity_penalty(box_lambda_scale=box_lambda_scale)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                active = int(torch.sum(torch.sigmoid(model.box_gate_logits) > 0.1))
            loss_value = float(loss.detach())
            print(
                f"epoch {epoch:4d} | loss {loss_value:.4f} "
                f"| box_λ_scale {box_lambda_scale:.2f} | active boxes {active}"
            )

    return model
