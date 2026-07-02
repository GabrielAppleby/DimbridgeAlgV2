import torch
import torch.nn as nn
import torch.nn.functional as F


def _softplus_inverse(x):
    """
    I just copied this from TF

    https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/math/generic.py#L531-L582
    """
    eps = torch.tensor(torch.finfo(x.dtype).eps)
    threshold = torch.log(eps) + 2.0
    is_too_small = x < torch.exp(threshold)
    is_too_large = x > -threshold
    x_safe = torch.where(is_too_small | is_too_large, torch.ones_like(x), x)
    y = x_safe + torch.log(-torch.expm1(-x_safe))
    return torch.where(is_too_small, torch.log(x), torch.where(is_too_large, x, y))


class SingleBox(nn.Module):

    def __init__(self, x_min, x_max, k=10.0):
        super().__init__()
        self.k = k
        centers_init = (x_min + x_max) / 2.0
        half_width_init = ((x_max - x_min) / 2.0).clamp(min=1e-4)
        self.centers = nn.Parameter(centers_init.clone())
        self.half_width_unconstrained = nn.Parameter(_softplus_inverse(half_width_init))

    def forward(self, x):
        half_width = F.softplus(self.half_width_unconstrained)
        lower = self.centers - half_width
        upper = self.centers + half_width
        left = torch.sigmoid(self.k * (x - lower))
        right = torch.sigmoid(self.k * (upper - x))
        inside = left * right
        return inside.prod(dim=1)

    def width_penalty(self, weight):
        half_width = F.softplus(self.half_width_unconstrained)
        return weight * (2.0 * half_width).sum()


class IterativeRangeModel(nn.Module):

    def __init__(self, n_features, k=10.0):
        super().__init__()
        self.n_features = n_features
        self.k = k
        self.boxes = nn.ModuleList()

    def forward(self, x):
        if len(self.boxes) == 0:
            return torch.zeros(x.shape[0])
        coverages = torch.stack([box(x) for box in self.boxes], dim=1)
        return 1.0 - torch.prod(1.0 - coverages, dim=1)


def _train_single_box(
    x,
    y,
    x_min,
    x_max,
    prior_coverage,
    lr,
    epochs,
    width_penalty_weight,
    overlap_penalty_weight,
    use_balanced_loss,
    k,
    iteration,
):
    box = SingleBox(x_min, x_max, k=k)
    optimizer = torch.optim.Adam(box.parameters(), lr=lr, weight_decay=0.0)

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

        preds = box(x)
        bce = F.binary_cross_entropy(preds, y, weight=instance_weights)
        width_pen = box.width_penalty(width_penalty_weight)

        overlap_pen = x.new_tensor(0.0)
        if prior_coverage is not None:
            overlap_pen = overlap_penalty_weight * (preds * prior_coverage).mean()

        loss = bce + width_pen + overlap_pen
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"  [iter {iteration + 1}] epoch {epoch:4d} | loss {float(loss):.4f}")

    return box


def train_model(
    model,
    x,
    y,
    lr=1e-2,
    epochs=500,
    use_balanced_loss=True,
    width_penalty_weight=1e-3,
    overlap_penalty_weight=1.0,
    min_positive_gain=0.02,
    max_iterations=20,
    prediction_threshold=0.5,
):
    x_min = x.min(dim=0).values
    x_max = x.max(dim=0).values

    n_positives = int(y.sum())
    covered_positive = torch.zeros(y.shape[0], dtype=torch.bool, device=x.device)

    for iteration in range(max_iterations):
        print(f"\n=== Iteration {iteration + 1} / {max_iterations} ===")

        prior_coverage = None
        if len(model.boxes) > 0:
            with torch.no_grad():
                coverages = torch.stack([box(x) for box in model.boxes], dim=1)
                prior_coverage = 1.0 - torch.prod(1.0 - coverages, dim=1)

        box = _train_single_box(
            x=x,
            y=y,
            x_min=x_min,
            x_max=x_max,
            prior_coverage=prior_coverage,
            lr=lr,
            epochs=epochs,
            width_penalty_weight=width_penalty_weight,
            overlap_penalty_weight=overlap_penalty_weight,
            use_balanced_loss=use_balanced_loss,
            k=model.k,
            iteration=iteration,
        )

        with torch.no_grad():
            new_pred = box(x) >= prediction_threshold
            positive_mask = y == 1
            newly_covered = new_pred & positive_mask & ~covered_positive
            gain = newly_covered.sum().item() / max(n_positives, 1)

        print(
            f"  Positive gain: {gain:.4f} "
            f"({newly_covered.sum().item()} new / {n_positives} total positives)"
        )

        for p in box.parameters():
            p.requires_grad_(False)
        model.boxes.append(box)
        covered_positive = covered_positive | (new_pred & positive_mask)

        if gain < min_positive_gain:
            print(
                f"  Stopping: gain {gain:.4f} < min_positive_gain {min_positive_gain:.4f}"
            )
            break

    return model
