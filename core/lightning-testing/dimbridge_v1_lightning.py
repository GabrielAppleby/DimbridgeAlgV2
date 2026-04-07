import pytorch_lightning as pl
import torch
from torch import nn, optim


class DimBridgeModule(pl.LightningModule):

    def __init__(
        self,
        n_features,
        mu_init,
        a_init=0.4,
        lr=1e-2,
        mu_weight_decay=0.01,
        a_weight_decay=0,
        momentum=0.9,
    ):
        super().__init__()
        self.n_features = n_features
        self.lr = lr
        self.mu_weight_decay = mu_weight_decay
        self.a_weight_decay = a_weight_decay
        self.momentum = momentum

        # Assigning mu_init before registering just to stop stupid linter
        # from complaininig...
        self.mu_init = mu_init
        self.register_buffer("mu_init", mu_init, persistent=False)

        self.a = nn.Parameter(
            a_init + 0.1 * (2 * torch.rand(n_features, dtype=torch.float32) - 1)
        )
        self.mu = nn.Parameter(
            mu_init + 0.1 * (2 * torch.rand(n_features, dtype=torch.float32) - 1)
        )

    def forward(self, x):
        b = 3
        return 1 / (1 + ((self.a.abs() * (x - self.mu).abs()).pow(b)).sum(1))

    def training_step(self, batch, batch_idx):
        x, labels = batch

        n_points = x.shape[0]
        n_selected = labels.sum()
        n_unselected = n_points - n_selected

        instance_weight = torch.ones(n_points, dtype=torch.float32, device=x.device)
        instance_weight[labels.bool()] = float(n_points) / float(n_selected)
        instance_weight[~labels.bool()] = float(n_points) / float(n_unselected)

        bce = nn.BCELoss(weight=instance_weight)

        preds = self.forward(x)
        labels_double = labels.to(torch.float32)
        loss = bce(preds, labels_double)

        loss = loss + (self.mu - self.mu_init).pow(2).mean() * 20

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(
            [
                {"params": self.mu, "weight_decay": self.mu_weight_decay},
                {"params": self.a, "weight_decay": self.a_weight_decay},
            ],
            lr=self.lr,
            momentum=self.momentum,
        )
        return optimizer
