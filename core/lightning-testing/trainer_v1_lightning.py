import pytorch_lightning as pl
from dimbridge_v1_lightning import DimBridgeModule
from torch.utils.data import DataLoader, TensorDataset


def train(x, labels, n_iter=1000, a_init=0.4, lr=1e-2):
    module = DimBridgeModule(
        n_features=x.shape[1], mu_init=x[labels].mean(0), a_init=a_init, lr=lr
    )

    dataset = TensorDataset(x, labels)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    trainer = pl.Trainer(
        max_epochs=n_iter,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        accelerator="cpu",
    )
    trainer.fit(module, dataloader)

    return module.mu.detach(), module.a.detach()
