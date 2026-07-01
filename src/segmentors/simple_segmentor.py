import lightning as L
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.segmentation import MeanIoU


class SimpleSegmentor(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
        criterion: nn.Module | None = None,
        hf_model: str | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.model = model.train()
        self.criterion = criterion
        self.hf_model = hf_model

        self.val_metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(
                    num_classes=n_classes,
                    average="macro",
                    ignore_index=None,
                    multidim_average="global",
                ),
                "recall": MulticlassRecall(
                    num_classes=n_classes,
                    average="macro",
                    ignore_index=None,
                    multidim_average="global",
                ),
                "precision": MulticlassPrecision(
                    num_classes=n_classes,
                    average="macro",
                    ignore_index=None,
                    multidim_average="global",
                ),
                "f1": MulticlassF1Score(
                    num_classes=n_classes,
                    average="macro",
                    ignore_index=None,
                    multidim_average="global",
                ),
                "miou": MeanIoU(
                    num_classes=n_classes,
                    include_background=True,
                    per_class=False,
                    input_format="index",
                ),
            },
            prefix="val_metrics/",
        )
        self.test_metrics = self.val_metrics.clone(prefix="test_metrics/")
        self.save_hyperparameters(
            ignore=["model", "criterion", "hf_model", "val_metrics", "test_metrics"]
        )

    def forward(self, x, labels=None):
        if labels is not None:
            return self.model(x, labels=labels)

        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch["image"], batch["mask"]

        if self.hf_model == "swin":
            output = self(X, labels=y)
            loss, logits = output.loss, output.logits

        else:
            logits = self(X)
            loss = self.criterion(logits, y)
        self.log(
            "losses/train_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch["image"], batch["mask"]
        loss, y_hat = self._forward_batch(X, y)

        self.log(
            "losses/val_loss",
            loss.item(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.val_metrics.update(y_hat, y)
        return None

    def test_step(self, batch, batch_idx):
        X, y = batch["image"], batch["mask"]
        loss, y_hat = self._forward_batch(X, y)

        self.log(
            "losses/test_loss",
            loss.item(),
            prog_bar=True,
            logger=True,
        )

        self.test_metrics.update(y_hat, y)
        return None

    def predict_step(self, batch, batch_idx):
        X, y = batch["image"], batch["mask"]
        _, y_hat = self._forward_batch(X, y)
        return y_hat, y

    def _forward_batch(self, image, mask):

        if self.hf_model == "swin":
            output = self(image, labels=mask)
            loss, logits = output.loss, output.logits

        else:
            logits = self(image)
            loss = self.criterion(logits, mask)

        y_hat = logits.argmax(dim=1)

        return loss, y_hat

    def on_validation_epoch_end(self):
        self.log_dict(
            self.val_metrics.compute(), prog_bar=True, logger=True, batch_size=1
        )
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(
            self.test_metrics.compute(), prog_bar=True, logger=True, batch_size=1
        )
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
