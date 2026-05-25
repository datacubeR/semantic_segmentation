import lightning as L
import torch
import torchio as tio
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.segmentation import MeanIoU


class Segmentator(L.LightningModule):
    def __init__(
        self,
        model,
        n_classes,
        criterion,
        patch_size=256,
        overlap=32,
        batch_size=16,
        lr=1e-3,
        weight_decay=1e-4,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.accuracy = MulticlassAccuracy(
            num_classes=n_classes,
            average="macro",
            ignore_index=None,
            multidim_average="global",
        )
        self.recall = MulticlassRecall(
            num_classes=n_classes,
            average="macro",
            ignore_index=None,
            multidim_average="global",
        )
        self.precision = MulticlassPrecision(
            num_classes=n_classes,
            average="macro",
            ignore_index=None,
            multidim_average="global",
        )
        self.f1 = MulticlassF1Score(
            num_classes=n_classes,
            average="macro",
            ignore_index=None,
            multidim_average="global",
        )
        self.miou = MeanIoU(
            num_classes=n_classes,
            include_background=True,
            per_class=False,
            input_format="index",
        )
        self.save_hyperparameters(ignore=["model", "criterion"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch["image"], batch["mask"]
        logits = self(X)
        loss = self.criterion(logits, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = (
            batch["image"].unsqueeze(-1).squeeze(0),
            batch["mask"].unsqueeze(-1),
        )

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image), mask=tio.LabelMap(tensor=mask)
        )
        sampler = tio.GridSampler(
            subject,
            patch_size=(self.hparams.patch_size, self.hparams.patch_size, 1),
            patch_overlap=(self.hparams.overlap, self.hparams.overlap, 0),
        )
        aggregator = tio.GridAggregator(sampler)
        with torch.no_grad():
            n_patches = 0
            total_loss = 0
            for patch in DataLoader(
                sampler, batch_size=self.hparams.batch_size, shuffle=False
            ):
                X, y = (
                    patch["image"][tio.DATA].to(self.device),
                    patch["mask"][tio.DATA].to(self.device),
                )
                X = X.squeeze(-1)
                y = y.squeeze(-1).squeeze(1).long()

                pred = self(X)
                aggregator.add_batch(
                    pred.detach().cpu().unsqueeze(-1), patch[tio.LOCATION]
                )

                loss = self.criterion(pred, y)
                total_loss += loss.item()
                n_patches += 1

        avg_loss = total_loss / n_patches
        self.log(
            "val_loss",
            avg_loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
            logger=True,
        )
        full_pred = aggregator.get_output_tensor().squeeze(-1)

        y_hat = full_pred.argmax(dim=0).unsqueeze(0).to(self.device)
        gt = mask.squeeze(-1).long()

        self.accuracy.update(y_hat, gt)
        self.recall.update(y_hat, gt)
        self.precision.update(y_hat, gt)
        self.f1.update(y_hat, gt)
        self.miou.update(y_hat, gt)
        return None

    def on_validation_epoch_end(self):
        accuracy = self.accuracy.compute()
        recall = self.recall.compute()
        precision = self.precision.compute()
        f1 = self.f1.compute()
        miou = self.miou.compute()

        self.log(
            "val_iou", miou, on_epoch=True, prog_bar=True, batch_size=1, logger=True
        )
        self.log(
            "val_accuracy",
            accuracy,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
            logger=True,
        )
        self.log(
            "val_recall",
            recall,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
            logger=True,
        )
        self.log(
            "val_precision",
            precision,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
            logger=True,
        )
        self.log(
            "val_f1",
            f1,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
            logger=True,
        )
        self.accuracy.reset()
        self.recall.reset()
        self.precision.reset()
        self.f1.reset()
        self.miou.reset()

    def on_fit_start(self):
        self.f1 = self.f1.to(self.device)
        self.accuracy = self.accuracy.to(self.device)
        self.recall = self.recall.to(self.device)
        self.precision = self.precision.to(self.device)
        self.miou = self.miou.to(self.device)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
