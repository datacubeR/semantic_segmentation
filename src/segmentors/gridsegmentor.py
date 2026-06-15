import lightning as L
import torch
import torch.nn as nn
import torchio as tio
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.segmentation import MeanIoU


class GridSegmentor(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
        image_processor: nn.Module | None = None,
        criterion: nn.Module | None = None,
        hf_model: str | None = None,
        patch_size: int = 256,
        overlap: int = 32,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.model = model.train()
        self.image_processor = image_processor
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

    # def forward(self, **kwargs):
    #     return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        X, y = batch["image"], batch["mask"]

        if self.hf_model == "swin":
            output = self(X, labels=y)
            loss, logits = output.loss, output.logits

        # elif self.hf_model == "mask2former":
        #     inputs = self.image_processor(X, segmentation_maps=y, return_tensors="pt")
        #     outputs = self(**inputs)
        #     loss = outputs.loss
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
        image, mask = (
            batch["image"].unsqueeze(-1).squeeze(0),
            batch["mask"].unsqueeze(-1),
        )

        avg_loss, y_hat = self._forward_grid_batch(image, mask)
        gt = mask.squeeze(-1).long()

        self.log(
            "losses/val_loss",
            avg_loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
            logger=True,
        )

        self.val_metrics.update(y_hat, gt)
        # self.accuracy.update(y_hat, gt)
        # self.recall.update(y_hat, gt)
        # self.precision.update(y_hat, gt)
        # self.f1.update(y_hat, gt)
        # self.miou.update(y_hat, gt)
        return None

    def test_step(self, batch, batch_idx):
        image, mask = (
            batch["image"].unsqueeze(-1).squeeze(0),
            batch["mask"].unsqueeze(-1),
        )

        avg_loss, y_hat = self._forward_grid_batch(image, mask)
        gt = mask.squeeze(-1).long()

        self.log(
            "losses/test_loss",
            avg_loss,
            prog_bar=True,
            batch_size=1,
            logger=True,
        )

        self.test_metrics.update(y_hat, gt)

        # self.accuracy.update(y_hat, gt)
        # self.recall.update(y_hat, gt)
        # self.precision.update(y_hat, gt)
        # self.f1.update(y_hat, gt)
        # self.miou.update(y_hat, gt)
        return None

    def _forward_grid_batch(self, image, mask):

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

                if self.hf_model == "swin":
                    output = self(X, labels=y)
                    loss, logits = output.loss, output.logits

                # elif self.hf_model == "mask2former":
                #     inputs = self.image_processor(
                #         X, segmentation_maps=y, return_tensors="pt"
                #     )
                #     outputs = self(**inputs)
                #     loss = outputs.loss
                #     logits = self._post_process(
                #         outputs,
                #         patch_size=self.hparams.patch_size,
                #         batch_size=X.shape[0],
                #     )
                else:
                    logits = self(X)
                    loss = self.criterion(logits, y)

                aggregator.add_batch(
                    logits.detach().cpu().unsqueeze(-1), patch[tio.LOCATION]
                )

                total_loss += loss.item()
                n_patches += 1

        avg_loss = total_loss / n_patches
        full_pred = aggregator.get_output_tensor().squeeze(-1)

        # if self.hf_model == "mask2former":
        #     y_hat = full_pred.long().to(self.device)
        # else:
        y_hat = full_pred.argmax(dim=0).unsqueeze(0).to(self.device)
        return avg_loss, y_hat

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

    # def on_validation_epoch_end(self):
    # accuracy = self.accuracy.compute()
    # recall = self.recall.compute()
    # precision = self.precision.compute()
    # f1 = self.f1.compute()
    # miou = self.miou.compute()

    # self.log(
    #     "metrics/val_iou",
    #     miou,
    #     on_epoch=True,
    #     prog_bar=True,
    #     batch_size=1,
    #     logger=True,
    # )
    # self.log(
    #     "metrics/val_accuracy",
    #     accuracy,
    #     on_epoch=True,
    #     prog_bar=False,
    #     batch_size=1,
    #     logger=True,
    # )
    # self.log(
    #     "metrics/val_recall",
    #     recall,
    #     on_epoch=True,
    #     prog_bar=False,
    #     batch_size=1,
    #     logger=True,
    # )
    # self.log(
    #     "metrics/val_precision",
    #     precision,
    #     on_epoch=True,
    #     prog_bar=False,
    #     batch_size=1,
    #     logger=True,
    # )
    # self.log(
    #     "metrics/val_f1",
    #     f1,
    #     on_epoch=True,
    #     prog_bar=True,
    #     batch_size=1,
    #     logger=True,
    # )
    # self.accuracy.reset()
    # self.recall.reset()
    # self.precision.reset()
    # self.f1.reset()
    # self.miou.reset()

    # def on_fit_start(self):
    #     self.val_metrics = self.val_metrics.to(self.device)
    #     self.test_metrics = self.test_metrics.to(self.device)

    # def on_fit_start(self):
    #     self.f1 = self.f1.to(self.device)
    #     self.accuracy = self.accuracy.to(self.device)
    #     self.recall = self.recall.to(self.device)
    #     self.precision = self.precision.to(self.device)
    #     self.miou = self.miou.to(self.device)

    # def _post_process(self, output, patch_size, batch_size):
    #     processor_output = self.image_processor.post_process_semantic_segmentation(
    #         output, target_sizes=[(patch_size, patch_size)] * batch_size
    #     )
    #     return torch.stack(processor_output).unsqueeze(1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
