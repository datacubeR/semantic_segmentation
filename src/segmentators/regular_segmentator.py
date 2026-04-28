import lightning as L
import torch.nn.functional as F


class RegularTrainingSegmentator(L.LightningModule):
    def __init__(
        self,
        model,
        criterion,
        augmentation,
        metric,
        optimizer_cls,
        optimizer_kwargs,
        add_channel_dim=False,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.augmentation = augmentation
        self.metric = metric
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.add_channel_dim = add_channel_dim

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch["image"], batch["mask"]

        if self.augmentation is not None:
            X = self.augmentation["intensity_augmentation"](X)
            X, y = self.augmentation["geom_augmentation"](X, y)

        y = y.squeeze(1)  # Remove channel dimension for CE loss

        ## For Dice I need to one-hot encode the target.
        if self.add_channel_dim:
            y_ohe = F.one_hot(y, num_classes=2).permute(0, 3, 1, 2)
            y_ohe = y_ohe.contiguous()

        logits = self(X)
        if self.add_channel_dim:
            loss = self.criterion(logits, y_ohe)
        else:
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
        X, y = batch["image"], batch["mask"]

        y = y.squeeze(1)  # Remove channel dimension for CE loss

        ## For Dice I need to one-hot encode the target.
        if self.add_channel_dim:
            y_ohe = F.one_hot(y, num_classes=2).permute(0, 3, 1, 2)
            y_ohe = y_ohe.contiguous()

        logits = self(X)
        if self.add_channel_dim:
            loss = self.criterion(logits, y_ohe)
        else:
            loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.metric(preds, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return None

    def on_fit_start(self):
        self.metric = self.metric.to(self.device)

    def on_validation_epoch_end(self):
        miou = self.metric.compute()
        self.log(
            "val_dice", miou, on_epoch=True, prog_bar=True, batch_size=1, logger=True
        )
        self.metric.reset()

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters(), **self.optimizer_kwargs)
        return optimizer
