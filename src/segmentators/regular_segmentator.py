import kornia.geometry as kg
import lightning as L
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid


class RegularTrainingSegmentator(L.LightningModule):
    def __init__(
        self,
        model,
        criterion,
        augmentation,
        metric,
        optimizer_cls,
        optimizer_kwargs,
        n_tb_images=4,
        tb_size=(128, 128),
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
        self.n_tb_images = n_tb_images
        self.tb_size = tb_size
        self.lr = optimizer_kwargs["lr"]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch["image"], batch["mask"]

        if self.augmentation is not None:
            X = self.augmentation["intensity_augmentation"](X)
            X, y = self.augmentation["geom_augmentation"](X, y)

        y = y.squeeze(1)  # Remove channel dimension for CE loss

        ## For DiceLoss I need to one-hot encode the target.
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

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        # Saves the first batch once for validation purposes in TensorBoard
        if batch_idx == 0 and self.current_epoch == 0:
            self.tensorboard_images, self.tensorboard_masks = (
                batch["image"][: self.n_tb_images],
                batch["mask"][: self.n_tb_images],
            )

    def on_validation_epoch_end(self):
        miou = self.metric.compute()
        self.log("val_metric", miou, on_epoch=True, prog_bar=True, logger=True)
        self.metric.reset()

        with torch.no_grad():
            logits = self(self.tensorboard_images.to(self.device))
            preds = logits.argmax(dim=1)

        resized_image = kg.resize(self.tensorboard_images, self.tb_size)
        resized_mask = (
            kg.resize(
                self.tensorboard_masks.float(), self.tb_size, interpolation="nearest"
            )
            .squeeze(1)
            .long()
        )
        resized_preds = (
            kg.resize(preds.float(), self.tb_size, interpolation="nearest")
            .squeeze(1)
            .long()
        )

        pred_rgb = self._mask_to_rgb(resized_preds)
        mask_rgb = self._mask_to_rgb(resized_mask)
        grid = make_grid(
            torch.cat([resized_image, mask_rgb, pred_rgb], dim=0),
            nrow=4,
            normalize=True,
        )
        self.logger.experiment.add_image("val/img|gt|pred", grid, self.current_epoch)

    def _mask_to_rgb(self, mask):
        colors = (
            torch.tensor(
                ## give me a list of several colors in rgb
                [
                    [0, 0, 0],  # class 0 -> black
                    [255, 0, 0],  # class 1 -> red
                    [0, 255, 0],  # class 2 -> green
                    [0, 0, 255],  # class 3 -> blue
                    [255, 255, 0],  # class 4 -> yellow
                    [255, 0, 255],  # class 5 -> magenta
                    [0, 255, 255],  # class 6 -> cyan
                    [128, 128, 128],  # class 7 -> gray
                    [68, 55, 39],  # class 8 -> brown
                    [255, 165, 0],  # class 9 -> orange
                    [128, 0, 128],  # class 10 -> purple
                ],
                dtype=torch.float32,
                device=mask.device,
            )
            / 255.0
        )

        # B, H, W = mask.shape
        rgb = colors[mask.long()]  # (B, H, W, 3)
        return rgb.permute(0, 3, 1, 2)  # (B, 3, H, W)

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters(), **self.optimizer_kwargs)
        return optimizer
