import argparse

import kornia.augmentation as K
import lightning as L
import segmentation_models_pytorch.losses as smpl
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.losses import DiceLoss
from torchinfo import summary
from torchmetrics.segmentation import DiceScore, MeanIoU

import datamodules.data_classes as DM
from segmentators import RegularTrainingSegmentator

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


VERSION = config["version"]
PROJECT_NAME = config["project_name"]

CHECKPOINT_PATH = (
    f"checkpoints/{PROJECT_NAME}/version_{VERSION}/checkpoints/last.ckpt"
    if config.get("resume")
    else None
)

train_deadtrees_image_glob = config["data"]["train_image_glob"]
train_deadtrees_mask_glob = config["data"]["train_mask_glob"]
val_deadtrees_image_glob = config["data"]["val_image_glob"]
val_deadtrees_mask_glob = config["data"]["val_mask_glob"]

deadtrees_dm = DM.RegularDataModule(
    dataset_cls=getattr(DM, config["data"]["dataset_cls"]),
    train_dataset_kwargs={
        "image_glob": train_deadtrees_image_glob,
        "mask_glob": train_deadtrees_mask_glob,
        "white_threshold": config["data"]["white_threshold"],
    },
    val_dataset_kwargs={
        "image_glob": val_deadtrees_image_glob,
        "mask_glob": val_deadtrees_mask_glob,
        "white_threshold": config["data"]["white_threshold"],
    },
    loader_kwargs={
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["training"]["num_workers"],
    },
)

model = Unet(
    config["encoder_name"],
    encoder_weights=config["encoder_weights"],
    in_channels=config["in_channels"],
    classes=config["classes"],
)


if config["augmentation"]["apply"]:
    horizontal_flip = config["augmentation"]["horizontal_flip"]
    vertical_flip = config["augmentation"]["vertical_flip"]
    brightness = config["augmentation"]["brightness"]
    contrast = config["augmentation"]["contrast"]
    saturation = config["augmentation"]["saturation"]
    hue = config["augmentation"]["hue"]
    gaussian_noise_mean = config["augmentation"]["gaussian_noise_mean"]
    gaussian_noise_std = config["augmentation"]["gaussian_noise_std"]
    gaussian_noise_p = config["augmentation"]["gaussian_noise_p"]


augmentation = (
    dict(
        geom_augmentation=K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"],
        ),
        intensity_augmentation=K.AugmentationSequential(
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.3),
            data_keys=["input"],  # solo imagen
        ),
    )
    if config["augmentation"]["apply"]
    else None
)


segmentator = RegularTrainingSegmentator(
    model=model,
    criterion=getattr(smpl, config["criterion"]["name"])(
        **config["criterion"]["kwargs"]
    ),
    augmentation=augmentation,
    optimizer_cls=getattr(torch.optim, config["architecture"]["optimizer"]),
    optimizer_kwargs=config["architecture"]["optimizer_kwargs"],
    metric=DiceScore(**config["architecture"]["metrics_kwargs"]),
    add_channel_dim=config["architecture"]["add_channel_dim"],
)


checkpoint_callback = ModelCheckpoint(
    dirpath=f"checkpoints/{PROJECT_NAME}/version_{VERSION}/checkpoints",
    filename="{epoch:02d}-{step}-{val_dice:.4f}-{val_loss:.4f}",
    save_top_k=1,  # keep best 3 models
    monitor="val_metric",  # or "val_loss"
    mode="max",  # IoU → maximize
    save_last=True,  # VERY IMPORTANT for resume
)
logger = TensorBoardLogger(save_dir="logs", version=VERSION, name=f"{PROJECT_NAME}")
trainer = L.Trainer(
    max_epochs=config["trainer"]["num_epochs"],
    logger=logger,
    callbacks=[checkpoint_callback],
    accelerator=config["trainer"]["accelerator"],
    devices=config["trainer"]["devices"],
    enable_progress_bar=True,
    deterministic=False,  ## Not possible for CE
    check_val_every_n_epoch=1,
    fast_dev_run=config["fast_dev_run"],
    overfit_batches=config["overfit_batches"],
    log_every_n_steps=1,
    precision=config["trainer"]["precision"],
    accumulate_grad_batches=config["trainer"]["accumulate_grad_batches"],
)

trainer.fit(segmentator, deadtrees_dm, ckpt_path=CHECKPOINT_PATH)

summary(model)
