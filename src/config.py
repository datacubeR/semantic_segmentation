from typing import Literal

import segmentation_models_pytorch as smp
import torch.nn as nn
from pydantic import BaseModel, ConfigDict


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CrossEntropyLossConfig(BaseModel):
    ignore_index: int = -100


class DiceLossConfig(BaseModel):
    mode: Literal["binary", "multiclass", "multilabel"]
    smooth: float = 0.0
    from_logits: bool
    ignore_index: int | None = None


class FocalLossConfig(BaseModel):
    mode: Literal["binary", "multiclass", "multilabel"]
    gamma: float = 2.0
    alpha: float | None = None


LossKwargs = CrossEntropyLossConfig | DiceLossConfig | FocalLossConfig


class LRSchedulerKwargs(BaseModel):
    mode: str
    factor: float
    patience: int
    threshold: float
    threshold_mode: str
    min_lr: float


class SegnetConfig(BaseModel):
    in_channels: int
    out_channels: int


class SMPConfig(BaseModel):
    encoder_name: str
    in_channels: int
    classes: int
    encoder_depth: int = 5
    encoder_weights: str | None = "imagenet"


class HFTransformersConfig(BaseModel):
    pretrained_model_name_or_path: str
    num_labels: int
    ignore_mismatched_sizes: bool = True
    loss_ignore_index: int = 255


ModelKwargs = SegnetConfig | SMPConfig | HFTransformersConfig


class TrainConfig(ConfigBaseModel):
    checkpoint_path: str | None = None

    debug: bool

    model_name: Literal[
        "unet", "unetpp", "swin", "upernet", "segformer", "segnet", "dpt", "deeplab"
    ]
    dataset_name: Literal["Vaihingen", "DeadTrees", "Potsdam", "LoveDA"]
    save_top_k: int

    in_channels: int
    n_classes: int

    patch_size: int | None = None
    overlap: int | None = None
    image_size: int | None = None

    use_scheduler: bool = False
    scheduler_monitor: str | None = None
    lr_scheduler_kwargs: LRSchedulerKwargs

    batch_size: int
    val_test_batch_size: int = 1
    max_epochs: int
    grad_accumulation_batches: int

    precision: Literal[
        "16-mixed",
        "32",
        "bf16-mixed",
    ]

    lr: float
    weight_decay: float

    model: str

    loss_function: Literal["cross_entropy", "dice_loss", "focal_loss"] | None
    loss_kwargs: LossKwargs | None

    model_kwargs: ModelKwargs

    def get_loss(self):
        if self.loss_function is None:
            return None

        loss_map = {
            "cross_entropy": nn.CrossEntropyLoss,
            "dice_loss": smp.losses.DiceLoss,
            "focal_loss": smp.losses.FocalLoss,  ## Optional: Maybe won't be used during experiments.
        }

        return loss_map[self.loss_function](**self.loss_kwargs.model_dump())
