from typing import Literal, Optional, Union

import segmentation_models_pytorch as smp
import torch.nn as nn
from pydantic import BaseModel


class CrossEntropyLossConfig(BaseModel):
    label_smoothing: float = 0.0


class DiceLossConfig(BaseModel):
    mode: Literal["binary", "multiclass", "multilabel"]
    smooth: float = 0.0
    from_logits: bool


class FocalLossConfig(BaseModel):
    mode: Literal["binary", "multiclass", "multilabel"]
    gamma: float = 2.0
    alpha: Optional[float] = None


LossKwargs = Union[CrossEntropyLossConfig, DiceLossConfig, FocalLossConfig]


class SegnetConfig(BaseModel):
    in_channels: int
    out_channels: int


class SMPConfig(BaseModel):
    encoder_name: str
    in_channels: int
    classes: int
    encoder_depth: int = 5
    encoder_weights: Optional[str] = "imagenet"


class HFTransformersConfig(BaseModel):
    pretrained_model_name_or_path: str
    num_labels: int
    ignore_mismatched_sizes: bool = True


ModelKwargs = Union[SegnetConfig, SMPConfig, HFTransformersConfig]


class TrainConfig(BaseModel):
    checkpoint_path: Optional[str] = None

    debug: bool

    model_name: Literal[
        "unet", "unetpp", "swin", "upernet", "segformer", "segnet", "mask2former"
    ]
    dataset_name: Literal["Vaihingen", "DeadTrees", "Potsdam", "LoveDA"]
    version: str

    in_channels: int
    n_classes: int

    patch_size: int
    overlap: int

    batch_size: int
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
