import segmentation_models_pytorch as smp
import torch.nn as nn


def get_loss(name, **loss_kwargs):
    loss = dict(
        cross_entropy=nn.CrossEntropyLoss,
        dice_loss=smp.losses.DiceLoss,
        focal_loss=smp.losses.FocalLoss,
    )
    return loss[name](**loss_kwargs)
