import segmentation_models_pytorch as smp
from transformers import (
    Mask2FormerForUniversalSegmentation,
    UperNetForSemanticSegmentation,
)

from .segnet import SegNet


def get_model(name, **model_kwargs):
    model = dict(
        segnet=SegNet,
        unet=smp.Unet,
        unetpp=smp.UnetPlusPlus,
        upernet=smp.UPerNet,
        segformer=smp.Segformer,
    )
    if name not in ["swin", "mask2former"]:
        return model[name](**model_kwargs)
    else:
        model = dict(
            swin=UperNetForSemanticSegmentation.from_pretrained(**model_kwargs),
            mask2former=Mask2FormerForUniversalSegmentation.from_pretrained(
                **model_kwargs
            ),
        )
        return model[name]
