import torchio as tio
from torchgeo.datasets import GeoNRW

from .basevision import BaseVisionDataset


class GeoNRW(BaseVisionDataset, GeoNRW):
    def __getitem__(self, idx):
        image, mask = (
            super().__getitem__(idx)["image"],
            super().__getitem__(idx)["mask"],
        )
        image = image.unsqueeze(-1).float()
        mask = mask.unsqueeze(-1).unsqueeze(0).float()
        return tio.Subject(
            image=tio.ScalarImage(tensor=image),
            mask=tio.LabelMap(tensor=mask),
        )

    @property
    def get_class_names(self):
        class_names = [
            "Background",
            "Impervious surfaces",
            "Building",
            "Low vegetation",
            "Tree",
            "Car",
        ]
        return class_names
