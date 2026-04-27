import torchio as tio
from torchgeo.datasets import LoveDA

from .basevision import BaseVisionDataset


class LoveDAds(BaseVisionDataset, LoveDA):
    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx).values()
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
            "Building",
            "Road",
            "Water",
            "Barren",
            "Forest",
            "Agricultural land",
        ]
        return class_names
