import numpy as np

from .basers import BaseRSDataset


class PotsdamVaihingen(BaseRSDataset):
    @staticmethod
    def _rgb_to_mask(mask):

        color_to_class = {
            (255, 255, 255): 0,
            (0, 0, 255): 1,
            (0, 255, 255): 2,
            (0, 255, 0): 3,
            (255, 255, 0): 4,
            (255, 0, 0): 5,
        }

        h, w, _ = mask.shape
        out = np.zeros((h, w), dtype=np.uint8)

        for color, label in color_to_class.items():
            matches = np.all(mask == color, axis=-1)
            out[matches] = label

        return out

    @property
    def get_class_names(self):
        class_names = [
            "Impervious surfaces",
            "Building",
            "Low vegetation",
            "Tree",
            "Car",
            "Clutter",
        ]
        return class_names
