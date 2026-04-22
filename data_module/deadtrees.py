import glob
from pathlib import Path

import rasterio

from .basers import BaseRSDataset


class DeadTrees(BaseRSDataset):
    def __init__(self, image_glob, white_threshold=0.1, transform=None):
        self.image_ids = self._detect_non_border_images(
            glob.glob(image_glob), white_threshold
        )
        self.image_paths = [
            f"Dataset_Templado/dataset_rgb/{img_id}.tif" for img_id in self.image_ids
        ]
        self.mask_paths = [
            f"Dataset_Templado/dataset_binary/{img_id}.tif" for img_id in self.image_ids
        ]
        self.transform = transform
        self.to_rgb = False

    @staticmethod
    def _detect_non_border_images(image_list, percentage_threshold=0.1):
        blank_image_ids = []
        for path in image_list:
            image = rasterio.open(path).read().transpose(1, 2, 0)
            ## Detecting Images with any number of white Pixels
            if (image.sum(axis=2) == 3 * 255).sum() / image[
                :, :, 0
            ].size <= percentage_threshold:
                blank_image_ids.append(Path(path).stem)

        return blank_image_ids

    @property
    def get_class_names(self):
        class_names = ["Non-Dead Tree", "Dead Tree"]
        return class_names
