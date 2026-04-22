import glob

import rasterio
import torch
import torchio as tio

from .basevision import BaseVisionDataset


class BaseRSDataset(BaseVisionDataset):
    def __init__(self, image_glob, mask_glob, to_rgb=False, transform=None):
        self.image_paths = sorted(glob.glob(image_glob))
        self.mask_paths = sorted(glob.glob(mask_glob))
        self.transform = transform
        self.to_rgb = to_rgb

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        # Load the image and mask using rasterio
        with rasterio.open(image_path) as img:
            ## CHW
            image = img.read()
        with rasterio.open(mask_path) as msk:
            # HWC for conversion
            if self.to_rgb:
                mask = self._rgb_to_mask(msk.read().transpose(1, 2, 0))
            else:
                mask = msk.read().squeeze()

        # Apply transformations if any
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        img = torch.from_numpy(image).unsqueeze(-1).float()
        mask = torch.from_numpy(mask).unsqueeze(-1).unsqueeze(0).float()

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img),
            mask=tio.LabelMap(tensor=mask),
        )
        return subject
