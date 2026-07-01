import glob

import rasterio
import torch

from .basevision import BaseVisionDataset


class BaseRSDataset(BaseVisionDataset):
    def __init__(
        self,
        image_glob: str,
        mask_glob: str,
        reduce_mask: bool = False,
        squeeze_mask: bool = False,
    ):
        self.image_paths = sorted(glob.glob(image_glob))
        self.mask_paths = sorted(glob.glob(mask_glob))
        self.reduce_mask = reduce_mask
        self.squeeze_mask = squeeze_mask

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
            if self.reduce_mask:
                mask = self._rgb_to_mask(msk.read().transpose(1, 2, 0))
            else:
                mask = msk.read().squeeze()

        img = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).unsqueeze(0).long()  # Masks should be Integers
        if self.squeeze_mask:
            mask = mask.squeeze(0)

        return dict(image=img, mask=mask)
