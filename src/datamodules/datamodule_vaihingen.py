import warnings

import torchio as tio
from rasterio.errors import NotGeoreferencedWarning

from .data_classes import GridDataModule, PotsdamVaihingen

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


train_vaihingen_image_glob = "Vaihingen_dataset/train/images/*.tif"
train_vaihingen_mask_glob = "Vaihingen_dataset/train/masks/*.tif"
val_vaihingen_image_glob = "Vaihingen_dataset/val/images/*.tif"
val_vaihingen_mask_glob = "Vaihingen_dataset/val/masks/*.tif"

vaihingen_dm = GridDataModule(
    dataset_cls=PotsdamVaihingen,
    train_dataset_kwargs={
        "image_glob": train_vaihingen_image_glob,
        "mask_glob": train_vaihingen_mask_glob,
        "reduce_mask": True,
    },
    val_dataset_kwargs={
        "image_glob": val_vaihingen_image_glob,
        "mask_glob": val_vaihingen_mask_glob,
        "reduce_mask": True,
    },
    loader_kwargs={
        "batch_size": 1,
        "num_workers": 0,
    },
)

if __name__ == "__main__":
    vaihingen_dm.setup()
    train_loader = vaihingen_dm.train_dataloader()

    for batch in train_loader:
        print(batch["image"].shape, batch["mask"].shape)
        break
