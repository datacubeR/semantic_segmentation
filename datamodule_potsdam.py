import warnings

import torchio as tio
from rasterio.errors import NotGeoreferencedWarning

from data_module import GridDataModule, PotsdamVaihingen

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


train_potsdam_image_glob = "Potsdam_dataset/train/images/*.tif"
train_potsdam_mask_glob = "Potsdam_dataset/train/masks/*.tif"

val_potsdam_image_glob = "Potsdam_dataset/val/images/*.tif"
val_potsdam_mask_glob = "Potsdam_dataset/val/masks/*.tif"

dm = GridDataModule(
    dataset_cls=PotsdamVaihingen,
    train_dataset_kwargs={
        "image_glob": train_potsdam_image_glob,
        "mask_glob": train_potsdam_mask_glob,
        "reduce_mask": True,
    },
    val_dataset_kwargs={
        "image_glob": val_potsdam_image_glob,
        "mask_glob": val_potsdam_mask_glob,
        "reduce_mask": True,
    },
    loader_kwargs={
        "batch_size": 16,
        "num_workers": 0,
    },
)
dm.setup()
train_loader = dm.train_dataloader()

count = 0
for batch in train_loader:
    print(batch["image"][tio.DATA].shape, batch["mask"][tio.DATA].shape)
