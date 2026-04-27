import warnings

import torchio as tio
from rasterio.errors import NotGeoreferencedWarning

from data_classes import GID15, GridDataModule

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


train_gid15_image_glob = "GID15/gid-15/GID/img_dir/train/*.tif"
train_gid15_mask_glob = "GID15/gid-15/GID/ann_dir/train/*.png"

val_gid15_image_glob = "GID15/gid-15/GID/img_dir/val/*.tif"
val_gid15_mask_glob = "GID15/gid-15/GID/ann_dir/val/*.png"

dm = GridDataModule(
    dataset_cls=GID15,
    train_dataset_kwargs={
        "image_glob": train_gid15_image_glob,
        "mask_glob": train_gid15_mask_glob,
    },
    val_dataset_kwargs={
        "image_glob": val_gid15_image_glob,
        "mask_glob": val_gid15_mask_glob,
    },
    dl_kwargs={
        "batch_size": 16,
        "num_workers": 0,
    },
)
dm.setup()
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

for batch in train_loader:
    print(batch["image"][tio.DATA].shape, batch["mask"][tio.DATA].shape)
    break

for batch in val_loader:
    print(batch["image"][tio.DATA].shape, batch["mask"][tio.DATA].shape)
    break
