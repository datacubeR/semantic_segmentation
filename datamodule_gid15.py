import warnings

import torchio as tio
from rasterio.errors import NotGeoreferencedWarning

from data_module import GID15, GridDataModule

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


gid15_image_glob = "GID15/gid-15/GID/img_dir/train/*.tif"
gid15_mask_glob = "GID15/gid-15/GID/ann_dir/train/*.png"

dm = GridDataModule(
    dataset_cls=GID15,
    dataset_kwargs={
        "image_glob": gid15_image_glob,
        "mask_glob": gid15_mask_glob,
    },
    dl_kwargs={
        "batch_size": 16,
        "num_workers": 0,
    },
)
dm.setup()
train_loader = dm.train_dataloader()

for batch in train_loader:
    print(batch["image"][tio.DATA].shape, batch["mask"][tio.DATA].shape)
