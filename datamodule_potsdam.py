import warnings

import torchio as tio
from rasterio.errors import NotGeoreferencedWarning

from data_module import GridDataModule, PotsdamVaihingen

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


potsdam_image_glob = "Potsdam_dataset/2_Ortho_RGB/*.tif"
potsdam_mask_glob = "Potsdam_dataset/5_Labels_all/*.tif"

dm = GridDataModule(
    dataset_cls=PotsdamVaihingen,
    dataset_kwargs={
        "image_glob": potsdam_image_glob,
        "mask_glob": potsdam_mask_glob,
        "to_rgb": True,
    },
    dl_kwargs={
        "batch_size": 16,
        "num_workers": 0,
    },
)
dm.setup()
train_loader = dm.train_dataloader()

count = 0
for batch in train_loader:
    print(batch["image"][tio.DATA].shape, batch["mask"][tio.DATA].shape)
