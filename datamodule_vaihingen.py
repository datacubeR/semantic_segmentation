import warnings

import torchio as tio
from rasterio.errors import NotGeoreferencedWarning

from data_module import GridDataModule, PotsdamVaihingen

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


vaihingen_image_glob = "Vaihingen_dataset/top/*.tif"
vaihingen_mask_glob = (
    "Vaihingen_dataset/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/*.tif"
)

dm = GridDataModule(
    dataset_cls=PotsdamVaihingen,
    dataset_kwargs={
        "image_glob": vaihingen_image_glob,
        "mask_glob": vaihingen_mask_glob,
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
