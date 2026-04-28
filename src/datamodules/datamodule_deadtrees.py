from .data_classes import DeadTrees, RegularDataModule

train_deadtrees_image_glob = "../DeadTrees/train/images/*.tif"
train_deadtrees_mask_glob = "../DeadTrees/train/masks/*.tif"
val_deadtrees_image_glob = "../DeadTrees/val/images/*.tif"
val_deadtrees_mask_glob = "../DeadTrees/val/masks/*.tif"

deadtrees_dm = RegularDataModule(
    dataset_cls=DeadTrees,
    train_dataset_kwargs={
        "image_glob": train_deadtrees_image_glob,
        "mask_glob": train_deadtrees_mask_glob,
        "white_threshold": 0.1,
    },
    val_dataset_kwargs={
        "image_glob": val_deadtrees_image_glob,
        "mask_glob": val_deadtrees_mask_glob,
        "white_threshold": 0.1,
    },
    loader_kwargs={
        "batch_size": 16,
        "num_workers": 10,
    },
)
