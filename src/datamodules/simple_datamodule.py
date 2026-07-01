import lightning as L
from torch.utils.data import DataLoader

from .data_classes import FullImageDataset


class SimpleDataModule(L.LightningDataModule):
    def __init__(self, dataset_name: str, batch_size=32, val_test_batch_size=32):
        super().__init__()
        self.save_hyperparameters()

        if dataset_name == "DeadTrees":
            extension = "tif"
        elif dataset_name == "LoveDA":
            extension = "png"

        self.train_image_glob = f"{dataset_name}/train/images/*.{extension}"
        self.train_mask_glob = f"{dataset_name}/train/masks/*.{extension}"
        self.val_image_glob = f"{dataset_name}/val/images/*.{extension}"
        self.val_mask_glob = f"{dataset_name}/val/masks/*.{extension}"
        self.test_image_glob = f"{dataset_name}/test/images/*.{extension}"
        self.test_mask_glob = f"{dataset_name}/test/masks/*.{extension}"

    def setup(self, stage=None):
        if stage == "fit":
            self.train_data = FullImageDataset(
                self.train_image_glob,
                self.train_mask_glob,
                reduce_mask=False,
                squeeze_mask=True,
            )

        if stage in ["fit", "validate"]:
            self.val_data = FullImageDataset(
                self.val_image_glob,
                self.val_mask_glob,
                reduce_mask=False,
                squeeze_mask=True,
            )

        if stage in ["test", "predict"]:
            self.test_data = FullImageDataset(
                self.test_image_glob,
                self.test_mask_glob,
                reduce_mask=False,
                squeeze_mask=True,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.val_test_batch_size,
            shuffle=False,
            num_workers=10,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.val_test_batch_size,
            shuffle=False,
            num_workers=10,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.val_test_batch_size,
            shuffle=False,
            num_workers=10,
            pin_memory=True,
        )
