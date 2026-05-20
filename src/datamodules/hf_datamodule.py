import glob

import lightning as L
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader


class HFDataModule(L.LightningDataModule):
    def __init__(self, train_path, val_path, batch_size=32):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = self._load_shards_into_dataset(self.hparams.train_path)

        if stage in ["fit", "validate"]:
            self.val_dataset = self._load_shards_into_dataset(self.hparams.val_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=10,
            pin_memory=True,
        )

    def _load_shards_into_dataset(self, path):
        paths = [load_from_disk(p) for p in glob.glob(path)]

        dataset = concatenate_datasets(paths)
        dataset.set_format(type="torch")
        return dataset
