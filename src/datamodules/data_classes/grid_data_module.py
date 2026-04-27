import lightning as L
from torch.utils.data import DataLoader


class GridDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_cls,
        patch_size=(256, 256, 1),
        overlap=(0, 0, 0),
        train_dataset_kwargs=None,
        val_dataset_kwargs=None,
        loader_kwargs=None,
    ):
        super().__init__()
        self.dataset_cls = dataset_cls
        self.train_dataset_kwargs = train_dataset_kwargs
        self.val_dataset_kwargs = val_dataset_kwargs
        self.loader_kwargs = loader_kwargs
        self.patch_size = patch_size
        self.overlap = overlap

    def setup(self, stage=None):
        self.train_dataset = self.dataset_cls(**self.train_dataset_kwargs)
        self.val_dataset = self.dataset_cls(**self.val_dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loader_kwargs)
