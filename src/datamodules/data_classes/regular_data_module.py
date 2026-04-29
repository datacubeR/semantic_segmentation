import lightning as L
from torch.utils.data import DataLoader


class RegularDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_cls,
        train_dataset_kwargs=None,
        val_dataset_kwargs=None,
        loader_kwargs=None,
    ):
        super().__init__()
        self.dataset_cls = dataset_cls
        self.train_dataset_kwargs = train_dataset_kwargs
        self.val_dataset_kwargs = val_dataset_kwargs
        self.loader_kwargs = loader_kwargs
        self.batch_size = loader_kwargs["batch_size"]

    def setup(self, stage=None):
        self.train_dataset = self.dataset_cls(**self.train_dataset_kwargs)
        self.val_dataset = self.dataset_cls(**self.val_dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, shuffle=True, drop_last=True, **self.loader_kwargs
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.loader_kwargs)
