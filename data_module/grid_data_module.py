import lightning as L

from .gridloader import GridLoader


class GridDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_cls,
        patch_size=(256, 256, 1),
        overlap=(0, 0, 0),
        dataset_kwargs=None,
        dl_kwargs=None,
    ):
        super().__init__()
        self.dataset_cls = dataset_cls
        self.dataset_kwargs = dataset_kwargs
        self.dl_kwargs = dl_kwargs
        self.patch_size = patch_size
        self.overlap = overlap

    def setup(self, stage=None):
        self.dataset = self.dataset_cls(**self.dataset_kwargs)

    def train_dataloader(self):
        return GridLoader(
            self.dataset,
            patch_size=self.patch_size,
            overlap=self.overlap,
            dl_kwargs=self.dl_kwargs,
        )
