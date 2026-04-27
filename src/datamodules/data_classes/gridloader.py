import torchio as tio
from torch.utils.data import DataLoader


class GridLoader:
    def __init__(self, subjects, patch_size, overlap, loader_kwargs):

        ## Fix cuando trabajo sin Batch Size
        if isinstance(subjects, tio.Subject):
            subjects = [subjects]
        self.subjects = subjects
        self.subjects = subjects
        self.patch_size = patch_size
        self.overlap = overlap
        self.loader_kwargs = loader_kwargs

    def __iter__(self):
        for subject in self.subjects:
            sampler = tio.GridSampler(
                subject,
                patch_size=self.patch_size,
                patch_overlap=self.overlap,
            )

            loader = DataLoader(
                sampler,
                **self.loader_kwargs,
            )

            for batch in loader:
                yield batch
