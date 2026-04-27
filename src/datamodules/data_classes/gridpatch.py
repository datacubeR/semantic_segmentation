import torch
import torchio as tio
from torch.utils.data import Dataset


class GridPatchDataset(Dataset):
    def __init__(self, subjects, patch_size=(256, 256, 1), patch_overlap=(10, 10, 0)):
        self.samplers = [
            tio.data.GridSampler(
                subject, patch_size=patch_size, patch_overlap=patch_overlap
            )
            for subject in subjects
        ]

        # Precompute cumulative lengths for indexing
        self.lengths = [len(s) for s in self.samplers]
        self.cumsum = torch.tensor(self.lengths).cumsum(0)

    def __len__(self):
        return int(self.cumsum[-1])

    def __getitem__(self, idx):
        # Find which sampler this index belongs to
        sampler_idx = torch.searchsorted(self.cumsum, idx, right=True).item()

        if sampler_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumsum[sampler_idx - 1].item()

        return self.samplers[sampler_idx][local_idx]
