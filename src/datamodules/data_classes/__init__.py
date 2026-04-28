from .deadtrees import DeadTrees
from .geonrw import GeoNRW
from .gid15 import GID15
from .gridloader import GridLoader
from .gridpatch import GridPatchDataset
from .loveda import LoveDAds
from .potsdamvaihingen import PotsdamVaihingen
from .regular_data_module import RegularDataModule

__all__ = [
    "PotsdamVaihingen",
    "DeadTrees",
    "LoveDAds",
    "GID15",
    "GeoNRW",
    "GridPatchDataset",
    "GridLoader",
    "RegularDataModule",
]
