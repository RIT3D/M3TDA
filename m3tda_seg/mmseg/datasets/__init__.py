# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional datasets

from .acdc import ACDCDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .cityscapes import CityscapesDataset
from .custom import CustomDataset
from .dark_zurich import DarkZurichDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .gta import GTADataset
from .synthia import SynthiaDataset
from .idd import IDDDataset
from .mtda_dataset import MTDADataset

__all__ = [
    'CustomDataset',
    'build_dataloader',
    'ConcatDataset',
    'RepeatDataset',
    'DATASETS',
    'build_dataset',
    'PIPELINES',
    'CityscapesDataset',
    'GTADataset',
    'SynthiaDataset',
    'MTDADataset',
    'ACDCDataset',
    'DarkZurichDataset',
    'IDDDataset',
]
