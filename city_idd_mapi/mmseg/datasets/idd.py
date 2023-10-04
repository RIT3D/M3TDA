# ---------------------------------------------------------------
# Modified from https://github.com/lhoyer/MIC/blob/master/seg/mmseg/datasets/acdc.py
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class IDDDataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(IDDDataset, self).__init__(
            img_suffix='_leftImg8bit.png',
            seg_map_suffix='_gtFine_labelcsTrainIds.png',
            **kwargs)
        self.valid_mask_size = [1080, 1920]
