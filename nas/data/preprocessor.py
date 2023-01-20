from __future__ import annotations
from functools import partial
from typing import List, Optional, Union, Callable, Tuple

import cv2


class Preprocessor:
    """ Class for dataset preprocessing. Take images and targets by batch from loader and apply preprocessing to them.
    Returns generator inherited from keras Sequence class"""

    def __init__(self, image_size: Union[Tuple[int, int], List[int, int]],
                 transformations: Union[List[Callable], Tuple[Callable]] = None):
        self._image_size: Tuple = image_size
        self._transformations = [partial(cv2.resize, dsize=self.image_size)]
        self._additional_transforms = [] if not transformations else transformations
        self._mode: str = 'default'

    @property
    def image_size(self) -> Tuple:
        if type(self._image_size) == tuple:
            return self._image_size
        else:
            return tuple(self._image_size)

    @image_size.setter
    def image_size(self, val: Union[Tuple[int, int], List[int, int]]):
        if type(val) == tuple:
            self._image_size = val
        else:
            self._image_size = tuple(val)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, val: str):
        self._mode = val

    @property
    def all_transformations(self) -> List[Union[partial, Callable]]:
        return self._transformations + self._additional_transforms

    def set_image_size(self, image_size: Tuple[float, float]):
        self._image_size = image_size
        return self

    def set_features_transformations(self, transformations: Optional[List[Callable]] = None):
        self._additional_transforms = transformations
        return self

    def transform_sample(self, sample):
        transformations = self.all_transformations
        if self._mode == 'evaluation':
            transformations = self._transformations
        for t in transformations:
            sample = t(sample)
        return sample

    def preprocess(self, features_batch, targets_batch):
        new_features_batch = [self.transform_sample(sample) for sample in features_batch]
        new_targets_batch = targets_batch
        return new_features_batch, new_targets_batch
