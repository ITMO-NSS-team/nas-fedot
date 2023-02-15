from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union, Callable, Tuple


class BasePreprocessor(ABC):
    def __init__(self, transformations: Union[List[Callable], Tuple[Callable]] = None):
        self._transformations = transformations

    @property
    def transformations(self):
        return self._transformations

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        raise NotImplementedError


class Preprocessor(BasePreprocessor):
    """ Class for dataset preprocessing. Take images and targets by batch from loader and apply preprocessing to them.
    Returns generator inherited from keras Sequence class"""

    def __init__(self, transformations: Union[List[Callable], Tuple[Callable]] = None):
        super().__init__(transformations)

    def transform_sample(self, sample):
        if self.transformations:
            for t in self.transformations:
                sample = t(sample)
        return sample

    def preprocess(self, features_batch, targets_batch):
        new_features_batch = [self.transform_sample(sample) for sample in features_batch]
        new_targets_batch = targets_batch
        return new_features_batch, new_targets_batch
