from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Optional, Type

from fedot.core.data.data import InputData

from nas.data import Preprocessor, ImageLoader
from nas.data.loader import BaseDataLoader


class BaseNNDataset(ABC):
    def __init__(self, dataset_cls: Callable, batch_size: int, loader: Type[BaseDataLoader], shuffle: bool):
        self.dataset_cls = dataset_cls
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.loader = loader

    @abstractmethod
    def build(self, data: InputData, **kwargs):
        """This method builds tensorflow or pytorch dataset"""
        raise NotImplementedError


class ImageDatasetBuilder(BaseNNDataset):
    def __init__(self, dataset_cls: Callable, batch_size: int = 32, loader: Type[BaseDataLoader] = ImageLoader,
                 shuffle: bool = True):
        super().__init__(dataset_cls, batch_size, loader, shuffle)
        self._data_preprocessor: Optional[Preprocessor] = None

    def set_data_preprocessor(self, preprocessor: Preprocessor):
        self._data_preprocessor = preprocessor
        return self

    def build(self, data, **kwargs):
        """Method for creating dataset object with given parameters for further model training/evaluating."""
        train_mode = {'train': True, 'val': False, 'test': False}
        mode = kwargs.get('mode')

        shuffle = train_mode.get(mode, self.shuffle)
        batch_size = kwargs.pop('batch_size', self.batch_size)
        data_preprocessor = None
        if self._data_preprocessor:
            data_preprocessor = deepcopy(self._data_preprocessor)
            data_preprocessor.mode = 'evaluation' if mode == 'test' else 'default'
        loader = self.loader(data)

        dataset = self.dataset_cls(batch_size=batch_size, shuffle=shuffle,
                                   preprocessor=data_preprocessor, loader=loader)
        return dataset
