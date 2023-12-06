from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Type, Union, Tuple, List

from fedot.core.data.data import InputData

from nas.data.loader import BaseDataLoader, ImageLoader
from nas.data.preprocessor import Preprocessor


class BaseNNDatasetBuilder(ABC):
    def __init__(self, dataset_cls: Callable, batch_size: int = 32,
                 loader: Type[BaseDataLoader] = None, shuffle: bool = False):
        self.dataset_cls = dataset_cls
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.loader = loader

    @abstractmethod
    def build(self, data: InputData, **kwargs):
        """This method builds tensorflow or pytorch dataset"""
        raise NotImplementedError


class ImageDatasetBuilder(BaseNNDatasetBuilder):
    def __init__(self, dataset_cls: Callable, image_size: Union[Tuple[int, int], List[int, int]],
                 loader: Type[BaseDataLoader] = ImageLoader, shuffle: bool = True):
        super().__init__(dataset_cls, loader=loader, shuffle=shuffle)
        self._data_preprocessor: Optional[Preprocessor] = None
        self._image_size = image_size
        self._mean = None
        self._std = None

    @property
    def image_size(self) -> Tuple:
        if type(self._image_size) == tuple:
            return self._image_size
        else:
            return tuple(self._image_size)

    def set_data_preprocessor(self, preprocessor: Preprocessor):
        self._data_preprocessor = preprocessor
        return self

    def build(self, data: InputData, shuffle: bool = True, **kwargs):
        """Method for creating dataset object with given parameters for further model training/evaluating."""
        train_mode = {'train': True, 'val': False, 'test': False}
        mode = kwargs.get('mode')

        if mode == 'test':
            data_preprocessor = None
        else:
            data_preprocessor = self._data_preprocessor
        loader = self.loader(data, self.image_size)

        dataset = self.dataset_cls(preprocessor=data_preprocessor, loader=loader)
        return dataset
