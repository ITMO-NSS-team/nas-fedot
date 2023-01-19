from __future__ import annotations

from typing import Callable, Optional, Type

from nas.data import Preprocessor, ImageLoader


class BaseNasDatasetBuilder:
    def __init__(self, dataset_cls: Callable, batch_size: int = 32, shuffle: bool = True):
        self._data_preprocessor: Optional[Preprocessor] = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._data_loader: Type[ImageLoader] = ImageLoader
        self._dataset_cls: Callable = dataset_cls

    def set_loader(self, loader: ImageLoader):
        self._data_loader = loader
        return self

    def set_data_preprocessor(self, preprocessor: Preprocessor):
        self._data_preprocessor = preprocessor
        return self

    def build(self, data, **kwargs):
        """Method for creating dataset object with given parameters for further model training/evaluating."""
        train_mode = {'train': True, 'val': False, 'test': False}
        mode = kwargs.get('mode')

        shuffle = train_mode.get(mode, self.shuffle)
        batch_size = kwargs.pop('batch_size', self.batch_size)
        if self._data_preprocessor:
            self._data_preprocessor.mode = 'evaluation' if mode == 'test' else 'default'
        data_loader = self._data_loader(data)

        dataset = self._dataset_cls(batch_size=batch_size, shuffle=shuffle,
                                    preprocessor=self._data_preprocessor, loader=data_loader)
        return dataset
