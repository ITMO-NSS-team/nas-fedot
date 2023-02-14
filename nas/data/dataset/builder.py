from __future__ import annotations

from copy import deepcopy
from typing import Callable, Optional, Type

from nas.data import Preprocessor, ImageLoader


class BaseNasDatasetBuilder:
    def __init__(self, dataset_cls: Callable, batch_size: int = 32, shuffle: bool = True):
        """

        :param dataset_cls: Dataset builder. Can be dataset builder function or
        dataset class inherited from either Keras Sequence or Pytorch Dataset classes.
        :param batch_size: Hyperparameter that determines the number of samples to work through.
        :param shuffle: Flag that determines to shuffle dataset or not.
        """
        self._data_preprocessor: Optional[Preprocessor] = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._input_data_converter: Type[ImageLoader] = ImageLoader
        self._dataset_cls: Callable = dataset_cls

    def set_input_data_converter(self, loader: ImageLoader):
        self._input_data_converter = loader
        return self

    def set_data_preprocessor(self, preprocessor: Preprocessor):
        """
        Initializes preprocessor with given operations
        :param preprocessor: class that contains list of transformations to apply.
        :return:
        """
        self._data_preprocessor = preprocessor
        return self

    def build(self, data, **kwargs):
        """
        Method for creating dataset object with given parameters for further model training/evaluating.
        :param data:
        :param kwargs:
        :return:
        """
        train_mode = {'train': True, 'val': False, 'test': False}
        mode = kwargs.get('mode')

        shuffle = train_mode.get(mode, self.shuffle)
        batch_size = kwargs.pop('batch_size', self.batch_size)
        data_preprocessor = None
        if self._data_preprocessor:
            data_preprocessor = deepcopy(self._data_preprocessor)
            data_preprocessor.mode = 'evaluation' if mode == 'test' else 'default'
        data_loader = self._input_data_converter(data)

        dataset = self._dataset_cls(batch_size=batch_size, shuffle=shuffle,
                                    preprocessor=data_preprocessor, loader=data_loader)
        return dataset
