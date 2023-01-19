from __future__ import annotations

import math
from typing import Tuple, TYPE_CHECKING

import sklearn
import tensorflow as tf

if TYPE_CHECKING:
    from nas.data import ImageLoader, Preprocessor


class KerasDataset(tf.keras.utils.Sequence):
    def __init__(self, loader: ImageLoader, preprocessor: Preprocessor,
                 batch_size: int = 8, shuffle: bool = True):
        self.batch_size = batch_size
        self._loader = loader
        self._preprocessor = preprocessor
        self._shuffle = shuffle

    def __len__(self):
        return math.floor(len(self._loader) / self.batch_size)

    def __getitem__(self, batch_id) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_x = [self._loader.get_feature(i) for i in
                   range(batch_id * self.batch_size, (batch_id + 1) * self.batch_size)]
        batch_y = [self._loader.get_target(i) for i in
                   range(batch_id * self.batch_size, (batch_id + 1) * self.batch_size)]
        batch_x, batch_y = self._preprocessor.preprocess(batch_x, batch_y)
        return tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_y)

    def on_epoch_end(self):
        if self._shuffle:
            self.shuffle_dataset()

    def shuffle_dataset(self):
        return sklearn.utils.shuffle(self._loader.features, self._loader.target)

    # Hotfixes
    @property
    def num_classes(self):
        return self._loader.num_classes

    @property
    def target(self):
        return self._loader.target

    @property
    def features(self):
        return self._loader.features

    # TODO
    @property
    def idx(self):
        return self._loader.idx

    @property
    def task(self):
        return self._loader.task

    @property
    def data_type(self):
        return self._loader.data_type
