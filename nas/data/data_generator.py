import math
from functools import partial
from typing import List, Optional, Union, Callable, Tuple

import cv2
import numpy as np
import sklearn
import tensorflow as tf
from fedot.core.data.data import InputData
from sklearn.preprocessing import OneHotEncoder


class Loader:
    """ Class for loading image dataset from InputData format. Implement loading by batches"""

    def __init__(self, dataset: InputData):
        self.idx = dataset.idx
        self.task = dataset.task
        self.data_type = dataset.data_type
        self.features = dataset.features.flatten()
        self.target = self.transform_targets(dataset.target)
        self.num_classes = dataset.num_classes

    def get_feature(self, idx):
        return cv2.imread(self.features[idx])

    def get_target(self, idx):
        return self.target[idx]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.get_feature(item), self.get_target(item)

    @staticmethod
    def transform_targets(targets):
        new_targets = np.reshape(targets, (-1, 1))
        num_classes = len(np.unique(targets))

        if num_classes > 2:
            encoder = OneHotEncoder(handle_unknown='error', dtype=int, sparse=False)
            new_targets = encoder.fit_transform(new_targets)
        return new_targets


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, loader: Loader, preprocessor, batch_size: int = 8, shuffle: bool = True):
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
        return self._preprocessor.preprocess(batch_x, batch_y)

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


class Preprocessor:
    """ Class for dataset preprocessing. Take images and targets by batch from loader and apply preprocessing to them.
    Returns generator inherited from keras Sequence class"""

    def __init__(self, ):
        self._image_size = None
        self._transformations = []

    @property
    def transformations(self) -> List[Union[partial, Callable]]:
        return self._transformations

    @transformations.setter
    def transformations(self, value: Union[List[Callable], Callable]):
        if hasattr(value, '__iter__'):
            self._transformations.extend(value)
        else:
            self._transformations.append(value)

    def set_image_size(self, image_size: Tuple[float, float]):
        self._image_size = image_size
        self.transformations = partial(cv2.resize, dsize=image_size)
        return self

    def set_features_transformations(self, transformations: Optional[List[Callable]] = None):
        self.transformations = transformations
        return self

    def transform_sample(self, sample):
        for t in self.transformations:
            sample = t(sample)
        return sample

    def preprocess(self, features_batch, targets_batch):
        new_features_batch = tf.convert_to_tensor([self.transform_sample(sample) for sample in features_batch])
        new_targets_batch = tf.convert_to_tensor(targets_batch)
        return new_features_batch, new_targets_batch
