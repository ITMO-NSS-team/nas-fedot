import functools
import math
import os
from typing import (List,
                    Optional,
                    Union)
import pathlib
from abc import abstractmethod, ABC
from dataclasses import dataclass
import pandas as pd

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.applications.vgg16
from sklearn import utils
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from fedot.core.data.data import InputData
from fedot.core.data.data import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.data.supplementary_data import SupplementaryData

from nas.utils.var import project_root
from nas.utils.utils import set_root
from nas.data.split_data import generator_kfold_split, SplitterGenerator

set_root(project_root)
supported_images = ['.jpg', '.jpeg', '.png', '.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpe', '.jp2', '.tiff']


@dataclass
class FEDOTDataset(ABC):
    """
    General class for loading path.
    """
    batch_size: Optional[int]
    features: Optional[str]
    targets: Optional[str]

    def __init__(self, path, transformations):
        self._path = path
        self.transformations = transformations

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @property
    def path(self):
        try:
            if not isinstance(self._path, pathlib.Path):
                return pathlib.Path(self._path)
            else:
                return self._path
        except TypeError:
            raise TypeError('Wrong path: ', self._path)

    def shuffle(self):
        self.features, self.targets = utils.shuffle(self.features, self.targets, random_state=42)


class ImageDataset(FEDOTDataset):
    def __init__(self, data_path: Union[List, pathlib.Path, str], batch_size: int = 32, transformations: List = None,
                 shuffle: bool = False):
        super(ImageDataset, self).__init__(data_path, transformations)
        self.features, self.targets = self._get_feature_target_pairs()
        self.batch_size = batch_size

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        batch_x = self.features[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.targets[idx * self.batch_size: (idx + 1) * self.batch_size]
        features_batch = [self._load_and_transform(img) for img in batch_x]

        return np.array(features_batch), batch_y

    def shuffle(self):
        return utils.shuffle(self.features, self.targets)

    def _get_feature_target_pairs(self):
        features = []
        targets = []
        for item in self.path.rglob('*.*'):
            if item.suffix in supported_images:
                features.append(str(item))
                targets.append(item.parent.name)
        targets = LabelEncoder().fit_transform(targets)
        return features, targets

    def _load_and_transform(self, file_name):
        img = cv2.imread(str(file_name))
        if self.transformations:
            for t in self.transformations:
                img = t(img)
        return img


class CSVDataset(FEDOTDataset):
    def __init__(self, images_path, metadata_path, batch_size: int = 32, transformations: List = None,
                 shuffle: bool = False):
        super().__init__(images_path, transformations)
        self.targets_path = pd.read_csv(metadata_path)
        self.features, self.targets = self._get_feature_target_pairs()
        self.batch_size = batch_size

    def __len__(self):
        return len(self.features)

    def _get_feature_target_pairs(self):
        features = list(self.targets_path['image_path'])
        targets = self.targets_path['species']
        targets = LabelEncoder().fit_transform(targets)
        return features, targets

    def __getitem__(self, idx):
        batch_x = self.features[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.targets[idx * self.batch_size: (idx + 1) * self.batch_size]
        features_batch = [self._load_and_transform(img) for img in batch_x]

        return np.array(features_batch), batch_y

    def shuffle(self):
        return utils.shuffle(self.features, self.targets)

    def _load_and_transform(self, file_name):
        file_name = self.path / file_name
        img = cv2.imread(str(file_name))
        if self.transformations:
            for t in self.transformations:
                img = t(img)
        return img


class DataLoader(Sequence):
    def __init__(self, data_generator: FEDOTDataset, shuffle: bool = False):
        self.data_generator = data_generator
        self._batch_id = 0

        self._shuffle = shuffle
        if self._shuffle:
            self.data_generator.shuffle()

    @property
    def num_classes(self):
        return len(np.unique(self.data_generator.targets))

    @property
    def batch_size(self):
        return self.data_generator.batch_size

    @property
    def batch_id(self):
        return self._batch_id

    # @batch_id.setter
    # def batch_id(self, val):
    #     self._batch_id = val

    @property
    def steps_per_epoch(self):
        return math.ceil(len(self.data_generator) / self.batch_size)

    def __getitem__(self, idx):
        features, targets = self.data_generator[idx]
        return tf.convert_to_tensor(features), tf.convert_to_tensor(targets)

    def __len__(self):
        return self.steps_per_epoch

    def on_epoch_end(self):
        if self._shuffle:
            self.data_generator.shuffle()


@dataclass
class DataLoaderInputData(InputData):
    data_generator: Optional[DataLoader] = None

    @property
    def num_classes(self) -> Optional[int]:
        return len(np.unique(self.data_generator.data_generator.targets))

    @staticmethod
    def input_data_from_generator(data_generator, task, data_type: DataTypesEnum = DataTypesEnum.image, **kwargs):
        supplementary_data = SupplementaryData()
        supplementary_data.column_types = kwargs
        idx = np.arange(0, len(data_generator.data_generator))
        return DataLoaderInputData(features=data_generator.data_generator.features, data_generator=data_generator,
                                   target=data_generator.data_generator.targets, data_type=data_type,
                                   supplementary_data=supplementary_data, idx=idx, task=task)

    def shuffle(self):
        shuffled_ind = np.random.permutation(len(self.features))
        idx, features, target = np.asarray(self.idx)[shuffled_ind], self.features[shuffled_ind], self.target[
            shuffled_ind]
        self.idx = idx
        self.features = features
        self.target = target


if __name__ == '__main__':
    task = TaskTypesEnum.classification
    dataset_path = '../datasets/butterfly_cls'
    resize = functools.partial(tensorflow.image.resize, size=(32, 32))
    dataset = ImageDataset(dataset_path, 16, transformations=[resize])
    data_loader = DataLoader(data_generator=dataset, shuffle=True)
    data = DataLoaderInputData.input_data_from_generator(data_generator=data_loader, task=Task(task),
                                                         data_type=DataTypesEnum.image,
                                                         supplementary_data={'_image_size': [32, 32, 3]})

    k_fold = StratifiedKFold(n_splits=10)
    k_fold.get_n_splits(data.idx)

    for train_idx, test_idx in k_fold.split(data.features, data.target):
        new_idx_train = data.idx[train_idx]
        new_idx_test = data.idx[test_idx]

    generator_splitter = generator_kfold_split(data, n_splits=10)

    # generator_splits_train = []
    # generator_splits_test = []
    # for train_idx, test_idx in generator_splitter:
    #     generator_splits_train.extend([dataset.data_generator[i] for i in train_idx])
    #     generator_splits_test.extend([dataset.data_generator[i] for i in test_idx])

    splitter = SplitterGenerator('holdout', shuffle=True, random_state=42)

    for train_data, test_data in splitter.split(data):
        a = train_data
        b = test_data

    # model = tensorflow.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',
    #                                                   input_shape=(32, 32, 3))
    #
    # train_generator, test_data = generator_train_test_split(dataset, shuffle_flag=True)
    #
    # for layer in model.layers:
    #     layer.trainable = True
    #
    # top_model = model.output
    # top_model = tensorflow.keras.layers.Flatten()(top_model)
    # top_model = tensorflow.keras.layers.Dense(4096, activation='relu')(top_model)
    # top_model = tensorflow.keras.layers.Dense(1072, activation='relu')(top_model)
    # top_model = tensorflow.keras.layers.Dropout(0.2)(top_model)
    # output_layer = tensorflow.keras.layers.Dense(12, activation='softmax')(top_model)
    #
    # model = tensorflow.keras.models.Model(inputs=model.input, outputs=output_layer)
    # model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy',
    #               objective=['accuracy'])
    #
    # model.fit(data_loader, epochs=5)
    #
    print('Done!')
