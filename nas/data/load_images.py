from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from fedot.core.data.data import InputData, Data
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from nas.utils.utils import project_root

root = project_root()
supported_images = {'.jpg', '.jpeg', '.png', '.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpe', '.jp2', '.tiff'}


@dataclass
class NasData(Data):
    """ Class for loading heavy datasets into FEDOT's InputData e.g. image datasets"""
    @staticmethod
    def data_from_folder(data_path: os.PathLike, task: Task) -> InputData:
        data_path = pathlib.Path(data_path) if not isinstance(data_path, pathlib.Path) else data_path
        data_type = DataTypesEnum.image
        features = []
        target = []
        for item in data_path.rglob('*.*'):
            if item.suffix in supported_images:
                features.append(str(item))
                target.append(item.parent.name)
        target = LabelEncoder().fit_transform(target)
        features = np.reshape(features, (-1, 1))
        idx = np.arange(0, len(features))
        return InputData(idx=idx, features=features, target=target, task=task, data_type=data_type)

    @staticmethod
    def data_from_csv(data_path: os.PathLike, features_col: str, target_col: str, task: Task) -> InputData:
        dataframe = pd.read_csv(data_path)
        data_type = DataTypesEnum.image
        features = list(dataframe[features_col])
        target = dataframe[target_col]
        target = LabelEncoder().fit_transform(target)
        features = np.expand_dims(features, -1)
        idx = np.arange(0, len(features))
        return InputData(idx=idx, features=features, target=target, task=task, data_type=data_type)


class ImageLoader:
    """Class for loading image dataset from InputData format. Implements loading by batches"""

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
