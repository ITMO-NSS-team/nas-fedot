from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from fedot.core.data.data import Data, InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task

supported_images = {'.jpg', '.jpeg', '.png', '.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpe', '.jp2', '.tiff'}


@dataclass
class InputDataNN(Data):
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
