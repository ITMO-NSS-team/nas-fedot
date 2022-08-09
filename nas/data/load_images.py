from functools import partial
from typing import List, Union, Optional
from dataclasses import dataclass

import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from fedot.core.data.data import InputData, Data
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from nas.utils.utils import project_root

root = project_root()
supported_images = {'.jpg', '.jpeg', '.png', '.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpe', '.jp2', '.tiff'}


@dataclass
class NNData(Data):
    @staticmethod
    def data_from_folder(data_path, task):
        data_path = pathlib.Path(data_path) if not isinstance(data_path, pathlib.Path) else data_path
        data_type = DataTypesEnum.image
        features = []
        target = []
        for item in data_path.rglob('*.*'):
            if item.suffix in supported_images:
                features.append(str(item))
                target.append(item.parent.name)
        target = LabelEncoder().fit_transform(target)
        # features = features
        features = np.reshape(features, (-1, 1))
        idx = np.arange(0, len(features))
        return InputData(idx=idx, features=features, target=target, task=task, data_type=data_type)

    @staticmethod
    def data_from_csv(data_path, features_col, target_col, task):
        dataframe = pd.read_csv(data_path)
        data_type = DataTypesEnum.image
        features = list(dataframe[features_col])
        target = dataframe[target_col]
        target = LabelEncoder().fit_transform(target)
        features = np.expand_dims(features, -1)
        idx = np.arange(0, len(features))
        return InputData(idx=idx, features=features, target=target, task=task, data_type=data_type)
