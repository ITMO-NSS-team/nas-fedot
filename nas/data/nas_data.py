from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from fedot.core.data.data import Data, InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

SUPPORTED_TYPES = {'.jpg', '.jpeg', '.png', '.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpe', '.jp2', '.tiff'}


@dataclass
class InputDataNN(Data):
    """ Class for loading heavy datasets into FEDOT's InputData e.g. image datasets"""

    @staticmethod
    def data_from_folder(data_path: os.PathLike, task: Task, csv_labels: str = None) -> InputData:
        if csv_labels:
            labels = pd.read_csv(csv_labels)
        data_path = pathlib.Path(data_path) if not isinstance(data_path, pathlib.Path) else data_path
        data_type = DataTypesEnum.image
        features = []
        target = []
        for item in data_path.rglob('*.*'):
            if item.suffix in SUPPORTED_TYPES:
                features.append(str(item))
                if csv_labels:
                    target.extend(labels[labels['id'] == int(item.name[:-4])]['label'].values)
                else:
                    target.append(item.parent.name)
        # target = OneHotEncoder().fit_transform(target).toarray()
        # target = LabelEncoder().fit_transform(target)
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
