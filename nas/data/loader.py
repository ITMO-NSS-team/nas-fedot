from __future__ import annotations

import cv2
import numpy as np
from fedot.core.data.data import InputData
from sklearn.preprocessing import OneHotEncoder


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
