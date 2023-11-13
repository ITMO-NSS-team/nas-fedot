from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np
from fedot.core.data.data import InputData
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class BaseDataLoader(ABC):
    """Base class that performs data loading from InputData."""

    def __init__(self, dataset: InputData, *args, **kwargs):
        self.dataset = dataset

    @property
    def features(self):
        return self.dataset.features

    @property
    def target(self):
        new_targets = np.reshape(self.dataset.target, (-1, 1))

        if self.dataset.num_classes > 2:
            encoder = OneHotEncoder(handle_unknown='error', dtype=int, sparse_output=False)
        else:
            encoder = LabelEncoder()
        new_targets = encoder.fit_transform(new_targets)
        return new_targets

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class ImageLoader(BaseDataLoader):
    """Class for loading image dataset from InputData format. Implements loading by batches"""

    def __init__(self, dataset: InputData, image_size: Tuple[int, int]):
        super().__init__(dataset)
        self._image_size = image_size

    @property
    def features(self):
        return self.dataset.features.flatten()

    def get_feature(self, idx):
        feature = cv2.imread(self.features[idx])
        feature = cv2.resize(feature, dsize=self._image_size)
        return np.transpose(feature, (2, 0, 1))

    def get_target(self, idx):
        return self.target[idx]

    def __getitem__(self, item):
        return self.get_feature(item), self.get_target(item)

    def __len__(self):
        return len(self.features)
