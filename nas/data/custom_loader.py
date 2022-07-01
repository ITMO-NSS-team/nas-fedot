import copy
import os
from pathlib import Path
from functools import partial

import numpy as np
from sklearn.preprocessing import LabelEncoder

import cv2

from fedot.core.data.data import InputData
from nas.data.load_images import NASDataLoader


class CustomLoader(NASDataLoader):
    def __init__(self, task, transformations, image_size):
        self.task = task
        self.transformations = transformations if transformations else []
        self.image_size = image_size
        if image_size:
            self.transformations.append(partial(cv2.resize, dsize=(image_size, image_size)))

    def _apply_transforms(self, image):
        for t in self.transformations:
            image = t(image)
        return image

    def _load_and_transform(self, path_to_image):
        image = cv2.imread(str(path_to_image))
        if self.transformations:
            image = self._apply_transforms(image)
        return image

    def load(self, path, labels_file=None, limit=None):
        path = Path(path) if not isinstance(path, Path) else path
        images_array = []
        labels_array = []
        if labels_file:
            labels_file = CustomLoader._train_test_lists(labels_file)
        for dir_name, folders, files in os.walk(path, True):
            if folders:
                labels = copy.deepcopy(folders)
                continue
            label = Path(dir_name).name
            cnt = 0
            for image_name in files:
                path_to_image = Path(path, label, image_name)
                if image_name in labels_file:
                    image = self._load_and_transform(path_to_image)
                    images_array.append(image)
                    labels_array.append(label)
                    if limit:
                        if cnt >= limit:
                            break

        labels_array = LabelEncoder().fit_transform(labels_array)
        images_array = np.array(images_array)
        input_data = InputData.from_image(images=images_array, labels=labels_array, task=self.task)
        input_data.supplementary_data = {'labels': labels,
                                         'image_size': [self.image_size, self.image_size, 3]}
        return input_data

    @staticmethod
    def _train_test_lists(path_to_file):
        with open(Path(path_to_file), 'r') as file:
            train_test_file = file.read()
        return train_test_file
