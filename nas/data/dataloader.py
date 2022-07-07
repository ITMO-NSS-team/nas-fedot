import copy
from typing import (List,
                    Optional,
                    Union)
import pathlib
import random
from abc import abstractmethod

import cv2
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split, KFold

from fedot.core.data.data import InputData
from fedot.core.data.data import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.data.supplementary_data import SupplementaryData

from nas.utils.var import project_root
from nas.utils.utils import set_root

set_root(project_root)
supported_images = ['.jpg', '.jpeg', '.png', '.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpe', '.jp2', '.tiff']


def split_data(features: List, ratio, shuffle):
    target = get_label_names(features)
    return train_test_split(features, target, test_size=1 - ratio, shuffle=shuffle, random_state=42)[:2]


def get_label_names(features):
    return [f.parent.name for f in features]


class FEDOTDataset:
    """
    General class for loading path.
    """

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class ImageDataset(FEDOTDataset):
    def __init__(self, data_path: Union[List, pathlib.Path, str], transformations: List = None, shuffle: bool = True):
        self._path = data_path
        self._samples = self._samples_from_path()
        self.transformations = transformations
        self.shuffle = shuffle

    @property
    def path(self):
        try:
            if not isinstance(self._path, pathlib.Path):
                return pathlib.Path(self._path)
        except TypeError:
            raise TypeError('Wrong path: ', self._path)

    def __len__(self):
        return len(self.samples[0])

    def __getitem__(self, idx):
        img_id = self.samples[0][idx]
        target = self._samples[1][idx]
        path = pathlib.Path(self.path, target, img_id)
        image = cv2.imread(str(path))
        if self.transformations:
            for t in self.transformations:
                image = t(image)
        return image, target

    def _samples_from_path(self):
        images = []
        targets = []
        for sample in self.path.rglob('*.*'):
            if sample.is_file() and sample.suffix in supported_images:
                images.append(sample.name)
                targets.append(sample.parent.name)
        return images, targets

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, values):
        self._samples = values

    @staticmethod
    def shuffle(dataset):
        return random.sample(dataset, len(dataset))


class DataLoader:
    def __init__(self, dataset: FEDOTDataset, batch_size=32, mode: str = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self._mode = mode
        self.batch_id = 0

    def __getitem__(self, idx):
        images_array = []
        labels_array = []
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            img, label = self.dataset[idx]
            images_array.append(img)
            labels_array.append(label)
        batch = [images_array, labels_array]
        self.batch_id += 1
        return batch

    def __len__(self):
        if self.batch_size == 1:
            return len(self.dataset)
        elif self.batch_size > 1:
            return len(self.dataset) // self.batch_size

    @property
    def steps_per_epoch(self):
        return len(self)


class GeneratorInputData(InputData):
    data_loader: Optional[DataLoader]

    @staticmethod
    def get_batch_from_generator(data_loader: DataLoader, task, data_type: DataTypesEnum = DataTypesEnum.image,
                                 **kwargs):
        supplementary_data = SupplementaryData(kwargs)
        idx = np.arange(0, len(data_loader))
        batch_id = data_loader.batch_id
        return InputData(idx=idx,
                         task=task,
                         data_type=data_type,
                         features=data_loader[batch_id][0],
                         target=data_loader[batch_id][1],
                         supplementary_data=supplementary_data)

    def shuffle(self):
        shuffled_ind = np.random.permutation(len(self.features))
        idx, features, target = np.asarray(self.idx)[shuffled_ind], self.features[shuffled_ind], self.target[
            shuffled_ind]
        self.idx = idx
        self.features = features
        self.target = target


def generator_train_test_split(generator):
    features, targets = generator.dataset.samples
    x_train, x_val, y_train, y_val = train_test_split(features, targets, test_size=.33, random_state=42, shuffle=True)
    train_gen = copy.deepcopy(generator)
    train_gen.dataset.samples = x_train, y_train
    val_gen = copy.deepcopy(generator)
    val_gen.dataset.samples = x_val, y_val
    return train_gen, val_gen


if __name__ == '__main__':
    task = TaskTypesEnum.classification
    dataset_path = '../datasets/CXR8'
    dataset = ImageDataset(dataset_path, None, True)
    data_loader = DataLoader(dataset, 32, 'train')
    data = GeneratorInputData.get_batch_from_generator(data_loader, task, data_type=DataTypesEnum.image,
                                                       original_labels_dict={})
    for i in tqdm.tqdm(data_loader):
        np.transpose(i[0])
        np.transpose(i[1])

    train_gen, val_gen = generator_train_test_split(data_loader)
    print('Done!')
