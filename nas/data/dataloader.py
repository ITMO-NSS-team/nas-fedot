from typing import (List,
                    Optional,
                    Union)
import pathlib
import random
from abc import abstractmethod
from dataclasses import dataclass

import cv2
import tqdm
from sklearn.model_selection import train_test_split

from fedot.core.data.data import InputData

from nas.utils.var import project_root
from nas.utils.utils import set_root

set_root(project_root)


def split_data(features: List, ratio, shuffle):
    target = get_label_names(features)
    return train_test_split(features, target, test_size=1 - ratio, shuffle=shuffle, random_state=42)[:2]


def get_data_list(path, shuffle=False):
    if isinstance(path, str):
        path = pathlib.Path(path)
    lst = ('*.png', '*.jpg ', '*.jpeg', '*.jfif', '*.pjpeg', '*.pjp', '*.svg')
    images = []
    for ext in lst:
        images.extend(path.rglob(ext))
    if shuffle:
        images = ImageLoader.shuffle(images)
    return images


def get_label_names(features):
    return [f.parent.name for f in features]


class NASDataLoader:
    """
    General class for loading data.
    """

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class ImageLoader(NASDataLoader):
    def __init__(self, data_path: Union[List, pathlib.Path, str], transformations: List = None, shuffle: bool = True):
        self.data = data_path if isinstance(data_path, List) else get_data_list(data_path)
        self.transformations = transformations
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path = self.data[item]
        image = cv2.imread(str(path))
        label = path.parent.name
        return image, label

    @staticmethod
    def shuffle(dataset):
        return random.sample(dataset, len(dataset))

    def load(self, idx, *args):
        mode = args[0] if args else 'train'
        image, label = self[idx]
        if self.transformations and mode == 'train':
            for t in self.transformations:
                image = t(image)
        return image, label


class DataLoader:
    def __init__(self, task, data_loader, batch_size=1, mode: str = None, *args):
        self.task = task
        self.loader = data_loader(*args)
        self.batch_size = batch_size
        self._mode = mode

    def __getitem__(self, item):
        images_array = []
        labels_array = []
        for i in range(item * self.batch_size, (item + 1) * self.batch_size):
            img, label = self.loader.load(item, self._mode)
            images_array.append(img)
            labels_array.append(label)
        return images_array, labels_array

    def __len__(self):
        if self.batch_size == 1:
            return len(self.loader)
        elif self.batch_size > 1:
            return len(self.loader) // self.batch_size

    @property
    def steps_per_epoch(self):
        return len(self)


@dataclass
class NASInputData(InputData):
    @staticmethod
    def from_generator(generator):
        raise NotImplementedError


if __name__ == '__main__':
    dataset_path = '../datasets/CXR8'
    train_dataset = get_data_list(dataset_path)
    train_dataset, val_dataset = split_data(train_dataset, 0.6, True)
    val_dataset, test_dataset = split_data(val_dataset, 0.6, True)
    d = ImageLoader
    t_loader = DataLoader(None, d, 128, 'train', train_dataset, None, True)
    v_loader = DataLoader(None, d, 16, 'val', val_dataset, None, True)
    test_loader = DataLoader(None, d, 16, 'test', test_dataset, None, True)
    for i in tqdm.tqdm(range(len(t_loader))):
        t_sample = t_loader[i]
    print('Done!')
