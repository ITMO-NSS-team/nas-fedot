import copy
import json
import os
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Union, Optional
from functools import partial
from abc import abstractmethod

import cv2
import numpy as np
import pandas as pd

from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from nas.utils.utils import project_root, is_image
from sklearn.preprocessing import LabelEncoder

root = project_root()


class NASDataLoader(InputData):
    @abstractmethod
    def __init__(self, task, transformations, image_size, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _apply_transforms(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _load_and_transform(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load(self, **kwargs):
        raise NotImplementedError


@dataclass
class ImageDataLoader(InputData):

    # TODO implement regression task support
    # TODO add read from pickle option
    @staticmethod
    def _apply_transforms(image, transformations: List):
        for t in transformations:
            image = t(image)
        return image

    @staticmethod
    def _load_and_transform(image_path: Path, transformations: List):
        image = cv2.imread(str(image_path))
        if transformations:
            image = ImageDataLoader._apply_transforms(image, transformations)
        return image

    @staticmethod
    def images_from_directory(task: Task = Task(TaskTypesEnum.classification), transformations: Optional[List] = None,
                              dir_path: str = None, image_size: Union[int, float] = None, color_mode: str = 'rgb',
                              samples_limit: int = None) -> InputData:
        """
        Read images from directory. The following dataset format is required:
        dataset-directory
            |_class-1-name
                |_images
            ....
            |_class-n-name
                |_images
        :param task: type of task to be solved
        :param transformations: list of transformations applied to each image
        :param dir_path: path to dataset
        :param color_mode: image color mode
        :param image_size: image size. if not specified, the first image's size will be picked as image size
        :param samples_limit: limit for samples per class
        :return: dataset as InputData object
        """
        images_array = []
        labels_array = []
        transformations = transformations if transformations else []
        if image_size:
            transformations.append(partial(cv2.resize, dsize=(image_size, image_size)))
        for dir_path, folders, files in os.walk(dir_path, topdown=True):
            dir_path = Path(dir_path)
            if folders:
                labels = copy.deepcopy(folders)
                continue
            label = dir_path.name
            cnt = 0
            for image in files:
                if is_image(image):
                    image_path = dir_path.parent / label / image
                    image = ImageDataLoader._load_and_transform(image_path, transformations)
                    images_array.append(image)
                    labels_array.append(label)
                    if samples_limit:
                        cnt += 1
                        if cnt >= samples_limit:
                            break
        labels_array = LabelEncoder().fit_transform(labels_array)
        images_array = np.array(images_array)
        data = InputData.from_image(images=images_array, labels=labels_array, task=task)
        data.supplementary_data = {'labels': labels,
                                   'image_size': [image_size, image_size, 3]}
        return data

    @staticmethod
    def images_from_csv(task: Task = Task(TaskTypesEnum.classification), img_path: str = None, labels_path: str = None,
                        img_id: Optional[str] = 'id', target: Optional[str] = 'target', labels: Optional[List] = None,
                        transformations: List = None, image_size: Union[int, float] = None,
                        samples_limit: int = None) -> InputData:
        """
        Load images from dataset.
        Images should be in one folder and class names with corresponding image have to be in csv file with following
        format:
            | image_name | class_name |
        """
        images_array = []
        labels_array = []
        image_path = Path(img_path)
        df = pd.read_csv(labels_path).reset_index()
        df = df.filter([img_id, target])
        transformations = [] if not transformations else transformations
        if image_size:
            transformations.append(partial(cv2.resize, dsize=(image_size, image_size)))

        for idx in df.index:
            image = df[img_id][idx]
            label = df[target][idx]
            image = ImageDataLoader._load_and_transform(image_path / image, transformations)
            images_array.append(image)
            labels_array.append(label)
        labels = copy.deepcopy(labels_array) if not labels else labels
        labels_array = LabelEncoder().fit_transform(labels_array)
        images_array = np.array(images_array)
        data = InputData.from_image(images=images_array, labels=labels_array, task=task)
        data.supplementary_data = {'labels': labels,
                                   'image_size': [image_size, image_size, 3]}
        return data

    @staticmethod
    def images_from_json(task: Task = Task(TaskTypesEnum.classification), img_path: str = None, labels_path: str = None,
                         labels: Optional[List] = None, transformations: List = None,
                         image_size: Union[int, float] = None, samples_limit: int = None) -> InputData:
        """
        Load images from dataset.
        Images should be in one folder and class names of corresponding images required
        in following format of json file:
            {image_name: 'class_name'}
        """
        images_array = []
        labels_array = []
        images_path = img_path if isinstance(img_path, Path) else Path(img_path)
        transformations = [] if not transformations else transformations
        if image_size:
            transformations.append(partial(cv2.resize, dsize=(image_size, image_size)))
        with open(labels_path, 'r') as json_data:
            json_data = json.load(json_data)
        for dir_root, folders, files in os.walk(images_path):
            for image in files:
                if image.endswith('.png'):
                    label = json_data[image[:-4]]
                    image = ImageDataLoader._load_and_transform(images_path / image, transformations)
                    images_array.append(image)
                    labels_array.append(label)
        labels = copy.deepcopy(labels_array) if not labels else labels
        images_array = np.array(images_array)
        labels_array = LabelEncoder().fit_transform(labels_array)
        data = InputData.from_image(images=images_array, labels=labels_array, task=task)
        data.supplementary_data = {'labels': labels,
                                   'image_size': [image_size, image_size, 3]}
        return data


# TODO
def convert_data_to_pickle(dataset: InputData, dataset_name: str):
    """function or class for convert InputData to pickle format fot further save"""
    x_arr = dataset.features
    y_arr = dataset.target
    pickle_dataset = []
    pickle_dataset.extend([x_arr, y_arr])
    save_path = f'./datasets/{dataset_name}_pickled.pickle'
    with open(save_path, 'wb') as pickle_data:
        pickle.dump(pickle_dataset, pickle_data)


if __name__ == '__main__':
    print('Converting dataset to pickle...')
