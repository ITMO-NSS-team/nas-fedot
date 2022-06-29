import copy
from tqdm import tqdm
import json
import os
import pickle
from pathlib import Path
from dataclasses import dataclass
from os.path import isfile, join
from typing import List, Union, Optional
from functools import partial

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fedot.core.data.data import Data, InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from nas.utils.utils import project_root
from sklearn.preprocessing import LabelEncoder

root = project_root()


@dataclass
class ImageDataLoader(InputData):

    # TODO implement regression task support
    # TODO add read from pickle option
    @staticmethod
    def _apply_transforms(image, transformations: List, **kwargs):
        for t in transformations:
            image = t(image)
        return image

    @staticmethod
    def load_and_transform(image_path: Path, transformations: List, **kwargs):
        image = cv2.imread(str(image_path))
        if transformations:
            image = ImageDataLoader._apply_transforms(image, transformations)
        return image

    @staticmethod
    def from_directory(task: Task = Task(TaskTypesEnum.classification), transformations: List = [],
                       dir_path: str = None, color_mode: str = 'rgb', image_size: Union[int, float] = None,
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
        if image_size:
            transformations.append(partial(cv2.resize, dsize=(image_size, image_size)))
        print('\nReading data from directory...\n')
        for dir_path, folders, files in os.walk(dir_path, topdown=True):
            dir_path = Path(dir_path)
            if folders:
                str_labels = copy.deepcopy(folders)
                continue
            label = dir_path.name
            cnt = 0
            for image in files:
                image_path = dir_path.parent / label / image
                image = ImageDataLoader.load_and_transform(image_path, transformations)
                images_array.append(image)
                labels_array.append(label)
                if samples_limit:
                    cnt += 1
                    if cnt >= samples_limit:
                        break
        labels_array = LabelEncoder().fit_transform(labels_array)
        images_array = np.array(images_array)
        data = InputData.from_image(images=images_array, labels=labels_array, task=task)
        data.supplementary_data = str_labels
        return data

    @staticmethod
    def image_from_csv(task: Task = Task(TaskTypesEnum.classification), img_path: str = None, labels_path: str = None,
                       img_id: Optional[str] = 'id', target: Optional[str] = 'target', labels: Optional[List] = None,
                       transformations: List = [], image_size: Union[int, float] = None,
                       samples_limit: int = None) -> InputData:
        """
        Load images from dataset.
        Images needed in one folder and class names with corresponding image have to be in csv file with following
        format:
            | image_name | class_name |
        """
        images_array = []
        labels_array = []
        image_path = Path(img_path)
        df = pd.read_csv(labels_path).reset_index()
        df = df.filter([img_id, target])
        if image_size:
            transformations.append(partial(cv2.resize, dsize=(image_size, image_size)))

        for idx in df.index:
            image = df[img_id][idx]
            label = df[target][idx]
            image = ImageDataLoader.load_and_transform(str(image_path / image), transformations)
            images_array.append(image)
            labels_array.append(label)
        labels = copy.deepcopy(labels_array) if not labels else labels
        labels_array = LabelEncoder().fit_transform(labels_array)
        images_array = np.array(images_array)
        data = InputData.from_image(images=images_array, labels=labels_array, task=task)
        data.supplementary_data = labels
        return data

    @staticmethod
    def images_from_pickle(task: TaskTypesEnum.classification, dir_path: str = None,
                           image_size: Union[int, float] = None, samples_limit: int = None) -> InputData:
        raise NotImplementedError

    @staticmethod
    def images_from_json(task: TaskTypesEnum.classification, dir_path: str = None,
                         image_size: Union[int, float] = None, samples_limit: int = None) -> InputData:
        raise NotImplementedError


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
