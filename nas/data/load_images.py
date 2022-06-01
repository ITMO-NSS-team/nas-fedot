import json
import os
import pickle
from dataclasses import dataclass
from os.path import isfile, join
from typing import List, Union

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from fedot.core.data.data import Data, InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from nas.utils.utils import project_root
from sklearn.preprocessing import LabelEncoder
from fedot.core.repository.dataset_types import DataTypesEnum

root = project_root()


@dataclass
class DataLoader(InputData):

    # TODO implement regression task support
    @staticmethod
    def from_directory(task: Task = Task(TaskTypesEnum.classification), dir_path: str = None,
                       image_size: Union[int, float] = None, samples_limit: int = None) -> InputData:
        """
        Read images from directory. The following dataset format is required:
        dataset-directory
            |_class-1-name
                |_images
            ....
            |_class-n-name
                |_images
        :param task: type of task to be solved
        :param dir_path: path to dataset
        :param image_size: image size. if not specified, the first image's size will be picked as image size
        :param samples_limit: limit for samples per class
        :return: dataset as InputData object
        """
        images_array = []
        labels_array = []
        for label in os.listdir(dir_path):
            path = os.path.join(dir_path, label)
            cnt = 0
            for image_name in os.listdir(path):
                image = cv2.imread(os.path.join(path, image_name))
                if image_size is None:
                    image_size = image.shape[0]
                elif image.shape[0] != image_size:
                    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
                    images_array.append(image)
                    labels_array.append(label)
                    if samples_limit:
                        cnt += 1
                        if cnt >= samples_limit:
                            break
        is_digit = labels_array[0].isdigit()
        if not is_digit:
            labels_array = LabelEncoder().fit_transform(labels_array)
        images_array = np.array(images_array)
        labels_array = np.array(labels_array)
        return InputData.from_image(images=images_array, labels=labels_array, task=task)

    @staticmethod
    def image_from_csv(task: TaskTypesEnum.classification):
        raise NotImplementedError

    @staticmethod
    def images_from_pickle():
        raise NotImplementedError


# TODO
def convert_data_to_pickle():
    """function or class for convert InputData to pickle format fot further save"""
    raise NotImplementedError


# TODO delete these functions
def str_to_digit(labels):
    if not labels[0].isdigit():
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels)
    return labels


def load_from_folder(dir_path, image_size: int = None, per_class_limit: int = None):
    images_array = []
    labels_array = []
    for label in os.listdir(dir_path):
        label_path = os.path.join(dir_path, label)
        cnt = 0
        for image_name in os.listdir(label_path):

            if per_class_limit:
                if cnt >= per_class_limit:
                    break
            image_name = os.path.join(label_path, image_name)
            image = cv2.imread(image_name)
            image_size = image.shape[0] if image_size is None else image_size
            if image_size is not None:
                if image.shape[0] != image_size:
                    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            images_array.append(image)
            labels_array.append(label)
            cnt += 1
    labels_array = str_to_digit(labels_array)
    images_array = np.array(images_array)
    labels_array = np.array(labels_array)
    return images_array, labels_array, image_size, len(np.unique(labels_array))


def load_images(file_path, size=120, number_of_classes=10, per_class_limit=None):
    if number_of_classes == 10:
        with open(os.path.join(root, 'dataset_files', 'labels_10.json'), 'r') as fp:
            labels_dict = json.load(fp)
        with open(os.path.join(root, 'dataset_files', 'encoded_labels_10.json'), 'r') as fp:
            encoded_labels = json.load(fp)
    elif number_of_classes == 3:
        with open(os.path.join(root, 'dataset_files', 'labels.json'), 'r') as fp:
            labels_dict = json.load(fp)
        with open(os.path.join(root, 'dataset_files', 'encoded_labels.json'), 'r') as fp:
            encoded_labels = json.load(fp)
    else:
        print('specify the number of classes correctly')
    images_array = []
    labels_array = []

    files = [f for f in os.listdir(file_path) if isfile(join(file_path, f))]
    files.sort()
    encountered_targets_count = {} if per_class_limit is not None else None
    for filename in files:
        image = cv2.imread(join(file_path, filename))
        image = cv2.resize(image, (size, size))
        label_names = labels_dict[filename[:-4]]
        for name in label_names:
            num_label = encoded_labels[name]
            if encountered_targets_count is not None:
                if name not in encountered_targets_count:
                    encountered_targets_count[name] = 1
                elif encountered_targets_count[name] >= per_class_limit:
                    continue
                else:
                    encountered_targets_count[name] += 1
            images_array.append(image)
            labels_array.append(num_label)
    images_array = np.array(images_array)
    labels_array = np.array(labels_array)
    return images_array, labels_array


def data2pickle(path, num_of_classes, per_class_limit=None, save=True):
    """
    Convert dataset to pickle format with given limit of samples per class
    """
    x_arr, y_arr = load_images(path, number_of_classes=num_of_classes, per_class_limit=per_class_limit)
    dataset = []
    dataset.extend([x_arr, y_arr])
    if per_class_limit is not None:
        output_path = os.path.join(root, 'datasets', f'{per_class_limit}_samples_per_class_{path[17:]}.pickle')
    else:
        os.path.join(root, 'datasets', f'{path[17:]}.pickle')
    if save:
        with open(output_path, 'wb') as pickle_data:
            pickle.dump(dataset, pickle_data)


def from_pickle(path):
    with open(path, 'rb') as dataset:
        images_array, labels_array = pickle.load(dataset)
    return images_array, labels_array


def from_images(file_path, num_classes, task_type: TaskTypesEnum = TaskTypesEnum.classification, per_class_limit=None):
    _, extension = os.path.splitext(file_path)
    if not extension:
        images, labels = load_images(file_path, size=120, number_of_classes=num_classes,
                                     per_class_limit=per_class_limit)
    elif extension == '.pickle':
        images, labels = from_pickle(file_path)
    else:
        raise ValueError('Wrong file path')
    images_train, images_val, labels_train, labels_val = train_test_split(images, labels,
                                                                          random_state=1, train_size=0.8)
    task = Task(task_type=task_type, task_params=None)
    train_input_data = Data.from_image(images=images_train, labels=labels_train, task=task)
    validation_input_data = Data.from_image(images=images_val, labels=labels_val, task=task)

    return train_input_data, validation_input_data


def from_directory(file_path, task_type: TaskTypesEnum = TaskTypesEnum.classification,
                   per_class_limit=None):
    images, labels, image_size, _ = load_from_folder(file_path, per_class_limit=per_class_limit)
    images_train, images_val, labels_train, labels_val = train_test_split(images, labels,
                                                                          random_state=1, train_size=0.8)
    task = Task(task_type=task_type, task_params=None)
    train_input_data = Data.from_image(images=images_train, labels=labels_train, task=task)
    validation_input_data = Data.from_image(images=images_val, labels=labels_val, task=task)

    return train_input_data, validation_input_data, image_size


if __name__ == '__main__':
    print('Converting dataset to pickle...')
    # TODO local path
    dataset_path = os.path.join('D:/work/datasets/Generated_dataset')
    data2pickle(dataset_path, 3, 15)
