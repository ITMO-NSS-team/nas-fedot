import json
import os
import pickle
from os.path import isfile, join

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from fedot.core.data.data import Data
from fedot.core.repository.tasks import Task, TaskTypesEnum
from nas.patches.utils import project_root

root = project_root()


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


if __name__ == '__main__':
    print('Converting dataset to pickle...')
    dataset_path = os.path.join('D:/work/datasets/Generated_dataset')
    data2pickle(dataset_path, 3, 15)
