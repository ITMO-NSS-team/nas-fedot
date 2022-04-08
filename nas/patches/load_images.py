import json
import os
import pickle
from os.path import isfile, join

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from fedot.core.data.data import Data
from fedot.core.repository.tasks import Task, TaskTypesEnum


def load_images(file_path, size=120, number_of_classes=10):
    if number_of_classes == 10:
        with open('dataset_files/labels_10.json', 'r') as fp:
            labels_dict = json.load(fp)
        with open('dataset_files/encoded_labels_10.json', 'r') as fp:
            encoded_labels = json.load(fp)
    elif number_of_classes == 3:
        with open('dataset_files/labels.json', 'r') as fp:
            labels_dict = json.load(fp)
        with open('dataset_files/encoded_labels.json', 'r') as fp:
            encoded_labels = json.load(fp)
    else:
        print('specify the number of classes correctly')
    images_array = []
    labels_array = []

    files = [f for f in os.listdir(file_path) if isfile(join(file_path, f))]
    files.sort()
    for filename in files:
        image = cv2.imread(join(file_path, filename))
        image = cv2.resize(image, (size, size))
        images_array.append(image)
        label_names = labels_dict[filename[:-4]]
        each_file_labels = [0 for _ in range(number_of_classes)]
        for name in label_names:
            num_label = encoded_labels[name]
            each_file_labels[num_label] = 1
        labels_array.append(each_file_labels)
    images_array = np.array(images_array)
    labels_array = np.array(labels_array)

    return images_array, labels_array


def from_pickle(path):
    with open(path, 'rb') as dataset:
        images_array, labels_array = pickle.load(dataset)
    return images_array, labels_array


def from_images(file_path, num_classes, task_type: TaskTypesEnum = TaskTypesEnum.classification):
    _, extension = os.path.splitext(file_path)
    if not extension:
        images, labels = load_images(file_path, size=120, number_of_classes=num_classes)
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
