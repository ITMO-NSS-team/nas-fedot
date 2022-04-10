import json
import os
from os.path import isfile, join

import cv2.cv2 as cv2
import numpy as np
from sklearn.model_selection import train_test_split

from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def load_images(file_path, img_size=120, number_of_classes=10, is_train=True):
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
    Xarr = []
    Yarr = []

    files = [f for f in os.listdir(file_path) if isfile(join(file_path, f))]
    files.sort()
    for filename in files:
        image = cv2.imread(join(file_path, filename))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (img_size, img_size))
        Xarr.append(image)
        label_names = labels_dict[filename[:-4]]
        each_file_labels = [0 for _ in range(number_of_classes)]
        for name in label_names:
            num_label = encoded_labels[name]
            # each_file_labels.append(num_label)
            each_file_labels[num_label] = 1
        Yarr.append(each_file_labels)
    Xarr = np.array(Xarr)
    Yarr = np.array(Yarr)
    # Xarr = Xarr.reshape(-1, img_size, img_size, 1)

    return Xarr, Yarr


def from_images(file_path, num_classes, img_size: int = 120, task_type: TaskTypesEnum = TaskTypesEnum.classification):
    Xtrain, Ytrain = load_images(file_path, img_size=img_size, number_of_classes=num_classes, is_train=True)
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, random_state=1, train_size=0.8)
    task = Task(task_type=task_type, task_params=None)
    train_input_data = InputData(idx=np.arange(0, len(Xtrain)), features=Xtrain, target=np.array(Ytrain),
                                 task=task, data_type=DataTypesEnum.image)
    val_input_data = InputData(idx=np.arange(0, len(Xval)), features=Xval, target=np.array(Yval),
                               task=task, data_type=DataTypesEnum.image)
    # test_input_data = InputData(idx=np.arange(0, len(Xtest)), features=Xtest, target=np.array(Ytest),
    #                            task_type=task_type)

    return train_input_data, val_input_data
