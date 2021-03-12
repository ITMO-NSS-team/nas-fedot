from fedot.core.models.data import InputData
from fedot.core.repository.task_types import TaskTypesEnum, MachineLearningTasksEnum
import numpy as np
import cv2
import json
import os
from os.path import isfile, join
from sklearn.model_selection import train_test_split


def load_images(file_path, size=120, is_train=True):
    with open('labels.json', 'r') as fp:
        labels_dict = json.load(fp)
    with open('encoded_labels.json', 'r') as fp:
        encoded_labels = json.load(fp)
    Xarr = []
    Yarr = []
    number_of_classes = 3
    files = [f for f in os.listdir(file_path) if isfile(join(file_path, f))]
    files.sort()
    for filename in files:
        image = cv2.imread(join(file_path, filename))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (size, size))
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
    # Xarr = Xarr.reshape(-1, size, size, 1)

    return Xarr, Yarr


def from_images(file_path, task_type: TaskTypesEnum = MachineLearningTasksEnum.classification):
    Xtrain, Ytrain = load_images(file_path, size=120, is_train=True)
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, random_state=1, train_size=0.8)
    train_input_data = InputData(idx=np.arange(0, len(Xtrain)), features=Xtrain, target=np.array(Ytrain),
                                 task_type=task_type)
    val_input_data = InputData(idx=np.arange(0, len(Xval)), features=Xval, target=np.array(Yval),
                               task_type=task_type)
    # test_input_data = InputData(idx=np.arange(0, len(Xtest)), features=Xtest, target=np.array(Ytest),
    #                            task_type=task_type)

    return train_input_data, val_input_data
