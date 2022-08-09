from copy import deepcopy
from typing import Callable
from functools import partial

from sklearn.model_selection import train_test_split, StratifiedKFold
from fedot.core.data.data import InputData


def k_fold_for_images(data: InputData, n_splits, shuffle=True):
    task = data.task
    supplementary_data = deepcopy(data.supplementary_data)
    data_type = data.data_type

    features = data.features
    target = data.target
    indices = data.idx

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=1)
    splitter.get_n_splits(features, target)

    for train_id, test_id in splitter.split(features, target, indices):
        x_train, y_train, train_id = [features[i] for i in train_id], [target[i] for i in train_id], indices[train_id]
        x_test, y_test, test_id = [features[i] for i in test_id], [target[i] for i in test_id], indices[test_id]
        train_data = InputData(idx=train_id, features=x_train, target=y_train, task=task,
                               supplementary_data=supplementary_data, data_type=data_type)
        test_data = InputData(idx=test_id, features=x_test, target=y_test, task=task,
                              supplementary_data=supplementary_data, data_type=data_type)
        yield train_data, test_data


def generator_kfold_split(data, n_splits=2, random_state=42, shuffle=True):
    features = data.features
    target = data.target
    indices = data.idx
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splitter.get_n_splits(indices)

    return splitter.split(features, target)


class SplitterGenerator:
    def __init__(self, split_method: str, **kwargs):
        self.splitter = split_method
        self.params = kwargs

    @staticmethod
    def _split_generator(input_data, x, y, idx):
        new_data = copy.deepcopy(input_data)
        new_data.data_generator.data_generator.features = x
        new_data.data_generator.data_generator.targets = y
        new_data.target = y
        new_data.features = x
        new_data.idx = idx

        return new_data

    def _holdout_split(self, data):
        generator = data.data_generator
        features, targets = generator.data_generator.features, generator.data_generator.targets
        indices = data.idx
        try:
            train_id, id_test, x_train, x_test, y_train, y_test = train_test_split(indices, features, targets,
                                                                                   **self.params)
        except TypeError:
            raise TypeError('Wrong params for holdout split')

        yield self._split_generator(data, x_train, y_train, train_id), self._split_generator(data, x_test, y_test,
                                                                                             id_test)

    def _stratified_kfold(self, data):
        features = data.features
        targets = data.target
        indices = data.idx

        try:
            splitter = StratifiedKFold(**self.params)
        except TypeError:
            raise TypeError('Wrong params for kfold split')
        splitter.get_n_splits(features, targets)
        for train_id, test_id in splitter.split(features, targets):
            x_train, y_train, train_id = [features[i] for i in train_id], [targets[i] for i in train_id], train_id
            x_test, y_test, test_id = [features[i] for i in test_id], [targets[i] for i in test_id], test_id
            yield self._split_generator(data, x_train, y_train, train_id), self._split_generator(data, x_test, y_test,
                                                                                                 test_id)

    def split(self, data):
        if self.splitter == 'k_fold':
            return self._stratified_kfold(data)
        elif self.splitter == 'holdout':
            return self._holdout_split(data)
        else:
            raise NotImplementedError('This split method is not implemented')
