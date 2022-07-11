import copy

from fedot.core.data.data import InputData
from sklearn.model_selection import train_test_split, StratifiedKFold


def _split_generator(input_data, features, targets, idx):
    new_data = copy.deepcopy(input_data)
    new_data.data_generator.data_generator.features = features
    new_data.data_generator.data_generator.targets = targets
    new_data.idx = idx

    return new_data


def generator_train_test_split(data, split_ratio=.8, shuffle_flag=False):
    data_generator = data.data_generator
    features, targets = data_generator.data_generator.features, data_generator.data_generator.targets
    idx = data.idx
    idx_train, idx_test, x_train, x_test, y_train, y_test = train_test_split(idx, features, targets,
                                                                             test_size=1 - split_ratio,
                                                                             shuffle=shuffle_flag, random_state=42)
    train_input_data = _split_generator(data, x_train, y_train, idx_train)
    test_input_data = _split_generator(data, x_test, y_test, idx_test)

    return train_input_data, test_input_data


class KFoldSplit:
    pass

# def loader_train_test_split(input_data, ratio: float = .7, shuffle=True):
#     features, targets = input_data.data_loader.dataset.samples
#     x_train, x_val, y_train, y_val = train_test_split(features, targets, test_size=1 - ratio, random_state=42,
#                                                       shuffle=shuffle)
#     train_input_data = copy.deepcopy(input_data)
#     val_input_data = copy.deepcopy(input_data)
#     train_input_data.data_loader.dataset.samples = [x_train, y_train]
#     val_input_data.data_loader.dataset.samples = [x_val, y_val]
#     return train_input_data, val_input_data
#
#
# def kfold_from_input_data(loader, n_splits: int = 5, shuffle=True):
#     k_fold = StratifiedKFold(n_splits, shuffle=shuffle, random_state=42)
#     k_fold.get_n_splits(*loader.data_loader.dataset.samples)
#     for i, j in k_fold.split(*loader.data_loader.dataset.samples):
#         print('!!')
#
#
# class DataLoaderKFold:
#     def __init__(self, n_splits=5, shuffle=True):
#         self.n_splits = n_splits
#         self.shuffle = shuffle
#
#     def split(self, *samples) -> InputData:
#         pass
