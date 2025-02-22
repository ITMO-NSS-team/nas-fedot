import cv2
import numpy as np
import torch
from fedot.core.data.data import InputData
from torchvision.transforms import Normalize


class NasImageNormalizer:
    """
    This class calculates mean and std over dataset and then
    applies normalization to given data sample with calculated mean and std.
    """

    def __init__(self, dataset: InputData, mean: float = None, std: float = None):
        dataset_mean = 0.
        dataset_std = 0.
        for image in dataset.features:
            image = cv2.cvtColor(cv2.imread(str(image[0])), cv2.COLOR_BGR2RGB)
            dataset_mean += np.mean(image)
            dataset_std += np.std(image)
        self.filter = Normalize(mean=dataset_mean / len(dataset.features), std=dataset_std / len(dataset.features))

    def __call__(self, sample: np.ndarray):
        return np.array(self.filter(torch.Tensor(sample)))


class MinMaxScaler:
    """
    This class scales data to the range [0, 1].
    """

    def __init__(self, dataset: InputData):
        """ Features a paths to images """
        self.min = 0
        self.max = 255

        for image in dataset.features:
            image = cv2.cvtColor(cv2.imread(str(image[0])), cv2.COLOR_BGR2RGB)
            self.min = min(self.min, np.min(image))
            self.max = max(self.max, np.max(image))

    def __call__(self, sample: np.ndarray):
        return (sample - self.min) / (self.max - self.min)


class MakeSingleChannel:
    """
    This class converts image to single channel.
    """

    def __call__(self, sample: np.ndarray):
        return sample.mean(axis=0, keepdims=True)
