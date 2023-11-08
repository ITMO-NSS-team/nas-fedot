import cv2
import numpy as np
from fedot.core.data.data import InputData
from torchvision.transforms import Normalize


class NasImageNormalizer:
    def __init__(self, dataset: InputData, mean: float = None, std: float = None):
        dataset_mean = 0.
        dataset_std = 0.
        for image in dataset.features:
            image = cv2.cvtColor(cv2.imread(str(image[0])), cv2.COLOR_BGR2RGB)
            dataset_mean += np.mean(image)
            dataset_std += np.std(image)
        self.filter = Normalize(mean=dataset_mean / len(dataset.features), std=dataset_std / len(dataset.features))

    def __call__(self, sample):
        return self.filter(sample)
