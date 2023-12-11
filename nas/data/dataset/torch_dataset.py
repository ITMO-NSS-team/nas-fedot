import albumentations
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from typing import List

from nas.data.loader import ImageLoader
from nas.data.preprocessor import Preprocessor


class TorchDataset(Dataset):
    def __init__(self, loader: ImageLoader, preprocessor: List = None):
        self._loader = loader
        self._preprocessor = albumentations.Compose(preprocessor)

    def __len__(self):
        return len(self._loader)

    def __getitem__(self, item):
        x = self._loader.get_feature(item)
        y = self._loader.get_target(item)
        if self._preprocessor:
            x = self._preprocessor(image=x)['image']
        return x, torch.Tensor(y)
