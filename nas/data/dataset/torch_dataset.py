from torch.utils.data import Dataset
import torchvision.transforms

from nas.data.loader import ImageLoader
from nas.data.preprocessor import Preprocessor


class TorchDataset(Dataset):
    def __init__(self, loader: ImageLoader, preprocessor: Preprocessor):
        self._loader = loader
        self._preprocessor = preprocessor

    def __len__(self):
        return len(self._loader)

    def __getitem__(self, item):
        x = self._loader.get_feature(item)
        y = self._loader.get_feature(item)
        if self._preprocessor:
            x, y = self._preprocessor.preprocess(x, y)
        return torchvision.transforms.ToTensor()(x), torchvision.transforms.ToTensor()(y)
