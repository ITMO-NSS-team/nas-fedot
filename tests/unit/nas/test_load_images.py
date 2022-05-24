import os

from fedot.core.data.data import InputData

from nas.patches.load_images import from_images
from nas.patches.utils import project_root

root = project_root()


def test_from_pickle():
    dataset_path = os.path.join(root, 'datasets', '15_samples_per_class_10cls_Generated_dataset.pickle')
    train_data, val_data = from_images(dataset_path, 10)
    assert type(train_data) == InputData
    assert type(val_data) == InputData
    assert train_data.num_classes == 10
    assert val_data.num_classes == 10
