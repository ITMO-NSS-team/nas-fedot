from pathlib import Path

import torch.utils.data
from fedot.core.repository.tasks import Task, TaskTypesEnum

from nas.data.dataset.builder import ImageDatasetBuilder
from nas.data.dataset.torch_dataset import TorchDataset
from nas.data.nas_data import InputDataNN
from nas.utils.utils import project_root


def test_torch_builder():
    """
    Test if dataset builder works properly with pytorch dataset generation.
    """
    path = Path(project_root()) / 'example_datasets' / 'butterfly_cls'
    input_data = InputDataNN.data_from_folder(path, Task(TaskTypesEnum.classification))
    builder_cls = ImageDatasetBuilder(TorchDataset, [64, 64])
    dataset = builder_cls.build(input_data)
    assert isinstance(dataset, torch.utils.data.Dataset)
    assert hasattr(dataset, '__getitem__')
    assert hasattr(dataset, '__len__')


def test_torch_dataset_inputs():
    """
    Test if data loader outputs image of given size
    """
    path = Path(project_root()) / 'example_datasets' / 'butterfly_cls'
    input_data = InputDataNN.data_from_folder(path, Task(TaskTypesEnum.classification))
    image_size = [[12, 12], [30, 30], [64, 64], [80, 80]]
    for i in image_size:
        builder_cls = ImageDatasetBuilder(TorchDataset, i)
        dataset = builder_cls.build(input_data)
        f, t = dataset[0]
        assert list(f.shape)[1:] == i
