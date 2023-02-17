import pathlib
from typing import Any

import cv2
from fedot.core.repository.tasks import Task, TaskTypesEnum

from nas.data import InputDataNN, Preprocessor
from nas.data.dataset.builder import ImageDatasetBuilder
from nas.data.dataset.tf_dataset import KerasDataset
from nas.utils.utils import set_root, project_root

set_root(project_root())


def data_transformation_one(sample: Any) -> Any:
    return cv2.resize(sample, (15, 15))


def data_transformation_two(sample: Any) -> Any:
    return cv2.resize(sample, (5, 5))


def test_preprocessor_mode_switch():
    data = InputDataNN.data_from_folder(pathlib.Path(project_root(), 'example_datasets/butterfly_cls'),
                                        Task(TaskTypesEnum.classification))
    dataset = ImageDatasetBuilder(KerasDataset, batch_size=1, image_size=[24, 24]).set_data_preprocessor(Preprocessor())
    generator_1 = dataset.build(data, mode='train')
    generator_2 = dataset.build(data, mode='test')
    assert generator_1._preprocessor
    assert generator_2._preprocessor is None


def test_dataset_batch_size_change():
    data = InputDataNN.data_from_folder(pathlib.Path(project_root(), 'example_datasets/butterfly_cls'),
                                        Task(TaskTypesEnum.classification))
    dataset = ImageDatasetBuilder(KerasDataset, batch_size=1, image_size=[24, 24])
    generator_1 = dataset.build(data, mode='train')
    generator_2 = dataset.build(data, mode='train', batch_size=2)
    generator_3 = dataset.build(data, mode='train', batch_size=3)
    assert len(generator_1[0][0]) == len(generator_1[0][1]) == 1
    assert len(generator_2[0][0]) == len(generator_2[0][1]) == 2
    assert len(generator_3[0][0]) == len(generator_3[0][1]) == 3
