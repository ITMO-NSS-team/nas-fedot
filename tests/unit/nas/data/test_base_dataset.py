import pathlib
from typing import Any

import cv2
from fedot.core.repository.tasks import Task, TaskTypesEnum

from nas.data import BaseNasImageData, Preprocessor
from nas.data.dataset.builder import BaseNasDatasetBuilder
from nas.data.dataset.tf_dataset import KerasDataset
from nas.utils.utils import set_root, project_root

set_root(project_root())


def data_transformation_one(sample: Any) -> Any:
    return cv2.resize(sample, (15, 15))


def data_transformation_two(sample: Any) -> Any:
    return cv2.resize(sample, (5, 5))


def test_preprocessor_image_shape_resize():
    preprocessor_one = Preprocessor((32, 32))
    preprocessor_two = Preprocessor([24, 24])
    root = project_root()
    sample = cv2.imread(str(pathlib.Path(f'{root}/example_datasets/butterfly_cls/ADONIS/1.jpg')))
    sample_one = preprocessor_one.transform_sample(sample)
    sample_two = preprocessor_two.transform_sample(sample)
    assert sample_one.shape[:-1] == (32, 32)
    assert sample_two.shape[:-1] == (24, 24)


def test_preprocessor_mode_switch():
    root = project_root()
    sample = cv2.imread(str(pathlib.Path(f'{root}/example_datasets/butterfly_cls/ADONIS/1.jpg')))
    preprocessor = Preprocessor((32, 32))
    preprocessor.set_features_transformations([data_transformation_one, data_transformation_two])
    new_sample_one = preprocessor.transform_sample(sample)
    preprocessor.mode = 'evaluation'
    new_sample_two = preprocessor.transform_sample(sample)
    assert new_sample_one.shape[:-1] == (5, 5)
    assert new_sample_two.shape[:-1] == (32, 32)


def test_dataset_batch_size_change():
    data = BaseNasImageData.data_from_folder(pathlib.Path(project_root(), 'example_datasets/butterfly_cls'),
                                             Task(TaskTypesEnum.classification))
    dataset = BaseNasDatasetBuilder(KerasDataset, batch_size=1)
    generator_1 = dataset.build(data, mode='train')
    generator_2 = dataset.build(data, mode='train', batch_size=2)
    generator_3 = dataset.build(data, mode='train', batch_size=3)
    assert len(generator_1[0][0]) == len(generator_1[0][1]) == 1
    assert len(generator_2[0][0]) == len(generator_2[0][1]) == 2
    assert len(generator_3[0][0]) == len(generator_3[0][1]) == 3


def test_dataset_mode_switch():
    root = project_root()
    data = BaseNasImageData.data_from_folder(pathlib.Path(root, 'example_datasets/butterfly_cls'),
                                             Task(TaskTypesEnum.classification))
    preprocessor = Preprocessor((32, 32))
    preprocessor.set_features_transformations([data_transformation_one, data_transformation_two])
    dataset = BaseNasDatasetBuilder(KerasDataset, batch_size=1).set_data_preprocessor(preprocessor)
    dataset_one = dataset.build(data, mode='test')
    dataset_two = dataset.build(data, mode='train')
    assert dataset_one._shuffle is False
    assert dataset_one._preprocessor._mode == 'evaluation'
    assert dataset_two._shuffle is True
    assert dataset_two._preprocessor._mode == 'default'
