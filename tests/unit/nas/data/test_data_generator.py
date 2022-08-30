import numpy as np
import pathlib
import tensorflow as tf

from fedot.core.repository.tasks import Task, TaskTypesEnum

import nas
from nas.data.data_generator import DataGenerator
from nas.utils.utils import set_root, project_root

set_root(project_root())
img_size = 16


def _setup_preprocessor():
    preprocessor = nas.data.data_generator.Preprocessor()
    preprocessor.set_image_size((img_size, img_size))
    preprocessor.set_features_transformations([tf.convert_to_tensor])
    return preprocessor


def _setup_input_data_dataset():
    task = Task(TaskTypesEnum.classification)
    dataset_path = pathlib.Path('example_datasets/butterfly_cls')
    return nas.data.load_images.NNData.data_from_folder(dataset_path, task)


def _setup_loader():
    dataset = _setup_input_data_dataset()
    return nas.data.data_generator.Loader(dataset)


def setup_data_generator():
    dataset_loader = _setup_loader()
    preprocessor = _setup_preprocessor()
    return DataGenerator(dataset_loader, preprocessor, batch_size=8)


def test_is_valid_loader_len():
    loader = _setup_loader()
    assert len(loader) == len(loader.features) == len(loader.target)


def test_is_all_samples_unique():
    loader = _setup_loader()
    assert len(np.unique(loader.features)) == len(loader)


def test_is_correct_number_of_classes():
    loader = _setup_loader()
    assert tuple(set([len(s) for _, s in loader]))[-1] == loader.num_classes


def test_targets_samples_dtype():
    generator = setup_data_generator()
    for i in range(len(generator)):
        samples, targets = generator[i]
        assert np.all([isinstance(s, tf.Tensor) for s in samples])
        assert np.all([isinstance(t, tf.Tensor) for t in targets])


def test_if_image_size_is_correct():
    generator = setup_data_generator()
    for i in range(len(generator)):
        sample, target = generator[i]
        assert len(set([s.numpy().shape for s in sample])) == 1
        assert tuple(set([s.numpy().shape for s in sample]))[-1] == (img_size, img_size, 3)


def test_is_correct_batch_size():
    generator = setup_data_generator()
    for i in range(len(generator)):
        sample, target = generator[i]
        assert len(sample) == generator.batch_size


def test_if_task_correct():
    generator = setup_data_generator()
    dataset = _setup_input_data_dataset()
    assert dataset.task == generator.task
