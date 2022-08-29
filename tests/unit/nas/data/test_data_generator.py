import numpy as np
import pathlib
import tensorflow as tf

from fedot.core.repository.tasks import Task, TaskTypesEnum

import nas
from nas.data.data_generator import DataGenerator
from nas.utils.utils import set_root, project_root

set_root(project_root())


def _setup_preprocessor():
    preprocessor = nas.data.data_generator.Preprocessor()
    preprocessor.set_image_size((20, 20))
    preprocessor.set_features_transformations([tf.convert_to_tensor])
    return preprocessor


def setup_loader():
    task = Task(TaskTypesEnum.classification)
    dataset_path = pathlib.Path('example_datasets/butterfly_cls')
    data = nas.data.load_images.NNData.data_from_folder(dataset_path, task)
    return nas.data.data_generator.Loader(data)


def setup_data_generator():

    dataset_loader = setup_loader()
    preprocessor = _setup_preprocessor()
    return DataGenerator(dataset_loader, preprocessor, batch_size=8)


def test_is_valid_loader_len():
    loader = setup_loader()
    assert len(loader) == len(loader.features) == len(loader.target)


def test_is_all_samples_unique():
    loader = setup_loader()
    assert len(np.unique(loader.features)) == len(loader)


def test_is_correct_number_of_classes():
    loader = setup_loader()
    assert tuple(set([len(s) for _, s in loader]))[-1] == loader.num_classes


def test_preprocessor():
    generator = setup_data_generator()
    for i in range(len(generator)):
        sample, target = generator[i]
        assert np.all([isinstance(s, tf.Tensor) for s in sample])
        assert np.all([isinstance(t, tf.Tensor) for t in target])
        assert len(set([s.numpy().shape for s in sample])) == 1
        assert tuple(set([s.numpy().shape for s in sample]))[-1] == (20, 20, 3)


def test_is_correct_batch_size():
    generator = setup_data_generator()
    for i in range(len(generator)):
        sample, target = generator[i]
        assert len(sample) == generator.batch_size
