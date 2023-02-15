import pathlib

import tensorflow as tf
from fedot.core.repository.tasks import Task, TaskTypesEnum

import nas
import nas.data.dataset.tf_dataset
import nas.data.nas_data
from nas.utils.utils import project_root, set_root

set_root(project_root())
image_size = 16
batch_size = 8


def _load_dataset_as_input_data():
    path = 'example_datasets/butterfly_cls'
    task = Task(TaskTypesEnum.classification)
    dataset_path = pathlib.Path(path)
    return nas.data.nas_data.InputDataNN.data_from_folder(dataset_path, task)


def _make_generator(mode: str):
    data = _load_dataset_as_input_data()
    to_tensor_transform = tf.convert_to_tensor
    preprocessor = nas.data.preprocessor.Preprocessor((image_size, image_size))
    preprocessor.set_image_size((image_size, image_size)).set_features_transformations([to_tensor_transform])
    shuffle = True

    return nas.data.setup_data(data, batch_size, preprocessor, mode,
                               nas.data.dataset.tf_dataset.KerasDataset, shuffle)


def test_if_generator_is_valid():
    generator = _make_generator('train')
    assert isinstance(generator, tf.keras.utils.Sequence)
    generator = _make_generator('test')
    assert isinstance(generator, tf.keras.utils.Sequence)
    generator = _make_generator('valid')
    assert isinstance(generator, tf.keras.utils.Sequence)
