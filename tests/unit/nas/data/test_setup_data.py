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
    preprocessor = nas.data.preprocessor.Preprocessor()
    shuffle = True

    return nas.data.setup_data(data, batch_size, preprocessor, mode,
                               nas.data.dataset.tf_dataset.KerasDataset, shuffle)
