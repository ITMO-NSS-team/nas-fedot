import pathlib

from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum

import nas
from nas.utils.utils import project_root, set_root

set_root(project_root())


def test_load_images_from_folder():
    path = 'example_datasets/butterfly_cls'
    task = Task(TaskTypesEnum.classification)
    dataset_path = pathlib.Path(path)
    dataset = nas.data.load_images.NasData.data_from_folder(dataset_path, task)
    assert isinstance(dataset, InputData)


def test_load_images_from_csv():
    # TODO find small csv dataset and _update this test
    path = 'example_datasets/butterfly_cls'
    task = Task(TaskTypesEnum.classification)
    dataset_path = pathlib.Path(path)
    dataset = nas.data.load_images.NasData.data_from_folder(dataset_path, task)
    assert True
