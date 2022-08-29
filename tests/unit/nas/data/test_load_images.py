import pathlib

from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.data.data import InputData

import nas
from nas.utils.utils import project_root, set_root

set_root(project_root())


def test_load_images_from_folder():
    path = 'example_datasets/butterfly_cls'
    task = Task(TaskTypesEnum.classification)
    dataset_path = pathlib.Path('example_datasets/butterfly_cls')
    dataset = nas.data.load_images.NNData.data_from_folder(dataset_path, task)
    assert isinstance(dataset, InputData)
    