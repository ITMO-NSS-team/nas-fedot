import pathlib

import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum

import nas
import nas.data.nas_data
from nas.utils.utils import project_root, set_root

set_root(project_root())


def test_load_images_from_folder():
    """
    Test if loaded Input data is correct
    """
    path = 'example_datasets/butterfly_cls'
    task = Task(TaskTypesEnum.classification)
    dataset_path = pathlib.Path(path)
    dataset = nas.data.nas_data.InputDataNN.data_from_folder(dataset_path, task)
    assert isinstance(dataset, InputData)
    assert len(dataset.features) == len(dataset.target) == len(dataset.idx)
    assert len(np.unique(dataset.target)) == dataset.num_classes
